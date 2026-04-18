"""Live training loop for mutated architectures.

Sub-modules split for readability:
- training_metrics.py: compute_speedrun_auc, fit_learning_curve, _average_router_entropy
- probes.py: run_multi_needle_probes, run_downstream_probes, _estimate_long_recall
"""

from __future__ import annotations

import gc
import json as std_json
import math
import time
import traceback
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .candidates import Candidate
from .data import DataModule, TokenBatch
from .dsl import ArchitectureSpec
from .evaluation import estimate_flops_per_token
from .models import BranchRouter, EvolutionModel, MoELayer, count_parameters
from .morphology import match_experts_to_parent, sort_moe_experts
from .optimizers import apply_gradient_transform_, apply_update_filter_, build_optimizer
from .parameter_golf import (
    DEFAULT_SLIDING_EVAL_MODE,
    DEFAULT_STANDARD_EVAL_MODE,
    ParameterGolfDataModule,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_parameter_golf_val,
    measure_quantized_artifact,
    quantize_state_dict_int8,
)
from .probes import _estimate_long_recall, run_downstream_probes, run_multi_needle_probes
from .training_metrics import _average_router_entropy, compute_speedrun_auc, fit_learning_curve

# Re-export for backward compatibility
__all__ = [
    "FullWeightTrainer",
    "_average_router_entropy",
    "_estimate_long_recall",
    "compute_speedrun_auc",
    "fit_learning_curve",
    "run_downstream_probes",
    "run_multi_needle_probes",
]


class FullWeightTrainer:
    """Runs a short, full-weight finetune for each candidate."""

    def __init__(
        self,
        checkpoint_dir: Path = Path("runs/checkpoints"),
        device: str | None = None,
        checkpoint_dtype: str = "fp16",
        steps: int = 50,
        eval_batches: int = 2,
        entropy_threshold: float = 0.5,
        entropy_patience: int = 3,
        instability_threshold: float = 5.0,
        no_improve_patience: int = 20,
        improvement_tolerance: float = 1e-3,
        speedrun_callback: Callable[[int, float, int], None] | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if device:
            self.device = torch.device(device)
        elif torch.backends.cuda.is_built() and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.checkpoint_dtype = checkpoint_dtype
        self.steps = steps
        self.eval_batches = eval_batches
        self.entropy_threshold = entropy_threshold
        self.entropy_patience = entropy_patience
        self.instability_threshold = instability_threshold
        self.no_improve_patience = no_improve_patience
        self.improvement_tolerance = improvement_tolerance
        self._eval_module_cache: dict[str, Any] = {}
        self._pg_lut_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.last_parameter_golf_error: str | None = None
        self.speedrun_callback = speedrun_callback

    def _micro_batch_tokens(self, spec: ArchitectureSpec) -> int:
        return max(1, int(spec.data.seq_len) * int(spec.data.batch_size))

    def _grad_accum_steps(self, spec: ArchitectureSpec) -> int:
        target_tokens = getattr(spec.train, "batch_tokens", None)
        micro_tokens = self._micro_batch_tokens(spec)
        if target_tokens is None:
            return 1
        return max(1, math.ceil(float(target_tokens) / float(micro_tokens)))

    def _parameter_golf_wallclock_seconds(self, spec: ArchitectureSpec) -> float | None:
        if spec.parameter_golf is None:
            return None
        configured = getattr(spec.parameter_golf, "max_wallclock_seconds", None)
        if configured is not None:
            return float(configured)
        if str(getattr(spec.parameter_golf, "track", "") or "").lower() == "10min":
            return 600.0
        return None

    def _parameter_golf_eval_protocol(self, spec: ArchitectureSpec) -> str:
        if spec.parameter_golf is None:
            return "mid_fidelity"
        return str(getattr(spec.parameter_golf, "eval_protocol", "mid_fidelity") or "mid_fidelity")

    def _parameter_golf_report_eval_modes(self, spec: ArchitectureSpec) -> list[str]:
        if spec.parameter_golf is None:
            return ["standard"]
        modes = list(getattr(spec.parameter_golf, "report_eval_modes", None) or ["standard"])
        ordered: list[str] = []
        for mode in modes:
            value = str(mode or "standard")
            if value not in ordered:
                ordered.append(value)
        if "standard" not in ordered:
            ordered.insert(0, "standard")
        return ordered

    def _parameter_golf_eval_token_budget(
        self,
        spec: ArchitectureSpec,
        *,
        fallback_eval_tokens: int | None,
    ) -> int | None:
        protocol = self._parameter_golf_eval_protocol(spec)
        if protocol == "truth_full":
            return None
        if fallback_eval_tokens is not None:
            return int(fallback_eval_tokens)
        if spec.parameter_golf is not None and spec.parameter_golf.val_batch_tokens is not None:
            return int(spec.parameter_golf.val_batch_tokens)
        return getattr(spec.data, "eval_tokens", None)

    def _predict_elapsed_seconds(self, samples: list[float], *, fallback_seconds: float) -> float:
        recent = [float(item) for item in samples[-4:] if float(item) > 0.0]
        if recent:
            return float(sum(recent) / len(recent))
        return max(float(fallback_seconds), 0.0)

    def _parameter_golf_val_batch_tokens(
        self,
        spec: ArchitectureSpec,
        *,
        fallback_eval_tokens: int | None = None,
    ) -> int:
        if spec.parameter_golf is None:
            if fallback_eval_tokens is not None:
                return max(int(spec.data.seq_len), int(fallback_eval_tokens))
            return max(1, int(spec.data.seq_len) * int(spec.data.batch_size))

        configured = getattr(spec.parameter_golf, "val_batch_tokens", None)
        if configured is not None:
            return max(int(spec.data.seq_len), int(configured))
        if fallback_eval_tokens is not None:
            return max(int(spec.data.seq_len), int(fallback_eval_tokens))
        eval_tokens = getattr(spec.data, "eval_tokens", None)
        if eval_tokens is not None:
            return max(int(spec.data.seq_len), int(eval_tokens))
        return max(1, int(spec.data.seq_len) * int(spec.data.batch_size))

    def _muon_momentum_for_step(
        self,
        spec: ArchitectureSpec,
        *,
        base_momentum: float | None,
        step_idx: int,
    ) -> float | None:
        if base_momentum is None:
            return None
        optimizer_cfg = getattr(spec.train, "optimizer", None)
        if optimizer_cfg is None:
            return base_momentum
        warmup_steps = int(getattr(optimizer_cfg, "muon_momentum_warmup_steps", 0) or 0)
        warmup_start = getattr(optimizer_cfg, "muon_momentum_warmup_start", None)
        if warmup_steps <= 0 or warmup_start is None:
            return base_momentum
        frac = min(float(max(step_idx, 0)) / float(max(warmup_steps, 1)), 1.0)
        return (1.0 - frac) * float(warmup_start) + frac * float(base_momentum)

    def _apply_optimizer_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: list[float],
        base_momentums: list[float | None],
        spec: ArchitectureSpec,
        *,
        step_idx: int,
        total_steps: int,
        elapsed_s: float,
    ) -> None:
        warmup = int(getattr(spec.train, "warmup", 0) or 0)
        warmdown = int(getattr(spec.train, "warmdown_steps", 0) or 0)
        scale = 1.0
        if warmup > 0:
            scale = min(scale, float(step_idx + 1) / float(warmup))
        wallclock_limit = self._parameter_golf_wallclock_seconds(spec)
        if warmdown > 0:
            if wallclock_limit is not None and elapsed_s > 0.0 and step_idx > 0:
                max_wallclock_ms = float(wallclock_limit) * 1000.0
                elapsed_ms = float(elapsed_s) * 1000.0
                step_ms = elapsed_ms / float(max(step_idx, 1))
                warmdown_ms = float(warmdown) * step_ms
                remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
                if remaining_ms <= warmdown_ms:
                    scale = min(scale, remaining_ms / max(warmdown_ms, 1.0e-9))
            elif total_steps > 0:
                remaining = max(0, total_steps - step_idx)
                if remaining < warmdown:
                    scale = min(scale, float(remaining) / float(warmdown))
        for idx, (group, base_lr) in enumerate(zip(optimizer.param_groups, base_lrs, strict=True)):
            group["lr"] = base_lr * scale
            base_momentum = base_momentums[idx]
            if "momentum" in group:
                momentum = self._muon_momentum_for_step(
                    spec,
                    base_momentum=base_momentum,
                    step_idx=step_idx,
                )
                if momentum is not None:
                    group["momentum"] = float(momentum)

    def _get_eval_module(
        self, spec: ArchitectureSpec, *, eval_batches: int | None = None
    ) -> tuple[Any, int]:
        """Return a deterministic eval DataModule + token budget.

        If `data.eval_shards` is provided, use it. Otherwise, reuse the training
        shards but sample with a different RNG seed so `ppl_eval` isn't identical
        to the training stream.
        """
        batches = int(self.eval_batches if eval_batches is None else eval_batches)
        batches = max(1, batches)
        eval_shards = getattr(spec.data, "eval_shards", []) or []
        eval_cfg = spec.data.model_copy(deep=True)
        if eval_shards:
            eval_cfg.shards = list(eval_shards)
        if getattr(eval_cfg, "packed", False):
            eval_cfg.packed_split = "val"
        eval_cfg.healing_shards = []
        eval_cfg.healing_tokens = None

        eval_tokens_cfg = getattr(spec.data, "eval_tokens", None)
        if isinstance(eval_tokens_cfg, int) and eval_tokens_cfg > 0:
            eval_tokens = int(eval_tokens_cfg)
        else:
            eval_tokens = int(spec.data.seq_len) * int(spec.data.batch_size) * batches

        seed_val = int(getattr(spec.train, "seed", 0) or 0)
        if spec.parameter_golf is not None:
            cache_payload = {
                "parameter_golf": spec.parameter_golf.model_dump(mode="python"),
                "seq_len": int(spec.data.seq_len),
                "batch_size": int(spec.data.batch_size),
            }
            cache_key = std_json.dumps(cache_payload, sort_keys=True)
            cache_key = f"{seed_val + 1}:{cache_key}"
            eval_module = self._eval_module_cache.get(cache_key)
            if eval_module is None:
                eval_module = ParameterGolfDataModule(
                    spec.parameter_golf,
                    seq_len=spec.data.seq_len,
                    batch_size=spec.data.batch_size,
                    seed=seed_val + 1,
                )
                self._eval_module_cache[cache_key] = eval_module
            eval_module.reset_rng(seed_val + 1)
            return eval_module, eval_tokens

        cache_key = std_json.dumps(eval_cfg.model_dump(mode="python"), sort_keys=True)
        cache_key = f"{seed_val + 1}:{cache_key}"
        eval_module = self._eval_module_cache.get(cache_key)
        if eval_module is None:
            eval_module = DataModule(eval_cfg, seed=seed_val + 1)
            self._eval_module_cache[cache_key] = eval_module
        eval_module.reset_rng(seed_val + 1)
        return eval_module, eval_tokens

    def _eval_batches(self, eval_module: Any, *, eval_tokens: int) -> Iterable[TokenBatch]:
        if isinstance(eval_module, ParameterGolfDataModule):
            return eval_module.batches(max_tokens=eval_tokens, split="val")
        return cast(Iterable[TokenBatch], eval_module.batches(max_tokens=eval_tokens))

    def _parameter_golf_exact_eval_device(self) -> torch.device:
        # Apple GPU memory is too tight for the exact challenge scorer at realistic
        # validation batch sizes, so run the final exact check on CPU there.
        if self.device.type == "mps":
            return torch.device("cpu")
        return self.device

    def _get_parameter_golf_luts(
        self,
        spec: ArchitectureSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if spec.parameter_golf is None:
            raise ValueError("parameter_golf config is required for challenge evaluation")
        eval_device = self._parameter_golf_exact_eval_device() if device is None else device
        cache_key = (
            f"{eval_device.type}:{eval_device.index}:{spec.parameter_golf.tokenizer_path}:"
            f"{spec.model.head.vocab}"
        )
        cached = self._pg_lut_cache.get(cache_key)
        if cached is not None:
            return cached
        luts = build_sentencepiece_luts(
            spec.parameter_golf.tokenizer_path,
            vocab_size=spec.model.head.vocab,
            device=eval_device,
        )
        self._pg_lut_cache[cache_key] = luts
        return luts

    def _build_parameter_golf_eval_model(
        self,
        spec: ArchitectureSpec,
        *,
        state_dict: dict[str, torch.Tensor],
        recurrence_steps: dict[int, int],
        device: torch.device,
    ) -> EvolutionModel:
        eval_model = EvolutionModel(spec.model).to(device)
        eval_model.load_state_dict(state_dict, strict=False)
        if recurrence_steps:
            eval_model.set_recurrence_steps(recurrence_steps)
        return eval_model

    def _parameter_golf_failure_metrics(self, spec: ArchitectureSpec) -> dict[str, float]:
        if spec.parameter_golf is None:
            return {}
        budget = float(spec.parameter_golf.artifact_budget_bytes)
        code_bytes = float(spec.parameter_golf.code_bytes)
        return {
            "val_loss": 1e9,
            "val_bpb": 1e9,
            "val_bpb_standard": 1e9,
            "post_quant_val_bpb": 1e9,
            "post_quant_val_bpb_standard": 1e9,
            "quantization_delta_bpb": 1e9,
            "artifact_code_bytes": code_bytes,
            "artifact_budget_bytes": budget,
            "artifact_payload_bytes": 1e9,
            "artifact_raw_torch_bytes": 1e9,
            "artifact_zlib_bytes": 1e9,
            "artifact_total_bytes": 1e9,
            "artifact_over_budget_bytes": 1e9,
            "parameter_golf_error": 1.0,
            "parameter_golf_eval_batch_tokens_used": 0.0,
            "parameter_golf_post_quant_eval_batch_tokens_used": 0.0,
            "parameter_golf_eval_backoff_count": 0.0,
        }

    def _parameter_golf_eval_batch_token_candidates(
        self,
        *,
        seq_len: int,
        batch_tokens: int,
    ) -> list[int]:
        minimum = max(1, int(seq_len))
        current = max(minimum, int(batch_tokens))
        candidates: list[int] = []
        while True:
            if current not in candidates:
                candidates.append(current)
            if current <= minimum:
                break
            current = max(minimum, current // 2)
        return candidates

    def _parameter_golf_retryable_eval_error(self, exc: Exception) -> bool:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        message = str(exc).lower()
        retryable_snippets = (
            "out of memory",
            "cuda error: out of memory",
            "mps backend out of memory",
            "resource exhausted",
        )
        return any(snippet in message for snippet in retryable_snippets)

    def _parameter_golf_clear_eval_memory(self, device: torch.device) -> None:
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            return
        if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _evaluate_parameter_golf_modes_with_backoff(
        self,
        model: EvolutionModel,
        *,
        spec: ArchitectureSpec,
        val_tokens: torch.Tensor,
        base_bytes_lut: torch.Tensor,
        has_leading_space_lut: torch.Tensor,
        is_boundary_token_lut: torch.Tensor,
        device: torch.device,
        batch_tokens: int,
        eval_token_budget: int | None,
        report_modes: list[str],
    ) -> tuple[dict[str, tuple[float, float]], int, int]:
        candidates = self._parameter_golf_eval_batch_token_candidates(
            seq_len=spec.data.seq_len,
            batch_tokens=batch_tokens,
        )
        last_error: Exception | None = None
        for backoff_count, candidate_batch_tokens in enumerate(candidates):
            try:
                results: dict[str, tuple[float, float]] = {}
                for eval_mode in report_modes:
                    results[eval_mode] = eval_parameter_golf_val(
                        model,
                        seq_len=spec.data.seq_len,
                        val_tokens=val_tokens,
                        base_bytes_lut=base_bytes_lut,
                        has_leading_space_lut=has_leading_space_lut,
                        is_boundary_token_lut=is_boundary_token_lut,
                        device=device,
                        batch_tokens=candidate_batch_tokens,
                        eval_mode=eval_mode,
                        total_eval_tokens=eval_token_budget,
                    )
                return results, candidate_batch_tokens, backoff_count
            except Exception as exc:
                last_error = exc
                if not self._parameter_golf_retryable_eval_error(exc):
                    raise
                if candidate_batch_tokens <= int(spec.data.seq_len):
                    raise
                self._parameter_golf_clear_eval_memory(device)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Parameter Golf evaluation failed before any scoring attempt.")

    def _evaluate_parameter_golf_metrics(
        self,
        model: EvolutionModel,
        spec: ArchitectureSpec,
        *,
        recurrence_steps: dict[int, int],
    ) -> dict[str, float]:
        if spec.parameter_golf is None:
            return {}
        self.last_parameter_golf_error = None
        eval_model: EvolutionModel | None = None
        quant_model: EvolutionModel | None = None
        try:
            eval_module, eval_tokens = self._get_eval_module(spec)
            if not isinstance(eval_module, ParameterGolfDataModule):
                raise TypeError("Parameter Golf evaluation requires ParameterGolfDataModule")
            val_tokens = eval_module.validation_tokens()
            eval_device = self._parameter_golf_exact_eval_device()
            state_dict = {
                key: value.detach().to(device="cpu") for key, value in model.state_dict().items()
            }
            if eval_device == self.device:
                eval_model = model
            else:
                eval_model = self._build_parameter_golf_eval_model(
                    spec,
                    state_dict=state_dict,
                    recurrence_steps=recurrence_steps,
                    device=eval_device,
                )
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
                self._get_parameter_golf_luts(spec, device=eval_device)
            )
            batch_tokens = self._parameter_golf_val_batch_tokens(
                spec,
                fallback_eval_tokens=eval_tokens,
            )
            eval_token_budget = self._parameter_golf_eval_token_budget(
                spec,
                fallback_eval_tokens=eval_tokens,
            )
            report_modes = self._parameter_golf_report_eval_modes(spec)
            val_results, val_batch_tokens_used, val_backoff_count = (
                self._evaluate_parameter_golf_modes_with_backoff(
                    eval_model,
                    spec=spec,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                    device=eval_device,
                    batch_tokens=batch_tokens,
                    eval_token_budget=eval_token_budget,
                    report_modes=report_modes,
                )
            )
            val_loss, val_bpb = val_results["standard"]
            artifact_metrics = measure_quantized_artifact(
                state_dict,
                code_bytes=spec.parameter_golf.code_bytes,
                artifact_budget_bytes=spec.parameter_golf.artifact_budget_bytes,
                parameter_golf=spec.parameter_golf,
            )
            quant_payload, _ = quantize_state_dict_int8(
                state_dict,
                parameter_golf=spec.parameter_golf,
            )
            restored_state = dequantize_state_dict_int8(quant_payload)
            quant_model = self._build_parameter_golf_eval_model(
                spec,
                state_dict=restored_state,
                recurrence_steps=recurrence_steps,
                device=eval_device,
            )
            post_quant_results, post_quant_batch_tokens_used, post_quant_backoff_count = (
                self._evaluate_parameter_golf_modes_with_backoff(
                    quant_model,
                    spec=spec,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                    device=eval_device,
                    batch_tokens=batch_tokens,
                    eval_token_budget=eval_token_budget,
                    report_modes=report_modes,
                )
            )
            _, post_quant_val_bpb = post_quant_results["standard"]
            metrics = dict(artifact_metrics)
            metrics.update(
                {
                    "val_loss": float(val_loss),
                    "val_bpb": float(val_bpb),
                    "val_bpb_standard": float(val_bpb),
                    "post_quant_val_bpb": float(post_quant_val_bpb),
                    "post_quant_val_bpb_standard": float(post_quant_val_bpb),
                    "quantization_delta_bpb": float(post_quant_val_bpb - val_bpb),
                    "parameter_golf_error": 0.0,
                    "parameter_golf_eval_batch_tokens_used": float(val_batch_tokens_used),
                    "parameter_golf_post_quant_eval_batch_tokens_used": float(
                        post_quant_batch_tokens_used
                    ),
                    "parameter_golf_eval_backoff_count": float(
                        val_backoff_count + post_quant_backoff_count
                    ),
                }
            )
            if DEFAULT_SLIDING_EVAL_MODE in val_results:
                metrics["val_bpb_sliding64"] = float(val_results[DEFAULT_SLIDING_EVAL_MODE][1])
            if DEFAULT_SLIDING_EVAL_MODE in post_quant_results:
                metrics["post_quant_val_bpb_sliding64"] = float(
                    post_quant_results[DEFAULT_SLIDING_EVAL_MODE][1]
                )
            return metrics
        except Exception as exc:
            self.last_parameter_golf_error = (
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc().strip()}"
            )
            return self._parameter_golf_failure_metrics(spec)
        finally:
            if eval_model is not None and eval_model is not model:
                del eval_model
            if quant_model is not None:
                del quant_model

    def _checkpoint_torch_dtype(self) -> torch.dtype:
        key = (self.checkpoint_dtype or "fp16").lower()
        if key in {"fp16", "float16"}:
            return torch.float16
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported checkpoint_dtype: {self.checkpoint_dtype}")

    def _checkpoint_state(self, model: nn.Module) -> dict[str, torch.Tensor]:
        dtype = self._checkpoint_torch_dtype()
        state = {}
        for key, value in model.state_dict().items():
            tensor = value.detach().to(device="cpu")
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            state[key] = tensor
        return state

    def train(
        self,
        candidate: Candidate,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        seed_state_path: Path | None = None,
    ) -> tuple[dict[str, float], Path]:
        seed_val = int(getattr(spec.train, "seed", 0) or 0)
        torch.manual_seed(seed_val)
        model = EvolutionModel(spec.model).to(self.device)
        model.train()
        model.set_grad_checkpointing(bool(getattr(spec.train, "grad_checkpoint", False)))
        sort_moe_experts(model)
        parent_state: dict[str, torch.Tensor] = {}
        if seed_state_path and seed_state_path.exists():
            # Load checkpoints onto CPU first to avoid transient peak device memory
            # during weight inheritance (especially on MPS).
            try:
                loaded_state: Any = torch.load(  # nosec B614 - checkpoints produced locally
                    seed_state_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                loaded_state = torch.load(  # nosec B614 - checkpoints produced locally
                    seed_state_path,
                    map_location="cpu",
                )
            state_obj: Any = loaded_state
            if (
                isinstance(state_obj, dict)
                and "state_dict" in state_obj
                and isinstance(state_obj["state_dict"], dict)
            ):
                state_obj = state_obj["state_dict"]
            if isinstance(state_obj, dict):
                parent_state = {k: v for k, v in state_obj.items() if isinstance(v, torch.Tensor)}
            current_state = model.state_dict()
            coerced: dict[str, torch.Tensor] = {}
            for key, value in parent_state.items():
                if key not in current_state:
                    continue
                target_dtype = current_state[key].dtype
                if value.is_floating_point() and value.dtype != target_dtype:
                    coerced[key] = value.to(dtype=target_dtype)
                else:
                    coerced[key] = value
            parent_state = coerced
            try:
                model.load_state_dict(parent_state, strict=False)
                match_experts_to_parent(model, parent_state)
            except RuntimeError:
                # If shapes changed (e.g., GQA/Sparsity mutations), load only compatible tensors.
                compatible_state = {
                    k: v
                    for k, v in parent_state.items()
                    if k in current_state and current_state[k].shape == v.shape
                }
                if compatible_state:
                    model.load_state_dict(compatible_state, strict=False)
                # fall back to random init for the rest
        optimizer = build_optimizer(model, spec.train)
        base_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
        base_momentums = [
            (
                float(momentum_value)
                if isinstance(momentum_value := group.get("momentum"), (int, float))
                else None
            )
            for group in optimizer.param_groups
        ]
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        start_time = time.perf_counter()
        tokens_seen = 0
        iterator = iter(batch_iter)
        stop_reason = ""
        best_loss = float("inf")
        no_improve = 0
        entropy_bad = 0
        nan_or_inf = False
        max_loss_jump = 0.0
        optimizer.zero_grad()
        total_steps = max(1, self.steps)
        grad_accum_steps = self._grad_accum_steps(spec)
        micro_batch_tokens = self._micro_batch_tokens(spec)
        effective_batch_tokens = micro_batch_tokens * grad_accum_steps
        # Track losses for learning curve fitting
        loss_history_steps: list[int] = []
        loss_history_values: list[float] = []
        # Track stability metrics
        recent_losses: list[float] = []  # Last N losses for variance calculation
        grad_norms: list[float] = []  # Gradient norms for spike detection
        grad_norm_spikes = 0  # Count of gradient norm spikes
        update_filter_keep_history: list[float] = []
        gradient_transform_apply_history: list[float] = []
        # Track improvement rate for adaptive rung budgets
        early_loss: float | None = None
        mid_loss: float | None = None
        autocast_ctx: Any = nullcontext()
        if self.device.type in {"cuda", "mps"}:
            dtype = torch.bfloat16 if bool(getattr(spec.train, "bf16", True)) else torch.float16
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=dtype)
        speedrun_interval = int(getattr(spec.train, "speedrun_eval_interval", 0) or 0)
        speedrun_eval_batches = int(
            getattr(spec.train, "speedrun_eval_batches", self.eval_batches) or self.eval_batches
        )
        speedrun_target_loss: float | None = None
        speedrun_target_loss_raw = getattr(spec.train, "speedrun_target_loss", None)
        if speedrun_target_loss_raw is not None:
            try:
                speedrun_target_loss = float(speedrun_target_loss_raw)
            except (TypeError, ValueError):
                speedrun_target_loss = None
        speedrun_target_ppl: float | None = None
        speedrun_target_ppl_raw = getattr(spec.train, "speedrun_target_ppl", None)
        if speedrun_target_ppl_raw is not None:
            try:
                speedrun_target_ppl = float(speedrun_target_ppl_raw)
            except (TypeError, ValueError):
                speedrun_target_ppl = None
        if speedrun_target_loss is None and speedrun_target_ppl is not None:
            speedrun_target_loss = math.log(speedrun_target_ppl)
        speedrun_enabled = speedrun_interval > 0
        speedrun_eval_failed = False
        speedrun_eval_module: Any | None = None
        speedrun_eval_tokens = 0
        speedrun_error = 0.0
        speedrun_reached = False
        speedrun_best_loss = float("inf")
        speedrun_eval_points: list[tuple[float, float]] = []
        speedrun_steps_to_target = 0.0
        speedrun_tokens_to_target = 0.0
        speedrun_time_to_target = 0.0
        speedrun_flops_to_target = 0.0
        speedrun_prev_eval_loss: float | None = None
        speedrun_prev_eval_step = 0
        speedrun_prev_eval_tokens = 0.0
        speedrun_prev_eval_time = 0.0
        speedrun_prev_eval_flops = 0.0
        flops_seen = 0.0
        wallclock_limit_s = self._parameter_golf_wallclock_seconds(spec)
        train_step_durations: list[float] = []
        speedrun_eval_durations: list[float] = []
        parameter_golf_periodic_durations: list[float] = []
        parameter_golf_val_every = 0
        parameter_golf_val_batch_tokens = 0
        parameter_golf_periodic_best_bpb = float("inf")
        parameter_golf_periodic_last_bpb = 1e9
        parameter_golf_periodic_last_loss = 1e9
        parameter_golf_periodic_count = 0
        parameter_golf_periodic_skipped = 0
        parameter_golf_periodic_error = 0.0
        parameter_golf_val_tokens: torch.Tensor | None = None
        parameter_golf_luts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if spec.parameter_golf is not None:
            parameter_golf_val_every = int(getattr(spec.parameter_golf, "val_loss_every", 0) or 0)
            parameter_golf_val_batch_tokens = self._parameter_golf_val_batch_tokens(spec)
            if parameter_golf_val_every > 0:
                try:
                    eval_module, _ = self._get_eval_module(spec)
                    if isinstance(eval_module, ParameterGolfDataModule):
                        parameter_golf_val_tokens = eval_module.validation_tokens()
                        parameter_golf_luts = self._get_parameter_golf_luts(spec)
                except Exception:
                    parameter_golf_periodic_error = 1.0

        # Multi-threshold speedrun tracking
        multi_thresholds_raw = getattr(spec.train, "speedrun_multi_thresholds", None)
        multi_thresholds: list[float] = []
        if multi_thresholds_raw:
            threshold_inputs: list[Any]
            if isinstance(multi_thresholds_raw, str | int | float):
                threshold_inputs = [multi_thresholds_raw]
            else:
                threshold_inputs = list(multi_thresholds_raw)
            parsed_thresholds: list[float] = []
            for raw in threshold_inputs:
                try:
                    threshold = float(raw)
                except (TypeError, ValueError):
                    continue
                if threshold > 0:
                    parsed_thresholds.append(threshold)
            multi_thresholds = sorted(parsed_thresholds, reverse=True)
        multi_threshold_reached: dict[float, bool] = dict.fromkeys(multi_thresholds, False)
        multi_threshold_tokens: dict[float, float | None] = dict.fromkeys(multi_thresholds, None)
        multi_threshold_flops: dict[float, float | None] = dict.fromkeys(multi_thresholds, None)
        # Track eval history for interpolation
        eval_history: list[tuple[float, float, float, float]] = []  # (tokens, flops, time, ppl)
        flops_per_token_est = estimate_flops_per_token(
            spec,
            recurrence_steps=self._recurrence_schedule(spec, self.steps, total_steps),
        )
        grad_total: float | torch.Tensor = 0.0
        optimizer_step_idx = 0
        while optimizer_step_idx < self.steps:
            recurrence_steps: dict[int, int] = {}
            if spec.model.recurrences:
                recurrence_steps = self._recurrence_schedule(spec, optimizer_step_idx, total_steps)
                model.set_recurrence_steps(recurrence_steps)
            elapsed_s = max(time.perf_counter() - start_time, 0.0)
            if wallclock_limit_s is not None and optimizer_step_idx > 0:
                predicted_step_s = self._predict_elapsed_seconds(
                    train_step_durations,
                    fallback_seconds=max(elapsed_s / float(max(optimizer_step_idx, 1)), 0.0),
                )
                remaining_s = float(wallclock_limit_s) - float(elapsed_s)
                if remaining_s <= max(predicted_step_s, 1e-3):
                    stop_reason = "wallclock_cap"
                    break
            self._apply_optimizer_schedule(
                optimizer,
                base_lrs,
                base_momentums,
                spec,
                step_idx=optimizer_step_idx,
                total_steps=total_steps,
                elapsed_s=elapsed_s,
            )

            micro_batches_taken = 0
            step_loss_total = 0.0
            step_tokens = 0
            step_started_at = time.perf_counter()
            while micro_batches_taken < grad_accum_steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    if micro_batches_taken <= 0:
                        stop_reason = (
                            "token_budget"
                            if getattr(batch_iter, "max_tokens", None)
                            else "data_exhausted"
                        )
                        break
                    stop_reason = (
                        "token_budget"
                        if getattr(batch_iter, "max_tokens", None)
                        else "data_exhausted"
                    )
                    break

                input_ids = batch.input_ids.to(self.device)
                attn_mask = batch.attention_mask.to(self.device)
                target_ids = (
                    batch.target_ids.to(self.device) if batch.target_ids is not None else None
                )
                with autocast_ctx:
                    logits = model(input_ids)
                    # Next-token prediction (causal LM): predict input_ids[t+1] from input_ids[:t+1]
                    if target_ids is None and logits.size(1) <= 1:
                        stop_reason = "data_exhausted"
                        break
                    if target_ids is None:
                        shifted_logits = logits[:, :-1, :].contiguous()
                        labels = input_ids[:, 1:].contiguous()
                        label_mask = attn_mask[:, 1:].contiguous()
                    else:
                        shifted_logits = logits.contiguous()
                        labels = target_ids.contiguous()
                        label_mask = attn_mask.contiguous()
                    labels = labels.masked_fill(label_mask == 0, -100)
                    loss = criterion(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        labels.view(-1),
                    )
                # Auxiliary MoE routing losses
                aux_loss = torch.tensor(0.0, device=self.device)
                lb_coeff = float(getattr(spec.train, "router_lb_coeff", 0.0) or 0.0)
                ent_coeff = float(getattr(spec.train, "router_entropy_coeff", 0.0) or 0.0)
                for mod in model.modules():
                    if not isinstance(mod, MoELayer):
                        continue
                    layer_ent = ent_coeff
                    layer_lb = lb_coeff
                    cfg = getattr(mod, "cfg", None)
                    if cfg is not None:
                        # Per-layer coefficients: treat as multiplier if a global coeff is set,
                        # otherwise treat as an absolute override.
                        aux = getattr(cfg, "router_aux_weight", None)
                        if aux is not None:
                            aux_val = float(aux)
                            layer_ent = layer_ent * aux_val if layer_ent > 0.0 else aux_val
                        lbw = getattr(cfg, "router_lb_weight", None)
                        if lbw is not None:
                            lb_val = float(lbw)
                            layer_lb = layer_lb * lb_val if layer_lb > 0.0 else lb_val
                        elif layer_lb == 0.0:
                            layer_lb = float(getattr(cfg, "balance", 0.0) or 0.0)
                    if layer_ent > 0.0 and hasattr(mod, "last_entropy"):
                        aux_loss = aux_loss + (-layer_ent) * mod.last_entropy
                    if layer_lb > 0.0 and hasattr(mod, "last_lb"):
                        aux_loss = aux_loss + layer_lb * mod.last_lb
                loss = loss + aux_loss
                current_loss = float(loss.item())
                if not math.isfinite(current_loss):
                    nan_or_inf = True
                    stop_reason = "nan_or_inf_loss"
                    optimizer.zero_grad()
                    break

                step_loss_total += current_loss
                micro_batches_taken += 1
                try:
                    step_tokens += int(attn_mask.sum().item())
                except Exception:
                    step_tokens += int(input_ids.numel())

                (loss / float(grad_accum_steps)).backward()

            if nan_or_inf or micro_batches_taken <= 0:
                break

            current_loss = step_loss_total / float(micro_batches_taken)

            # Track loss for learning curve fitting (every 5 steps to avoid overhead)
            if (optimizer_step_idx + 1) % 5 == 0 or optimizer_step_idx == 0:
                loss_history_steps.append(optimizer_step_idx + 1)
                loss_history_values.append(current_loss)

            # Track recent losses for stability (keep last 20)
            recent_losses.append(current_loss)
            if len(recent_losses) > 20:
                recent_losses.pop(0)

            # Track early and mid losses for improvement rate
            if optimizer_step_idx == min(5, total_steps // 4):
                early_loss = current_loss
            if optimizer_step_idx == total_steps // 2:
                mid_loss = current_loss

            if float(spec.train.clip) > 0.0:
                grad_total = clip_grad_norm_(model.parameters(), spec.train.clip)
                if hasattr(grad_total, "item"):
                    grad_norm = float(grad_total.item())
                else:
                    grad_norm = float(grad_total)
            else:
                grad_sq_sum = 0.0
                for param in model.parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    grad_sq_sum += float(grad.detach().float().pow(2).sum().item())
                grad_norm = math.sqrt(max(grad_sq_sum, 0.0))
                grad_total = grad_norm
            if not math.isfinite(grad_norm):
                nan_or_inf = True
                stop_reason = "nan_or_inf_grad"
                optimizer.zero_grad()
                break

            # Track gradient norms for stability analysis
            grad_norms.append(grad_norm)
            # Detect spikes: grad_norm > 3x running average
            if len(grad_norms) >= 5:
                avg_norm = sum(grad_norms[-5:]) / 5
                if grad_norm > 3.0 * avg_norm and avg_norm > 0.1:
                    grad_norm_spikes += 1

            if grad_norm > self.instability_threshold:
                stop_reason = f"high_grad({grad_norm:.2f})"
                optimizer.zero_grad()
                break
            transform_frac = apply_gradient_transform_(optimizer, spec.train)
            gradient_transform_apply_history.append(float(transform_frac))
            keep_frac = apply_update_filter_(optimizer, spec.train)
            update_filter_keep_history.append(float(keep_frac))
            optimizer.step()
            optimizer.zero_grad()
            train_step_durations.append(max(time.perf_counter() - step_started_at, 1e-6))
            tokens_seen += step_tokens
            if step_tokens > 0:
                flops_seen += float(step_tokens) * estimate_flops_per_token(
                    spec, recurrence_steps=recurrence_steps
                )
            if (
                speedrun_enabled
                and not speedrun_eval_failed
                and (optimizer_step_idx + 1) % speedrun_interval == 0
            ):
                if speedrun_eval_module is None:
                    try:
                        speedrun_eval_module, speedrun_eval_tokens = self._get_eval_module(
                            spec,
                            eval_batches=speedrun_eval_batches,
                        )
                    except Exception:
                        speedrun_error = 1.0
                        speedrun_eval_failed = True
                        continue
                if wallclock_limit_s is not None:
                    elapsed_now = max(time.perf_counter() - start_time, 0.0)
                    predicted_speedrun_s = self._predict_elapsed_seconds(
                        speedrun_eval_durations,
                        fallback_seconds=self._predict_elapsed_seconds(
                            train_step_durations,
                            fallback_seconds=max(
                                elapsed_now / float(max(optimizer_step_idx + 1, 1)),
                                0.0,
                            ),
                        ),
                    )
                    if float(wallclock_limit_s) - float(elapsed_now) <= predicted_speedrun_s:
                        speedrun_error = 1.0
                        continue
                eval_started_at = time.perf_counter()
                try:
                    speedrun_eval_module.reset_rng(seed_val + 1)
                    loss_val = self._evaluate_loss(
                        model,
                        spec,
                        self._eval_batches(
                            speedrun_eval_module,
                            eval_tokens=speedrun_eval_tokens,
                        ),
                        criterion,
                        eval_batches=speedrun_eval_batches,
                        empty_value=1e9,
                    )
                except Exception:
                    speedrun_error = 1.0
                    speedrun_eval_failed = True
                    loss_val = 1e9
                speedrun_eval_durations.append(max(time.perf_counter() - eval_started_at, 1e-6))
                if loss_val < speedrun_best_loss:
                    speedrun_best_loss = loss_val
                if math.isfinite(float(loss_val)):
                    speedrun_eval_points.append((float(tokens_seen), float(loss_val)))
                if self.speedrun_callback is not None and math.isfinite(float(loss_val)):
                    self.speedrun_callback(
                        int(optimizer_step_idx + 1),
                        float(loss_val),
                        int(tokens_seen),
                    )
                if (
                    not speedrun_reached
                    and speedrun_target_loss is not None
                    and math.isfinite(float(loss_val))
                    and loss_val <= speedrun_target_loss
                ):
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    eval_step = int(optimizer_step_idx + 1)
                    eval_tokens = float(tokens_seen)
                    eval_flops = float(flops_seen)
                    eval_time = float(elapsed)
                    frac = 1.0
                    if (
                        speedrun_prev_eval_loss is not None
                        and math.isfinite(speedrun_prev_eval_loss)
                        and speedrun_prev_eval_loss > speedrun_target_loss
                        and math.isfinite(loss_val)
                    ):
                        denom = speedrun_prev_eval_loss - float(loss_val)
                        if denom > 0.0:
                            frac = (speedrun_prev_eval_loss - speedrun_target_loss) / denom
                            frac = max(0.0, min(1.0, frac))
                    speedrun_reached = True
                    speedrun_steps_to_target = float(speedrun_prev_eval_step) + frac * float(
                        eval_step - speedrun_prev_eval_step
                    )
                    speedrun_tokens_to_target = speedrun_prev_eval_tokens + frac * (
                        eval_tokens - speedrun_prev_eval_tokens
                    )
                    speedrun_flops_to_target = speedrun_prev_eval_flops + frac * (
                        eval_flops - speedrun_prev_eval_flops
                    )
                    speedrun_time_to_target = speedrun_prev_eval_time + frac * (
                        eval_time - speedrun_prev_eval_time
                    )
                if math.isfinite(float(loss_val)):
                    speedrun_prev_eval_loss = float(loss_val)
                    speedrun_prev_eval_step = int(optimizer_step_idx + 1)
                    speedrun_prev_eval_tokens = float(tokens_seen)
                    speedrun_prev_eval_time = max(time.perf_counter() - start_time, 1e-6)
                    speedrun_prev_eval_flops = float(flops_seen)

                    # Track eval history for multi-threshold interpolation
                    current_ppl = math.exp(loss_val) if loss_val < 20 else float("inf")
                    eval_history.append(
                        (
                            float(tokens_seen),
                            float(flops_seen),
                            speedrun_prev_eval_time,
                            current_ppl,
                        )
                    )

                    # Check multi-thresholds
                    for thresh in multi_thresholds:
                        if multi_threshold_reached[thresh]:
                            continue
                        if current_ppl <= thresh:
                            # Interpolate to find exact crossing point
                            if len(eval_history) >= 2:
                                prev_tokens, prev_flops, prev_time, prev_ppl = eval_history[-2]
                                curr_tokens, curr_flops, curr_time, curr_ppl = eval_history[-1]
                                if prev_ppl > thresh >= curr_ppl and prev_ppl != curr_ppl:
                                    frac = (prev_ppl - thresh) / (prev_ppl - curr_ppl)
                                    interp_tokens = prev_tokens + frac * (curr_tokens - prev_tokens)
                                    interp_flops = prev_flops + frac * (curr_flops - prev_flops)
                                else:
                                    interp_tokens = float(tokens_seen)
                                    interp_flops = float(flops_seen)
                            else:
                                interp_tokens = float(tokens_seen)
                                interp_flops = float(flops_seen)

                            multi_threshold_reached[thresh] = True
                            multi_threshold_tokens[thresh] = interp_tokens
                            multi_threshold_flops[thresh] = interp_flops
            if (
                parameter_golf_val_every > 0
                and parameter_golf_val_tokens is not None
                and parameter_golf_luts is not None
                and (optimizer_step_idx + 1) % parameter_golf_val_every == 0
            ):
                elapsed_now = max(time.perf_counter() - start_time, 0.0)
                predicted_periodic_s = self._predict_elapsed_seconds(
                    parameter_golf_periodic_durations,
                    fallback_seconds=max(
                        self._predict_elapsed_seconds(
                            train_step_durations,
                            fallback_seconds=max(
                                elapsed_now / float(max(optimizer_step_idx + 1, 1)), 0.0
                            ),
                        )
                        * 2.0,
                        1.0,
                    ),
                )
                if (
                    wallclock_limit_s is not None
                    and float(wallclock_limit_s) - float(elapsed_now) <= predicted_periodic_s
                ):
                    parameter_golf_periodic_skipped += 1
                else:
                    eval_started_at = time.perf_counter()
                    try:
                        val_loss, val_bpb = eval_parameter_golf_val(
                            model,
                            seq_len=spec.data.seq_len,
                            val_tokens=parameter_golf_val_tokens,
                            base_bytes_lut=parameter_golf_luts[0],
                            has_leading_space_lut=parameter_golf_luts[1],
                            is_boundary_token_lut=parameter_golf_luts[2],
                            device=self.device,
                            batch_tokens=parameter_golf_val_batch_tokens,
                            eval_mode=DEFAULT_STANDARD_EVAL_MODE,
                            total_eval_tokens=self._parameter_golf_eval_token_budget(
                                spec,
                                fallback_eval_tokens=spec.data.eval_tokens,
                            ),
                        )
                        parameter_golf_periodic_last_loss = float(val_loss)
                        parameter_golf_periodic_last_bpb = float(val_bpb)
                        parameter_golf_periodic_best_bpb = min(
                            parameter_golf_periodic_best_bpb,
                            float(val_bpb),
                        )
                        parameter_golf_periodic_count += 1
                    except Exception:
                        parameter_golf_periodic_error = 1.0
                    parameter_golf_periodic_durations.append(
                        max(time.perf_counter() - eval_started_at, 1e-6)
                    )
            # Router entropy guard
            step_entropy = _average_router_entropy(model)
            if step_entropy is not None and step_entropy < self.entropy_threshold:
                entropy_bad += 1
                if entropy_bad >= self.entropy_patience:
                    stop_reason = f"low_entropy({step_entropy:.2f})"
                    break
            else:
                entropy_bad = 0
            if best_loss < float("inf"):
                jump = current_loss - best_loss
                if jump > max_loss_jump:
                    max_loss_jump = jump
            if current_loss + self.improvement_tolerance < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.no_improve_patience:
                    stop_reason = "no_improve"
                    break
            if (
                wallclock_limit_s is not None
                and (time.perf_counter() - start_time) >= wallclock_limit_s
            ):
                stop_reason = "wallclock_cap"
                break
            optimizer_step_idx += 1
            if stop_reason in {"token_budget", "data_exhausted"}:
                break
        if nan_or_inf:
            raise RuntimeError(f"Non-finite loss/grad encountered ({stop_reason}).")
        train_duration = max(time.perf_counter() - start_time, 1e-6)
        throughput = tokens_seen / train_duration
        model.set_recurrence_steps(self._recurrence_schedule(spec, self.steps, total_steps))
        ppl_train = self._evaluate_perplexity(model, spec, batch_iter, criterion)
        ppl_eval = ppl_train
        ppl_eval_error = 0.0
        try:
            eval_module, eval_tokens = self._get_eval_module(spec)
            ppl_eval = self._evaluate_perplexity(
                model,
                spec,
                self._eval_batches(eval_module, eval_tokens=eval_tokens),
                criterion,
            )
        except Exception:
            # Best-effort: if eval dataset shards are unavailable (common in unit tests
            # or offline environments), fall back to the training-stream perplexity.
            ppl_eval = ppl_train
            ppl_eval_error = 1.0
        perplexity = ppl_eval
        speedrun_end_eval_loss = math.log(max(float(perplexity), 1e-12))
        long_recall_proxy = _estimate_long_recall(spec)
        passkey_metrics = self._passkey_probe(model, spec)
        # Long-recall score: use the probe if it beats the structural proxy.
        # This avoids collapsing the signal to 0.0 early when the probe hasn't learned yet.
        passkey_acc = float(passkey_metrics.get("passkey_acc", 0.0) or 0.0)
        long_recall = max(float(long_recall_proxy), passkey_acc)
        checkpoint_path = self.checkpoint_dir / f"{candidate.ident}.pt"
        state = self._checkpoint_state(model)
        torch.save(state, checkpoint_path)
        del state
        current_recurrence_steps = dict(getattr(model, "recurrence_steps", {}))
        # Aggregate router telemetry
        router_entropy = 0.0
        router_lb = 0.0
        router_load_max = 0.0
        router_load_min = 1.0
        router_load_cv = 0.0
        capacity_overflow = 0.0
        max_grad_norm = 0.0
        try:
            if hasattr(grad_total, "item"):
                max_grad_norm = float(grad_total.item())
            else:
                max_grad_norm = float(grad_total)
        except Exception:
            max_grad_norm = 0.0
        count_entropy = 0
        count_moe = 0
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, MoELayer | BranchRouter):
                    last = getattr(mod, "last_entropy", None)
                    if isinstance(last, torch.Tensor):
                        router_entropy += float(last.item())
                        count_entropy += 1
                if isinstance(mod, MoELayer):
                    if hasattr(mod, "last_lb"):
                        router_lb += float(mod.last_lb.item())
                    if hasattr(mod, "last_load"):
                        load = mod.last_load
                        max_val = float(load.max().item())
                        min_val = float(load.min().item())
                        mean_val = float(load.mean().item())
                        std_val = float(load.std().item())
                        router_load_max = max(router_load_max, max_val)
                        router_load_min = min(router_load_min, min_val)
                        if mean_val > 0.0:
                            router_load_cv = max(router_load_cv, std_val / max(mean_val, 1e-9))
                    if hasattr(mod, "last_overflow"):
                        capacity_overflow += float(mod.last_overflow)
                    count_moe += 1
        if count_entropy:
            router_entropy /= count_entropy
        if count_moe:
            router_lb /= count_moe
            capacity_overflow /= count_moe
        else:
            router_load_min = 0.0
        reason_code = 0.0
        if stop_reason.startswith("high_grad"):
            reason_code = 1.0
        elif stop_reason.startswith("low_entropy"):
            reason_code = 2.0
        elif stop_reason == "no_improve":
            reason_code = 3.0
        elif stop_reason == "token_budget":
            reason_code = 4.0
        elif stop_reason == "data_exhausted":
            reason_code = 5.0
        elif stop_reason == "wallclock_cap":
            reason_code = 6.0
        moe_penalty = 1.0 + 0.01 * spec.model.moe_block_count()
        # Fit learning curve to predict long-term performance
        learning_curve_metrics = fit_learning_curve(
            loss_history_steps,
            loss_history_values,
            target_steps=self.steps * 10,  # Extrapolate to 10x current training
        )
        total_duration = max(time.perf_counter() - start_time, 1e-6)
        post_eval_duration = max(total_duration - train_duration, 0.0)
        metrics = {
            "ppl_code": perplexity,
            "ppl_train": ppl_train,
            "ppl_eval": ppl_eval,
            "ppl_eval_error": ppl_eval_error,
            "ppl": perplexity,
            "ppl_math": perplexity * moe_penalty,
            "ppl_math_proxy": perplexity * moe_penalty,
            "throughput": throughput,
            "params": float(count_parameters(model)),
            "ram": float(count_parameters(model) * 2 / (1024**3)),
            "flops_per_token_est": float(flops_per_token_est),
            "long_recall": long_recall,
            "long_recall_proxy": long_recall_proxy,
            "router_entropy": router_entropy,
            "router_lb": router_lb,
            "router_load_max": router_load_max,
            "router_load_min": router_load_min,
            "router_load_cv": router_load_cv,
            "capacity_overflow": capacity_overflow,
            "max_grad_norm": max_grad_norm,
            "instability": max_grad_norm,
            "stop_reason_code": reason_code,
            "nan_seen": 1.0 if nan_or_inf else 0.0,
            "loss_spike": max(0.0, max_loss_jump),
            "opt_mask_keep_observed_avg": (
                float(sum(update_filter_keep_history)) / float(len(update_filter_keep_history))
                if update_filter_keep_history
                else 1.0
            ),
            "opt_mask_keep_observed_last": (
                float(update_filter_keep_history[-1]) if update_filter_keep_history else 1.0
            ),
            "opt_grad_transform_applied_avg": (
                float(sum(gradient_transform_apply_history))
                / float(len(gradient_transform_apply_history))
                if gradient_transform_apply_history
                else 0.0
            ),
            "opt_grad_transform_applied_last": (
                float(gradient_transform_apply_history[-1])
                if gradient_transform_apply_history
                else 0.0
            ),
            "grad_accum_steps": float(grad_accum_steps),
            "train_micro_batch_tokens": float(micro_batch_tokens),
            "train_effective_batch_tokens": float(effective_batch_tokens),
            "wallclock_seconds_used": float(train_duration),
            "wallclock_seconds_budget": (
                float(wallclock_limit_s) if wallclock_limit_s is not None else 0.0
            ),
            "wallclock_stop_triggered": 1.0 if stop_reason == "wallclock_cap" else 0.0,
            "train_wallclock_seconds_used": float(train_duration),
            "post_eval_wallclock_seconds_used": float(post_eval_duration),
            "total_job_seconds_used": float(total_duration),
        }
        if spec.parameter_golf is not None:
            metrics.update(
                {
                    "parameter_golf_val_every_steps": float(parameter_golf_val_every),
                    "parameter_golf_val_batch_tokens": float(parameter_golf_val_batch_tokens),
                    "parameter_golf_periodic_val_count": float(parameter_golf_periodic_count),
                    "parameter_golf_periodic_val_skipped": float(parameter_golf_periodic_skipped),
                    "parameter_golf_periodic_val_last_loss": float(
                        parameter_golf_periodic_last_loss
                    ),
                    "parameter_golf_periodic_val_last_bpb": float(parameter_golf_periodic_last_bpb),
                    "parameter_golf_periodic_val_best_bpb": (
                        float(parameter_golf_periodic_best_bpb)
                        if math.isfinite(parameter_golf_periodic_best_bpb)
                        else 1e9
                    ),
                    "parameter_golf_periodic_val_error": float(parameter_golf_periodic_error),
                }
            )
        # Add learning curve metrics
        metrics.update(learning_curve_metrics)

        # Compute stability metrics
        loss_variance = 0.0
        if len(recent_losses) >= 3:
            mean_loss = sum(recent_losses) / len(recent_losses)
            loss_variance = sum((loss - mean_loss) ** 2 for loss in recent_losses) / len(
                recent_losses
            )

        grad_norm_mean = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        grad_norm_std = 0.0
        if len(grad_norms) >= 3:
            grad_norm_std = (
                sum((g - grad_norm_mean) ** 2 for g in grad_norms) / len(grad_norms)
            ) ** 0.5

        # Compute instability score: composite of loss variance, grad spikes, and max grad
        instability_score = (
            0.3 * min(loss_variance, 10.0) / 10.0  # Normalized loss variance
            + 0.3 * min(grad_norm_spikes, 10) / 10.0  # Normalized spike count
            + 0.4 * min(max_grad_norm, self.instability_threshold) / self.instability_threshold
        )

        # Compute improvement rate for adaptive rung budgets
        improvement_rate = 0.0
        if early_loss is not None and mid_loss is not None and early_loss > 0:
            # Relative improvement from early to mid training
            improvement_rate = max(0.0, (early_loss - mid_loss) / early_loss)
        elif len(loss_history_values) >= 5:
            # Fallback: compute from first and last few losses
            first_avg = sum(loss_history_values[:3]) / 3
            last_avg = sum(loss_history_values[-3:]) / 3
            if first_avg > 0:
                improvement_rate = max(0.0, (first_avg - last_avg) / first_avg)

        metrics.update(
            {
                "loss_variance": float(loss_variance),
                "grad_norm_mean": float(grad_norm_mean),
                "grad_norm_std": float(grad_norm_std),
                "grad_norm_spikes": float(grad_norm_spikes),
                "instability_score": float(instability_score),
                "improvement_rate": float(improvement_rate),
            }
        )
        if speedrun_enabled:
            missing_penalty = 1e9
            missing_penalty_flops = missing_penalty * max(float(flops_per_token_est), 1.0)
            missing_penalty_time = missing_penalty
            best_eval_loss = (
                float(speedrun_best_loss) if math.isfinite(speedrun_best_loss) else missing_penalty
            )
            target_loss = float(speedrun_target_loss) if speedrun_target_loss is not None else None
            if target_loss is not None and math.isfinite(best_eval_loss):
                loss_gap = max(0.0, best_eval_loss - target_loss)
            else:
                loss_gap = missing_penalty
            gap_cap = 2.0
            gap_beta = 4.0

            tokens_budget = getattr(batch_iter, "max_tokens", None)
            if tokens_budget is None:
                tokens_budget = getattr(spec.train, "max_tokens", None)
            if tokens_budget is None:
                tokens_budget = tokens_seen
            tokens_budget = max(float(tokens_budget), float(tokens_seen), 1.0)

            flops_per_token_run = float(flops_seen) / max(float(tokens_seen), 1.0)
            if not math.isfinite(flops_per_token_run) or flops_per_token_run <= 0.0:
                flops_per_token_run = float(flops_per_token_est)
            flops_budget = tokens_budget * max(float(flops_per_token_run), 1.0)
            time_budget = tokens_budget / max(float(throughput), 1e-6)

            speedrun_loss_auc = missing_penalty
            if speedrun_eval_points and math.isfinite(tokens_budget) and tokens_budget > 0.0:
                token_cap = float(tokens_budget)
                cleaned = [
                    (max(0.0, min(float(t), token_cap)), float(loss))
                    for t, loss in speedrun_eval_points
                    if math.isfinite(float(loss))
                ]
                cleaned.sort(key=lambda pair: pair[0])
                points: list[tuple[float, float]] = []
                for t, loss in cleaned:
                    if points and abs(t - points[-1][0]) < 1e-9:
                        points[-1] = (t, loss)
                    else:
                        points.append((t, loss))
                if points:
                    if points[0][0] > 0.0:
                        points = [(0.0, points[0][1]), *points]
                    if points[-1][0] < token_cap:
                        points.append((token_cap, points[-1][1]))
                    auc = 0.0
                    prev_t, prev_l = points[0]
                    for t, loss in points[1:]:
                        dt = max(0.0, float(t) - float(prev_t))
                        auc += 0.5 * (float(prev_l) + float(loss)) * dt
                        prev_t, prev_l = t, loss
                        if prev_t >= token_cap:
                            break
                    speedrun_loss_auc = float(auc) / max(token_cap, 1.0)

            if speedrun_reached:
                speedrun_score = float(speedrun_flops_to_target)
                speedrun_time_score = float(speedrun_time_to_target)
            elif math.isfinite(float(speedrun_best_loss)):
                penalty = math.exp(gap_beta * min(float(loss_gap), gap_cap))
                speedrun_score = max(flops_budget, 1.0) * penalty
                speedrun_time_score = max(time_budget, 1.0) * penalty
            else:
                speedrun_score = missing_penalty_flops
                speedrun_time_score = missing_penalty_time
            metrics.update(
                {
                    "speedrun_reached": 1.0 if speedrun_reached else 0.0,
                    "speedrun_steps_to_target": (
                        speedrun_steps_to_target if speedrun_reached else missing_penalty
                    ),
                    "speedrun_tokens_to_target": (
                        speedrun_tokens_to_target if speedrun_reached else missing_penalty
                    ),
                    "speedrun_time_to_target": (
                        speedrun_time_to_target if speedrun_reached else missing_penalty
                    ),
                    "speedrun_flops_to_target": (
                        speedrun_flops_to_target if speedrun_reached else missing_penalty_flops
                    ),
                    "speedrun_best_eval_loss": best_eval_loss,
                    "speedrun_end_eval_loss": float(speedrun_end_eval_loss),
                    "speedrun_loss_auc": float(speedrun_loss_auc),
                    "speedrun_loss_gap": float(loss_gap),
                    "speedrun_score": float(speedrun_score),
                    "speedrun_time_score": float(speedrun_time_score),
                    "speedrun_error": speedrun_error,
                }
            )

            # Add multi-threshold metrics if configured
            if multi_thresholds:
                tokens_list = [
                    multi_threshold_tokens.get(t) for t in sorted(multi_thresholds, reverse=True)
                ]
                auc = compute_speedrun_auc(sorted(multi_thresholds, reverse=True), tokens_list)
                metrics["speedrun_auc"] = auc if math.isfinite(auc) else missing_penalty

                # Add per-threshold metrics
                for thresh in multi_thresholds:
                    thresh_key = str(int(thresh))
                    reached = multi_threshold_reached.get(thresh, False)
                    metrics[f"speedrun_reached_{thresh_key}"] = 1.0 if reached else 0.0
                    tokens_val = multi_threshold_tokens.get(thresh)
                    metrics[f"speedrun_tokens_{thresh_key}"] = (
                        tokens_val if tokens_val is not None else missing_penalty
                    )
                    flops_val = multi_threshold_flops.get(thresh)
                    metrics[f"speedrun_flops_{thresh_key}"] = (
                        flops_val if flops_val is not None else missing_penalty_flops
                    )

                # Count how many thresholds were reached
                metrics["speedrun_thresholds_reached"] = float(
                    sum(1 for t in multi_thresholds if multi_threshold_reached.get(t, False))
                )
        if spec.parameter_golf is not None:
            metrics.update(
                self._evaluate_parameter_golf_metrics(
                    model,
                    spec,
                    recurrence_steps=current_recurrence_steps,
                )
            )
        metrics.update(passkey_metrics)

        # Run downstream probes if enabled
        if bool(getattr(spec.train, "downstream_probes", False)):
            n_examples = int(getattr(spec.train, "downstream_probe_examples", 5) or 5)
            try:
                # Need to reload model for probes (it was deleted for memory)
                probe_model = EvolutionModel(spec.model).to(self.device)
                probe_model.eval()
                state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                probe_model.load_state_dict(state, strict=False)
                downstream_metrics = run_downstream_probes(
                    probe_model, spec, self.device, n_examples=n_examples
                )
                metrics.update(downstream_metrics)
                del probe_model
            except Exception:
                metrics.update(
                    {
                        "code_probe_acc": 0.0,
                        "math_probe_acc": 0.0,
                        "recall_probe_acc": 0.0,
                        "downstream_avg_acc": 0.0,
                        "downstream_error": 1.0,
                    }
                )
        if spec.model.recurrences:
            metrics.update(self._recurrence_evaluations(model, spec, batch_iter, criterion))
        result = (metrics, checkpoint_path)
        # Best-effort cleanup to prevent allocator growth across many candidates.
        del parent_state
        del optimizer
        del criterion
        del model
        gc.collect()
        if (
            self.device.type == "cuda"
            and torch.backends.cuda.is_built()
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        if (
            self.device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()
        return result

    def _recurrence_schedule(
        self, spec: ArchitectureSpec, step_idx: int, total_steps: int
    ) -> dict[int, int]:
        schedule: dict[int, int] = {}
        if not spec.model.recurrences:
            return schedule
        progress = step_idx / max(1, total_steps)
        for idx, cfg in enumerate(spec.model.recurrences):
            base = max(1, cfg.train_recurrence)
            target = max(1, cfg.max_train_recurrence)
            frac = cfg.curriculum_fraction
            if frac > 0 and progress < frac:
                ratio = progress / frac
                steps = int(round(base + (target - base) * ratio))
            else:
                steps = target
            schedule[idx] = max(1, steps)
        return schedule

    def _evaluate_perplexity(
        self,
        model: nn.Module,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        criterion: nn.Module,
    ) -> float:
        loss = self._evaluate_loss(
            model, spec, batch_iter, criterion, eval_batches=self.eval_batches, empty_value=1e9
        )
        if not math.isfinite(loss) or loss >= 1e8:
            return 1e9
        return float(torch.exp(torch.tensor(loss)).item())

    def _evaluate_loss(
        self,
        model: nn.Module,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        criterion: nn.Module,
        *,
        eval_batches: int,
        empty_value: float,
    ) -> float:
        eval_loss = 0.0
        batches = 0
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
            with torch.no_grad():
                iterator = iter(batch_iter)
                for _ in range(max(1, int(eval_batches))):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    input_ids = batch.input_ids.to(self.device)
                    attn_mask = batch.attention_mask.to(self.device)
                    target_ids = (
                        batch.target_ids.to(self.device) if batch.target_ids is not None else None
                    )
                    logits = model(input_ids)
                    if target_ids is None and logits.size(1) <= 1:
                        continue
                    if target_ids is None:
                        shifted_logits = logits[:, :-1, :].contiguous()
                        labels = input_ids[:, 1:].contiguous()
                        label_mask = attn_mask[:, 1:].contiguous()
                    else:
                        shifted_logits = logits.contiguous()
                        labels = target_ids.contiguous()
                        label_mask = attn_mask.contiguous()
                    labels = labels.masked_fill(label_mask == 0, -100)
                    loss = criterion(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        labels.view(-1),
                    )
                    loss_val = float(loss.item())
                    if not math.isfinite(loss_val):
                        return 1e9
                    eval_loss += loss_val
                    batches += 1
        finally:
            if was_training:
                model.train()
        if batches == 0:
            return float(empty_value)
        return float(eval_loss / batches)

    def _recurrence_evaluations(
        self,
        model: nn.Module,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        criterion: nn.Module,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not spec.model.recurrences:
            return metrics
        rec_values = spec.model.recurrences[0].test_recurrences or [1]
        rec_values = sorted({max(1, v) for v in rec_values})
        base_value = rec_values[0]
        best_value = rec_values[-1]
        base_ppl = None
        best_ppl = None
        for value in rec_values:
            steps = dict.fromkeys(range(len(spec.model.recurrences)), value)
            if isinstance(model, EvolutionModel):
                model.set_recurrence_steps(steps)
            ppl = self._evaluate_perplexity(model, spec, batch_iter, criterion)
            metrics[f"ppl_code_rec_{value}"] = ppl
            if value == base_value:
                base_ppl = ppl
            if value == best_value:
                best_ppl = ppl
        if base_ppl is not None and best_ppl is not None:
            metrics["recurrence_gain"] = float(base_ppl - best_ppl)
        return metrics

    def _passkey_probe(self, model: nn.Module, spec: ArchitectureSpec) -> dict[str, float]:
        steps = int(getattr(spec.train, "passkey_eval_steps", 0) or 0)
        if steps <= 0:
            return {}
        try:
            eval_batches = int(getattr(spec.train, "passkey_eval_batches", 8) or 8)
            seq_len = int(getattr(spec.train, "passkey_eval_seq_len", None) or spec.data.seq_len)
            seq_len = max(4, seq_len)
            min_distance = int(getattr(spec.train, "passkey_eval_min_distance", 0) or 0)
            lr = float(getattr(spec.train, "passkey_eval_lr", None) or spec.train.lr)
            batch_size = int(
                getattr(spec.train, "passkey_eval_batch_size", None) or spec.data.batch_size
            )
            batch_size = max(1, batch_size)
            vocab = int(spec.model.head.vocab)
            if vocab < 8:
                return {"passkey_acc": 0.0, "passkey_loss": 1e9, "passkey_error": 1.0}

            seed_val = int(getattr(spec.train, "seed", 0) or 0)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed_val + 17)

            query_marker_id = vocab - 2
            noise_vocab = max(1, vocab - 3)
            vocab_limit = getattr(spec.train, "passkey_eval_vocab_limit", None)
            if vocab_limit is not None:
                try:
                    limit = int(vocab_limit)
                    if limit > 0:
                        noise_vocab = max(1, min(noise_vocab, limit))
                except (TypeError, ValueError):
                    pass
            criterion = nn.CrossEntropyLoss()

            def sample_batch() -> TokenBatch:
                ids = torch.randint(
                    0,
                    noise_vocab,
                    (batch_size, seq_len),
                    generator=generator,
                    dtype=torch.long,
                )
                passkey = torch.randint(
                    0,
                    noise_vocab,
                    (batch_size,),
                    generator=generator,
                    dtype=torch.long,
                )
                # Long-context copy task: place the passkey at position 0 and query at the end.
                #
                # This is intentionally easy-to-learn in short probes and still stresses
                # long-range credit assignment as seq_len grows.
                #
                # NOTE: `min_distance` is kept for config compatibility; position 0 always
                # satisfies it as long as seq_len is sufficiently large.
                if min_distance > 0 and (seq_len - 2) < min_distance:
                    raise ValueError("passkey_eval_min_distance exceeds available context length")
                ids[:, 0] = passkey
                ids[:, -2] = int(query_marker_id)
                ids[:, -1] = passkey
                attn = torch.ones_like(ids)
                return TokenBatch(input_ids=ids, attention_mask=attn, uids=["passkey"] * batch_size)

            probe_model = EvolutionModel(spec.model).to(self.device)
            probe_model.train()
            probe_model.set_grad_checkpointing(False)
            with torch.no_grad():
                probe_model.load_state_dict(model.state_dict(), strict=False)

            probe_schedule = spec.train.model_copy(deep=True)
            probe_schedule.lr = lr
            probe_schedule.weight_decay = 0.0
            optimizer = build_optimizer(probe_model.parameters(), probe_schedule)

            for _ in range(steps):
                batch = sample_batch()
                input_ids = batch.input_ids.to(self.device)
                logits = probe_model(input_ids)
                if logits.size(1) < 2:
                    continue
                pred = logits[:, -2, :noise_vocab]
                targets = input_ids[:, -1]
                loss = criterion(pred, targets)
                if not math.isfinite(float(loss.item())):
                    raise ValueError("passkey_probe loss is non-finite")
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    probe_model.parameters(), float(getattr(spec.train, "clip", 1.0) or 1.0)
                )
                optimizer.step()

            correct = 0
            total = 0
            eval_loss = 0.0
            probe_model.eval()
            with torch.no_grad():
                for _ in range(eval_batches):
                    batch = sample_batch()
                    input_ids = batch.input_ids.to(self.device)
                    logits = probe_model(input_ids)
                    if logits.size(1) < 2:
                        continue
                    pred = logits[:, -2, :noise_vocab]
                    targets = input_ids[:, -1]
                    batch_loss = float(criterion(pred, targets).item())
                    if not math.isfinite(batch_loss):
                        raise ValueError("passkey_probe eval loss is non-finite")
                    eval_loss += batch_loss
                    predicted = pred.argmax(dim=-1)
                    correct += int((predicted == targets).sum().item())
                    total += int(targets.numel())
            acc = float(correct) / float(max(1, total))
            loss_val = eval_loss / float(max(1, eval_batches))

            # Run multi-needle probes on the trained probe model
            multi_needle_results = run_multi_needle_probes(
                probe_model, spec, self.device, probe_types=["first", "middle", "last", "pattern"]
            )

            results = {
                "passkey_acc": acc,
                "passkey_loss": loss_val,
            }
            results.update(multi_needle_results)
            return results
        except Exception:
            if (
                self.device.type == "cuda"
                and torch.backends.cuda.is_built()
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
            if (
                self.device.type == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
            return {"passkey_acc": 0.0, "passkey_loss": 1e9, "passkey_error": 1.0}
