"""Live training loop for mutated architectures."""

from __future__ import annotations

import gc
import json as std_json
import math
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .candidates import Candidate
from .data import DataModule, TokenBatch
from .dsl import ArchitectureSpec
from .models import BranchRouter, EvolutionModel, MoELayer, count_parameters
from .morphology import match_experts_to_parent, sort_moe_experts
from .optimizers import build_optimizer


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
        self._eval_module_cache: dict[str, DataModule] = {}

    def _get_eval_module(self, spec: ArchitectureSpec) -> tuple[DataModule, int]:
        """Return a deterministic eval DataModule + token budget.

        If `data.eval_shards` is provided, use it. Otherwise, reuse the training
        shards but sample with a different RNG seed so `ppl_eval` isn't identical
        to the training stream.
        """
        eval_shards = getattr(spec.data, "eval_shards", []) or []
        eval_cfg = spec.data.model_copy(deep=True)
        if eval_shards:
            eval_cfg.shards = list(eval_shards)
        eval_cfg.healing_shards = []
        eval_cfg.healing_tokens = None

        eval_tokens_cfg = getattr(spec.data, "eval_tokens", None)
        if isinstance(eval_tokens_cfg, int) and eval_tokens_cfg > 0:
            eval_tokens = int(eval_tokens_cfg)
        else:
            eval_tokens = (
                int(spec.data.seq_len) * int(spec.data.batch_size) * max(1, self.eval_batches)
            )

        seed_val = int(getattr(spec.train, "seed", 0) or 0)
        cache_key = std_json.dumps(eval_cfg.model_dump(mode="python"), sort_keys=True)
        cache_key = f"{seed_val + 1}:{cache_key}"
        eval_module = self._eval_module_cache.get(cache_key)
        if eval_module is None:
            eval_module = DataModule(eval_cfg, seed=seed_val + 1)
            self._eval_module_cache[cache_key] = eval_module
        eval_module.reset_rng(seed_val + 1)
        return eval_module, eval_tokens

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
        torch.manual_seed(int(getattr(spec.train, "seed", 0) or 0))
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
        optimizer = build_optimizer(model.parameters(), spec.train)
        base_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
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
        autocast_ctx: Any = nullcontext()
        if self.device.type in {"cuda", "mps"}:
            dtype = torch.bfloat16 if bool(getattr(spec.train, "bf16", True)) else torch.float16
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=dtype)
        for step_idx in range(self.steps):
            if spec.model.recurrences:
                model.set_recurrence_steps(self._recurrence_schedule(spec, step_idx, total_steps))
            try:
                batch = next(iterator)
            except StopIteration:
                stop_reason = (
                    "token_budget" if getattr(batch_iter, "max_tokens", None) else "data_exhausted"
                )
                break
            warmup = int(getattr(spec.train, "warmup", 0) or 0)
            if warmup > 0:
                scale = min(1.0, float(step_idx + 1) / float(warmup))
                for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base_lr * scale
            input_ids = batch.input_ids.to(self.device)
            attn_mask = batch.attention_mask.to(self.device)
            with autocast_ctx:
                logits = model(input_ids)
                # Next-token prediction (causal LM): predict input_ids[t+1] from input_ids[:t+1]
                if logits.size(1) <= 1:
                    stop_reason = "data_exhausted"
                    break
                shifted_logits = logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()
                label_mask = attn_mask[:, 1:].contiguous()
                labels = labels.masked_fill(label_mask == 0, -100)
                loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1))
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
            loss.backward()
            grad_total = clip_grad_norm_(model.parameters(), spec.train.clip)
            if hasattr(grad_total, "item"):
                grad_norm = float(grad_total.item())
            else:
                grad_norm = float(grad_total)
            if not math.isfinite(grad_norm):
                nan_or_inf = True
                stop_reason = "nan_or_inf_grad"
                optimizer.zero_grad()
                break
            if grad_norm > self.instability_threshold:
                stop_reason = f"high_grad({grad_norm:.2f})"
                optimizer.zero_grad()
                break
            optimizer.step()
            optimizer.zero_grad()
            try:
                tokens_seen += int(attn_mask.sum().item())
            except Exception:
                tokens_seen += input_ids.numel()
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
        if nan_or_inf:
            raise RuntimeError(f"Non-finite loss/grad encountered ({stop_reason}).")
        duration = max(time.perf_counter() - start_time, 1e-6)
        throughput = tokens_seen / duration
        model.set_recurrence_steps(self._recurrence_schedule(spec, self.steps, total_steps))
        ppl_train = self._evaluate_perplexity(model, spec, batch_iter, criterion)
        ppl_eval = ppl_train
        ppl_eval_error = 0.0
        try:
            eval_module, eval_tokens = self._get_eval_module(spec)
            ppl_eval = self._evaluate_perplexity(
                model,
                spec,
                eval_module.batches(max_tokens=eval_tokens),
                criterion,
            )
        except Exception:
            # Best-effort: if eval dataset shards are unavailable (common in unit tests
            # or offline environments), fall back to the training-stream perplexity.
            ppl_eval = ppl_train
            ppl_eval_error = 1.0
        perplexity = ppl_eval
        long_recall_proxy = _estimate_long_recall(spec)
        passkey_metrics = self._passkey_probe(model, spec)
        long_recall = float(passkey_metrics.get("passkey_acc", long_recall_proxy))
        checkpoint_path = self.checkpoint_dir / f"{candidate.ident}.pt"
        state = self._checkpoint_state(model)
        torch.save(state, checkpoint_path)
        del state
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
                if isinstance(mod, (MoELayer, BranchRouter)):
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
        moe_penalty = 1.0 + 0.01 * spec.model.moe_block_count()
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
        }
        metrics.update(passkey_metrics)
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
        eval_loss = 0.0
        eval_batches = 0
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
            with torch.no_grad():
                iterator = iter(batch_iter)
                for _ in range(self.eval_batches):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    input_ids = batch.input_ids.to(self.device)
                    attn_mask = batch.attention_mask.to(self.device)
                    logits = model(input_ids)
                    if logits.size(1) <= 1:
                        continue
                    shifted_logits = logits[:, :-1, :].contiguous()
                    labels = input_ids[:, 1:].contiguous()
                    label_mask = attn_mask[:, 1:].contiguous()
                    labels = labels.masked_fill(label_mask == 0, -100)
                    loss = criterion(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        labels.view(-1),
                    )
                    loss_val = float(loss.item())
                    if not math.isfinite(loss_val):
                        return 1e9
                    eval_loss += loss_val
                    eval_batches += 1
        finally:
            if was_training:
                model.train()
        if eval_batches == 0:
            return 1.0
        return float(torch.exp(torch.tensor(eval_loss / eval_batches)).item())

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
            return {
                "passkey_acc": acc,
                "passkey_loss": loss_val,
            }
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


def _average_router_entropy(model: nn.Module) -> float | None:
    with torch.no_grad():
        entropies = []
        for mod in model.modules():
            if isinstance(mod, (MoELayer, BranchRouter)) and hasattr(mod, "last_entropy"):
                last = getattr(mod, "last_entropy", None)
                if isinstance(last, torch.Tensor):
                    entropies.append(float(last.item()))
    if not entropies:
        return None
    return sum(entropies) / len(entropies)


def _estimate_long_recall(spec: ArchitectureSpec) -> float:
    layers = max(1, spec.model.n_layers)
    memory_blocks = sum(
        1
        for block in spec.model.blocks
        for extra in block.extras
        if getattr(extra, "type", None)
        in {"retro", "assoc_memory", "memory_tokens", "chunk_memory"}
    )
    ssm_blocks = sum(1 for block in spec.model.blocks if block.ssm is not None)
    rec_spans = len(spec.model.recurrences)
    extra_types = {
        getattr(extra, "type", type(extra).__name__)
        for block in spec.model.blocks
        for extra in block.extras
    }
    density = (memory_blocks + 0.5 * ssm_blocks + 0.5 * rec_spans) / layers
    diversity_bonus = 0.1 * len(extra_types)
    return float(min(2.0, density + diversity_bonus))
