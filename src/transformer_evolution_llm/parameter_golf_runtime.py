"""Runtime helpers for Parameter Golf benchmark and exported workspaces."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .candidates import Candidate
from .dsl import load_architecture_spec
from .models import EvolutionModel
from .parameter_golf_export import build_official_submission_plan
from .parameter_golf import (
    ParameterGolfDataModule,
    artifact_size_calibration_table,
    estimate_calibrated_artifact_total_bytes_for_spec,
    estimate_artifact_total_bytes_for_spec,
    resolve_parameter_golf_glob,
    resolve_parameter_golf_path,
)
from .parameter_golf_seeded import seed_lane_metadata
from .trainer import FullWeightTrainer


def preflight_parameter_golf_config(config_path: str | Path) -> dict[str, Any]:
    """Resolve challenge inputs and report the current size picture before running."""
    spec = load_architecture_spec(config_path)
    if spec.parameter_golf is None:
        raise ValueError("config must include a parameter_golf block.")

    train_shards = resolve_parameter_golf_glob(spec.parameter_golf.train_shards_glob)
    val_shards = resolve_parameter_golf_glob(spec.parameter_golf.val_shards_glob)
    tokenizer_path = resolve_parameter_golf_path(spec.parameter_golf.tokenizer_path)
    payload_bytes_est, total_bytes_est = estimate_artifact_total_bytes_for_spec(spec)
    calibrated_payload_est, calibrated_total_est = estimate_calibrated_artifact_total_bytes_for_spec(
        spec
    )
    budget = int(spec.parameter_golf.artifact_budget_bytes)
    max_est_gate = spec.evolution.rung0_thresholds.get("max_artifact_total_bytes_est")
    max_exact_gate = spec.evolution.rung0_thresholds.get("max_artifact_total_bytes")
    micro_batch_tokens = int(spec.data.seq_len) * int(spec.data.batch_size)
    target_batch_tokens = int(getattr(spec.train, "batch_tokens", 0) or micro_batch_tokens)
    grad_accum_steps = max(1, (target_batch_tokens + micro_batch_tokens - 1) // micro_batch_tokens)
    official_plan = build_official_submission_plan(spec)

    return {
        "config": str(Path(config_path).resolve()),
        "track": str(spec.parameter_golf.track),
        **seed_lane_metadata(spec),
        "train_shard_count": len(train_shards),
        "val_shard_count": len(val_shards),
        "resolved_train_glob_first": str(train_shards[0]),
        "resolved_val_glob_first": str(val_shards[0]),
        "resolved_tokenizer_path": str(tokenizer_path),
        "artifact_payload_bytes_est": int(payload_bytes_est),
        "artifact_total_bytes_est": int(total_bytes_est),
        "artifact_payload_bytes_calibrated_est": int(calibrated_payload_est),
        "artifact_total_bytes_calibrated_est": int(calibrated_total_est),
        "artifact_budget_bytes": budget,
        "estimated_over_budget_bytes": max(0, int(total_bytes_est) - budget),
        "estimated_over_budget_bytes_calibrated": max(0, int(calibrated_total_est) - budget),
        "estimated_within_budget": bool(total_bytes_est <= budget),
        "estimated_within_budget_calibrated": bool(calibrated_total_est <= budget),
        "train_micro_batch_tokens": micro_batch_tokens,
        "train_target_batch_tokens": target_batch_tokens,
        "grad_accum_steps_est": grad_accum_steps,
        "parameter_golf_val_batch_tokens": int(
            spec.parameter_golf.val_batch_tokens or spec.data.eval_tokens or target_batch_tokens
        ),
        "parameter_golf_val_loss_every": int(spec.parameter_golf.val_loss_every),
        "parameter_golf_train_log_every": int(spec.parameter_golf.train_log_every),
        "tied_embedding_export_dtype": str(spec.parameter_golf.tied_embedding_export_dtype),
        "export_quant_mode": str(spec.parameter_golf.export_quant_mode),
        "eval_protocol": str(spec.parameter_golf.eval_protocol),
        "report_eval_modes": list(spec.parameter_golf.report_eval_modes),
        "parameter_golf_max_wallclock_seconds": (
            float(spec.parameter_golf.max_wallclock_seconds)
            if spec.parameter_golf.max_wallclock_seconds is not None
            else (600.0 if str(spec.parameter_golf.track).lower() == "10min" else None)
        ),
        "gate0_max_artifact_total_bytes_est": (
            float(max_est_gate) if max_est_gate is not None else None
        ),
        "gate0_max_artifact_total_bytes": (
            float(max_exact_gate) if max_exact_gate is not None else None
        ),
        "official_submission": official_plan,
        "artifact_calibration": artifact_size_calibration_table(),
        "parameter_golf_root": os.environ.get("TEVO_PARAMETER_GOLF_ROOT"),
        "packed_root": os.environ.get("TEVO_PACKED_ROOT"),
    }


def run_parameter_golf_benchmark(
    config_path: str | Path,
    *,
    out_path: str | Path | None = None,
    checkpoint_dir: str | Path = "runs/parameter_golf_checkpoints",
    steps: int | None = None,
    eval_batches: int = 2,
    device: str | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train one Parameter Golf spec and return a benchmark summary."""
    spec = load_architecture_spec(config_path)
    if spec.parameter_golf is None:
        raise ValueError("config must include a parameter_golf block.")
    if seed is not None:
        spec.train.seed = int(seed)

    seed_val = int(getattr(spec.train, "seed", 0) or 0)
    data_module = ParameterGolfDataModule(
        spec.parameter_golf,
        seq_len=spec.data.seq_len,
        batch_size=spec.data.batch_size,
        seed=seed_val,
    )
    token_budget = _resolve_benchmark_token_budget(spec, steps=steps, max_tokens=max_tokens)
    trainer = FullWeightTrainer(
        checkpoint_dir=Path(checkpoint_dir),
        device=device,
        steps=int(steps) if steps is not None else 50,
        eval_batches=eval_batches,
        entropy_threshold=spec.train.entropy_threshold,
        entropy_patience=spec.train.entropy_patience,
        instability_threshold=spec.train.instability_threshold,
        no_improve_patience=spec.train.no_improve_patience,
        improvement_tolerance=spec.train.improvement_tolerance,
    )

    started = time.time()
    candidate = Candidate(ident="parameter-golf-benchmark", spec=spec)
    preflight = preflight_parameter_golf_config(config_path)
    metrics, checkpoint = trainer.train(
        candidate,
        spec,
        data_module.batches(max_tokens=token_budget, split="train"),
    )
    duration_s = time.time() - started
    summary = {
        "config": str(Path(config_path).resolve()),
        "duration_s": float(duration_s),
        "steps": int(trainer.steps),
        "eval_batches": int(eval_batches),
        "token_budget": token_budget,
        "checkpoint": str(checkpoint),
        "preflight": preflight,
        "metrics": metrics,
    }
    if trainer.last_parameter_golf_error:
        summary["parameter_golf_error_message"] = trainer.last_parameter_golf_error

    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
    return summary


def _load_checkpoint_payload(
    checkpoint_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")  # nosec B614
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload at {checkpoint_path}")
    state_dict = payload.get("model_state")
    if isinstance(state_dict, dict):
        recurrence_steps = payload.get("recurrence_steps")
        if not isinstance(recurrence_steps, dict):
            recurrence_steps = {}
        return state_dict, {int(key): int(value) for key, value in recurrence_steps.items()}
    if all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        return payload, {}
    raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a readable state dict.")


def rescore_parameter_golf_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    out_path: str | Path | None = None,
    device: str | None = None,
    val_batch_tokens: int | None = None,
    eval_protocol: str | None = None,
) -> dict[str, Any]:
    """Re-run exact Parameter Golf scoring for an already trained checkpoint."""
    spec = load_architecture_spec(config_path)
    if spec.parameter_golf is None:
        raise ValueError("config must include a parameter_golf block.")
    if val_batch_tokens is not None:
        spec.parameter_golf.val_batch_tokens = int(val_batch_tokens)
    if eval_protocol is not None:
        spec.parameter_golf.eval_protocol = str(eval_protocol)

    trainer = FullWeightTrainer(device=device, steps=1, eval_batches=1)
    state_dict, recurrence_steps = _load_checkpoint_payload(checkpoint_path)
    model = EvolutionModel(spec.model).to(trainer.device)
    model.load_state_dict(state_dict, strict=False)

    started = time.time()
    metrics = trainer._evaluate_parameter_golf_metrics(
        model,
        spec,
        recurrence_steps=recurrence_steps,
    )
    duration_s = time.time() - started
    summary = {
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "duration_s": float(duration_s),
        "preflight": preflight_parameter_golf_config(config_path),
        "metrics": metrics,
    }
    if trainer.last_parameter_golf_error:
        summary["parameter_golf_error_message"] = trainer.last_parameter_golf_error
    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
    return summary


def _resolve_benchmark_token_budget(
    spec: Any,
    *,
    steps: int | None,
    max_tokens: int | None,
) -> int | None:
    """Pick a token budget that can actually cover the requested benchmark steps.

    If the caller explicitly provides ``max_tokens``, respect it. Otherwise, when
    ``steps`` is provided, make sure the budget can cover at least
    ``steps * batch_size * seq_len`` tokens so the run does not stop early just
    because the config carries a small search-time cap.
    """
    if max_tokens is not None:
        return int(max_tokens)

    config_budget = getattr(spec.train, "max_tokens", None)
    if steps is None:
        return int(config_budget) if config_budget is not None else None

    tokens_per_step = int(spec.data.seq_len) * int(spec.data.batch_size)
    target_tokens = int(getattr(spec.train, "batch_tokens", 0) or tokens_per_step)
    tokens_per_step = max(tokens_per_step, target_tokens)
    required_budget = int(steps) * tokens_per_step
    if config_budget is None:
        return required_budget
    return max(int(config_budget), required_budget)


def main(
    *,
    config_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> None:
    """CLI entrypoint used by exported workspaces and local scripts."""
    if config_path is None:
        parser = argparse.ArgumentParser(description="Run a TEVO Parameter Golf benchmark.")
        parser.add_argument("config", type=Path, help="Path to a TEVO spec with parameter_golf.")
        parser.add_argument("--out", type=Path, default=Path("runs/parameter_golf_benchmark.json"))
        parser.add_argument(
            "--checkpoint-dir", type=Path, default=Path("runs/parameter_golf_checkpoints")
        )
        parser.add_argument(
            "--preflight-only",
            action="store_true",
            help="Resolve data paths and print size estimates without training.",
        )
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--eval-batches", type=int, default=2)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--max-tokens", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()
        if args.preflight_only:
            print(json.dumps(preflight_parameter_golf_config(args.config), indent=2))
            return
        summary = run_parameter_golf_benchmark(
            args.config,
            out_path=args.out,
            checkpoint_dir=args.checkpoint_dir,
            steps=args.steps,
            eval_batches=args.eval_batches,
            device=args.device,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return

    summary = run_parameter_golf_benchmark(config_path, out_path=out_path)
    print(json.dumps(summary, indent=2))


__all__ = [
    "main",
    "preflight_parameter_golf_config",
    "rescore_parameter_golf_checkpoint",
    "run_parameter_golf_benchmark",
]
