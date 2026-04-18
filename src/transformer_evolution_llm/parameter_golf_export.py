"""Parameter Golf export helpers and official-style submission planning."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import ujson as json

from .dsl import ArchitectureSpec, DenseFFNConfig, load_architecture_spec, save_architecture_spec
from .parameter_golf import (
    estimate_artifact_total_bytes_for_spec,
    estimate_calibrated_artifact_total_bytes_for_spec,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_BASE_CODE_BYTES = 47_642
OFFICIAL_EXPORT_METADATA = "parameter_golf_export.json"
SUPPORTED_OFFICIAL_PATCHES = {
    "fp16_tied_embedding_export",
    "muon_weight_decay",
    "sliding64_eval",
}


class ParameterGolfExportError(ValueError):
    """Raised when a spec cannot be exported to the Parameter Golf workspace format."""


def load_parameter_golf_export_spec(
    source_path: str | Path,
    *,
    candidate_id: str | None = None,
) -> ArchitectureSpec:
    """Load a spec directly or from a frontier entry."""
    path = Path(source_path)
    if path.suffix != ".json":
        return load_architecture_spec(path)

    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return load_architecture_spec(path)

    if candidate_id is None:
        if len(payload) != 1:
            raise ParameterGolfExportError(
                "Frontier export requires --candidate-id when the frontier has multiple entries."
            )
        entry = payload[0]
    else:
        entry = next((row for row in payload if row.get("id") == candidate_id), None)
        if entry is None:
            raise ParameterGolfExportError(
                f"Candidate {candidate_id!r} was not found in frontier {path}."
            )
    spec_data = entry.get("spec")
    if not isinstance(spec_data, dict):
        raise ParameterGolfExportError("Frontier entry is missing a spec payload.")
    return ArchitectureSpec(**spec_data)


def resolve_official_train_gpt_path(path: str | Path | None = None) -> Path | None:
    """Locate a local copy of the official Parameter Golf `train_gpt.py`."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get("TEVO_PARAMETER_GOLF_OFFICIAL_TRAIN_PY")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    env_root = os.environ.get("TEVO_PARAMETER_GOLF_OFFICIAL_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / "train_gpt.py")

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents[:4]):
        candidates.append(parent / "parameter-golf" / "train_gpt.py")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _estimated_payload_bytes(spec: ArchitectureSpec) -> int:
    payload_est, total_est = estimate_calibrated_artifact_total_bytes_for_spec(spec)
    code_bytes = int(spec.parameter_golf.code_bytes) if spec.parameter_golf is not None else 0
    if total_est >= payload_est:
        return max(0, int(total_est) - code_bytes)
    return int(payload_est)


def _uniform_values(values: Iterable[Any]) -> list[Any]:
    ordered: list[Any] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _common_block_errors(spec: ArchitectureSpec, *, fatal: list[str]) -> None:
    if getattr(spec.model, "bigram_hash", None) is not None:
        fatal.append("bigram_hash is not supported by the official submission lane")
    if spec.model.hyper is not None:
        fatal.append("hyper-connections are not supported by the official submission lane")
    if spec.model.kv_policy is not None:
        fatal.append("kv_policy is not supported by the official submission lane")
    if spec.model.recurrences:
        fatal.append("recurrence is not supported by the official submission lane")
    if any(block.share_with is not None for block in spec.model.blocks):
        fatal.append("shared blocks are not supported by the official submission lane")


def _supported_patch_reasons(spec: ArchitectureSpec) -> list[str]:
    if spec.parameter_golf is None:
        return []
    reasons: list[str] = []
    tied_export_dtype = str(spec.parameter_golf.tied_embedding_export_dtype or "int8").lower()
    if tied_export_dtype == "fp16":
        reasons.append("fp16_tied_embedding_export")
    report_modes = list(getattr(spec.parameter_golf, "report_eval_modes", None) or [])
    if "sliding64" in report_modes:
        reasons.append("sliding64_eval")

    effective_weight_decay = float(
        spec.train.optimizer.weight_decay
        if spec.train.optimizer.weight_decay is not None
        else spec.train.weight_decay
    )
    if str(spec.train.optimizer.name or "adamw").lower() == "muon" and effective_weight_decay > 0.0:
        reasons.append("muon_weight_decay")
    return reasons


def _unsupported_patch_reasons(spec: ArchitectureSpec) -> list[str]:
    reasons: list[str] = []
    grad_transform_mode = str(
        getattr(spec.train.optimizer.gradient_transform, "mode", "identity") or "identity"
    ).lower()
    if grad_transform_mode != "identity":
        reasons.append(f"gradient_transform={grad_transform_mode}")
    update_filter_mode = str(
        getattr(spec.train.optimizer.update_filter, "mode", "none") or "none"
    ).lower()
    if update_filter_mode != "none":
        reasons.append(f"update_filter={update_filter_mode}")
    quant_mode = str(getattr(spec.parameter_golf, "export_quant_mode", "int8") or "int8").lower()
    if quant_mode != "int8":
        reasons.append(f"export_quant_mode={quant_mode}")
    return reasons


def build_official_submission_plan(
    spec: ArchitectureSpec,
    *,
    official_train_py: str | Path | None = None,
    artifact_zlib_bytes: float | None = None,
) -> dict[str, Any]:
    """Project a TEVO spec into a compact official-style submission plan."""
    if spec.parameter_golf is None:
        raise ParameterGolfExportError("parameter_golf config is required for official export.")
    if not spec.model.blocks:
        raise ParameterGolfExportError("export requires at least one transformer block.")

    fatal_reasons: list[str] = []
    patch_reasons: list[str] = []
    _common_block_errors(spec, fatal=fatal_reasons)

    attn_blocks = []
    ffn_blocks = []
    for idx, block in enumerate(spec.model.blocks):
        if block.attn is None:
            fatal_reasons.append(f"block {idx} is missing attention")
            continue
        if block.ffn is None or not isinstance(block.ffn, DenseFFNConfig):
            fatal_reasons.append(f"block {idx} must use a dense FFN")
            continue
        if block.ffn_memory is not None:
            fatal_reasons.append(f"block {idx} uses ffn_memory")
        if block.ssm is not None:
            fatal_reasons.append(f"block {idx} uses SSM")
        if block.extras:
            fatal_reasons.append(f"block {idx} uses extra modules")
        attn_blocks.append(block.attn)
        ffn_blocks.append(block.ffn)

    attn_kinds = _uniform_values(str(attn.kind or "MHA").upper() for attn in attn_blocks)
    if len(attn_kinds) > 1:
        fatal_reasons.append("all blocks must share the same attention kind")
    if attn_kinds and attn_kinds[0] not in {"MHA", "GQA", "MQA"}:
        fatal_reasons.append(f"attention kind {attn_kinds[0]!r} is not supported")

    num_heads_values = _uniform_values(int(attn.heads) for attn in attn_blocks)
    if len(num_heads_values) > 1:
        fatal_reasons.append("all blocks must share the same head count")
    num_heads = num_heads_values[0] if num_heads_values else 0

    kv_groups_values = _uniform_values(int(attn.kv_groups or 1) for attn in attn_blocks)
    if len(kv_groups_values) > 1:
        fatal_reasons.append("all blocks must share the same kv_groups setting")
    kv_groups = kv_groups_values[0] if kv_groups_values else 1

    rope_values = _uniform_values(str(attn.rope or "standard").lower() for attn in attn_blocks)
    if len(rope_values) > 1:
        fatal_reasons.append("all blocks must share the same rope mode")
    elif rope_values and rope_values[0] not in {"standard", "none"}:
        fatal_reasons.append(f"rope mode {rope_values[0]!r} is not supported")

    rope_theta_values = _uniform_values(
        float(attn.rope_theta) if attn.rope_theta is not None else 10_000.0 for attn in attn_blocks
    )
    if len(rope_theta_values) > 1:
        fatal_reasons.append("all blocks must share the same rope_theta")

    softcap_values = _uniform_values(
        float(softmax.softcap)
        for attn in attn_blocks
        if (softmax := getattr(attn, "softmax", None)) is not None and softmax.softcap is not None
    )
    if len(softcap_values) > 1:
        fatal_reasons.append("all blocks must share the same softcap")
    if not softcap_values:
        fatal_reasons.append("official submission lane requires a positive softcap")

    qk_norm_values = _uniform_values(
        str(softmax.qk_norm or "none").lower()
        for attn in attn_blocks
        if (softmax := getattr(attn, "softmax", None)) is not None
    )
    if len(qk_norm_values) > 1:
        fatal_reasons.append("all blocks must share the same qk_norm policy")
    elif qk_norm_values and qk_norm_values[0] != "rms":
        fatal_reasons.append("official submission lane currently requires qk_norm=rms")

    for idx, attn in enumerate(attn_blocks):
        if attn.selector != "none" or attn.gating_pos != "none":
            fatal_reasons.append(f"block {idx} uses selector or gating")
        if bool(getattr(attn, "alibi", False)):
            fatal_reasons.append(f"block {idx} uses ALiBi")
        sparsity = str(getattr(attn, "sparsity", "none") or "none").lower()
        if sparsity != "none":
            fatal_reasons.append(f"block {idx} uses sparse attention")

    ffn_hidden_values = _uniform_values(int(ffn.hidden) for ffn in ffn_blocks)
    if len(ffn_hidden_values) > 1:
        fatal_reasons.append("all blocks must share the same FFN hidden size")
    ffn_activation_values = _uniform_values(
        str(getattr(ffn, "activation", "swiglu") or "swiglu").lower() for ffn in ffn_blocks
    )
    if len(ffn_activation_values) > 1:
        fatal_reasons.append("all blocks must share the same FFN activation")
    elif ffn_activation_values and ffn_activation_values[0] != "relu_squared":
        fatal_reasons.append("official submission lane currently requires relu_squared FFNs")

    model_dim = int(spec.model.emb.dim)
    hidden = ffn_hidden_values[0] if ffn_hidden_values else 0
    if model_dim <= 0 or hidden <= 0 or hidden % model_dim != 0:
        fatal_reasons.append("FFN hidden size must be an integer multiple of model_dim")
    mlp_mult = hidden // model_dim if model_dim > 0 else 0

    optimizer_name = str(getattr(spec.train.optimizer, "name", "adamw") or "adamw").lower()
    if optimizer_name != "muon":
        fatal_reasons.append("official submission lane currently requires optimizer=muon")
    supported_patch_reasons = _supported_patch_reasons(spec)
    unsupported_patch_reasons = _unsupported_patch_reasons(spec)
    patch_reasons.extend(supported_patch_reasons)
    patch_reasons.extend(unsupported_patch_reasons)

    env_overrides: dict[str, str] = {
        "SEED": str(int(getattr(spec.train, "seed", 0) or 0)),
        "VOCAB_SIZE": str(int(spec.model.head.vocab)),
        "NUM_LAYERS": str(int(spec.model.n_layers)),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(int(num_heads)),
        "NUM_KV_HEADS": str(
            int(
                num_heads
                if attn_kinds and attn_kinds[0] == "MHA"
                else max(1, num_heads // kv_groups)
            )
        ),
        "MLP_MULT": str(int(mlp_mult)),
        "TIE_EMBEDDINGS": "1" if bool(getattr(spec.model.head, "tie_embeddings", True)) else "0",
        "ROPE_BASE": str(float(rope_theta_values[0] if rope_theta_values else 10_000.0)),
        "LOGIT_SOFTCAP": str(float(softcap_values[0] if softcap_values else 30.0)),
        "WARMUP_STEPS": str(int(getattr(spec.train, "warmup", 0) or 0)),
        "WARMDOWN_ITERS": str(int(getattr(spec.train, "warmdown_steps", 0) or 0)),
        "TRAIN_SEQ_LEN": str(int(spec.data.seq_len)),
        "TRAIN_BATCH_TOKENS": str(
            int(
                getattr(spec.train, "batch_tokens", 0) or (spec.data.seq_len * spec.data.batch_size)
            )
        ),
        "GRAD_CLIP_NORM": str(float(getattr(spec.train, "clip", 0.0) or 0.0)),
        "MATRIX_LR": str(float(getattr(spec.train, "matrix_lr", None) or spec.train.lr)),
        "SCALAR_LR": str(float(getattr(spec.train, "scalar_lr", None) or spec.train.lr)),
    }
    if spec.parameter_golf.val_batch_tokens is not None or spec.data.eval_tokens is not None:
        env_overrides["VAL_BATCH_SIZE"] = str(
            int(
                spec.parameter_golf.val_batch_tokens
                or spec.data.eval_tokens
                or spec.data.seq_len * spec.data.batch_size
            )
        )
    if spec.parameter_golf.val_loss_every > 0:
        env_overrides["VAL_LOSS_EVERY"] = str(int(spec.parameter_golf.val_loss_every))
    if spec.parameter_golf.train_log_every > 0:
        env_overrides["TRAIN_LOG_EVERY"] = str(int(spec.parameter_golf.train_log_every))
    if "sliding64_eval" in supported_patch_reasons:
        env_overrides["EVAL_STRIDE"] = "64"

    max_tokens = getattr(spec.train, "max_tokens", None)
    batch_tokens = int(
        getattr(spec.train, "batch_tokens", 0) or (spec.data.seq_len * spec.data.batch_size)
    )
    if isinstance(max_tokens, int) and max_tokens > 0 and batch_tokens > 0:
        env_overrides["ITERATIONS"] = str(
            max(1, (int(max_tokens) + batch_tokens - 1) // batch_tokens)
        )

    wallclock_s = getattr(spec.parameter_golf, "max_wallclock_seconds", None)
    if wallclock_s is None and str(spec.parameter_golf.track).lower() == "10min":
        wallclock_s = 600.0
    if wallclock_s is not None:
        env_overrides["MAX_WALLCLOCK_SECONDS"] = str(float(wallclock_s))

    if bool(getattr(spec.model.head, "tie_embeddings", True)):
        tied_lr = getattr(spec.train, "tied_embedding_lr", None)
        if tied_lr is not None:
            env_overrides["TIED_EMBED_LR"] = str(float(tied_lr))
        env_overrides["TIED_EMBED_INIT_STD"] = str(
            float(getattr(spec.model.emb, "init_std", 0.02) or 0.02)
        )
    else:
        embed_lr = getattr(spec.train, "embed_lr", None)
        head_lr = getattr(spec.train, "head_lr", None)
        if embed_lr is not None:
            env_overrides["EMBED_LR"] = str(float(embed_lr))
        if head_lr is not None:
            env_overrides["HEAD_LR"] = str(float(head_lr))

    betas = getattr(spec.train.optimizer, "betas", None)
    if isinstance(betas, tuple) and len(betas) >= 2:
        env_overrides["BETA1"] = str(float(betas[0]))
        env_overrides["BETA2"] = str(float(betas[1]))
    optimizer_eps = getattr(spec.train.optimizer, "eps", None)
    if optimizer_eps is not None:
        env_overrides["ADAM_EPS"] = str(float(optimizer_eps))
    muon_momentum = getattr(spec.train.optimizer, "muon_momentum", None)
    if muon_momentum is not None:
        env_overrides["MUON_MOMENTUM"] = str(float(muon_momentum))
    if getattr(spec.train.optimizer, "muon_ns_steps", None) is not None:
        env_overrides["MUON_BACKEND_STEPS"] = str(int(spec.train.optimizer.muon_ns_steps))
    muon_warmup_start = getattr(spec.train.optimizer, "muon_momentum_warmup_start", None)
    if muon_warmup_start is not None:
        env_overrides["MUON_MOMENTUM_WARMUP_START"] = str(float(muon_warmup_start))
    if int(getattr(spec.train.optimizer, "muon_momentum_warmup_steps", 0) or 0) > 0:
        env_overrides["MUON_MOMENTUM_WARMUP_STEPS"] = str(
            int(spec.train.optimizer.muon_momentum_warmup_steps)
        )
    effective_weight_decay = float(
        spec.train.optimizer.weight_decay
        if spec.train.optimizer.weight_decay is not None
        else spec.train.weight_decay
    )
    if "muon_weight_decay" in supported_patch_reasons:
        env_overrides["MUON_WEIGHT_DECAY"] = str(effective_weight_decay)
    if "fp16_tied_embedding_export" in supported_patch_reasons:
        env_overrides["TIED_EMBED_EXPORT_DTYPE"] = "fp16"

    official_train_path = resolve_official_train_gpt_path(official_train_py)
    patch_bytes_est = 0
    if official_train_path is not None:
        base_text = official_train_path.read_text(encoding="utf-8")
        patched = _apply_supported_official_patches(base_text, supported_patch_reasons)
        patch_bytes_est = max(
            0,
            len(patched.encode("utf-8")) - len(base_text.encode("utf-8")),
        )
        code_bytes_est = int(
            len(_inject_official_env_overrides(patched, env_overrides).encode("utf-8"))
        )
    else:
        base_code_bytes = int(DEFAULT_OFFICIAL_BASE_CODE_BYTES)
        prelude = _official_env_prelude(env_overrides)
        patch_bytes_est = int(
            sum(_official_patch_size_estimate(reason) for reason in supported_patch_reasons)
        )
        code_bytes_est = int(base_code_bytes + len(prelude.encode("utf-8")) + patch_bytes_est)
    payload_bytes_est = _estimated_payload_bytes(spec)
    total_bytes_est = int(payload_bytes_est + code_bytes_est)
    code_allowance = int(spec.parameter_golf.code_bytes)
    artifact_budget = int(spec.parameter_golf.artifact_budget_bytes)
    code_over_budget = max(0, code_bytes_est - code_allowance)
    total_over_budget = max(0, total_bytes_est - artifact_budget)

    exact_total_bytes = None
    exact_over_budget = None
    if artifact_zlib_bytes is not None:
        exact_total_bytes = int(round(float(artifact_zlib_bytes))) + code_bytes_est
        exact_over_budget = max(0, exact_total_bytes - artifact_budget)

    exportable = not fatal_reasons and not unsupported_patch_reasons
    eligible_est = exportable and code_over_budget <= 0 and total_over_budget <= 0
    eligible_exact = (
        exportable
        and code_over_budget <= 0
        and exact_over_budget is not None
        and exact_over_budget <= 0
    )
    return {
        "mode": "official",
        "official_train_py": str(official_train_path) if official_train_path is not None else None,
        "env_overrides": env_overrides,
        "env_override_count": len(env_overrides),
        "fatal_reasons": fatal_reasons,
        "patch_reasons": patch_reasons,
        "supported_patch_reasons": supported_patch_reasons,
        "unsupported_patch_reasons": unsupported_patch_reasons,
        "requires_patch": bool(patch_reasons),
        "exportable": bool(exportable),
        "eligible_est": bool(eligible_est),
        "eligible_exact": bool(eligible_exact),
        "code_bytes_est": int(code_bytes_est),
        "patch_bytes_est": int(patch_bytes_est),
        "code_bytes_allowance": int(code_allowance),
        "code_over_budget_bytes_est": int(code_over_budget),
        "artifact_payload_bytes_est": int(payload_bytes_est),
        "artifact_total_bytes_est": int(total_bytes_est),
        "artifact_over_budget_bytes_est": int(total_over_budget),
        "artifact_budget_bytes": int(artifact_budget),
        "artifact_total_bytes_exact": (
            int(exact_total_bytes) if exact_total_bytes is not None else None
        ),
        "artifact_over_budget_bytes_exact": (
            int(exact_over_budget) if exact_over_budget is not None else None
        ),
    }


def official_submission_metrics(
    spec: ArchitectureSpec,
    *,
    official_train_py: str | Path | None = None,
    artifact_zlib_bytes: float | None = None,
) -> dict[str, float]:
    """Return numeric official-submission metrics for ranking and gating."""
    plan = build_official_submission_plan(
        spec,
        official_train_py=official_train_py,
        artifact_zlib_bytes=artifact_zlib_bytes,
    )
    metrics = {
        "official_submission_exportable": 1.0 if plan["exportable"] else 0.0,
        "official_submission_requires_patch": 1.0 if plan["requires_patch"] else 0.0,
        "official_submission_code_bytes_est": float(plan["code_bytes_est"]),
        "official_submission_patch_bytes_est": float(plan["patch_bytes_est"]),
        "official_submission_code_over_budget_bytes_est": float(plan["code_over_budget_bytes_est"]),
        "official_submission_total_bytes_est": float(plan["artifact_total_bytes_est"]),
        "official_submission_over_budget_bytes_est": float(plan["artifact_over_budget_bytes_est"]),
        "official_submission_env_override_count": float(plan["env_override_count"]),
        "main_track_eligible_est": 1.0 if plan["eligible_est"] else 0.0,
        "official_submission_fatal_reason_count": float(len(plan["fatal_reasons"])),
        "official_submission_patch_reason_count": float(len(plan["patch_reasons"])),
        "official_submission_supported_patch_reason_count": float(
            len(plan.get("supported_patch_reasons", []))
        ),
        "official_submission_unsupported_patch_reason_count": float(
            len(plan.get("unsupported_patch_reasons", []))
        ),
    }
    exact_total = plan.get("artifact_total_bytes_exact")
    exact_over = plan.get("artifact_over_budget_bytes_exact")
    if exact_total is not None:
        metrics["official_submission_total_bytes_exact"] = float(exact_total)
    if exact_over is not None:
        metrics["official_submission_over_budget_bytes_exact"] = float(exact_over)
        metrics["main_track_eligible_exact"] = 1.0 if plan["eligible_exact"] else 0.0
    return metrics


def _replace_once(base_text: str, needle: str, replacement: str, *, reason: str) -> str:
    if needle not in base_text:
        raise ParameterGolfExportError(
            f"official patch {reason!r} could not find its anchor in train_gpt.py"
        )
    return base_text.replace(needle, replacement, 1)


def _official_patch_size_estimate(reason: str) -> int:
    if reason == "fp16_tied_embedding_export":
        return 220
    if reason == "muon_weight_decay":
        return 320
    if reason == "sliding64_eval":
        return 0
    return 0


def _apply_supported_official_patches(base_text: str, patch_reasons: list[str]) -> str:
    rendered = base_text
    for reason in patch_reasons:
        if reason == "fp16_tied_embedding_export":
            rendered = _patch_fp16_tied_embedding_export(rendered)
            continue
        if reason == "muon_weight_decay":
            rendered = _patch_muon_weight_decay(rendered)
            continue
        if reason == "sliding64_eval":
            continue
        raise ParameterGolfExportError(f"unsupported official patch reason: {reason}")
    return rendered


def _patch_fp16_tied_embedding_export(base_text: str) -> str:
    header = (
        "def keep_float_tensor("
        "name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]"
        ") -> Tensor:\n"
    )
    insertion = "\n".join(
        [
            header.rstrip("\n"),
            '    tied_embed_export_dtype = os.environ.get("TIED_EMBED_EXPORT_DTYPE", "").lower()',
            '    if tied_embed_export_dtype == "fp16" and name == "tok_emb.weight":',
            "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()",
            "",
        ]
    )
    return _replace_once(
        base_text,
        header,
        insertion,
        reason="fp16_tied_embedding_export",
    )


def _patch_muon_weight_decay(base_text: str) -> str:
    hyper_line = '    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))\n'
    hyper_replacement = "\n".join(
        [
            hyper_line.rstrip("\n"),
            '    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))',
            "",
        ]
    )
    rendered = _replace_once(
        base_text,
        hyper_line,
        hyper_replacement,
        reason="muon_weight_decay:hyperparameters",
    )
    step_anchor = "\n".join(
        [
            "        if args.grad_clip_norm > 0:",
            (
                "            torch.nn.utils.clip_grad_norm_("
                "base_model.parameters(), args.grad_clip_norm)"
            ),
            "        for opt in optimizers:",
            "            opt.step()",
        ]
    )
    step_replacement = "\n".join(
        [
            "        if args.grad_clip_norm > 0:",
            (
                "            torch.nn.utils.clip_grad_norm_("
                "base_model.parameters(), args.grad_clip_norm)"
            ),
            "        if args.muon_weight_decay != 0.0:",
            "            with torch.no_grad():",
            "                for p in matrix_params:",
            "                    p.add_(p, alpha=-args.matrix_lr * scale * args.muon_weight_decay)",
            "        for opt in optimizers:",
            "            opt.step()",
        ]
    )
    return _replace_once(
        rendered,
        step_anchor,
        step_replacement,
        reason="muon_weight_decay:step",
    )


def _official_env_prelude(env_overrides: dict[str, str]) -> str:
    overrides_blob = json.dumps(env_overrides, sort_keys=True, indent=2)
    return "\n".join(
        [
            "",
            "# TEVO official-style environment overrides",
            "import os",
            "_TEVO_ENV_OVERRIDES = " + overrides_blob,
            "for _tevo_key, _tevo_value in _TEVO_ENV_OVERRIDES.items():",
            "    os.environ.setdefault(_tevo_key, str(_tevo_value))",
            "",
        ]
    )


def _inject_official_env_overrides(base_script: str, env_overrides: dict[str, str]) -> str:
    marker = "from __future__ import annotations\n"
    prelude = _official_env_prelude(env_overrides)
    if marker in base_script:
        idx = base_script.index(marker) + len(marker)
        return base_script[:idx] + prelude + base_script[idx:]
    return prelude + base_script


def _validate_tevo_export_spec(spec: ArchitectureSpec) -> None:
    if spec.parameter_golf is None:
        raise ParameterGolfExportError("parameter_golf config is required for export.")
    if spec.model.hyper is not None:
        raise ParameterGolfExportError("hyper-connections are not yet supported by the exporter.")
    if spec.model.kv_policy is not None:
        raise ParameterGolfExportError("kv_policy is not yet supported by the exporter.")
    if not spec.model.blocks:
        raise ParameterGolfExportError("export requires at least one transformer block.")

    supported_attn_kinds = {"MHA", "GQA", "MQA", "MLA"}
    supported_optimizers = {"adamw", "lion", "muon"}
    opt_name = str(getattr(spec.train.optimizer, "name", "adamw") or "adamw").lower()
    if opt_name not in supported_optimizers:
        raise ParameterGolfExportError(f"optimizer {opt_name!r} is not exportable yet.")

    for idx, block in enumerate(spec.model.blocks):
        if block.attn is None:
            raise ParameterGolfExportError(f"block {idx} is missing attention.")
        if block.attn.kind not in supported_attn_kinds:
            raise ParameterGolfExportError(
                f"block {idx} attention kind {block.attn.kind!r} is not exportable yet."
            )
        if block.attn.selector != "none" or block.attn.gating_pos != "none":
            raise ParameterGolfExportError(
                f"block {idx} uses selector or branch gating, which export v1 excludes."
            )
        if block.attn.alibi:
            raise ParameterGolfExportError(f"block {idx} uses ALiBi, which export v1 excludes.")
        if str(block.attn.sparsity or "none").lower() != "none":
            raise ParameterGolfExportError(
                f"block {idx} uses sparse attention, which export v1 excludes."
            )
        if block.ssm is not None:
            raise ParameterGolfExportError(f"block {idx} uses SSM, which export v1 excludes.")
        if block.ffn_memory is not None:
            raise ParameterGolfExportError(
                f"block {idx} uses ffn_memory, which export v1 excludes."
            )
        if block.extras:
            raise ParameterGolfExportError(
                f"block {idx} uses extra modules, which export v1 excludes."
            )
        if block.ffn is None or not isinstance(block.ffn, DenseFFNConfig):
            raise ParameterGolfExportError(
                f"block {idx} must use a dense FFN to be exportable in v1."
            )


def _copy_package_tree(out_dir: Path) -> Path:
    dst = out_dir / "src" / "transformer_evolution_llm"
    shutil.copytree(
        PACKAGE_ROOT,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    return dst


def _write_tevo_wrapper(out_dir: Path) -> Path:
    wrapper = out_dir / "train_gpt.py"
    wrapper.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import sys",
                "from pathlib import Path",
                "",
                "ROOT = Path(__file__).resolve().parent",
                'sys.path.insert(0, str(ROOT / "src"))',
                "",
                "from transformer_evolution_llm.parameter_golf_runtime import main",
                "",
                'if __name__ == "__main__":',
                "    main(",
                '        config_path=ROOT / "parameter_golf_spec.yaml",',
                '        out_path=ROOT / "parameter_golf_run.json",',
                "    )",
                "",
            ]
        )
    )
    return wrapper


def _write_tevo_readme(out_dir: Path) -> Path:
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Parameter Golf Export",
                "",
                "This workspace was exported from TEVO.",
                "",
                "- `train_gpt.py` runs the exported TEVO spec.",
                "- `parameter_golf_spec.yaml` is the frozen architecture and search config.",
                "- `parameter_golf_export.json` records byte estimates and export metadata.",
                "",
            ]
        )
    )
    return readme


def _code_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*.py"):
        total += path.stat().st_size
    return int(total)


def _export_tevo_workspace(
    spec: ArchitectureSpec,
    *,
    source_path: str | Path,
    out_dir: Path,
    candidate_id: str | None,
) -> dict[str, Any]:
    _validate_tevo_export_spec(spec)
    _copy_package_tree(out_dir)
    wrapper_path = _write_tevo_wrapper(out_dir)
    _write_tevo_readme(out_dir)

    exported_spec = spec.model_copy(deep=True)
    exported_spec_path = out_dir / "parameter_golf_spec.yaml"
    save_architecture_spec(exported_spec, exported_spec_path)

    code_bytes = _code_bytes(out_dir)
    parameter_golf_cfg = exported_spec.parameter_golf
    if parameter_golf_cfg is None:
        raise ParameterGolfExportError("exported spec is missing parameter_golf configuration")
    parameter_golf_cfg.code_bytes = int(code_bytes)
    save_architecture_spec(exported_spec, exported_spec_path)

    payload_bytes_est, total_bytes_est = estimate_artifact_total_bytes_for_spec(exported_spec)
    metadata = {
        "mode": "tevo",
        "source_path": str(Path(source_path).resolve()),
        "candidate_id": candidate_id,
        "workspace": str(out_dir.resolve()),
        "wrapper_path": str(wrapper_path.resolve()),
        "spec_path": str(exported_spec_path.resolve()),
        "code_bytes": int(code_bytes),
        "artifact_payload_bytes_est": int(payload_bytes_est),
        "artifact_total_bytes_est": int(total_bytes_est),
        "artifact_budget_bytes": int(parameter_golf_cfg.artifact_budget_bytes),
    }
    metadata_path = out_dir / OFFICIAL_EXPORT_METADATA
    metadata_path.write_text(json.dumps(metadata, indent=2))
    metadata["metadata_path"] = str(metadata_path.resolve())
    return metadata


def _export_official_workspace(
    spec: ArchitectureSpec,
    *,
    source_path: str | Path,
    out_dir: Path,
    candidate_id: str | None,
    official_train_py: str | Path | None,
) -> dict[str, Any]:
    plan = build_official_submission_plan(spec, official_train_py=official_train_py)
    if not plan["exportable"]:
        reasons = [
            *plan["fatal_reasons"],
            *plan.get("unsupported_patch_reasons", []),
        ]
        raise ParameterGolfExportError(
            "spec is not eligible for the official submission lane: " + "; ".join(reasons)
        )
    official_train_path = resolve_official_train_gpt_path(official_train_py)
    if official_train_path is None:
        raise ParameterGolfExportError(
            "official export requires a local copy of parameter-golf/train_gpt.py "
            "(set TEVO_PARAMETER_GOLF_OFFICIAL_TRAIN_PY or pass --official-train-py)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    base_text = official_train_path.read_text(encoding="utf-8")
    patched_text = _apply_supported_official_patches(
        base_text,
        list(plan.get("supported_patch_reasons", [])),
    )
    rendered_train_py = _inject_official_env_overrides(patched_text, plan["env_overrides"])
    train_py_path = out_dir / "train_gpt.py"
    train_py_path.write_text(rendered_train_py, encoding="utf-8")

    exported_spec = spec.model_copy(deep=True)
    parameter_golf_cfg = exported_spec.parameter_golf
    if parameter_golf_cfg is None:
        raise ParameterGolfExportError("exported spec is missing parameter_golf configuration")
    parameter_golf_cfg.code_bytes = int(train_py_path.stat().st_size)
    spec_path = out_dir / "parameter_golf_spec.yaml"
    save_architecture_spec(exported_spec, spec_path)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Parameter Golf Official-Style Export",
                "",
                "This folder is a compact official-style submission render from TEVO.",
                "",
                "Included files:",
                "- `train_gpt.py`: injected official runner snapshot plus TEVO env overrides",
                "- `parameter_golf_spec.yaml`: frozen source spec for provenance",
                "- `submission.json`: stub metadata to fill after a truth run",
                "- `parameter_golf_export.json`: export metadata and byte accounting",
                "",
            ]
        ),
        encoding="utf-8",
    )

    submission = {
        "author": "TODO",
        "github_id": "TODO",
        "name": str(spec.model.name),
        "blurb": "Generated from TEVO official-style export. Fill after a truth run.",
        "date": None,
        "val_loss": None,
        "val_bpb": None,
        "bytes_total": int(plan["artifact_total_bytes_est"]),
        "bytes_code": int(train_py_path.stat().st_size),
    }
    submission_path = out_dir / "submission.json"
    submission_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")

    payload_bytes_est = _estimated_payload_bytes(exported_spec)
    metadata = dict(plan)
    metadata.update(
        {
            "mode": "official",
            "source_path": str(Path(source_path).resolve()),
            "candidate_id": candidate_id,
            "workspace": str(out_dir.resolve()),
            "train_gpt_path": str(train_py_path.resolve()),
            "spec_path": str(spec_path.resolve()),
            "submission_path": str(submission_path.resolve()),
            "readme_path": str(readme.resolve()),
            "code_bytes": int(train_py_path.stat().st_size),
            "artifact_payload_bytes_est": int(payload_bytes_est),
            "artifact_total_bytes_est": int(payload_bytes_est + train_py_path.stat().st_size),
        }
    )
    metadata_path = out_dir / OFFICIAL_EXPORT_METADATA
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path.resolve())
    return metadata


def export_parameter_golf_workspace(
    source_path: str | Path,
    out_dir: str | Path,
    *,
    candidate_id: str | None = None,
    mode: Literal["official", "tevo"] = "official",
    official_train_py: str | Path | None = None,
) -> dict[str, Any]:
    """Export a TEVO spec to either the official or the legacy TEVO workspace format."""
    spec = load_parameter_golf_export_spec(source_path, candidate_id=candidate_id)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if mode == "tevo":
        return _export_tevo_workspace(
            spec,
            source_path=source_path,
            out_dir=out_path,
            candidate_id=candidate_id,
        )
    return _export_official_workspace(
        spec,
        source_path=source_path,
        out_dir=out_path,
        candidate_id=candidate_id,
        official_train_py=official_train_py,
    )


__all__ = [
    "DEFAULT_OFFICIAL_BASE_CODE_BYTES",
    "OFFICIAL_EXPORT_METADATA",
    "ParameterGolfExportError",
    "build_official_submission_plan",
    "export_parameter_golf_workspace",
    "load_parameter_golf_export_spec",
    "official_submission_metrics",
    "resolve_official_train_gpt_path",
]
