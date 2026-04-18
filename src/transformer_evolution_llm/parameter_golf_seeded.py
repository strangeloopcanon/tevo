"""Helpers for seeded Parameter Golf search lanes and motif transfer."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from json import dumps as json_dumps
from typing import Any

from .dsl import ArchitectureSpec


def seed_lane_metadata(spec: ArchitectureSpec) -> dict[str, Any]:
    """Return user-facing lane labels for seeded Parameter Golf runs."""
    cfg = spec.parameter_golf
    if cfg is None:
        return {}
    return {
        "seed_family": cfg.seed_family,
        "lane_kind": cfg.lane_kind,
        "exportable_family": cfg.exportable_family,
        "tied_embedding_export_dtype": cfg.tied_embedding_export_dtype,
    }


def motif_signature(spec: ArchitectureSpec) -> str:
    """Hash the transferable training motif of a Parameter Golf candidate."""
    payload = {
        "seq_len": int(spec.data.seq_len),
        "batch_tokens": int(getattr(spec.train, "batch_tokens", 0) or 0),
        "embed_init_std": float(getattr(spec.model.emb, "init_std", 0.02) or 0.02),
        "model_norm": str(getattr(spec.model, "norm", "layernorm") or "layernorm"),
        "tied_embedding_export_dtype": (
            getattr(spec.parameter_golf, "tied_embedding_export_dtype", "int8")
            if spec.parameter_golf is not None
            else "int8"
        ),
        "train": {
            "lr": float(spec.train.lr),
            "matrix_lr": _maybe_float(spec.train.matrix_lr),
            "scalar_lr": _maybe_float(spec.train.scalar_lr),
            "embed_lr": _maybe_float(spec.train.embed_lr),
            "head_lr": _maybe_float(spec.train.head_lr),
            "tied_embedding_lr": _maybe_float(spec.train.tied_embedding_lr),
            "warmup": int(spec.train.warmup),
            "warmdown_steps": int(spec.train.warmdown_steps),
            "clip": float(spec.train.clip),
            "weight_decay": float(spec.train.weight_decay),
        },
        "optimizer": {
            "name": str(spec.train.optimizer.name),
            "weight_decay": _maybe_float(spec.train.optimizer.weight_decay),
            "muon_momentum": _maybe_float(spec.train.optimizer.muon_momentum),
            "muon_ns_steps": int(spec.train.optimizer.muon_ns_steps),
            "muon_momentum_warmup_start": _maybe_float(
                spec.train.optimizer.muon_momentum_warmup_start
            ),
            "muon_momentum_warmup_steps": int(spec.train.optimizer.muon_momentum_warmup_steps),
            "gradient_transform": deepcopy(
                spec.train.optimizer.gradient_transform.model_dump(mode="python")
            ),
            "update_filter": deepcopy(spec.train.optimizer.update_filter.model_dump(mode="python")),
        },
        "attn": [
            {
                "kind": str(block.attn.kind),
                "heads": int(block.attn.heads),
                "head_dim": int(block.attn.head_dim),
                "kv_groups": int(block.attn.kv_groups or 1),
                "qk_norm_max": _maybe_float(block.attn.qk_norm_max),
                "softmax": (
                    deepcopy(block.attn.softmax.model_dump(mode="python"))
                    if block.attn.softmax is not None
                    else None
                ),
            }
            for block in spec.model.blocks
            if block.attn is not None
        ],
        "ffn": [
            {
                "hidden": int(block.ffn.hidden),
                "activation": getattr(block.ffn, "activation", None),
                "input_source": getattr(block.ffn, "input_source", "residual"),
            }
            for block in spec.model.blocks
            if block.ffn is not None
        ],
    }
    blob = json_dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


def incubator_promotion_summary(
    spec: ArchitectureSpec,
    *,
    post_quant_val_bpb: float | None,
    appearance_count: int = 1,
) -> dict[str, Any]:
    """Decide whether an incubator motif is strong enough to transfer."""
    cfg = spec.parameter_golf
    if cfg is None or cfg.lane_kind != "incubator":
        return {
            "eligible": True,
            "delta_bpb": None,
            "appearance_count": int(appearance_count),
            "reason": "not_incubator",
        }

    anchor = cfg.incubator_anchor_post_quant_val_bpb
    if post_quant_val_bpb is None or anchor is None:
        return {
            "eligible": False,
            "delta_bpb": None,
            "appearance_count": int(appearance_count),
            "reason": "missing_anchor_or_score",
        }

    delta = float(anchor) - float(post_quant_val_bpb)
    min_delta = float(cfg.motif_promote_min_delta)
    if delta >= min_delta:
        return {
            "eligible": True,
            "delta_bpb": delta,
            "appearance_count": int(appearance_count),
            "reason": "beats_anchor",
        }
    if int(appearance_count) >= 2:
        return {
            "eligible": True,
            "delta_bpb": delta,
            "appearance_count": int(appearance_count),
            "reason": "rediscovered",
        }
    return {
        "eligible": False,
        "delta_bpb": delta,
        "appearance_count": int(appearance_count),
        "reason": "needs_transfer",
    }


def transfer_parameter_golf_motif(
    source_spec: ArchitectureSpec,
    target_spec: ArchitectureSpec,
    *,
    include_context: bool = False,
    include_structure: bool = False,
) -> ArchitectureSpec:
    """Copy transferable training motifs from one seed family into another."""
    child = target_spec.model_copy(deep=True)
    child.train.lr = float(source_spec.train.lr)
    child.train.embed_lr = _maybe_float(source_spec.train.embed_lr)
    child.train.head_lr = _maybe_float(source_spec.train.head_lr)
    child.train.matrix_lr = _maybe_float(source_spec.train.matrix_lr)
    child.train.scalar_lr = _maybe_float(source_spec.train.scalar_lr)
    child.train.tied_embedding_lr = _maybe_float(source_spec.train.tied_embedding_lr)
    child.train.warmup = int(source_spec.train.warmup)
    child.train.warmdown_steps = int(source_spec.train.warmdown_steps)
    child.train.clip = float(source_spec.train.clip)
    child.train.weight_decay = float(source_spec.train.weight_decay)
    child.train.optimizer = source_spec.train.optimizer.model_copy(deep=True)
    child.model.emb.init_std = float(getattr(source_spec.model.emb, "init_std", 0.02) or 0.02)
    child.model.norm = str(getattr(source_spec.model, "norm", child.model.norm) or child.model.norm)

    if child.parameter_golf is not None and source_spec.parameter_golf is not None:
        child.parameter_golf.tied_embedding_export_dtype = (
            source_spec.parameter_golf.tied_embedding_export_dtype
        )

    if include_context:
        child.data.seq_len = int(source_spec.data.seq_len)
        child.data.batch_size = int(source_spec.data.batch_size)
        child.data.eval_tokens = source_spec.data.eval_tokens
        child.train.batch_tokens = source_spec.train.batch_tokens
        if child.parameter_golf is not None and source_spec.parameter_golf is not None:
            child.parameter_golf.val_batch_tokens = source_spec.parameter_golf.val_batch_tokens

    if include_structure:
        child.model.emb.init_std = float(getattr(source_spec.model.emb, "init_std", 0.02) or 0.02)
        child.model.blocks = [block.model_copy(deep=True) for block in source_spec.model.blocks]

    for target_block, source_block in zip(
        child.model.blocks, source_spec.model.blocks, strict=False
    ):
        if target_block.attn is not None and source_block.attn is not None:
            target_block.attn.qk_norm_max = source_block.attn.qk_norm_max
            if source_block.attn.softmax is None:
                target_block.attn.softmax = None
            else:
                target_block.attn.softmax = source_block.attn.softmax.model_copy(deep=True)
            if include_structure:
                target_block.attn.kind = source_block.attn.kind
                target_block.attn.heads = source_block.attn.heads
                target_block.attn.head_dim = source_block.attn.head_dim
                target_block.attn.kv_groups = source_block.attn.kv_groups
        if target_block.ffn is not None and source_block.ffn is not None:
            target_block.ffn.hidden = source_block.ffn.hidden
            if hasattr(target_block.ffn, "activation") and hasattr(source_block.ffn, "activation"):
                target_block.ffn.activation = source_block.ffn.activation
            if hasattr(target_block.ffn, "input_source") and hasattr(source_block.ffn, "input_source"):
                target_block.ffn.input_source = source_block.ffn.input_source
    return child


def _maybe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "incubator_promotion_summary",
    "motif_signature",
    "seed_lane_metadata",
    "transfer_parameter_golf_motif",
]
