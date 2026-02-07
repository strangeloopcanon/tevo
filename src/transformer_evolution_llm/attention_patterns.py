"""Helpers to canonicalize declarative attention pattern settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttentionPattern:
    """Resolved attention sparsity settings after stencil bridging."""

    sparsity: str
    sw: int | None
    block_size: int | None
    block_stride: int | None
    dilation: int | None
    global_stride: int | None


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_attention_pattern(attn: Any) -> AttentionPattern:
    """Resolve runtime sparsity knobs, including declarative stencil fallbacks."""
    sparsity = str(getattr(attn, "sparsity", "none") or "none")
    sw = _as_int_or_none(getattr(attn, "sw", None))
    block_size = _as_int_or_none(getattr(attn, "block_size", None))
    block_stride = _as_int_or_none(getattr(attn, "block_stride", None))
    dilation = _as_int_or_none(getattr(attn, "dilation", None))
    global_stride = _as_int_or_none(getattr(attn, "global_stride", None))

    stencil = getattr(attn, "stencil", None)
    if stencil is not None:
        stencil_kind = str(getattr(stencil, "kind", "full") or "full")
        kind_map = {
            "local": "sliding",
            "dilated": "dilated",
            "block": "block",
            "ring": "block",
            "sliding": "sliding",
            "hybrid": "local_global",
        }
        if sparsity == "none":
            mapped = kind_map.get(stencil_kind)
            if mapped is not None:
                sparsity = mapped
        if sw is None:
            sw = _as_int_or_none(getattr(stencil, "window", None))
        if block_size is None:
            block_size = _as_int_or_none(getattr(stencil, "block", None))
        if block_stride is None:
            block_stride = _as_int_or_none(getattr(stencil, "stride", None))
        if dilation is None:
            dilation = _as_int_or_none(getattr(stencil, "dilation", None))
        if global_stride is None:
            global_stride = _as_int_or_none(getattr(stencil, "globals", None))

    if sparsity == "none" and sw is not None:
        sparsity = "sliding"

    return AttentionPattern(
        sparsity=sparsity,
        sw=sw,
        block_size=block_size,
        block_stride=block_stride,
        dilation=dilation,
        global_stride=global_stride,
    )
