"""Crossover helpers for combining parent architectures and checkpoints."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .dsl import ArchitectureSpec, BlockConfig
from .models import EvolutionModel

ParentKey = Literal["a", "b"]


@dataclass
class CrossoverPlan:
    """Aligned crossover plan including child blocks and weight-transfer map."""

    blocks: list[BlockConfig]
    source_map: list[tuple[ParentKey, int]]
    report: dict[str, Any]


def splice_blocks(
    spec_a: ArchitectureSpec,
    spec_b: ArchitectureSpec,
    rng: random.Random,
) -> tuple[list[BlockConfig], int, int]:
    """Legacy positional splice helper kept for backward compatibility."""
    cut_a = rng.randrange(1, len(spec_a.model.blocks) + 1)
    cut_b = rng.randrange(1, len(spec_b.model.blocks) + 1)
    blocks = spec_a.model.blocks[:cut_a] + spec_b.model.blocks[cut_b:]
    return blocks, cut_a, cut_b


def _block_similarity(a: BlockConfig, b: BlockConfig) -> float:
    score = 0.0
    a_origin = getattr(a, "origin_id", None)
    b_origin = getattr(b, "origin_id", None)
    if a_origin and b_origin and a_origin == b_origin:
        score += 4.0

    a_parent = getattr(a, "parent_origin", None)
    b_parent = getattr(b, "parent_origin", None)
    if a_origin and b_parent and a_origin == b_parent:
        score += 1.5
    if b_origin and a_parent and b_origin == a_parent:
        score += 1.5

    a_attn = a.attn
    b_attn = b.attn
    if bool(a_attn) == bool(b_attn):
        score += 0.5
    if a_attn and b_attn:
        if str(getattr(a_attn, "kind", "MHA") or "MHA") == str(
            getattr(b_attn, "kind", "MHA") or "MHA"
        ):
            score += 1.0
        if int(getattr(a_attn, "heads", 0) or 0) == int(getattr(b_attn, "heads", 0) or 0):
            score += 0.4
        if int(getattr(a_attn, "head_dim", 0) or 0) == int(getattr(b_attn, "head_dim", 0) or 0):
            score += 0.4

    a_ffn = getattr(a.ffn, "type", "none") if a.ffn is not None else "none"
    b_ffn = getattr(b.ffn, "type", "none") if b.ffn is not None else "none"
    if str(a_ffn) == str(b_ffn):
        score += 1.0

    if bool(a.ssm) == bool(b.ssm):
        score += 0.4

    a_extras = {getattr(extra, "type", type(extra).__name__) for extra in a.extras}
    b_extras = {getattr(extra, "type", type(extra).__name__) for extra in b.extras}
    if a_extras or b_extras:
        inter = len(a_extras & b_extras)
        union = len(a_extras | b_extras)
        if union > 0:
            score += float(inter) / float(union)

    return score


def aligned_splice_blocks(
    spec_a: ArchitectureSpec,
    spec_b: ArchitectureSpec,
    rng: random.Random,
    *,
    preferred_parent: ParentKey = "a",
) -> CrossoverPlan:
    """Greedy homology-aligned crossover using structural similarity."""
    blocks_a = spec_a.model.blocks
    blocks_b = spec_b.model.blocks

    pair_scores: list[tuple[float, int, int]] = []
    for i, block_a in enumerate(blocks_a):
        for j, block_b in enumerate(blocks_b):
            score = _block_similarity(block_a, block_b)
            if score <= 0.0:
                continue
            pair_scores.append((score, i, j))
    pair_scores.sort(key=lambda item: (item[0], -abs(item[1] - item[2])), reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for score, i, j in pair_scores:
        if i in used_a or j in used_b:
            continue
        if score < 1.0:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, score))

    disjoint_a = [idx for idx in range(len(blocks_a)) if idx not in used_a]
    disjoint_b = [idx for idx in range(len(blocks_b)) if idx not in used_b]

    entries: list[tuple[float, ParentKey, int, BlockConfig]] = []
    matched_report: list[dict[str, Any]] = []
    prefer_prob = 0.7
    for a_idx, b_idx, score in matched:
        pick_preferred = rng.random() < prefer_prob
        if preferred_parent == "a":
            chosen_parent: ParentKey = "a" if pick_preferred else "b"
        else:
            chosen_parent = "b" if pick_preferred else "a"
        chosen_idx = a_idx if chosen_parent == "a" else b_idx
        chosen_block = copy.deepcopy(blocks_a[a_idx] if chosen_parent == "a" else blocks_b[b_idx])
        order = (float(a_idx) + float(b_idx)) / 2.0
        entries.append((order, chosen_parent, chosen_idx, chosen_block))
        matched_report.append(
            {
                "a_idx": a_idx,
                "b_idx": b_idx,
                "score": round(float(score), 4),
                "chosen_parent": chosen_parent,
                "chosen_idx": chosen_idx,
            }
        )

    preferred = blocks_a if preferred_parent == "a" else blocks_b
    preferred_key: ParentKey = preferred_parent
    for idx in (disjoint_a if preferred_parent == "a" else disjoint_b):
        entries.append((float(idx), preferred_key, idx, copy.deepcopy(preferred[idx])))

    other = blocks_b if preferred_parent == "a" else blocks_a
    other_key: ParentKey = "b" if preferred_parent == "a" else "a"
    disjoint_other = disjoint_b if preferred_parent == "a" else disjoint_a
    for idx in disjoint_other:
        if rng.random() < 0.4:
            entries.append((float(idx) + 0.01, other_key, idx, copy.deepcopy(other[idx])))

    if not entries:
        # Fallback in pathological cases.
        entries.append((0.0, preferred_key, 0, copy.deepcopy(preferred[0])))

    entries.sort(key=lambda item: item[0])
    child_blocks = [entry[3] for entry in entries]
    source_map: list[tuple[ParentKey, int]] = [(entry[1], int(entry[2])) for entry in entries]

    report = {
        "method": "aligned_greedy",
        "preferred_parent": preferred_parent,
        "matched": matched_report,
        "disjoint_a": disjoint_a,
        "disjoint_b": disjoint_b,
        "child_blocks": len(child_blocks),
    }
    return CrossoverPlan(blocks=child_blocks, source_map=source_map, report=report)


def crossover_specs(
    spec_a: ArchitectureSpec,
    spec_b: ArchitectureSpec,
    rng: random.Random,
) -> ArchitectureSpec:
    plan = aligned_splice_blocks(spec_a, spec_b, rng, preferred_parent="a")
    data = spec_a.model_dump(mode="python")
    data["model"]["blocks"] = [block.model_dump(mode="python") for block in plan.blocks]
    return ArchitectureSpec(**data)


def _transfer_blocks_by_source_map(
    child_state: dict[str, torch.Tensor],
    parent_state: dict[str, torch.Tensor],
    source_idx: int,
    target_idx: int,
) -> tuple[int, int]:
    transferred = 0
    dropped = 0
    prefix = f"blocks.{source_idx}."
    target_prefix = f"blocks.{target_idx}."
    for key, value in parent_state.items():
        if not key.startswith(prefix):
            continue
        new_key = key.replace(prefix, target_prefix, 1)
        if new_key in child_state and child_state[new_key].shape == value.shape:
            child_state[new_key] = value.to(dtype=child_state[new_key].dtype).clone()
            transferred += 1
        else:
            dropped += 1
    return transferred, dropped


def _transfer_non_block_tensors(
    child_state: dict[str, torch.Tensor],
    parent_state: dict[str, torch.Tensor],
) -> tuple[int, int]:
    transferred = 0
    dropped = 0
    for key, value in parent_state.items():
        if key.startswith("blocks."):
            continue
        if key in child_state and child_state[key].shape == value.shape:
            child_state[key] = value.to(dtype=child_state[key].dtype).clone()
            transferred += 1
        else:
            dropped += 1
    return transferred, dropped


def merge_checkpoints_with_report(
    child_spec: ArchitectureSpec,
    parent_a_ckpt: Path | None,
    parent_b_ckpt: Path | None,
    out_path: Path,
    *,
    source_map: list[tuple[ParentKey, int]] | None = None,
    preferred_parent: ParentKey = "a",
    cut_a: int | None = None,
    cut_b: int | None = None,
    parent_b_blocks: int | None = None,
    checkpoint_dtype: str = "fp16",
) -> tuple[Path | None, dict[str, Any]]:
    """Merge parent checkpoints based on source-map alignment."""
    model = EvolutionModel(child_spec.model)
    child_state = model.state_dict()

    state_a: dict[str, torch.Tensor] | None = None
    state_b: dict[str, torch.Tensor] | None = None
    if parent_a_ckpt and parent_a_ckpt.exists():
        loaded = torch.load(
            parent_a_ckpt, map_location="cpu"
        )  # nosec B614 - trusted local checkpoint
        if isinstance(loaded, dict):
            state_a = loaded
    if parent_b_ckpt and parent_b_ckpt.exists():
        loaded = torch.load(
            parent_b_ckpt, map_location="cpu"
        )  # nosec B614 - trusted local checkpoint
        if isinstance(loaded, dict):
            state_b = loaded

    if source_map is not None:
        resolved_source_map: list[tuple[ParentKey, int]] = list(source_map)
    elif cut_a is not None and cut_b is not None:
        resolved_source_map = []
        for idx in range(int(cut_a)):
            resolved_source_map.append(("a", idx))
        if parent_b_blocks is not None:
            for idx in range(int(cut_b), int(parent_b_blocks)):
                resolved_source_map.append(("b", idx))
    else:
        resolved_source_map = [("a", idx) for idx in range(len(child_spec.model.blocks))]

    transferred = 0
    dropped = 0
    if preferred_parent == "a":
        primary_state = state_a or state_b
    else:
        primary_state = state_b or state_a
    if primary_state is not None:
        t, d = _transfer_non_block_tensors(child_state, primary_state)
        transferred += t
        dropped += d

    for child_idx, (source_parent, source_idx) in enumerate(resolved_source_map):
        parent_state = state_a if source_parent == "a" else state_b
        if parent_state is None:
            continue
        t, d = _transfer_blocks_by_source_map(
            child_state,
            parent_state,
            source_idx=int(source_idx),
            target_idx=int(child_idx),
        )
        transferred += t
        dropped += d

    key = (checkpoint_dtype or "fp16").lower()
    if key in {"fp16", "float16"}:
        dtype = torch.float16
    elif key in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif key in {"fp32", "float32"}:
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported checkpoint_dtype: {checkpoint_dtype}")
    for name, value in list(child_state.items()):
        if value.is_floating_point():
            child_state[name] = value.to(dtype=dtype)

    torch.save(child_state, out_path)
    report = {
        "source_map_size": len(resolved_source_map),
        "transferred_tensors": int(transferred),
        "dropped_tensors": int(dropped),
        "used_parent_a": bool(state_a),
        "used_parent_b": bool(state_b),
    }
    return out_path, report


def merge_checkpoints(
    child_spec: ArchitectureSpec,
    cut_a: int,
    cut_b: int,
    parent_a_blocks: int,
    parent_b_blocks: int,
    parent_a_ckpt: Path | None,
    parent_b_ckpt: Path | None,
    out_path: Path,
    checkpoint_dtype: str = "fp16",
) -> Path | None:
    """Backward-compatible merge API used by older tests/callers."""
    path, _ = merge_checkpoints_with_report(
        child_spec=child_spec,
        parent_a_ckpt=parent_a_ckpt,
        parent_b_ckpt=parent_b_ckpt,
        out_path=out_path,
        cut_a=cut_a,
        cut_b=cut_b,
        parent_b_blocks=parent_b_blocks,
        checkpoint_dtype=checkpoint_dtype,
    )
    return path
