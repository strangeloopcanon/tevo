import random
from pathlib import Path

import torch

from transformer_evolution_llm.crossover import (
    aligned_splice_blocks,
    merge_checkpoints,
    merge_checkpoints_with_report,
    splice_blocks,
)


def test_merge_checkpoints(tmp_path: Path, tiny_spec):
    rng = random.Random(0)  # noqa: S311
    blocks, cut_a, cut_b = splice_blocks(tiny_spec, tiny_spec, rng)
    spec_data = tiny_spec.model_dump(mode="python")
    spec_data["model"]["blocks"] = [block.model_dump(mode="python") for block in blocks]
    child_spec = type(tiny_spec)(**spec_data)
    parent_ckpt = _save_state(tmp_path / "parent.pt", tiny_spec)
    out = merge_checkpoints(
        child_spec=child_spec,
        cut_a=cut_a,
        cut_b=cut_b,
        parent_a_blocks=len(tiny_spec.model.blocks),
        parent_b_blocks=len(tiny_spec.model.blocks),
        parent_a_ckpt=parent_ckpt,
        parent_b_ckpt=parent_ckpt,
        out_path=tmp_path / "child.pt",
    )
    assert out is not None and out.exists()


def test_aligned_splice_blocks_returns_source_map(tiny_spec):
    spec_a = tiny_spec.model_copy(deep=True)
    spec_b = tiny_spec.model_copy(deep=True)
    spec_a.model.blocks = [spec_a.model.blocks[0].model_copy(deep=True) for _ in range(2)]
    spec_b.model.blocks = [spec_b.model.blocks[0].model_copy(deep=True) for _ in range(2)]
    spec_a.model.blocks[0].attn.kind = "GQA"
    spec_b.model.blocks[0].attn.kind = "MHA"
    rng = random.Random(1)  # noqa: S311
    plan = aligned_splice_blocks(spec_a, spec_b, rng, preferred_parent="a")
    assert plan.blocks
    assert len(plan.source_map) == len(plan.blocks)
    assert plan.report.get("method") == "aligned_greedy"


def test_merge_checkpoints_with_report(tmp_path: Path, tiny_spec):
    parent_ckpt = _save_state(tmp_path / "parent.pt", tiny_spec)
    child_spec = tiny_spec.model_copy(deep=True)
    out, report = merge_checkpoints_with_report(
        child_spec=child_spec,
        parent_a_ckpt=parent_ckpt,
        parent_b_ckpt=parent_ckpt,
        out_path=tmp_path / "child-report.pt",
        source_map=[("a", 0)],
        preferred_parent="a",
    )
    assert out is not None and out.exists()
    assert report["transferred_tensors"] >= 0


def _save_state(path: Path, spec):
    from transformer_evolution_llm.models import EvolutionModel

    model = EvolutionModel(spec.model)
    torch.save(model.state_dict(), path)
    return path
