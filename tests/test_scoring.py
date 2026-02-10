from transformer_evolution_llm.dsl import ArchitectureSpec, StencilConfig
from transformer_evolution_llm.scoring import (
    archive_novelty,
    behavioral_descriptor,
    graph_entropy,
    structural_distance,
)


def _direct_and_stencil_specs(
    tiny_spec: ArchitectureSpec,
) -> tuple[ArchitectureSpec, ArchitectureSpec]:
    direct = tiny_spec.model_copy(deep=True)
    block_direct = direct.model.blocks[0]
    assert block_direct.attn is not None
    block_direct.attn.sparsity = "local_global"
    block_direct.attn.sw = 8
    block_direct.attn.global_stride = 16

    stencil = tiny_spec.model_copy(deep=True)
    block_stencil = stencil.model.blocks[0]
    assert block_stencil.attn is not None
    block_stencil.attn.sparsity = "none"
    block_stencil.attn.sw = None
    block_stencil.attn.global_stride = None
    block_stencil.attn.stencil = StencilConfig(kind="hybrid", window=8, globals=16)
    return direct, stencil


def test_structural_distance_treats_stencil_equivalent_as_same(
    tiny_spec: ArchitectureSpec,
) -> None:
    direct, stencil = _direct_and_stencil_specs(tiny_spec)
    assert structural_distance(direct, stencil) == 0.0


def test_graph_entropy_uses_resolved_sparsity_pattern(tiny_spec: ArchitectureSpec) -> None:
    direct, stencil = _direct_and_stencil_specs(tiny_spec)
    assert graph_entropy(direct) == graph_entropy(stencil)


def test_behavioral_descriptor_fixed_shape(tiny_spec: ArchitectureSpec) -> None:
    descriptor = behavioral_descriptor(tiny_spec)
    assert isinstance(descriptor, list)
    assert len(descriptor) == 13


def test_archive_novelty_decreases_for_similar_points() -> None:
    anchor = [1.0, 2.0, 3.0]
    archive = [[1.0, 2.0, 3.0], [1.01, 2.01, 2.99], [0.99, 2.0, 3.02]]
    near = archive_novelty(anchor, archive, k=2)
    far = archive_novelty([10.0, 20.0, 30.0], archive, k=2)
    assert near <= far
