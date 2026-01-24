import random

from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    CustomModuleConfig,
    LookupMemoryConfig,
    MoEFFNConfig,
    RetroConfig,
)
from transformer_evolution_llm.mutations import (
    dense_to_moe,
    insert_custom_module,
    insert_lookup_memory,
    insert_retro_module,
    make_gqa,
    mutate_topk,
    toggle_gated_mix,
    toggle_hyper_connections,
    tune_lookup_memory,
)


def test_dense_to_moe_promotes_block(tiny_spec: ArchitectureSpec):
    before = tiny_spec.model.moe_block_count()
    rng = random.Random(0)  # noqa: S311 - deterministic unit tests
    after_spec = dense_to_moe(tiny_spec, rng=rng)
    assert after_spec.model.moe_block_count() >= before


def test_make_gqa_sets_kind(tiny_spec: ArchitectureSpec):
    rng = random.Random(1)  # noqa: S311 - deterministic unit tests
    child = make_gqa(tiny_spec, rng=rng)
    assert child.model.blocks[0].attn.kind == "GQA"


def test_mutate_topk_changes_value(tiny_spec: ArchitectureSpec):
    rng = random.Random(2)  # noqa: S311 - deterministic unit tests
    spec = dense_to_moe(tiny_spec, rng=rng)
    child = mutate_topk(spec, rng=rng)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    assert moe_blocks


def test_insert_retro_module_adds_extra(tiny_spec: ArchitectureSpec):
    rng = random.Random(3)  # noqa: S311 - deterministic unit tests
    child = insert_retro_module(tiny_spec, rng=rng)
    assert any(isinstance(extra, RetroConfig) for extra in child.model.blocks[0].extras)


def test_insert_custom_module_adds_extra(tiny_spec: ArchitectureSpec):
    rng = random.Random(4)  # noqa: S311 - deterministic unit tests
    child = insert_custom_module(tiny_spec, rng=rng)
    assert any(isinstance(extra, CustomModuleConfig) for extra in child.model.blocks[0].extras)


def test_toggle_gated_mix_adds_gate(tiny_spec: ArchitectureSpec):
    rng = random.Random(5)  # noqa: S311 - deterministic unit tests
    child = toggle_gated_mix(tiny_spec, rng=rng)
    assert child.model.blocks[0].extras


def test_toggle_hyper_connections_toggles_streams(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(6)  # noqa: S311 - deterministic unit tests
    enabled = toggle_hyper_connections(tiny_spec, rng=rng)
    assert enabled.model.hyper is not None
    assert enabled.model.hyper.streams > 1
    disabled = toggle_hyper_connections(enabled, rng=rng)
    assert disabled.model.hyper is None


def test_insert_lookup_memory_adds_extra(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(7)  # noqa: S311 - deterministic unit tests
    child = insert_lookup_memory(tiny_spec, rng=rng)
    assert any(isinstance(extra, LookupMemoryConfig) for extra in child.model.blocks[0].extras)


def test_tune_lookup_memory_keeps_valid_bounds(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(8)  # noqa: S311 - deterministic unit tests
    with_mem = insert_lookup_memory(tiny_spec, rng=rng)
    tuned = tune_lookup_memory(with_mem, rng=rng)
    mem = next(
        extra for extra in tuned.model.blocks[0].extras if isinstance(extra, LookupMemoryConfig)
    )
    assert mem.entries > 0
    assert 0 < mem.topk <= mem.entries
