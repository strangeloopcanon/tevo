import random

from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    CustomModuleConfig,
    LookupMemoryConfig,
    MoEFFNConfig,
    RetroConfig,
)
from transformer_evolution_llm.mutations import (
    add_additional_recurrence,
    dense_to_moe,
    insert_custom_module,
    insert_lookup_memory,
    insert_retro_module,
    make_gqa,
    mutate_topk,
    toggle_gated_mix,
    toggle_hyper_connections,
    toggle_optimizer,
    tune_clip,
    tune_lookup_memory,
    tune_optimizer,
    tune_warmup,
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


def test_toggle_optimizer_flips_name(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(9)  # noqa: S311 - deterministic unit tests
    first = toggle_optimizer(tiny_spec, rng=rng)
    assert first.train.optimizer.name == "lion"
    second = toggle_optimizer(first, rng=rng)
    assert second.train.optimizer.name == "adamw"


def test_tune_optimizer_keeps_valid_bounds(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(10)  # noqa: S311 - deterministic unit tests
    child = tune_optimizer(tiny_spec, rng=rng)
    assert child.train.optimizer.name in {"adamw", "lion"}
    if child.train.optimizer.lr is not None:
        assert child.train.optimizer.lr > 0.0
        assert child.train.optimizer.lr <= 5e-3
    if child.train.optimizer.weight_decay is not None:
        assert child.train.optimizer.weight_decay >= 0.0
        assert child.train.optimizer.weight_decay <= 0.2
    if child.train.optimizer.eps is not None:
        assert child.train.optimizer.eps > 0.0
    if child.train.optimizer.betas is not None:
        beta1, beta2 = child.train.optimizer.betas
        assert 0.0 <= beta1 < 1.0
        assert 0.0 <= beta2 < 1.0


def test_tune_warmup_changes_value(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(11)  # noqa: S311 - deterministic unit tests
    child = tune_warmup(tiny_spec, rng=rng)
    assert child.train.warmup >= 0
    assert child.train.warmup != tiny_spec.train.warmup


def test_tune_clip_keeps_positive(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(12)  # noqa: S311 - deterministic unit tests
    child = tune_clip(tiny_spec, rng=rng)
    assert child.train.clip > 0.0


def test_add_additional_recurrence_keeps_train_le_max(tiny_spec: ArchitectureSpec) -> None:
    base = tiny_spec.model_copy(deep=True)
    base.model.blocks = [base.model.blocks[0], base.model.blocks[0].model_copy(deep=True)]
    for seed in range(200):
        child = add_additional_recurrence(base, rng=random.Random(seed))  # noqa: S311
        assert child.model.recurrences, "expected recurrence to be added"
        rec = child.model.recurrences[-1]
        assert rec.train_recurrence <= rec.max_train_recurrence
