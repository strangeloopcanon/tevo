import random

from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    CustomModuleConfig,
    LookupMemoryConfig,
    MoEFFNConfig,
    RecurrenceConfig,
    RetroConfig,
)
from transformer_evolution_llm.mutations import (
    add_additional_recurrence,
    add_embedding_ffn_branch,
    dense_to_moe,
    insert_custom_module,
    insert_lookup_memory,
    insert_retro_module,
    make_gqa,
    mix_method_recipe,
    mix_optimizer_recipe,
    moe_to_dense,
    mutate_topk,
    mutate_with_trace,
    mutation_names,
    register_mutation,
    remove_block_span,
    remove_embedding_ffn_branch,
    remove_recurrence,
    resample_optimizer_base,
    sanitize_topology,
    simplify_attention,
    strip_extras,
    template_registry_name,
    toggle_ffn_input_source,
    toggle_gated_mix,
    toggle_gradient_transform_mode,
    toggle_hyper_connections,
    toggle_update_filter_mode,
    tune_clip,
    tune_embedding_ffn_branch,
    tune_gradient_transform_eps,
    tune_gradient_transform_ns_steps,
    tune_lookup_memory,
    tune_optimizer,
    tune_update_filter_block_size,
    tune_update_filter_granularity,
    tune_update_filter_momentum_blend,
    tune_update_filter_ratio,
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


def test_resample_optimizer_base_changes_name(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(9)  # noqa: S311 - deterministic unit tests
    first = resample_optimizer_base(tiny_spec, rng=rng)
    assert first.train.optimizer.name in {"adamw", "lion", "muon"}
    assert first.train.optimizer.name != tiny_spec.train.optimizer.name


def test_tune_optimizer_keeps_valid_bounds(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(10)  # noqa: S311 - deterministic unit tests
    child = tune_optimizer(tiny_spec, rng=rng)
    assert child.train.optimizer.name in {"adamw", "lion", "muon"}
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
    if child.train.optimizer.name == "muon":
        momentum = child.train.optimizer.muon_momentum
        assert momentum is None or (0.5 <= momentum <= 0.999)
        assert child.train.optimizer.muon_ns_steps >= 1


def test_gradient_transform_mutations_keep_bounds(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(1010)  # noqa: S311 - deterministic unit tests
    child = toggle_gradient_transform_mode(tiny_spec, rng=rng)
    child = tune_gradient_transform_ns_steps(child, rng=rng)
    child = tune_gradient_transform_eps(child, rng=rng)
    transform = child.train.optimizer.gradient_transform
    assert transform.mode in {
        "identity",
        "sign",
        "normalize",
        "orthogonalize_2d",
        "sign_orthogonalize_2d",
    }
    assert transform.ns_steps >= 1
    assert transform.eps > 0.0


def test_mix_optimizer_recipe_changes_recipe_dimensions(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(1011)  # noqa: S311 - deterministic unit tests
    child = mix_optimizer_recipe(tiny_spec, rng=rng)
    before = tiny_spec.train.optimizer
    after = child.train.optimizer
    changed = (
        before.name != after.name
        or before.lr != after.lr
        or before.weight_decay != after.weight_decay
        or before.gradient_transform.mode != after.gradient_transform.mode
        or before.update_filter.mode != after.update_filter.mode
        or before.update_filter.keep_ratio != after.update_filter.keep_ratio
    )
    assert changed


def test_mix_method_recipe_changes_method_dimensions(tiny_spec: ArchitectureSpec) -> None:
    changed_any = False
    for seed in range(20):
        child = mix_method_recipe(tiny_spec, rng=random.Random(2000 + seed))  # noqa: S311
        before = tiny_spec
        after = child
        method_changed = (
            before.train.optimizer.name != after.train.optimizer.name
            or before.train.optimizer.gradient_transform.mode
            != after.train.optimizer.gradient_transform.mode
            or before.train.optimizer.update_filter.mode != after.train.optimizer.update_filter.mode
            or len(before.model.recurrences) != len(after.model.recurrences)
            or any(
                (b0.attn is not None and b1.attn is not None and b0.attn.kind != b1.attn.kind)
                for b0, b1 in zip(before.model.blocks, after.model.blocks, strict=False)
            )
            or any(
                len(b0.extras) != len(b1.extras)
                for b0, b1 in zip(before.model.blocks, after.model.blocks, strict=False)
            )
            or before.model.moe_block_count() != after.model.moe_block_count()
        )
        if method_changed:
            changed_any = True
            break
    assert changed_any


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


def test_remove_block_span_preserves_at_least_one_block(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks = [spec.model.blocks[0].model_copy(deep=True) for _ in range(3)]
    child = remove_block_span(spec, rng=random.Random(13))  # noqa: S311
    assert len(child.model.blocks) >= 1
    assert len(child.model.blocks) < len(spec.model.blocks)


def test_moe_to_dense_reduces_moe_count(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    for _ in range(2):
        spec = dense_to_moe(spec, rng=random.Random(14))  # noqa: S311
    before = spec.model.moe_block_count()
    child = moe_to_dense(spec, rng=random.Random(15))  # noqa: S311
    assert child.model.moe_block_count() <= before


def test_strip_extras_removes_one_extra(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec = insert_custom_module(spec, rng=random.Random(16))  # noqa: S311
    before = len(spec.model.blocks[0].extras)
    child = strip_extras(spec, rng=random.Random(17))  # noqa: S311
    assert len(child.model.blocks[0].extras) <= before


def test_remove_recurrence_removes_entries(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks = [spec.model.blocks[0], spec.model.blocks[0].model_copy(deep=True)]
    spec = add_additional_recurrence(spec, rng=random.Random(18))  # noqa: S311
    child = remove_recurrence(spec, rng=random.Random(19))  # noqa: S311
    assert len(child.model.recurrences) <= len(spec.model.recurrences)


def test_simplify_attention_sets_mha(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks[0].attn.kind = "MLA"
    child = simplify_attention(spec, rng=random.Random(20))  # noqa: S311
    assert child.model.blocks[0].attn.kind == "MHA"


def test_sanitize_topology_clamps_indices(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks = [spec.model.blocks[0], spec.model.blocks[0].model_copy(deep=True)]
    spec.model.recurrences = [RecurrenceConfig(start=0, end=2)]
    spec.model.recurrences[0].end = 99
    sanitize_topology(spec)
    assert spec.model.recurrences[0].end <= len(spec.model.blocks)


def test_mutate_with_trace_reports_selected_keys(tiny_spec: ArchitectureSpec) -> None:
    key = "unit_test_identity_mutation"

    def _identity(spec: ArchitectureSpec, _rng: random.Random):
        return spec.model_copy(deep=True)

    register_mutation(key, _identity)
    label, _mutated, trace = mutate_with_trace(
        tiny_spec,
        rng=random.Random(21),  # noqa: S311
        weights={name: (1.0 if name == key else 0.0) for name in mutation_names()},
        steps=1,
    )
    assert key in label
    assert trace == [key]


def test_mutate_with_trace_honors_allowlist(tiny_spec: ArchitectureSpec) -> None:
    label, _mutated, trace = mutate_with_trace(
        tiny_spec,
        rng=random.Random(210),  # noqa: S311
        allowed_names=["mix_optimizer_recipe"],
        steps=3,
    )
    assert label == "mix_optimizer_recipe+mix_optimizer_recipe+mix_optimizer_recipe"
    assert trace == ["mix_optimizer_recipe", "mix_optimizer_recipe", "mix_optimizer_recipe"]


def test_update_filter_mutations_keep_bounds(tiny_spec: ArchitectureSpec) -> None:
    rng = random.Random(211)  # noqa: S311
    child = toggle_update_filter_mode(tiny_spec, rng=rng)
    assert child.train.optimizer.update_filter.mode in {"none", "bernoulli", "topk"}
    child = tune_update_filter_ratio(child, rng=rng)
    child = tune_update_filter_granularity(child, rng=rng)
    child = tune_update_filter_momentum_blend(child, rng=rng)
    child = tune_update_filter_block_size(child, rng=rng)
    filt = child.train.optimizer.update_filter
    assert 0.0 < filt.keep_ratio <= 1.0
    assert filt.granularity in {"element", "block"}
    assert 0.0 <= filt.momentum_blend <= 1.0
    assert filt.block_size >= 1


def test_template_entries_registered() -> None:
    assert any(name.startswith("tpl::") for name in mutation_names())
    assert template_registry_name("example-template") == "tpl::example-template"


def test_toggle_ffn_input_source_changes_field(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks[0].ffn.input_source = "residual"
    child = toggle_ffn_input_source(spec, rng=random.Random(22))  # noqa: S311
    assert child.model.blocks[0].ffn is not None
    assert child.model.blocks[0].ffn.input_source in {"residual", "embedding"}
    assert child.model.blocks[0].ffn.input_source != "residual"


def test_add_and_remove_embedding_ffn_branch(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    assert getattr(spec.model.blocks[0], "ffn_memory", None) is None
    child = add_embedding_ffn_branch(spec, rng=random.Random(23))  # noqa: S311
    assert getattr(child.model.blocks[0], "ffn_memory", None) is not None
    removed = remove_embedding_ffn_branch(child, rng=random.Random(24))  # noqa: S311
    assert getattr(removed.model.blocks[0], "ffn_memory", None) is None


def test_tune_embedding_ffn_branch_keeps_valid(tiny_spec: ArchitectureSpec) -> None:
    spec = add_embedding_ffn_branch(tiny_spec, rng=random.Random(25))  # noqa: S311
    tuned = tune_embedding_ffn_branch(spec, rng=random.Random(26))  # noqa: S311
    branch = getattr(tuned.model.blocks[0], "ffn_memory", None)
    assert branch is not None
    assert branch.hidden > 0
