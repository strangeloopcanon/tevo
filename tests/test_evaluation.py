from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    HyperConnectionsConfig,
    RecurrenceConfig,
    StencilConfig,
)
from transformer_evolution_llm.evaluation import (
    StaticChecker,
    estimate_flops_per_token,
    estimate_params,
    kv_bytes_per_token,
    throughput_proxy,
)


def test_static_checker_metrics(tiny_spec: ArchitectureSpec) -> None:
    params = estimate_params(tiny_spec)
    kv = kv_bytes_per_token(tiny_spec)
    tps = throughput_proxy(tiny_spec, tiny_spec.data.seq_len)
    checker = StaticChecker()
    result = checker.run(tiny_spec)
    assert params > 0
    assert kv > 0
    assert tps > 0
    assert isinstance(result.metrics, dict)


def test_static_checker_sparsity_bounds(tiny_spec: ArchitectureSpec) -> None:
    # Sliding window must be > 0
    spec = tiny_spec.model_copy(deep=True)
    block = spec.model.blocks[0]
    assert block.attn is not None
    block.attn.sparsity = "sliding"
    block.attn.sw = 0
    checker = StaticChecker()
    result = checker.run(spec)
    assert not result.ok
    assert any("sliding_window must be > 0" in reason for reason in result.reasons)

    # local_global requires positive sw and valid global_stride
    spec2 = tiny_spec.model_copy(deep=True)
    block2 = spec2.model.blocks[0]
    assert block2.attn is not None
    block2.attn.sparsity = "local_global"
    block2.attn.sw = -1
    block2.attn.global_stride = spec2.data.seq_len + 1
    result2 = checker.run(spec2)
    assert not result2.ok
    assert any("local_global requires positive sliding_window" in r for r in result2.reasons)
    assert any("local_global requires 0 < global_stride <= seq_len" in r for r in result2.reasons)


def test_estimate_params_includes_hyper_connections(tiny_spec: ArchitectureSpec) -> None:
    base = estimate_params(tiny_spec)
    spec = tiny_spec.model_copy(deep=True)
    spec.model.hyper = HyperConnectionsConfig(
        streams=4, diag_bias=4.0, noise_std=0.0, update_scale=4.0
    )
    hyper = estimate_params(spec)
    expected = float(spec.model.n_layers * (4 * 4 + 2 * 4) + 4)
    assert hyper == base + expected


def test_estimate_flops_interprets_stencil_like_equivalent_sparsity(
    tiny_spec: ArchitectureSpec,
) -> None:
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

    assert estimate_flops_per_token(direct) == estimate_flops_per_token(stencil)


def test_static_checker_validates_local_global_from_stencil(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    block = spec.model.blocks[0]
    assert block.attn is not None
    block.attn.sparsity = "none"
    block.attn.sw = None
    block.attn.global_stride = None
    block.attn.stencil = StencilConfig(kind="hybrid")

    result = StaticChecker().run(spec)
    assert not result.ok
    assert any("local_global requires positive sliding_window" in r for r in result.reasons)


def test_estimate_params_counts_recurrence_adapters(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks.append(spec.model.blocks[0].model_copy(deep=True))
    base = estimate_params(spec)
    spec.model.recurrences = [
        RecurrenceConfig(
            start=0,
            end=2,
            adapter="gated",
            concat_prelude=True,
            train_recurrence=1,
            max_train_recurrence=2,
        )
    ]
    expected = 2.0 * float(spec.model.emb.dim * (spec.model.emb.dim * 2))
    assert estimate_params(spec) == base + expected


def test_estimate_params_counts_shared_blocks_once(tiny_spec: ArchitectureSpec) -> None:
    single = tiny_spec.model_copy(deep=True)

    shared = tiny_spec.model_copy(deep=True)
    shared_block = shared.model.blocks[0].model_copy(deep=True)
    shared_block.share_with = 0
    shared.model.blocks.append(shared_block)

    duplicated = tiny_spec.model_copy(deep=True)
    duplicated.model.blocks.append(duplicated.model.blocks[0].model_copy(deep=True))

    assert estimate_params(shared) == estimate_params(single)
    assert estimate_params(duplicated) > estimate_params(shared)

    metrics = StaticChecker().run(shared).metrics
    assert metrics["effective_depth"] == 2.0
    assert metrics["physical_depth"] == 1.0
    assert metrics["shared_blocks"] == 1.0
