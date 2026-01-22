from transformer_evolution_llm.dsl import ArchitectureSpec, HyperConnectionsConfig
from transformer_evolution_llm.evaluation import (
    StaticChecker,
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
