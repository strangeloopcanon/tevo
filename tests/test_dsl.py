from pathlib import Path

import pytest

from transformer_evolution_llm import api
from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    BlockConfig,
    CondConfig,
    CondOpConfig,
    CondRegConfig,
    CondSourceConfig,
    CustomModuleConfig,
    DenseFFNConfig,
    DepthRouterConfig,
    GateStep,
    HierarchyConfig,
    HierarchyLevelConfig,
    HyperConnectionsConfig,
    KVPolicyConfig,
    MacroConfig,
    MixerConfig,
    MixUnitConfig,
    ProjectionConfig,
    ResidualConfig,
    SoftmaxConfig,
    SoftmaxKernelConfig,
    StencilConfig,
)


def test_spec_summary_counts_moe_blocks(tiny_spec: ArchitectureSpec, tmp_path: Path) -> None:
    spec = tiny_spec
    assert spec.summary()["moe_blocks"] == 0

    block = spec.model.blocks[0]
    assert isinstance(block.ffn, DenseFFNConfig)
    block.ffn = DenseFFNConfig(type="dense", hidden=4096)
    spec_path = tmp_path / "spec.yaml"
    api.save_spec(spec, spec_path)
    loaded = api.load_spec(spec_path)
    assert isinstance(loaded, ArchitectureSpec)
    assert loaded.model.n_layers == 1


def test_block_accepts_custom_extras() -> None:
    block = BlockConfig(
        ffn=None,
        extras=[CustomModuleConfig(name="wild", params={"depth": 2})],
    )
    assert block.extras[0].name == "wild"


def test_macro_primitives_roundtrip(tiny_spec: ArchitectureSpec, tmp_path: Path) -> None:
    spec = tiny_spec
    spec.model.kv_policy = KVPolicyConfig(cache="window", window=4096, quant="nf4")
    spec.model.macro = MacroConfig(
        depth_router=DepthRouterConfig(kind="token", budget=0.7, tau=1.0, min_layers=1),
        hierarchy=HierarchyConfig(
            levels=[HierarchyLevelConfig(every=4, downsample=0.5, up_proj=True)]
        ),
        residual=ResidualConfig(kind="single", pre_ln=True),
        cond=CondConfig(
            source=CondSourceConfig(kind="pool-mlp", H=128),
            reg=CondRegConfig(kind="freebits", kappa=0.5),
            ops=[CondOpConfig(where="pre_mixer", op="lora", r=4)],
        ),
        mix_unit=MixUnitConfig(
            kind="par",
            merge="WeightedAdd",
            choices=[
                MixerConfig(
                    kind="Attention",
                    heads=4,
                    head_dim=32,
                    stencil=StencilConfig(kind="sliding", window=256, stride=64),
                    softmax=SoftmaxConfig(
                        type="kernel",
                        qk_norm="rms",
                        kernel=SoftmaxKernelConfig(name="favor", features=64),
                    ),
                    projection=ProjectionConfig(type="low_rank", rank=8),
                ),
                MixerConfig(kind="Retention", heads=4, head_dim=32, chunk=512, mode="parallel"),
            ],
        ),
    )

    spec_path = tmp_path / "spec.yaml"
    api.save_spec(spec, spec_path)
    loaded = api.load_spec(spec_path)
    assert loaded.model.kv_policy is not None
    assert loaded.model.kv_policy.cache == "window"
    assert loaded.model.kv_policy.quant == "nf4"
    assert loaded.model.macro is not None
    assert loaded.model.macro.depth_router is not None
    assert loaded.model.macro.depth_router.kind == "token"
    assert loaded.model.macro.hierarchy is not None
    assert loaded.model.macro.hierarchy.levels[0].every == 4
    assert loaded.model.macro.cond is not None
    assert loaded.model.macro.cond.source is not None
    assert loaded.model.macro.cond.source.kind == "pool_mlp"


def test_hyper_connections_roundtrip(tiny_spec: ArchitectureSpec, tmp_path: Path) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.model.hyper = HyperConnectionsConfig(
        streams=4, diag_bias=6.0, noise_std=1e-3, update_scale=4.0
    )
    spec_path = tmp_path / "spec.yaml"
    api.save_spec(spec, spec_path)
    loaded = api.load_spec(spec_path)
    assert loaded.model.hyper is not None
    assert loaded.model.hyper.streams == 4
    assert loaded.summary()["hyper_streams"] == 4


def test_evolution_population_must_be_positive(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.population = 0
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ArchitectureSpec.model_validate(spec.model_dump(mode="python"))


def test_gate_schedule_must_be_sorted(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.gate_schedule = [
        GateStep(generation=5, thresholds={"min_layers": 4.0}),
        GateStep(generation=2, thresholds={"min_layers": 2.0}),
    ]
    with pytest.raises(ValueError, match="gate_schedule must be sorted"):
        ArchitectureSpec.model_validate(spec.model_dump(mode="python"))
