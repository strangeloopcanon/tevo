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
    ParameterGolfConfig,
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


def test_update_filter_roundtrip(tiny_spec: ArchitectureSpec, tmp_path: Path) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.optimizer.name = "muon"
    spec.train.optimizer.update_filter.mode = "topk"
    spec.train.optimizer.update_filter.keep_ratio = 0.4
    spec.train.optimizer.update_filter.granularity = "block"
    spec.train.optimizer.update_filter.block_size = 64
    spec.train.optimizer.update_filter.momentum_blend = 0.5

    spec_path = tmp_path / "opt_filter.yaml"
    api.save_spec(spec, spec_path)
    loaded = api.load_spec(spec_path)
    assert loaded.train.optimizer.name == "muon"
    assert loaded.train.optimizer.update_filter.mode == "topk"
    assert loaded.train.optimizer.update_filter.keep_ratio == pytest.approx(0.4)
    assert loaded.train.optimizer.update_filter.granularity == "block"
    assert loaded.train.optimizer.update_filter.block_size == 64
    assert loaded.train.optimizer.update_filter.momentum_blend == pytest.approx(0.5)


def test_mutation_weights_must_be_positive(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.mutation_weights = {"mix_optimizer_recipe": 0.0}
    with pytest.raises(ValueError, match="must be > 0"):
        ArchitectureSpec.model_validate(spec.model_dump(mode="python"))


def test_parameter_golf_roundtrip(tiny_spec: ArchitectureSpec, tmp_path: Path) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.clip = 0.0
    spec.train.matrix_lr = 0.04
    spec.train.scalar_lr = 0.03
    spec.train.tied_embedding_lr = 0.05
    spec.train.optimizer.muon_momentum_warmup_start = 0.85
    spec.train.optimizer.muon_momentum_warmup_steps = 500
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="fp16_tied_embedding",
        lane_kind="exportable",
        exportable_family="fp16_tied_embedding",
        tied_embedding_export_dtype="fp16",
        max_wallclock_seconds=600.0,
        val_batch_tokens=524_288,
        val_loss_every=200,
        train_log_every=50,
    )

    spec_path = tmp_path / "parameter_golf.yaml"
    api.save_spec(spec, spec_path)
    loaded = api.load_spec(spec_path)
    assert loaded.parameter_golf is not None
    assert loaded.parameter_golf.track == "10min"
    assert loaded.parameter_golf.max_wallclock_seconds == pytest.approx(600.0)
    assert loaded.parameter_golf.val_batch_tokens == 524_288
    assert loaded.parameter_golf.val_loss_every == 200
    assert loaded.parameter_golf.seed_family == "fp16_tied_embedding"
    assert loaded.parameter_golf.exportable_family == "fp16_tied_embedding"
    assert loaded.parameter_golf.tied_embedding_export_dtype == "fp16"
    assert loaded.parameter_golf.export_quant_mode == "int8"
    assert loaded.parameter_golf.eval_protocol == "mid_fidelity"
    assert loaded.parameter_golf.report_eval_modes == ["standard"]
    assert loaded.train.clip == pytest.approx(0.0)
    assert loaded.train.matrix_lr == pytest.approx(0.04)
    assert loaded.train.scalar_lr == pytest.approx(0.03)
    assert loaded.train.tied_embedding_lr == pytest.approx(0.05)
    assert loaded.train.optimizer.muon_momentum_warmup_start == pytest.approx(0.85)
    assert loaded.train.optimizer.muon_momentum_warmup_steps == 500
    assert loaded.summary()["physical_layers"] == loaded.model.physical_block_count()


def test_parameter_golf_short_budget_defaults_to_scout_fast() -> None:
    cfg = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        max_wallclock_seconds=180.0,
        report_eval_modes=["sliding64"],
    )
    assert cfg.eval_protocol == "scout_fast"
    assert cfg.report_eval_modes == ["standard", "sliding64"]


def test_midfidelity_parameter_golf_configs_load_with_600_second_protocols() -> None:
    growth = api.load_spec(Path("configs/pg_midfidelity_growth_search.yaml"))
    leader_overlap = api.load_spec(Path("configs/pg_midfidelity_public_leader_overlap_search.yaml"))

    assert growth.parameter_golf is not None
    assert growth.parameter_golf.max_wallclock_seconds == pytest.approx(600.0)
    assert growth.parameter_golf.eval_protocol == "mid_fidelity"

    assert leader_overlap.parameter_golf is not None
    assert leader_overlap.parameter_golf.max_wallclock_seconds == pytest.approx(600.0)
    assert leader_overlap.parameter_golf.eval_protocol == "mid_fidelity"
    assert leader_overlap.parameter_golf.report_eval_modes == ["standard", "sliding64"]


def test_shared_block_must_match_source(tiny_spec: ArchitectureSpec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    block = spec.model.blocks[0].model_copy(deep=True)
    block.ffn = DenseFFNConfig(type="dense", hidden=1024)
    block.share_with = 0
    spec.model.blocks.append(block)
    with pytest.raises(ValueError, match="shares weights"):
        ArchitectureSpec.model_validate(spec.model_dump(mode="python"))
