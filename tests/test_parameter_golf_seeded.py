from __future__ import annotations

from pathlib import Path

from transformer_evolution_llm.api import load_spec
from transformer_evolution_llm.dsl import ParameterGolfConfig
from transformer_evolution_llm.parameter_golf_export import build_official_submission_plan
from transformer_evolution_llm.parameter_golf_seeded import (
    incubator_promotion_summary,
    motif_signature,
    seed_lane_metadata,
    transfer_parameter_golf_motif,
)


def _seeded_spec(tiny_spec, *, lane_kind: str = "exportable"):
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        seed_family="seed-family",
        lane_kind=lane_kind,
        exportable_family="seed-family" if lane_kind == "exportable" else None,
        incubator_anchor_post_quant_val_bpb=3.65 if lane_kind == "incubator" else None,
    )
    return spec


def test_seed_lane_metadata_reports_family_labels(tiny_spec) -> None:
    spec = _seeded_spec(tiny_spec)

    metadata = seed_lane_metadata(spec)

    assert metadata["seed_family"] == "seed-family"
    assert metadata["lane_kind"] == "exportable"
    assert metadata["exportable_family"] == "seed-family"


def test_motif_signature_changes_with_transferable_recipe(tiny_spec) -> None:
    baseline = _seeded_spec(tiny_spec)
    variant = _seeded_spec(tiny_spec)
    variant.train.warmdown_steps = 3600

    assert motif_signature(baseline) != motif_signature(variant)

    variant = _seeded_spec(tiny_spec)
    variant.model.norm = "rmsnorm"

    assert motif_signature(baseline) != motif_signature(variant)

    variant = _seeded_spec(tiny_spec)
    if variant.model.blocks[0].ffn is None:
        raise TypeError("expected FFN block in tiny spec")
    variant.model.blocks[0].ffn.input_source = "embedding"

    assert motif_signature(baseline) != motif_signature(variant)


def test_incubator_promotion_summary_uses_delta_threshold(tiny_spec) -> None:
    spec = _seeded_spec(tiny_spec, lane_kind="incubator")

    promoted = incubator_promotion_summary(spec, post_quant_val_bpb=3.60)
    blocked = incubator_promotion_summary(spec, post_quant_val_bpb=3.649, appearance_count=1)
    rediscovered = incubator_promotion_summary(spec, post_quant_val_bpb=3.649, appearance_count=2)

    assert promoted["eligible"] is True
    assert promoted["reason"] == "beats_anchor"
    assert blocked["eligible"] is False
    assert blocked["reason"] == "needs_transfer"
    assert rediscovered["eligible"] is True
    assert rediscovered["reason"] == "rediscovered"


def test_transfer_parameter_golf_motif_copies_training_recipe(tiny_spec) -> None:
    source = _seeded_spec(tiny_spec, lane_kind="incubator")
    target = _seeded_spec(tiny_spec)
    source.train.matrix_lr = 0.06
    source.train.warmdown_steps = 3600
    source.model.emb.init_std = 0.004
    source.model.norm = "rmsnorm"
    source.train.optimizer.update_filter.mode = "bernoulli"
    source.train.optimizer.update_filter.keep_ratio = 0.9
    if source.model.blocks[0].ffn is None:
        raise TypeError("expected FFN block in tiny spec")
    source.model.blocks[0].ffn.input_source = "embedding"

    transferred = transfer_parameter_golf_motif(source, target)

    assert transferred.train.matrix_lr == 0.06
    assert transferred.train.warmdown_steps == 3600
    assert transferred.model.emb.init_std == 0.004
    assert transferred.model.norm == "rmsnorm"
    assert transferred.train.optimizer.update_filter.mode == "bernoulli"
    assert transferred.train.optimizer.update_filter.keep_ratio == 0.9
    if transferred.model.blocks[0].ffn is None:
        raise TypeError("expected transferred FFN block in tiny spec")
    assert transferred.model.blocks[0].ffn.input_source == "embedding"
    assert transferred.parameter_golf is not None
    assert transferred.parameter_golf.exportable_family == "seed-family"


def test_seed_family_configs_load_cleanly() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    paths = [
        config_dir / "pg_seed_exact_official_baseline.yaml",
        config_dir / "pg_seed_fp16_tied_embedding.yaml",
        config_dir / "pg_seed_long_context.yaml",
        config_dir / "pg_seed_ten_layer_compression.yaml",
        config_dir / "pg_seed_incubator_keep90.yaml",
    ]

    specs = [load_spec(path) for path in paths]

    assert specs[0].parameter_golf is not None
    assert specs[0].parameter_golf.seed_family == "exact_official_baseline"
    assert specs[1].parameter_golf is not None
    assert specs[1].parameter_golf.tied_embedding_export_dtype == "fp16"
    assert build_official_submission_plan(specs[1])["exportable"] is True
    assert (
        "fp16_tied_embedding_export"
        in build_official_submission_plan(specs[1])["supported_patch_reasons"]
    )
    assert specs[4].parameter_golf is not None
    assert specs[4].parameter_golf.lane_kind == "incubator"
