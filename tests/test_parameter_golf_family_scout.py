from __future__ import annotations

from pathlib import Path

from transformer_evolution_llm.parameter_golf_family_scout import (
    FAMILY_SCOUT_LANES,
    build_local_family_command,
    build_runpod_family_command,
    build_scout_plan,
    build_smoke_assets,
    build_smoke_family_spec,
    family_scout_lanes,
    recommended_refine_families,
)


def test_family_scout_lane_order_prioritizes_exportable_public_families() -> None:
    lanes = family_scout_lanes()
    assert [lane.family_id for lane in lanes[:3]] == [
        "fp16_tied_embedding",
        "ten_layer_compression",
        "long_context",
    ]
    assert lanes[-1].family_id == "incubator_keep90"


def test_family_scout_plan_builds_commands() -> None:
    plan = build_scout_plan(stage="scout", device="cpu", seed=7, family_ids=["fp16_tied_embedding"])
    assert len(plan) == 1
    row = plan[0]
    assert row["family_id"] == "fp16_tied_embedding"
    assert row["local_command"][:3] == [
        "python",
        "scripts/run_live.py",
        "configs/pg_seed_fp16_tied_embedding.yaml",
    ]
    assert "--device" in row["local_command"]
    assert "--pod-id" in row["runpod_command"]


def test_build_smoke_family_spec_rewrites_paths_and_small_budget(tmp_path: Path) -> None:
    assets = build_smoke_assets(tmp_path / "assets")
    spec = build_smoke_family_spec(
        FAMILY_SCOUT_LANES[0].config_path,
        assets,
    )
    assert spec.parameter_golf is not None
    assert spec.parameter_golf.train_shards_glob == str(assets["train_path"])
    assert spec.parameter_golf.val_shards_glob == str(assets["val_path"])
    assert spec.parameter_golf.tokenizer_path == str(assets["tokenizer_path"])
    assert spec.data.seq_len == 16
    assert spec.data.batch_size == 2
    assert spec.train.max_tokens == 256
    assert spec.parameter_golf.max_wallclock_seconds == 30.0


def test_recommended_refine_families_prefers_exportable_improving_lanes() -> None:
    families = recommended_refine_families(
        [
            {
                "family_id": "incubator_keep90",
                "exportable": False,
                "main_track_eligible_exact": 0.0,
                "improvement_bpb": 0.3,
                "best_post_quant_val_bpb": 3.5,
            },
            {
                "family_id": "ten_layer_compression",
                "exportable": True,
                "main_track_eligible_exact": 1.0,
                "improvement_bpb": 0.2,
                "best_post_quant_val_bpb": 1.5,
            },
            {
                "family_id": "fp16_tied_embedding",
                "exportable": True,
                "main_track_eligible_exact": 1.0,
                "improvement_bpb": 0.25,
                "best_post_quant_val_bpb": 1.6,
            },
        ]
    )
    assert families == ["fp16_tied_embedding", "ten_layer_compression"]
