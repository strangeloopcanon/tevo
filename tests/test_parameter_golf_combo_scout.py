from __future__ import annotations

from pathlib import Path

from transformer_evolution_llm.api import load_spec
from transformer_evolution_llm.parameter_golf_combo_scout import (
    COMBO_SCOUT_LANES,
    build_combo_plan,
    combo_scout_lanes,
    recommended_combo_refine_families,
)


def test_combo_lane_order_centers_on_winning_line() -> None:
    lanes = combo_scout_lanes()
    assert [lane.family_id for lane in lanes] == [
        "combo_ffn_input_practical",
        "combo_ffn_input_hybrid",
        "combo_ffn_input_wild",
    ]


def test_combo_plan_builds_commands() -> None:
    plan = build_combo_plan(stage="scout", device="cpu", seed=11, family_ids=["combo_ffn_input_practical"])
    assert len(plan) == 1
    row = plan[0]
    assert row["family_id"] == "combo_ffn_input_practical"
    assert row["budget"]["mutation_steps"] == 2
    assert row["local_command"][:3] == [
        "python",
        "scripts/run_live.py",
        "configs/pg_combo_ffn_input_practical.yaml",
    ]
    assert "--pod-id" in row["runpod_command"]


def test_combo_configs_load_cleanly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = [
        repo_root / "configs/pg_combo_ffn_input_practical.yaml",
        repo_root / "configs/pg_combo_ffn_input_hybrid.yaml",
        repo_root / "configs/pg_combo_ffn_input_wild.yaml",
    ]
    specs = [load_spec(path) for path in config_paths]
    assert specs[0].parameter_golf is not None
    assert specs[0].parameter_golf.seed_family == "combo_ffn_input_practical"
    assert specs[0].parameter_golf.lane_kind == "exportable"
    assert specs[1].parameter_golf is not None
    assert specs[1].parameter_golf.lane_kind == "incubator"
    assert specs[2].parameter_golf is not None
    assert specs[2].evolution.mutation_steps == 4


def test_recommended_combo_refine_prefers_exportable_then_transferable() -> None:
    families = recommended_combo_refine_families(
        [
            {
                "family_id": "combo_ffn_input_hybrid",
                "exportable": False,
                "main_track_eligible_exact": 0.0,
                "motif_transfer_eligible": 1.0,
                "improvement_bpb": 0.4,
                "best_post_quant_val_bpb": 3.1,
            },
            {
                "family_id": "combo_ffn_input_practical",
                "exportable": True,
                "main_track_eligible_exact": 1.0,
                "motif_transfer_eligible": 0.0,
                "improvement_bpb": 0.15,
                "best_post_quant_val_bpb": 3.4,
            },
            {
                "family_id": "combo_backup",
                "exportable": True,
                "main_track_eligible_exact": 1.0,
                "motif_transfer_eligible": 0.0,
                "improvement_bpb": 0.2,
                "best_post_quant_val_bpb": 3.2,
            },
        ]
    )
    assert families == ["combo_backup", "combo_ffn_input_practical"]
