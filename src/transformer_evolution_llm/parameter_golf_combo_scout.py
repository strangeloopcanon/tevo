"""Helpers for broader Parameter Golf combo scout searches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .api import build_data_module_for_spec, save_spec
from .candidates import Candidate
from .orchestrator import EvolutionRunner
from .parameter_golf_family_scout import (
    FamilyScoutLane,
    ScoutBudget,
    build_local_family_command,
    build_runpod_family_command,
    build_smoke_assets,
    build_smoke_family_spec,
    select_best_candidate,
)
from .trainer import FullWeightTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]


COMBO_SCOUT_LANES: tuple[FamilyScoutLane, ...] = (
    FamilyScoutLane(
        family_id="combo_ffn_input_practical",
        config_path=REPO_ROOT / "configs/pg_combo_ffn_input_practical.yaml",
        priority=1,
        note=(
            "Main branch: start from the cleaned-up winner and search broadly "
            "among small, practical moves."
        ),
        exportable=True,
        scout=ScoutBudget(generations=12, steps=90, eval_batches=2, mutation_steps=2),
        refine=ScoutBudget(generations=24, steps=140, eval_batches=3, mutation_steps=2),
    ),
    FamilyScoutLane(
        family_id="combo_ffn_input_hybrid",
        config_path=REPO_ROOT / "configs/pg_combo_ffn_input_hybrid.yaml",
        priority=2,
        note=(
            "Hybrid branch: mix practical changes with a few structural bets "
            "that still seem transferable."
        ),
        exportable=False,
        scout=ScoutBudget(generations=10, steps=90, eval_batches=2, mutation_steps=2),
        refine=ScoutBudget(generations=20, steps=140, eval_batches=3, mutation_steps=2),
    ),
    FamilyScoutLane(
        family_id="combo_ffn_input_wild",
        config_path=REPO_ROOT / "configs/pg_combo_ffn_input_wild.yaml",
        priority=3,
        note="Wild branch: let several things change at once and look for surprising new motifs.",
        exportable=False,
        scout=ScoutBudget(generations=8, steps=90, eval_batches=2, mutation_steps=4),
        refine=ScoutBudget(generations=16, steps=140, eval_batches=3, mutation_steps=4),
    ),
)


def combo_scout_lanes(
    family_ids: list[str] | None = None,
) -> list[FamilyScoutLane]:
    """Return the default broader combo-scout lane order."""
    if family_ids is None:
        return list(COMBO_SCOUT_LANES)
    requested = {str(item).strip() for item in family_ids if str(item).strip()}
    if not requested:
        return list(COMBO_SCOUT_LANES)
    return [lane for lane in COMBO_SCOUT_LANES if lane.family_id in requested]


def combo_stage_budget(lane: FamilyScoutLane, stage: str) -> ScoutBudget:
    """Return the stage budget for one combo lane."""
    if str(stage).lower() == "refine":
        return lane.refine
    return lane.scout


def build_combo_plan(
    *,
    stage: str = "scout",
    device: str = "cuda",
    seed: int = 0,
    family_ids: list[str] | None = None,
    out_root: str | Path = "runs/parameter_golf_combo_scouts",
) -> list[dict[str, Any]]:
    """Build the full staged combo-scout plan."""
    rows: list[dict[str, Any]] = []
    for lane in combo_scout_lanes(family_ids):
        budget = combo_stage_budget(lane, stage)
        rows.append(
            {
                "family_id": lane.family_id,
                "priority": lane.priority,
                "exportable": lane.exportable,
                "note": lane.note,
                "budget": {
                    "generations": budget.generations,
                    "steps": budget.steps,
                    "eval_batches": budget.eval_batches,
                    "mutation_steps": budget.mutation_steps,
                },
                "config_path": str(lane.config_path.relative_to(REPO_ROOT)),
                "local_command": build_local_family_command(
                    lane,
                    stage=stage,
                    device=device,
                    seed=seed,
                    out_root=out_root,
                ),
                "runpod_command": build_runpod_family_command(
                    lane,
                    stage=stage,
                    seed=seed,
                    pod_id="<pod-id>",
                ),
            }
        )
    return rows


def run_local_smoke_combo_scouts(
    out_root: str | Path,
    *,
    family_ids: list[str] | None = None,
    generations: int = 1,
    steps: int = 1,
    eval_batches: int = 1,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Run a tiny local smoke search for each combo lane."""
    out_root = Path(out_root)
    assets = build_smoke_assets(out_root / "assets")
    summaries: list[dict[str, Any]] = []
    for lane in combo_scout_lanes(family_ids):
        lane_root = out_root / lane.family_id
        lane_root.mkdir(parents=True, exist_ok=True)
        smoke_spec = build_smoke_family_spec(lane.config_path, assets)
        smoke_spec_path = lane_root / f"{lane.family_id}_smoke.yaml"
        save_spec(smoke_spec, smoke_spec_path)

        runner = EvolutionRunner(
            base_spec=smoke_spec,
            evolution_cfg=smoke_spec.evolution,
            mode="live",
            seed=seed,
        )
        runner.trainer = FullWeightTrainer(
            checkpoint_dir=lane_root / "checkpoints",
            device=device,
            steps=steps,
            eval_batches=eval_batches,
            entropy_threshold=smoke_spec.train.entropy_threshold,
            entropy_patience=smoke_spec.train.entropy_patience,
            instability_threshold=smoke_spec.train.instability_threshold,
            no_improve_patience=smoke_spec.train.no_improve_patience,
            improvement_tolerance=smoke_spec.train.improvement_tolerance,
        )
        runner.data_module = cast(
            Any,
            build_data_module_for_spec(smoke_spec, seed=seed),
        )
        runner.run(generations=generations)

        frontier_path = lane_root / "frontier.json"
        lineage_path = lane_root / "lineage.json"
        runner.save_frontier(frontier_path)
        runner.save_lineage(lineage_path)
        summary = summarize_combo_run(lane, runner)
        summary["smoke_spec_path"] = str(smoke_spec_path)
        summary["frontier_path"] = str(frontier_path)
        summary["lineage_path"] = str(lineage_path)
        summaries.append(summary)

    ordered = sorted(
        summaries,
        key=lambda item: (
            0 if item["exportable"] else 1,
            -float(item.get("improvement_bpb") or 0.0),
            float(item.get("best_post_quant_val_bpb") or 1e9),
        ),
    )
    report = {
        "families": ordered,
        "recommended_refine_families": recommended_combo_refine_families(ordered),
    }
    report_path = out_root / "smoke_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def summarize_combo_run(
    lane: FamilyScoutLane,
    runner: EvolutionRunner,
) -> dict[str, Any]:
    """Summarize one combo run around its seed and best discovered candidate."""
    evaluated = [
        cand for cand in runner._history if cand.metrics.get("post_quant_val_bpb") is not None
    ]
    completed = [cand for cand in evaluated if cand.status == "completed"]
    anchor = next((cand for cand in runner._history if cand.parent is None), None)
    best = select_best_candidate(completed)
    if best is None:
        best = select_best_candidate(evaluated)
    anchor_bpb = _metric(anchor, "post_quant_val_bpb")
    best_bpb = _metric(best, "post_quant_val_bpb")
    improvement = None
    if anchor_bpb is not None and best_bpb is not None:
        improvement = anchor_bpb - best_bpb
    return {
        "family_id": lane.family_id,
        "priority": lane.priority,
        "exportable": lane.exportable,
        "anchor_candidate": anchor.ident if anchor is not None else None,
        "anchor_post_quant_val_bpb": anchor_bpb,
        "best_candidate": best.ident if best is not None else None,
        "best_post_quant_val_bpb": best_bpb,
        "improvement_bpb": improvement,
        "main_track_eligible_exact": _metric(best, "main_track_eligible_exact"),
        "motif_transfer_eligible": _metric(best, "motif_transfer_eligible"),
        "candidate_count": len(evaluated),
        "completed_candidate_count": len(completed),
    }


def recommended_combo_refine_families(
    summaries: list[dict[str, Any]],
    *,
    top_k: int = 2,
) -> list[str]:
    """Choose the next combo lanes to keep after the scout pass."""
    ranked = sorted(
        summaries,
        key=lambda row: (
            0 if bool(row.get("exportable")) else 1,
            -float(row.get("main_track_eligible_exact") or 0.0),
            -float(row.get("motif_transfer_eligible") or 0.0),
            -float(row.get("improvement_bpb") or 0.0),
            float(row.get("best_post_quant_val_bpb") or 1e9),
        ),
    )
    return [str(row["family_id"]) for row in ranked[:top_k]]


def _metric(candidate: Candidate | None, name: str) -> float | None:
    if candidate is None:
        return None
    value = candidate.metrics.get(name)
    if value is None:
        return None
    return float(value)


__all__ = [
    "COMBO_SCOUT_LANES",
    "build_combo_plan",
    "combo_scout_lanes",
    "combo_stage_budget",
    "recommended_combo_refine_families",
    "run_local_smoke_combo_scouts",
    "summarize_combo_run",
]
