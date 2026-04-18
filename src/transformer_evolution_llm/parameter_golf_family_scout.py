"""Helpers for running seeded Parameter Golf family scout searches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

from .api import build_data_module_for_spec, load_spec, save_spec
from .candidates import Candidate
from .dsl import ArchitectureSpec
from .orchestrator import EvolutionRunner
from .trainer import FullWeightTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScoutBudget:
    """Compact run budget for one scout/refinement lane."""

    generations: int
    steps: int
    eval_batches: int
    mutation_steps: int = 1


@dataclass(frozen=True)
class FamilyScoutLane:
    """One seeded search family with stage-specific budgets."""

    family_id: str
    config_path: Path
    priority: int
    note: str
    exportable: bool
    scout: ScoutBudget
    refine: ScoutBudget


FAMILY_SCOUT_LANES: tuple[FamilyScoutLane, ...] = (
    FamilyScoutLane(
        family_id="fp16_tied_embedding",
        config_path=REPO_ROOT / "configs/pg_seed_fp16_tied_embedding.yaml",
        priority=1,
        note="Most direct public baseline-beating idea; easiest exportable win path.",
        exportable=True,
        scout=ScoutBudget(generations=6, steps=80, eval_batches=2),
        refine=ScoutBudget(generations=18, steps=140, eval_batches=3),
    ),
    FamilyScoutLane(
        family_id="ten_layer_compression",
        config_path=REPO_ROOT / "configs/pg_seed_ten_layer_compression.yaml",
        priority=2,
        note="Close to the strongest public shape while still staying submission-small.",
        exportable=True,
        scout=ScoutBudget(generations=6, steps=80, eval_batches=2),
        refine=ScoutBudget(generations=18, steps=140, eval_batches=3),
    ),
    FamilyScoutLane(
        family_id="long_context",
        config_path=REPO_ROOT / "configs/pg_seed_long_context.yaml",
        priority=3,
        note="Publicly promising, but slower and riskier, so scout it after the top two.",
        exportable=True,
        scout=ScoutBudget(generations=4, steps=60, eval_batches=2),
        refine=ScoutBudget(generations=12, steps=120, eval_batches=2),
    ),
    FamilyScoutLane(
        family_id="exact_official_baseline",
        config_path=REPO_ROOT / "configs/pg_seed_exact_official_baseline.yaml",
        priority=4,
        note="Control lane for motif transfer and reality checks.",
        exportable=True,
        scout=ScoutBudget(generations=4, steps=60, eval_batches=2),
        refine=ScoutBudget(generations=12, steps=120, eval_batches=2),
    ),
    FamilyScoutLane(
        family_id="incubator_keep90",
        config_path=REPO_ROOT / "configs/pg_seed_incubator_keep90.yaml",
        priority=5,
        note="Motif incubator only; useful for transfer ideas, not direct submission.",
        exportable=False,
        scout=ScoutBudget(generations=4, steps=60, eval_batches=2),
        refine=ScoutBudget(generations=8, steps=100, eval_batches=2),
    ),
)


def family_scout_lanes(
    family_ids: list[str] | None = None,
) -> list[FamilyScoutLane]:
    """Return the default seeded Parameter Golf lane order."""
    if family_ids is None:
        return list(FAMILY_SCOUT_LANES)

    requested = {str(item).strip() for item in family_ids if str(item).strip()}
    if not requested:
        return list(FAMILY_SCOUT_LANES)

    selected = [lane for lane in FAMILY_SCOUT_LANES if lane.family_id in requested]
    return selected


def family_stage_budget(lane: FamilyScoutLane, stage: str) -> ScoutBudget:
    """Return the stage budget for one family lane."""
    if str(stage).lower() == "refine":
        return lane.refine
    return lane.scout


def build_local_family_command(
    lane: FamilyScoutLane,
    *,
    stage: str = "scout",
    device: str = "cuda",
    seed: int = 0,
    out_root: str | Path = "runs/parameter_golf_family_scouts",
) -> list[str]:
    """Build the recommended local live-search command for one family."""
    budget = family_stage_budget(lane, stage)
    out_root = Path(out_root)
    lane_root = out_root / stage / lane.family_id
    frontier = lane_root / "frontier.json"
    lineage = lane_root / "lineage.json"
    checkpoints = lane_root / "checkpoints"
    return [
        "python",
        "scripts/run_live.py",
        str(lane.config_path.relative_to(REPO_ROOT)),
        "--generations",
        str(budget.generations),
        "--steps",
        str(budget.steps),
        "--eval-batches",
        str(budget.eval_batches),
        "--mutation-steps",
        str(budget.mutation_steps),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--out",
        str(frontier),
        "--lineage-out",
        str(lineage),
        "--checkpoint-dir",
        str(checkpoints),
    ]


def build_runpod_family_command(
    lane: FamilyScoutLane,
    *,
    stage: str = "scout",
    seed: int = 0,
    pod_id: str = "<pod-id>",
) -> list[str]:
    """Build the recommended Runpod remote-search command for one family."""
    budget = family_stage_budget(lane, stage)
    run_id = f"{stage}_{lane.family_id}"
    return [
        "python",
        "scripts/runpod_parameter_golf.py",
        "tevo-evolution",
        str(lane.config_path.relative_to(REPO_ROOT)),
        "--pod-id",
        str(pod_id),
        "--generations",
        str(budget.generations),
        "--steps",
        str(budget.steps),
        "--eval-batches",
        str(budget.eval_batches),
        "--mutation-steps",
        str(budget.mutation_steps),
        "--seed",
        str(seed),
        "--run-id",
        run_id,
    ]


def build_scout_plan(
    *,
    stage: str = "scout",
    device: str = "cuda",
    seed: int = 0,
    family_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the full staged family scout plan."""
    rows: list[dict[str, Any]] = []
    for lane in family_scout_lanes(family_ids):
        budget = family_stage_budget(lane, stage)
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
                ),
                "runpod_command": build_runpod_family_command(
                    lane,
                    stage=stage,
                    seed=seed,
                ),
            }
        )
    return rows


def build_smoke_assets(out_dir: str | Path) -> dict[str, Path]:
    """Create a tiny local Parameter Golf dataset for cheap scout smoke tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "parameter golf loves compact models",
                "shared weights can stretch depth",
                "tiny transformers still learn",
                "fp16 embeddings can help the baseline",
                "long context and extra depth may matter",
            ]
        ),
        encoding="utf-8",
    )
    model_prefix = out_dir / "toy_sp"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=64,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        hard_vocab_limit=False,
    )
    tokenizer_path = model_prefix.with_suffix(".model")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    train_tokens: list[int] = []
    val_tokens: list[int] = []
    train_lines = [
        "parameter golf loves compact models",
        "shared weights can stretch depth",
        "fp16 embeddings can help the baseline",
    ]
    val_lines = [
        "tiny transformers still learn",
        "long context and extra depth may matter",
    ]
    for _ in range(8):
        for line in train_lines:
            train_tokens.extend(tokenizer.encode(line, out_type=int))
    for _ in range(6):
        for line in val_lines:
            val_tokens.extend(tokenizer.encode(line, out_type=int))

    train_path = out_dir / "pg_train_000.bin"
    val_path = out_dir / "pg_val_000.bin"
    _write_pg_shard(train_path, train_tokens)
    _write_pg_shard(val_path, val_tokens)
    return {
        "train_path": train_path,
        "val_path": val_path,
        "tokenizer_path": tokenizer_path,
    }


def build_smoke_family_spec(
    config_path: str | Path,
    assets: dict[str, Path],
) -> ArchitectureSpec:
    """Rewrite a family config into a cheap local smoke-run spec."""
    spec = load_spec(config_path)
    spec.model.name = f"{spec.model.name}-smoke"
    vocab_size = spm.SentencePieceProcessor(model_file=str(assets["tokenizer_path"])).vocab_size()
    spec.model.emb.vocab = int(vocab_size)
    spec.model.head.vocab = int(vocab_size)
    spec.train.bf16 = False
    spec.train.grad_checkpoint = False
    spec.train.max_tokens = 256
    spec.train.batch_tokens = 64
    spec.data.seq_len = 16
    spec.data.batch_size = 2
    spec.data.eval_tokens = 64
    if spec.parameter_golf is None:
        raise ValueError("Smoke family specs require a parameter_golf block.")
    spec.parameter_golf.train_shards_glob = str(assets["train_path"])
    spec.parameter_golf.val_shards_glob = str(assets["val_path"])
    spec.parameter_golf.tokenizer_path = str(assets["tokenizer_path"])
    spec.parameter_golf.val_batch_tokens = 64
    spec.parameter_golf.val_loss_every = 1
    spec.parameter_golf.train_log_every = 1
    spec.parameter_golf.max_wallclock_seconds = 30.0
    spec.evolution.rung1_tokens = 64
    spec.evolution.rung2_tokens = 128
    return spec


def run_local_smoke_scouts(
    out_root: str | Path,
    *,
    family_ids: list[str] | None = None,
    generations: int = 1,
    steps: int = 1,
    eval_batches: int = 1,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Run a tiny local smoke search for each family lane."""
    out_root = Path(out_root)
    assets = build_smoke_assets(out_root / "assets")
    summaries: list[dict[str, Any]] = []
    for lane in family_scout_lanes(family_ids):
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
        runner.data_module = build_data_module_for_spec(smoke_spec, seed=seed)
        runner.run(generations=generations)

        frontier_path = lane_root / "frontier.json"
        lineage_path = lane_root / "lineage.json"
        runner.save_frontier(frontier_path)
        runner.save_lineage(lineage_path)
        summary = summarize_family_run(lane, runner)
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
        "recommended_refine_families": recommended_refine_families(ordered),
    }
    report_path = out_root / "smoke_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def summarize_family_run(
    lane: FamilyScoutLane,
    runner: EvolutionRunner,
) -> dict[str, Any]:
    """Summarize one family run around its seed and best discovered candidate."""
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


def select_best_candidate(candidates: list[Candidate]) -> Candidate | None:
    """Pick the most promising candidate in a family-local run."""
    if not candidates:
        return None
    return min(candidates, key=_candidate_rank_key)


def recommended_refine_families(
    summaries: list[dict[str, Any]],
    *,
    top_k: int = 2,
) -> list[str]:
    """Choose the next families to keep after the scout pass."""
    exportable = [row for row in summaries if bool(row.get("exportable"))]
    if not exportable:
        return []
    ranked = sorted(
        exportable,
        key=lambda row: (
            -float(row.get("main_track_eligible_exact") or 0.0),
            -float(row.get("improvement_bpb") or 0.0),
            float(row.get("best_post_quant_val_bpb") or 1e9),
        ),
    )
    return [str(row["family_id"]) for row in ranked[:top_k]]


def _candidate_rank_key(candidate: Candidate) -> tuple[float, float, float, float, float, float]:
    return (
        -float(candidate.metrics.get("main_track_eligible_exact", 0.0) or 0.0),
        -float(candidate.metrics.get("motif_transfer_eligible", 0.0) or 0.0),
        float(candidate.metrics.get("post_quant_val_bpb", 1e9) or 1e9),
        float(
            candidate.metrics.get("official_submission_unsupported_patch_reason_count", 1e9) or 1e9
        ),
        float(candidate.metrics.get("artifact_total_bytes", 1e9) or 1e9),
        -float(candidate.metrics.get("throughput", 0.0) or 0.0),
    )


def _metric(candidate: Candidate | None, name: str) -> float | None:
    if candidate is None:
        return None
    value = candidate.metrics.get(name)
    if value is None:
        return None
    return float(value)


def _write_pg_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype="<u2").tofile(handle)


__all__ = [
    "FAMILY_SCOUT_LANES",
    "FamilyScoutLane",
    "ScoutBudget",
    "build_local_family_command",
    "build_runpod_family_command",
    "build_scout_plan",
    "build_smoke_assets",
    "build_smoke_family_spec",
    "family_scout_lanes",
    "family_stage_budget",
    "recommended_refine_families",
    "run_local_smoke_scouts",
    "select_best_candidate",
    "summarize_family_run",
]
