"""Helpers for staging and auditing the first TEVO -> autoresearch-mlx transfer run."""

import csv
import difflib
import json as std_json
import os
import re
import shlex
import shutil
import statistics
import subprocess  # nosec B404 - workflow runner executes trusted local benchmark commands.
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ujson as json
from pydantic import ValidationError

from .dsl import ArchitectureSpec
from .train_recipe import (
    TrainRecipe,
    TrainRecipeCompatibilityError,
    TrainRecipeTarget,
    load_train_recipe,
    render_train_recipe_to_path,
    save_train_recipe,
    train_recipe_from_spec,
)

_SELECTION_LABELS = ("quality", "compute", "balanced")
_ARM_LABELS = ("baseline", "quality", "compute", "balanced")
_IGNORE_NAMES = shutil.ignore_patterns(
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
    "run.log",
    "results.tsv",
)
_VAL_BPB_RE = re.compile(r"^val_bpb:\s*([-+0-9.eE]+)\s*$", flags=re.MULTILINE)
_PEAK_VRAM_MB_RE = re.compile(r"^peak_vram_mb:\s*([-+0-9.eE]+)\s*$", flags=re.MULTILINE)
_TEVO_REGION_RE = re.compile(
    r"^# === TEVO TRAIN RECIPE: (?P<name>[A-Z_]+) START ===\n"
    r"(?P<body>.*?)"
    r"^# === TEVO TRAIN RECIPE: (?P=name) END ===$",
    flags=re.MULTILINE | re.DOTALL,
)


class MlxTransferError(ValueError):
    """Base error for the TEVO -> autoresearch-mlx workflow."""


@dataclass(frozen=True)
class CompatibleRecipeCandidate:
    """Frontier entry that can be rendered into downstream train.py targets."""

    candidate_id: str
    frontier_index: int
    recipe: TrainRecipe
    metrics: dict[str, float]
    recipe_key: str


@dataclass(frozen=True)
class SelectedTransferRecipe:
    """One labeled TrainRecipe selection for the first public transfer run."""

    label: str
    candidate_id: str
    frontier_index: int
    recipe: TrainRecipe
    metrics: dict[str, float]
    recipe_key: str


@dataclass(frozen=True)
class BenchmarkRunResult:
    """One concrete autoresearch-mlx benchmark attempt."""

    arm: str
    run_index: int
    status: str
    val_bpb: float | None
    peak_memory_gb: float | None
    log_path: Path


def load_frontier_recipe_candidates(frontier_path: str | Path) -> list[CompatibleRecipeCandidate]:
    """Load all bridge-compatible frontier entries as train recipes."""
    frontier_path = Path(frontier_path)
    payload = json.loads(frontier_path.read_text())
    if not isinstance(payload, list):
        raise MlxTransferError(f"Frontier JSON must be a list: {frontier_path}")
    compatible = _load_recipe_candidates_from_entries(payload, source_path=frontier_path)
    if len({item.recipe_key for item in compatible}) >= 3:
        return compatible

    state_path = frontier_path.with_name("frontier.state.json")
    if state_path.exists():
        state_payload = json.loads(state_path.read_text())
        history = state_payload.get("history")
        if isinstance(history, list):
            compatible = _merge_recipe_candidates(
                compatible,
                _load_recipe_candidates_from_entries(history, source_path=state_path),
            )
    if len({item.recipe_key for item in compatible}) >= 3:
        return compatible

    lineage_path = frontier_path.with_name("lineage.json")
    if lineage_path.exists():
        lineage_payload = json.loads(lineage_path.read_text())
        nodes = lineage_payload.get("nodes")
        if isinstance(nodes, list):
            compatible = _merge_recipe_candidates(
                compatible,
                _load_recipe_candidates_from_entries(nodes, source_path=lineage_path),
            )
    return compatible


def _load_recipe_candidates_from_entries(
    entries: Sequence[object],
    *,
    source_path: Path,
) -> list[CompatibleRecipeCandidate]:
    compatible: list[CompatibleRecipeCandidate] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        spec_payload = entry.get("spec")
        if not isinstance(spec_payload, dict):
            continue
        candidate_id = str(entry.get("id") or f"frontier-{idx}")
        try:
            spec = ArchitectureSpec(**spec_payload)
            recipe = train_recipe_from_spec(
                spec,
                candidate_id=candidate_id,
                frontier_path=source_path,
                metrics=_numeric_metrics(entry.get("metrics")),
            )
        except (ValidationError, TrainRecipeCompatibilityError):
            continue
        compatible.append(
            CompatibleRecipeCandidate(
                candidate_id=candidate_id,
                frontier_index=idx,
                recipe=recipe,
                metrics=dict(recipe.source.metrics if recipe.source is not None else {}),
                recipe_key=_recipe_model_key(recipe),
            )
        )
    return compatible


def _merge_recipe_candidates(
    existing: Sequence[CompatibleRecipeCandidate],
    extra: Sequence[CompatibleRecipeCandidate],
) -> list[CompatibleRecipeCandidate]:
    merged: list[CompatibleRecipeCandidate] = list(existing)
    seen_candidate_ids = {item.candidate_id for item in existing}
    for item in extra:
        if item.candidate_id in seen_candidate_ids:
            continue
        merged.append(item)
        seen_candidate_ids.add(item.candidate_id)
    return merged


def select_public_transfer_recipes(
    frontier_path: str | Path,
) -> list[SelectedTransferRecipe]:
    """Select the quality/compute/balanced recipe trio for the first MLX transfer run."""
    compatible = load_frontier_recipe_candidates(frontier_path)
    unique_recipe_keys = {item.recipe_key for item in compatible}
    if len(unique_recipe_keys) < 3:
        raise MlxTransferError(
            "Need at least three recipe-distinct compatible frontier entries "
            "for the MLX transfer demo."
        )

    quality_sorted = sorted(
        compatible,
        key=lambda item: (
            _metric_value(item.metrics, "ppl_code"),
            item.frontier_index,
            item.candidate_id,
        ),
    )
    compute_sorted = sorted(
        compatible,
        key=lambda item: (
            _metric_value(item.metrics, "speedrun_flops_to_target"),
            item.frontier_index,
            item.candidate_id,
        ),
    )
    ppl_rank = {item.candidate_id: rank for rank, item in enumerate(quality_sorted, start=1)}
    flops_rank = {item.candidate_id: rank for rank, item in enumerate(compute_sorted, start=1)}
    balanced_sorted = sorted(
        compatible,
        key=lambda item: (
            ppl_rank[item.candidate_id] + flops_rank[item.candidate_id],
            ppl_rank[item.candidate_id],
            flops_rank[item.candidate_id],
            item.frontier_index,
            item.candidate_id,
        ),
    )
    ordered = {
        "quality": quality_sorted,
        "compute": compute_sorted,
        "balanced": balanced_sorted,
    }

    selected: list[SelectedTransferRecipe] = []
    used_recipe_keys: set[str] = set()
    for label in _SELECTION_LABELS:
        match = next(
            (item for item in ordered[label] if item.recipe_key not in used_recipe_keys), None
        )
        if match is None:
            raise MlxTransferError(f"Could not find a distinct {label} recipe candidate.")
        used_recipe_keys.add(match.recipe_key)
        selected.append(
            SelectedTransferRecipe(
                label=label,
                candidate_id=match.candidate_id,
                frontier_index=match.frontier_index,
                recipe=match.recipe,
                metrics=match.metrics,
                recipe_key=match.recipe_key,
            )
        )
    return selected


def export_public_transfer_recipes(
    frontier_path: str | Path,
    out_dir: str | Path,
) -> tuple[list[SelectedTransferRecipe], Path]:
    """Write the selected TrainRecipe trio plus a manifest to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_public_transfer_recipes(frontier_path)
    manifest_rows: list[dict[str, Any]] = []
    for item in selected:
        recipe_path = out_dir / f"{item.label}__{_safe_slug(item.candidate_id)}.train_recipe.yaml"
        save_train_recipe(item.recipe, recipe_path)
        manifest_rows.append(
            {
                "label": item.label,
                "candidate_id": item.candidate_id,
                "frontier_index": item.frontier_index,
                "recipe_path": str(recipe_path),
                "metrics": item.metrics,
                "recipe_model": item.recipe.model.model_dump(mode="python"),
            }
        )
    manifest_path = out_dir / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frontier_path": str(Path(frontier_path).resolve()),
                "selected": manifest_rows,
            },
            indent=2,
        )
    )
    return selected, manifest_path


def render_public_transfer_variants(
    mlx_repo: str | Path,
    recipe_manifest_path: str | Path,
    out_dir: str | Path,
) -> tuple[dict[str, Path], Path]:
    """Render baseline + labeled TEVO train.py variants for autoresearch-mlx."""
    mlx_repo = Path(mlx_repo)
    out_dir = Path(out_dir)
    train_py_path = mlx_repo / "train.py"
    prepare_py_path = mlx_repo / "prepare.py"
    if not train_py_path.exists():
        raise MlxTransferError(f"autoresearch-mlx train.py not found at {train_py_path}")
    if not prepare_py_path.exists():
        raise MlxTransferError(f"autoresearch-mlx prepare.py not found at {prepare_py_path}")

    manifest = json.loads(Path(recipe_manifest_path).read_text())
    selected = manifest.get("selected")
    if not isinstance(selected, list):
        raise MlxTransferError(f"Invalid recipe manifest at {recipe_manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    baseline_path = out_dir / "baseline.train.py"
    baseline_path.write_text(train_py_path.read_text())
    rendered["baseline"] = baseline_path

    render_rows: list[dict[str, Any]] = []
    for row in selected:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        recipe_path_raw = row.get("recipe_path")
        if label not in _SELECTION_LABELS or not recipe_path_raw:
            continue
        recipe_path = Path(str(recipe_path_raw))
        out_path = out_dir / f"{label}.train.py"
        recipe = load_train_recipe(recipe_path)
        render_train_recipe_to_path(
            recipe,
            target=TrainRecipeTarget.AUTORESEARCH_MLX,
            train_py_path=train_py_path,
            out_path=out_path,
        )
        rendered[label] = out_path
        render_rows.append(
            {"label": label, "recipe_path": str(recipe_path), "rendered_train_py": str(out_path)}
        )

    manifest_path = out_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mlx_repo": str(mlx_repo.resolve()),
                "source_train_py": str(train_py_path.resolve()),
                "rendered": render_rows,
            },
            indent=2,
        )
    )
    return rendered, manifest_path


def stage_public_transfer_workspaces(
    *,
    mlx_repo: str | Path,
    rendered_variants: Mapping[str, str | Path],
    out_dir: str | Path,
) -> Path:
    """Create isolated baseline/seeded autoresearch-mlx workspaces for benchmarking."""
    mlx_repo = Path(mlx_repo)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_variant = Path(rendered_variants["baseline"])
    if not baseline_variant.exists():
        raise MlxTransferError("Missing rendered baseline train.py variant.")

    manifest_rows: list[dict[str, Any]] = []
    for arm in _ARM_LABELS:
        variant_path = Path(
            rendered_variants["baseline"] if arm == "baseline" else rendered_variants[arm]
        )
        if not variant_path.exists():
            raise MlxTransferError(f"Missing rendered train.py for arm {arm}.")
        repo_copy = out_dir / arm / "repo"
        _copy_repo_tree(mlx_repo, repo_copy)
        shutil.copy2(variant_path, repo_copy / "train.py")
        manifest_rows.append(
            {
                "arm": arm,
                "repo_path": str(repo_copy.resolve()),
                "train_py": str((repo_copy / "train.py").resolve()),
                "variant_source": str(variant_path.resolve()),
            }
        )
    manifest_path = out_dir / "arm_manifest.json"
    manifest_path.write_text(
        json.dumps({"mlx_repo": str(mlx_repo.resolve()), "arms": manifest_rows}, indent=2)
    )
    return manifest_path


def run_public_transfer_benchmarks(
    arm_manifest_path: str | Path,
    *,
    out_dir: str | Path,
    repeat: int = 3,
    timeout_seconds: int = 600,
    command: Sequence[str] | str = ("uv", "run", "train.py"),
) -> tuple[list[BenchmarkRunResult], Path, Path, Path]:
    """Run repeated baseline/seeded autoresearch-mlx benchmarks and summarize medians."""
    arm_manifest = json.loads(Path(arm_manifest_path).read_text())
    arms = arm_manifest.get("arms")
    if not isinstance(arms, list):
        raise MlxTransferError(f"Invalid arm manifest at {arm_manifest_path}")
    parsed_command = _normalize_command(command)
    out_dir = Path(out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkRunResult] = []
    for row in arms:
        if not isinstance(row, dict):
            continue
        arm = str(row.get("arm") or "").strip()
        repo_path = Path(str(row.get("repo_path") or ""))
        if arm not in _ARM_LABELS or not repo_path.exists():
            continue
        for run_index in range(1, int(repeat) + 1):
            log_path = logs_dir / f"{arm}_run{run_index}.log"
            result = _run_arm_once(
                arm=arm,
                run_index=run_index,
                repo_path=repo_path,
                command=parsed_command,
                timeout_seconds=timeout_seconds,
                log_path=log_path,
            )
            results.append(result)

    rows = _benchmark_rows(results)
    results_path = out_dir / "benchmark_results.json"
    results_path.write_text(json.dumps(rows, indent=2))
    tsv_path = out_dir / "benchmark_results.tsv"
    _write_benchmark_tsv(tsv_path, rows)

    summary_payload = summarize_public_transfer_results(results)
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    markdown_path = out_dir / "benchmark_summary.md"
    markdown_path.write_text(render_public_transfer_summary_markdown(summary_payload))
    return results, results_path, summary_path, markdown_path


def summarize_public_transfer_results(results: Sequence[BenchmarkRunResult]) -> dict[str, Any]:
    """Aggregate per-arm median metrics for the first MLX transfer benchmark."""
    grouped: dict[str, list[BenchmarkRunResult]] = {}
    for result in results:
        grouped.setdefault(result.arm, []).append(result)

    summaries: list[dict[str, Any]] = []
    for arm in _ARM_LABELS:
        arm_results = grouped.get(arm, [])
        val_bpbs = [float(item.val_bpb) for item in arm_results if item.val_bpb is not None]
        peak_memory = [
            float(item.peak_memory_gb) for item in arm_results if item.peak_memory_gb is not None
        ]
        summaries.append(
            {
                "arm": arm,
                "runs": len(arm_results),
                "successful_runs": sum(1 for item in arm_results if item.status == "ok"),
                "median_val_bpb": _median_or_none(val_bpbs),
                "best_val_bpb": (min(val_bpbs) if val_bpbs else None),
                "median_peak_memory_gb": _median_or_none(peak_memory),
                "all_val_bpb": val_bpbs,
            }
        )

    seeded = [
        row for row in summaries if row["arm"] != "baseline" and row["median_val_bpb"] is not None
    ]
    baseline_row = next((row for row in summaries if row["arm"] == "baseline"), None)
    seeded.sort(key=lambda row: (float(row["median_val_bpb"]), row["arm"]))
    winner = seeded[0]["arm"] if seeded else None
    baseline_median = baseline_row["median_val_bpb"] if baseline_row is not None else None
    winner_delta = None
    if winner is not None and baseline_median is not None:
        winner_row = next(row for row in seeded if row["arm"] == winner)
        winner_delta = float(winner_row["median_val_bpb"]) - float(baseline_median)

    return {
        "arms": summaries,
        "winner": winner,
        "winner_delta_vs_baseline": winner_delta,
    }


def render_public_transfer_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown summary table for the first public MLX transfer run."""
    arms = summary.get("arms") or []
    lines = [
        "# MLX Transfer Benchmark Summary",
        "",
        "| Arm | Successful Runs | Median val_bpb | Best val_bpb | Median Peak Memory (GB) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in arms:
        if not isinstance(row, dict):
            continue
        lines.append(
            (
                "| {arm} | {successful_runs}/{runs} | {median_val_bpb} | "
                "{best_val_bpb} | {median_peak_memory_gb} |"
            ).format(
                arm=row.get("arm", "-"),
                successful_runs=row.get("successful_runs", 0),
                runs=row.get("runs", 0),
                median_val_bpb=_fmt_metric(row.get("median_val_bpb")),
                best_val_bpb=_fmt_metric(row.get("best_val_bpb")),
                median_peak_memory_gb=_fmt_metric(row.get("median_peak_memory_gb")),
            )
        )
    winner = summary.get("winner")
    if winner:
        lines.extend(
            [
                "",
                f"Winner: `{winner}`",
                "Median delta vs baseline: "
                f"`{_fmt_metric(summary.get('winner_delta_vs_baseline'))}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_winning_seed_diff(
    arm_manifest_path: str | Path,
    benchmark_summary_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Write a small unified diff between the baseline train.py and the winning seeded arm."""
    manifest = json.loads(Path(arm_manifest_path).read_text())
    summary = json.loads(Path(benchmark_summary_path).read_text())
    winner = summary.get("winner")
    if not isinstance(winner, str) or not winner:
        raise MlxTransferError("Benchmark summary does not contain a winning seeded arm.")

    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise MlxTransferError(f"Invalid arm manifest at {arm_manifest_path}")
    arm_by_name = {
        str(row.get("arm")): Path(str(row.get("train_py") or ""))
        for row in arms
        if isinstance(row, dict)
    }
    baseline_path = arm_by_name.get("baseline")
    winner_path = arm_by_name.get(winner)
    if baseline_path is None or winner_path is None:
        raise MlxTransferError(
            "Could not resolve baseline and winning train.py paths from arm manifest."
        )
    baseline_lines = baseline_path.read_text().splitlines(keepends=True)
    winner_lines = winner_path.read_text().splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            baseline_lines,
            winner_lines,
            fromfile=str(baseline_path),
            tofile=str(winner_path),
            n=3,
        )
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(diff)
    return out_path


def stage_public_transfer_continuation(
    arm_manifest_path: str | Path,
    benchmark_summary_path: str | Path,
    out_dir: str | Path,
    *,
    repo_root: str | Path,
) -> Path:
    """Create a winning-seed continuation workspace plus a TEVO-aware prompt."""
    manifest = json.loads(Path(arm_manifest_path).read_text())
    summary = json.loads(Path(benchmark_summary_path).read_text())
    winner = summary.get("winner")
    if not isinstance(winner, str) or not winner:
        raise MlxTransferError("Benchmark summary does not contain a winning seeded arm.")

    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise MlxTransferError(f"Invalid arm manifest at {arm_manifest_path}")
    arm_rows = {
        str(row.get("arm")): row
        for row in arms
        if isinstance(row, dict) and row.get("arm") in _ARM_LABELS
    }
    winner_row = arm_rows.get(winner)
    baseline_row = arm_rows.get("baseline")
    if winner_row is None or baseline_row is None:
        raise MlxTransferError("Arm manifest is missing the winner or baseline arm.")

    winner_repo = Path(str(winner_row.get("repo_path") or ""))
    baseline_repo = Path(str(baseline_row.get("repo_path") or ""))
    if not winner_repo.exists():
        raise MlxTransferError(f"Winner workspace missing at {winner_repo}")

    out_dir = Path(out_dir)
    continuation_repo = out_dir / "repo"
    _copy_repo_tree(winner_repo, continuation_repo)

    seed_snapshot_path = out_dir / "tevo_seed.train.py"
    shutil.copy2(continuation_repo / "train.py", seed_snapshot_path)
    zone_snapshot_path = out_dir / "tevo_zone_snapshot.json"
    zone_snapshot_path.write_text(
        json.dumps(extract_tevo_regions(seed_snapshot_path.read_text()), indent=2)
    )
    results_tsv_path = out_dir / "results.tsv"
    results_tsv_path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    prompt_path = out_dir / "program.tevo_seeded.md"
    prompt_path.write_text(
        _render_seeded_prompt(
            repo_root=Path(repo_root),
            continuation_repo=continuation_repo,
            baseline_repo=baseline_repo,
            seed_snapshot_path=seed_snapshot_path,
            zone_snapshot_path=zone_snapshot_path,
            results_tsv_path=results_tsv_path,
        )
    )
    manifest_path = out_dir / "continuation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "winner": winner,
                "winner_repo": str(winner_repo.resolve()),
                "baseline_repo": str(baseline_repo.resolve()),
                "continuation_repo": str(continuation_repo.resolve()),
                "seed_snapshot_path": str(seed_snapshot_path.resolve()),
                "zone_snapshot_path": str(zone_snapshot_path.resolve()),
                "program_path": str(prompt_path.resolve()),
                "results_tsv": str(results_tsv_path.resolve()),
            },
            indent=2,
        )
    )
    return manifest_path


def summarize_continuation_results(results_tsv_path: str | Path) -> dict[str, Any]:
    """Parse a continuation `results.tsv` into a best-so-far trajectory."""
    results_tsv_path = Path(results_tsv_path)
    if not results_tsv_path.exists():
        raise MlxTransferError(f"Continuation results file not found: {results_tsv_path}")

    trajectory: list[dict[str, Any]] = []
    best_val_bpb: float | None = None
    with results_tsv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader, start=1):
            raw_val = str(row.get("val_bpb", "") or "").strip()
            try:
                val_bpb = float(raw_val)
            except ValueError:
                val_bpb = None
            if val_bpb is not None and val_bpb > 0.0:
                best_val_bpb = val_bpb if best_val_bpb is None else min(best_val_bpb, val_bpb)
            trajectory.append(
                {
                    "experiment_index": index,
                    "commit": str(row.get("commit", "") or ""),
                    "status": str(row.get("status", "") or ""),
                    "description": str(row.get("description", "") or ""),
                    "val_bpb": val_bpb,
                    "best_val_bpb": best_val_bpb,
                }
            )
    return {
        "results_tsv": str(results_tsv_path.resolve()),
        "trajectory": trajectory,
        "best_val_bpb": best_val_bpb,
        "experiments": len(trajectory),
    }


def render_continuation_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown view of the seeded continuation trajectory."""
    lines = [
        "# Seeded Continuation Summary",
        "",
        "| Exp | Status | val_bpb | Best so far | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("trajectory") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {experiment_index} | {status} | {val_bpb} | {best_val_bpb} | {description} |".format(
                experiment_index=row.get("experiment_index", "-"),
                status=row.get("status", "-"),
                val_bpb=_fmt_metric(row.get("val_bpb")),
                best_val_bpb=_fmt_metric(row.get("best_val_bpb")),
                description=str(row.get("description", "") or "").replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def build_public_transfer_report(run_root: str | Path) -> tuple[dict[str, Any], str]:
    """Assemble the public-facing TEVO -> autoresearch-mlx report from run artifacts."""
    run_root = Path(run_root)
    report: dict[str, Any] = {
        "run_root": str(run_root.resolve()),
        "tevo_seed_summary": None,
        "mlx_benchmark_summary": None,
        "continuation_summary": None,
    }
    tevo_seed_path = run_root / "tevo_seed_summary.json"
    if tevo_seed_path.exists():
        report["tevo_seed_summary"] = json.loads(tevo_seed_path.read_text())

    benchmark_summary_path = run_root / "mlx_results" / "benchmark_summary.json"
    if benchmark_summary_path.exists():
        report["mlx_benchmark_summary"] = json.loads(benchmark_summary_path.read_text())

    continuation_results_path = run_root / "continuation" / "results.tsv"
    if continuation_results_path.exists():
        report["continuation_summary"] = summarize_continuation_results(continuation_results_path)

    lines = ["# Public TEVO -> autoresearch-mlx Report", ""]
    seed_summary = report["tevo_seed_summary"]
    if isinstance(seed_summary, dict):
        metrics = seed_summary.get("metrics") or {}
        lines.extend(
            [
                "## TEVO seed on openwebtext_10m",
                "",
                f"- `ppl_code`: `{_fmt_metric(metrics.get('ppl_code'))}`",
                "- `speedrun_flops_to_target`: "
                f"`{_fmt_metric(metrics.get('speedrun_flops_to_target'))}`",
                "",
            ]
        )
    bench_summary = report["mlx_benchmark_summary"]
    if isinstance(bench_summary, dict):
        lines.extend(
            [
                "## MLX baseline vs seeded arms",
                "",
                render_public_transfer_summary_markdown(bench_summary).rstrip(),
                "",
            ]
        )
    continuation_summary = report["continuation_summary"]
    if isinstance(continuation_summary, dict):
        lines.extend(
            [
                "## Seeded continuation trajectory",
                "",
                render_continuation_summary_markdown(continuation_summary).rstrip(),
                "",
            ]
        )
    return report, "\n".join(lines).rstrip() + "\n"


def extract_tevo_regions(text: str) -> dict[str, str]:
    """Return TEVO-owned marker regions keyed by their region name."""
    regions: dict[str, str] = {}
    for match in _TEVO_REGION_RE.finditer(text):
        name = str(match.group("name"))
        body = str(match.group("body"))
        regions[name] = body.strip()
    return regions


def audit_tevo_regions(seed_text: str, candidate_text: str) -> dict[str, Any]:
    """Compare TEVO-owned regions between the seed snapshot and a later train.py."""
    seed_regions = extract_tevo_regions(seed_text)
    candidate_regions = extract_tevo_regions(candidate_text)
    changed = sorted(
        {
            name
            for name in set(seed_regions) | set(candidate_regions)
            if seed_regions.get(name, "") != candidate_regions.get(name, "")
        }
    )
    return {
        "changed_regions": changed,
        "seed_region_count": len(seed_regions),
        "candidate_region_count": len(candidate_regions),
    }


def audit_tevo_regions_from_paths(
    seed_train_py_path: str | Path,
    candidate_train_py_path: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare TEVO-owned regions between two train.py files and optionally persist JSON."""
    seed_train_py_path = Path(seed_train_py_path)
    candidate_train_py_path = Path(candidate_train_py_path)
    payload = audit_tevo_regions(
        seed_train_py_path.read_text(), candidate_train_py_path.read_text()
    )
    payload.update(
        {
            "seed_train_py": str(seed_train_py_path.resolve()),
            "candidate_train_py": str(candidate_train_py_path.resolve()),
        }
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    return payload


def parse_autoresearch_log(text: str) -> dict[str, float | None]:
    """Extract the core autoresearch metrics from one train.py log."""
    val_match = _VAL_BPB_RE.search(text)
    peak_match = _PEAK_VRAM_MB_RE.search(text)
    val_bpb = float(val_match.group(1)) if val_match else None
    peak_memory_gb = float(peak_match.group(1)) / 1024.0 if peak_match else None
    return {"val_bpb": val_bpb, "peak_memory_gb": peak_memory_gb}


def detect_tevo_device(preferred: str = "auto") -> str:
    """Choose a local TEVO live-run device with MPS-first fallback."""
    choice = str(preferred or "auto").strip().lower()
    if choice and choice != "auto":
        return choice
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def normalize_command(command: Sequence[str] | str) -> list[str]:
    """Expose command normalization for CLI wiring and tests."""
    return _normalize_command(command)


def cost_conscious_modal_budget(
    generations: int,
    steps: int,
    eval_batches: int,
) -> dict[str, int]:
    """Clamp the first public Modal pass to a small, cheap exploratory budget."""
    return {
        "generations": max(1, min(int(generations), 4)),
        "steps": max(1, min(int(steps), 120)),
        "eval_batches": max(1, min(int(eval_batches), 4)),
    }


def _normalize_command(command: Sequence[str] | str) -> list[str]:
    if isinstance(command, str):
        parsed = shlex.split(command)
    else:
        parsed = [str(token) for token in command]
    if not parsed:
        raise MlxTransferError("Benchmark command may not be empty.")
    return parsed


def _metric_value(metrics: dict[str, float], key: str) -> float:
    try:
        return float(metrics.get(key, float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _numeric_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    numeric: dict[str, float] = {}
    for key, value in payload.items():
        try:
            numeric[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return numeric


def _recipe_model_key(recipe: TrainRecipe) -> str:
    return std_json.dumps(
        recipe.model.model_dump(mode="python"), sort_keys=True, separators=(",", ":")
    )


def _safe_slug(value: str) -> str:
    slug = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value).strip("_")
    return slug or "candidate"


def _copy_repo_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=_IGNORE_NAMES)
    source_venv = source / ".venv"
    if source_venv.exists():
        os.symlink(source_venv, dest / ".venv")


def _run_arm_once(
    *,
    arm: str,
    run_index: int,
    repo_path: Path,
    command: Sequence[str],
    timeout_seconds: int,
    log_path: Path,
) -> BenchmarkRunResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603
            list(command),
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        log_text = (completed.stdout or "") + (completed.stderr or "")
        log_path.write_text(log_text)
        parsed = parse_autoresearch_log(log_text)
        status = "ok" if completed.returncode == 0 and parsed["val_bpb"] is not None else "crash"
        return BenchmarkRunResult(
            arm=arm,
            run_index=run_index,
            status=status,
            val_bpb=parsed["val_bpb"],
            peak_memory_gb=parsed["peak_memory_gb"],
            log_path=log_path,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(stdout + stderr)
        return BenchmarkRunResult(
            arm=arm,
            run_index=run_index,
            status="timeout",
            val_bpb=None,
            peak_memory_gb=None,
            log_path=log_path,
        )


def _benchmark_rows(results: Sequence[BenchmarkRunResult]) -> list[dict[str, Any]]:
    return [
        {
            "arm": item.arm,
            "run_index": item.run_index,
            "status": item.status,
            "val_bpb": item.val_bpb,
            "peak_memory_gb": item.peak_memory_gb,
            "log_path": str(item.log_path),
        }
        for item in results
    ]


def _write_benchmark_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = "arm\trun_index\tval_bpb\tpeak_memory_gb\tstatus\tlog_path\n"
    lines = [header]
    for row in rows:
        lines.append(
            "{arm}\t{run_index}\t{val_bpb}\t{peak_memory_gb}\t{status}\t{log_path}\n".format(
                arm=row.get("arm", ""),
                run_index=row.get("run_index", ""),
                val_bpb=_fmt_metric(row.get("val_bpb")),
                peak_memory_gb=_fmt_metric(row.get("peak_memory_gb")),
                status=row.get("status", ""),
                log_path=row.get("log_path", ""),
            )
        )
    path.write_text("".join(lines))


def _median_or_none(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values]
    if not clean:
        return None
    return float(statistics.median(clean))


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:.1f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _render_seeded_prompt(
    *,
    repo_root: Path,
    continuation_repo: Path,
    baseline_repo: Path,
    seed_snapshot_path: Path,
    zone_snapshot_path: Path,
    results_tsv_path: Path,
) -> str:
    return f"""# TEVO-seeded autoresearch-mlx continuation

This workspace starts from a TEVO-rendered `train.py` seed instead of the stock baseline.

## Canon

- `prepare.py` is immutable.
- `train.py` is the only file you edit.
- The goal is still the lowest `val_bpb`.
- Keep the stock baseline reference repo untouched at: `{baseline_repo}`
- Log every experiment to: `{results_tsv_path}`

## First 12 experiments

- Treat all regions bounded by `# === TEVO TRAIN RECIPE: ... START/END ===` as frozen.
- You may edit only non-TEVO regions of `train.py`.
- After each experiment, audit whether any TEVO-owned region changed:

```bash
uv run --project {repo_root} python -m transformer_evolution_llm.cli mlx-transfer-audit \
  {seed_snapshot_path} {continuation_repo / "train.py"} \
  --out {zone_snapshot_path}
```

- If the audit reports changed regions during experiments 1-12, discard the change and revert.

## Experiments 13-24

- If there is still no improvement after 12 experiments, full-file edits are allowed.
- Continue logging whether TEVO-owned regions changed, but do not automatically
  discard on that basis.
- Record TEVO-zone edits explicitly in the `description` column of `results.tsv`.

## Reference files

- Seed snapshot: `{seed_snapshot_path}`
- TEVO zone snapshot JSON: `{zone_snapshot_path}`
- Current continuation repo: `{continuation_repo}`
"""
