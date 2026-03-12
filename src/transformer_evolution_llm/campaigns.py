"""Optional campaign helpers for packaging and comparing existing TEVO runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from json import dumps as std_json_dumps
from pathlib import Path
from typing import Any

import ujson as json
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from .candidates import ObjectiveDirection
from .dsl import ArchitectureSpec, load_architecture_spec, save_architecture_spec
from .train_recipe import (
    TrainRecipeCompatibilityError,
    save_train_recipe,
    train_recipe_from_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTIER_TOP_K = 5
DEFAULT_AGGREGATE_TOP_K = 10
BUNDLE_FRONTIER_FILENAME = "frontier_top.json"
LEGACY_BUNDLE_FRONTIER_FILENAME = "frontier_top5.json"
CHAMPION_RECIPE_FILENAME = "champion.train_recipe.yaml"


class CampaignBudget(BaseModel):
    """Shared search budget for a comparable campaign."""

    generations: int | None = Field(default=None, ge=1)
    steps: int | None = Field(default=None, ge=1)
    eval_batches: int | None = Field(default=None, ge=1)


class CampaignLane(BaseModel):
    """One contributor-owned lane inside a campaign."""

    lane_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    focus: str = Field(min_length=1)
    seed: int | None = None
    notes: list[str] = Field(default_factory=list)


class CampaignManifest(BaseModel):
    """Metadata needed to compare independently submitted runs."""

    schema_version: int = 1
    campaign_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    base_config: str = Field(min_length=1)
    config_fingerprint: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    objectives: dict[str, ObjectiveDirection] = Field(default_factory=dict)
    budget: CampaignBudget = Field(default_factory=CampaignBudget)
    dataset: str | None = None
    tokenizer: str | None = None
    seq_len: int | None = Field(default=None, gt=0)
    lanes: list[CampaignLane] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_campaign(self) -> CampaignManifest:
        if self.primary_metric not in self.objectives:
            raise ValueError("primary_metric must exist in objectives")
        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("lane ids must be unique")
        return self

    def lane_by_id(self, lane_id: str) -> CampaignLane:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        raise ValueError(f"Unknown lane_id {lane_id!r} for campaign {self.campaign_id}")


class MetricBounds(BaseModel):
    """Best and worst values observed for a metric."""

    best: float
    worst: float


class CandidateSummary(BaseModel):
    """Compact candidate snapshot used by campaign bundles and reports."""

    lane_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    spec_fingerprint: str = Field(min_length=1)
    primary_metric: float
    metrics: dict[str, float] = Field(default_factory=dict)
    rung: int = 0
    status: str = "unknown"
    parent: str | None = None
    mutation_trace: list[str] = Field(default_factory=list)
    bridge_compatible: bool = False
    source_frontier: str | None = None


class LineageSummary(BaseModel):
    """Compact lineage statistics for a submitted run."""

    node_count: int = 0
    completed_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    mutation_counts: dict[str, int] = Field(default_factory=dict)


class SubmissionManifest(BaseModel):
    """Pointers back to the original run and the derived bundle files."""

    schema_version: int = 1
    campaign_id: str = Field(min_length=1)
    lane_id: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    config_path: str | None = None
    config_fingerprint: str = Field(min_length=1)
    source_run_root: str = Field(min_length=1)
    source_frontier_path: str = Field(min_length=1)
    source_lineage_path: str | None = None
    source_state_path: str | None = None
    source_run_manifest_path: str | None = None
    budget: CampaignBudget = Field(default_factory=CampaignBudget)
    objectives: dict[str, ObjectiveDirection] = Field(default_factory=dict)
    bundle_files: dict[str, str] = Field(default_factory=dict)


class SubmissionSummary(BaseModel):
    """Human-sized summary of one submitted lane."""

    schema_version: int = 1
    campaign_id: str = Field(min_length=1)
    lane_id: str = Field(min_length=1)
    lane_title: str = Field(min_length=1)
    lane_focus: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    frontier_count: int = Field(ge=0)
    metric_bounds: dict[str, MetricBounds] = Field(default_factory=dict)
    champion: CandidateSummary
    top_candidates: list[CandidateSummary] = Field(default_factory=list)
    bridge_candidate_ids: list[str] = Field(default_factory=list)
    lineage: LineageSummary = Field(default_factory=LineageSummary)
    champion_spec_bundle_path: str = Field(min_length=1)
    champion_train_recipe_bundle_path: str | None = None


class RejectedSubmission(BaseModel):
    """Why an artifact directory was ignored during aggregation."""

    path: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class AggregatedSubmission(BaseModel):
    """Accepted lane bundle inside an aggregate report."""

    lane_id: str = Field(min_length=1)
    lane_title: str = Field(min_length=1)
    lane_focus: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    summary_path: str = Field(min_length=1)
    frontier_top_path: str = Field(min_length=1)
    frontier_count: int = Field(ge=0)
    champion: CandidateSummary
    bridge_candidate_ids: list[str] = Field(default_factory=list)


class AggregateReport(BaseModel):
    """Combined view of all comparable campaign submissions."""

    schema_version: int = 1
    campaign_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    objective_direction: ObjectiveDirection
    config_fingerprint: str = Field(min_length=1)
    expected_lanes: int = Field(ge=0)
    submitted_lanes: int = Field(ge=0)
    missing_lanes: list[str] = Field(default_factory=list)
    rejected_submissions: list[RejectedSubmission] = Field(default_factory=list)
    submissions: list[AggregatedSubmission] = Field(default_factory=list)
    pooled_top_candidates: list[CandidateSummary] = Field(default_factory=list)
    bridge_candidate_count: int = Field(ge=0, default=0)


class ShortlistEntry(BaseModel):
    """Downstream validation recommendation from an aggregate report."""

    rank: int = Field(ge=1)
    lane_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    primary_metric: float
    bridge_compatible: bool = False
    source_frontier: str | None = None


@dataclass(frozen=True)
class RunArtifactPaths:
    """Best-effort locations for the standard files next to a run."""

    run_root: Path
    frontier: Path
    lineage: Path | None
    state: Path | None
    manifest: Path | None


def load_campaign_manifest(path: str | Path) -> CampaignManifest:
    """Load a campaign manifest from YAML or JSON."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    return CampaignManifest(**payload)


def save_campaign_manifest(manifest: CampaignManifest, path: str | Path) -> None:
    """Persist a campaign manifest to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="python")
    if path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2))
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def config_fingerprint(config_path: str | Path) -> str:
    """Hash a TEVO config while ignoring lineage-only metadata."""
    return spec_fingerprint(load_architecture_spec(config_path))


def spec_fingerprint(spec: ArchitectureSpec) -> str:
    """Stable candidate fingerprint used for cross-run dedupe."""
    payload = _strip_lineage_metadata(spec.model_dump(mode="json"))
    blob = std_json_dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(blob.encode("utf-8")).hexdigest()


def resolve_run_artifact_paths(run_or_frontier: str | Path) -> RunArtifactPaths:
    """Resolve frontier, lineage, state, and manifest paths from a run dir or frontier file."""
    run_or_frontier = Path(run_or_frontier)
    if run_or_frontier.is_dir():
        run_root = run_or_frontier
        frontier = run_root / "frontier.json"
    else:
        frontier = run_or_frontier
        run_root = frontier.parent
    if not frontier.exists():
        raise FileNotFoundError(f"frontier.json not found at {frontier}")
    lineage = _first_existing(
        frontier.with_name(f"{frontier.stem}_lineage.json"),
        frontier.parent / "lineage.json",
    )
    state = _first_existing(
        frontier.with_name(f"{frontier.stem}.state.json"),
        frontier.parent / "runner.state.json",
    )
    manifest = _first_existing(frontier.with_name(f"{frontier.stem}.manifest.json"))
    return RunArtifactPaths(
        run_root=run_root.resolve(),
        frontier=frontier.resolve(),
        lineage=lineage.resolve() if lineage else None,
        state=state.resolve() if state else None,
        manifest=manifest.resolve() if manifest else None,
    )


def build_submission_bundle(
    *,
    campaign_manifest_path: str | Path,
    lane_id: str,
    run_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    frontier_top_k: int = DEFAULT_FRONTIER_TOP_K,
) -> tuple[SubmissionManifest, SubmissionSummary]:
    """Shrink a local run into a tracked campaign bundle."""
    campaign_manifest_path = Path(campaign_manifest_path)
    campaign = load_campaign_manifest(campaign_manifest_path)
    lane = campaign.lane_by_id(lane_id)
    artifacts = resolve_run_artifact_paths(run_path)
    frontier_entries = _load_frontier_entries(artifacts.frontier)
    if not frontier_entries:
        raise ValueError(f"Frontier is empty at {artifacts.frontier}")

    resolved_config = _resolve_config_path(
        explicit_config=config_path,
        run_manifest_path=artifacts.manifest,
        campaign_manifest_path=campaign_manifest_path,
        campaign=campaign,
    )
    spec = load_architecture_spec(resolved_config)
    actual_fingerprint = spec_fingerprint(spec)
    if actual_fingerprint != campaign.config_fingerprint:
        raise ValueError(
            "Run config fingerprint does not match campaign base config. "
            f"Expected {campaign.config_fingerprint}, got {actual_fingerprint}."
        )
    _validate_campaign_spec(campaign, spec)

    run_manifest_payload = _load_optional_json(artifacts.manifest)
    run_budget = _budget_from_run_manifest(run_manifest_payload)
    _validate_budget(campaign.budget, run_budget)

    direction = campaign.objectives[campaign.primary_metric]
    ranked_entries = sorted(
        frontier_entries,
        key=lambda entry: _entry_sort_key(
            entry=entry,
            metric=campaign.primary_metric,
            direction=direction,
        ),
    )
    top_entries = ranked_entries[: max(1, int(frontier_top_k))]
    bridge_candidate_ids = [
        str(entry.get("id"))
        for entry in ranked_entries
        if _is_bridge_compatible_entry(entry, frontier_path=artifacts.frontier)
    ]
    top_candidates = [
        _candidate_summary(
            entry,
            lane_id=lane.lane_id,
            primary_metric=campaign.primary_metric,
            direction=direction,
            source_frontier=artifacts.frontier,
        )
        for entry in top_entries
    ]
    champion = top_candidates[0]
    lineage_summary = _summarize_lineage_payload(_load_lineage_payload(artifacts.lineage))
    metric_bounds = _metric_bounds(
        frontier_entries,
        objectives=campaign.objectives,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_top_frontier_path = output_dir / LEGACY_BUNDLE_FRONTIER_FILENAME
    if legacy_top_frontier_path.exists():
        legacy_top_frontier_path.unlink()
    top_frontier_path = output_dir / BUNDLE_FRONTIER_FILENAME
    top_frontier_path.write_text(json.dumps(top_entries, indent=2))

    lineage_summary_path = output_dir / "lineage_summary.json"
    lineage_summary_path.write_text(json.dumps(lineage_summary.model_dump(mode="python"), indent=2))

    champion_entry = top_entries[0]
    champion_spec = ArchitectureSpec(**champion_entry["spec"])
    champion_spec_path = output_dir / "champion_spec.yaml"
    save_architecture_spec(champion_spec, champion_spec_path)

    champion_recipe_path = output_dir / CHAMPION_RECIPE_FILENAME
    try:
        recipe = train_recipe_from_spec(
            champion_spec,
            candidate_id=str(champion_entry.get("id") or ""),
            frontier_path=artifacts.frontier,
            metrics=_numeric_metrics(champion_entry.get("metrics")),
        )
    except (TrainRecipeCompatibilityError, ValidationError):
        recipe = None
    if recipe is not None:
        save_train_recipe(recipe, champion_recipe_path)
        recipe_written = True
    elif champion_recipe_path.exists():
        champion_recipe_path.unlink()
        recipe_written = False
    else:
        recipe_written = False

    submission_summary = SubmissionSummary(
        campaign_id=campaign.campaign_id,
        lane_id=lane.lane_id,
        lane_title=lane.title,
        lane_focus=lane.focus,
        primary_metric=campaign.primary_metric,
        frontier_count=len(frontier_entries),
        metric_bounds=metric_bounds,
        champion=champion.model_copy(update={"bridge_compatible": recipe_written}),
        top_candidates=top_candidates,
        bridge_candidate_ids=bridge_candidate_ids,
        lineage=lineage_summary,
        champion_spec_bundle_path=champion_spec_path.name,
        champion_train_recipe_bundle_path=(champion_recipe_path.name if recipe_written else None),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(submission_summary.model_dump(mode="python"), indent=2))

    bundle_files = {
        "summary": summary_path.name,
        "frontier_top": top_frontier_path.name,
        "lineage_summary": lineage_summary_path.name,
        "champion_spec": champion_spec_path.name,
    }
    if recipe_written:
        bundle_files["champion_train_recipe"] = champion_recipe_path.name
    submission_manifest = SubmissionManifest(
        campaign_id=campaign.campaign_id,
        lane_id=lane.lane_id,
        primary_metric=campaign.primary_metric,
        config_path=str(resolved_config),
        config_fingerprint=actual_fingerprint,
        source_run_root=str(artifacts.run_root),
        source_frontier_path=str(artifacts.frontier),
        source_lineage_path=str(artifacts.lineage) if artifacts.lineage is not None else None,
        source_state_path=str(artifacts.state) if artifacts.state is not None else None,
        source_run_manifest_path=(
            str(artifacts.manifest) if artifacts.manifest is not None else None
        ),
        budget=run_budget,
        objectives=campaign.objectives,
        bundle_files=bundle_files,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(submission_manifest.model_dump(mode="python"), indent=2))
    return submission_manifest, submission_summary


def aggregate_campaign_submissions(
    *,
    campaign_manifest_path: str | Path,
    artifacts_root: str | Path,
    output_dir: str | Path,
    top_k: int = DEFAULT_AGGREGATE_TOP_K,
) -> AggregateReport:
    """Merge comparable lane bundles into a single campaign report."""
    campaign_manifest_path = Path(campaign_manifest_path)
    campaign = load_campaign_manifest(campaign_manifest_path)
    artifacts_root = Path(artifacts_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    direction = campaign.objectives[campaign.primary_metric]

    rejected: list[RejectedSubmission] = []
    submissions: list[AggregatedSubmission] = []
    pooled_candidates: dict[str, CandidateSummary] = {}
    seen_lanes: set[str] = set()

    if artifacts_root.exists():
        lane_dirs = sorted(path for path in artifacts_root.iterdir() if path.is_dir())
    else:
        lane_dirs = []
    for lane_dir in lane_dirs:
        manifest_path = lane_dir / "manifest.json"
        summary_path = lane_dir / "summary.json"
        frontier_top_path = _resolve_bundle_frontier_path(lane_dir)
        if not manifest_path.exists() and not summary_path.exists() and frontier_top_path is None:
            continue
        if not manifest_path.exists() or not summary_path.exists() or frontier_top_path is None:
            rejected.append(
                RejectedSubmission(
                    path=str(lane_dir),
                    reason=(
                        "bundle is incomplete; expected manifest.json, summary.json, "
                        f"and {BUNDLE_FRONTIER_FILENAME}"
                    ),
                )
            )
            continue
        try:
            submission_manifest = SubmissionManifest(**json.loads(manifest_path.read_text()))
            submission_summary = SubmissionSummary(**json.loads(summary_path.read_text()))
            frontier_top_entries = _load_frontier_entries(frontier_top_path)
            _validate_submission_for_aggregate(
                campaign=campaign,
                manifest=submission_manifest,
                summary=submission_summary,
                seen_lanes=seen_lanes,
            )
        except (OSError, ValueError, ValidationError) as exc:
            rejected.append(RejectedSubmission(path=str(lane_dir), reason=str(exc)))
            continue

        seen_lanes.add(submission_manifest.lane_id)
        submissions.append(
            AggregatedSubmission(
                lane_id=submission_summary.lane_id,
                lane_title=submission_summary.lane_title,
                lane_focus=submission_summary.lane_focus,
                manifest_path=str(manifest_path.resolve()),
                summary_path=str(summary_path.resolve()),
                frontier_top_path=str(frontier_top_path.resolve()),
                frontier_count=submission_summary.frontier_count,
                champion=submission_summary.champion,
                bridge_candidate_ids=submission_summary.bridge_candidate_ids,
            )
        )

        for entry in frontier_top_entries:
            candidate = _candidate_summary(
                entry,
                lane_id=submission_summary.lane_id,
                primary_metric=campaign.primary_metric,
                direction=direction,
                source_frontier=Path(submission_manifest.source_frontier_path),
            )
            existing = pooled_candidates.get(candidate.spec_fingerprint)
            if existing is None or _candidate_beats(candidate, existing, direction=direction):
                pooled_candidates[candidate.spec_fingerprint] = candidate

    sorted_candidates = sorted(
        pooled_candidates.values(),
        key=lambda item: _summary_sort_key(item, direction=direction),
    )
    pooled_top = sorted_candidates[: max(1, int(top_k))]
    missing_lanes = [
        lane.lane_id
        for lane in campaign.lanes
        if lane.lane_id not in {item.lane_id for item in submissions}
    ]
    report = AggregateReport(
        campaign_id=campaign.campaign_id,
        title=campaign.title,
        primary_metric=campaign.primary_metric,
        objective_direction=direction,
        config_fingerprint=campaign.config_fingerprint,
        expected_lanes=len(campaign.lanes),
        submitted_lanes=len(submissions),
        missing_lanes=missing_lanes,
        rejected_submissions=rejected,
        submissions=submissions,
        pooled_top_candidates=pooled_top,
        bridge_candidate_count=sum(1 for item in sorted_candidates if item.bridge_compatible),
    )
    report_path = output_dir / "aggregate_report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="python"), indent=2))
    return report


def build_shortlist_from_report(
    *,
    aggregate_report_path: str | Path,
    output_path: str | Path,
    top_k: int = 3,
    bridge_only: bool = True,
) -> list[ShortlistEntry]:
    """Select the next candidates to validate downstream."""
    report = AggregateReport(**json.loads(Path(aggregate_report_path).read_text()))
    candidates = list(report.pooled_top_candidates)
    if bridge_only:
        candidates = [candidate for candidate in candidates if candidate.bridge_compatible]
    shortlist = [
        ShortlistEntry(
            rank=index + 1,
            lane_id=item.lane_id,
            candidate_id=item.candidate_id,
            primary_metric=item.primary_metric,
            bridge_compatible=item.bridge_compatible,
            source_frontier=item.source_frontier,
        )
        for index, item in enumerate(candidates[: max(1, int(top_k))])
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([entry.model_dump(mode="python") for entry in shortlist], indent=2)
    )
    return shortlist


def _candidate_summary(
    entry: dict[str, Any],
    *,
    lane_id: str,
    primary_metric: str,
    direction: ObjectiveDirection,
    source_frontier: Path,
) -> CandidateSummary:
    spec_payload = entry.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("Frontier entry is missing a spec payload")
    spec = ArchitectureSpec(**spec_payload)
    metrics = _numeric_metrics(entry.get("metrics"))
    return CandidateSummary(
        lane_id=lane_id,
        candidate_id=str(entry.get("id") or ""),
        spec_fingerprint=spec_fingerprint(spec),
        primary_metric=_metric_value(metrics, primary_metric, direction),
        metrics=metrics,
        rung=int(entry.get("rung", 0) or 0),
        status=str(entry.get("status") or "unknown"),
        parent=str(entry.get("parent")) if entry.get("parent") is not None else None,
        mutation_trace=_string_list(entry.get("mutation_trace")),
        bridge_compatible=_is_bridge_compatible_spec(
            spec, entry=entry, frontier_path=source_frontier
        ),
        source_frontier=str(source_frontier),
    )


def _validate_submission_for_aggregate(
    *,
    campaign: CampaignManifest,
    manifest: SubmissionManifest,
    summary: SubmissionSummary,
    seen_lanes: set[str],
) -> None:
    if manifest.campaign_id != campaign.campaign_id or summary.campaign_id != campaign.campaign_id:
        raise ValueError("submission campaign_id does not match the aggregate target")
    if manifest.config_fingerprint != campaign.config_fingerprint:
        raise ValueError("submission config_fingerprint does not match the campaign")
    if summary.primary_metric != campaign.primary_metric:
        raise ValueError("submission primary_metric does not match the campaign")
    if manifest.lane_id != summary.lane_id:
        raise ValueError("submission lane id mismatch between manifest and summary")
    campaign.lane_by_id(manifest.lane_id)
    if manifest.lane_id in seen_lanes:
        raise ValueError(f"duplicate lane submission for {manifest.lane_id}")
    _validate_budget(campaign.budget, manifest.budget)


def _validate_campaign_spec(campaign: CampaignManifest, spec: ArchitectureSpec) -> None:
    if campaign.objectives and dict(spec.evolution.objectives or {}) != dict(campaign.objectives):
        raise ValueError("Run config objectives do not match the campaign objectives")
    if campaign.tokenizer is not None and str(spec.data.tokenizer) != campaign.tokenizer:
        raise ValueError("Run tokenizer does not match the campaign tokenizer")
    if campaign.seq_len is not None and int(spec.data.seq_len) != campaign.seq_len:
        raise ValueError("Run seq_len does not match the campaign seq_len")
    if campaign.dataset is not None:
        dataset_fields = [
            str(spec.data.packed_train_path or ""),
            str(spec.data.packed_val_path or ""),
            ",".join(
                str(shard.name)
                for shard in getattr(spec.data, "shards", [])
                if getattr(shard, "name", None) is not None
            ),
        ]
        if campaign.dataset not in " ".join(dataset_fields):
            raise ValueError("Run dataset metadata does not match the campaign dataset")


def _validate_budget(expected: CampaignBudget, actual: CampaignBudget) -> None:
    for field in ("generations", "steps", "eval_batches"):
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value is None or actual_value is None:
            continue
        if expected_value != actual_value:
            raise ValueError(
                f"Run budget mismatch for {field}: expected {expected_value}, got {actual_value}"
            )


def _budget_from_run_manifest(payload: dict[str, Any] | None) -> CampaignBudget:
    if not isinstance(payload, dict):
        return CampaignBudget()
    return CampaignBudget(
        generations=_optional_int(payload.get("generations")),
        steps=_optional_int(payload.get("steps")),
        eval_batches=_optional_int(payload.get("eval_batches")),
    )


def _resolve_config_path(
    *,
    explicit_config: str | Path | None,
    run_manifest_path: Path | None,
    campaign_manifest_path: Path,
    campaign: CampaignManifest,
) -> Path:
    if explicit_config is not None:
        return _resolve_external_path(explicit_config, anchor=campaign_manifest_path.parent)
    if run_manifest_path is not None:
        payload = _load_optional_json(run_manifest_path)
        config = payload.get("config") if isinstance(payload, dict) else None
        if config:
            return _resolve_external_path(str(config), anchor=run_manifest_path.parent)
    raise ValueError(
        "Could not determine which TEVO config produced this run. "
        "Provide --config when packaging runs that do not include frontier.manifest.json. "
        f"Campaign base config: {campaign.base_config}"
    )


def _resolve_external_path(raw_path: str | Path, *, anchor: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    probes = (
        anchor / candidate,
        REPO_ROOT / candidate,
        Path.cwd() / candidate,
    )
    for probe in probes:
        if probe.exists():
            return probe.resolve()
    return (anchor / candidate).resolve()


def _metric_bounds(
    entries: list[dict[str, Any]],
    *,
    objectives: dict[str, ObjectiveDirection],
) -> dict[str, MetricBounds]:
    bounds: dict[str, MetricBounds] = {}
    for metric, direction in objectives.items():
        values: list[float] = []
        for entry in entries:
            metrics = _numeric_metrics(entry.get("metrics"))
            if metric not in metrics:
                continue
            values.append(metrics[metric])
        if not values:
            continue
        if direction == "min":
            bounds[metric] = MetricBounds(best=min(values), worst=max(values))
        else:
            bounds[metric] = MetricBounds(best=max(values), worst=min(values))
    return bounds


def _entry_sort_key(
    *,
    entry: dict[str, Any],
    metric: str,
    direction: ObjectiveDirection,
) -> tuple[float, str]:
    value = _metric_value(_numeric_metrics(entry.get("metrics")), metric, direction)
    candidate_id = str(entry.get("id") or "")
    if direction == "min":
        return (value, candidate_id)
    return (-value, candidate_id)


def _summary_sort_key(
    summary: CandidateSummary,
    *,
    direction: ObjectiveDirection,
) -> tuple[float, str, str]:
    if direction == "min":
        return (summary.primary_metric, summary.lane_id, summary.candidate_id)
    return (-summary.primary_metric, summary.lane_id, summary.candidate_id)


def _candidate_beats(
    lhs: CandidateSummary,
    rhs: CandidateSummary,
    *,
    direction: ObjectiveDirection,
) -> bool:
    if direction == "min":
        return lhs.primary_metric < rhs.primary_metric
    return lhs.primary_metric > rhs.primary_metric


def _load_frontier_entries(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Frontier JSON must be a list: {path}")
    return [entry for entry in payload if isinstance(entry, dict)]


def _load_lineage_payload(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        nodes = payload.get("nodes", [])
    else:
        nodes = payload
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, dict)]


def _summarize_lineage_payload(nodes: list[dict[str, Any]]) -> LineageSummary:
    statuses: Counter[str] = Counter()
    mutations: Counter[str] = Counter()
    completed = 0
    for node in nodes:
        status = str(node.get("status") or "unknown")
        statuses[status] += 1
        if status == "completed":
            completed += 1
        mutations.update(_string_list(node.get("mutation_trace")))
    return LineageSummary(
        node_count=len(nodes),
        completed_count=completed,
        status_counts=dict(statuses),
        mutation_counts=dict(mutations),
    )


def _numeric_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in payload.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _metric_value(
    metrics: dict[str, float],
    metric: str,
    direction: ObjectiveDirection,
) -> float:
    if metric in metrics:
        return metrics[metric]
    return float("inf") if direction == "min" else float("-inf")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return None
    return payload


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_bundle_frontier_path(bundle_dir: Path) -> Path | None:
    return _first_existing(
        bundle_dir / BUNDLE_FRONTIER_FILENAME,
        bundle_dir / LEGACY_BUNDLE_FRONTIER_FILENAME,
    )


def _strip_lineage_metadata(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _strip_lineage_metadata(value)
            for key, value in payload.items()
            if key not in {"origin_id", "parent_origin"}
        }
    if isinstance(payload, list):
        return [_strip_lineage_metadata(item) for item in payload]
    return payload


def _string_list(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if isinstance(item, str)]


def _is_bridge_compatible_entry(entry: dict[str, Any], *, frontier_path: Path) -> bool:
    spec_payload = entry.get("spec")
    if not isinstance(spec_payload, dict):
        return False
    try:
        spec = ArchitectureSpec(**spec_payload)
    except ValidationError:
        return False
    return _is_bridge_compatible_spec(spec, entry=entry, frontier_path=frontier_path)


def _is_bridge_compatible_spec(
    spec: ArchitectureSpec,
    *,
    entry: dict[str, Any],
    frontier_path: Path,
) -> bool:
    try:
        train_recipe_from_spec(
            spec,
            candidate_id=str(entry.get("id") or ""),
            frontier_path=frontier_path,
            metrics=_numeric_metrics(entry.get("metrics")),
        )
    except (TrainRecipeCompatibilityError, ValidationError):
        return False
    return True
