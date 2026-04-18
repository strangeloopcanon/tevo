"""Helpers for multi-seed Parameter Golf champion searches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import load_spec
from .candidates import Candidate
from .dsl import ArchitectureSpec
from .orchestrator import EvolutionRunner


@dataclass(frozen=True)
class ChampionSeedSource:
    """One prior frontier entry to import into a champions pool."""

    name: str
    frontier_path: Path
    candidate_id: str | None = None


def load_champion_seed_manifest(path: str | Path) -> list[ChampionSeedSource]:
    """Load a simple JSON manifest describing frontier-backed champion seeds."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("Champion seed manifest must be a JSON list.")
    seeds: list[ChampionSeedSource] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Champion seed entry {index} must be an object.")
        name = str(entry.get("name") or "").strip()
        frontier_raw = str(entry.get("frontier_path") or "").strip()
        if not name:
            raise ValueError(f"Champion seed entry {index} is missing name.")
        if not frontier_raw:
            raise ValueError(f"Champion seed entry {index} is missing frontier_path.")
        candidate_raw = entry.get("candidate_id")
        candidate_id = None
        if candidate_raw is not None and str(candidate_raw).strip():
            candidate_id = str(candidate_raw).strip()
        seeds.append(
            ChampionSeedSource(
                name=name,
                frontier_path=Path(frontier_raw),
                candidate_id=candidate_id,
            )
        )
    return seeds


def build_champion_state(
    *,
    base_config_path: str | Path,
    seed_manifest_path: str | Path,
    state_out: str | Path,
    checkpoint_dir: str | Path,
    seed: int = 0,
) -> Path:
    """Build a saved runner state seeded with multiple champion candidates."""
    base_spec = load_spec(base_config_path)
    seed_sources = load_champion_seed_manifest(seed_manifest_path)
    if not seed_sources:
        raise ValueError("Champion seed manifest is empty.")

    pool = [
        build_champion_candidate(
            base_spec=base_spec,
            source=seed_source,
            index=index,
        )
        for index, seed_source in enumerate(seed_sources, start=1)
    ]
    if not pool:
        raise ValueError("No champion seeds could be constructed.")

    runner = EvolutionRunner(
        base_spec=pool[0].spec.model_copy(deep=True),
        evolution_cfg=pool[0].spec.evolution,
        mode="simulate",
        seed=seed,
    )
    runner.checkpoint_dir = Path(checkpoint_dir)
    runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runner.pool = list(pool)
    runner._history = list(pool)
    runner._parents = {candidate.ident: [] for candidate in pool}
    runner.frontier._entries = list(pool)
    runner._rebuild_seen_specs()
    out_path = Path(state_out)
    runner.save_state(out_path)
    return out_path


def build_champion_candidate(
    *,
    base_spec: ArchitectureSpec,
    source: ChampionSeedSource,
    index: int,
) -> Candidate:
    """Construct one completed candidate from a prior frontier entry."""
    entry = load_frontier_entry(source.frontier_path, source.candidate_id)
    spec_payload = entry.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError(
            f"Frontier entry {source.name!r} in {source.frontier_path} is missing spec data."
        )
    metrics_payload = entry.get("metrics")
    metrics = _coerce_metrics(metrics_payload)
    source_spec = ArchitectureSpec(**spec_payload)
    normalized = normalize_seed_spec(base_spec, source_spec, source_name=source.name)
    metadata = dict(entry.get("metadata") or {})
    metadata.update(
        {
            "champion_seed_name": source.name,
            "champion_seed_frontier": str(source.frontier_path),
            "champion_seed_candidate_id": str(entry.get("id") or ""),
            "champion_seed_source_family": (
                source_spec.parameter_golf.seed_family
                if source_spec.parameter_golf is not None
                else None
            ),
        }
    )
    ident = f"champion-{index:02d}-{_slug(source.name)}"
    return Candidate(
        ident=ident,
        spec=normalized,
        rung=2,
        status="completed",
        metrics=metrics,
        metadata=metadata,
        mutation_trace=[],
    )


def load_frontier_entry(frontier_path: str | Path, candidate_id: str | None = None) -> dict[str, Any]:
    """Load one frontier entry, defaulting to the best post-quant score."""
    path = Path(frontier_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Frontier JSON must be a list: {path}")
    entries = [entry for entry in payload if isinstance(entry, dict)]
    if not entries:
        raise ValueError(f"Frontier {path} does not contain any entries.")
    if candidate_id is not None:
        for entry in entries:
            if str(entry.get("id") or "") == candidate_id:
                return entry
        raise ValueError(f"Candidate {candidate_id!r} not found in frontier {path}.")
    return min(entries, key=_frontier_sort_key)


def normalize_seed_spec(
    base_spec: ArchitectureSpec,
    source_spec: ArchitectureSpec,
    *,
    source_name: str,
) -> ArchitectureSpec:
    """Keep the source recipe, but normalize the run settings to the new lane."""
    spec = source_spec.model_copy(deep=True)
    spec.data = base_spec.data.model_copy(deep=True)
    spec.parameter_golf = (
        base_spec.parameter_golf.model_copy(deep=True)
        if base_spec.parameter_golf is not None
        else None
    )
    spec.evolution = base_spec.evolution.model_copy(deep=True)
    spec.priors = base_spec.priors.model_copy(deep=True)
    spec.model.name = f"{base_spec.model.name}-{_slug(source_name)}"
    if spec.parameter_golf is not None:
        spec.parameter_golf.seed_family = f"champion::{_slug(source_name)}"
        if base_spec.parameter_golf is not None:
            spec.parameter_golf.lane_kind = base_spec.parameter_golf.lane_kind
            spec.parameter_golf.exportable_family = base_spec.parameter_golf.exportable_family
    return spec


def _frontier_sort_key(entry: dict[str, Any]) -> tuple[float, float]:
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return (1e9, 1e9)
    post_quant = _metric_value(metrics.get("post_quant_val_bpb"))
    artifact_total = _metric_value(metrics.get("artifact_total_bytes"))
    return (post_quant, artifact_total)


def _coerce_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        numeric = _metric_value(value)
        if numeric == 1e9 and value not in {1e9, 1.0e9}:
            continue
        metrics[key] = numeric
    return metrics


def _metric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1e9


def _slug(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
            continue
        if char in {"-", "_"}:
            safe.append(char)
            continue
        safe.append("-")
    collapsed = "".join(safe).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed or "seed"


__all__ = [
    "ChampionSeedSource",
    "build_champion_candidate",
    "build_champion_state",
    "load_champion_seed_manifest",
    "load_frontier_entry",
    "normalize_seed_spec",
]
