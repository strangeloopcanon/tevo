"""Candidate and frontier tracking utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .dsl import ArchitectureSpec

Status = Literal["pending", "running", "completed", "failed"]


@dataclass
class Candidate:
    """Represents a single architecture variant."""

    ident: str
    spec: ArchitectureSpec
    parent: str | None = None
    parent_checkpoint: Path | None = None
    seed_state_path: Path | None = None
    rung: int = 0
    status: Status = "pending"
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint: Path | None = None
    mutation_trace: list[str] = field(default_factory=list)
    crossover_report: dict[str, Any] | None = None

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def score(self, weights: dict[str, float]) -> float:
        if self.status != "completed":
            return -math.inf
        total = 0.0
        for metric, weight in weights.items():
            value = self.metrics.get(metric)
            if value is None:
                return -math.inf
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return -math.inf
            if not math.isfinite(value_f):
                return -math.inf
            total += value_f * float(weight)
        return total

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.ident,
            "parent": self.parent,
            "parent_checkpoint": str(self.parent_checkpoint) if self.parent_checkpoint else None,
            "seed_state_path": str(self.seed_state_path) if self.seed_state_path else None,
            "rung": self.rung,
            "status": self.status,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "spec": self.spec.model_dump(mode="python"),
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "mutation_trace": self.mutation_trace,
            "crossover_report": self.crossover_report,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Candidate:
        spec_data = data.get("spec")
        if not isinstance(spec_data, dict):
            raise ValueError("Candidate spec missing in state.")
        ident = str(data.get("id"))
        parent_raw = data.get("parent")
        parent = str(parent_raw) if parent_raw is not None else None
        parent_checkpoint = (
            Path(data["parent_checkpoint"]) if data.get("parent_checkpoint") else None
        )
        seed_state_path = Path(data["seed_state_path"]) if data.get("seed_state_path") else None
        checkpoint = Path(data["checkpoint"]) if data.get("checkpoint") else None
        metrics = data.get("metrics", {}) or {}
        if not isinstance(metrics, dict):
            metrics = {}
        metadata = data.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        mutation_trace = data.get("mutation_trace", []) or []
        if not isinstance(mutation_trace, list):
            mutation_trace = []
        mutation_trace = [str(name) for name in mutation_trace if isinstance(name, str)]
        crossover_report = data.get("crossover_report")
        if not isinstance(crossover_report, dict):
            crossover_report = None
        return cls(
            ident=ident,
            spec=ArchitectureSpec(**spec_data),
            parent=parent,
            parent_checkpoint=parent_checkpoint,
            seed_state_path=seed_state_path,
            rung=int(data.get("rung", 0)),
            status=data.get("status", "pending"),
            metrics=metrics,
            metadata=metadata,
            checkpoint=checkpoint,
            mutation_trace=mutation_trace,
            crossover_report=crossover_report,
        )


ObjectiveDirection = Literal["max", "min"]


class ParetoFrontier:
    """Maintains a Pareto-optimal set of candidates."""

    def __init__(self, objectives: dict[str, ObjectiveDirection]) -> None:
        self.objectives = objectives
        self._entries: list[Candidate] = []

    @property
    def entries(self) -> list[Candidate]:
        return list(self._entries)

    def update(self, candidate: Candidate) -> None:
        # Ignore candidates missing objective metrics or producing NaN/Inf.
        for metric in self.objectives:
            val = candidate.metrics.get(metric)
            if val is None:
                return
            try:
                if not math.isfinite(float(val)):
                    return
            except (TypeError, ValueError):
                return
        dominated = []
        for idx, existing in enumerate(self._entries):
            if self._dominates(candidate, existing):
                dominated.append(idx)
            elif self._dominates(existing, candidate):
                return
        for idx in reversed(dominated):
            del self._entries[idx]
        self._entries.append(candidate)

    def _dominates(self, a: Candidate, b: Candidate) -> bool:
        better_or_equal = True
        strictly_better = False
        for metric, direction in self.objectives.items():
            a_val = a.metrics.get(metric)
            b_val = b.metrics.get(metric)
            if a_val is None or b_val is None:
                continue
            if direction == "max":
                better = a_val >= b_val
                strictly = a_val > b_val
            else:
                better = a_val <= b_val
                strictly = a_val < b_val
            if not better:
                better_or_equal = False
                break
            if strictly:
                strictly_better = True
        return better_or_equal and strictly_better

    def to_json(self) -> list[dict[str, Any]]:
        return [entry.serialize() for entry in self._entries]
