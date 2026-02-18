"""Aggregate optimizer motifs across one or more frontier JSON files."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Summarize optimizer/filter motif convergence across frontiers.")


def _bucket_ratio(value: float) -> str:
    if value >= 0.95:
        return "0.95-1.00"
    if value >= 0.80:
        return "0.80-0.95"
    if value >= 0.60:
        return "0.60-0.80"
    if value >= 0.40:
        return "0.40-0.60"
    if value >= 0.20:
        return "0.20-0.40"
    return "0.00-0.20"


def _bucket_blend(value: float) -> str:
    return f"{round(float(value) * 4.0) / 4.0:.2f}"


def _bucket_eps(value: float) -> str:
    clamped = max(1e-12, float(value))
    power = round(float(abs(math.log10(clamped))))
    return f"1e-{int(power)}"


def _optimizer_signature(spec: dict[str, Any]) -> dict[str, Any]:
    train = spec.get("train", {}) if isinstance(spec, dict) else {}
    optimizer = train.get("optimizer", {}) if isinstance(train, dict) else {}
    gradient_transform = (
        optimizer.get("gradient_transform", {}) if isinstance(optimizer, dict) else {}
    )
    update_filter = optimizer.get("update_filter", {}) if isinstance(optimizer, dict) else {}

    name = str(optimizer.get("name", "adamw") or "adamw").lower()
    grad_mode = str(gradient_transform.get("mode", "identity") or "identity").lower()
    grad_ns_steps = int(gradient_transform.get("ns_steps", 5) or 5)
    grad_eps = float(gradient_transform.get("eps", 1.0e-8) or 1.0e-8)
    mode = str(update_filter.get("mode", "none") or "none").lower()
    granularity = str(update_filter.get("granularity", "element") or "element").lower()
    keep_ratio = float(update_filter.get("keep_ratio", 1.0) or 1.0)
    momentum_blend = float(update_filter.get("momentum_blend", 0.0) or 0.0)

    ratio_bucket = _bucket_ratio(keep_ratio)
    blend_bucket = _bucket_blend(momentum_blend)
    eps_bucket = _bucket_eps(grad_eps)
    key = (
        f"opt={name}|gt={grad_mode}|gt_ns={grad_ns_steps}|gt_eps={eps_bucket}|"
        f"mode={mode}|gran={granularity}|"
        f"keep={ratio_bucket}|blend={blend_bucket}"
    )
    return {
        "key": key,
        "optimizer_name": name,
        "grad_mode": grad_mode,
        "grad_ns_steps": grad_ns_steps,
        "grad_eps": grad_eps,
        "grad_eps_bucket": eps_bucket,
        "mode": mode,
        "granularity": granularity,
        "keep_ratio": keep_ratio,
        "keep_ratio_bucket": ratio_bucket,
        "momentum_blend": momentum_blend,
        "momentum_blend_bucket": blend_bucket,
    }


def _entry_sort_key(entry: dict[str, Any]) -> tuple[float, float]:
    metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
    flops = float(metrics.get("speedrun_flops_to_target", 1e18) or 1e18)
    ppl = float(metrics.get("ppl_code", 1e18) or 1e18)
    return (flops, ppl)


def _load_frontier(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise typer.BadParameter(f"{path} is not a frontier list JSON.")
    return [row for row in data if isinstance(row, dict)]


@app.command()
def main(
    frontier: list[Path] = typer.Option(
        ...,
        "--frontier",
        exists=True,
        readable=True,
        help="Frontier JSON path (repeatable).",
    ),
    top_k: int = typer.Option(5, min=1, help="How many top entries to include per frontier."),
    out: Path = typer.Option(
        Path("runs/optimizer_motif_report.json"), help="Output JSON report path."
    ),
) -> None:
    motif_counts: Counter[str] = Counter()
    motif_frontiers: dict[str, set[str]] = defaultdict(set)
    per_frontier: list[dict[str, Any]] = []

    for frontier_path in frontier:
        entries = _load_frontier(frontier_path)
        sorted_entries = sorted(entries, key=_entry_sort_key)
        top_entries = []
        for entry in sorted_entries[:top_k]:
            spec = entry.get("spec", {})
            if not isinstance(spec, dict):
                continue
            sig = _optimizer_signature(spec)
            metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
            motif_counts[sig["key"]] += 1
            motif_frontiers[sig["key"]].add(str(frontier_path))
            top_entries.append(
                {
                    "id": entry.get("id"),
                    "speedrun_flops_to_target": float(
                        metrics.get("speedrun_flops_to_target", 1e18) or 1e18
                    ),
                    "ppl_code": float(metrics.get("ppl_code", 1e18) or 1e18),
                    "signature": sig,
                }
            )

        best = top_entries[0] if top_entries else None
        per_frontier.append(
            {
                "frontier": str(frontier_path),
                "entries": len(entries),
                "best": best,
                "top": top_entries,
            }
        )

    motif_summary = []
    for key, total_count in motif_counts.items():
        frontiers_seen = sorted(motif_frontiers[key])
        motif_summary.append(
            {
                "motif": key,
                "entry_count": int(total_count),
                "frontier_count": len(frontiers_seen),
                "frontiers": frontiers_seen,
            }
        )
    motif_summary.sort(
        key=lambda item: (-int(item["frontier_count"]), -int(item["entry_count"]), item["motif"])
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frontier_count": len(frontier),
        "top_k": int(top_k),
        "per_frontier": per_frontier,
        "motifs": motif_summary,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    typer.echo(f"Wrote optimizer motif report to {out}")
    if motif_summary:
        top = motif_summary[0]
        typer.echo(
            "Most convergent motif: "
            f"{top['motif']} (frontiers={top['frontier_count']}, entries={top['entry_count']})"
        )


if __name__ == "__main__":
    app()
