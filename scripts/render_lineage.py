"""Render a lineage JSON into a Mermaid graph (.mmd).

Input JSON shape (as produced by EvolutionRunner.save_lineage):
- legacy list payload:
  [{"id": "child", "parents": ["parentA", "parentB"], ...}, ...]
- current object payload:
  {"nodes": [...], "novelty_archive": [...], ...}

Usage:
  python scripts/render_lineage.py --lineage runs/frontier_phi_creative_unbiased_lineage_from_log.json \
    --out runs/creative_unbiased_lineage.mmd
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(help="Render lineage JSON to Mermaid graph")


@app.command()
def main(
    lineage: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(...),
    title: Optional[str] = typer.Option(None),
) -> None:
    payload = json.loads(lineage.read_text())
    if isinstance(payload, dict):
        nodes = payload.get("nodes", [])
    else:
        nodes = payload
    if not isinstance(nodes, list):
        nodes = []
    lines: list[str] = []
    lines.append("graph LR")
    if title:
        lines.append(f"%% {title}")
    # Add nodes
    ids = {n.get("id") for n in nodes if n.get("id")}
    for n in nodes:
        cid = n.get("id")
        if not cid:
            continue
        label = cid
        lines.append(f"  {cid}[\"{label}\"]")
    # Add edges
    edge_count = 0
    for n in nodes:
        cid = n.get("id")
        parents = n.get("parents") or []
        for p in parents:
            if p and cid and (p in ids):
                lines.append(f"  {p} --> {cid}")
                edge_count += 1
    out.write_text("\n".join(lines) + "\n")
    typer.echo(f"Wrote Mermaid graph with {len(ids)} nodes and {edge_count} edges to {out}")


if __name__ == "__main__":
    app()
