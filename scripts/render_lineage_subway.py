"""Render lineage JSON as a generation-ordered Mermaid 'subway' diagram."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import typer


app = typer.Typer(help="Render lineage JSON into a generation-grouped Mermaid diagram.")


def _load_nodes(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("nodes", [])
    if not isinstance(data, list):
        raise typer.BadParameter("Lineage JSON must be a list of nodes.")
    return data


def _compute_generations(nodes: list[dict]) -> dict[str, int]:
    parents: Dict[str, List[str]] = {}
    for node in nodes:
        ident = node.get("id")
        if not ident:
            continue
        parents[ident] = [p for p in node.get("parents", []) if isinstance(p, str)]

    memo: dict[str, int] = {}

    def dfs(node_id: str, stack: set[str] | None = None) -> int:
        if node_id in memo:
            return memo[node_id]
        stack = stack or set()
        if node_id in stack:
            return 0
        stack.add(node_id)
        parent_ids = parents.get(node_id) or []
        if not parent_ids:
            memo[node_id] = 0
        else:
            gens = [dfs(pid, stack) for pid in parent_ids if pid in parents]
            memo[node_id] = (max(gens) + 1) if gens else 0
        stack.remove(node_id)
        return memo[node_id]

    for node_id in parents:
        dfs(node_id)
    return memo


def _label(node_id: str) -> str:
    prefix = node_id.split("-", 1)[0]
    suffix = node_id[-4:]
    return f"{prefix}\\n{suffix}"


@app.command()
def main(
    lineage: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("runs/lineage_subway.mmd")),
    title: str = typer.Option("Lineage Subway"),
    max_nodes: int = typer.Option(400, help="Cap nodes rendered to keep diagram readable."),
) -> None:
    nodes = _load_nodes(lineage)
    if len(nodes) > max_nodes:
        typer.echo(f"Lineage has {len(nodes)} nodes; rendering first {max_nodes}.")
        nodes = nodes[:max_nodes]
    gens = _compute_generations(nodes)
    gen_to_nodes: dict[int, list[str]] = {}
    for node in nodes:
        ident = node.get("id")
        if not ident:
            continue
        gen = gens.get(ident, 0)
        gen_to_nodes.setdefault(gen, []).append(ident)

    lines: list[str] = []
    lines.append("graph LR")
    lines.append(f"  %% {title}")
    for gen in sorted(gen_to_nodes):
        lines.append(f"  subgraph Gen{gen}")
        for ident in gen_to_nodes[gen]:
            label = _label(ident)
            lines.append(f'    {ident}["{label}"]')
        lines.append("  end")
    # edges
    for node in nodes:
        cid = node.get("id")
        if not cid:
            continue
        for parent in node.get("parents", []) or []:
            if parent and cid in gens and parent in gens:
                lines.append(f"  {parent} --> {cid}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    typer.echo(f"Wrote subway diagram with {len(nodes)} nodes to {out}")


if __name__ == "__main__":
    app()
