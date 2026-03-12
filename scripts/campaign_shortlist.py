"""Create a downstream-validation shortlist from an aggregate campaign report."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from transformer_evolution_llm.campaigns import build_shortlist_from_report

app = typer.Typer(help="Select the next candidates to validate downstream.")


@app.command()
def main(
    aggregate_report: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Path to aggregate_report.json."),
    ],
    out: Annotated[
        Path | None,
        typer.Option(help="Output JSON path for the shortlist."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(min=1, help="How many candidates to include."),
    ] = 3,
    bridge_only: Annotated[
        bool,
        typer.Option(
            "--bridge-only/--allow-any",
            help="Keep only candidates that the TrainRecipe bridge can express.",
        ),
    ] = True,
) -> None:
    """Write a compact shortlist for downstream validation."""
    if out is None:
        out = aggregate_report.with_name("shortlist.json")
    shortlist = build_shortlist_from_report(
        aggregate_report_path=aggregate_report,
        output_path=out,
        top_k=top_k,
        bridge_only=bridge_only,
    )
    typer.echo(f"Shortlist written to {out}")
    typer.echo(f"- candidates: {len(shortlist)}")


if __name__ == "__main__":
    app()
