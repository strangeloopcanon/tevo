"""Aggregate comparable campaign submission bundles into one report."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from transformer_evolution_llm.campaigns import (
    DEFAULT_AGGREGATE_TOP_K,
    aggregate_campaign_submissions,
    load_campaign_manifest,
)

app = typer.Typer(help="Merge comparable campaign submission bundles.")


@app.command()
def main(
    campaign: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Path to campaign manifest YAML/JSON."),
    ],
    artifacts_root: Annotated[
        Path | None,
        typer.Option(help="Directory containing one lane bundle per subdirectory."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for aggregate_report.json."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(min=1, help="How many pooled candidates to keep in the report."),
    ] = DEFAULT_AGGREGATE_TOP_K,
) -> None:
    """Read campaign bundles and emit one aggregate report."""
    campaign_manifest = load_campaign_manifest(campaign)
    if artifacts_root is None:
        artifacts_root = Path("artifacts/campaigns") / campaign_manifest.campaign_id
    if out_dir is None:
        out_dir = artifacts_root / "_aggregate"
    report = aggregate_campaign_submissions(
        campaign_manifest_path=campaign,
        artifacts_root=artifacts_root,
        output_dir=out_dir,
        top_k=top_k,
    )
    typer.echo(f"Aggregate report written to {out_dir / 'aggregate_report.json'}")
    typer.echo(f"- accepted lanes: {report.submitted_lanes}/{report.expected_lanes}")
    typer.echo(f"- rejected bundles: {len(report.rejected_submissions)}")
    typer.echo(f"- pooled candidates: {len(report.pooled_top_candidates)}")


if __name__ == "__main__":
    app()
