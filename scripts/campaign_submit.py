"""Build a compact campaign bundle from an existing TEVO run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from transformer_evolution_llm.campaigns import (
    DEFAULT_FRONTIER_TOP_K,
    build_submission_bundle,
    load_campaign_manifest,
)

app = typer.Typer(help="Package one local run as a tracked campaign submission bundle.")


@app.command()
def main(
    campaign: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Path to campaign manifest YAML/JSON."),
    ],
    lane_id: Annotated[str, typer.Argument(help="Lane id defined in the campaign manifest.")],
    run: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Run directory or frontier.json path."),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory for the compact submission bundle."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Optional explicit config path when the run manifest is missing."),
    ] = None,
    frontier_top_k: Annotated[
        int,
        typer.Option(min=1, help="How many ranked frontier entries to keep in the bundle."),
    ] = DEFAULT_FRONTIER_TOP_K,
) -> None:
    """Shrink a run directory into a small, Git-tracked artifact bundle."""
    campaign_manifest = load_campaign_manifest(campaign)
    if out_dir is None:
        out_dir = Path("artifacts/campaigns") / campaign_manifest.campaign_id / lane_id
    manifest, summary = build_submission_bundle(
        campaign_manifest_path=campaign,
        lane_id=lane_id,
        run_path=run,
        output_dir=out_dir,
        config_path=config,
        frontier_top_k=frontier_top_k,
    )
    typer.echo(f"Submission bundle written to {out_dir}")
    typer.echo(f"- lane: {summary.lane_id}")
    typer.echo(f"- champion: {summary.champion.candidate_id}")
    typer.echo(f"- frontier entries: {summary.frontier_count}")
    typer.echo(f"- manifest: {out_dir / 'manifest.json'}")
    typer.echo(f"- source frontier: {manifest.source_frontier_path}")


if __name__ == "__main__":
    app()
