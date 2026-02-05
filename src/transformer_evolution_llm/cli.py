"""Command-line utilities for the transformer_evolution_llm package."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import ujson as json
from rich.console import Console
from rich.table import Table

from . import get_version
from .api import convert_checkpoints, export_frontier_seed, prune_checkpoints, run_evolution
from .cache_builder import synthesize_cache

app = typer.Typer(help="Evolutionary search loop utilities")
console = Console()


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(get_version())


@app.command()
def run(
    config: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    generations: Annotated[int, typer.Option(min=1)] = 4,
    mode: Annotated[str, typer.Option(help="Evaluation backend (simulate/live).")] = "simulate",
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option()] = Path("runs/frontier.json"),
) -> None:
    """Execute an evolutionary search run."""
    run_evolution(config_path=config, generations=generations, mode=mode, seed=seed, out_path=out)


@app.command("cache")
def cache_cmd(
    out_dir: Annotated[Path, typer.Argument()] = Path("cache/phi_tiny_topk16"),
    samples: Annotated[int, typer.Option(min=1)] = 128,
    seq_len: Annotated[int, typer.Option(min=16)] = 2048,
    topk: Annotated[int, typer.Option(min=1)] = 8,
    vocab: Annotated[int, typer.Option(min=256)] = 100_352,
) -> None:
    """Build a synthetic teacher logit cache for pipeline testing."""
    synthesize_cache(out_dir, samples=samples, seq_len=seq_len, topk=topk, vocab=vocab)


@app.command()
def frontier(path: Annotated[Path, typer.Argument()] = Path("runs/frontier.json")) -> None:
    """Print the latest Pareto frontier file."""
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist")
    data = json.loads(path.read_text())
    table = Table(title=f"Frontier ({path})")
    table.add_column("ID")
    table.add_column("Parent")
    table.add_column("ppl_code")
    table.add_column("throughput")
    for row in data:
        table.add_row(
            row["id"],
            str(row.get("parent", "-")),
            f"{row.get('metrics', {}).get('ppl_code', 0.0):.2f}",
            f"{row.get('metrics', {}).get('throughput', 0.0):.2f}",
        )
    console.print(table)


@app.command("export-seed")
def export_seed(
    frontier_path: Annotated[Path, typer.Argument(help="Path to Pareto frontier JSON file.")],
    candidate_id: Annotated[str, typer.Argument(help="Candidate identifier to export.")],
    out: Annotated[
        Path,
        typer.Argument(help="Destination config path (e.g., configs/seed_<id>.yaml)."),
    ],
    checkpoint_dir: Annotated[
        Path,
        typer.Option(help="Directory containing candidate checkpoints."),
    ] = Path("runs/checkpoints"),
) -> None:
    """Export a frontier candidate spec + checkpoint pointer as a seed config."""
    export_frontier_seed(
        frontier_path=frontier_path,
        candidate_id=candidate_id,
        out_path=out,
        checkpoint_dir=checkpoint_dir,
    )
    console.print(f"[bold green]Seed config written:[/] {out}")


@app.command("resume-state")
def resume_state(
    state_path: Annotated[Path, typer.Argument(help="Path to saved runner state JSON.")],
    generations: Annotated[int, typer.Option(min=1)] = 4,
    mode: Annotated[str, typer.Option(help="Evaluation backend (simulate/full).")] = "simulate",
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option()] = Path("runs/frontier.json"),
) -> None:
    """Resume from a saved state and continue for more generations."""
    from .orchestrator import EvolutionRunner

    runner = EvolutionRunner.load_state(state_path, mode=mode)
    runner.rng.seed(seed)
    runner.run(generations)
    runner.save_frontier(out)
    console.print(f"[bold green]Frontier written:[/] {out}")


@app.command("prune-checkpoints")
def prune_checkpoints_cmd(
    checkpoint_dir: Annotated[
        Path, typer.Argument(help="Directory containing checkpoints to prune.")
    ] = Path("runs/checkpoints"),
    frontier_path: Annotated[
        Path | None,
        typer.Option(help="Frontier JSON to keep (ids referenced will be retained)."),
    ] = None,
    state_path: Annotated[
        Path | None,
        typer.Option(help="State JSON to keep (ids referenced will be retained)."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Show what would be removed without deleting.")
    ] = False,
) -> None:
    """Delete checkpoints not referenced by the provided frontier/state."""
    kept, removed = prune_checkpoints(
        checkpoint_dir=checkpoint_dir,
        frontier_path=frontier_path,
        state_path=state_path,
        dry_run=dry_run,
    )
    console.print(f"[bold]Kept:[/] {len(kept)} checkpoints")
    console.print(
        f"[bold]Removed:[/] {len(removed)} checkpoints" + (" (dry run)" if dry_run else "")
    )
    if dry_run and removed:
        console.print("Would remove:")
        for path in removed:
            console.print(f"- {path}")


@app.command("cleanup-run")
def cleanup_run_cmd(
    manifest: Annotated[Path, typer.Argument(help="Path to runs/*.manifest.json")],
    keep: Annotated[
        str,
        typer.Option(help="What to keep: frontier | state | frontier+state."),
    ] = "frontier",
    apply: Annotated[bool, typer.Option(help="Apply deletions (default is dry run).")] = False,
) -> None:
    """Prune a run's checkpoints using its manifest metadata."""
    if not manifest.exists():
        raise typer.BadParameter(f"{manifest} does not exist")
    payload = json.loads(manifest.read_text())
    checkpoint_dir_raw = payload.get("checkpoint_dir")
    frontier_raw = payload.get("frontier")
    if not checkpoint_dir_raw or not frontier_raw:
        raise typer.BadParameter("manifest missing checkpoint_dir/frontier fields")
    checkpoint_dir = Path(checkpoint_dir_raw)
    frontier_path = Path(frontier_raw)
    state_path = frontier_path.with_name(frontier_path.stem + ".state.json")
    frontier_arg: Path | None = None
    state_arg: Path | None = None
    keep_norm = (keep or "frontier").lower()
    if keep_norm in {"frontier", "frontier+state"}:
        frontier_arg = frontier_path
    if keep_norm in {"state", "frontier+state"} and state_path.exists():
        state_arg = state_path
    # If applying deletions, compute size first (before files are removed).
    preview_kept, preview_removed = prune_checkpoints(
        checkpoint_dir=checkpoint_dir,
        frontier_path=frontier_arg,
        state_path=state_arg,
        dry_run=True,
    )
    removed_bytes = 0
    for path in preview_removed:
        try:
            removed_bytes += path.stat().st_size
        except OSError:
            continue
    kept, removed = preview_kept, preview_removed
    if apply and removed:
        kept, removed = prune_checkpoints(
            checkpoint_dir=checkpoint_dir,
            frontier_path=frontier_arg,
            state_path=state_arg,
            dry_run=False,
        )
    gb = removed_bytes / (1024**3)
    console.print(f"[bold]Kept:[/] {len(kept)} checkpoints")
    console.print(
        f"[bold]Removed:[/] {len(removed)} checkpoints"
        + (" (dry run)" if not apply else "")
        + f" (≈{gb:.2f} GiB)"
    )


@app.command("convert-checkpoints")
def convert_checkpoints_cmd(
    checkpoint_dir: Annotated[
        Path, typer.Argument(help="Directory containing candidate checkpoints.")
    ],
    dtype: Annotated[str, typer.Option(help="Target dtype: fp16 | bf16 | fp32.")] = "fp16",
    apply: Annotated[bool, typer.Option(help="Apply conversion (default is dry run).")] = False,
) -> None:
    """Downcast checkpoint tensors to shrink disk usage."""
    paths, before, after = convert_checkpoints(
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        dry_run=not apply,
    )
    gb_before = before / (1024**3)
    gb_after = after / (1024**3)
    gb_saved = (before - after) / (1024**3)
    console.print(f"[bold]Checkpoints:[/] {len(paths)}")
    console.print(
        f"[bold]Size:[/] {gb_before:.2f} GiB -> {gb_after:.2f} GiB"
        + (" (dry run)" if not apply else "")
    )
    if apply:
        console.print(f"[bold]Saved:[/] ≈{gb_saved:.2f} GiB")


def main() -> None:
    """Entry point for `python -m transformer_evolution_llm.cli`."""
    app()


if __name__ == "__main__":
    main()
