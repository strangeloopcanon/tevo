"""Build a multi-seed Parameter Golf runner state from prior frontier winners."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from transformer_evolution_llm.parameter_golf_champions import build_champion_state  # noqa: E402

app = typer.Typer(help="Compose a multi-seed Parameter Golf champions state.")


@app.command()
def main(
    base_config: Path = typer.Argument(..., exists=True, readable=True),
    seed_manifest: Path = typer.Argument(..., exists=True, readable=True),
    state_out: Path = typer.Argument(...),
    checkpoint_dir: Path = typer.Option(
        Path("runs/checkpoints_champions"),
        help="Checkpoint directory recorded into the saved runner state.",
    ),
    seed: int = typer.Option(0, help="Random seed to bake into the runner state."),
) -> None:
    """Write one saved runner state seeded with multiple prior frontier winners."""
    out_path = build_champion_state(
        base_config_path=base_config,
        seed_manifest_path=seed_manifest,
        state_out=state_out,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    typer.echo(out_path)


if __name__ == "__main__":
    app()
