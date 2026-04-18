"""Plan and smoke-test broader Parameter Golf combo scout searches."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from transformer_evolution_llm.parameter_golf_combo_scout import (  # noqa: E402
    build_combo_plan,
    run_local_smoke_combo_scouts,
)

app = typer.Typer(help="Broader Parameter Golf combo scout helpers.", no_args_is_help=True)
console = Console()


@app.command("plan")
def plan_cmd(
    stage: Annotated[str, typer.Option(help="scout or refine")] = "scout",
    device: Annotated[
        str, typer.Option(help="Suggested local device for the command output.")
    ] = "cuda",
    seed: Annotated[int, typer.Option()] = 0,
    family: Annotated[
        list[str] | None,
        typer.Option("--family", help="Optional family id filter; repeatable."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option(help="Print the plan as JSON instead of a table.")
    ] = False,
) -> None:
    """Show the recommended broader combo search order and commands."""
    plan = build_combo_plan(stage=stage, device=device, seed=seed, family_ids=family)
    if as_json:
        console.print_json(data=plan)
        return

    table = Table(title=f"Parameter Golf {stage.title()} Combo Plan")
    table.add_column("Priority", justify="right")
    table.add_column("Family")
    table.add_column("Exportable")
    table.add_column("Budget")
    table.add_column("Why")
    for row in plan:
        budget = row["budget"]
        table.add_row(
            str(row["priority"]),
            str(row["family_id"]),
            "yes" if row["exportable"] else "motif-only",
            f"{budget['generations']}g/{budget['steps']}s/{budget['eval_batches']}e/{budget['mutation_steps']}m",
            str(row["note"]),
        )
    console.print(table)
    for row in plan:
        console.print(f"[bold]{row['family_id']}[/]")
        console.print("  local : " + " ".join(str(part) for part in row["local_command"]))
        console.print("  runpod: " + " ".join(str(part) for part in row["runpod_command"]))


@app.command("smoke")
def smoke_cmd(
    out_root: Annotated[Path, typer.Option(help="Directory to write smoke artifacts into.")] = Path(
        "runs/parameter_golf_combo_scout_smoke"
    ),
    generations: Annotated[int, typer.Option(min=1)] = 1,
    steps: Annotated[int, typer.Option(min=1)] = 1,
    eval_batches: Annotated[int, typer.Option(min=1)] = 1,
    device: Annotated[str, typer.Option(help="cpu, mps, or cuda")] = "cpu",
    seed: Annotated[int, typer.Option()] = 0,
    family: Annotated[
        list[str] | None,
        typer.Option("--family", help="Optional family id filter; repeatable."),
    ] = None,
) -> None:
    """Run a tiny local smoke search for the combo lanes."""
    report = run_local_smoke_combo_scouts(
        out_root,
        family_ids=family,
        generations=generations,
        steps=steps,
        eval_batches=eval_batches,
        device=device,
        seed=seed,
    )
    console.print_json(data=report)


@app.command("write-plan")
def write_plan_cmd(
    out_path: Annotated[Path, typer.Argument(help="Output path for the JSON plan.")],
    stage: Annotated[str, typer.Option(help="scout or refine")] = "scout",
    device: Annotated[str, typer.Option()] = "cuda",
    seed: Annotated[int, typer.Option()] = 0,
    family: Annotated[
        list[str] | None,
        typer.Option("--family", help="Optional family id filter; repeatable."),
    ] = None,
) -> None:
    """Persist the recommended combo plan as JSON."""
    plan = build_combo_plan(stage=stage, device=device, seed=seed, family_ids=family)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    console.print(f"[bold green]Wrote combo plan:[/] {out_path}")


if __name__ == "__main__":
    app()
