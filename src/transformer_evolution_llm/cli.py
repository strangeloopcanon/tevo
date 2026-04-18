"""Command-line utilities for the transformer_evolution_llm package."""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - CLI orchestrates trusted local repo entrypoints.
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
import ujson as json
from rich.console import Console
from rich.table import Table

from . import get_version
from .api import (
    convert_checkpoints,
    export_frontier_seed,
    export_parameter_golf_workspace,
    export_train_recipe,
    load_spec,
    preflight_parameter_golf_config,
    prune_checkpoints,
    render_train_recipe,
    run_evolution,
    run_parameter_golf_benchmark,
    save_spec,
    transfer_parameter_golf_motif,
)
from .cache_builder import synthesize_cache
from .cuda_transfer import (
    AutoresearchFlavor,
    CudaTransferError,
    build_public_cuda_transfer_report,
    prepare_autoresearch_at_home_handoff,
    render_public_cuda_variants,
    resolve_autoresearch_source_repo,
    run_public_cuda_transfer_modal_benchmarks,
    train_recipe_target_for_flavor,
)
from .mlx_transfer import (
    audit_tevo_regions_from_paths,
    build_public_transfer_report,
    cost_conscious_modal_budget,
    detect_tevo_device,
    export_public_transfer_recipes,
    render_continuation_summary_markdown,
    render_public_transfer_variants,
    run_public_transfer_benchmarks,
    stage_public_transfer_continuation,
    stage_public_transfer_workspaces,
    summarize_continuation_results,
    write_winning_seed_diff,
)
from .parameter_golf_runtime import rescore_parameter_golf_checkpoint
from .train_recipe import TrainRecipeTarget, load_train_recipe, render_train_recipe_fragment

app = typer.Typer(help="Evolutionary search loop utilities")
console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _modal_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise typer.BadParameter(
            "Modal TEVO runs require --config to live inside this repo so it can be "
            "mounted into the Modal image."
        ) from exc


def _modal_run_slug(path: Path) -> str:
    slug = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "_"
        for ch in str(path.resolve().name or "mlx_transfer")
    )
    return slug.strip("_") or "mlx_transfer"


def _copy_required_artifact(source: Path, dest: Path, *, label: str) -> None:
    if not source.exists():
        raise typer.BadParameter(f"Missing {label} artifact at {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


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


@app.command("parameter-golf-benchmark")
def parameter_golf_benchmark_cmd(
    config: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    out: Annotated[Path, typer.Option()] = Path("runs/parameter_golf_benchmark.json"),
    checkpoint_dir: Annotated[Path, typer.Option()] = Path("runs/parameter_golf_checkpoints"),
    steps: Annotated[int | None, typer.Option(min=1)] = None,
    eval_batches: Annotated[int, typer.Option(min=1)] = 2,
    device: Annotated[str | None, typer.Option()] = None,
    max_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Train a single Parameter Golf spec and write a benchmark summary."""
    summary = run_parameter_golf_benchmark(
        config,
        out_path=out,
        checkpoint_dir=checkpoint_dir,
        steps=steps,
        eval_batches=eval_batches,
        device=device,
        max_tokens=max_tokens,
        seed=seed,
    )
    console.print(f"[bold green]Parameter Golf summary written:[/] {out}")
    console.print(
        f"val_bpb={summary['metrics'].get('val_bpb', float('nan')):.6f} "
        f"artifact_total_bytes={summary['metrics'].get('artifact_total_bytes', float('nan')):.0f}"
    )


@app.command("parameter-golf-rescore")
def parameter_golf_rescore_cmd(
    config: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    checkpoint: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    out: Annotated[Path, typer.Option()] = Path("runs/parameter_golf_rescore.json"),
    device: Annotated[str | None, typer.Option()] = None,
    val_batch_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    eval_protocol: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Re-score an existing Parameter Golf checkpoint without retraining it."""
    summary = rescore_parameter_golf_checkpoint(
        config,
        checkpoint,
        out_path=out,
        device=device,
        val_batch_tokens=val_batch_tokens,
        eval_protocol=eval_protocol,
    )
    console.print(f"[bold green]Parameter Golf rescore written:[/] {out}")
    console.print(
        f"val_bpb={summary['metrics'].get('val_bpb', float('nan')):.6f} "
        f"artifact_total_bytes={summary['metrics'].get('artifact_total_bytes', float('nan')):.0f}"
    )
    if "parameter_golf_error_message" in summary:
        console.print(f"[yellow]parameter_golf_error:[/] {summary['parameter_golf_error_message']}")


@app.command("parameter-golf-preflight")
def parameter_golf_preflight_cmd(
    config: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
) -> None:
    """Resolve Parameter Golf inputs and print size estimates without training."""
    report = preflight_parameter_golf_config(config)
    console.print_json(data=report)


@app.command("parameter-golf-export")
def parameter_golf_export_cmd(
    source: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    out: Annotated[Path, typer.Argument(help="Output directory for the exported workspace.")],
    candidate_id: Annotated[
        str | None,
        typer.Option(help="Optional frontier candidate id when exporting from frontier JSON."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(help="Export mode: official for compact submission lane, tevo for legacy."),
    ] = "official",
    official_train_py: Annotated[
        Path | None,
        typer.Option(
            help="Optional local path to parameter-golf/train_gpt.py for official exports."
        ),
    ] = None,
) -> None:
    """Export a TEVO spec into an official-style or legacy Parameter Golf workspace."""
    if mode not in {"official", "tevo"}:
        raise typer.BadParameter("mode must be 'official' or 'tevo'")
    metadata = export_parameter_golf_workspace(
        source,
        out,
        candidate_id=candidate_id,
        mode=cast(Literal["official", "tevo"], mode),
        official_train_py=official_train_py,
    )
    console.print(f"[bold green]Parameter Golf workspace written:[/] {out}")
    console.print(
        f"code_bytes={metadata['code_bytes']} "
        f"artifact_total_bytes_est={metadata['artifact_total_bytes_est']}"
    )


@app.command("parameter-golf-transfer-motif")
def parameter_golf_transfer_motif_cmd(
    source: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    target: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    out: Annotated[Path, typer.Argument(help="Output path for the transferred spec.")],
    include_context: Annotated[bool, typer.Option()] = False,
    include_structure: Annotated[bool, typer.Option()] = False,
) -> None:
    """Copy a discovered Parameter Golf motif into another seed family."""
    transferred = transfer_parameter_golf_motif(
        load_spec(source),
        load_spec(target),
        include_context=include_context,
        include_structure=include_structure,
    )
    save_spec(transferred, out)
    console.print(f"[bold green]Transferred motif spec written:[/] {out}")


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


@app.command("train-recipe-export")
def export_train_recipe_cmd(
    source: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            help="TEVO config path or frontier JSON path.",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(help="Output recipe file path (single export) or directory (multi-export)."),
    ] = None,
    candidate_id: Annotated[
        str | None,
        typer.Option(help="Optional frontier candidate id to export."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(min=1, help="How many compatible frontier recipes to export."),
    ] = 1,
    metric: Annotated[
        str,
        typer.Option(help="Metric used to shortlist compatible frontier entries."),
    ] = "ppl_code",
) -> None:
    """Export TrainRecipe YAML artifacts from a TEVO spec or frontier."""
    if out is None:
        if top_k > 1 or source.suffix == ".json":
            out = source.parent / f"{source.stem}_train_recipes"
        else:
            out = source.with_name(source.stem + ".train_recipe.yaml")
    written = export_train_recipe(
        source_path=source,
        out_path=out,
        candidate_id=candidate_id,
        top_k=top_k,
        metric=metric,
    )
    for path in written:
        console.print(f"[bold green]Train recipe written:[/] {path}")


@app.command("train-recipe-render")
def render_train_recipe_cmd(
    recipe: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Path to a TrainRecipe YAML/JSON."),
    ],
    backend: Annotated[
        TrainRecipeTarget,
        typer.Option(help="Which downstream train.py flavor to render for."),
    ],
    train_py: Annotated[
        Path | None,
        typer.Option(help="Existing downstream train.py to patch in-place (or via --out)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Optional output path for the rendered file or fragment."),
    ] = None,
) -> None:
    """Render a TrainRecipe into a downstream train.py or emit the TEVO-owned zones."""
    loaded_recipe = load_train_recipe(recipe)
    if train_py is None:
        fragment = render_train_recipe_fragment(loaded_recipe, backend)
        if out is None:
            typer.echo(fragment)
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fragment)
        console.print(f"[bold green]Train recipe fragment written:[/] {out}")
        return

    rendered_path = render_train_recipe(
        recipe_path=recipe,
        target=backend,
        train_py_path=train_py,
        out_path=out,
    )
    console.print(f"[bold green]Rendered train.py written:[/] {rendered_path}")


@app.command("mlx-transfer-prepare")
def mlx_transfer_prepare_cmd(
    mlx_repo: Annotated[
        Path,
        typer.Argument(
            ..., exists=True, readable=True, file_okay=False, help="autoresearch-mlx checkout."
        ),
    ],
    run_root: Annotated[
        Path,
        typer.Option(help="Artifact directory for the first public MLX transfer run."),
    ] = Path("runs/mlx_transfer"),
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            readable=True,
            help="TEVO config used for the bridge-safe discovery sweep.",
        ),
    ] = Path("configs/exp_train_recipe_bridge_owt_10m_v1.yaml"),
    frontier: Annotated[
        Path | None,
        typer.Option(
            help="Existing frontier JSON to reuse instead of launching a fresh TEVO sweep."
        ),
    ] = None,
    lineage: Annotated[
        Path | None,
        typer.Option(help="Optional lineage JSON to copy alongside a reused frontier."),
    ] = None,
    modal: Annotated[
        bool,
        typer.Option(
            help="Run the upstream TEVO seed benchmark and sweep on Modal instead of locally."
        ),
    ] = False,
    modal_gpu: Annotated[
        str,
        typer.Option(help="Modal GPU preset for the upstream TEVO discovery sweep."),
    ] = "A10G",
    modal_local_out_dir: Annotated[
        Path | None,
        typer.Option(help="Local cache directory for downloaded Modal TEVO artifacts."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="TEVO live-run device: auto, mps, cpu, or cuda."),
    ] = "auto",
    generations: Annotated[int, typer.Option(min=1)] = 8,
    steps: Annotated[int, typer.Option(min=1)] = 120,
    eval_batches: Annotated[int, typer.Option(min=1)] = 4,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Stage the first public TEVO -> autoresearch-mlx transfer run."""
    run_root.mkdir(parents=True, exist_ok=True)
    frontier_out = run_root / "frontier.json"
    lineage_out = run_root / "frontier_lineage.json"
    state_out = run_root / "frontier.state.json"
    tevo_mode = "reused_frontier" if frontier is not None else ("modal" if modal else "local")
    if tevo_mode == "local":
        prepared_device = detect_tevo_device(device)
    elif tevo_mode == "modal":
        prepared_device = f"modal:{modal_gpu}"
    else:
        prepared_device = "reused-frontier"
    modal_budget: dict[str, int] | None = None
    modal_artifacts_root: Path | None = None

    if frontier is None:
        seed_summary_path = run_root / "tevo_seed_summary.json"
        seed_history_path = run_root / "tevo_seed_history.json"
        if modal:
            modal_budget = cost_conscious_modal_budget(
                generations=generations,
                steps=steps,
                eval_batches=eval_batches,
            )
            if modal_budget != {
                "generations": int(generations),
                "steps": int(steps),
                "eval_batches": int(eval_batches),
            }:
                console.print(
                    "[yellow]Modal TEVO discovery budget capped for the first public pass:[/] "
                    f"{modal_budget}"
                )
            modal_artifacts_root = (modal_local_out_dir or (run_root / "modal")).resolve()
            modal_artifacts_root.mkdir(parents=True, exist_ok=True)
            modal_env = dict(os.environ, TEVO_MODAL_GPU=str(modal_gpu))
            modal_config = _modal_repo_path(config)
            run_slug = _modal_run_slug(run_root)
            bench_run_id = f"{run_slug}_seed_bench"
            benchmark_cmd = [
                "modal",
                "run",
                "scripts/modal_run_benchmark.py",
                "--config-path",
                modal_config,
                "--steps",
                str(modal_budget["steps"]),
                "--eval-batches",
                str(modal_budget["eval_batches"]),
                "--seed",
                str(int(seed)),
                "--run-id",
                bench_run_id,
                "--download",
                "--local-out-dir",
                str(modal_artifacts_root),
            ]
            subprocess.run(  # noqa: S603  # nosec B603
                benchmark_cmd,
                check=True,
                cwd=REPO_ROOT,
                env=modal_env,
            )
            _copy_required_artifact(
                modal_artifacts_root / bench_run_id / "summary.json",
                seed_summary_path,
                label="Modal TEVO seed summary",
            )
            _copy_required_artifact(
                modal_artifacts_root / bench_run_id / "history.json",
                seed_history_path,
                label="Modal TEVO seed history",
            )

            sweep_run_id = f"{run_slug}_sweep"
            live_cmd = [
                "modal",
                "run",
                "scripts/modal_run_live.py",
                "--config-path",
                modal_config,
                "--generations",
                str(modal_budget["generations"]),
                "--steps",
                str(modal_budget["steps"]),
                "--eval-batches",
                str(modal_budget["eval_batches"]),
                "--seed",
                str(int(seed)),
                "--run-id",
                sweep_run_id,
                "--download",
                "--local-out-dir",
                str(modal_artifacts_root),
                "--lineage",
            ]
            subprocess.run(  # noqa: S603  # nosec B603
                live_cmd,
                check=True,
                cwd=REPO_ROOT,
                env=modal_env,
            )
            _copy_required_artifact(
                modal_artifacts_root / sweep_run_id / "frontier.json",
                frontier_out,
                label="Modal TEVO frontier",
            )
            _copy_required_artifact(
                modal_artifacts_root / sweep_run_id / "frontier.state.json",
                state_out,
                label="Modal TEVO state",
            )
            lineage_source = modal_artifacts_root / sweep_run_id / "frontier_lineage.json"
            if not lineage_source.exists():
                lineage_source = modal_artifacts_root / sweep_run_id / "lineage.json"
            if lineage_source.exists():
                _copy_required_artifact(
                    lineage_source,
                    lineage_out,
                    label="Modal TEVO lineage",
                )
        else:
            benchmark_cmd = [
                "python3",
                "scripts/run_benchmark.py",
                str(config),
                "--steps",
                str(int(steps)),
                "--eval-batches",
                str(int(eval_batches)),
                "--device",
                prepared_device,
                "--out",
                str(seed_summary_path),
                "--history-out",
                str(seed_history_path),
            ]
            subprocess.run(benchmark_cmd, check=True, cwd=REPO_ROOT)  # noqa: S603  # nosec B603

            live_cmd = [
                "python3",
                "scripts/run_live.py",
                str(config),
                "--generations",
                str(int(generations)),
                "--steps",
                str(int(steps)),
                "--eval-batches",
                str(int(eval_batches)),
                "--device",
                prepared_device,
                "--seed",
                str(int(seed)),
                "--out",
                str(frontier_out),
                "--checkpoint-dir",
                str(run_root / "checkpoints"),
                "--lineage-out",
                str(lineage_out),
                "--state-out",
                str(state_out),
            ]
            subprocess.run(live_cmd, check=True, cwd=REPO_ROOT)  # noqa: S603  # nosec B603
    else:
        if frontier.resolve() != frontier_out.resolve():
            shutil.copy2(frontier, frontier_out)
        else:
            frontier_out = frontier
        if lineage is not None:
            if lineage.resolve() != lineage_out.resolve():
                shutil.copy2(lineage, lineage_out)
            else:
                lineage_out = lineage

    selected, selection_manifest_path = export_public_transfer_recipes(
        frontier_out,
        run_root / "train_recipes",
    )
    rendered_variants, render_manifest_path = render_public_transfer_variants(
        mlx_repo,
        selection_manifest_path,
        run_root / "rendered_train_py",
    )
    arm_manifest_path = stage_public_transfer_workspaces(
        mlx_repo=mlx_repo,
        rendered_variants=rendered_variants,
        out_dir=run_root / "mlx_arms",
    )
    prepare_manifest = {
        "config": str(config.resolve()),
        "device": prepared_device,
        "tevo_mode": tevo_mode,
        "modal_gpu": (str(modal_gpu) if tevo_mode == "modal" else None),
        "modal_budget": modal_budget,
        "modal_artifacts_root": (
            str(modal_artifacts_root) if modal_artifacts_root is not None else None
        ),
        "frontier": str(frontier_out.resolve()),
        "lineage": str(lineage_out.resolve()) if lineage_out.exists() else None,
        "state": str(state_out.resolve()) if state_out.exists() else None,
        "selection_manifest": str(selection_manifest_path.resolve()),
        "render_manifest": str(render_manifest_path.resolve()),
        "arm_manifest": str(arm_manifest_path.resolve()),
        "selected_labels": [item.label for item in selected],
        "selected_candidates": [item.candidate_id for item in selected],
    }
    prepare_manifest_path = run_root / "transfer_prepare_manifest.json"
    prepare_manifest_path.write_text(json.dumps(prepare_manifest, indent=2))
    console.print(f"[bold green]Prepared MLX transfer run:[/] {run_root}")
    console.print(f"Frontier: {frontier_out}")
    console.print(f"Arm manifest: {arm_manifest_path}")


@app.command("mlx-transfer-benchmark")
def mlx_transfer_benchmark_cmd(
    run_root: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            file_okay=False,
            help="Prepared transfer run directory.",
        ),
    ],
    repeat: Annotated[int, typer.Option(min=1)] = 3,
    timeout_minutes: Annotated[int, typer.Option(min=1)] = 10,
    command: Annotated[
        str,
        typer.Option(help="Benchmark command to run inside each arm repo."),
    ] = "uv run train.py",
) -> None:
    """Run the baseline + seeded autoresearch-mlx benchmark arms and summarize medians."""
    arm_manifest_path = run_root / "mlx_arms" / "arm_manifest.json"
    if not arm_manifest_path.exists():
        raise typer.BadParameter(f"Missing arm manifest: {arm_manifest_path}")
    _, _, summary_path, markdown_path = run_public_transfer_benchmarks(
        arm_manifest_path,
        out_dir=run_root / "mlx_results",
        repeat=repeat,
        timeout_seconds=int(timeout_minutes) * 60,
        command=command,
    )
    summary_payload = json.loads(summary_path.read_text())
    winner = summary_payload.get("winner")
    if winner:
        diff_path = write_winning_seed_diff(
            arm_manifest_path,
            summary_path,
            run_root / "mlx_results" / "winning_seed.diff",
        )
        continuation_manifest = stage_public_transfer_continuation(
            arm_manifest_path,
            summary_path,
            run_root / "continuation",
            repo_root=REPO_ROOT,
        )
        console.print(f"[bold green]Winning seed diff:[/] {diff_path}")
        console.print(f"[bold green]Continuation workspace:[/] {continuation_manifest}")
    else:
        console.print(
            "[yellow]No winning seeded arm yet; continuation workspace was not staged.[/]"
        )

    report_payload, report_markdown = build_public_transfer_report(run_root)
    report_path = run_root / "public_report.md"
    report_json_path = run_root / "public_report.json"
    report_path.write_text(report_markdown)
    report_json_path.write_text(json.dumps(report_payload, indent=2))
    console.print(f"[bold green]Benchmark summary written:[/] {markdown_path}")
    console.print(f"[bold green]Public report written:[/] {report_path}")


@app.command("mlx-transfer-audit")
def mlx_transfer_audit_cmd(
    seed_train_py: Annotated[
        Path,
        typer.Argument(
            ..., exists=True, readable=True, help="Seed snapshot with frozen TEVO-owned regions."
        ),
    ],
    candidate_train_py: Annotated[
        Path,
        typer.Argument(
            ..., exists=True, readable=True, help="Current train.py to compare against the seed."
        ),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional JSON output path.")] = None,
) -> None:
    """Audit whether TEVO-owned marker regions changed between two train.py files."""
    payload = audit_tevo_regions_from_paths(seed_train_py, candidate_train_py, out)
    console.print_json(json=json.dumps(payload, indent=2))


@app.command("mlx-transfer-continuation-summary")
def mlx_transfer_continuation_summary_cmd(
    results_tsv: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            help="Continuation results.tsv from the seeded MLX workspace.",
        ),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional Markdown output path.")] = None,
) -> None:
    """Summarize the seeded continuation best-so-far trajectory."""
    payload = summarize_continuation_results(results_tsv)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_continuation_summary_markdown(payload))
        console.print(f"[bold green]Continuation summary written:[/] {out}")
    console.print_json(json=json.dumps(payload, indent=2))


@app.command("mlx-transfer-report")
def mlx_transfer_report_cmd(
    run_root: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            file_okay=False,
            help="Prepared transfer run directory.",
        ),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional Markdown output path.")] = None,
) -> None:
    """Build the public-facing report from TEVO, MLX benchmark, and continuation artifacts."""
    payload, markdown = build_public_transfer_report(run_root)
    if out is None:
        out = run_root / "public_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    payload_path = out.with_suffix(".json")
    payload_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[bold green]Public report written:[/] {out}")


@app.command("cuda-transfer-prepare")
def cuda_transfer_prepare_cmd(
    run_root: Annotated[
        Path,
        typer.Option(help="Artifact directory for the first public CUDA transfer run."),
    ] = Path("runs/cuda_transfer"),
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            readable=True,
            help="TEVO config used for the bridge-safe discovery sweep.",
        ),
    ] = Path("configs/exp_train_recipe_bridge_owt_10m_v1.yaml"),
    autoresearch_repo: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            readable=True,
            file_okay=False,
            help="Optional local autoresearch checkout to render against.",
        ),
    ] = None,
    autoresearch_flavor: Annotated[
        AutoresearchFlavor,
        typer.Option(help="Named CUDA train.py flavor to target when resolving a repo preset."),
    ] = AutoresearchFlavor.UPSTREAM,
    autoresearch_repo_url: Annotated[
        str | None,
        typer.Option(
            help="Optional Git URL override for the CUDA autoresearch repo Modal will benchmark."
        ),
    ] = None,
    autoresearch_ref: Annotated[
        str,
        typer.Option(help="Git ref or commit for the CUDA autoresearch repo."),
    ] = "master",
    frontier: Annotated[
        Path | None,
        typer.Option(
            help="Existing frontier JSON to reuse instead of launching a fresh Modal TEVO sweep."
        ),
    ] = None,
    lineage: Annotated[
        Path | None,
        typer.Option(help="Optional lineage JSON to copy alongside a reused frontier."),
    ] = None,
    modal_gpu: Annotated[
        str,
        typer.Option(help="Modal GPU preset for the TEVO discovery sweep."),
    ] = "A10G",
    modal_local_out_dir: Annotated[
        Path | None,
        typer.Option(help="Local cache directory for downloaded Modal TEVO artifacts."),
    ] = None,
    generations: Annotated[int, typer.Option(min=1)] = 8,
    steps: Annotated[int, typer.Option(min=1)] = 120,
    eval_batches: Annotated[int, typer.Option(min=1)] = 4,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Stage the first public TEVO -> autoresearch CUDA transfer run."""
    run_root.mkdir(parents=True, exist_ok=True)
    source_repo, repo_metadata = resolve_autoresearch_source_repo(
        out_dir=run_root,
        flavor=autoresearch_flavor,
        local_repo=autoresearch_repo,
        repo_url=autoresearch_repo_url,
        repo_ref=autoresearch_ref,
    )
    frontier_out = run_root / "frontier.json"
    lineage_out = run_root / "frontier_lineage.json"
    state_out = run_root / "frontier.state.json"
    tevo_mode = "reused_frontier" if frontier is not None else "modal"
    modal_budget: dict[str, int] | None = None
    modal_artifacts_root: Path | None = None

    if frontier is None:
        seed_summary_path = run_root / "tevo_seed_summary.json"
        seed_history_path = run_root / "tevo_seed_history.json"
        modal_budget = cost_conscious_modal_budget(
            generations=generations,
            steps=steps,
            eval_batches=eval_batches,
        )
        if modal_budget != {
            "generations": int(generations),
            "steps": int(steps),
            "eval_batches": int(eval_batches),
        }:
            console.print(
                "[yellow]Modal TEVO discovery budget capped for the first public CUDA pass:[/] "
                f"{modal_budget}"
            )
        modal_artifacts_root = (modal_local_out_dir or (run_root / "modal" / "tevo")).resolve()
        modal_artifacts_root.mkdir(parents=True, exist_ok=True)
        modal_env = dict(os.environ, TEVO_MODAL_GPU=str(modal_gpu))
        modal_config = _modal_repo_path(config)
        run_slug = _modal_run_slug(run_root)
        bench_run_id = f"{run_slug}_seed_bench"
        benchmark_cmd = [
            "modal",
            "run",
            "scripts/modal_run_benchmark.py",
            "--config-path",
            modal_config,
            "--steps",
            str(modal_budget["steps"]),
            "--eval-batches",
            str(modal_budget["eval_batches"]),
            "--seed",
            str(int(seed)),
            "--run-id",
            bench_run_id,
            "--download",
            "--local-out-dir",
            str(modal_artifacts_root),
        ]
        subprocess.run(  # noqa: S603  # nosec B603
            benchmark_cmd,
            check=True,
            cwd=REPO_ROOT,
            env=modal_env,
        )
        _copy_required_artifact(
            modal_artifacts_root / bench_run_id / "summary.json",
            seed_summary_path,
            label="Modal TEVO seed summary",
        )
        _copy_required_artifact(
            modal_artifacts_root / bench_run_id / "history.json",
            seed_history_path,
            label="Modal TEVO seed history",
        )

        sweep_run_id = f"{run_slug}_sweep"
        live_cmd = [
            "modal",
            "run",
            "scripts/modal_run_live.py",
            "--config-path",
            modal_config,
            "--generations",
            str(modal_budget["generations"]),
            "--steps",
            str(modal_budget["steps"]),
            "--eval-batches",
            str(modal_budget["eval_batches"]),
            "--seed",
            str(int(seed)),
            "--run-id",
            sweep_run_id,
            "--download",
            "--local-out-dir",
            str(modal_artifacts_root),
            "--lineage",
        ]
        subprocess.run(  # noqa: S603  # nosec B603
            live_cmd,
            check=True,
            cwd=REPO_ROOT,
            env=modal_env,
        )
        _copy_required_artifact(
            modal_artifacts_root / sweep_run_id / "frontier.json",
            frontier_out,
            label="Modal TEVO frontier",
        )
        _copy_required_artifact(
            modal_artifacts_root / sweep_run_id / "frontier.state.json",
            state_out,
            label="Modal TEVO state",
        )
        lineage_source = modal_artifacts_root / sweep_run_id / "frontier_lineage.json"
        if not lineage_source.exists():
            lineage_source = modal_artifacts_root / sweep_run_id / "lineage.json"
        if lineage_source.exists():
            _copy_required_artifact(
                lineage_source,
                lineage_out,
                label="Modal TEVO lineage",
            )
    else:
        if frontier.resolve() != frontier_out.resolve():
            shutil.copy2(frontier, frontier_out)
        else:
            frontier_out = frontier
        if lineage is not None:
            if lineage.resolve() != lineage_out.resolve():
                shutil.copy2(lineage, lineage_out)
            else:
                lineage_out = lineage

    selected, selection_manifest_path = export_public_transfer_recipes(
        frontier_out,
        run_root / "train_recipes",
    )
    rendered_variants, render_manifest_path = render_public_cuda_variants(
        source_repo,
        selection_manifest_path,
        run_root / "rendered_train_py",
        target=train_recipe_target_for_flavor(autoresearch_flavor),
    )
    arm_manifest_path = stage_public_transfer_workspaces(
        mlx_repo=source_repo,
        rendered_variants=rendered_variants,
        out_dir=run_root / "cuda_arms",
    )
    prepare_manifest = {
        "config": str(config.resolve()),
        "device": f"modal:{modal_gpu}",
        "tevo_mode": tevo_mode,
        "modal_gpu": str(modal_gpu),
        "modal_budget": modal_budget,
        "modal_artifacts_root": (
            str(modal_artifacts_root) if modal_artifacts_root is not None else None
        ),
        "frontier": str(frontier_out.resolve()),
        "lineage": str(lineage_out.resolve()) if lineage_out.exists() else None,
        "state": str(state_out.resolve()) if state_out.exists() else None,
        "selection_manifest": str(selection_manifest_path.resolve()),
        "render_manifest": str(render_manifest_path.resolve()),
        "arm_manifest": str(arm_manifest_path.resolve()),
        "selected_labels": [item.label for item in selected],
        "selected_candidates": [item.candidate_id for item in selected],
        "autoresearch_source_repo": str(source_repo.resolve()),
        "autoresearch_flavor": repo_metadata.get("autoresearch_flavor"),
        "autoresearch_repo_url": repo_metadata.get("repo_url"),
        "autoresearch_repo_ref": repo_metadata.get("repo_ref"),
        "autoresearch_source_kind": repo_metadata.get("source_kind"),
    }
    prepare_manifest_path = run_root / "transfer_prepare_manifest.json"
    prepare_manifest_path.write_text(json.dumps(prepare_manifest, indent=2))
    console.print(f"[bold green]Prepared CUDA transfer run:[/] {run_root}")
    console.print(f"Frontier: {frontier_out}")
    console.print(f"CUDA source repo: {source_repo}")
    console.print(f"Arm manifest: {arm_manifest_path}")


@app.command("autoresearch-at-home-handoff")
def autoresearch_at_home_handoff_cmd(
    frontier: Annotated[
        Path,
        typer.Option(
            ...,
            exists=True,
            readable=True,
            help="Frontier JSON containing the chosen TEVO candidate.",
        ),
    ],
    candidate_id: Annotated[
        str,
        typer.Option(..., help="Bridge-compatible frontier candidate id to hand off."),
    ],
    run_root: Annotated[
        Path,
        typer.Option(help="Artifact directory for the handoff bundle."),
    ] = Path("runs/at_home_handoff"),
    lineage: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            readable=True,
            help="Optional lineage JSON to reference in the handoff summary.",
        ),
    ] = None,
    autoresearch_repo: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            readable=True,
            file_okay=False,
            help="Optional local autoresearch@home checkout to stage against.",
        ),
    ] = None,
    autoresearch_flavor: Annotated[
        AutoresearchFlavor,
        typer.Option(help="Named autoresearch-family preset to stage against."),
    ] = AutoresearchFlavor.AT_HOME,
    autoresearch_repo_url: Annotated[
        str | None,
        typer.Option(help="Optional Git URL override for the staged handoff repo."),
    ] = None,
    autoresearch_ref: Annotated[
        str,
        typer.Option(help="Git ref or commit for the staged handoff repo."),
    ] = "master",
) -> None:
    """Export one TEVO candidate into a runnable autoresearch@home workspace."""
    try:
        _, manifest_path, summary_path = prepare_autoresearch_at_home_handoff(
            frontier_path=frontier,
            candidate_id=candidate_id,
            out_dir=run_root,
            local_repo=autoresearch_repo,
            repo_url=autoresearch_repo_url,
            repo_ref=autoresearch_ref,
            flavor=autoresearch_flavor,
            lineage_path=lineage,
        )
    except CudaTransferError as exc:
        raise typer.BadParameter(str(exc)) from exc

    manifest = json.loads(manifest_path.read_text())
    console.print(f"[bold green]Prepared autoresearch@home handoff:[/] {run_root}")
    console.print(f"Candidate: {candidate_id}")
    console.print(f"Staged repo: {manifest['staged_repo']}")
    console.print(f"Manifest: {manifest_path}")
    console.print(f"Summary: {summary_path}")


@app.command("cuda-transfer-benchmark")
def cuda_transfer_benchmark_cmd(
    run_root: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            file_okay=False,
            help="Prepared CUDA transfer run directory.",
        ),
    ],
    repeat: Annotated[int, typer.Option(min=1)] = 3,
    timeout_minutes: Annotated[int, typer.Option(min=1)] = 10,
    modal_gpu: Annotated[
        str,
        typer.Option(help="Modal GPU preset for CUDA autoresearch benchmarking."),
    ] = "A100-80GB",
    modal_local_out_dir: Annotated[
        Path | None,
        typer.Option(help="Local cache directory for downloaded Modal CUDA benchmark artifacts."),
    ] = None,
) -> None:
    """Run the baseline + seeded CUDA autoresearch benchmark arms through Modal."""
    arm_manifest_path = run_root / "cuda_arms" / "arm_manifest.json"
    if not arm_manifest_path.exists():
        raise typer.BadParameter(f"Missing arm manifest: {arm_manifest_path}")
    prepare_manifest_path = run_root / "transfer_prepare_manifest.json"
    if not prepare_manifest_path.exists():
        raise typer.BadParameter(f"Missing prepare manifest: {prepare_manifest_path}")
    prepare_manifest = json.loads(prepare_manifest_path.read_text())
    repo_url = str(prepare_manifest.get("autoresearch_repo_url") or "").strip()
    repo_ref = str(prepare_manifest.get("autoresearch_repo_ref") or "master").strip()
    if not repo_url:
        raise typer.BadParameter(
            "The prepared run is missing autoresearch_repo_url, "
            "so Modal cannot clone the CUDA repo."
        )

    _, _, summary_path, markdown_path = run_public_cuda_transfer_modal_benchmarks(
        arm_manifest_path,
        repo_url=repo_url,
        repo_ref=repo_ref,
        out_dir=run_root / "cuda_results",
        repo_root=REPO_ROOT,
        modal_gpu=modal_gpu,
        repeat=repeat,
        timeout_minutes=timeout_minutes,
        modal_local_out_dir=modal_local_out_dir or (run_root / "modal" / "autoresearch"),
    )
    summary_payload = json.loads(summary_path.read_text())
    winner = summary_payload.get("winner")
    if winner:
        diff_path = write_winning_seed_diff(
            arm_manifest_path,
            summary_path,
            run_root / "cuda_results" / "winning_seed.diff",
        )
        console.print(f"[bold green]Winning seed diff:[/] {diff_path}")
    else:
        console.print("[yellow]No winning seeded CUDA arm yet.[/]")

    report_payload, report_markdown = build_public_cuda_transfer_report(run_root)
    report_path = run_root / "cuda_public_report.md"
    report_json_path = run_root / "cuda_public_report.json"
    report_path.write_text(report_markdown)
    report_json_path.write_text(json.dumps(report_payload, indent=2))
    console.print(f"[bold green]Benchmark summary written:[/] {markdown_path}")
    console.print(f"[bold green]CUDA public report written:[/] {report_path}")


@app.command("cuda-transfer-report")
def cuda_transfer_report_cmd(
    run_root: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            file_okay=False,
            help="Prepared CUDA transfer run directory.",
        ),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional Markdown output path.")] = None,
) -> None:
    """Build the public-facing CUDA transfer report from TEVO + Modal artifacts."""
    payload, markdown = build_public_cuda_transfer_report(run_root)
    if out is None:
        out = run_root / "cuda_public_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    payload_path = out.with_suffix(".json")
    payload_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[bold green]CUDA public report written:[/] {out}")


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
