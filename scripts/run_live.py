"""Utility to run live evolutionary sweeps with custom trainer settings."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

import typer

from transformer_evolution_llm.api import load_spec
from transformer_evolution_llm.data import DataModule
from transformer_evolution_llm.orchestrator import EvolutionRunner, default_objectives
from transformer_evolution_llm.trainer import FullWeightTrainer

app = typer.Typer(help="Launch live (full-weight) evolutionary runs with custom trainer settings.")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _detect_attention_impl(device: str) -> str:
    """Best-effort attention kernel description for audit logs."""
    try:
        import torch

        if torch.backends.cuda.is_built() and device.startswith("cuda"):
            try:
                from torch.backends.cuda import sdp_kernel

                if hasattr(sdp_kernel, "is_flash_sdp_available") and sdp_kernel.is_flash_sdp_available():
                    return "flash"
                if hasattr(sdp_kernel, "is_mem_efficient_sdp_available") and sdp_kernel.is_mem_efficient_sdp_available():
                    return "sdpa"
                return "cuda_math"
            except Exception:
                return "cuda_sdpa"
        if device.startswith("mps"):
            return "mps_mha"
    except Exception:
        pass
    return "vanilla_mha"


def _prune_checkpoints_to_frontier(frontier: Path, checkpoint_dir: Path) -> None:
    """Delete checkpoint files not referenced by the frontier JSON.

    Keeps only the checkpoint paths present in the saved frontier entries.
    """
    try:
        entries = json.loads(frontier.read_text())
    except Exception:
        return
    if not isinstance(entries, list):
        return
    keep: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ckpt = entry.get("checkpoint")
        if not ckpt:
            continue
        try:
            keep.add(Path(str(ckpt)).resolve())
        except Exception:
            continue
    removed = 0
    for path in checkpoint_dir.glob("*.pt"):
        if path.resolve() in keep:
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    if removed:
        typer.echo(f"Pruned {removed} non-frontier checkpoints from {checkpoint_dir}")


@app.command()
def main(
    config: Path = typer.Argument(..., exists=True, readable=True, help="YAML/JSON DSL config"),
    resume_from: Path | None = typer.Option(None, help="Path to saved runner state JSON to resume from."),
    state_out: Path | None = typer.Option(None, help="Optional path to write runner state after completion."),
    generations: int = typer.Option(18, min=1),
    steps: int = typer.Option(120, min=1, help="Gradient steps per candidate."),
    eval_batches: int = typer.Option(4, min=1),
    device: str = typer.Option("mps", help="Device for training (mps/cpu/cuda)."),
    out: Path = typer.Option(Path("runs/frontier_live.json")),
    checkpoint_dir: Path = typer.Option(Path("runs/checkpoints_live")),
    seed: int = typer.Option(0),
    cleanup_old_checkpoints: bool = typer.Option(
        True,
        help="After the run, delete other checkpoint directories (names starting with 'checkpoints') in the same parent folder.",
    ),
    prune_checkpoints_to_frontier: bool = typer.Option(
        False,
        help="After the run, keep only frontier checkpoint files inside checkpoint_dir.",
    ),
    lineage_out: Path | None = typer.Option(
        None,
        help="Optional path to write a full lineage JSON (all candidates, parent links).",
    ),
    score_weight_ppl: float = typer.Option(1.0, help="Weight for ppl_code/ppl_math in parent selection."),
    score_weight_throughput: float = typer.Option(1.0, help="Weight for throughput in parent selection."),
    score_weight_long_recall: float = typer.Option(1.0, help="Weight for long_recall in parent selection."),
    score_weight_ram: float = typer.Option(1.0, help="Weight for RAM in parent selection."),
    score_weight_layers: float = typer.Option(1.0, help="Weight for layer count in parent selection."),
    score_weight_moe: float = typer.Option(1.0, help="Weight for MoE block count in parent selection."),
    score_weight_novelty: float = typer.Option(1.0, help="Weight for novelty in parent selection."),
    score_weight_instability: float = typer.Option(1.0, help="Weight for instability (minimize) in parent selection."),
    mutation_weight: list[str] = typer.Option(
        None,
        help="Optional mutation weights as name=weight (repeatable). Example: --mutation-weight dense_to_moe=3.0",
    ),
    mutation_steps: int = typer.Option(
        1,
        min=1,
        help="Number of mutations to apply per child (chained sequentially).",
    ),
    parent_selection: str | None = typer.Option(
        None,
        help="Parent selection strategy (overrides config if set): weighted | pareto_uniform | lexicase | map_elites",
    ),
    score_weight_prior: float = typer.Option(
        0.0, help="Weight for prior_distance (minimize). 0 disables its effect."
    ),
) -> None:
    """Run an evolutionary sweep with the live trainer."""
    # Avoid tokenizers forking warnings when we later call subprocess (e.g., git).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    commit = _git_commit()
    if resume_from:
        runner = EvolutionRunner.load_state(resume_from, mode="live")
        spec = runner.base_spec
        # Resume implies deterministic continuation; ignore CLI seed.
        seed = seed  # keep for manifest only
    else:
        spec = load_spec(config)
        runner = EvolutionRunner(
            spec,
            spec.evolution,
            mode="live",
            seed=seed,
            score_weight_overrides=None,
        )
    # Only override runner.score_weights when the user explicitly changes one of
    # the weights from its default. This keeps configs that define custom
    # objectives (e.g., passkey_loss) working out of the box.
    defaults = {
        "ppl": 1.0,
        "throughput": 1.0,
        "long_recall": 1.0,
        "ram": 1.0,
        "layers": 1.0,
        "moe_blocks": 1.0,
        "novelty": 1.0,
        "instability": 1.0,
        "prior_distance": 0.0,
    }
    override_requested = any(
        abs(val - defaults[key]) > 1e-12
        for key, val in (
            ("ppl", score_weight_ppl),
            ("throughput", score_weight_throughput),
            ("long_recall", score_weight_long_recall),
            ("ram", score_weight_ram),
            ("layers", score_weight_layers),
            ("moe_blocks", score_weight_moe),
            ("novelty", score_weight_novelty),
            ("instability", score_weight_instability),
            ("prior_distance", score_weight_prior),
        )
    )
    if override_requested:
        direction_fallback = dict(default_objectives())
        direction_fallback["prior_distance"] = "min"

        def _signed(metric: str, weight: float) -> float:
            direction = runner.objective_dir.get(metric) or direction_fallback.get(metric, "max")
            sign = 1.0 if direction == "max" else -1.0
            return sign * float(weight)

        score_weights = dict(runner.score_weights)
        score_weights.update(
            {
                "ppl_code": _signed("ppl_code", score_weight_ppl),
                "ppl_math": _signed("ppl_math", score_weight_ppl),
                "throughput": _signed("throughput", score_weight_throughput),
                "long_recall": _signed("long_recall", score_weight_long_recall),
                "ram": _signed("ram", score_weight_ram),
                "layers": _signed("layers", score_weight_layers),
                "moe_blocks": _signed("moe_blocks", score_weight_moe),
                "novelty": _signed("novelty", score_weight_novelty),
                "instability": _signed("instability", score_weight_instability),
            }
        )
        if score_weight_prior != 0.0:
            score_weights["prior_distance"] = _signed("prior_distance", score_weight_prior)
        runner.score_weights = score_weights
    # Optional mutation mix override: list of "name=weight"
    if mutation_weight:
        weights: dict[str, float] = {}
        for item in mutation_weight:
            if "=" not in item:
                continue
            name, val = item.split("=", 1)
            try:
                weights[name] = float(val)
            except ValueError:
                continue
        if weights:
            runner.mutation_weights = weights
    runner.mutation_steps = mutation_steps
    # Override selection strategy from CLI if provided
    if parent_selection is not None:
        runner.cfg.parent_selection = parent_selection
    train_cfg = spec.train
    runner.trainer = FullWeightTrainer(
        steps=steps,
        eval_batches=eval_batches,
        checkpoint_dir=checkpoint_dir,
        device=device,
        entropy_threshold=train_cfg.entropy_threshold,
        entropy_patience=train_cfg.entropy_patience,
        instability_threshold=train_cfg.instability_threshold,
        no_improve_patience=train_cfg.no_improve_patience,
        improvement_tolerance=train_cfg.improvement_tolerance,
    )
    runner.data_module = DataModule(spec.data, seed=int(getattr(spec.train, "seed", 0) or 0))
    # Log device/capabilities (best effort)
    try:
        import torch

        dev = device
        msg = f"[runner] device={dev}"
        impl = _detect_attention_impl(dev)
        msg += f", attention_impl={impl}"
        typer.echo(msg)
    except Exception:
        pass
    runner.checkpoint_dir = checkpoint_dir
    run_error: str | None = None
    try:
        runner.run(generations=generations)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        typer.echo(f"[runner] run crashed: {run_error}")
        typer.echo(traceback.format_exc())
    try:
        runner.save_frontier(out)
    except Exception as exc:
        typer.echo(f"[runner] failed to write frontier: {exc}")
    if state_out is None:
        state_out = out.with_name(out.stem + ".state.json")
    try:
        runner.save_state(state_out)
    except Exception:
        pass
    if lineage_out is None:
        # default next to frontier
        lineage_out = out.with_name(out.stem + "_lineage.json")
    try:
        runner.save_lineage(lineage_out)
    except Exception:
        pass
    if run_error is not None:
        try:
            out.with_name(out.stem + ".error.txt").write_text(run_error + "\n")
        except Exception:
            pass
    manifest_path = out.with_name(out.stem + ".manifest.json")
    manifest = {
        "config": str(config.resolve()),
        "generations": generations,
        "steps": steps,
        "eval_batches": eval_batches,
        "device": device,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "seed": seed,
        "parent_selection": parent_selection,
        "score_weights": {
            "ppl": score_weight_ppl,
            "throughput": score_weight_throughput,
            "long_recall": score_weight_long_recall,
            "ram": score_weight_ram,
            "layers": score_weight_layers,
            "moe_blocks": score_weight_moe,
            "novelty": score_weight_novelty,
            "instability": score_weight_instability,
            "prior_distance": score_weight_prior,
        },
        "attention_impl": _detect_attention_impl(device),
        "cleanup_old_checkpoints": cleanup_old_checkpoints,
        "frontier": str(out.resolve()),
        "lineage": str(lineage_out.resolve()),
    }
    if commit:
        manifest["git_commit"] = commit
    manifest_path.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"Frontier written to {out}")
    if prune_checkpoints_to_frontier:
        _prune_checkpoints_to_frontier(out, checkpoint_dir)
    if cleanup_old_checkpoints:
        _cleanup_old_checkpoint_roots(checkpoint_dir)
    if run_error is not None:
        raise RuntimeError(run_error)


def _cleanup_old_checkpoint_roots(current_root: Path) -> None:
    parent = current_root.parent
    if not parent.exists():
        return
    removed: list[Path] = []
    for sibling in parent.iterdir():
        if sibling == current_root or not sibling.is_dir():
            continue
        if not sibling.name.startswith("checkpoints"):
            continue
        shutil.rmtree(sibling, ignore_errors=True)
        removed.append(sibling)
    if removed:
        formatted = ", ".join(str(path) for path in removed)
        typer.echo(f"Removed old checkpoint dirs: {formatted}")


if __name__ == "__main__":
    app()
