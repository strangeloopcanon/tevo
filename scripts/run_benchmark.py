"""Run a fixed-architecture benchmark with optional speedrun logging."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import typer

from transformer_evolution_llm.api import load_spec
from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.data import DataModule
from transformer_evolution_llm.trainer import FullWeightTrainer

app = typer.Typer(help="Train a single config and log benchmark metrics.")


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


@app.command()
def main(
    config: Path = typer.Argument(..., exists=True, readable=True, help="YAML/JSON DSL config"),
    steps: int = typer.Option(240, min=1),
    eval_batches: int = typer.Option(4, min=1),
    device: str = typer.Option("cpu"),
    out: Path = typer.Option(Path("runs/benchmark.json")),
    history_out: Path | None = typer.Option(None, help="Optional JSON path for eval history."),
    checkpoint_dir: Path = typer.Option(Path("runs/bench_checkpoints")),
    seed: int | None = typer.Option(None, help="Override train seed."),
    max_tokens: int | None = typer.Option(None, help="Override token budget for data iterator."),
) -> None:
    spec = load_spec(config)
    if seed is not None:
        spec.train.seed = int(seed)
    seed_val = int(getattr(spec.train, "seed", 0) or 0)

    data_module = DataModule(spec.data, seed=seed_val)
    token_budget = int(max_tokens) if max_tokens is not None else getattr(spec.train, "max_tokens", None)
    batch_iter = data_module.batches(max_tokens=token_budget)

    history: list[dict[str, float]] = []

    def record(step: int, loss: float, tokens_seen: int) -> None:
        eval_ppl = float("inf")
        if math.isfinite(loss) and loss < 50:
            eval_ppl = float(math.exp(loss))
        history.append(
            {
                "step": float(step),
                "eval_loss": float(loss),
                "eval_ppl": eval_ppl,
                "tokens_seen": float(tokens_seen),
            }
        )

    trainer = FullWeightTrainer(
        steps=steps,
        eval_batches=eval_batches,
        checkpoint_dir=checkpoint_dir,
        device=device,
        entropy_threshold=spec.train.entropy_threshold,
        entropy_patience=spec.train.entropy_patience,
        instability_threshold=spec.train.instability_threshold,
        no_improve_patience=spec.train.no_improve_patience,
        improvement_tolerance=spec.train.improvement_tolerance,
        speedrun_callback=record,
    )

    candidate = Candidate(ident="benchmark", spec=spec)
    started = time.time()
    metrics, checkpoint = trainer.train(candidate, spec, batch_iter)
    duration_s = time.time() - started

    val_ppl = float(metrics.get("ppl_eval", metrics.get("ppl", 0.0)))
    val_loss = float(math.log(val_ppl)) if val_ppl > 0.0 else float("inf")

    summary = {
        "config": str(config.resolve()),
        "git_commit": _git_commit(),
        "device": device,
        "steps": steps,
        "eval_batches": eval_batches,
        "token_budget": token_budget,
        "duration_s": duration_s,
        "val_ppl": val_ppl,
        "val_loss": val_loss,
        "metrics": metrics,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    if history_out is not None:
        history_out.parent.mkdir(parents=True, exist_ok=True)
        history_out.write_text(json.dumps(history, indent=2))

    typer.echo(f"Benchmark summary written to {out}")
    if history_out is not None:
        typer.echo(f"Eval history written to {history_out}")


if __name__ == "__main__":
    app()
