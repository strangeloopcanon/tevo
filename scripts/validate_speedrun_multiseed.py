"""Run and summarize multi-seed speedrun benchmarks.

This is intended for *validation*, not for use inside evolution:
- Run a small set of candidate configs across a few seeds (e.g., 3)
- Summarize time/tokens-to-target stability and trade-offs

Examples:
  # Benchmark all exported frontier seeds across 3 seeds on Modal (A10G)
  TEVO_MODAL_GPU=A10G python scripts/validate_speedrun_multiseed.py \
    --config-dir configs/frontiers/exp_nanogpt_speedrun_owt10m_v7_20260125 \
    --seeds 0,1,2 --steps 360 --eval-batches 8 --max-tokens 900000 --modal
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import typer


app = typer.Typer(help="Run multi-seed speedrun benchmarks and summarize results.")

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchResult:
    config_path: Path
    seed: int
    summary_path: Path
    metrics: dict[str, float]


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise typer.BadParameter("Provide at least one seed (e.g., --seeds 0,1,2)")
    return seeds


def _load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    metrics = data.get("metrics") or {}
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _median(values: list[float]) -> float:
    values = [float(v) for v in values if v is not None]
    if not values:
        return float("inf")
    return float(statistics.median(values))


def _repo_relative(path: Path) -> str:
    """Return a path relative to the repo root when possible.

    Modal workers mount the repo under /repo, so passing absolute local paths
    (e.g., /Users/...) will fail inside the container.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _run_modal_benchmark(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    eval_batches: int,
    max_tokens: int | None,
    run_id: str,
    local_out_dir: Path,
) -> Path:
    env = dict(os.environ)
    cmd = [
        "modal",
        "run",
        "scripts/modal_run_benchmark.py",
        "--config-path",
        _repo_relative(config_path),
        "--steps",
        str(int(steps)),
        "--eval-batches",
        str(int(eval_batches)),
        "--seed",
        str(int(seed)),
        "--run-id",
        run_id,
        "--download",
        "--local-out-dir",
        str(local_out_dir),
    ]
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(int(max_tokens))])
    subprocess.run(cmd, check=True, env=env)
    return local_out_dir / run_id / "summary.json"


def _run_local_benchmark(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    eval_batches: int,
    max_tokens: int | None,
    run_id: str,
    local_out_dir: Path,
) -> Path:
    out_root = local_out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    history_path = out_root / "history.json"
    cmd = [
        "python",
        "scripts/run_benchmark.py",
        str(config_path),
        "--steps",
        str(int(steps)),
        "--eval-batches",
        str(int(eval_batches)),
        "--seed",
        str(int(seed)),
        "--out",
        str(summary_path),
        "--history-out",
        str(history_path),
    ]
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(int(max_tokens))])
    subprocess.run(cmd, check=True)
    return summary_path


@app.command()
def main(
    config_dir: Path | None = typer.Option(
        None, exists=True, file_okay=False, help="Directory of YAML configs to benchmark."
    ),
    config: list[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Config path(s) to benchmark (repeatable).",
    ),
    seeds: str = typer.Option("0,1,2", help="Comma-separated seeds (e.g., 0,1,2)."),
    steps: int = typer.Option(360, min=1),
    eval_batches: int = typer.Option(8, min=1),
    max_tokens: int | None = typer.Option(None, help="Override token budget for the benchmark."),
    modal: bool = typer.Option(False, help="Run on Modal via scripts/modal_run_benchmark.py."),
    out_dir: Path = typer.Option(Path("runs/modal/multiseed"), help="Local output root."),
    run_tag: str = typer.Option("multiseed", help="Prefix for run ids and output files."),
    reuse: bool = typer.Option(True, help="Reuse existing summary.json if present."),
    csv_out: Path | None = typer.Option(
        None, help="Optional CSV output path (defaults under out_dir)."
    ),
) -> None:
    configs: list[Path] = []
    if config_dir is not None:
        configs.extend(sorted(config_dir.glob("*.yaml")))
        configs.extend(sorted(config_dir.glob("*.yml")))
    configs.extend(list(config or []))
    if not configs:
        raise typer.BadParameter("Provide --config-dir or at least one --config path.")
    configs = [path.resolve() for path in configs]
    uniq: list[Path] = []
    seen: set[Path] = set()
    for path in configs:
        if path in seen:
            continue
        seen.add(path)
        uniq.append(path)
    configs = uniq

    seed_list = _parse_seeds(seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    results: list[BenchResult] = []
    for cfg_path in configs:
        for seed in seed_list:
            run_id = f"{run_tag}_{cfg_path.stem}_s{seed}_{ts}"
            summary_path = out_dir / run_id / "summary.json"
            if reuse and summary_path.exists():
                metrics = _load_metrics(summary_path)
                results.append(BenchResult(cfg_path, seed, summary_path, metrics))
                continue
            if modal:
                summary_path = _run_modal_benchmark(
                    config_path=cfg_path,
                    seed=seed,
                    steps=steps,
                    eval_batches=eval_batches,
                    max_tokens=max_tokens,
                    run_id=run_id,
                    local_out_dir=out_dir,
                )
            else:
                summary_path = _run_local_benchmark(
                    config_path=cfg_path,
                    seed=seed,
                    steps=steps,
                    eval_batches=eval_batches,
                    max_tokens=max_tokens,
                    run_id=run_id,
                    local_out_dir=out_dir,
                )
            metrics = _load_metrics(summary_path)
            results.append(BenchResult(cfg_path, seed, summary_path, metrics))

    # Per-run rows
    rows: list[dict[str, object]] = []
    for res in results:
        m = res.metrics
        rows.append(
            {
                "config": str(res.config_path),
                "seed": int(res.seed),
                "speedrun_reached": float(m.get("speedrun_reached", 0.0)),
                "speedrun_time_to_target": float(m.get("speedrun_time_to_target", float("inf"))),
                "speedrun_tokens_to_target": float(m.get("speedrun_tokens_to_target", float("inf"))),
                "speedrun_steps_to_target": float(m.get("speedrun_steps_to_target", float("inf"))),
                "speedrun_loss_auc": float(m.get("speedrun_loss_auc", float("inf"))),
                "ppl_code": float(m.get("ppl_code", float("inf"))),
                "throughput": float(m.get("throughput", 0.0)),
                "summary_path": str(res.summary_path),
            }
        )

    # Aggregate per config
    by_cfg: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_cfg.setdefault(str(row["config"]), []).append(row)

    aggregates: list[dict[str, object]] = []
    for cfg_path, cfg_rows in by_cfg.items():
        reached = [r for r in cfg_rows if float(r["speedrun_reached"]) > 0.5]
        aggregates.append(
            {
                "config": cfg_path,
                "runs": len(cfg_rows),
                "reached_rate": float(len(reached)) / float(len(cfg_rows)),
                "median_time_to_target": _median([float(r["speedrun_time_to_target"]) for r in reached]),
                "median_tokens_to_target": _median(
                    [float(r["speedrun_tokens_to_target"]) for r in reached]
                ),
                "median_loss_auc": _median([float(r["speedrun_loss_auc"]) for r in cfg_rows]),
                "median_ppl_code": _median([float(r["ppl_code"]) for r in cfg_rows]),
                "median_throughput": _median([float(r["throughput"]) for r in cfg_rows]),
            }
        )

    aggregates.sort(key=lambda r: (float(r["median_loss_auc"]), float(r["median_ppl_code"])))

    if csv_out is None:
        csv_out = out_dir / f"{run_tag}_{ts}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)

    typer.echo(f"Wrote CSV summary to {csv_out}")
    for row in aggregates:
        typer.echo(
            f"{Path(str(row['config'])).name}: reached={row['reached_rate']:.2f} "
            f"auc={row['median_loss_auc']:.4f} ppl={row['median_ppl_code']:.0f} "
            f"tgt_s={row['median_time_to_target']:.1f} thr={row['median_throughput']:.0f}"
        )


if __name__ == "__main__":
    app()
