# First Public MLX Transfer Run

So what: the repo can now stage the first public `TEVO -> autoresearch-mlx` proof end to end: run or reuse a bridge-safe TEVO sweep, pick three recipe-distinct candidates, render them into MLX `train.py`, benchmark baseline vs seeded arms, and scaffold a TEVO-aware continuation workspace from the winner.

## What This Workflow Does

The workflow is intentionally narrow:

- upstream discovery stays inside the bridge-safe `TrainRecipe` family
- downstream validation stays inside `autoresearch-mlx/train.py`
- `prepare.py` remains immutable
- MLX arms are benchmarked against the stock baseline on the same machine

Artifacts are written under a single run directory so the public report can be assembled later.

## 1. Prepare The Run

If you want a fresh TEVO sweep plus MLX arm staging:

```bash
evo-loop mlx-transfer-prepare /path/to/autoresearch-mlx \
  --run-root runs/mlx_transfer_demo \
  --config configs/exp_train_recipe_bridge_owt_10m_v1.yaml \
  --device auto \
  --generations 8 \
  --steps 120 \
  --eval-batches 4 \
  --seed 0
```

If you want the upstream TEVO discovery work on Modal, while keeping the MLX benchmark local:

```bash
evo-loop mlx-transfer-prepare /path/to/autoresearch-mlx \
  --run-root runs/mlx_transfer_demo \
  --config configs/exp_train_recipe_bridge_owt_10m_v1.yaml \
  --modal \
  --modal-gpu A10G \
  --generations 8 \
  --steps 120 \
  --eval-batches 4 \
  --seed 0
```

The Modal path is intentionally cost-conscious for the first public proof:

- only the upstream TEVO seed benchmark and sweep run remotely
- the downstream `autoresearch-mlx` arms still run on your local Mac
- the first remote pass is capped to `generations <= 4`, `steps <= 120`, and `eval_batches <= 4`
- `A10G` is the default GPU unless you opt into something larger

If you already have a frontier and want to skip the TEVO live run:

```bash
evo-loop mlx-transfer-prepare /path/to/autoresearch-mlx \
  --run-root runs/mlx_transfer_demo \
  --frontier runs/existing/frontier.json \
  --lineage runs/existing/frontier_lineage.json
```

This creates:

- `frontier.json` and optional `frontier_lineage.json`
- `frontier.state.json` when the sweep produced one
- `train_recipes/selection_manifest.json`
- `rendered_train_py/`
- `mlx_arms/arm_manifest.json`

The selected trio is fixed to:

- `quality`: lowest `ppl_code`
- `compute`: lowest `speedrun_flops_to_target`
- `balanced`: lowest rank-sum across both metrics

If two frontier entries collapse to the same `TrainRecipe.model`, the workflow skips duplicates and picks the next compatible entry.

## 2. Benchmark The MLX Arms

Run the stock baseline plus the three TEVO-seeded arms:

```bash
evo-loop mlx-transfer-benchmark runs/mlx_transfer_demo \
  --repeat 3 \
  --timeout-minutes 10
```

Default benchmark command is `uv run train.py`.

This writes:

- `mlx_results/benchmark_results.tsv`
- `mlx_results/benchmark_summary.json`
- `mlx_results/benchmark_summary.md`
- `mlx_results/winning_seed.diff`
- `public_report.md`
- `public_report.json`

If a seeded arm wins on median `val_bpb`, the command also stages:

- `continuation/repo/`
- `continuation/tevo_seed.train.py`
- `continuation/program.tevo_seeded.md`
- `continuation/results.tsv`

## 3. Audit TEVO-Owned Zones During Continuation

For the first 12 continuation experiments, keep TEVO-owned template zones frozen.

Audit the current `train.py` against the seed snapshot:

```bash
evo-loop mlx-transfer-audit \
  runs/mlx_transfer_demo/continuation/tevo_seed.train.py \
  runs/mlx_transfer_demo/continuation/repo/train.py \
  --out runs/mlx_transfer_demo/continuation/tevo_zone_snapshot.json
```

The audit reports which TEVO marker regions changed, if any.

## 4. Summaries

Summarize the seeded continuation trajectory:

```bash
evo-loop mlx-transfer-continuation-summary \
  runs/mlx_transfer_demo/continuation/results.tsv \
  --out runs/mlx_transfer_demo/continuation/summary.md
```

Rebuild the combined public report at any time:

```bash
evo-loop mlx-transfer-report runs/mlx_transfer_demo
```

## Notes

- The first public proof is intentionally **single-platform MLX on local M4**.
- TEVO `ppl_code` and MLX `val_bpb` are not compared directly; each system is compared against its own baseline.
- The workflow keeps a clean stock baseline MLX workspace separate from the seeded continuation workspace.
