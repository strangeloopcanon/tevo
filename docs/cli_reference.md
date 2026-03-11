# CLI Reference

So what: the repo now has a real CLI surface for core TEVO runs, `TrainRecipe` export/render, and `autoresearch` transfer workflows. This page is the fast map of the commands you are most likely to need.

## `evo-loop` CLI

The package installs an `evo-loop` CLI with these top-level command families:

```bash
evo-loop --help
```

- Core search:
  - `run`
  - `resume-state`
  - `frontier`
  - `export-seed`
- `TrainRecipe` bridge:
  - `train-recipe-export`
  - `train-recipe-render`
- Transfer workflows:
  - `cuda-transfer-prepare`
  - `cuda-transfer-benchmark`
  - `cuda-transfer-report`
  - `mlx-transfer-prepare`
  - `mlx-transfer-benchmark`
  - `mlx-transfer-audit`
  - `mlx-transfer-continuation-summary`
  - `mlx-transfer-report`
- Maintenance:
  - `prune-checkpoints`
  - `cleanup-run`
  - `convert-checkpoints`
  - `cache`
  - `version`

## Core TEVO Run

Run an evolutionary search directly:

```bash
evo-loop run configs/live_smoke.yaml \
  --device cpu \
  --generations 3 \
  --steps 40 \
  --eval-batches 2 \
  --seed 0
```

## Inspecting Results

```bash
evo-loop frontier runs/<run>/frontier.json

evo-loop export-seed runs/<run>/frontier.json \
  <candidate_id> \
  configs/seed_winner.yaml
```

Use the reporting scripts when you want more context than the CLI summary:

```bash
python scripts/report_motifs.py runs/<run>/frontier.json \
  --lineage runs/<run>/frontier_lineage.json --top 15
```

## TrainRecipe Bridge

```bash
evo-loop train-recipe-export runs/<run>/frontier.json \
  --candidate-id <candidate_id> \
  --out artifacts/train_recipes/<candidate_id>.yaml

evo-loop train-recipe-render artifacts/train_recipes/<candidate_id>.yaml \
  --backend autoresearch_cuda \
  --train-py /path/to/autoresearch/train.py
```

See [train_recipe_bridge.md](train_recipe_bridge.md) for the compatibility rules and projection behavior.

## CUDA Transfer Workflow

```bash
evo-loop cuda-transfer-prepare \
  --run-root runs/cuda_transfer_demo \
  --config configs/exp_train_recipe_bridge_owt_10m_v1.yaml \
  --modal-gpu A10G

evo-loop cuda-transfer-benchmark runs/cuda_transfer_demo \
  --repeat 3 \
  --timeout-minutes 10

evo-loop cuda-transfer-report runs/cuda_transfer_demo
```

See [cuda_transfer_demo.md](cuda_transfer_demo.md) and [motif_transfer_demo.md](motif_transfer_demo.md) for the recommended public-facing CUDA path.

## MLX Transfer Workflow

```bash
evo-loop mlx-transfer-prepare /path/to/autoresearch-mlx \
  --run-root runs/mlx_transfer_demo \
  --config configs/exp_train_recipe_bridge_owt_10m_v1.yaml

evo-loop mlx-transfer-benchmark runs/mlx_transfer_demo

evo-loop mlx-transfer-report runs/mlx_transfer_demo
```

See [mlx_transfer_demo.md](mlx_transfer_demo.md) for the full flow.

## Disk Hygiene

Runs can accumulate checkpoints quickly. Useful cleanup tools:

```bash
evo-loop cleanup-run runs/<run_dir>/frontier.manifest.json --apply

evo-loop prune-checkpoints runs/<run_dir>/checkpoints \
  --state-path runs/<run_dir>/frontier.state.json

evo-loop convert-checkpoints runs/<run_dir>/checkpoints --dtype fp16 --apply
```
