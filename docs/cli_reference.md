# CLI Reference

Command-line tools for running evolution, inspecting results, and managing artifacts.

## `evo-loop` CLI

The package installs an `evo-loop` CLI tool with several subcommands:

```bash
evo-loop convert-checkpoints runs/<run>/checkpoints --dtype fp16 --apply

evo-loop --help
```

## Main Script: `scripts/run_live.py`

Key arguments:

```
--device          Device to use (cpu, cuda, mps)
--generations     Number of evolution generations
--steps           Training steps per candidate
--eval-batches    Number of batches for evaluation
--seed            Random seed
--out             Output frontier JSON path
--lineage-out     Output lineage JSON path
--checkpoint-dir  Directory for model checkpoints
--prune-checkpoints-to-frontier  Delete non-frontier checkpoints after run
--mutation-steps  Number of mutations to chain per child
--parent-selection  Selection strategy (weighted, pareto_uniform, lexicase, map_elites)
```

## Inspecting Results

The output JSON files contain the full frontier. You can inspect the lineage or export specific candidates.

```bash
python scripts/export_seed.py runs/<run>/frontier.json \
  --id <candidate_id> \
  --out-config configs/seed_winner.yaml
```

## Reproducing a Run

To reproduce a specific architecture, export its spec (and optionally its checkpoint) and rerun a short sweep from that seed. The runner always evaluates the seed first before spawning children.

```bash
RUN="runs/replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python scripts/run_live.py configs/seed_winner.yaml \
  --device mps --generations 1 --steps 240 --eval-batches 4 \
  --out "$RUN/frontier.json" --checkpoint-dir "$RUN/checkpoints" \
  2>&1 | tee "$RUN/live.log"
```

## Motif Reports

```bash
python scripts/report_motifs.py runs/<RUN>/frontier.json \
  --lineage runs/<RUN>/frontier_lineage.json --top 15
```

## Disk Hygiene

Runs can accumulate checkpoints quickly. Useful cleanup tools:

```bash
python scripts/archive_run.py runs/<run_dir> --delete-checkpoints

evo-loop convert-checkpoints runs/<run_dir>/checkpoints --dtype fp16 --apply
```
