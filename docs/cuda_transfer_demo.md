# First Public CUDA Transfer Run

So what: the repo can now stage a TEVO -> `autoresearch` CUDA proof that stays close to Karpathy's canon. TEVO discovery runs on Modal, compatible recipes render into upstream-style CUDA `train.py`, and the baseline plus seeded arms are benchmarked on Modal against the real 5-minute `val_bpb` loop.

## Current Validated Path

The cleanest current CUDA story is a motif-transfer probe, not a full-model transfer.

The first direct nanochat-scale render OOMed on `A100-80GB`, which taught us that the bridge should transfer the motif while normalizing scale into the upstream `autoresearch` scaffold. After adding CUDA compatibility projection, a projected frontier sibling improved the pinned upstream baseline:

| Probe | Result |
|-------|--------|
| Stock upstream CUDA `autoresearch` baseline | `val_bpb = 1.117953` |
| Projected TEVO motif-transfer probe | `val_bpb = 1.113392` |
| Delta | `-0.004561` better |
| Peak VRAM | `43.96 GiB -> 35.89 GiB` |

Treat that as the current proof-of-concept: the bridge can transfer a discovered motif usefully when scale is normalized.

## Why CUDA Here

Upstream `autoresearch` is fundamentally a CUDA benchmark:

- `prepare.py` is immutable and owns the cache under `~/.cache/autoresearch`
- `train.py` is the only mutable file
- the score is `val_bpb`
- each run is a fixed 5-minute single-GPU benchmark

That makes CUDA-on-Modal the cleanest first public validation lane.

## GPU Guidance

- Use cheaper GPUs such as `A10G` for TEVO discovery sweeps.
- Use `A100-80GB` for the downstream CUDA `autoresearch` benchmark lane.
- Do not assume `A10G` or `L40S` can fit stock upstream `autoresearch`; our probes OOMed there.

## 1. Prepare The Run

This command:

- runs the bridge-safe TEVO discovery sweep on Modal
- exports the `quality` / `compute` / `balanced` recipe trio
- clones upstream `autoresearch` locally if needed
- renders CUDA `train.py` variants
- stages local arm workspaces for diffs and artifact review

```bash
evo-loop cuda-transfer-prepare \
  --run-root runs/cuda_transfer_demo \
  --config configs/exp_train_recipe_bridge_owt_10m_v1.yaml \
  --modal-gpu A10G \
  --generations 8 \
  --steps 120 \
  --eval-batches 4 \
  --seed 0
```

Notes:

- the first TEVO discovery pass is capped to `generations <= 4`, `steps <= 120`, and `eval_batches <= 4`
- upstream `autoresearch` is cloned from `https://github.com/karpathy/autoresearch.git` by default
- if you already have a local CUDA `autoresearch` checkout, pass `--autoresearch-repo /path/to/autoresearch`
- oversized CUDA recipes are projected automatically before `train.py` is patched, so the transfer stays baseline-adjacent instead of silently inflating the model scale

If you already have a TEVO frontier and want to skip the discovery sweep:

```bash
evo-loop cuda-transfer-prepare \
  --run-root runs/cuda_transfer_demo \
  --frontier runs/existing/frontier.json \
  --lineage runs/existing/frontier_lineage.json
```

## 2. Benchmark On Modal

This command benchmarks the stock baseline plus the three TEVO-seeded CUDA arms on Modal:

```bash
evo-loop cuda-transfer-benchmark runs/cuda_transfer_demo \
  --repeat 3 \
  --timeout-minutes 10
```

The CUDA benchmark step:

- clones the recorded `autoresearch` repo URL and commit on Modal
- runs `python prepare.py`
- injects the rendered `train.py`
- runs repeated 5-minute CUDA `train.py` benchmarks
- downloads summaries and logs locally

Notes:

- Use `A100-80GB` for the CUDA benchmark lane unless you have already validated a different GPU tier against stock upstream `autoresearch`.
- The TEVO discovery sweep can still stay on a cheaper GPU such as `A10G`; only the downstream CUDA benchmark lane needs the larger card.
- The most honest first comparison is stock baseline vs one projected TEVO probe. Expand to the full seeded trio once you have a candidate family that survives the actual TEVO frontier objectives.

Artifacts land under:

- `cuda_results/benchmark_results.json`
- `cuda_results/benchmark_results.tsv`
- `cuda_results/benchmark_summary.json`
- `cuda_results/benchmark_summary.md`
- `cuda_results/winning_seed.diff`
- `cuda_public_report.md`
- `cuda_public_report.json`

## 3. Rebuild The Report

```bash
evo-loop cuda-transfer-report runs/cuda_transfer_demo
```

## Notes

- TEVO `ppl_code` and CUDA `val_bpb` are not compared directly; each system is compared against its own baseline.
- The local cloned `autoresearch_source/` tree is only for rendering and diffs; the actual benchmark clones the recorded repo URL/ref on Modal.
- This workflow is intentionally narrower than the MLX continuation flow: it optimizes first for a credible CUDA transfer proof.
- The current public proof is documented in [docs/motif_transfer_demo.md](motif_transfer_demo.md).
