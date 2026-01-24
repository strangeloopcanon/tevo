# NanoGPT-Style Benchmark Contract

So what: we now have a small, repeatable benchmark harness to compare architectures *inside this repo* using speedrun-style time/tokens-to-target, plus a fixed token-budget eval.

## Benchmarks

### A) OpenWebText packed stream (NanoGPT-style)
- Packed GPT-2 token stream from OpenWebText.
- Random block sampling on train; eval on a held-out packed split.
- Config: `configs/bench_nanogpt_owt.yaml`.

Canonical baseline (NanoGPT-like stack, 12×MHA, seq_len=1024):
- Config: `configs/bench_nanogpt_owt_baseline.yaml`.

### B) HF mix (existing shards)
- Uses the current shard mix to keep continuity with older runs.
- Same speedrun logging and metrics.
- Config: `configs/bench_nanogpt_hfmix.yaml`.

## What this is / is not

- **Is:** a consistent, local benchmark to compare candidates under the same recipe.
- **Is not:** a leaderboard or a direct comparison to NanoGPT/FineWeb speedruns (dataset, tokenizer, steps, and hardware differ).

## How to run

Local:
```
PYTHONPATH=src python scripts/run_benchmark.py \
  configs/bench_nanogpt_owt.yaml \
  --device mps \
  --steps 240 \
  --eval-batches 4 \
  --out runs/bench/owt_summary.json \
  --history-out runs/bench/owt_history.json
```

Modal:
```
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_benchmark.py \
  --config-path configs/bench_nanogpt_owt.yaml \
  --steps 240 \
  --eval-batches 4 \
  --run-id bench_owt \
  --download
```

## Reporting checklist

- Record config path, git commit, device, steps, token budget.
- Report `val_loss`, `val_ppl`, `throughput_tok_s` (from summary JSON).
- Report `speedrun_*` metrics. If the target is not reached, those fields stay at the sentinel `1e9`.

## Evolution objective

Use `configs/exp_nanogpt_speedrun_owt.yaml` for evolution runs that include `speedrun_tokens_to_target` and `speedrun_time_to_target` in the objective set. Keep `ppl_stop_threshold: null` so the speedrun probes run.

## Latest NanoGPT-objective frontier (A10G, longer budget)

So what: even with a lower target and longer budgets, the frontier still favors dense MHA stacks; novelty shows up mostly as layer-count changes and small toggles.

- Run: `runs/modal/modal_nanogpt_speedrun_long3/` (48 generations, 180 steps, A10G).
- Frontier size: 8; `speedrun_tokens_to_target` spans 40960–81920 (some never reach target); `ppl_code` ~2402–4989.
- Variants that survived: Alibi/precision toggles, layer-count shifts, and a graph-module insertion; no MoE/SSM/retro/recurrence blocks in the frontier.
- Limitation: still a proxy under short training; it does not measure long-context memory or downstream tasks.

<details>
<summary>Data prep, cache layout, and knobs</summary>

Packed-token files live under `runs/packed/` by default. OpenWebText is:
- `runs/packed/openwebtext/train.bin`
- `runs/packed/openwebtext/val.bin`

Build the cache:
```
PYTHONPATH=src python scripts/prepare_packed_data.py \
  --dataset openwebtext \
  --out-dir runs/packed/openwebtext \
  --streaming \
  --max-train-tokens 10000000 \
  --max-val-tokens 1000000
```

Notes:
- The packed files are just token ids, so sizes are predictable (`uint16` ≈ 2 bytes/token). The example above is ~22MB on disk.
- For a quick smoke cache, drop those caps down (for example `--max-train-tokens 200000 --max-val-tokens 50000`).

On Modal, point `packed_train_path` / `packed_val_path` at a mounted volume (for example `/runs/packed/openwebtext/train.bin`).
If you keep relative paths like `runs/packed/...`, set `TEVO_PACKED_ROOT=/runs` inside the container.

Speedrun targets can be expressed as `speedrun_target_ppl` or `speedrun_target_loss` (pick one). When `speedrun_target_ppl` is set, we convert via `loss = log(ppl)`.
</details>
