# NanoGPT-Style Benchmark Contract

So what: we now have a small, repeatable benchmark harness to compare architectures *inside this repo* using speedrun-style time/tokens-to-target, plus a fixed token-budget eval.

## Benchmarks

### A) OpenWebText packed stream (NanoGPT-style)
- Packed GPT-2 token stream from OpenWebText.
- Random block sampling on train; eval on a held-out packed split.
- Config: `configs/bench_nanogpt_owt.yaml`.

Canonical baseline (NanoGPT-like stack, 12×MHA, seq_len=1024):
- Config: `configs/bench_nanogpt_owt_baseline.yaml`.

Larger packed stream (recommended for non-toy runs):
- Packed data: `runs/packed/openwebtext_10m/` (see “Data prep” below).
- Baseline config: `configs/bench_nanogpt_owt_baseline_10m.yaml`.
- Evolution config: `configs/exp_nanogpt_speedrun_owt_10m.yaml`.
- V3 evolution configs (compute-to-target): `configs/exp_nanogpt_speedrun_owt_10m_v3.yaml` and `configs/exp_selector_style_owt_10m_v3.yaml`.
- `speedrun_target_ppl` is calibrated so the baseline does not hit it on the first eval interval.

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
- Report `speedrun_*` metrics (including `speedrun_flops_to_target` and `flops_per_token_est`). If the target is not reached, the `*_to_target` fields stay at large sentinels.

## Evolution objective

Use `configs/exp_nanogpt_speedrun_owt.yaml` for evolution runs that include `speedrun_tokens_to_target` and `speedrun_time_to_target` in the objective set. Keep `ppl_stop_threshold: null` so the speedrun probes run.

V3 (recommended): use `configs/exp_nanogpt_speedrun_owt_10m_v3.yaml` to minimize estimated compute-to-threshold via `speedrun_flops_to_target` (and still track `ppl_code`). The FLOPs estimate is a crude trunk-only proxy intended for relative comparisons.

## Recent Modal runs (A10G)

So what: speedrun-to-target can find early-learning wins. Historically, `speedrun_tokens_to_target` was bucketed by the eval interval; we now interpolate between eval points so the metric is less discretized.

- NanoGPT objective on `openwebtext_10m` (`configs/exp_nanogpt_speedrun_owt_10m.yaml`): `runs/modal/modal_nanogpt_speedrun_owt10m_dyn1/` (96 generations, 360 steps). Frontier collapsed to 1 dominant model (`tune_kv-72-a316`), which hit the target at 40,960 tokens vs 57,344 for the seed and improved short-budget `ppl_code` (~769 vs ~1616). Trade-off: much lower measured throughput (~7.9k vs ~17.8k tok/s).
- DeepSeek-style objective on `openwebtext_10m` (`configs/exp_deepseek_style_owt_10m.yaml`): `runs/modal/modal_deepseek_style_owt10m_dyn1/` (96 generations, 360 steps). Frontier size 11; the best KV-efficient points used 1× MLA and reached `kv_bytes_per_token` as low as 31,232 (vs 36,864 for the seed) while keeping throughput ~17k tok/s, with `ppl_code` in the ~1.3k–1.8k range.
- Historical tiny packed OWT: `runs/modal/modal_nanogpt_speedrun_long3/` (48 generations, 180 steps): frontier size 8; mostly dense attention stacks with small toggles (Alibi/precision/graph), no MoE/SSM/retro/recurrence blocks in the frontier.
- V3 smoke validation (FLOPs-to-target objective, 10M packed OWT):
  - `configs/exp_nanogpt_speedrun_owt_10m_v3.yaml`: `runs/modal/modal_speedrun_owt10m_v3_seed1/` (1 generation, 360 steps). Seed hits the target with interpolated `speedrun_tokens_to_target` (~46.9k) and reports `speedrun_flops_to_target`.
  - `configs/exp_selector_style_owt_10m_v3.yaml`: `runs/modal/modal_selector_owt10m_v3_seed1/` (1 generation, 360 steps). Same target, plus selector/KV objectives are active.

<details>
<summary>Data prep, cache layout, and knobs</summary>

Packed-token files live under `runs/packed/` by default. OpenWebText is:
- `runs/packed/openwebtext/train.bin`
- `runs/packed/openwebtext/val.bin`

Build the cache:
```
PYTHONPATH=src python scripts/prepare_packed_data.py \
  --dataset openwebtext \
  --out-dir runs/packed/openwebtext_10m \
  --streaming \
  --val-fraction 0.1 \
  --max-train-tokens 10000000 \
  --max-val-tokens 1000000
```

Notes:
- The packed files are just token ids, so sizes are predictable (`uint16` ≈ 2 bytes/token). The example above is ~22MB on disk.
- For a quick smoke cache, drop those caps down (for example `--max-train-tokens 200000 --max-val-tokens 50000`).

On Modal (writes into the `tevo-runs` volume under `/runs/packed/...`):
```
modal run scripts/modal_prepare_packed_data.py \
  --out-dir openwebtext_10m \
  --max-train-tokens 10000000 \
  --max-val-tokens 1000000 \
  --val-fraction 0.1 \
  --download-metadata
```

On Modal, point `packed_train_path` / `packed_val_path` at a mounted volume (for example `/runs/packed/openwebtext_10m/train.bin`).
If you keep relative paths like `runs/packed/...`, set `TEVO_PACKED_ROOT=/runs` inside the container.

Speedrun targets can be expressed as `speedrun_target_ppl` or `speedrun_target_loss` (pick one). When `speedrun_target_ppl` is set, we convert via `loss = log(ppl)`.
</details>
