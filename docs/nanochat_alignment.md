# Nanochat Alignment (Pinned Reference)

This note pins the upstream reference we aligned to and documents exact vs approximate mapping into this repo's DSL.

## Upstream Pin
- Repo: `karpathy/nanochat`
- Branch: `master`
- Commit: `aeff095e97a3721202e188b8348948536dc90a83`
- Files used:
  - `runs/speedrun.sh`
  - `scripts/base_train.py`
  - `nanochat/gpt.py`

## Upstream Targets We Mapped
- `speedrun.sh` current base run:
  - `--depth=26`
  - `--target-param-data-ratio=8.25`
  - `--device-batch-size=16`
  - `--fp8`
- Base model defaults in `base_train.py` / `gpt.py`:
  - `aspect_ratio=64` => `n_embd ~= depth*64`, rounded to multiple of `head_dim`
  - `head_dim=128`
  - `max_seq_len=2048`
  - `window_pattern=SSSL`
  - `vocab_size=32768`
  - untied embedding/head
  - RMSNorm + QK norm + `relu^2` MLP + Muon/Adam optimizer split

## DSL Equivalents Added
- `configs/ref_nanochat_gpt2grade_d20_aeff095e.yaml`
  - GPT-2-grade neighborhood (d20)
- `configs/ref_nanochat_speedrun_d26_aeff095e.yaml`
  - Current `speedrun.sh` architecture neighborhood (d26)
- `configs/exp_nanochat_gpt2grade_d20_modal_evolve_aeff095e.yaml`
  - Cost-controlled Modal evolution recipe derived from the same family

## Parameter Cross-Check (this repo model implementation)
Validated by constructing `EvolutionModel` from each config:
- d20: **477,392,128** params
- d26: **973,414,528** params

These are in the expected nanochat scale bands (`d20~477M`, `d22~599M`, larger for `d26`).

## Exact vs Approximate Mapping
Exact (or near-exact):
- depth, embedding scale, head dimension, sequence length, vocab size, untied head
- `window_pattern=SSSL` approximated structurally via per-layer pattern `[S,S,S,L]` with `sliding_window=1024` (half context), with the final layer forced to full context

Approximate (DSL/runtime limitation):
- nanochat weightless RMSNorm is mapped to this repo's RMSNorm variant (learnable scale)
- nanochat `relu^2` mapped to DSL `relu`
- Muon+Adam parameter-group optimizer mapped to single AdamW optimizer in DSL
- nanochat QK norm and value-embedding path are not one-to-one represented in current DSL model path
- nanochat custom tokenizer training path is not one-command reproducible in this repo's HF tokenizer field; configs keep runnable defaults while preserving vocab shape

## Why Two References (d20 + d26)
- d20 keeps you in the GPT-2-grade parameter regime for cheaper search loops.
- d26 matches current `speedrun.sh` script intent more closely for eventual high-end benchmark attempts.

## Ready-to-Run Evolution Preset (FineWeb + Staggered Budgets)
- Config: `configs/exp_nanochat_gpt2grade_d20_modal_evolve_fineweb_staggered_gpt2vocab_aeff095e.yaml`
- Runner: `run_nanochat_fineweb_modal_matrix.sh`

So what: this preset is tuned to avoid the earlier failure modes and provide useful evolutionary pressure on Modal:
- Uses FineWeb-EDU packed data path (`runs/packed/fineweb_edu_10m/*`) for nanochat-like pretraining signal.
- Keeps vocab aligned with GPT-2 packed token ids (`vocab=50257`) to avoid CUDA index asserts.
- Uses staggered training pressure:
  - `adaptive_rung_budget: true`
  - `adaptive_rung_fast_promote_threshold: 0.12`
  - `adaptive_rung_slow_stop_threshold: 0.015`
  - promotion rung enabled (`promotion_prob: 0.35`, 1.5x steps/tokens multiplier)

Example:
```bash
./run_nanochat_fineweb_modal_matrix.sh
```

Environment overrides:
- `TEVO_MODAL_GPU` (default `A10G`)
- `TEVO_GENERATIONS` (default `16`)
- `TEVO_STEPS` (default `180`)
- `TEVO_SEEDS` as CSV (default `0,1`)
- `TEVO_SKIP_DATA_PREP=1` to reuse existing packed data
