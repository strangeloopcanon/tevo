# Example Frontier Survivors

Illustrative survivors from evolution runs, showing the kinds of architectures the loop discovers.

## Long-Context Sweep (11-entry Pareto frontier, ~65-85M params)

Source (metrics + specs): `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/frontier_arch.json` (generated from `configs/exp_longctx_overnight_m4_full_deck.yaml`).

### Quality-lean memory stack (best `ppl_code`)

Source: `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/duplicate_block_span+toggle_kv_policy+add_extra_combo-292-4963.yaml`.
- Depth: 5 blocks; Memory blocks: 5/5; MoE blocks: 1/5; KV policy: `window=1024` + `int8`.
- Proxy metrics: `ppl_code≈121.45`, `passkey_loss≈7.79` (~83.4M params).

```mermaid
flowchart LR
  E1[Embed] --> D1[Depth: 5 blocks]
  D1 --> R1[Memory: 5/5 blocks]
  D1 --> M1[MoE: 1/5 blocks]
  D1 --> K1[KV policy: window=1024 int8]
  R1 --> O1[Head]
  M1 --> O1
  K1 --> O1
```

### Probe-lean hybrid (best `passkey_loss`)

Source: `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/duplicate_block_span+toggle_qk_norm+add_extra_combo-91-10b3.yaml`.
- Depth: 10 blocks; Memory blocks: 4/10; Recurrences: 1; MLA blocks: 2/10; Selector blocks: 2/10; QK-norm blocks: 1/10.
- Proxy metrics: `passkey_loss≈5.43`, `ppl_code≈194.03` (~65.3M params).

```mermaid
flowchart LR
  E2[Embed] --> D2[Depth: 10 blocks]
  D2 --> R2[Memory: 4/10 blocks]
  D2 --> A2[Alt attn: MLA 2/10]
  D2 --> S2[Selectors: 2/10]
  D2 --> C2[Recurrence spans: 1]
  D2 --> Q2[QK-norm: 1/10]
  R2 --> O2[Head]
  A2 --> O2
  S2 --> O2
  C2 --> O2
  Q2 --> O2
```

### Deeper routed memory stack (balanced quality)

Source: `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/insert_assoc_memory+tune_retro+tune_branch_router-375-1123.yaml`.
- Depth: 13 blocks; Memory blocks: 4/13; MLA blocks: 1/13; Extras: assoc-memory + branch-router + layer-scale.
- Proxy metrics: `ppl_code≈123.58`, `passkey_loss≈7.52` (~75.8M params).

```mermaid
flowchart LR
  E3[Embed] --> D3[Depth: 13 blocks]
  D3 --> R3[Memory: 4/13 blocks]
  D3 --> A3[Alt attn: MLA 1/13]
  D3 --> X3[Routing/stability extras]
  X3 --> BR3[Branch router]
  X3 --> AM3[Assoc memory]
  X3 --> LS3[LayerScale]
  R3 --> O3[Head]
  A3 --> O3
  BR3 --> O3
  AM3 --> O3
  LS3 --> O3
```

## Behavioral Memory Sweep (Modal A10G, 64 generations)

Purely behavioral selection (loss + memory/speed + novelty/entropy) repeatedly discovered *embedding-conditioned FFNs* (FFNs that read token embeddings instead of the residual stream). This trait was not an explicit objective.

- Best `ppl_code`: `1331 -> 791`
- `long_recall`: `0.0 -> 1.175`
- `kv_bytes_per_token`: `8192 -> 7168`

Motif: early embedding-conditioned FFNs + mixed `MHA/GQA` attention + lightweight memory extras.

Repro:
```bash
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_live.py \
  --config-path configs/exp_behavioral_memory_modal_v1.yaml \
  --generations 64 --steps 160 --eval-batches 4 --seed 0 \
  --download --local-out-dir runs/modal \
  --cleanup-old-checkpoints --prune-checkpoints-to-frontier --lineage
```
