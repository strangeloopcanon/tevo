# Scale Policy

This doc captures the soft priors we use while exploring architectures on surrogate-scale models and how to adapt them for scale-hop runs. Three regimes are relevant today:

| Regime | Params | Purpose |
| --- | --- | --- |
| Surrogate | ~56 M | Fast evolutionary search (laptop, ~200-300 steps) |
| GPT-2-grade | ~477 M | Mid-scale validation; confirms search results transfer |
| Scale-hop | ~973 M | Pre-production sanity check before multi-billion training |

## Concrete dimension tables

### Surrogate (~56 M)
| Knob | Value |
| --- | --- |
| `d_model` | 256 |
| `heads` | 4 |
| `head_dim` | 64 |
| `layers` | 8 |
| `ffn_hidden` | 1024 (4 x d) |
| `kv_groups` | 1-2 |
| `vocab` | 50 257 |

### GPT-2-grade (~477 M)
| Knob | Value |
| --- | --- |
| `d_model` | 1280 |
| `heads` | 10 |
| `head_dim` | 128 |
| `layers` | 20 |
| `ffn_hidden` | 5120 (4 x d) |
| `kv_groups` | 1 |
| `vocab` | 32 768 |
| `norm` | rmsnorm |
| `rope` | standard |
| `optimizer` | AdamW, lr=3e-4, betas=(0.8, 0.95) |
| `weight_decay` | 0.2 |
| `max_tokens` | 4 B |

### Scale-hop (~973 M)
| Knob | Value |
| --- | --- |
| `d_model` | 1536 |
| `heads` | 24 |
| `head_dim` | 64 |
| `layers` | 31 |
| `ffn_hidden` | 6144 (4 x d) |
| `kv_groups` | 2-4 |
| `vocab` | 32 768 |
| `norm` | rmsnorm |
| `rope` | standard |
| `optimizer` | AdamW or Muon, lr=2e-4 |
| `weight_decay` | 0.1 |
| `max_tokens` | 10-20 B |

## Priors encoded in the DSL

| Knob | Default | Rationale |
| --- | --- | --- |
| `model.norm` | `layernorm` (RMSNorm optional) | LayerNorm is the baseline; RMSNorm often stabilizes larger stacks. |
| `attn.head_dim` | 64 | Keeps QK compute efficient and portable; matches most modern LLMs. |
| `ffn.hidden` | `4 x d_model` | Standard transformer width multiplier. |
| `attn.kv_groups` | 1-2 | KV compression halves KV cache without hurting quality on surrogates. At 973 M, groups of 2-4 save memory. |
| `attn.sparsity` | `local_global` default window ~sqrt(seq_len) | Balances local detail and global sentinels. |
| `attn.global_stride` | ~sqrt(seq_len) | One global token per sqrt(L) positions keeps receptive field broad. |
| `attn.rope_theta` | 10 000 (jitter allowed) | Base RoPE value; slight jitter explores length generalisation. |
| `train.priors.tokens_per_param` | 4-20 | At surrogate scale use 4; at 477 M use ~8; at 973 M use 10-20. |

All priors are soft: the search can deviate when it helps. The `scripts/fit_scaling.py` utility re-fits these trends from run history.

## Token and compute budgeting

| Regime | Token budget | Steps (approx) | GPU-hours (A10G) |
| --- | --- | --- | --- |
| Surrogate (56 M) | 400 M | 200-300 | <1 per candidate |
| GPT-2-grade (477 M) | 4 B | ~16 000 | 8-12 per candidate |
| Scale-hop (973 M) | 10-20 B | ~40 000 | 30-60 per candidate |

Rung schedule (all regimes, same ratios):
- Rung 0: static validity checks (params, KV budget, structural constraints).
- Rung 1: 20% of configured steps (quick filter). Candidates with `ppl_code > 2.5` stop early.
- Rung 2: full steps.

## Scale-hop checklist (477 M -> 973 M)

1. Export a seed (spec + checkpoint) from the GPT-2-grade frontier.
2. Increase width from 1280 to 1536, heads from 10 to 24 (head_dim=64), layers from 20 to 31, FFN hidden from 5120 to 6144.
3. Set `kv_groups=2` or higher to control KV cache at 973 M.
4. Increase `train.max_tokens` to 10-20 B (roughly 10-20x tokens per parameter).
5. Keep `local_global` window scaled with sqrt(seq_len); for 2k+ context, set `window_scale` closer to 4.
6. Run `scripts/fit_scaling.py` on GPT-2-grade frontiers; record predicted ppl/throughput for the 973 M target.
7. Train the 973 M model for a 30-60 GPU-hour budget (e.g. 5-10k steps) to validate stability and trend adherence.
8. If using Muon optimizer at 973 M, set lr=0.02 and use AdamW for embeddings/biases.

Document the run (frontier + lineage + fit outputs) before committing to multi-billion parameter training.

## GPT-2-grade specifics

The `ref_nanochat_gpt2grade_d20` config provides the baseline for 477 M runs:
- 20 layers alternating between sliding-window attention and full attention
- ReLU activation (consider relu_squared or SwiGLU for better frontier candidates)
- Trained on packed OpenWebText at seq_len=2048
- Speedrun evaluation at 16-step intervals

When running evolution at 477 M, reduce population size (8-12 candidates) and increase per-candidate budget to compensate for the higher cost per evaluation.
