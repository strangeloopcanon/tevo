# Run History & Evolution Log

So what: this page is a historical log of earlier TEVO sweeps and benchmark runs. The current shareable proof for the repo is the motif-transfer result in [motif_transfer_demo.md](motif_transfer_demo.md), not one of the older frontiers summarized here.

## Local Artifacts

| Run | Config | Frontier Size | Notable Findings |
|-----|--------|---------------|------------------|
| `frontier_phi_seeded128` | `seed_xover-48-9237.yaml` (128 gens) | 25 | Shallow retro-heavy stacks; `ppl_code≈1.0` |
| `frontier_phi_gated128` | `live_phi_tiny.yaml` (128 gens) | 1 | Deep (12 layers) but unstable |
| `frontier_phi_entropy_v2` | `seed_xover-48-9237.yaml` (160 gens) | 99 | Balanced mix of shallow retro and deep MoE/SSM hybrids (up to 30 layers) |

## Modal Runs (packed OWT benchmarks)

| Run | Config | Frontier Size | Notable Findings |
|-----|--------|---------------|------------------|
| `modal_nanogpt_speedrun_long1` | `exp_nanogpt_speedrun_owt.yaml` (24 gens, 120 steps) | 7 | Dense MHA stacks (9-12 layers); no MoE/SSM/retro; KV window + Alibi variants |
| `modal_nanogpt_speedrun_long3` | `exp_nanogpt_speedrun_owt.yaml` (48 gens, 180 steps) | 8 | Dense MHA stacks with layer-count shifts; no MoE/SSM/retro in frontier; small toggles (Alibi/precision/graph) |
| `modal_nanogpt_speedrun_valfix_full1` | `exp_nanogpt_speedrun_owt.yaml` (48 gens, 1000 steps) | 1 | Mixed `kv_groups` + memory extras + `branch_router` hit target at 40,960 vs 61,440 tokens on tiny OWT subset. |
| `modal_nanogpt_speedrun_owt10m_full1` | `exp_nanogpt_speedrun_owt_10m.yaml` (48 gens, 240 steps) | 1 | 10M/1M packed OWT: winner adds 1x MLA block + one Alibi block; hits target at 40,960 vs 61,440 tokens and improves `ppl_code` vs seed. |
| `modal_nanogpt_speedrun_owt10m_dyn1` | `exp_nanogpt_speedrun_owt_10m.yaml` (96 gens, 360 steps) | 1 | 10M/1M packed OWT: calibrated target (2.5k) yields multiple token buckets; winner improves `ppl_code` strongly but is slower (throughput trade-off). |
| `modal_deepseek_style_owt10m_dyn1` | `exp_deepseek_style_owt_10m.yaml` (96 gens, 360 steps) | 11 | DeepSeek-style pressure (KV bytes + throughput + selector): frontier contains MLA/GQA variants that reduce `kv_bytes_per_token` while keeping throughput high. |
| `modal_speedrun_owt10m_v3_full1` | `exp_nanogpt_speedrun_owt_10m_v3.yaml` (96 gens, 360 steps) | 3 | V3 compute-to-target (`speedrun_flops_to_target`): frontier stayed mostly dense MHA; `memory_tokens` shows up as a recurring "speed" assist. |
| `modal_selector_owt10m_v3_full1` | `exp_selector_style_owt_10m_v3.yaml` (96 gens, 360 steps) | 17 | V3 selector-style pressure: larger frontier with MLA + KV-policy quant points (e.g., `kv_policy.quant=nf4` + 1x GQA) plus some memory modules. |

## Historical Selector-Style v3 Frontier Snapshot

Best-quality point from the selector-style v3 run (`modal_selector_owt10m_v3_full1`):

- **Seed config:** `configs/frontiers/exp_selector_style_owt10m_v3_20260124/toggle_alibi-14-c5d6.yaml`
- **Shape:** 12 blocks @ d_model=768. Features: Selector Attention (block 0), ALiBi (blocks 1, 8), MLA (block 8, `kv_latent_dim=192`).
- **Metrics (A10G):** `ppl_code≈1277`, `throughput≈16.5k tok/s`, `kv_bytes/tok≈34.5k`, `speedrun_flops≈8.66e12`, `tokens_to_target≈47k`.

KV-efficient point from the same historical frontier (`kv_policy` is inference-side):

- **Seed config:** `configs/frontiers/exp_selector_style_owt10m_v3_20260124/toggle_kv_policy-92-99c7.yaml`
- **Shape:** 12 blocks. Features: 1x GQA (`kv_groups=3`), plus Retro, Memory Tokens, Layer Scale, and Gated Attention; `kv_policy.quant=nf4`.
- **Metrics (A10G):** `ppl_code≈1424`, `throughput≈16.8k tok/s`, `kv_bytes/tok≈8.7k` (4-bit quant), `speedrun_flops≈8.50e12`, `tokens_to_target≈45.7k`.

## Architecture Highlights (Historical)

| Theme | What we learned |
|-------|-----------------|
| **Triple-retro loops** | Best single-block models carry three separate retro rails feeding the same residual |
| **MoE + SSM hybrids** | 5-6 block stacks with dual MoE cores and Mamba SSMs reach ppl≈1.87 |
| **Checkpoint pruning** | Old checkpoints are removed automatically; long sweeps no longer eat disk |

## Historical Runs (Referenced in earlier versions)

- `frontier_phi_creative_canon.json` -- Composite objective + throughput experiments
- `frontier_phi_creative_super_recur_mps.json` -- Deep recurrence sweeps
- `frontier_phi_promotion_mps.json` -- Promotion rung experiments
- Various Pareto/lexicase sweeps with diverse hybrids (MoE + SSM + retro + sparsity + recurrence)
