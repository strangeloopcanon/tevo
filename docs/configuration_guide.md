# Configuration Guide

How to configure evolution experiments: config file layout, objective/selection profiles, search regimes, and tuning knobs.

## Config File Layout

Experiments are configured via YAML files in `configs/`. The main sections are:

| Section | Purpose | Key Fields |
|---------|---------|------------|
| `model` | Architecture spec | `emb` (embedding), `blocks` (layer configs), `head` |
| `train` | Training hyperparams | `lr`, `warmup`, `max_tokens`, `instability_threshold` |
| `data` | Dataset config | `tokenizer`, `seq_len`, `batch_size`, `shards` |
| `evolution` | Evolution settings | `population`, `rung1_tokens`, `rung2_tokens`, `pareto_objectives` |

Example minimal config structure:
```yaml
model:
  name: my-experiment
  emb: { dim: 256, vocab: 50257 }
  blocks:
    - attn: { kind: GQA, heads: 8, head_dim: 32 }
      ffn: { type: moe, hidden: 1024, n_experts: 4, k: 2 }
train:
  lr: 0.001
  max_tokens: 65536
evolution:
  population: 8
  pareto_objectives: [ppl_code, throughput, ram]
```

See [`src/transformer_evolution_llm/dsl.py`](../src/transformer_evolution_llm/dsl.py) for the full schema definition.

## Search Regimes

The same engine can be run in a "constrained optimization" posture or a more "exploration-heavy" posture purely via config.

- **Constrained optimization (default posture):**
  - Fixed `rung0_thresholds`, stronger quality/efficiency objectives, and selection tuned for stable gains.
  - Best when you want reproducible improvements around a known baseline family.
- **Exploration-heavy posture (config-driven):**
  - Enable `gate_schedule`, novelty-heavy objectives, MAP-Elites complexity banding, and a broader mutation mix.
  - Best when you want multiple structurally distinct families and gradual complexification pressure.

In both regimes, the search space is bounded by the DSL + registered mutations.

### Extending the search space (adding a new primitive)

At a high level:

1) Add a typed config to the DSL (`src/transformer_evolution_llm/dsl.py`).
2) Implement the module in the model code (`src/transformer_evolution_llm/models.py`).
3) Add any static checks / cost estimates (`src/transformer_evolution_llm/evaluation.py`).
4) Add a mutation operator so evolution can discover it (`src/transformer_evolution_llm/mutations.py` or `src/transformer_evolution_llm/template_mutation.py`).

## Objective & Selection Cookbook

In practice, the frontier is mostly controlled by three things: `rung0_thresholds` (what is allowed), `objectives` (what is rewarded), and `parent_selection` (how pressure is applied).

### Profile A: Quality + Compute (default strong baseline)

Use when you want better short-budget learning efficiency without over-optimizing for one serving metric.

```yaml
evolution:
  rung0_thresholds:
    max_params: 150000000
    max_kv_bytes_per_token: 45000
    min_throughput_proxy: 1.0
    min_layers: 12
  rung1_tokens: 300000
  rung2_tokens: 900000
  population: 24
  topk_keep: 0.45
  crossover_prob: 0.4
  parent_selection: map_elites
  archive_max_elites: 64
  structural_elite_k: 4
  adaptive_mutation: true
  objectives:
    ppl_code: min
    speedrun_flops_to_target: min
```

### Profile B: Serving-Oriented (quality + KV + throughput)

Use when deployment memory/latency matters and you still need reasonable quality.

```yaml
evolution:
  rung0_thresholds:
    max_params: 150000000
    max_kv_bytes_per_token: 45000
    min_throughput_proxy: 1.0
    min_layers: 12
    min_selector_blocks: 1
  rung1_tokens: 300000
  rung2_tokens: 900000
  population: 24
  topk_keep: 0.45
  crossover_prob: 0.4
  parent_selection: map_elites
  archive_max_elites: 64
  structural_elite_k: 4
  objectives:
    ppl_code: min
    speedrun_flops_to_target: min
    kv_bytes_per_token: min
    throughput: max
```

### Profile C: Diversity Discovery (find new motifs)

Use when you want multiple structurally different lineages instead of one dominant family.

```yaml
evolution:
  rung0_thresholds:
    max_params: 150000000
    max_kv_bytes_per_token: 45000
    min_throughput_proxy: 1.0
    min_layers: 10
  rung1_tokens: 300000
  rung2_tokens: 900000
  population: 24
  topk_keep: 0.8
  crossover_prob: 0.3
  parent_selection: epsilon_lexicase
  epsilon_lexicase_epsilon: 0.05
  structural_elite_k: 4
  objectives:
    ppl_code: min
    novelty: max
    graph_entropy: max
    throughput: max
```

### Profile D: Progressive Complexity Exploration

Use when you want minimal-to-complex pressure, novelty niches, and broader structural exploration in one run.

```yaml
evolution:
  rung0_thresholds:
    max_params: 160000000
    max_kv_bytes_per_token: 50000
    min_throughput_proxy: 0.8
    min_layers: 2
  gate_schedule:
    - generation: 0
      thresholds: { min_layers: 2 }
    - generation: 15
      thresholds: { min_layers: 4, min_moe_blocks: 1 }
    - generation: 30
      thresholds: { min_layers: 8, min_moe_blocks: 2 }
  parent_selection: map_elites
  map_elites_complexity_band: true
  complexity_band_width: 4.0
  topk_keep: 0.8
  crossover_prob: 0.35
  adaptive_mutation: true
  register_template_entries: true
  objectives:
    ppl_code: min
    novelty: max
    graph_entropy: max
```

## Parent Selection Cheat Sheet

| Strategy | When to use | Trade-off |
|---|---|---|
| `map_elites` | Best default for broad search and niche retention | More exploration, slower collapse |
| `epsilon_lexicase` | Noisy metrics and multi-niche pressure | More variance run-to-run |
| `lexicase` | Strong niche pressure with low noise | Can be brittle with noisy objectives |
| `pareto_uniform` | Objective scales differ a lot | Weaker exploitation |
| `weighted` | Late-phase exploitation | Sensitive to metric scales |

## Knob Priority (What Usually Matters Most)

1. Set hard gates first (`rung0_thresholds`) for non-negotiables.
2. Keep `objectives` to 2-4 metrics that match the run goal.
3. Pick `parent_selection` for the phase: exploration (`map_elites` / `epsilon_lexicase`) vs exploitation (`weighted`).
4. Tune exploration pressure with `topk_keep` (higher = broader search).
5. Tune compute fidelity with `rung1_tokens`/`rung2_tokens`; complex motifs usually need more rung budget.
6. Use `adaptive_mutation` and `structural_elite_k` to avoid early collapse.

## Score-Weight Notes

- `scripts/run_live.py` supports CLI score-weight overrides for common metrics (`ppl`, `throughput`, `long_recall`, `ram`, `layers`, `moe_blocks`, `novelty`, `instability`, `prior_distance`).
- If your objective list includes very large-scale metrics (for example `speedrun_flops_to_target`), avoid `weighted` selection unless you intentionally normalize; `map_elites`, `pareto_uniform`, or `epsilon_lexicase` are usually safer.

## Sparse Attention Patterns

The DSL supports `sparsity: none|sliding|block|local_global|dilated|local_block`.
- `local_global` combines a local window with periodic global tokens.
- `dilated` allows attention to tokens that share the same index mod `dilation`.

## Optimizers

You can switch optimizers via the DSL:
```yaml
optimizer:
  name: lion    # or adamw
  lr: 3.0e-4
  betas: [0.9, 0.99]
```
Evolution can mutate optimizer recipes compositionally via:
- `resample_optimizer_base` (base family resampling)
- `tune_optimizer` (hparam jitter)
- `toggle_gradient_transform_mode` / `tune_gradient_transform_*`
- update-filter mutations (`toggle_update_filter_mode`, `tune_update_filter_*`)
- `mix_optimizer_recipe` (chains multiple recipe edits in one mutation)

## Example Sweep Commands

### Long-Context Sweep (Mac M4 / MPS)

Designed to stay disk-safe by pruning checkpoints to just the frontier.

```bash
export TOKENIZERS_PARALLELISM=false
RUN="runs/exp_longctx_full_deck_2h_m4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

HF_TOKEN="$HF_TOKEN" python scripts/run_live.py configs/exp_longctx_overnight_m4_full_deck.yaml \
  --device mps --generations 400 --steps 240 --eval-batches 4 --seed 4242 \
  --mutation-steps 2 \
  --out "$RUN/frontier.json" \
  --lineage-out "$RUN/frontier_lineage.json" \
  --state-out "$RUN/frontier.state.json" \
  --checkpoint-dir "$RUN/checkpoints" \
  --prune-checkpoints-to-frontier \
  2>&1 | tee "$RUN/live.log"

python scripts/report_motifs.py "$RUN/frontier.json" --lineage "$RUN/frontier_lineage.json" --top 15
```

Replace `--device mps` with `--device cuda` on NVIDIA GPUs.

### NanoGPT-Style Benchmark

```bash
BENCH="runs/bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH"
PYTHONPATH=src python scripts/run_benchmark.py \
  configs/bench_nanogpt_owt_baseline.yaml \
  --device cpu \
  --steps 40 \
  --eval-batches 2 \
  --out "$BENCH/summary.json" \
  --history-out "$BENCH/history.json"
```

See [nanogpt_benchmark.md](nanogpt_benchmark.md) for the packed-token data contract and the actual benchmark configs/budgets.
