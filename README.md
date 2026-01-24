# Transformer Evolution LLM
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/transformer-evolution-llm)

An evolutionary architecture-search loop for Transformer-like language models.

Architectures are defined in typed YAML (Pydantic). The loop mutates/crosses over these specs, trains each candidate for a short budget, scores it on multiple objectives, and keeps a Pareto frontier. To keep the search practical, children can inherit weights from parents and crossover merges checkpoints when possible.

## Why This Exists

This repo is a sandbox for answering questions like: “under a fixed training recipe and constraints, what kinds of architectural motifs survive?”

It’s meant to be:
- A way to explore trade-offs (quality vs speed vs memory) without hand-writing dozens of model variants.
- A way to make results inspectable: every run emits a frontier plus lineage describing how candidates were produced.

The goal is to use laptop-scale surrogates (~65–100M params) as a fast feedback loop to discover motifs that can then be scaled up.

## Key Features

- **Typed DSL**: Architectures are defined in YAML/JSON using a strict Pydantic-based schema (`src/transformer_evolution_llm/dsl.py`), ensuring all generated candidates are valid by construction.
- **Weight Inheritance + Checkpoint-Aware Crossover**: Children can inherit parent weights; crossover attempts to merge state dicts so you don’t always restart from scratch.
- **Multi-Objective Optimization**: Maintains a Pareto frontier over metrics like perplexity, throughput, RAM proxy, and structural complexity.
- **Rich Mutation Primitives**: Includes structural mutations (add/remove blocks, split layers), component swaps (Attention ↔ MoE ↔ SSM), and hyperparameter tuning.
- **Template Learning (experimental)**: Optionally adjusts and persists mutation templates based on which template mutations improve objectives (`evolution.template_learning`, `evolution.template_learning_save_path`).
- **Graph Module Primitive (experimental)**: A built-in `graph_module` component that can be inserted as a custom extra so search can explore small operator graphs, not just fixed blocks.
- **Audit Lineage**: Every discovery comes with a full JSON lineage and visualization tools, explaining exactly *how* a specific architecture was derived.
- **NanoGPT-style benchmark path (implemented)**: A repeatable speedrun-style metric (`tokens/time to target`) for comparing training efficiency inside this repo (see `docs/nanogpt_benchmark.md`).

## What You Get From A Run

A run writes a small set of artifacts under `runs/<run_id>/`:
- `frontier.json`: the non-dominated candidates (each has a full `spec` + `metrics` + `id`).
- `lineage.json` / `frontier_lineage.json`: parent links and mutation/crossover history.
- `frontier.manifest.json`: the run recipe + environment snapshot.
- `checkpoints/` (optional): model checkpoints (often pruned to frontier-only).

See “Run Folder: What’s What” below for the full list.

## Scope (What This Is / Is Not)

- **Bounded search space:** evolution can only produce what the DSL + mutation set can express. To explore a new primitive (e.g. a new memory operator), you have to add it to the codebase first.
- **Short-budget proxies:** most metrics come from short surrogate training. Expect the frontier to move when you increase data, steps, or change objectives.
- **Not a leaderboard:** the NanoGPT-style benchmark is for repeatable *within-repo* comparisons. The micro-benchmark numbers in this README used a tiny packed OpenWebText subset; regenerate a larger cache before drawing conclusions.
- **Not interpretability tooling:** this repo can discover motifs; it does not (yet) provide mechanistic explanations for why a motif works.

<details>
<summary>Extending the search space (adding a new primitive)</summary>

At a high level:

1) Add a typed config to the DSL (`src/transformer_evolution_llm/dsl.py`).
2) Implement the module in the model code (`src/transformer_evolution_llm/models.py`).
3) Add any static checks / cost estimates (`src/transformer_evolution_llm/evaluation.py`).
4) Add a mutation operator so evolution can discover it (`src/transformer_evolution_llm/mutations.py` or `src/transformer_evolution_llm/template_mutation.py`).

</details>

## Installation

### Prerequisites
- Python 3.11+
- A compatible accelerator (CUDA, MPS, or CPU)
- `uv` (used by `make setup`; install via `brew install uv` or see [astral.sh/uv](https://astral.sh/uv/))
- [Optional] A Hugging Face token (for dataset access): `HF_TOKEN`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/strangeloopcanon/transformer-evolution-llm.git
cd transformer-evolution-llm

# 2. Create the venv + install dependencies (uses uv)
make setup
source .venv/bin/activate
```

Optional environment:

- `export HF_TOKEN="..."` (only needed for Hugging Face dataset access)
- `export TOKENIZERS_PARALLELISM=false` (avoids tokenizers fork warnings)

## Quick Start

### 10-minute tour (end-to-end)

So what: run a tiny search, inspect what survived, then rerun/benchmark one candidate.

1) Run a smoke evolution loop (CPU):

```bash
export TOKENIZERS_PARALLELISM=false
RUN="runs/live_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

python scripts/run_live.py configs/live_smoke.yaml \
  --device cpu --generations 3 --steps 40 --eval-batches 2 --seed 0 \
  --out "$RUN/frontier.json" \
  --lineage-out "$RUN/frontier_lineage.json" \
  --state-out "$RUN/frontier.state.json" \
  --checkpoint-dir "$RUN/checkpoints" \
  --prune-checkpoints-to-frontier \
  2>&1 | tee "$RUN/live.log"

python scripts/report_motifs.py "$RUN/frontier.json" --lineage "$RUN/frontier_lineage.json" --top 10
```

2) Export one candidate as a new seed and rerun it:

```bash
REPLAY="runs/replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPLAY"
# Pick an id printed by report_motifs.py above.
CANDIDATE_ID="<paste_id_here>"
python scripts/export_seed.py "$RUN/frontier.json" --id "$CANDIDATE_ID" --out-config "$REPLAY/seed.yaml"
python scripts/run_live.py "$REPLAY/seed.yaml" \
  --device cpu --generations 1 --steps 40 --eval-batches 2 --seed 0 \
  --out "$REPLAY/frontier.json" \
  --checkpoint-dir "$REPLAY/checkpoints" \
  2>&1 | tee "$REPLAY/live.log"
```

3) (Optional) Run the NanoGPT-style benchmark recipe for a single config:

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

See `docs/nanogpt_benchmark.md` for the packed-token data contract and the actual benchmark configs/budgets.

### Run a Long-Context Sweep (Mac M4 / MPS)
This matches the example long‑context frontier section below and is designed to stay disk-safe by pruning checkpoints to just the frontier.

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

Note: replace `--device mps` with `--device cuda` on NVIDIA GPUs.

Monitor the run:
```bash
tail -f "$RUN/live.log"
```

If you want to archive the discovered specs and reclaim disk immediately after the run:

```bash
python scripts/archive_run.py "$RUN" --delete-checkpoints
```

## Run Folder: What’s What

Each run directory under `runs/` is self-contained. The key artifacts are:

- `frontier.json`: The final Pareto frontier (the candidates worth looking at). Each entry includes a full `spec` (the architecture), `metrics`, and an `id`.
- `frontier_lineage.json`: The full genealogy graph (parents, mutations/crossover, checkpoints, and whether nodes completed). Use this to explain *how* an architecture was derived.
- `frontier.manifest.json`: Run metadata (config path, generations/steps, device, git commit, output paths).
- `frontier.state.json`: Internal search state (archive/map-elites bins, RNG state, etc). Useful for debugging/restarts; not usually needed for sharing results.
- `checkpoints/`: Training checkpoints (often pruned down to just frontier members if `--prune-checkpoints-to-frontier` was used).
- `live.log`: Full run log (training/eval events, errors, throughput, etc).
- `motifs.txt`: A human-readable motif summary (attempted vs completed vs frontier coverage).

Helpful commands:

```bash
# Motif summary + example IDs
python scripts/report_motifs.py runs/<RUN>/frontier.json --lineage runs/<RUN>/frontier_lineage.json --top 15

# Reclaim disk after you’ve archived what you want
python scripts/archive_run.py runs/<RUN> --delete-checkpoints
```

## System Architecture

The evolution loop follows this flow:

```mermaid
flowchart LR
    subgraph Evolution Loop
        SEED[Seed Config] --> MUT[Mutate/Crossover]
        MUT --> TRAIN[Train Rungs 0→1→2]
        TRAIN --> EVAL[Evaluate Metrics]
        EVAL --> SELECT[Pareto Selection]
        SELECT --> |Next Gen| MUT
    end
    SELECT --> FRONTIER[Pareto Frontier]
    FRONTIER --> EXPORT[Export Winners]
```

### Components

1.  **Declarative Spec (DSL)**: The `dsl.py` module defines the genome. It describes the model topology (blocks, layers, mixers) and training hyperparameters.
2.  **Evolution Loop**: `orchestrator.py` manages the population. It selects parents using strategies like weighted sampling, `pareto_uniform`, `lexicase`, and `map_elites`.
3.  **Mutation & Crossover**:
    *   `mutations.py`: Modifies the DSL (e.g., changes a layer from Dense Attention to MoE).
    *   `crossover.py`: Splices two architectures and intelligently merges their state dicts.
4.  **Evaluation (Rungs)**: Candidates pass through "rungs" of increasing cost.
    *   **Static Analysis**: Cheap checks for params/flops.
    *   **Rung 0/1**: Short training runs to filter unstable or poor-performing models.
    *   **Full Training**: Longer runs for promising candidates.
5.  **Pareto Frontier**: The system maintains a set of non-dominated solutions (e.g., best perplexity for a given parameter count) in `frontier.json`.

## Configuration Reference

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

See [`src/transformer_evolution_llm/dsl.py`](src/transformer_evolution_llm/dsl.py) for the full schema definition.

## Example Long-Context Frontier (Illustrative Survivors)

These are illustrative survivors from the newest long‑context sweep (11‑entry Pareto frontier at ~65–85M params), archived as YAML for inspection/reseeding.

Source (metrics + specs): `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/frontier_arch.json` (generated from `configs/exp_longctx_overnight_m4_full_deck.yaml`).

- **Quality‑lean memory stack (best `ppl_code`)**  
  Source: `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/duplicate_block_span+toggle_kv_policy+add_extra_combo-292-4963.yaml`.  
  - Depth: 5 blocks; Memory blocks: 5/5; MoE blocks: 1/5; KV policy: `window=1024` + `int8`.  
  - Proxy metrics: `ppl_code≈121.45`, `passkey_loss≈7.79` (≈83.4M params).

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

- **Probe‑lean hybrid (best `passkey_loss`)**  
  Source: `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/duplicate_block_span+toggle_qk_norm+add_extra_combo-91-10b3.yaml`.  
  - Depth: 10 blocks; Memory blocks: 4/10; Recurrences: 1; MLA blocks: 2/10; Selector blocks: 2/10; QK‑norm blocks: 1/10.  
  - Proxy metrics: `passkey_loss≈5.43`, `ppl_code≈194.03` (≈65.3M params).

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

- **Deeper routed memory stack (balanced quality)**  
  Source: `configs/frontiers/exp_longctx_full_deck_2h_m4_20251217_003818/insert_assoc_memory+tune_retro+tune_branch_router-375-1123.yaml`.  
  - Depth: 13 blocks; Memory blocks: 4/13; MLA blocks: 1/13; Extras: assoc‑memory + branch‑router + layer‑scale.  
  - Proxy metrics: `ppl_code≈123.58`, `passkey_loss≈7.52` (≈75.8M params).

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

## Scientific Goals & Findings

### What we set out to learn
- Can we evolve genuinely new LLM blueprints—beyond familiar transformer tweaks—using only ~100 M parameter surrogates?
- Do hybrids (retro memory + sparse attention + MoE/SSM toggles) outperform single tricks when tokens are scarce?
- Is a fully-auditable lineage enough to explain breakthroughs?

### What we're seeing so far
- **Explicit memory is stable**: When long-context probes are used, survivors almost always carry memory primitives (retro + token/chunk/assoc variants).
- **Selection pressure dominates**: Different selection strategies (`map_elites` vs. lexicase) maintain different niches.
- **Convergent Evolution**: The system reliably rediscovers known distinct classes of building blocks (explicit memory, routing/gating, depth reuse) without being explicitly told to look for them.

### So what (what this implies)
- These runs are an *architecture microscope*: at ~65–85M params and a few hundred steps, the loop finds convergent motifs (memory, routing/gating, depth reuse) without being told to chase any named target.
- The frontier is a function of constraints: rung0 gates (params/KV/throughput + optional minima) decide what survives long enough to train; objective weights decide what gets rewarded.
- When a motif appears in lineage but not the frontier, it's usually filtered by instability (NaNs) or rung0 gates before it can pay off; relaxing gates or increasing rung budgets changes the reachable regime.

For more detailed takeaways, see [docs/evolution_takeaways.md](docs/evolution_takeaways.md).

## Project Structure

The codebase is separated into the core library (`src/`) and execution scripts (`scripts/`).

```
├── configs/                  # YAML configurations for evolution experiments
├── docs/                     # Documentation and deeper insights
├── scripts/
│   ├── run_live.py           # Main entry point for evolutionary runs
│   ├── run_ablation.py       # Tool to ablate specific features from a candidate
│   ├── export_seed.py        # Export a winner as a new seed for future runs
│   └── ...
├── src/transformer_evolution_llm/
│   ├── dsl.py                # The architecture specification language (Pydantic)
│   ├── models.py             # PyTorch implementations of evolvable blocks
│   ├── orchestrator.py       # Evolution engine (population, selection, frontier)
│   ├── trainer.py            # Training loop with weight inheritance
│   ├── mutations.py          # Genetic operators (modify DSL)
│   └── ...
└── tests/                    # Pytest suite
```

## Usage & Workflows

### Inspecting Results
The output JSON files contain the full frontier. You can inspect the lineage or export specific candidates.

```bash
# Export the best candidate as a new seed configuration
python scripts/export_seed.py runs/<run>/frontier.json \
  --id <candidate_id> \
  --out-config configs/seed_winner.yaml
```

### Reproducing a Run
To reproduce a specific architecture, export its spec (and optionally its checkpoint) and rerun a short sweep from that seed. The runner always evaluates the seed first before spawning children.

```bash
RUN="runs/replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
python scripts/run_live.py configs/seed_winner.yaml \
  --device mps --generations 1 --steps 240 --eval-batches 4 \
  --out "$RUN/frontier.json" --checkpoint-dir "$RUN/checkpoints" \
  2>&1 | tee "$RUN/live.log"
```

### Disk Hygiene
Runs can accumulate checkpoints quickly. Useful cleanup tools:

```bash
# Archive a run's frontier specs to configs/frontiers/<run_name>/ and delete checkpoints.
python scripts/archive_run.py runs/<run_dir> --delete-checkpoints

# Downcast checkpoints to shrink disk usage (fp16 works well on MPS).
evo-loop convert-checkpoints runs/<run_dir>/checkpoints --dtype fp16 --apply
```

## What's Next / Roadmap

### Scale-Hop Plans
The current phase runs on laptop-scale surrogates (~65–100M parameters). The next steps:
1. **350M–1B sanity runs** on GPU (see [docs/gpu_run_plan.md](docs/gpu_run_plan.md))—validate that motifs discovered at small scale transfer.
2. **Speedrun-style eval (implemented)**: Measure time/tokens-to-target under a NanoGPT-like recipe, so architectures are judged on training efficiency. Next: scale the packed-token cache and calibrate targets so this metric has useful dynamic range (see `docs/nanogpt_benchmark.md`).
3. **Multi-GPU evolution** with ZeRO-1 or FSDP for larger populations.

### Architectures of Interest

The DSL includes knobs inspired by recent architecture work:

| Architecture | Key Ideas | DSL Components |
|--------------|-----------|----------------|
| [**Titans**](https://arxiv.org/abs/2501.00663) (Google) | Neural long-term memory, attention-as-memory | `retro`, `assoc_memory`, `memory_tokens` |
| [**DeepSeek MoE**](https://arxiv.org/abs/2401.06066) | Fine-grained experts, shared expert routing | `moe` FFN, `router` configs, `shared` experts |
| [**Mamba/SSM hybrids**](https://arxiv.org/abs/2312.00752) | State-space models mixed with attention | `ssm` blocks, `toggle_ssm` mutations |
| [**MLA (Multi-head Latent Attention)**](https://arxiv.org/abs/2405.04434) | Latent KV compression | `MLA` attention kind, `kv_policy` |

Evolution can recombine these ingredients under different constraints/objectives.

## Contributing

We welcome contributions! Please see [AGENTS.md](AGENTS.md) for our operational guidelines for autonomous agents, which also serve as good best practices for human contributors.

### Development Commands
- **Setup**: `make setup`
- **Format + lint + types + security**: `make check`
- **Tests**: `make test`

---

## Appendix

<details>
<summary>Glossary & Key Concepts</summary>

### Core Terms

| Term | Definition |
|------|------------|
| **DSL** | Domain-Specific Language—the typed YAML/JSON schema for specifying architectures. See `dsl.py`. |
| **Rungs** | Tiered evaluation budgets. Rung 0 = static analysis (cheap), Rung 1 = short training, Rung 2 = full training. Candidates are promoted through rungs if they pass gates. |
| **Pareto Frontier** | The set of non-dominated solutions across multiple objectives. A candidate is on the frontier if no other candidate beats it on *all* objectives. |
| **Lamarckian Inheritance** | Children inherit trained weights from parents (not just the genome). This dramatically reduces compute since we don't train from scratch. |
| **Mutations** | Genetic operators that modify architecture specs: `dense_to_moe`, `add_recurrence`, `toggle_ssm`, etc. See `mutations.py`. |
| **Crossover** | Splicing two parent architectures to create a child with features from both. Includes intelligent weight merging. |

### Architecture Components

| Component | Description |
|-----------|-------------|
| **MoE (Mixture of Experts)** | FFN with multiple "expert" sub-networks; a router selects top-k experts per token. |
| **SSM (State Space Model)** | Mamba-style recurrent layers as an alternative to attention. |
| **MLA (Multi-head Latent Attention)** | Attention with compressed/latent KV projections (DeepSeek-style). |
| **Retro** | Retrieval-augmented memory that caches and retrieves past hidden states. |
| **Selectors** | Sparse attention variants that select a subset of tokens to attend to. |

</details>

<details>
<summary>CLI Reference (evo-loop)</summary>

The package installs an `evo-loop` CLI tool with several subcommands:

```bash
# Convert checkpoints to a different dtype (e.g., fp16 to save disk)
evo-loop convert-checkpoints runs/<run>/checkpoints --dtype fp16 --apply

# Other commands (run with --help for details)
evo-loop --help
```

### Main Script: `scripts/run_live.py`

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

</details>

<details>
<summary>Troubleshooting</summary>

### Common Issues

**Tokenizer warnings (`TOKENIZERS_PARALLELISM`)**
```bash
export TOKENIZERS_PARALLELISM=false
```

**Out of Memory (OOM)**
- Reduce `batch_size` in config
- Enable `grad_ckpt: true` in train config
- Use `--device cpu` for smoke tests

**NaN losses / Instability**
- Lower `instability_threshold` in train config to catch unstable models earlier
- Check that learning rate isn't too high
- Some mutations produce unstable architectures—this is expected; they get filtered

**MPS (Apple Silicon) quirks**
- Some operations fall back to CPU; this is normal
- FlashAttention doesn't work on MPS; uses PyTorch SDPA instead
- fp16 checkpoints work well for disk savings

**CUDA out of memory**
- Reduce population size
- Use gradient checkpointing
- Try smaller `max_tokens` budget

### Debugging tips
- Check `runs/<run>/live.log` for detailed output
- Use `--generations 1` to test a single generation
- The `stop_reason` field in frontier.json indicates why training stopped (0=normal, 1=instability, 3=early stop)

</details>

<details>
<summary>Deep-Dive Documentation</summary>

Additional documentation in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [evolution_takeaways.md](docs/evolution_takeaways.md) | Detailed lessons learned from multiple evolutionary sweeps—how to set up evolution to discover specific architecture families. |
| [gpu_run_plan.md](docs/gpu_run_plan.md) | Playbook for running evolution on GPU-rich machines (A100, 4090) with larger models (350M–1B params). |
| [modal_run.md](docs/modal_run.md) | How to run evolution and benchmarks on Modal GPUs with volume setup. |
| [nanogpt_benchmark.md](docs/nanogpt_benchmark.md) | NanoGPT-style benchmark contract (packed OpenWebText + HF mix) and speedrun metrics. |
| [scale_policy.md](docs/scale_policy.md) | Soft priors and scaling policies for moving from laptop surrogates to production scale. |

</details>

<details>
<summary>Operator Notes (Scale, Knobs, Ablations)</summary>

### Scale & portability
The current phase runs on single-machine surrogates (~100 M parameters). To scale:
1. Swap in a bigger spec in `configs/`.
2. Keep `grad_ckpt` on.
3. Re-tune `--score-weight-*` for production priorities.

### NanoGPT-style benchmark (implemented)

So what: we now have a fixed, repeatable benchmark path (packed OpenWebText + HF mix) that logs *time/tokens to target* alongside `val_ppl`, so architectures can be compared on training efficiency, not just short-run perplexity.

Use `docs/nanogpt_benchmark.md` for the contract and commands.

*Update (Jan 2026):* We now log a compute proxy (`speedrun_flops_to_target`) in addition to `speedrun_tokens_to_target`, and `tokens_to_target` is interpolated between eval points (less discretization than the old “bucketed” counts).

On Modal (A10G) using the non-toy packed OpenWebText stream (`openwebtext_10m`, 10M train / 1M val tokens), a speedrun-style objective can find large early-learning wins under short budgets. Example: with a calibrated target (`speedrun_target_ppl=2500`, eval interval 4), the best NanoGPT-objective run hit the target at **40,960 tokens vs 57,344** for the seed and also improved short-budget `ppl_code` (~769 vs ~1616). This is still an early-convergence proxy, not a claim about scaled training.

### Sparse attention patterns
The DSL supports `sparsity: none|sliding|block|local_global|dilated|local_block`.
- `local_global` combines a local window with periodic global tokens.
- `dilated` allows attention to tokens that share the same index mod `dilation`.

### Optimizers
You can switch optimizers via the DSL:
```yaml
optimizer:
  name: lion    # or adamw
  lr: 3.0e-4
  betas: [0.9, 0.99]
```

### Scaling tools
Fit scaling-law priors from existing runs:
```bash
python scripts/fit_scaling.py runs/<run_1>/frontier.json runs/<run_2>/frontier.json
```

</details>

<details>
<summary>Run History & Evolution Log</summary>

### Local Artifacts

| Run | Config | Frontier Size | Notable Findings |
|-----|--------|---------------|------------------|
| `frontier_phi_seeded128` | `seed_xover-48-9237.yaml` (128 gens) | 25 | Shallow retro-heavy stacks; `ppl_code≈1.0` |
| `frontier_phi_gated128` | `live_phi_tiny.yaml` (128 gens) | 1 | Deep (12 layers) but unstable |
| `frontier_phi_entropy_v2` | `seed_xover-48-9237.yaml` (160 gens) | 99 | Balanced mix of shallow retro and deep MoE/SSM hybrids (up to 30 layers) |

### Modal Runs (NanoGPT objective)

| Run | Config | Frontier Size | Notable Findings |
|-----|--------|---------------|------------------|
| `modal_nanogpt_speedrun_long1` | `exp_nanogpt_speedrun_owt.yaml` (24 gens, 120 steps) | 7 | Dense MHA stacks (9–12 layers); no MoE/SSM/retro; KV window + Alibi variants |
| `modal_nanogpt_speedrun_long3` | `exp_nanogpt_speedrun_owt.yaml` (48 gens, 180 steps) | 8 | Dense MHA stacks with layer-count shifts; no MoE/SSM/retro in frontier; small toggles (Alibi/precision/graph) |
| `modal_nanogpt_speedrun_valfix_full1` | `exp_nanogpt_speedrun_owt.yaml` (48 gens, 1000 steps) | 1 | Mixed `kv_groups` + memory extras + `branch_router` hit target at 40,960 vs 61,440 tokens on tiny OWT subset. |
| `modal_nanogpt_speedrun_owt10m_full1` | `exp_nanogpt_speedrun_owt_10m.yaml` (48 gens, 240 steps) | 1 | 10M/1M packed OWT: winner adds 1× MLA block + one Alibi block; hits target at 40,960 vs 61,440 tokens and improves `ppl_code` vs seed. |
| `modal_nanogpt_speedrun_owt10m_dyn1` | `exp_nanogpt_speedrun_owt_10m.yaml` (96 gens, 360 steps) | 1 | 10M/1M packed OWT: calibrated target (2.5k) yields multiple token buckets; winner improves `ppl_code` strongly but is slower (throughput trade-off). |
| `modal_deepseek_style_owt10m_dyn1` | `exp_deepseek_style_owt_10m.yaml` (96 gens, 360 steps) | 11 | DeepSeek-style pressure (KV bytes + throughput + selector): frontier contains MLA/GQA variants that reduce `kv_bytes_per_token` while keeping throughput high. |

### Architecture Highlights (Historical)

| Theme | What we learned |
|-------|-----------------|
| **Triple-retro loops** | Best single-block models carry three separate retro rails feeding the same residual |
| **MoE + SSM hybrids** | 5–6 block stacks with dual MoE cores and Mamba SSMs reach ppl≈1.87 |
| **Checkpoint pruning** | Old checkpoints are removed automatically; long sweeps no longer eat disk |

### Historical Runs (Referenced in earlier versions)

- `frontier_phi_creative_canon.json` – Composite objective + throughput experiments
- `frontier_phi_creative_super_recur_mps.json` – Deep recurrence sweeps
- `frontier_phi_promotion_mps.json` – Promotion rung experiments
- Various Pareto/lexicase sweeps with diverse hybrids (MoE + SSM + retro + sparsity + recurrence)

</details>
