# Transformer Evolution LLM
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/transformer-evolution-llm)

An evolutionary architecture-search loop for Transformer-like language models.

Architectures are defined in typed YAML (Pydantic). The loop mutates/crosses over these specs, trains each candidate for a short budget, scores it on multiple objectives, and keeps a Pareto frontier. Children inherit weights from parents and crossover merges checkpoints when possible.

## Punchline

Purely behavioral selection (loss + memory/speed + novelty/entropy) repeatedly discovers *embedding-conditioned FFNs* -- FFNs that read token embeddings instead of the residual stream. This trait was not an explicit objective. See [docs/example_frontiers.md](docs/example_frontiers.md) for full metrics and repro commands.

## Why This Exists

This repo is a sandbox for answering: "under a fixed training recipe and constraints, what kinds of architectural motifs survive?"

- Explore trade-offs (quality vs speed vs memory) without hand-writing dozens of model variants.
- Make results inspectable: every run emits a frontier plus lineage describing how candidates were produced.

The goal is to use laptop-scale surrogates (~65-100M params) as a fast feedback loop to discover motifs that can then be scaled up.

## Installation

**Prerequisites:** Python 3.11+, a compatible accelerator (CUDA / MPS / CPU), and [`uv`](https://astral.sh/uv/).

```bash
git clone https://github.com/strangeloopcanon/transformer-evolution-llm.git
cd transformer-evolution-llm
make setup
source .venv/bin/activate
```

Optional: `export HF_TOKEN="..."` for Hugging Face dataset access, `export TOKENIZERS_PARALLELISM=false` to suppress tokenizer fork warnings.

## Quick Start

Run a smoke evolution loop (CPU, ~2 minutes):

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

python scripts/report_motifs.py "$RUN/frontier.json" \
  --lineage "$RUN/frontier_lineage.json" --top 10
```

Then export a winner and rerun it:

```bash
CANDIDATE_ID="<paste_id_from_report>"
python scripts/export_seed.py "$RUN/frontier.json" \
  --id "$CANDIDATE_ID" --out-config configs/seed_winner.yaml
python scripts/run_live.py configs/seed_winner.yaml \
  --device cpu --generations 1 --steps 40 --eval-batches 2 --seed 0 \
  --out runs/replay/frontier.json
```

See [docs/configuration_guide.md](docs/configuration_guide.md) for longer sweep commands (MPS, CUDA, Modal).

## What You Get From A Run

Each run writes artifacts under `runs/<run_id>/`:

| File | Purpose |
|------|---------|
| `frontier.json` | Non-dominated candidates (spec + metrics + id) |
| `frontier_lineage.json` | Genealogy graph (parents, mutations, crossover reports) |
| `frontier.manifest.json` | Run recipe + environment snapshot |
| `checkpoints/` | Model checkpoints (often pruned to frontier-only) |
| `live.log` | Full run log |

## System Architecture

```mermaid
flowchart LR
    subgraph evo [Evolution Loop]
        SEED[Seed Config] --> MUT[Mutate/Crossover]
        MUT --> TRAIN[Train Rungs 0-1-2]
        TRAIN --> EVAL[Evaluate Metrics]
        EVAL --> SELECT[Pareto Selection]
        SELECT --> |Next Gen| MUT
    end
    SELECT --> FRONTIER[Pareto Frontier]
    FRONTIER --> EXPORT[Export Winners]
```

| Component | What it does |
|-----------|-------------|
| **DSL** (`dsl.py`) | Typed architecture genome (blocks, layers, mixers, training hparams) |
| **Orchestrator** (`orchestrator.py`) | Population management, selection (`map_elites`, `lexicase`, etc.) |
| **Mutations** (`mutations.py`) | Genetic operators that modify specs (grow, shrink, toggle) |
| **Crossover** (`crossover.py`) | Splice two architectures + merge checkpoints |
| **Evaluation** (`evaluation.py`) | Runged filtering: static analysis, short training, full training |

## Project Structure

```
configs/                          YAML experiment configurations
docs/                             Deep-dive documentation
scripts/
    run_live.py                   Main entry point for evolution
    export_seed.py                Export a winner as a new seed
    report_motifs.py              Motif analysis
    run_benchmark.py              NanoGPT-style benchmark
    archive_run.py                Archive runs and reclaim disk
src/transformer_evolution_llm/
    dsl.py                        Architecture specification (Pydantic)
    models.py                     PyTorch block implementations
    orchestrator.py               Evolution engine
    trainer.py                    Training loop with weight inheritance
    mutations.py                  Genetic operators
    crossover.py                  Crossover + checkpoint merge
tests/                            Pytest suite
```

## Scope

- **Bounded search space:** evolution can only produce what the DSL + mutation set can express. To explore a new primitive, add it to the codebase first.
- **Short-budget proxies:** metrics come from short surrogate training. The frontier moves when you increase data, steps, or change objectives.
- **Not a leaderboard:** the NanoGPT-style benchmark is for repeatable *within-repo* comparisons.

## Documentation

| Topic | Document |
|-------|----------|
| Configuration, objectives, selection profiles | [docs/configuration_guide.md](docs/configuration_guide.md) |
| Architecture comparison (GPT-2 vs nanochat vs TEVO), glossary | [docs/architecture_comparison.md](docs/architecture_comparison.md) |
| Example frontier survivors | [docs/example_frontiers.md](docs/example_frontiers.md) |
| Findings & operational lessons | [docs/evolution_takeaways.md](docs/evolution_takeaways.md) |
| Optimizer & method discovery | [docs/optimizer_method_discovery.md](docs/optimizer_method_discovery.md) |
| Run history & evolution log | [docs/run_history.md](docs/run_history.md) |
| Running on Modal GPUs | [docs/modal_run.md](docs/modal_run.md) |
| GPU scale-up plan (350M-1B) | [docs/gpu_run_plan.md](docs/gpu_run_plan.md) |
| NanoGPT-style benchmark contract | [docs/nanogpt_benchmark.md](docs/nanogpt_benchmark.md) |
| Scaling policies | [docs/scale_policy.md](docs/scale_policy.md) |
| nanochat alignment notes | [docs/nanochat_alignment.md](docs/nanochat_alignment.md) |
| CLI reference & workflows | [docs/cli_reference.md](docs/cli_reference.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |

## Roadmap

1. **350M-1B sanity runs** on GPU -- validate that motifs discovered at small scale transfer. See [docs/gpu_run_plan.md](docs/gpu_run_plan.md).
2. **Speedrun-style eval** -- scale the packed-token cache and calibrate targets for useful dynamic range. See [docs/nanogpt_benchmark.md](docs/nanogpt_benchmark.md).
3. **Multi-GPU evolution** with ZeRO-1 or FSDP for larger populations.

## Contributing

We welcome contributions! See [AGENTS.md](AGENTS.md) for operational guidelines.

```bash
make setup       # Create venv + install deps
make check       # Format + lint + types + security
make test        # Run test suite
```
