# Optimizer & Method Discovery

How to use the evolution loop for optimizer-only search and wide cross-family method invention.

## Optimizer Rediscovery Protocol (Optimizer-Only Search)

Run a constrained discovery mode where architecture stays fixed and evolution explores compositional optimizer recipes (base family + gradient transforms + update filtering).

- Config: `configs/exp_optimizer_discovery_owt_10m_v1.yaml`
- Search is enforced by `evolution.mutation_allowlist` (not just weight biasing).
- Primary objective is compute-to-target (`speedrun_flops_to_target`) with quality guardrails.

Local smoke:
```bash
python scripts/run_live.py configs/exp_optimizer_discovery_owt_10m_v1.yaml \
  --device mps --generations 4 --steps 80 --eval-batches 2 --seed 0 \
  --out runs/optdisc_smoke_seed0/frontier.json \
  --lineage-out runs/optdisc_smoke_seed0/frontier_lineage.json
```

Motif aggregation across runs:
```bash
python scripts/report_optimizer_motifs.py \
  --frontier runs/modal/modal_optdisc_v1_seed0/frontier.json \
  --frontier runs/modal/modal_optdisc_v1_seed1/frontier.json \
  --frontier runs/modal/modal_optdisc_v1_seed2/frontier.json \
  --out runs/modal/optdisc_v1_report.json
```

## Wide Method Discovery (Cross-Family Invention)

Instead of optimizing a single known trick, this mode pushes evolution to compose new methods across optimizer recipe, attention, memory modules, routing, and recurrence.

- Config: `configs/exp_method_discovery_openwebtext_exec_v1.yaml`
- Core mutation: `mix_method_recipe` (multi-step cross-family composition)
- Selection balances quality + compute + novelty (`ppl_code`, `speedrun_flops_to_target`, `novelty`, `graph_entropy`, `throughput`, `instability`).

Local smoke:
```bash
python scripts/run_live.py configs/exp_method_discovery_openwebtext_exec_v1.yaml \
  --device mps --generations 2 --steps 24 --eval-batches 1 --seed 7 \
  --out runs/methoddisc_smoke_seed7/frontier.json \
  --lineage-out runs/methoddisc_smoke_seed7/frontier_lineage.json
```

## Method Discovery Proof (Feb 18, 2026)

This repo can run open-ended method search (not a paper copy) and discover multiple improved training-method motifs under one shared setup.

### Repro (3 seeds, same config, same budget)

```bash
for seed in 0 1 2; do
  python scripts/run_live.py configs/exp_method_discovery_openwebtext_exec_v1.yaml \
    --device mps --generations 8 --steps 80 --eval-batches 1 --seed "$seed" \
    --out "runs/methoddisc_v1_seed${seed}/frontier.json" \
    --lineage-out "runs/methoddisc_v1_seed${seed}/frontier_lineage.json" \
    --checkpoint-dir "runs/methoddisc_v1_seed${seed}/checkpoints" \
    --no-cleanup-old-checkpoints --no-prune-checkpoints-to-frontier
done

python scripts/report_optimizer_motifs.py \
  --frontier runs/methoddisc_v1_seed0/frontier.json \
  --frontier runs/methoddisc_v1_seed1/frontier.json \
  --frontier runs/methoddisc_v1_seed2/frontier.json \
  --out runs/methoddisc_v1_3seed_optimizer_motifs.json
```

### Results (`ppl_code`, lower is better)

| Seed | Baseline | Best candidate | Best `ppl_code` | Delta vs seed |
|------|----------|----------------|-----------------|---------------|
| 0 | `seed-1-edf6` (`2610.58`) | `mix_optimizer_recipe-9-b961` | `1814.18` | `-30.51%` |
| 1 | `seed-1-cc63` (`2632.07`) | `tune_optimizer-7-c9fc` | `2277.05` | `-13.49%` |
| 2 | `seed-1-da45` (`2617.77`) | `tune_update_filter_ratio-6-5560` | `2290.58` | `-12.50%` |

### What emerged

- A mask-style motif recurred in seed 1/2: `update_filter.mode=bernoulli`, `keep_ratio=0.5` (strong gain over seed).
- The best single candidate was a different method family: `gradient_transform=orthogonalize_2d` (seed 0), showing this search can discover beyond one expected trick.
- `mix_method_recipe` was high-variance in this run (4 failures), which is useful signal for where to add stability guards in future sweeps.

### Summary

We used our own DSL + evolution loop to discover improved training methods in-repo. On 3 independent seeds (same config, 8 generations, 80 steps), best candidates improved `ppl_code` by -30.5%, -13.5%, and -12.5% vs each seed baseline. Two distinct motifs emerged: (1) masked-update optimizer behavior (`bernoulli`, `keep_ratio=0.5`) repeated across seeds, and (2) a different high-performing family (`orthogonalize_2d` gradient transform) produced the best overall gain. This is short-budget evidence of discovery ability, not a claim of universal optimality.

## Scale & Portability

The current phase runs on single-machine surrogates (~100M parameters). To scale:
1. Swap in a bigger spec in `configs/`.
2. Keep `grad_ckpt` on.
3. Re-tune `--score-weight-*` for production priorities.

### NanoGPT-style benchmark notes

We now have a fixed, repeatable benchmark path (packed OpenWebText + HF mix) that logs *time/tokens to target* alongside `val_ppl`, so architectures can be compared on training efficiency, not just short-run perplexity. See [nanogpt_benchmark.md](nanogpt_benchmark.md) for the contract and commands.

*Update (Jan 2026):* We now log a compute proxy (`speedrun_flops_to_target`) in addition to `speedrun_tokens_to_target`, and `tokens_to_target` is interpolated between eval points (less discretization than the old "bucketed" counts).

On Modal (A10G) using the non-toy packed OpenWebText stream (`openwebtext_10m`, 10M train / 1M val tokens), a speedrun-style objective can find large early-learning wins under short budgets. Example: with a calibrated target (`speedrun_target_ppl=2500`, eval interval 4), the best NanoGPT-objective run hit the target at **40,960 tokens vs 57,344** for the seed and also improved short-budget `ppl_code` (~769 vs ~1616). This is still an early-convergence proxy, not a claim about scaled training.

### Scaling tools

Fit scaling-law priors from existing runs:
```bash
python scripts/fit_scaling.py runs/<run_1>/frontier.json runs/<run_2>/frontier.json
```
