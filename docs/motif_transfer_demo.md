# Motif Transfer Demo

So what: this repo now has a clean proof that TEVO can discover a motif under a cheap proxy benchmark, export it as a `TrainRecipe`, project it into the upstream CUDA `autoresearch` size envelope, and improve stock `train.py` on the real 5-minute benchmark.

## The Result

The current proof-of-concept uses:

- a nanochat-aligned TEVO discovery run on Modal
- the frontier sibling `tune_attn_sparsity-5-ecda`
- upstream `autoresearch` commit `c12eef778edafc89cd7ce036a7f500ddb5397a65`
- a single `A100-80GB` CUDA probe

| Probe | Result |
|-------|--------|
| Stock upstream CUDA `autoresearch` baseline | `val_bpb = 1.117953` |
| Projected TEVO motif-transfer probe | `val_bpb = 1.113392` |
| Delta | `-0.004561` better |
| Peak VRAM | `43.96 GiB -> 35.89 GiB` |

This is not a claim that TEVO has broadly beaten `autoresearch`. It is a narrower and more useful claim: **the TEVO -> TrainRecipe -> upstream `train.py` bridge can transfer a discovered motif usefully when scale is normalized.**

## What Failed First

The first direct transfer was honest, but it was the wrong transfer unit.

The nanochat frontier sibling rendered into upstream CUDA `train.py` as a model with `519,048,360` parameters. That version OOMed on `A100-80GB`, so it was not a meaningful comparison to stock upstream `autoresearch`.

That failure told us the bridge should not transfer full model scale. It should transfer the motif.

## What Changed

The CUDA renderer now projects oversized recipes into the upstream `autoresearch` size envelope before patching `train.py`.

For this probe, the discovered motif was kept while the scale was normalized into a downstream-safe scaffold:

- `DEPTH = 8`
- `MODEL_DIM = 512`
- `N_HEAD = 4`
- `N_KV_HEAD = 4`
- `MLP_HIDDEN = 2048`
- `WINDOW_PATTERN = "SSSSLSLL"`

The important part is that the transfer preserved the discovered locality pattern instead of trying to preserve every upstream TEVO dimension.

## Why This Matters

This result is the first clean evidence that the new repo direction is worth sharing:

- TEVO is not just generating abstract YAML that never touches real code.
- `TrainRecipe` is not just a serialization format; it is the bridge between structured search and real downstream `train.py`.
- The right transfer primitive is a motif, not a whole model scale.

That is the design split we wanted:

- **TEVO** searches broadly in a typed space.
- **TrainRecipe** carries only the shared, renderer-safe knobs.
- **`autoresearch`** judges whether the motif survives contact with a real training script.

## Current Boundaries

Be explicit about what this proof does and does not establish.

- It is one successful projected probe, not yet a repeated three-run median.
- It uses a pinned upstream `autoresearch` commit, not a moving target.
- It shows motif transfer into CUDA `autoresearch`, not yet a full TEVO-seeded overnight `autoresearch` hill-climb.
- It does not claim direct comparability between TEVO proxy metrics and downstream `val_bpb`.

Those are good next steps, but they are not required for the repo to be understandable or interesting today.

## Where To Look

- [README.md](../README.md) for the top-level repo story
- [docs/train_recipe_bridge.md](train_recipe_bridge.md) for the bridge rules and projection behavior
- [docs/cuda_transfer_demo.md](cuda_transfer_demo.md) for the Modal CUDA workflow

## Proof Artifacts

The small curated proof bundle lives under [artifacts/motif_transfer_proof](../artifacts/motif_transfer_proof/README.md):

- [baseline_summary.json](../artifacts/motif_transfer_proof/baseline_summary.json)
- [full_scale_probe_summary.json](../artifacts/motif_transfer_proof/full_scale_probe_summary.json)
- [projected_probe_summary.json](../artifacts/motif_transfer_proof/projected_probe_summary.json)
- [tune_attn_sparsity.train_recipe.yaml](../artifacts/motif_transfer_proof/tune_attn_sparsity.train_recipe.yaml)
- [tune_attn_sparsity.projected.train.py](../artifacts/motif_transfer_proof/tune_attn_sparsity.projected.train.py)
