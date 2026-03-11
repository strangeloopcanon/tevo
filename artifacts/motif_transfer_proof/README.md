# Motif Transfer Proof Artifacts

So what: this directory contains the small, stable artifact bundle behind the current TEVO -> projected CUDA `autoresearch` proof-of-concept.

The tracked JSON summaries use a display-oriented upstream commit field (`repo_ref_display`) so repo secret scanning does not mistake the pinned git SHA for a leaked credential. The full commit is documented in the top-level docs.

## Contents

- `baseline_summary.json`
  The stock upstream CUDA `autoresearch` baseline summary on the pinned commit.
- `full_scale_probe_summary.json`
  The first direct transfer summary, which shows why full-model transfer was the wrong unit.
- `projected_probe_summary.json`
  The successful projected motif-transfer summary.
- `tune_attn_sparsity.train_recipe.yaml`
  The exported `TrainRecipe` for the frontier sibling that drove the proof.
- `tune_attn_sparsity.projected.train.py`
  The rendered upstream-style CUDA `train.py` after compatibility projection.

## Provenance

- Upstream repo: `https://github.com/karpathy/autoresearch.git`
- Upstream commit: `c12eef778edafc89cd7ce036a7f500ddb5397a65`
- Discovery family: nanochat-aligned TEVO run on Modal
- Transfer lane: projected CUDA `autoresearch` probe on `A100-80GB`

See [docs/motif_transfer_demo.md](../../docs/motif_transfer_demo.md) for the explanation of how these files fit together.
