# TrainRecipe Bridge

So what: TEVO can now export a constrained `TrainRecipe` artifact from compatible frontier candidates and render that recipe into the current `autoresearch` CUDA `train.py`, the matching `autoresearch@home` CUDA `train.py`, or the `autoresearch-mlx` `train.py` layout without hand-editing the file every time. The bridge now preserves downstream training defaults unless a recipe explicitly overrides them, and oversized CUDA recipes are projected into the upstream-size CUDA envelope so motif-transfer probes stay comparable to stock `train.py`.

## What The Bridge Owns

The v1 bridge owns only the train.py-safe architecture surface:

- global depth / model width / head geometry
- global `n_kv_head` / GQA-style attention layout
- shared `S`/`L` window pattern
- shared MLP hidden size + activation
- global norm kind
- QK-norm toggle
- backend-safe value-embedding toggles when explicitly set in a recipe

It does **not** try to render TEVO-only primitives such as retro, selectors, recurrence, MoE, MLA, or custom modules into downstream `train.py`.

## CUDA Compatibility Projection

When rendering for `autoresearch_cuda`, the bridge now checks whether a recipe exceeds the safe upstream envelope. If it does, the renderer down-projects the recipe before patching `train.py`:

- depth is capped to the upstream CUDA scaffold
- model width is capped to the stock CUDA width budget
- MLP width is capped to the stock CUDA hidden-size budget
- GQA layout is projected onto the safe head geometry
- the window pattern is compressed to the projected depth, preserving late-layer long-context motifs when possible

This keeps the downstream probe focused on **motif transfer** instead of accidentally benchmarking a much larger model than stock upstream `autoresearch`.

## Export A Recipe

Single compatible candidate from a frontier:

```bash
evo-loop train-recipe-export runs/<run>/frontier.json \
  --candidate-id <candidate_id> \
  --out runs/<run>/<candidate_id>.train_recipe.yaml
```

Shortlist the top three compatible candidates:

```bash
evo-loop train-recipe-export runs/<run>/frontier.json \
  --top-k 3 \
  --metric ppl_code \
  --out runs/<run>/train_recipes
```

Notes:

- Incompatible frontier entries are skipped automatically.
- When TEVO cannot derive a downstream knob confidently, the recipe leaves it empty and the renderer falls back to the target backend's current default value.
- `train_recipe_from_spec(...)` now leaves optimizer and batch-size fields empty on purpose; this keeps exported CUDA recipes aligned with upstream `autoresearch` defaults instead of importing TEVO's internal microbatch or LR scales.

## Render Into Downstream `train.py`

Patch a CUDA `autoresearch/train.py` checkout:

```bash
evo-loop train-recipe-render runs/<run>/<candidate_id>.train_recipe.yaml \
  --backend autoresearch_cuda \
  --train-py /path/to/autoresearch/train.py
```

Patch a CUDA `autoresearch@home/train.py` checkout:

```bash
evo-loop train-recipe-render runs/<run>/<candidate_id>.train_recipe.yaml \
  --backend autoresearch_at_home_cuda \
  --train-py /path/to/autoresearch-at-home/train.py
```

Patch an Apple Silicon `autoresearch-mlx/train.py` checkout:

```bash
evo-loop train-recipe-render runs/<run>/<candidate_id>.train_recipe.yaml \
  --backend autoresearch_mlx \
  --train-py /path/to/autoresearch-mlx/train.py
```

If you omit `--train-py`, the command prints the TEVO-owned template zones instead of patching a file.

`autoresearch_cuda` and `autoresearch_at_home_cuda` currently share the same renderer logic and projection rules. For oversized CUDA recipes, the rendered file reflects the projected downstream-safe scaffold rather than the raw TEVO model scale.

## Stable Template Zones

Rendered files now contain TEVO-owned marker blocks such as:

- `TEVO TRAIN RECIPE: CONSTANTS`
- `TEVO TRAIN RECIPE: NORM`
- `TEVO TRAIN RECIPE: VALUE_EMBED`
- `TEVO TRAIN RECIPE: MLP`
- `TEVO TRAIN RECIPE: QK_NORM`
- `TEVO TRAIN RECIPE: MODEL_CONFIG`

This keeps the runtime plumbing in the downstream repo intact while giving TEVO stable regions to update on later iterations.

## Search Preset

Use `configs/exp_train_recipe_bridge_owt_10m_v1.yaml` for the first shared-family search pass.

That preset keeps the search inside recipe-safe global mutations:

- global attention-shape edits
- global window-pattern edits
- global FFN edits
- global norm / QK-norm toggles
- depth changes

Crossover is disabled there on purpose so TEVO does not synthesize per-block hybrids that the shared bridge cannot render faithfully.
