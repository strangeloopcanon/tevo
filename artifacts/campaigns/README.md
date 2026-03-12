# Campaign Artifacts

So what: this directory is for small, reviewable campaign bundles derived from existing TEVO runs. It is not for raw run directories, checkpoints, or large logs.

Each lane bundle should contain only compact derived files such as:

- `manifest.json`
- `summary.json`
- `frontier_top.json`
- `lineage_summary.json`
- `champion_spec.yaml`
- optional `champion.train_recipe.yaml`

Keep `runs/` and checkpoints local. This campaign layer is derived from the normal TEVO run artifacts, not a second run system. Only commit the bundle that another collaborator can review or aggregate.
