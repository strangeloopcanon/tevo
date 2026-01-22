# Modal GPU runs

This repo includes a small Modal harness to run `scripts/run_live.py` on a single GPU and persist outputs in Modal Volumes.

## Run
```bash
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_live.py \
  --config-path configs/exp_longctx_overnight_m4_full_deck_template_learning.yaml \
  --generations 96 --steps 720 --eval-batches 8 \
  --download
```

This writes `frontier.json` (+ `frontier.state.json`, and optional `lineage.json`) into a persisted volume, and (with `--download`) downloads them into `runs/modal/<run_id>/` locally.

<details>
<summary>Configuration knobs</summary>

- GPU selection: `TEVO_MODAL_GPU` (e.g. `A10G`, `A100`, `H100`)
- Torch wheel source: `TEVO_TORCH_INDEX_URL` (default: `https://download.pytorch.org/whl/cu124`)
- Torch version: `TEVO_TORCH_VERSION` (default: `2.6.0+cu124`)
- Volumes: `TEVO_MODAL_RUNS_VOLUME` (default: `tevo-runs`), `TEVO_MODAL_HF_VOLUME` (default: `tevo-hf-cache`)
- Timeout (seconds): `TEVO_MODAL_TIMEOUT_S` (default: 12h)
</details>
