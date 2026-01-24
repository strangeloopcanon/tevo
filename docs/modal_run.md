# Modal GPU runs

So what: you can run evolution or benchmarks on a single Modal GPU and keep artifacts in Modal Volumes.

## Packed data prep (CPU)
```bash
modal run scripts/modal_prepare_packed_data.py --out-dir openwebtext_10m --max-train-tokens 10000000 --max-val-tokens 1000000 --val-fraction 0.1 --download-metadata
```

## Evolution run
```bash
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_live.py \
  --config-path configs/exp_longctx_overnight_m4_full_deck_template_learning.yaml \
  --generations 96 --steps 720 --eval-batches 8 \
  --download
```

This writes `frontier.json` (+ `frontier.state.json`, and optional `lineage.json`) into a persisted volume, and (with `--download`) downloads them into `runs/modal/<run_id>/` locally.

`runs/modal/` is treated as a local artifact directory (large JSON/checkpoints); keep it out of git.

## Benchmark run
```bash
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_benchmark.py \
  --config-path configs/bench_nanogpt_owt.yaml \
  --steps 240 --eval-batches 4 \
  --run-id bench_owt \
  --download
```

This writes `summary.json` + `history.json` into a persisted volume and downloads them to `runs/modal/<run_id>/` when `--download` is set.

<details>
<summary>GPU presets (A10G / A100)</summary>

Evolution:
```
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_live.py --config-path configs/exp_longctx_overnight_m4_full_deck_template_learning.yaml --generations 96 --steps 720 --eval-batches 8 --download
TEVO_MODAL_GPU=A100 modal run scripts/modal_run_live.py --config-path configs/exp_longctx_overnight_m4_full_deck_template_learning.yaml --generations 96 --steps 720 --eval-batches 8 --download
```

Benchmark:
```
TEVO_MODAL_GPU=A10G modal run scripts/modal_run_benchmark.py --config-path configs/bench_nanogpt_owt.yaml --steps 240 --eval-batches 4 --run-id bench_owt --download
TEVO_MODAL_GPU=A100 modal run scripts/modal_run_benchmark.py --config-path configs/bench_nanogpt_owt.yaml --steps 240 --eval-batches 4 --run-id bench_owt --download
```
</details>

<details>
<summary>Configuration knobs</summary>

- GPU selection: `TEVO_MODAL_GPU` (e.g. `A10G`, `A100`, `H100`)
- Torch wheel source: `TEVO_TORCH_INDEX_URL` (default: `https://download.pytorch.org/whl/cu124`)
- Torch version: `TEVO_TORCH_VERSION` (default: `2.6.0+cu124`)
- Volumes: `TEVO_MODAL_RUNS_VOLUME` (default: `tevo-runs`), `TEVO_MODAL_HF_VOLUME` (default: `tevo-hf-cache`)
- Timeout (seconds): `TEVO_MODAL_TIMEOUT_S` (default: 12h)
- Packed-token root: `TEVO_PACKED_ROOT` (default: `/runs` in the Modal image)
</details>
