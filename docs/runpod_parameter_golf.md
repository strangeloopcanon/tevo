# Runpod Parameter Golf

This repo now includes a small Runpod helper script for cheap Parameter Golf smoke runs:

```bash
python scripts/runpod_parameter_golf.py --help
```

It is meant for the first practical loop:

1. Create or attach to a `1xH100` pod.
2. Download a tiny official challenge dataset subset.
3. Run one official baseline smoke test.
4. Sync this TEVO repo to the pod.
5. Run one TEVO benchmark or a short TEVO evolution sweep.

For final leaderboard attempts, you still need the official challenge flow and a reproducible `8xH100` run. This helper is for cheap iteration and early truth checks.

## Safety

The helper reads `RUNPOD_API_KEY` from the environment. It does not take the key as a normal CLI argument.

```bash
export RUNPOD_API_KEY=...
```

If you already created a pod in the Runpod web console, you can skip the API entirely and pass `--host` plus `--port` instead of `--pod-id`.

## Recommended First Run

Create a pod payload without spending money yet:

```bash
python scripts/runpod_parameter_golf.py create-pod \
  --name tevo-pg-smoke \
  --dry-run
```

Create the pod for real:

```bash
python scripts/runpod_parameter_golf.py create-pod \
  --name tevo-pg-smoke
```

Once you have a pod id, wait for SSH:

```bash
python scripts/runpod_parameter_golf.py wait-ready <pod_id>
```

Download a small official dataset subset and prepare the official repo:

```bash
python scripts/runpod_parameter_golf.py official-setup \
  --pod-id <pod_id> \
  --train-shards 1
```

Run the official baseline smoke test:

```bash
python scripts/runpod_parameter_golf.py official-smoke \
  --pod-id <pod_id> \
  --run-id baseline_sp1024_smoke \
  --iterations 200 \
  --max-wallclock-seconds 180
```

Sync this repo to the pod:

```bash
python scripts/runpod_parameter_golf.py sync-repo \
  --pod-id <pod_id>
```

Prepare TEVO against the official dataset cache:

```bash
python scripts/runpod_parameter_golf.py tevo-setup \
  --pod-id <pod_id>
```

Run one TEVO benchmark:

```bash
python scripts/runpod_parameter_golf.py tevo-benchmark \
  configs/pg_lane2_shared_depth.yaml \
  --pod-id <pod_id> \
  --steps 120 \
  --eval-batches 2
```

Run a short TEVO evolution sweep:

```bash
python scripts/runpod_parameter_golf.py tevo-evolution \
  configs/pg_lane2_shared_depth.yaml \
  --pod-id <pod_id> \
  --generations 24 \
  --steps 120 \
  --eval-batches 2
```

## Helpful Notes

- `tevo-benchmark` and `tevo-evolution` print a local size estimate before trying the remote run. If you do not have the challenge shards locally, they fall back to a size-only check instead of blocking the remote run.
- `sync-repo` excludes `.git`, virtualenvs, caches, and local `runs/`.
- `official-setup` installs the small Python package set needed for the official repo if you are using a plain PyTorch image instead of the official Runpod template.
- `ssh` can open an interactive shell, or run one remote command:

```bash
python scripts/runpod_parameter_golf.py ssh --pod-id <pod_id>
```

```bash
python scripts/runpod_parameter_golf.py ssh --pod-id <pod_id> nvidia-smi
```

## Cheapest Sensible Order

- First smoke: `official-setup` + `official-smoke`
- Then TEVO truth check: `sync-repo` + `tevo-setup` + `tevo-benchmark`
- Then short search: `tevo-evolution`
- Only after that: longer runs, wider sweeps, or an `8xH100` submission attempt
