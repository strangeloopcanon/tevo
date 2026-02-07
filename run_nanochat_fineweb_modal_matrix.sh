#!/usr/bin/env bash
set -euo pipefail

# So what: one command to prepare nanochat-like FineWeb packed data and launch
# a small multi-seed Modal evolution matrix with staggered rung budgets.

CONFIG_PATH="${TEVO_CONFIG_PATH:-configs/exp_nanochat_gpt2grade_d20_modal_evolve_fineweb_staggered_gpt2vocab_aeff095e.yaml}"
GPU="${TEVO_MODAL_GPU:-A10G}"
TIMEOUT_S="${TEVO_MODAL_TIMEOUT_S:-21600}"
RUN_PREFIX="${TEVO_RUN_PREFIX:-modal_nanochat_fineweb_d20_staggered}"

GENERATIONS="${TEVO_GENERATIONS:-16}"
STEPS="${TEVO_STEPS:-180}"
EVAL_BATCHES="${TEVO_EVAL_BATCHES:-4}"
SEEDS_CSV="${TEVO_SEEDS:-0,1}"

DATASET="${TEVO_DATASET:-karpathy/fineweb-edu-100b-shuffle}"
OUT_DIR="${TEVO_OUT_DIR:-fineweb_edu_10m}"
MAX_TRAIN_TOKENS="${TEVO_MAX_TRAIN_TOKENS:-10000000}"
MAX_VAL_TOKENS="${TEVO_MAX_VAL_TOKENS:-1000000}"
VAL_FRACTION="${TEVO_VAL_FRACTION:-0.1}"
SKIP_DATA_PREP="${TEVO_SKIP_DATA_PREP:-0}"

if [[ "${SKIP_DATA_PREP}" != "1" ]]; then
  echo "[setup] Preparing packed FineWeb data on Modal volume..."
  modal run scripts/modal_prepare_packed_data.py \
    --dataset "${DATASET}" \
    --out-dir "${OUT_DIR}" \
    --max-train-tokens "${MAX_TRAIN_TOKENS}" \
    --max-val-tokens "${MAX_VAL_TOKENS}" \
    --val-fraction "${VAL_FRACTION}" \
    --download-metadata
else
  echo "[setup] Skipping data prep (TEVO_SKIP_DATA_PREP=1)."
fi

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
STAMP="$(date +%Y%m%d_%H%M%S)"
GPU_SLUG="$(printf '%s' "${GPU}" | tr '[:upper:]' '[:lower:]')"

for seed in "${SEEDS[@]}"; do
  run_id="${RUN_PREFIX}_${GPU_SLUG}_g${GENERATIONS}_s${STEPS}_seed${seed}_${STAMP}"
  echo "[run] Launching ${run_id}"
  TEVO_MODAL_GPU="${GPU}" \
  TEVO_MODAL_TIMEOUT_S="${TIMEOUT_S}" \
  modal run scripts/modal_run_live.py \
    --config-path "${CONFIG_PATH}" \
    --generations "${GENERATIONS}" \
    --steps "${STEPS}" \
    --eval-batches "${EVAL_BATCHES}" \
    --seed "${seed}" \
    --run-id "${run_id}" \
    --prune-checkpoints-to-frontier \
    --lineage \
    --download
done

echo "[done] Completed matrix for seeds: ${SEEDS_CSV}"
echo "[done] Artifacts downloaded under runs/modal/"
