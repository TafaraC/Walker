#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-7}"
SEED_STEPS="${SEED_STEPS:-20000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/dagger}"
DATASET="${DATASET:-${ARTIFACT_DIR}/aggregated_data.npz}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${ARTIFACT_DIR}/policy.pth}"
FINAL_MODEL="${FINAL_MODEL:-dagger_policy.pth}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for value_name in TOTAL_ITERATIONS SEED_STEPS ROLLOUT_STEPS EPOCHS BATCH_SIZE; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[Error] $value_name must be a positive integer." >&2
        exit 2
    fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[Error] Python interpreter not found: $PYTHON_BIN" >&2
    exit 2
fi
mkdir -p "$ARTIFACT_DIR"

echo "[Preflight] Python interpreter: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[Preflight] Checking FC Portugal native modules for this Python version..."
"$PYTHON_BIN" -c 'from scripts.commons.Script import Script; Script.build_cpp_modules()'
"$PYTHON_BIN" -c 'from cpp.ball_predictor import ball_predictor; from cpp.localization import localization; from cpp.a_star import a_star'

if [[ "${SKIP_SIM_CONFIG_CHECK:-0}" != "1" ]]; then
    "$PYTHON_BIN" check_simulator_config.py
fi

if [[ "${RESET_DAGGER_DATASET:-0}" == "1" && -e "$DATASET" ]]; then
    backup="${DATASET}.bak.$(date +%Y%m%d-%H%M%S)"
    mv -- "$DATASET" "$backup"
    echo "[Info] Existing DAgger dataset moved to: $backup"
fi

echo "=========================================="
echo " STARTING DAGGER TRAINING PIPELINE"
echo "=========================================="

if [[ ! -f "$DATASET" ]]; then
    echo "[Phase 0] Collecting $SEED_STEPS FC Portugal oracle seed steps..."
    "$PYTHON_BIN" collect_seed.py \
        --steps "$SEED_STEPS" --dataset "$DATASET" --seed "$SEED"
else
    echo "[Phase 0] Using existing seed/aggregate dataset: $DATASET"
    echo "[Phase 0] Use RESET_DAGGER_DATASET=1 for a clean experiment."
fi

echo "[Phase 0] Training initial BC policy..."
"$PYTHON_BIN" train_dagger.py \
    --iter 0 --dataset "$DATASET" --output "$MODEL_OUTPUT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --validation-fraction "$VALIDATION_FRACTION" --seed "$SEED" \
    --metrics-output "${ARTIFACT_DIR}/training_metrics_iter_0.json"

for ((i=1; i<=TOTAL_ITERATIONS; i++)); do
    echo ""
    echo "=========================================="
    echo " DAGGER ITERATION $i / $TOTAL_ITERATIONS"
    echo "=========================================="

    echo "[Step 1/2] Rolling out mixture policy and querying oracle..."
    "$PYTHON_BIN" rollout_dagger.py \
        --iter "$i" --steps "$ROLLOUT_STEPS" \
        --model "$MODEL_OUTPUT" --dataset "$DATASET" \
        --seed "$((SEED + i))" \
        --metrics-output "${ARTIFACT_DIR}/rollout_metrics_iter_${i}.json"

    echo "[Step 2/2] Training on the newly aggregated dataset..."
    "$PYTHON_BIN" train_dagger.py \
        --iter "$i" --dataset "$DATASET" --output "$MODEL_OUTPUT" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --validation-fraction "$VALIDATION_FRACTION" --seed "$SEED" \
        --metrics-output "${ARTIFACT_DIR}/training_metrics_iter_${i}.json"
done

cp -- "$MODEL_OUTPUT" "$FINAL_MODEL"

echo "=========================================="
echo " DAGGER PIPELINE COMPLETE"
echo " Final policy: $FINAL_MODEL"
echo "=========================================="
