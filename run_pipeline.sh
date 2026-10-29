#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# All values can be overridden as environment variables.
N_RUNS="${N_RUNS:-10}"
STEPS_PER_COLLECTOR="${STEPS_PER_COLLECTOR:-5000}"
DATASET="${DATASET:-expert_walk_data.npz}"
TRAIN_DATASET="${TRAIN_DATASET:-expert_train_data.npz}"
TEST_DATASET="${TEST_DATASET:-expert_test_data.npz}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
SPLIT_BLOCK_SIZE="${SPLIT_BLOCK_SIZE:-250}"
MODEL_OUTPUT="${MODEL_OUTPUT:-bc_walk_policy.pth}"
METRICS_OUTPUT="${METRICS_OUTPUT:-artifacts/bc/training_metrics.json}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! [[ "$N_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] N_RUNS must be a positive integer." >&2
    exit 2
fi
if ! [[ "$STEPS_PER_COLLECTOR" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] STEPS_PER_COLLECTOR must be a positive integer." >&2
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[Error] Python interpreter not found: $PYTHON_BIN" >&2
    exit 2
fi
mkdir -p "$(dirname -- "$METRICS_OUTPUT")"

echo "[Preflight] Python interpreter: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[Preflight] Checking FC Portugal native modules for this Python version..."
"$PYTHON_BIN" -c 'from scripts.commons.Script import Script; Script.build_cpp_modules()'
"$PYTHON_BIN" -c 'from cpp.ball_predictor import ball_predictor; from cpp.localization import localization; from cpp.a_star import a_star'

if [[ "${SKIP_SIM_CONFIG_CHECK:-0}" != "1" ]]; then
    "$PYTHON_BIN" check_simulator_config.py
fi

if [[ "${RESET_DATASET:-0}" == "1" && -e "$DATASET" ]]; then
    backup="${DATASET}.bak.$(date +%Y%m%d-%H%M%S)"
    mv -- "$DATASET" "$backup"
    echo "[Info] Existing dataset moved to: $backup"
fi

echo "=========================================="
echo " PIPELINE START: $N_RUNS ITERATIONS "
echo "=========================================="

for ((i=1; i<=N_RUNS; i++))
do
   echo ""
   echo "------------------------------------------"
   echo " RUN $i / $N_RUNS"
   echo "------------------------------------------"
   
   echo "[Step 1/2] Executing Long-Range Collection..."
   "$PYTHON_BIN" collect_long.py --steps "$STEPS_PER_COLLECTOR" --dataset "$DATASET" --seed "$((SEED + i * 2))"

   echo "[Step 2/2] Executing Short-Range Turns Collection..."
   "$PYTHON_BIN" collect_short.py --steps "$STEPS_PER_COLLECTOR" --dataset "$DATASET" --seed "$((SEED + i * 2 + 1))"
done

echo ""
echo "=========================================="
echo " ALL COLLECTION RUNS COMPLETE "
echo " Creating fixed expert train/test split... "
echo "=========================================="

"$PYTHON_BIN" split_expert_data.py \
    --source "$DATASET" --train-output "$TRAIN_DATASET" \
    --test-output "$TEST_DATASET" --test-fraction "$TEST_FRACTION" \
    --block-size "$SPLIT_BLOCK_SIZE" --seed "$SEED"

"$PYTHON_BIN" train.py \
    --dataset "$TRAIN_DATASET" \
    --output "$MODEL_OUTPUT" \
    --metrics-output "$METRICS_OUTPUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --seed "$SEED"

echo ""
echo "=========================================="
echo " PIPELINE FINISHED SUCCESSFULLY "
echo "=========================================="
