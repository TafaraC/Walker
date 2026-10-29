#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LABEL="${LABEL:-dagger_mixed}"
SEED_ROUNDS="${SEED_ROUNDS:-5}"
LONG_STEPS_PER_ROUND="${LONG_STEPS_PER_ROUND:-10000}"
SHORT_STEPS_PER_ROUND="${SHORT_STEPS_PER_ROUND:-10000}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-7}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/dagger_mixed}"
DATASET="${DATASET:-${ARTIFACT_DIR}/aggregated_data.npz}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${ARTIFACT_DIR}/policy.pth}"
FINAL_MODEL="${FINAL_MODEL:-dagger_mixed_policy.pth}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
SEED="${SEED:-142}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for value_name in SEED_ROUNDS LONG_STEPS_PER_ROUND SHORT_STEPS_PER_ROUND TOTAL_ITERATIONS ROLLOUT_STEPS EPOCHS BATCH_SIZE; do
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
"$PYTHON_BIN" -c 'from scripts.commons.Script import Script; Script.build_cpp_modules()'
"$PYTHON_BIN" -c 'from cpp.ball_predictor import ball_predictor; from cpp.localization import localization; from cpp.a_star import a_star'
if [[ "${SKIP_SIM_CONFIG_CHECK:-0}" != "1" ]]; then
    "$PYTHON_BIN" check_simulator_config.py
fi

if [[ "${RESET_DAGGER_DATASET:-0}" == "1" ]]; then
    backup_dir="experiments/${LABEL}_backup_$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    shopt -s nullglob
    artifacts=(
        "$DATASET" "$MODEL_OUTPUT" "${MODEL_OUTPUT%.pth}_iter_"*.pth
        "${ARTIFACT_DIR}/training_metrics_iter_"*.json
        "${ARTIFACT_DIR}/rollout_metrics_iter_"*.json
    )
    shopt -u nullglob
    moved=0
    for artifact in "${artifacts[@]}"; do
        if [[ -e "$artifact" ]]; then
            mv -- "$artifact" "$backup_dir/"
            moved=1
        fi
    done
    if [[ "$moved" == "0" ]]; then
        rmdir "$backup_dir"
    else
        echo "[Info] Previous $LABEL state moved to: $backup_dir"
    fi
fi

echo "=========================================================="
echo " DAGGER MIXED-SEED PIPELINE"
echo " Oracle seed: $SEED_ROUNDS x ($LONG_STEPS_PER_ROUND long + $SHORT_STEPS_PER_ROUND short)"
echo " Learner rollouts: $TOTAL_ITERATIONS x $ROLLOUT_STEPS steps"
echo "=========================================================="

if [[ ! -f "$DATASET" ]]; then
    for ((round=1; round<=SEED_ROUNDS; round++)); do
        echo "[Seed $round/$SEED_ROUNDS] Long-range oracle collection..."
        "$PYTHON_BIN" collect_long.py \
            --steps "$LONG_STEPS_PER_ROUND" --dataset "$DATASET" \
            --seed "$((SEED + round * 2))"
        echo "[Seed $round/$SEED_ROUNDS] Short-turn oracle collection..."
        "$PYTHON_BIN" collect_short.py \
            --steps "$SHORT_STEPS_PER_ROUND" --dataset "$DATASET" \
            --seed "$((SEED + round * 2 + 1))"
    done
else
    echo "[Seed] Using existing mixed aggregate: $DATASET"
fi

echo "[Iteration 0] Training on the mixed oracle seed..."
"$PYTHON_BIN" train_dagger.py \
    --iter 0 --dataset "$DATASET" --output "$MODEL_OUTPUT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --validation-fraction "$VALIDATION_FRACTION" --seed "$SEED" \
    --metrics-output "${ARTIFACT_DIR}/training_metrics_iter_0.json" --label "$LABEL"

for ((iteration=1; iteration<=TOTAL_ITERATIONS; iteration++)); do
    echo "=========================================================="
    echo " $LABEL ITERATION $iteration / $TOTAL_ITERATIONS"
    echo "=========================================================="
    "$PYTHON_BIN" rollout_dagger.py \
        --iter "$iteration" --steps "$ROLLOUT_STEPS" \
        --model "$MODEL_OUTPUT" --dataset "$DATASET" \
        --seed "$((SEED + 100 + iteration))" \
        --metrics-output "${ARTIFACT_DIR}/rollout_metrics_iter_${iteration}.json" \
        --label "$LABEL"
    "$PYTHON_BIN" train_dagger.py \
        --iter "$iteration" --dataset "$DATASET" --output "$MODEL_OUTPUT" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --validation-fraction "$VALIDATION_FRACTION" --seed "$SEED" \
        --metrics-output "${ARTIFACT_DIR}/training_metrics_iter_${iteration}.json" \
        --label "$LABEL"
done

cp -- "$MODEL_OUTPUT" "$FINAL_MODEL"
echo "[Complete] Final mixed-seed policy: $FINAL_MODEL"
