#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITERATIONS="${ITERATIONS:-10}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4000}"
EXPERT_DATA="${EXPERT_DATA:-expert_train_data.npz}"
GAIL_LABEL="${GAIL_LABEL:-gail_dagger}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/${GAIL_LABEL}}"
INIT_MODEL="${INIT_MODEL:-dagger_policy.pth}"
POLICY_MODEL="${POLICY_MODEL:-${ARTIFACT_DIR}/policy.pth}"
FINAL_MODEL="${FINAL_MODEL:-${GAIL_LABEL}_policy.pth}"
DISC_MODEL="${DISC_MODEL:-${ARTIFACT_DIR}/discriminator.pth}"
OPTIMIZER_STATE="${OPTIMIZER_STATE:-${ARTIFACT_DIR}/optimizer_state.pth}"
ROLLOUT_DATA="${ROLLOUT_DATA:-${ARTIFACT_DIR}/rollout_batch.npz}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
DISC_EPOCHS="${DISC_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-0.000003}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-0.0003}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-0.0001}"
ENTROPY_COEFFICIENT="${ENTROPY_COEFFICIENT:-0.001}"
BC_COEFFICIENT="${BC_COEFFICIENT:-0.05}"
GRADIENT_PENALTY_COEFFICIENT="${GRADIENT_PENALTY_COEFFICIENT:-10.0}"
TARGET_KL="${TARGET_KL:-0.02}"
ROLLBACK_KL="${ROLLBACK_KL:-0.05}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for value_name in ITERATIONS ROLLOUT_STEPS PPO_EPOCHS DISC_EPOCHS BATCH_SIZE; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[Error] $value_name must be a positive integer." >&2
        exit 2
    fi
done

if [[ ! -f "$EXPERT_DATA" ]]; then
    echo "[Error] Expert dataset not found: $EXPERT_DATA" >&2
    exit 1
fi
if [[ "$INIT_MODEL" != "random" && ! -f "$INIT_MODEL" && ! -f "$POLICY_MODEL" ]]; then
    echo "[Error] Initialization policy not found: $INIT_MODEL" >&2
    exit 1
fi
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

if [[ "${RESET_GAIL:-0}" == "1" ]]; then
    backup_dir="experiments/${GAIL_LABEL}_backup_$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    moved=0
    shopt -s nullglob
    artifacts=(
        "$POLICY_MODEL" "$DISC_MODEL" "$OPTIMIZER_STATE" "$ROLLOUT_DATA"
        "${ARTIFACT_DIR}/policy_iter_"*.pth "${ARTIFACT_DIR}/discriminator_iter_"*.pth
        "${ARTIFACT_DIR}/rollout_metrics_iter_"*.json
        "${ARTIFACT_DIR}/training_metrics_iter_"*.json
    )
    shopt -u nullglob
    for artifact in "${artifacts[@]}"; do
        if [[ -e "$artifact" ]]; then
            mv -- "$artifact" "$backup_dir/"
            moved=1
        fi
    done
    if [[ "$moved" == "1" ]]; then
        echo "[Info] Previous GAIL state moved to: $backup_dir"
    else
        rmdir "$backup_dir"
    fi
fi

echo "=========================================="
echo " STARTING GAIL TRAINING PIPELINE"
echo " Variant: $GAIL_LABEL"
echo " Actor initialization: $INIT_MODEL"
echo "=========================================="

# Reassert derived paths here as well as at startup. Besides documenting the
# loop invariant, this keeps a long-running shell compatible if the pipeline
# file is updated between iterations during interactive development.
: "${GAIL_LABEL:=gail_dagger}"
: "${ARTIFACT_DIR:=artifacts/${GAIL_LABEL}}"
mkdir -p "$ARTIFACT_DIR"

for ((i=1; i<=ITERATIONS; i++)); do
    echo ""
    echo "------------------------------------------"
    echo " GAIL ITERATION $i / $ITERATIONS"
    echo "------------------------------------------"

    echo "[Step 1/2] Collecting synchronous on-policy trajectories..."
    "$PYTHON_BIN" rollout_gail.py \
        --iter "$i" --steps "$ROLLOUT_STEPS" --seed "$((SEED + i))" \
        --model "$POLICY_MODEL" --init-model "$INIT_MODEL" \
        --expert-data "$EXPERT_DATA" --rollout-output "$ROLLOUT_DATA" \
        --metrics-output "${ARTIFACT_DIR}/rollout_metrics_iter_${i}.json" \
        --label "$GAIL_LABEL"

    echo "[Step 2/2] Updating discriminator and PPO actor-critic..."
    "$PYTHON_BIN" train_gail.py \
        --iter "$i" --expert-data "$EXPERT_DATA" --rollout-data "$ROLLOUT_DATA" \
        --model "$POLICY_MODEL" --discriminator "$DISC_MODEL" \
        --init-model "$INIT_MODEL" --optimizer-state "$OPTIMIZER_STATE" \
        --ppo-epochs "$PPO_EPOCHS" --discriminator-epochs "$DISC_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --actor-learning-rate "$ACTOR_LEARNING_RATE" \
        --critic-learning-rate "$CRITIC_LEARNING_RATE" \
        --discriminator-learning-rate "$DISC_LEARNING_RATE" \
        --entropy-coefficient "$ENTROPY_COEFFICIENT" \
        --bc-coefficient "$BC_COEFFICIENT" \
        --gradient-penalty-coefficient "$GRADIENT_PENALTY_COEFFICIENT" \
        --target-kl "$TARGET_KL" --rollback-kl "$ROLLBACK_KL" --seed "$SEED" \
        --metrics-output "${ARTIFACT_DIR}/training_metrics_iter_${i}.json" \
        --label "$GAIL_LABEL"
done

cp -- "$POLICY_MODEL" "$FINAL_MODEL"

echo "=========================================="
echo " GAIL TRAINING COMPLETE"
echo " Final policy: $FINAL_MODEL"
echo "=========================================="
