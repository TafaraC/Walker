#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
shopt -s nullglob

EVAL_STEPS="${EVAL_STEPS:-3000}"
EVAL_SEEDS="${EVAL_SEEDS:-101 202 303 404 505}"
EXPERT_TRAIN="${EXPERT_TRAIN:-expert_train_data.npz}"
EXPERT_TEST="${EXPERT_TEST:-expert_test_data.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-nominal}"
FIELD_SLIP="${FIELD_SLIP:-0.1}"
EVALUATE_CHECKPOINTS="${EVALUATE_CHECKPOINTS:-0}"
RESUME_EVALUATION="${RESUME_EVALUATION:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! [[ "$EVAL_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] EVAL_STEPS must be a positive integer." >&2
    exit 2
fi
for required in "$EXPERT_TRAIN" "$EXPERT_TEST" bc_walk_policy.pth dagger_policy.pth; do
    if [[ ! -f "$required" ]]; then
        echo "[Error] Required evaluation artifact not found: $required" >&2
        exit 1
    fi
done

echo "[Preflight] Python interpreter: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
"$PYTHON_BIN" -c 'from scripts.commons.Script import Script; Script.build_cpp_modules()'
"$PYTHON_BIN" -c 'from cpp.ball_predictor import ball_predictor; from cpp.localization import localization; from cpp.a_star import a_star'
if [[ "${SKIP_SIM_CONFIG_CHECK:-0}" != "1" ]]; then
    "$PYTHON_BIN" check_simulator_config.py
fi

if [[ "${RESET_EVALUATION:-1}" == "1" && -d "$OUTPUT_DIR" ]]; then
    mv -- "$OUTPUT_DIR" "${OUTPUT_DIR}_backup_$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR/raw"

read -r -a seeds <<< "$EVAL_SEEDS"

run_policy() {
    local kind="$1"
    local label="$2"
    local model="${3:-}"
    for seed in "${seeds[@]}"; do
        if [[ "$RESUME_EVALUATION" == "1" && -f "$OUTPUT_DIR/raw/${label}_seed_${seed}.json" ]]; then
            echo "[Evaluation] Skipping completed ${label} seed ${seed}"
            continue
        fi
        args=(
            --policy "$kind" --label "$label" --expert-data "$EXPERT_TRAIN"
            --steps "$EVAL_STEPS" --seed "$seed" --output-dir "$OUTPUT_DIR/raw"
            --environment-id "$ENVIRONMENT_ID" --field-slip "$FIELD_SLIP"
        )
        if [[ -n "$model" ]]; then
            args+=(--model "$model")
        fi
        "$PYTHON_BIN" evaluate_policy.py "${args[@]}"
    done
}

# The baselines must use the same seeds as every learned policy.
run_policy random random
run_policy oracle oracle
run_policy bc bc bc_walk_policy.pth
run_policy dagger dagger dagger_policy.pth
if [[ -f dagger_mixed_policy.pth ]]; then
    run_policy dagger dagger_mixed dagger_mixed_policy.pth
else
    echo "[Evaluation] Skipping unavailable variant: dagger_mixed_policy.pth"
fi
gail_variants=(gail_dagger gail_bc gail_random)
for label in "${gail_variants[@]}"; do
    model="${label}_policy.pth"
    if [[ -f "$model" ]]; then
        run_policy gail "$label" "$model"
    else
        echo "[Evaluation] Skipping unavailable variant: $model"
    fi
done

if [[ "$EVALUATE_CHECKPOINTS" == "1" ]]; then
    for model in artifacts/dagger/policy_iter_*.pth; do
        iteration="${model#artifacts/dagger/policy_iter_}"
        iteration="${iteration%.pth}"
        run_policy dagger "dagger_iter_${iteration}" "$model"
    done
    for model in artifacts/dagger_mixed/policy_iter_*.pth; do
        iteration="${model#artifacts/dagger_mixed/policy_iter_}"
        iteration="${iteration%.pth}"
        run_policy dagger "dagger_mixed_iter_${iteration}" "$model"
    done
    for variant in "${gail_variants[@]}"; do
        for model in "artifacts/${variant}/policy_iter_"*.pth; do
            iteration="${model#artifacts/${variant}/policy_iter_}"
            iteration="${iteration%.pth}"
            run_policy gail "${variant}_iter_${iteration}" "$model"
        done
    done
fi

"$PYTHON_BIN" summarize_evaluation.py \
    --raw-dir "$OUTPUT_DIR/raw" --expert-test "$EXPERT_TEST" \
    --expert-train "$EXPERT_TRAIN" --output-json "$OUTPUT_DIR/comparison.json" \
    --output-csv "$OUTPUT_DIR/comparison.csv"

echo "[Evaluation] Complete: $OUTPUT_DIR/comparison.json"
