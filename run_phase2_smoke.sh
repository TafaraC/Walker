#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDITION_ID="${CONDITION_ID:?Set CONDITION_ID, for example high_traction}"
FIELD_SLIP="${FIELD_SLIP:?Set FIELD_SLIP, for example 0.025}"
SMOKE_STEPS="${SMOKE_STEPS:-1000}"
SMOKE_SEED="${SMOKE_SEED:-901}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="experiments/phase2_domain_robustness/smoke/${CONDITION_ID}"

"$PYTHON_BIN" configure_field_slip.py status --expect "$FIELD_SLIP" \
    --output "$OUTPUT_DIR/environment_manifest.json"
"$PYTHON_BIN" check_simulator_config.py
mkdir -p "$OUTPUT_DIR/raw"
"$PYTHON_BIN" evaluate_policy.py \
    --policy oracle --label oracle \
    --expert-data expert_train_data.npz \
    --steps "$SMOKE_STEPS" --seed "$SMOKE_SEED" \
    --output-dir "$OUTPUT_DIR/raw" \
    --environment-id "$CONDITION_ID" --field-slip "$FIELD_SLIP"

echo "[Phase 2 smoke] Complete: $OUTPUT_DIR"
