#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDITION_ID="${CONDITION_ID:?Set CONDITION_ID, for example high_traction}"
FIELD_SLIP="${FIELD_SLIP:?Set FIELD_SLIP, for example 0.025}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PHASE_DIR="experiments/phase2_domain_robustness/conditions/${CONDITION_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-${PHASE_DIR}/evaluation}"

"$PYTHON_BIN" configure_field_slip.py status --expect "$FIELD_SLIP" \
    --output "$PHASE_DIR/environment_manifest.json"
"$PYTHON_BIN" check_simulator_config.py

ENVIRONMENT_ID="$CONDITION_ID" FIELD_SLIP="$FIELD_SLIP" \
OUTPUT_DIR="$OUTPUT_DIR" RESET_EVALUATION="${RESET_EVALUATION:-1}" \
RESUME_EVALUATION="${RESUME_EVALUATION:-1}" \
EVALUATE_CHECKPOINTS=0 "$SCRIPT_DIR/run_evaluation.sh"

echo "[Phase 2 condition] Complete: $PHASE_DIR"
