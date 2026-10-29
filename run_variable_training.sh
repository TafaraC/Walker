#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/pamnt/miniconda3/envs/fcportugal/bin/python3}"
exec "$PYTHON_BIN" -u -m variable_training.run "$@"
