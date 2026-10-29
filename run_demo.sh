#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/home/pamnt/miniconda3/envs/fcportugal/bin/python3"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON=python3
fi
exec "${PYTHON_BIN:-$DEFAULT_PYTHON}" "$SCRIPT_DIR/demo_agents.py" "$@"
