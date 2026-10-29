#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/pamnt/miniconda3/envs/fcportugal/bin/python3}"
ARGS=()
INCLUDE_SCRATCH=0
for ARG in "$@"; do
    if [[ "$ARG" == --include-scratch ]]; then INCLUDE_SCRATCH=1; else ARGS+=("$ARG"); fi
done
if [[ "$INCLUDE_SCRATCH" == 1 ]]; then
    exec "$PYTHON_BIN" -u -m scratch_comparison.run training "${ARGS[@]}"
fi
exec "$PYTHON_BIN" -u -m consolidated.run training "$@"
