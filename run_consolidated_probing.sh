#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=()
INCLUDE_SCRATCH=0
for ARG in "$@"; do
    if [[ "$ARG" == --include-scratch ]]; then INCLUDE_SCRATCH=1; else ARGS+=("$ARG"); fi
done
if [[ "$INCLUDE_SCRATCH" == 1 ]]; then
    exec "${PYTHON_BIN:-/home/pamnt/miniconda3/envs/fcportugal/bin/python3}" -u -m scratch_comparison.probe "${ARGS[@]}"
fi
exec "${PYTHON_BIN:-/home/pamnt/miniconda3/envs/fcportugal/bin/python3}" -u -m consolidated_probing.run "$@"
