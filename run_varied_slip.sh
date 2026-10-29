#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/pamnt/miniconda3/envs/fcportugal/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi
case "${1:-}" in
  smoke|fixed|sampled) exec "$PYTHON_BIN" run_varied_slip.py "$@" ;;
  report-fixed) shift; exec "$PYTHON_BIN" summarize_varied_slip.py fixed "$@" ;;
  report-sampled) shift; exec "$PYTHON_BIN" summarize_varied_slip.py sampled "$@" ;;
  plan) shift; exec "$PYTHON_BIN" summarize_varied_slip.py plan "$@" ;;
  *) echo "Usage: $0 {smoke|fixed|sampled|report-fixed|report-sampled|plan} [options]"; exit 2 ;;
esac
