#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GAIL_LABEL=gail_bc INIT_MODEL=bc_walk_policy.pth exec ./run_gail.sh
