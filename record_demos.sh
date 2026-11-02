#!/usr/bin/env bash
# Run the viewer for each demo clip in turn, for screen recording in RoboViz.
# Each run starts the simulator, waits for Enter (start recording, then press Enter),
# walks for 30 active seconds and shuts down before the next one starts.
#
# Usage: ./record_demos.sh [--list] [--from N]
#   --list    print the numbered clip plan and exit
#   --from N  start at clip N (to redo or continue after stopping)
# Environment: SEED (default 101) sets the targets, the same for every clip;
#              ALTERED_SLIP (default 10) is the fixed slip for the altered-surface clips.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
SEED="${SEED:-101}"
ALTERED_SLIP="${ALTERED_SLIP:-10}"

# title | menu label | slip
CLIPS=(
  "1 Original field: expert|historical: FC Portugal expert|original"
  "1 Original field: BC, matched updates|controls/bc_original_matched|original"
  "1 Original field: DAgger, 7 rounds|original/hidden/seed_42/dagger_mixed|original"
  "1 Original field: GAIL-BC|original/hidden/seed_42/gail_bc|original"
  "2 Control: zero action|controls/zero_native|original"
  "2 Control: untrained network, seed 42|convergence/engine_random_network_42|original"
  "2 Control: uniform random actions|historical: Uniform-random action baseline|original"
  "3 Slip ${ALTERED_SLIP}: BC, 100 epochs|original/hidden/seed_42/bc|${ALTERED_SLIP}"
  "3 Slip ${ALTERED_SLIP}: BC, matched updates|controls/bc_original_matched|${ALTERED_SLIP}"
  "4 Slip ${ALTERED_SLIP}: expert|historical: FC Portugal expert|${ALTERED_SLIP}"
  "4 Slip ${ALTERED_SLIP}: BC, validation-stopped|convergence/bc_original_convergence|${ALTERED_SLIP}"
)

FROM=1
LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST=1 ;;
    --from) FROM="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$LIST" = 1 ]; then
  for i in "${!CLIPS[@]}"; do
    IFS='|' read -r title label slip <<< "${CLIPS[$i]}"
    printf '%2d. %-45s %s (slip %s)\n' "$((i + 1))" "$title" "$label" "$slip"
  done
  exit 0
fi

# Check every label exists before starting any simulator.
AVAILABLE="$(./run_viewer.sh --list | sed -E 's/^ *[0-9]+\. //')" || { echo "Could not list viewer models" >&2; exit 1; }
missing=0
for clip in "${CLIPS[@]}"; do
  IFS='|' read -r title label slip <<< "$clip"
  if ! grep -Fxq -- "$label" <<< "$AVAILABLE"; then
    echo "Not in the viewer menu: $label ($title)" >&2
    missing=1
  fi
done
[ "$missing" = 0 ] || { echo "Fix the labels above (see ./run_viewer.sh --list)." >&2; exit 1; }

failed=()
for i in "${!CLIPS[@]}"; do
  n=$((i + 1))
  [ "$n" -lt "$FROM" ] && continue
  IFS='|' read -r title label slip <<< "${CLIPS[$i]}"
  echo
  echo "=== Clip ${n}/${#CLIPS[@]}: ${title} (seed ${SEED}, slip ${slip}) ==="
  if ! ./run_viewer.sh --model "$label" --slip "$slip" --seed "$SEED"; then
    echo "Clip ${n} failed; continuing. Redo it with: ./record_demos.sh --from ${n}" >&2
    failed+=("$n")
  fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All clips finished."
else
  echo "Finished with failures in clips: ${failed[*]}"
  exit 1
fi
