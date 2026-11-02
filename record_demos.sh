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
# Output: everything shown in the console is also saved to logs/demos/<time>/console.log,
#         and clips.tsv there records each clip's model, slip, seed, times, status,
#         targets reached, falls and the viewer's own log folder.
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

RUN_DIR="logs/demos/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/clips.tsv"
printf 'clip\ttitle\tmodel\tslip\tseed\tstarted\tfinished\tstatus\ttargets\tfalls\tviewer_log\n' > "$SUMMARY"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
echo "Recording session: $RUN_DIR (seed ${SEED}, altered slip ${ALTERED_SLIP}, git $(git rev-parse --short HEAD 2>/dev/null || echo unknown))"

failed=()
for i in "${!CLIPS[@]}"; do
  n=$((i + 1))
  [ "$n" -lt "$FROM" ] && continue
  IFS='|' read -r title label slip <<< "${CLIPS[$i]}"
  echo
  echo "=== Clip ${n}/${#CLIPS[@]}: ${title} (seed ${SEED}, slip ${slip}) ==="
  started="$(date '+%Y-%m-%d %H:%M:%S')"
  clip_log="$RUN_DIR/clip_$(printf '%02d' "$n").log"
  if ./run_viewer.sh --model "$label" --slip "$slip" --seed "$SEED" 2>&1 | tee "$clip_log"; then
    status=ok
  else
    status=failed
    echo "Clip ${n} failed; continuing. Redo it with: ./record_demos.sh --from ${n}" >&2
    failed+=("$n")
  fi
  finished="$(date '+%Y-%m-%d %H:%M:%S')"
  last="$(grep -E '\[Demo\] [0-9]+/[0-9]+ steps' "$clip_log" | tail -1)"
  targets="$(sed -nE 's/.*targets=([0-9]+).*/\1/p' <<< "$last")"
  falls="$(sed -nE 's/.*falls=([0-9]+).*/\1/p' <<< "$last")"
  viewer_log="$(sed -nE 's/^\[Viewer\] Logs: //p' "$clip_log" | tail -1)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$title" "$label" "$slip" "$SEED" \
    "$started" "$finished" "$status" "${targets:--}" "${falls:--}" "${viewer_log:--}" >> "$SUMMARY"
  echo "--- Clip ${n}: ${status}, targets ${targets:--}, falls ${falls:--}"
done
echo
echo "Clip record: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All clips finished."
else
  echo "Finished with failures in clips: ${failed[*]}"
  exit 1
fi
