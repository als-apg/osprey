#!/usr/bin/env bash
# Live dashboard updater (issue #259 model matrix). Every 60s: pull the newest
# results from the Mac Studio and re-render dashboard.html. Exits after the
# matrix signals completion (one final render) or a max-duration safety cap.
# Runs locally — the dashboard opens in a browser on this Mac.
set -uo pipefail
# Repo root, resolved from this script's location (scripts/benchmark/ -> repo).
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Remote box running the matrix, and its OSPREY checkout (relative to the remote
# login home unless absolute). Override for a different host/layout.
REMOTE="${OSPREY_BENCH_REMOTE:-macstudio}"
REMOTE_REPO="${OSPREY_BENCH_REMOTE_REPO:-projects/osprey}"
# All regenerable run-exhaust lives under one gitignored output dir co-located
# with the benchmark tooling, so the repo root stays clean (artifacts are
# regenerable from scripts/benchmark/, never committed).
ARTIFACTS="$REPO/scripts/benchmark/_out"
RESULTS="$ARTIFACTS/results_box"
OUT="$ARTIFACTS/dashboard.html"
LOG="$ARTIFACTS/live_updater.log"      # outside RESULTS so rsync --delete can't nuke it
mkdir -p "$RESULTS"
MAX_SECONDS="${1:-115200}"             # 32h safety
DEADLINE=$(( $(date +%s) + MAX_SECONDS ))

render() {
  rsync -az --delete "$REMOTE:$REMOTE_REPO/results/" "$RESULTS/" 2>>"$LOG"
  python3 "$REPO/scripts/benchmark/matrix_dashboard.py" --results-dir "$RESULTS" --out "$OUT" >>"$LOG" 2>&1
}

echo "=== live updater start $(date) -> $OUT ===" >> "$LOG"
while :; do
  render
  # Multi-driver aware: several driver invocations (matrix + qwen + ref) share
  # this matrix.log, and EACH writes MATRIX_EXIT_0 when ITS combos finish. So the
  # marker alone means "a driver finished", not "all done". Only exit when the
  # marker exists AND no combo is still in flight (every START has an END).
  done_mark=$(grep -c MATRIX_EXIT_0 "$RESULTS/matrix.log" 2>/dev/null || true)
  running=$(comm -23 \
    <(grep '>> START' "$RESULTS/matrix.log" 2>/dev/null | sed 's/.*START //;s/ [0-9:]*$//' | sort -u) \
    <(grep '>> END'   "$RESULTS/matrix.log" 2>/dev/null | sed 's/.*END   //;s/ rc=.*//'   | sort -u) \
    | wc -l | tr -d ' ')
  echo "$(date '+%H:%M:%S') rendered (marker=${done_mark:-0} running=${running:-?})" >> "$LOG"
  [ "${done_mark:-0}" -ge 1 ] && [ "${running:-1}" -eq 0 ] && { echo "all drivers done — final render, exiting" >> "$LOG"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "max duration reached, exiting" >> "$LOG"; break; }
  sleep "${DASH_INTERVAL:-60}"
done
