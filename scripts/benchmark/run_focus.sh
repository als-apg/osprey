#!/usr/bin/env bash
# Focused manual e2e run: one forced model, a hand-picked set of test nodes.
# Reproduces the als-apg ref-lane env contract from run_e2e_for_model.sh, but
# runs only the node(s) passed as args (e.g. the timezone-sensitive scenario
# tests) so we can confirm the vacuum_burst gate without the full ~30 min suite.
#
# Usage (on the box, from repo root):
#   MODEL=claude-opus-4-6 bash scripts/benchmark/run_focus.sh \
#       tests/e2e/test_vacuum_burst_scenario.py
#
# Env:
#   MODEL   forced model id (als-apg ids: claude-{haiku-4-5-20251001,sonnet-4-6,opus-4-6})
#   TAG     label for output files (default: derived from MODEL)
set -uo pipefail
REPO="${OSPREY_BENCH_REMOTE_REPO:-$HOME/projects/osprey}"
cd "$REPO" || { echo "FATAL: cannot cd $REPO" >&2; exit 2; }
PY="$REPO/.venv/bin/python"
MODEL="${MODEL:?set MODEL=<als-apg model id>}"
NODES=("$@"); [ "${#NODES[@]}" -gt 0 ] || { echo "FATAL: pass test node(s)" >&2; exit 2; }

# --- als-apg ref env contract (mirrors run_e2e_for_model.sh) ---------------
[ -n "${ALS_APG_API_KEY:-}" ] || ALS_APG_API_KEY="$(cat "$HOME/.als_apg_key")"
export ALS_APG_API_KEY
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$ALS_APG_API_KEY}"
export ALS_APG_BASE_URL="${ALS_APG_BASE_URL:-https://llm.gianlucamartino.com}"
export OSPREY_E2E_JUDGE_MODEL="${OSPREY_E2E_JUDGE_MODEL:-claude-haiku-4-5-20251001}"
export OSPREY_E2E_FORCE_PROVIDER=als-apg
export OSPREY_E2E_FORCE_MODEL="$MODEL"
unset OSPREY_E2E_PROXY_UPSTREAM
case "$MODEL" in
  *opus*)   export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-6}" ;;
  *sonnet*) export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-3}" ;;
  *)        export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-1}" ;;
esac

TAG="${TAG:-${MODEL//\//__}}"
OUT="$REPO/results_focus"; mkdir -p "$OUT"
XML="$OUT/${TAG}.xml"
export OSPREY_E2E_LIVE="$OUT/${TAG}.live.jsonl"; : > "$OSPREY_E2E_LIVE"
echo ">> MODEL=$MODEL budget_scale=$OSPREY_E2E_BUDGET_SCALE nodes=${NODES[*]}" >&2
echo ">> junit=$XML" >&2

"$PY" -m pytest "${NODES[@]}" \
  -p no:cacheprovider -q --no-header -o addopts="" \
  --timeout="${E2E_TEST_TIMEOUT:-1800}" --timeout-method=signal \
  --junitxml="$XML"
echo ">> DONE $MODEL rc=$? $(date +%H:%M:%S)" >&2
