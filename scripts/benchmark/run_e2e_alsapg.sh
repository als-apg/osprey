#!/usr/bin/env bash
# Run the FULL OSPREY e2e kit against ONE Anthropic model via NATIVE als-apg
# (issue #259 reference bracket). Sibling of run_e2e_for_model.sh, but:
#   * routes to als-apg directly (NO CBORG, NO translation proxy) — the refs
#     never needed CBORG, and CBORG's anthropic-direct path breaks several
#     safety/feedback tests that als-apg passes (validated 2026-06-18).
#   * derives its pytest --ignore list from scripts/benchmark/matrix_e2e_config.json via
#     check_e2e_coverage.py (single-sourced exclusions + startup coverage warning).
#
#   ALS_APG_API_KEY=sk-... scripts/benchmark/run_e2e_alsapg.sh claude-opus-4-6 1
#
# Outputs (under results/, same schema as run_e2e_for_model.sh so the dashboard
# reads both): results/<safe>__seed<seed>.{xml,json,live.jsonl}
set -uo pipefail

MODEL="${1:?usage: run_e2e_alsapg.sh <ALS_APG_MODEL_ID> [SEED]}"
SEED="${2:-1}"

cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv python at $PY" >&2; exit 2; }

# --- credentials: als-apg ONLY ------------------------------------------
[ -n "${ALS_APG_API_KEY:-}" ] || { echo "ALS_APG_API_KEY unset (required for als-apg run)" >&2; exit 2; }
export ALS_APG_API_KEY
# requires_api / requires_als_apg collection gates are satisfied by ALS_APG_API_KEY.
# Native als-apg endpoint (AWS proxy, NO /v1 — anthropic SDK appends /v1/messages).
export ALS_APG_BASE_URL="${ALS_APG_BASE_URL:-https://llm.gianlucamartino.com}"
# Scenario LLM-judge: a valid als-apg model id (native routing, no override needed).
export OSPREY_E2E_JUDGE_MODEL="${OSPREY_E2E_JUDGE_MODEL:-claude-haiku-4-5-20251001}"

# --- suite-wide provider/model override: force every tier to MODEL via als-apg
export OSPREY_E2E_FORCE_PROVIDER=als-apg
export OSPREY_E2E_FORCE_MODEL="$MODEL"
unset OSPREY_E2E_PROXY_UPSTREAM   # no translation proxy for anthropic-protocol

# Pricier reference models cost several times more per token, so the haiku-tuned
# max_budget_usd caps blow mid-query ("Reached maximum budget") — a cost artifact
# that deflates the score for reasons unrelated to capability. Scale the cap (and
# the matching cost-ceiling assertions) by model, mirroring run_e2e_for_model.sh.
# Honored by e2e_budget_scale() in tests/e2e/sdk_helpers.py. Overridable from env.
case "$MODEL" in
  *opus*)   export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-6}" ;;
  *sonnet*) export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-3}" ;;
  *)        export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-1}" ;;
esac
echo ">> budget_scale=$OSPREY_E2E_BUDGET_SCALE" >&2

SAFE="${MODEL//\//__}"
mkdir -p "$REPO/results"
XML="$REPO/results/${SAFE}__seed${SEED}.xml"
JSON="$REPO/results/${SAFE}__seed${SEED}.json"
export OSPREY_E2E_LIVE="$REPO/results/${SAFE}__seed${SEED}.live.jsonl"
: > "$OSPREY_E2E_LIVE"

echo ">> model=$MODEL seed=$SEED route=als-apg-direct base=$ALS_APG_BASE_URL" >&2

# --- coverage guard: single-sourced exclusions + startup warning ---------
# Audit + warnings go to stderr (captured in the run log); --ignore args to stdout.
# NB: macstudio runs macOS bash 3.2 (no `mapfile`). The --ignore tokens contain no
# spaces, so collect into a plain string and rely on word-splitting at the call site.
IGNORE_ARGS="$("$PY" "$REPO/scripts/benchmark/check_e2e_coverage.py" \
  --repo-root "$REPO" --config "$REPO/scripts/benchmark/matrix_e2e_config.json" --print-ignore-args | tr '\n' ' ')"

# Optional pre-run hook (e.g. re-anchor ARIEL demo-logbook timestamps to "now").
if [ -n "${OSPREY_E2E_PRERUN_HOOK:-}" ] && [ -x "${OSPREY_E2E_PRERUN_HOOK}" ]; then
  echo ">> prerun hook: ${OSPREY_E2E_PRERUN_HOOK}" >&2
  "${OSPREY_E2E_PRERUN_HOOK}" >&2 || echo ">> WARN: prerun hook failed (rc=$?)" >&2
fi

export E2E_TEST_TIMEOUT="${E2E_TEST_TIMEOUT:-1800}"
START=$(date +%s)
# MUST be `pytest tests/e2e/` (path), NEVER `-m e2e` (registry leaks). --ignore
# tokens come from the config via check_e2e_coverage.py (do not hand-edit here).
# shellcheck disable=SC2086  # intentional word-split of space-free --ignore tokens
"$PY" -m pytest tests/e2e/ \
  $IGNORE_ARGS \
  -p no:cacheprovider -q --no-header \
  -o addopts="" \
  --timeout="${E2E_TEST_TIMEOUT}" --timeout-method=signal \
  --junitxml="$XML"
PYTEST_RC=$?
END=$(date +%s)
WALL=$((END - START))

# --- summarize JUnit XML -> summary JSON (schema matches run_e2e_for_model.sh)
MODEL="$MODEL" SEED="$SEED" ROUTE="als-apg" XML="$XML" JSON="$JSON" \
WALL="$WALL" PYTEST_RC="$PYTEST_RC" "$PY" - <<'PYEOF'
import json, os, xml.etree.ElementTree as ET
xml, jsonp = os.environ["XML"], os.environ["JSON"]
summary = {
    "model": os.environ["MODEL"], "seed": int(os.environ["SEED"]),
    "route": os.environ["ROUTE"], "pytest_rc": int(os.environ["PYTEST_RC"]),
    "total_duration_s": int(os.environ["WALL"]),
    "passed": 0, "failed": 0, "timeout": 0, "skipped": 0, "errors": 0, "total": 0,
    "tests": [],
}
KEY = {"passed": "passed", "failed": "failed", "timeout": "timeout",
       "skipped": "skipped", "error": "errors"}
try:
    root = ET.parse(xml).getroot()
    suites = root.findall("testsuite") or ([root] if root.tag == "testsuite" else [])
    for ts in suites:
        for tc in ts.findall("testcase"):
            name = f"{tc.get('classname','')}::{tc.get('name','')}"
            dur = round(float(tc.get("time", 0) or 0), 2)
            fail = tc.find("failure")
            if fail is not None:
                msg = (fail.get("message", "") + " " + (fail.text or "")).lower()
                outcome = "timeout" if "timeout" in msg else "failed"
            elif tc.find("error") is not None:
                outcome = "error"
            elif tc.find("skipped") is not None:
                outcome = "skipped"
            else:
                outcome = "passed"
            summary[KEY[outcome]] += 1
            summary["total"] += 1
            summary["tests"].append({"name": name, "outcome": outcome, "duration_s": dur})
except FileNotFoundError:
    summary["error"] = "junit xml not produced (collection/import crash)"
with open(jsonp, "w") as f:
    json.dump(summary, f, indent=2)
print(f">> summary: passed={summary['passed']} failed={summary['failed']} "
      f"timeout={summary['timeout']} skipped={summary['skipped']} errors={summary['errors']} "
      f"total={summary['total']} wall={summary['total_duration_s']}s -> {jsonp}")
PYEOF

exit 0
