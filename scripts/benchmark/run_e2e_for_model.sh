#!/usr/bin/env bash
# Run the FULL OSPREY e2e suite (tests/e2e/) against ONE CBORG model id and
# emit machine-readable results. Part of the issue #259 model matrix.
#
#   scripts/benchmark/run_e2e_for_model.sh <MODEL_ID> [SEED]
#
# Examples:
#   scripts/benchmark/run_e2e_for_model.sh cborg-coder 1     # open model -> via proxy
#   scripts/benchmark/run_e2e_for_model.sh claude-haiku 1    # anthropic-protocol -> direct
#
# Mechanism (no per-test edits — see tests/e2e/sdk_helpers.py + conftest.py):
#   OSPREY_E2E_FORCE_PROVIDER=cborg   build every project with the cborg provider
#   OSPREY_E2E_FORCE_MODEL=<id>       collapse all tiers (haiku/sonnet/opus)->id
#   OSPREY_E2E_PROXY_UPSTREAM=...     (open models only) session fixture starts
#                                     the translation proxy + rewrites base URL
#   ALS_APG_API_KEY/ANTHROPIC_API_KEY shimmed so requires_* gates don't skip;
#                                     real routing still goes to CBORG.
#
# Outputs (under results/, relative to repo root):
#   results/<safe_model>__seed<seed>.xml    JUnit XML (per-test outcomes)
#   results/<safe_model>__seed<seed>.json   summary {model,seed,passed,failed,
#       skipped,errors,total,total_duration_s,tests:[{name,outcome,duration_s}]}
#
# NEVER `uv run` — invoke the venv python directly (per macstudio convention).
set -uo pipefail

MODEL="${1:?usage: run_e2e_for_model.sh <MODEL_ID> [SEED]}"
SEED="${2:-1}"

cd "$(dirname "$0")/../.." || exit 2
REPO="$PWD"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv python at $PY" >&2; exit 2; }

# --- provider selection: subjects via CBORG, refs via als-apg ------------
# Anthropic reference models route through als-apg (their clean Anthropic home),
# NOT CBORG. CBORG routing systematically degrades the agent's channel-finder /
# scenario behaviour (feedback_hook + sector7 fail on CBORG yet pass 7/7 on
# als-apg with the identical project) — a provider-routing artifact, not an
# OSPREY bug. Open-model subjects stay on CBORG. Select with MATRIX_PROVIDER.
PROVIDER="${MATRIX_PROVIDER:-cborg}"

if [ "$PROVIDER" = "als-apg" ]; then
  # --- als-apg route (Anthropic references) ---
  if [ -z "${ALS_APG_API_KEY:-}" ] && [ -f "$HOME/.als_apg_key" ]; then
    ALS_APG_API_KEY="$(cat "$HOME/.als_apg_key")"
  fi
  [ -n "${ALS_APG_API_KEY:-}" ] || { echo "ALS_APG_API_KEY unset and ~/.als_apg_key missing" >&2; exit 2; }
  export ALS_APG_API_KEY
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$ALS_APG_API_KEY}"  # satisfy requires_* gates
  export ALS_APG_BASE_URL="${ALS_APG_BASE_URL:-https://llm.gianlucamartino.com}"
  export OSPREY_E2E_JUDGE_MODEL="${OSPREY_E2E_JUDGE_MODEL:-claude-haiku-4-5-20251001}"
  export OSPREY_E2E_FORCE_PROVIDER=als-apg
  export OSPREY_E2E_FORCE_MODEL="$MODEL"
  unset OSPREY_E2E_PROXY_UPSTREAM
  ROUTE="als-apg"
else
  # --- credentials (cborg) ---
  if [ -z "${CBORG_API_KEY:-}" ] && [ -f "$HOME/.cborg_key" ]; then
    CBORG_API_KEY="$(cat "$HOME/.cborg_key")"
  fi
  [ -n "${CBORG_API_KEY:-}" ] || { echo "CBORG_API_KEY unset and ~/.cborg_key missing" >&2; exit 2; }
  export CBORG_API_KEY
  # Shims: satisfy the requires_als_apg / requires_anthropic collection gates so
  # the suite is COLLECTED, not skipped. Routing is overridden to CBORG regardless,
  # and both vars are in MANAGED_ENV_VARS (scrubbed before the CLI launches).
  export ALS_APG_API_KEY="${ALS_APG_API_KEY:-$CBORG_API_KEY}"
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$CBORG_API_KEY}"
  # Scenario tests run an als-apg LLM judge (tests/e2e/judge.py) which otherwise
  # targets an als-apg-only endpoint + a dated model id CBORG rejects. On this
  # cborg-only box, route the judge to CBORG with a valid model id. Env-gated, so
  # CI/als-apg behavior is unchanged when these are unset.
  export ALS_APG_BASE_URL="${ALS_APG_BASE_URL:-https://api.cborg.lbl.gov/v1}"
  export OSPREY_E2E_JUDGE_MODEL="${OSPREY_E2E_JUDGE_MODEL:-google/claude-haiku-4-5}"

  export OSPREY_E2E_FORCE_PROVIDER=cborg
  export OSPREY_E2E_FORCE_MODEL="$MODEL"

  # --- routing: proxy for open models, direct for anthropic-protocol -------
  # CBORG serves claude-*/*claude* on the Anthropic /v1/messages route (no proxy);
  # everything else (cborg-coder, gemma-*, gpt-oss-*, ...) is OpenAI-protocol and
  # must be translated by the proxy.
  case "$MODEL" in
    *claude*) ROUTE="direct"; unset OSPREY_E2E_PROXY_UPSTREAM ;;
    *)        ROUTE="proxy";  export OSPREY_E2E_PROXY_UPSTREAM="https://api.cborg.lbl.gov/v1" ;;
  esac
fi

# --- per-query budget scales with model price ----------------------------
# The e2e suite's max_budget_usd caps (0.25/1.0/2.0) are tuned for the haiku
# default model. Pricier reference models cost several times more per token, so
# a multi-step task blows the cap and hard-errors mid-query ("Reached maximum
# budget") — a cost artifact that deflates the model's score for reasons
# unrelated to capability. Scale the cap (and the matching cost-ceiling
# assertions) by model. Honored by e2e_budget_scale() in tests/e2e/sdk_helpers.py.
# Default 1.0 (open/haiku-tier models). Overridable from the environment.
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
# live per-test stream (conftest appends one JSON line per test as it finishes),
# so the dashboard fills in mid-run. Fresh file per run.
export OSPREY_E2E_LIVE="$REPO/results/${SAFE}__seed${SEED}.live.jsonl"
: > "$OSPREY_E2E_LIVE"

echo ">> model=$MODEL seed=$SEED route=$ROUTE" >&2
echo ">> junit=$XML" >&2

# Optional pre-run hook: e.g. re-anchor the ARIEL demo-logbook timestamps to "now"
# so the scenario tests' "this morning" searches hit (the shared Postgres ingest is
# static and drifts out of "today"). Portable: only runs if the env var points at an
# executable. Ships as scripts/benchmark/ariel_refresh_timestamps.sh, e.g.
# OSPREY_E2E_PRERUN_HOOK="$REPO/scripts/benchmark/ariel_refresh_timestamps.sh".
if [ -n "${OSPREY_E2E_PRERUN_HOOK:-}" ] && [ -x "${OSPREY_E2E_PRERUN_HOOK}" ]; then
  echo ">> prerun hook: ${OSPREY_E2E_PRERUN_HOOK}" >&2
  "${OSPREY_E2E_PRERUN_HOOK}" >&2 || echo ">> WARN: prerun hook failed (rc=$?)" >&2
fi

START=$(date +%s)
# MUST be `pytest tests/e2e/` (path), NEVER `-m e2e` (causes registry leaks).
# --timeout (pytest-timeout): bound slow tests so a weak model can't stall the
# suite. SIGNAL method (SIGALRM, main thread) FAILS the individual test and lets
# the session CONTINUE — the thread method instead hard-kills the whole pytest
# process on the first overrun, leaving an empty junit (total=0). The e2e async
# tests run via pytest-asyncio in the main thread, so SIGALRM interrupts them.
# Model-matrix scope (issue #259): only the 36 tests that call an LLM AND route
# through the forced model-under-test. Excluded (measure nothing about the model):
#   test_llm_providers.py      — own provider/model axis (parametrized), not ours
#   test_ariel_search.py       — Postgres keyword/embedding search, no agent
#   test_dockerfile_e2e.py     — pure `docker build` smoke, no LLM
#   test_claude_code_build_integration.py — own init_project hardcoded to als-apg
#   test_llm_channel_namer.py  — config/pattern + naming, doesn't honor override
# Still `pytest tests/e2e/` (path) for registry-safe collection; --ignore prunes.
# Exclusions are single-sourced in scripts/benchmark/matrix_e2e_config.json (shared with the
# als-apg runner) and validated at startup by check_e2e_coverage.py, which warns if
# the run stops covering the full e2e kit minus explicit removals. macOS bash 3.2 has
# no `mapfile`; the --ignore tokens are space-free, so word-split a plain string.
IGNORE_ARGS="$("$PY" "$REPO/scripts/benchmark/check_e2e_coverage.py" \
  --repo-root "$REPO" --config "$REPO/scripts/benchmark/matrix_e2e_config.json" --print-ignore-args | tr '\n' ' ')"
# shellcheck disable=SC2086  # intentional word-split of space-free --ignore tokens
"$PY" -m pytest tests/e2e/ \
  $IGNORE_ARGS \
  -p no:cacheprovider -q --no-header \
  -o addopts="" \
  --timeout="${E2E_TEST_TIMEOUT:-1800}" --timeout-method=signal \
  --junitxml="$XML"
PYTEST_RC=$?
END=$(date +%s)
WALL=$((END - START))

# --- summarize JUnit XML -> summary JSON ---------------------------------
MODEL="$MODEL" SEED="$SEED" ROUTE="$ROUTE" XML="$XML" JSON="$JSON" \
WALL="$WALL" PYTEST_RC="$PYTEST_RC" "$PY" - <<'PYEOF'
import json, os, xml.etree.ElementTree as ET

xml, jsonp = os.environ["XML"], os.environ["JSON"]
summary = {
    "model": os.environ["MODEL"],
    "seed": int(os.environ["SEED"]),
    "route": os.environ["ROUTE"],
    "pytest_rc": int(os.environ["PYTEST_RC"]),
    "total_duration_s": int(os.environ["WALL"]),
    "passed": 0, "failed": 0, "timeout": 0, "skipped": 0, "errors": 0, "total": 0,
    "tests": [],
}
# per-test outcome -> summary count key (timeout is split out of failed so a
# "too slow" result is distinguishable from a genuine assertion failure)
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
                # Match the pytest-timeout signature, NOT the bare word "timeout":
                # a test's own source/comments (echoed into the traceback) can
                # contain "timeout" and mislabel a genuine assertion failure as a
                # wall-clock timeout. pytest-timeout emits "... from pytest-timeout".
                outcome = "timeout" if "from pytest-timeout" in msg else "failed"
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
