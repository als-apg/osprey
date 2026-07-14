#!/usr/bin/env bash
# Worker: run the FULL OSPREY e2e suite (tests/e2e/) for ONE (model, seed) cell
# and emit machine-readable results. Part of the issue #259 model matrix.
#
#   scripts/benchmark/run_e2e_for_model.sh <MODEL_ID> [SEED]
#
# This is a DUMB worker: every routing decision (provider, proxy-vs-direct,
# credentials, judge endpoint, budget scale) is made by the launcher
# scripts/benchmark/matrix.py and handed in via the environment below. The
# worker recomputes nothing — it just isolates the ARIEL DB, runs pytest, and
# summarizes. To run a single cell for debugging, use the launcher's filter:
#
#   scripts/benchmark/matrix.py --only <model> --parallel 1
#
# Environment contract (all set by matrix.py / cell_env):
#   OSPREY_E2E_FORCE_PROVIDER   build every project with this provider   (required)
#   OSPREY_E2E_FORCE_MODEL      collapse all tiers (haiku/sonnet/opus)->id (required)
#   OSPREY_E2E_PROXY_UPSTREAM   (proxy route only) upstream OpenAI endpoint;
#                               its presence selects proxy vs direct routing
#   OSPREY_E2E_PROXY_KEY        upstream auth for the proxy ("" for local servers)
#   ALS_APG_API_KEY / ALS_APG_BASE_URL / OSPREY_E2E_JUDGE_MODEL   LLM judge wiring
#   ANTHROPIC_API_KEY           collection-gate shim (real routing overridden)
#   OSPREY_E2E_BUDGET_SCALE     per-query max_budget_usd multiplier (default 1)
#
# Outputs (under results/, relative to repo root):
#   results/<safe_model>__seed<seed>.xml    JUnit XML (per-test outcomes)
#   results/<safe_model>__seed<seed>.json   summary {model,seed,passed,failed,...}
#   results/<safe_model>__seed<seed>.live.jsonl  one JSON line per test as it ends
#
# NEVER `uv run` — invoke the venv python directly (per macstudio convention).
set -uo pipefail

MODEL="${1:?usage: run_e2e_for_model.sh <MODEL_ID> [SEED]}"
SEED="${2:-1}"

cd "$(dirname "$0")/../.." || exit 2
REPO="$PWD"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv python at $PY" >&2; exit 2; }

# --- launcher-set routing (no credential/case logic lives here anymore) --
if [ -z "${OSPREY_E2E_FORCE_PROVIDER:-}" ] || [ -z "${OSPREY_E2E_FORCE_MODEL:-}" ]; then
  echo "FATAL: OSPREY_E2E_FORCE_PROVIDER/OSPREY_E2E_FORCE_MODEL unset." >&2
  echo "       This worker is driven by scripts/benchmark/matrix.py — run that," >&2
  echo "       e.g.  scripts/benchmark/matrix.py --only $MODEL --parallel 1" >&2
  exit 2
fi
# Route is implied by whether the launcher provided a proxy upstream.
if [ -n "${OSPREY_E2E_PROXY_UPSTREAM:-}" ]; then ROUTE="proxy"; else ROUTE="direct"; fi
export OSPREY_E2E_BUDGET_SCALE="${OSPREY_E2E_BUDGET_SCALE:-1}"
echo ">> budget_scale=$OSPREY_E2E_BUDGET_SCALE" >&2

SAFE="${MODEL//\//__}"
mkdir -p "$REPO/results"
XML="$REPO/results/${SAFE}__seed${SEED}.xml"
JSON="$REPO/results/${SAFE}__seed${SEED}.json"
# live per-test stream (conftest appends one JSON line per test as it finishes),
# so the dashboard fills in mid-run. Fresh file per run.
export OSPREY_E2E_LIVE="$REPO/results/${SAFE}__seed${SEED}.live.jsonl"
: > "$OSPREY_E2E_LIVE"

echo ">> model=$MODEL seed=$SEED provider=$OSPREY_E2E_FORCE_PROVIDER route=$ROUTE" >&2
echo ">> junit=$XML" >&2

# --- per-cell ARIEL database isolation (issue #259) ----------------------
# Concurrent matrix cells share one Postgres server. The scenario tests call
# apply_scenarios(seed_logbook=True), which PURGES the logbook AND DROPS the
# text_embeddings_* tables. With a single shared DB that races every other
# cell's logbook tests: a purge in cell A drops the embedding table that cell
# B's delegation/retrieval test is mid-way through using ("semantic search
# module not enabled"). Give each (model,seed) cell its OWN database so a purge
# can only affect its own cell. The built test projects honor OSPREY_ARIEL_DB_URI
# via tests/e2e/sdk_helpers._override_ariel_db_uri (rewrites the rendered
# config.yml so the agent's ARIEL MCP server and apply_scenarios both use it).
#
# Disable with OSPREY_BENCH_SHARED_DB=1 (falls back to the legacy shared DB).
PG_BIN="${OSPREY_BENCH_PG_BIN:-$HOME/bin/pg16-edb/pgsql/bin}"
PROV_DIR=""
if [ "${OSPREY_BENCH_SHARED_DB:-0}" != "1" ]; then
  # Postgres unquoted identifiers fold to lowercase and forbid '-'/'/'; sanitize.
  CELL_DB="ariel_$(printf '%s' "${SAFE}_seed${SEED}" | tr -C 'a-zA-Z0-9' '_' | tr 'A-Z' 'a-z')"
  CELL_DB="${CELL_DB%_}"
  export OSPREY_ARIEL_DB_URI="postgresql://ariel:ariel@localhost:5432/${CELL_DB}"
  echo ">> per-cell ARIEL DB: $CELL_DB" >&2

  if "$PG_BIN/psql" -h localhost -p 5432 -U ariel -d postgres -v ON_ERROR_STOP=1 \
       -c "DROP DATABASE IF EXISTS ${CELL_DB} WITH (FORCE);" \
       -c "CREATE DATABASE ${CELL_DB} OWNER ariel;" >&2; then
    # Provision schema + seeded (nominal) logbook + embeddings against the
    # per-cell DB, using a throwaway project whose config points at it. Order
    # matters: sim apply purges embeddings, so reembed must come last.
    PROV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ariel_prov_${CELL_DB}.XXXXXX")"
    if "$PY" -m osprey.cli.main build prov --preset control-assistant \
         --skip-deps --skip-lifecycle --output-dir "$PROV_DIR" \
         --set provider=als-apg --set model=haiku >&2; then
      PROV_CFG="$PROV_DIR/prov/config.yml"
      "$PY" - "$PROV_CFG" "$OSPREY_ARIEL_DB_URI" <<'PYEOF' >&2
import sys
path, uri = sys.argv[1], sys.argv[2]
default = "postgresql://ariel:ariel@localhost:5432/ariel"
text = open(path).read()
assert default in text, f"default ARIEL uri not found in {path}"
open(path, "w").write(text.replace(default, uri))
PYEOF
      ( cd "$PROV_DIR/prov" \
        && "$PY" -m osprey.cli.main ariel migrate \
        && "$PY" -m osprey.cli.main sim apply nominal --yes \
        && "$PY" -m osprey.cli.main ariel reembed --model nomic-embed-text --dimension 768 ) >&2 \
        || echo ">> WARNING: per-cell ARIEL provisioning failed; logbook tests in this cell will skip/gate" >&2
    else
      echo ">> WARNING: provisioning project build failed; logbook tests will skip/gate" >&2
    fi
  else
    echo ">> WARNING: could not create per-cell DB ${CELL_DB}; logbook tests will skip/gate" >&2
  fi
fi

START=$(date +%s)
# MUST be `pytest tests/e2e/` (path), NEVER `-m e2e` (causes registry leaks).
# --timeout (pytest-timeout): bound slow tests so a weak model can't stall the
# suite. SIGNAL method (SIGALRM, main thread) FAILS the individual test and lets
# the session CONTINUE — the thread method instead hard-kills the whole pytest
# process on the first overrun, leaving an empty junit (total=0). The e2e async
# tests run via pytest-asyncio in the main thread, so SIGALRM interrupts them.
# Model-matrix scope (issue #259): run the LLM-driven e2e tests that route
# through the forced model-under-test, excluding files that measure nothing
# about the model (own provider axis, no-LLM smokes, hardcoded-model tests).
# The exclusion list with per-file reasons is single-sourced in
# matrix_e2e_config.json — do not enumerate it here (it drifts).
# Still `pytest tests/e2e/` (path) for registry-safe collection; --ignore prunes.
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
MODEL="$MODEL" SEED="$SEED" ROUTE="$ROUTE" PROVIDER="$OSPREY_E2E_FORCE_PROVIDER" \
XML="$XML" JSON="$JSON" \
WALL="$WALL" PYTEST_RC="$PYTEST_RC" "$PY" - <<'PYEOF'
import json, os, xml.etree.ElementTree as ET

xml, jsonp = os.environ["XML"], os.environ["JSON"]
summary = {
    "model": os.environ["MODEL"],
    "seed": int(os.environ["SEED"]),
    "provider": os.environ.get("PROVIDER", ""),
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

# --- per-cell cleanup ----------------------------------------------------
# Drop the per-cell database and provisioning project so 21 cells don't leave
# 21 stale DBs behind. Keep them for post-mortem with OSPREY_BENCH_KEEP_CELL_DB=1.
if [ "${OSPREY_BENCH_KEEP_CELL_DB:-0}" != "1" ] && [ -n "${CELL_DB:-}" ]; then
  "$PG_BIN/psql" -h localhost -p 5432 -U ariel -d postgres \
    -c "DROP DATABASE IF EXISTS ${CELL_DB} WITH (FORCE);" >&2 2>/dev/null || true
  [ -n "$PROV_DIR" ] && rm -rf "$PROV_DIR"
fi

exit 0
