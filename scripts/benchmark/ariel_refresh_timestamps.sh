#!/usr/bin/env bash
# Re-anchor the ARIEL demo-logbook arc so DEMO-026 (the "beam dump") lands at
# NOW() - 1h, with DEMO-027/028 (investigation / repair) following. Run before
# each scenario e2e run / matrix row via OSPREY_E2E_PRERUN_HOOK.
#
# Why: init_project() rebases the logbook SEED file to "now", but the shared
# ARIEL Postgres is ingested ONCE and then its timestamps freeze. As days pass
# DEMO-026 drifts out of "today", so the rf_cavity agent's "what happened this
# morning" search returns nothing and (even Opus) correctly concludes "no dump
# today" and fails the scenario. See SCENARIO_E2E_INVESTIGATION.md.
#
# This is the committed stopgap (PR_PLAN Fix 4). The fully self-contained form
# is an autouse pytest fixture on the scenario test that re-anchors the ingested
# rows itself; this script keeps the matrix relaunch from depending on a
# host-local file in the meantime.
#
# Connection defaults to the standard ARIEL DB; override via PGHOST/PGPORT/
# PGUSER/PGDATABASE, and the psql binary via OSPREY_BENCH_PG_BIN.
set -uo pipefail
PSQL="${OSPREY_BENCH_PG_BIN:-$HOME/bin/pg16-edb/pgsql/bin}/psql"
[ -x "$PSQL" ] || PSQL=psql   # fall back to a psql on PATH
"$PSQL" -h "${PGHOST:-localhost}" -p "${PGPORT:-5432}" -U "${PGUSER:-ariel}" -d "${PGDATABASE:-ariel}" <<'EOSQL'
UPDATE enhanced_entries SET timestamp = NOW() - INTERVAL '1 hour' WHERE entry_id = 'DEMO-026';
UPDATE enhanced_entries SET timestamp = NOW() + INTERVAL '1 day 6 hours 40 minutes' WHERE entry_id = 'DEMO-027';
UPDATE enhanced_entries SET timestamp = NOW() + INTERVAL '2 days 10 hours 40 minutes' WHERE entry_id = 'DEMO-028';
SELECT entry_id, timestamp, LEFT(raw_text, 50) FROM enhanced_entries WHERE entry_id IN ('DEMO-026','DEMO-027','DEMO-028') ORDER BY timestamp;
EOSQL
