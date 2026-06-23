#!/usr/bin/env bash
# Restart the FULL CBORG e2e model matrix FRESH on the benchmark box, with every
# bug-fix in place: Postgres up (ARIEL), judge routed to cborg (runner sets
# ALS_APG_BASE_URL + OSPREY_E2E_JUDGE_MODEL), and the ARIEL demo-logbook
# timestamps refreshed before each run (OSPREY_E2E_PRERUN_HOOK). Archives the
# prior results first.
#
# Run from the worktree (drives the remote over ssh):
#   bash scripts/benchmark/cborg_matrix_restart.sh
#
# Host/layout overrides (defaults match the macstudio benchmark box):
#   OSPREY_BENCH_REMOTE        ssh host                       (default: macstudio)
#   OSPREY_BENCH_REMOTE_REPO   remote OSPREY checkout          (default: $HOME/projects/osprey)
#   OSPREY_BENCH_PG_BIN        remote Postgres bin dir         (default: $HOME/bin/pg16-edb/pgsql/bin)
#   OSPREY_BENCH_HOOK          remote ARIEL refresh hook       (default: the committed scripts/benchmark/ariel_refresh_timestamps.sh)
# The OSPREY_BENCH_REMOTE_REPO/PG_BIN/HOOK overrides are read on the REMOTE shell.
set -uo pipefail

ssh "${OSPREY_BENCH_REMOTE:-macstudio}" 'bash -s' <<'REMOTE'
set -uo pipefail
REPO="${OSPREY_BENCH_REMOTE_REPO:-$HOME/projects/osprey}"
PG_BIN="${OSPREY_BENCH_PG_BIN:-$HOME/bin/pg16-edb/pgsql/bin}"
HOOK="${OSPREY_BENCH_HOOK:-$REPO/scripts/benchmark/ariel_refresh_timestamps.sh}"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO" >&2; exit 2; }
# Clear stale bytecode so freshly-synced src/ fixes actually apply. osprey is an
# editable install here; rsync'd .py edits can otherwise be shadowed by stale
# __pycache__ (a build reads the old compiled module), silently running old code.
find src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
[ -x "$HOOK" ] || { echo "FATAL: refresh hook missing at $HOOK" >&2; exit 2; }
"$PG_BIN/pg_isready" -h localhost -p 5432 -U ariel >/dev/null 2>&1 \
  || { echo "FATAL: Postgres not accepting connections on 5432" >&2; exit 2; }

TS=$(date +%Y%m%d_%H%M%S)
if [ -d results ]; then mv results "results_prefix_$TS" && echo "archived old results -> results_prefix_$TS"; fi
mkdir -p results

# clear any stale matrix sessions (leave ariel_refresher running)
for s in e2e_matrix qwen_matrix matrix_subjects matrix_refs \
         ref_claude-haiku-4-5 ref_claude-sonnet-4-6 ref_claude-opus-4-6; do
  tmux kill-session -t "$s" 2>/dev/null
done

# Concurrency: each full-suite run fans out to ~3-4 subagents, so total concurrent
# runs * ~4 = concurrent CBORG calls. The first attempt at 6 runs hit 20-30 concurrent
# -> heavy 429s. 4-wide subjects (~16 concurrent CBORG calls) is the confirmed ceiling;
# WATCH the 429 count and drop via MATRIX_PARALLEL_SUBJ if they appear.
# Subjects: 6 open models x 3 seeds, 4-wide.
tmux new-session -d -s matrix_subjects \
  "cd '$REPO' && OSPREY_E2E_PRERUN_HOOK='$HOOK' \
   MATRIX_MODELS='gpt-oss-20b gemma-4 cborg-coder gpt-oss-120b google/qwen-3-coder google/qwen-3' \
   MATRIX_SEEDS='1 2 3' MATRIX_PARALLEL=${MATRIX_PARALLEL_SUBJ:-4} bash scripts/benchmark/cborg_matrix_driver.sh"

# References: haiku/sonnet/opus x 1 seed, 1-wide (Anthropic ceiling bracket).
# Routed via als-apg (MATRIX_PROVIDER=als-apg), NOT CBORG: CBORG routing degrades
# the agent's channel-finder/scenario behaviour (feedback_hook + sector7 fail on
# CBORG, pass 7/7 on als-apg), which would deflate the reference bar for a
# provider-routing artifact rather than model capability. als-apg model ids have
# no google/ prefix; opus-4-6 is the newest opus als-apg serves.
tmux new-session -d -s matrix_refs \
  "cd '$REPO' && OSPREY_E2E_PRERUN_HOOK='$HOOK' MATRIX_PROVIDER=als-apg \
   MATRIX_MODELS='claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6' \
   MATRIX_SEEDS='1' MATRIX_PARALLEL=${MATRIX_PARALLEL_REF:-1} bash scripts/benchmark/cborg_matrix_driver.sh"

sleep 3
echo "=== sessions ==="; tmux ls | grep -E "matrix_|ariel_refresher"
echo "=== matrix.log ==="; tail -8 results/matrix.log
REMOTE
