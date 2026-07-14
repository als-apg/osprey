#!/usr/bin/env bash
# Restart the FULL CBORG e2e model matrix FRESH on the benchmark box, with every
# bug-fix in place: Postgres up (ARIEL), judge routed to cborg (runner sets
# ALS_APG_BASE_URL + OSPREY_E2E_JUDGE_MODEL). Archives the prior results first.
#
# Run from the worktree (drives the remote over ssh):
#   bash scripts/benchmark/matrix_restart.sh
#
# Host/layout overrides (defaults match the macstudio benchmark box):
#   OSPREY_BENCH_REMOTE        ssh host                       (default: macstudio)
#   OSPREY_BENCH_REMOTE_REPO   remote OSPREY checkout          (default: $HOME/projects/osprey)
#   OSPREY_BENCH_PG_BIN        remote Postgres bin dir         (default: $HOME/bin/pg16-edb/pgsql/bin)
#   MATRIX_PARALLEL_SUBJ/REF/DS4   per-lane parallelism            (default: 4 / 1 / 1)
# The OSPREY_BENCH_REMOTE_REPO/PG_BIN overrides are read on the REMOTE shell.
set -uo pipefail

ssh "${OSPREY_BENCH_REMOTE:-macstudio}" 'bash -s' <<'REMOTE'
set -uo pipefail
REPO="${OSPREY_BENCH_REMOTE_REPO:-$HOME/projects/osprey}"
PG_BIN="${OSPREY_BENCH_PG_BIN:-$HOME/bin/pg16-edb/pgsql/bin}"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO" >&2; exit 2; }
# Clear stale bytecode so freshly-synced src/ fixes actually apply. osprey is an
# editable install here; rsync'd .py edits can otherwise be shadowed by stale
# __pycache__ (a build reads the old compiled module), silently running old code.
find src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
"$PG_BIN/pg_isready" -h localhost -p 5432 -U ariel >/dev/null 2>&1 \
  || { echo "FATAL: Postgres not accepting connections on 5432" >&2; exit 2; }

TS=$(date +%Y%m%d_%H%M%S)
if [ -d results ]; then mv results "results_prefix_$TS" && echo "archived old results -> results_prefix_$TS"; fi
mkdir -p results

# clear any stale matrix sessions
for s in e2e_matrix qwen_matrix matrix_subjects matrix_refs matrix_ds4 \
         ref_claude-haiku-4-5 ref_claude-sonnet-4-6 ref_claude-opus-4-6; do
  tmux kill-session -t "$s" 2>/dev/null
done

# The whole grid (models, providers, seeds, routes, budgets, judges) lives in
# scripts/benchmark/matrix.yaml. Here we just launch the lanes — cborg subjects,
# als-apg refs, and the local ds4 server — each at its own parallelism via the
# launcher's --provider filter. They share results/ and resume independently.
#
# Concurrency: each full-suite run fans out to ~3-4 subagents, so total concurrent
# runs * ~4 = concurrent CBORG calls. The first attempt at 6 runs hit 20-30 concurrent
# -> heavy 429s. 4-wide subjects (~16 concurrent CBORG calls) is the confirmed ceiling;
# WATCH the 429 count and drop via MATRIX_PARALLEL_SUBJ if they appear.
# Subjects: open models on CBORG (OpenAI-protocol -> proxy), 3 seeds, 4-wide.
tmux new-session -d -s matrix_subjects \
  "cd '$REPO' && .venv/bin/python scripts/benchmark/matrix.py \
   --provider cborg --parallel ${MATRIX_PARALLEL_SUBJ:-4}"

# References: Anthropic ceiling bracket via als-apg (NOT CBORG: CBORG routing
# degrades the agent's channel-finder/scenario behaviour — feedback_hook +
# sector7 fail on CBORG, pass 7/7 on als-apg — which would deflate the reference
# bar for a provider-routing artifact rather than model capability). 1-wide to
# keep als-apg load low so the ceiling isn't confounded by rate-limit scatter.
tmux new-session -d -s matrix_refs \
  "cd '$REPO' && .venv/bin/python scripts/benchmark/matrix.py \
   --provider als-apg --parallel ${MATRIX_PARALLEL_REF:-1}"

# Local DeepSeek V4 on the SHARED ds4 server (keyless localhost). 1-wide: the
# 153 GB model serves serially on one GPU, so concurrent cells would only thrash
# it. cborg/als-apg lanes are network calls and don't contend for the local GPU,
# so this lane runs alongside them fine. Skip the lane (with a loud note) if the
# shared server isn't answering — don't queue cells that will all fail.
if curl -fsS -m 10 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  tmux new-session -d -s matrix_ds4 \
    "cd '$REPO' && .venv/bin/python scripts/benchmark/matrix.py \
     --provider ds4 --parallel ${MATRIX_PARALLEL_DS4:-1}"
else
  echo "WARNING: ds4 server not answering on http://127.0.0.1:8000 — skipping" >&2
  echo "         the ds4 lane. Check: sudo launchctl print system/com.ds4.server" >&2
  echo "         and /Users/Shared/ds4-setup-for-thellert.md" >&2
fi

sleep 3
echo "=== sessions ==="; tmux ls | grep -E "matrix_"
echo "=== matrix.log ==="; tail -8 results/matrix.log
REMOTE
