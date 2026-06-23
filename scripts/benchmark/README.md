# Model-capability benchmark (`scripts/benchmark/`)

Runs the **full OSPREY end-to-end suite** (`tests/e2e/`) against a matrix of
models and renders per-test pass rates to a self-contained `dashboard.html`.
Originally built for issue #259 (CBORG self-hosted models in Claude Code).

The point is a **reference ceiling**: the Anthropic Claude models (Opus/Sonnet)
should score ~100% on a clean stack. A sub-100% reference cell is almost always
an *our-end* bug, not model incapacity — so the benchmark doubles as a
whole-stack integration test. (It surfaced the hook-interpreter dark-layer
safety bug, the python-execute kill-switch bypass, the tier-1 VAC data gap, and
the ARIEL Postgres prerequisite, among others.)

## How a run is forced onto a model

There is **no per-test edit**. The chokepoint lives in the production e2e
harness (`tests/e2e/sdk_helpers.py`, `tests/e2e/conftest.py`, `tests/e2e/judge.py`),
driven entirely by environment variables:

| Env var | Effect | Read in |
| --- | --- | --- |
| `OSPREY_E2E_FORCE_PROVIDER` | build every project with this provider | `sdk_helpers.init_project` |
| `OSPREY_E2E_FORCE_MODEL` | collapse all tiers (haiku/sonnet/opus) → this id | `sdk_helpers._apply_e2e_overrides` |
| `OSPREY_E2E_PROXY_UPSTREAM` | (open models) start the translation proxy + rewrite base URL | `conftest._e2e_translation_proxy` |
| `OSPREY_E2E_BUDGET_SCALE` | multiply per-query `max_budget_usd` (pricier refs need headroom) | `sdk_helpers.e2e_budget_scale` |
| `OSPREY_E2E_JUDGE_MODEL` / `ALS_APG_BASE_URL` | redirect the LLM judge to a reachable endpoint | `judge.py` |
| `OSPREY_E2E_LIVE` | append one JSON line per test as it finishes (live dashboard feed) | `conftest.pytest_runtest_logreport` |

All default to inert, so CI behaviour is byte-for-byte unchanged when unset.

## Toolchain

| Script | Role |
| --- | --- |
| `run_e2e_for_model.sh <model> [seed]` | run the full e2e suite for ONE model; emits `results/<model>__seed<n>.{xml,json,live.jsonl}`. Routes open models through the proxy, `claude-*` direct, `MATRIX_PROVIDER=als-apg` for the Anthropic references. |
| `cborg_matrix_driver.sh` | loop `run_e2e_for_model.sh` over a `MATRIX_MODELS` × `MATRIX_SEEDS` grid with bounded `MATRIX_PARALLEL`. **Resumable** — combos whose summary json exists are skipped. |
| `cborg_matrix_restart.sh` | one-shot clean restart on the benchmark box over ssh: archive prior results, clear stale bytecode, verify Postgres, launch subjects + reference tmux sessions. |
| `cborg_dashboard.py --results-dir DIR --out FILE` | render `results/*.json` → a self-contained `dashboard.html` (per-model summary + per-test heatmap). |
| `cborg_dashboard_live.sh [max_seconds]` | local loop: rsync results from the remote every `DASH_INTERVAL`s and re-render until the matrix signals done. |
| `check_e2e_coverage.py` | derive the pytest `--ignore` list from `matrix_e2e_config.json` and warn if the run stops covering the full e2e kit minus the explicit, documented exclusions. |
| `matrix_e2e_config.json` | single source of truth for the excluded e2e files (each with a reason). |
| `ariel_refresh_timestamps.sh` | re-anchor the demo-logbook arc (DEMO-026 → `NOW() - 1h`) so the scenario columns don't drift out of "today"; wired via `OSPREY_E2E_PRERUN_HOOK` (the restart script points `OSPREY_BENCH_HOOK` here by default). |

## Running it

```bash
# Single model, locally:
scripts/benchmark/run_e2e_for_model.sh cborg-coder 1

# A grid:
MATRIX_MODELS="gpt-oss-20b gemma-4" MATRIX_SEEDS="1 2" MATRIX_PARALLEL=4 \
  scripts/benchmark/cborg_matrix_driver.sh

# Full clean restart on the benchmark box + live local dashboard:
bash scripts/benchmark/cborg_matrix_restart.sh
scripts/benchmark/cborg_dashboard_live.sh
```

## Prerequisites

- A CBORG key (`~/.cborg_key`) for open models; an als-apg key for the Anthropic
  reference bracket (`MATRIX_PROVIDER=als-apg`).
- A live, seeded **ARIEL Postgres** for the scenario columns (rf_cavity);
  `osprey deploy up && osprey ariel migrate && osprey ariel quickstart`. The
  scenario test skips with an actionable message if it is absent.
- The scenario logbook timestamps drift out of "today" in a static DB;
  `ariel_refresh_timestamps.sh` re-anchors them. The matrix wires it via
  `OSPREY_E2E_PRERUN_HOOK` automatically (`cborg_matrix_restart.sh` defaults
  `OSPREY_BENCH_HOOK` to the committed copy). The fully self-contained form
  (an autouse pytest fixture) is PR_PLAN Fix 4, still to do.

## Host / layout overrides

The ssh-orchestration scripts default to the macstudio benchmark box. Override:

| Env var | Default | Where read |
| --- | --- | --- |
| `OSPREY_BENCH_REMOTE` | `macstudio` | local (ssh host) |
| `OSPREY_BENCH_REMOTE_REPO` | `$HOME/projects/osprey` | remote |
| `OSPREY_BENCH_PG_BIN` | `$HOME/bin/pg16-edb/pgsql/bin` | remote |
| `OSPREY_BENCH_HOOK` | `$REPO/scripts/benchmark/ariel_refresh_timestamps.sh` | remote |
