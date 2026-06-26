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

## The portable contract: environment variables

There is **no per-test edit**, and **no provider is hard-wired**. The chokepoint
lives in the production e2e harness (`tests/e2e/sdk_helpers.py`,
`tests/e2e/conftest.py`, `tests/e2e/judge.py`), driven entirely by environment
variables. This table is the real, provider-agnostic API; `matrix.py` generates
these per cell from `matrix.yaml` (see below), but nothing stops you from setting
them by hand.

| Env var | Effect | Read in |
| --- | --- | --- |
| `OSPREY_E2E_FORCE_PROVIDER` | build every project with this provider | `sdk_helpers.init_project` |
| `OSPREY_E2E_FORCE_MODEL` | collapse all tiers (haiku/sonnet/opus) → this id | `sdk_helpers._apply_e2e_overrides` |
| `OSPREY_E2E_PROXY_UPSTREAM` | (open models) start the translation proxy + rewrite base URL | `conftest._e2e_translation_proxy` |
| `OSPREY_E2E_PROXY_KEY` | upstream auth for the proxy (`""` for keyless local servers — honored by presence, not truthiness) | `conftest._e2e_translation_proxy` |
| `OSPREY_E2E_BUDGET_SCALE` | multiply per-query `max_budget_usd` (pricier refs need headroom) | `sdk_helpers.e2e_budget_scale` |
| `OSPREY_E2E_JUDGE_MODEL` / `ALS_APG_BASE_URL` | redirect the LLM judge to a reachable endpoint | `judge.py` |
| `OSPREY_E2E_LIVE` | append one JSON line per test as it finishes (live dashboard feed) | `conftest.pytest_runtest_logreport` |

All default to inert, so CI behaviour is byte-for-byte unchanged when unset.

## The declarative layer: `matrix.yaml`

The whole run is one file, `scripts/benchmark/matrix.yaml`, driven by
`matrix.py`. Each `models:` row names a `provider` and a model `id`; the launcher
resolves credentials, derives the route (proxy for OpenAI-protocol models,
direct for Anthropic), wires the LLM judge, expands the seed-major resumable
grid, and hands each (model, seed) cell the env-var contract above. **Adding a
model = adding a row. Adding a provider (deepseek/ollama/vllm/unsloth) = adding a
`providers:` entry.** No script edits — the `case`/lane logic that used to live in
the bash now lives in the config + launcher.

```yaml
providers:
  cborg:   { base_url: https://api.cborg.lbl.gov/v1, key_file: ~/.cborg_key,
             protocol: openai,    judge: { via: cborg,   model: google/claude-haiku-4-5 } }
  als-apg: { base_url: https://llm.gianlucamartino.com, key_file: ~/.als_apg_key,
             protocol: anthropic, judge: { via: als-apg, model: claude-haiku-4-5-20251001 } }
  ds4:     { base_url: http://127.0.0.1:8000/v1, keyless: true,
             protocol: openai,    judge: { via: cborg,   model: google/claude-haiku-4-5 } }
models:
  - { id: gpt-oss-20b,     provider: cborg }                              # 3 seeds (default)
  - { id: claude-opus-4-6, provider: als-apg, seeds: 1, budget_scale: 6 } # pricey ref
  # - { id: deepseek-v4-flash, provider: ds4 }                            # local DeepSeek
```

`protocol` is per-model-overridable — required because CBORG serves `claude-*` on
the Anthropic route and everything else on the OpenAI route under one provider.
`judge.via` names a provider whose endpoint+key back the grader, which is always
a reachable remote, **never the model under test** (so a local `ds4` cell judges
via `cborg`). `budget_scale` multiplies the per-query `$` cap so a pricey
reference model doesn't blow the haiku-tuned ceiling mid-query (local models stay
at the default `1` — free).

## Layer A — portable mechanism

Provider- and host-agnostic. These work anywhere the env-var contract above is
set; nothing in them assumes a particular provider, key file, or machine.

| Script | Role |
| --- | --- |
| `matrix.py [--config FILE] [--only SUBSTR] [--provider P] [--parallel N] [--dry-run]` | parse `matrix.yaml`, expand the seed-major grid, and run one isolated worker per not-yet-done cell. **Resumable** — cells whose summary json exists are skipped. Writes the `results/matrix.log` markers the live dashboard watches. |
| `matrix.yaml` | the whole run, declaratively (providers, models, seeds, budgets, judges). |
| `matrix_dashboard.py --results-dir DIR --out FILE` | render `results/*.json` → a self-contained `dashboard.html` (per-model summary + per-test heatmap). |
| `check_e2e_coverage.py` | derive the pytest `--ignore` list from `matrix_e2e_config.json` and warn if the run stops covering the full e2e kit minus the explicit, documented exclusions. |
| `matrix_e2e_config.json` | single source of truth for the excluded e2e files (each with a reason). |

## Layer B — operator/site glue (examples to copy & adapt)

These encode how *one* operator wired the matrix for the macstudio benchmark box
and the CBORG / als-apg providers. The ssh host, key files, and Postgres paths
are specific to that setup (overridable via `OSPREY_BENCH_*`, see [Host / layout
overrides](#host--layout-overrides)). Copy and adapt them for your own box.

| Script | Role |
| --- | --- |
| `run_e2e_for_model.sh <model> [seed]` | the per-cell **worker** — driven by `matrix.py`, which sets every routing/judge/budget env var. Isolates the ARIEL DB, runs `pytest tests/e2e/`, summarizes JUnit→JSON. Errors if run without the launcher env (use `matrix.py --only <model>` to debug one cell). |
| `matrix_curate_models.py` | curate CBORG's model catalogue into the canonical, chat-capable subject set (fetched via `ssh macstudio`, which is on-network). |
| `matrix_restart.sh` | one-shot clean restart on the benchmark box over ssh: archive prior results, clear stale bytecode, verify Postgres, launch the subject + reference lanes (`matrix.py --provider cborg` / `--provider als-apg`) in tmux. |
| `matrix_dashboard_live.sh [max_seconds]` | local loop: rsync results from the remote every `DASH_INTERVAL`s and re-render until the matrix signals done. |
| `matrix_run.py` | minimal single-tool harness probe (claude CLI → proxy → one model → in-process `read_pv` MCP tool); a fast "does this model call a tool at all" smoke check. |

## Running it

```bash
# See the resolved plan (no run):
scripts/benchmark/matrix.py --dry-run

# One model (substring filter), serially:
scripts/benchmark/matrix.py --only cborg-coder --parallel 1

# The whole grid, then render the dashboard:
scripts/benchmark/matrix.py --parallel 4
scripts/benchmark/matrix_dashboard.py --results-dir results --out dashboard.html

# Full clean restart on the benchmark box + live local dashboard:
bash scripts/benchmark/matrix_restart.sh
scripts/benchmark/matrix_dashboard_live.sh
```

To benchmark a local DeepSeek (`ds4`) server: start it
(`ds4-server --ctx 100000 …`), uncomment the `ds4` rows in `matrix.yaml`, and
`scripts/benchmark/matrix.py --provider ds4 --parallel 1` (local inference is
slow; keep it serial so serving + testing don't contend).

## Prerequisites

- A CBORG key (`~/.cborg_key`) for open models; an als-apg key (`~/.als_apg_key`)
  for the Anthropic reference bracket. A local server (e.g. `ds4`) needs no key.
- A live, seeded **ARIEL Postgres** for the scenario columns (rf_cavity,
  vacuum_burst); `osprey deploy up && osprey ariel migrate && osprey ariel
  quickstart`. Those tests skip with an actionable message if it is absent. The
  scenario tests re-anchor the demo-logbook to "now" themselves at setup (via
  `activate_scenarios`), so no external timestamp-refresh cron is needed.

## Host / layout overrides

The ssh-orchestration scripts default to the macstudio benchmark box. Override:

| Env var | Default | Where read |
| --- | --- | --- |
| `OSPREY_BENCH_REMOTE` | `macstudio` | local (ssh host) |
| `OSPREY_BENCH_REMOTE_REPO` | `$HOME/projects/osprey` | remote |
| `OSPREY_BENCH_PG_BIN` | `$HOME/bin/pg16-edb/pgsql/bin` | remote |
