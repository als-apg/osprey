# End-to-End (E2E) Tests

## 🚨 CRITICAL: How to Run These Tests

```bash
# ✅ CORRECT: Always use direct path
pytest tests/e2e/ -v

# ❌ WRONG: Do NOT use -m e2e marker
pytest -m e2e  # This causes test collection issues and failures!
```

**Why?** Using `-m e2e` causes pytest to collect tests in the wrong order, leading to registry initialization failures and mysterious "Registry contains no nodes" errors. Always run e2e tests using the direct path `pytest tests/e2e/`.

---

## Overview

E2E tests validate complete workflows through the Osprey framework including:
- Code generation with different generators (basic, Claude Code)
- Full framework initialization and registry loading
- Real LLM API calls (requires API keys)
- Actual code execution and artifact generation

## Running E2E Tests

### ⚠️ IMPORTANT: Test Isolation

E2E tests must be run **separately** from unit tests due to complex framework initialization and registry state management.

**✅ Correct way to run e2e tests:**
```bash
# Run all e2e tests
pytest tests/e2e/ -v

# Run MCP capability generation tests
pytest tests/e2e/test_mcp_capability_generation.py -v

# Run only smoke tests (faster)
pytest tests/e2e/ -m e2e_smoke -v

# Run with verbose output
pytest tests/e2e/ -v -s --e2e-verbose
```

**❌ DO NOT run with unit tests:**
```bash
# This will cause registry isolation issues
pytest tests/  # Runs both unit AND e2e - will fail
```

### Why Separate?

E2E tests create full framework instances with:
- Complete registry initialization
- Service registration (Python executor, code generators, etc.)
- File system operations

Running e2e tests together with unit tests can cause:
- Registry state leakage between tests
- Service initialization conflicts
- Async fixture lifecycle issues

## Benchmark-lane markers (required for matrix-scope tests)

The model-benchmark matrix (`scripts/benchmark/`) scores two lanes separately,
declared by a pytest marker on every e2e test that is not in the matrix
exclusion list (`scripts/benchmark/matrix_e2e_config.json`):

- `@pytest.mark.agentic_benchmark` — the test is a genuine model-capability
  task: the agent must choose and sequence tools, reason about results, and
  produce an answer the assertions (or an LLM judge) actually evaluate. A weak
  model plausibly fails it.
- `@pytest.mark.harness_benchmark` — an agent runs, but the assertion is
  model-independent OSPREY behavior (safety hook fires, write is blocked,
  audit record written). Any responding model passes it.

**When you add an e2e test**, either mark it with exactly one of these or add
its file to the exclusion config with a reason. The lane gate
(`scripts/benchmark/check_e2e_coverage.py --check-lanes`, also run as a unit
test in `tests/benchmark/test_matrix_lanes.py` and at matrix-cell startup)
fails on unmarked in-scope tests.

## Test Categories

### Tutorial Workflows (`test_tutorials.py`)

Tests complete tutorial experiences:
- **BPM Timeseries Tutorial**: Multi-capability workflow (channel finding + archiver + plotting)
- **Hello World Weather**: Beginner tutorial with mock API integration
- **Simple Smoke Test**: Quick validation of basic framework functionality

Uses LLM judges to evaluate:
- Workflow completion
- Expected artifacts produced
- Response quality

### MCP Capability Generation (`test_mcp_capability_generation.py`)

Tests MCP (Model Context Protocol) integration pipeline:
- **Full MCP Workflow**: Generate MCP server → Launch server → Generate capability → Execute query
- **Simulated Mode**: Quick smoke test using built-in simulated tools

Validates:
- MCP server generation and launch
- Capability generation from live MCP server
- Automatic registry integration
- End-to-end query execution using MCP capability
- LLM judge verification of responses

### Channel Finder Benchmarks (`test_channel_finder_benchmarks.py`)

Tests hierarchical channel finder performance and accuracy:
- Pattern matching across different facility naming conventions
- Benchmark validation against known test datasets
- Performance metrics for large-scale channel queries

## Local-only tests (skipped in CI)

The default GitHub Actions runner has Docker, Python, and ``ALS_APG_API_KEY``
— but no Postgres, no Ollama, no Confluence access, and no SQLite-backed
research databases. These tests skip cleanly in CI but are runnable
locally with the right backend stack:

| File | Skip reason in CI | Local requirements |
| --- | --- | --- |
| ``claude_code/test_agent_delegation.py`` | All 6 sub-agents need backend services | ARIEL Postgres @ 5432, AccelPapers SQLite (``$ACCELPAPERS_DB``), DePlot @ 8095, Confluence (``$CONFLUENCE_ACCESS_TOKEN``), MML SQLite (``$MATLAB_MML_DB``) |
| ``test_ariel_search.py`` | RAG sub-tests need Ollama embeddings | Postgres @ 5433, Ollama @ 11434 with ``nomic-embed-text`` pulled |
| ``test_ariel_e2e_pipeline.py`` | Postgres ingestion pipeline | Postgres @ 5432/5433 (config-dependent) |
| ``test_rf_cavity_correlation_scenario.py`` | Logbook arc needs a running ARIEL DB | ARIEL Postgres @ 5432 (seeded automatically at setup from the scenario bundle; see below) |
| ``test_vacuum_burst_scenario.py`` | Skipped on CI alongside its sibling | ARIEL Postgres @ 5432 (seeded automatically at setup with the ambient logbook) |

Each file's docstring lists the precise requirements; missing backends
yield a `skipped` not a `failed`. To run them locally, bring up the
relevant service via the OSPREY ``docker-compose`` services tree
(``services/postgres``, etc.) and re-run with ``pytest tests/e2e/<file> -v``.

## Scenario tests & the simulation engine

The control_assistant scenario tests (``test_vacuum_burst_scenario.py`` and
``test_rf_cavity_correlation_scenario.py``) get
their archiver ground truth from the **data-driven simulation engine**, not from
hard-coded connector code. Scenarios are **self-contained bundles** under
``data/simulation/scenarios/<name>/`` — each owns its telemetry overlay
(``scenario.json``) and, optionally, its logbook narrative (``logbook.json``).
Each test builds a project from the ``control_assistant`` preset and calls
``activate_scenarios(project, "<name>"...)`` to compose and apply one or more
fault bundles (``vacuum-burst`` / ``rf-thermal``) before running the operator
prompt. They build at **tier 3** so every simulated channel is discoverable
through the channel finder — the vacuum gauges live only in tier 2+.

The scenarios' statistical signatures (SR07/DCCT anti-correlation, the C1
excursion positions, derived-channel consistency) are pinned deterministically
and cheaply — no LLM — by ``tests/simulation/test_control_assistant_scenarios.py``
and ``test_scenario_composition.py``. Run those first when a scenario e2e
regresses: if the contract tests pass, the data substrate is sound and the miss
is the agent's (the scenario tests are ``flaky(reruns=2)`` to absorb the rare
stochastic bail-out); if they fail, fix the scenario bundle — never the e2e
prompts.

**Logbook seeding (automatic).** ``activate_scenarios`` calls
``apply_scenarios(seed_logbook=True)``: it composes the active scenarios, writes
the simulator state with a shared apply-time anchor, and **purges + reseeds**
the ARIEL logbook from the active bundles' own entries — so the narrative the
agent searches always matches the telemetry it reads, against one clock. No
manual ``purge && ingest`` step, and no stale/wrong-preset DB footgun: each test
reseeds deterministically at setup. A running ARIEL Postgres at 5432 is still
required (the seed has to land somewhere); only the seeding is automatic.

## Scan-stack e2e family (VA + bluesky bridge + Tiled)

These deploy a real stack — virtual accelerator, bluesky bridge, Tiled — and
drive it. They are heavy (Docker builds, minutes each), so each gets its own CI
lane rather than riding the bulk `e2e-tests` job, which `--ignore`s every one of
them. The map below records which lane **runs** which file; `ci.yml` is the
source of truth.

| Module | CI lane |
| --- | --- |
| `test_bluesky_deploy.py` | `bluesky-deploy-e2e` |
| `test_bluesky_panels_deploy.py` | `bluesky-panels-deploy-e2e` |
| `test_va_substrate_equivalence.py` | `va-substrate-equivalence-e2e` |
| `test_orm_roundtrip.py`, then `test_grid_scan_roundtrip.py` | `orm-roundtrip-e2e` (both, sequentially, one deploy per file) |
| `test_bluesky_queue_e2e.py` | `bluesky-queue-e2e` |
| `test_tiled_roundtrip.py` | `tiled-roundtrip-e2e` |
| `test_bluesky_catalog_e2e.py` | `bluesky-catalog-e2e` |
| `test_bluesky_sandbox_escape_e2e.py` | `bluesky-sandbox-escape-e2e` |
| `test_scan_stack_agentic.py` | `scan-agentic-e2e` |

Two entries are worth reading twice. `test_grid_scan_roundtrip.py` is **adopted
into** `orm-roundtrip-e2e` rather than given a lane of its own: it runs as a
second, sequential step after `test_orm_roundtrip.py`, so the two stacks never
contend for the same CA port (5064). And `bluesky-queue-e2e` drives the queue
stack with **no LLM in the loop** — it is a plain protocol test, unlike the
agentic lane below.

### `test_scan_stack_agentic.py` — the agentic member

The only module in the family that puts an **agent** in the loop. An operator
asks in plain language for a measurement on a **healthy** stack; the agent must
discover the tools, stage a draft of the right plan class, queue it, start the
queue, read the run back, and say what it shows. Nothing is broken, so there is
no hidden answer: a run that "concludes" a fault is wrong, and a run that never
took a measurement has nothing to conclude from.

It grades in the two layers described under [Best Practices](#best-practices) —
a deterministic floor over the tool trace, plus one judge criterion over the
prose. The floor is a **plan-class predicate**, never a plan name: correctors
driven against BPM detectors is the orbit-response class, two or more distinct
setpoint axes is the grid class. A structurally equivalent plan under a
different name still passes, and the two predicates are mutually exclusive, so
neither live test can be satisfied by the other's run.

Both halves are dry-verified offline against the same contracts the live tests
use, so you can iterate without Docker or a live run:

```bash
# Structural floor — hand-built traces. No Docker, no API key, no agent.
.venv/bin/pytest tests/e2e/test_scan_stack_agentic.py -k floor

# Judge rubric — hand-written conclusions, one failing control per criterion.
# Needs the judge provider's credentials (ALS_APG_API_KEY), nothing else.
.venv/bin/pytest tests/e2e/test_scan_stack_agentic.py -k judge
```

## Per-PR LLM cost expectation

The ``e2e-tests`` job is a hard-fail gate — as is every e2e lane: the
``all-checks-passed`` gate has no advisory tier. Per PR, expect ~14
SDK-driven safety scenario runs plus the SDK workflow/delegation
pipelines (cheap at als-apg/haiku rates). The channel-finder MCP
benchmarks (~90 queries from ``test_channel_finder_mcp_benchmarks.py``,
30 queries × 3 pipelines) no longer run per PR — they run on demand in
the ``channel-finder-benchmarks`` job (Actions tab "Run workflow" with
``run_benchmarks``, or ``gh workflow run ci.yml -f run_benchmarks=true``);
trigger it before a release or after touching the channel-finder/MCP
code, and read the scores in the job log. Channel-finder thresholds were
re-tuned against
als-apg/haiku in 2026-04 — if the model or provider changes, re-tune
(see test docstrings for the calibration date stamp).

## Configuration

### API Keys

E2E tests require API access. Set the appropriate environment variable:

```bash
# For ALS-APG (CI default — AWS Bedrock proxy reachable from anywhere)
export ALS_APG_API_KEY="your-key"

# For CBORG (local dev only — IP allowlist blocks GitHub Actions runners)
export CBORG_API_KEY="your-key"

# Or for Anthropic
export ANTHROPIC_API_KEY="your-key"
```

### Provider × model matrix (opt-in)

`test_llm_providers.py` is skipped by default. It is the one file that makes
paid API calls on **every** provider whose key is in the environment (all other
e2e tests pick a single provider by preference order), so it only runs when
explicitly requested — typically after changing `src/osprey/models/providers/`
or adding a provider:

```bash
OSPREY_LLM_MATRIX_ENABLE=1 pytest tests/e2e/test_llm_providers.py -v
```

### Additional Dependencies

Some E2E tests require additional dependencies:

```bash
# For MCP capability generation tests
pip install fastmcp

# Claude Code generator is included in core — no additional installation needed
```

### Test Options

```bash
# Use specific LLM provider for judge evaluations
pytest tests/e2e/ --judge-provider=anthropic --judge-model=claude-sonnet-4

# Show detailed judge reasoning
pytest tests/e2e/ --judge-verbose

# Show real-time progress during test execution
pytest tests/e2e/ --e2e-verbose
```

## Writing E2E Tests

### Template

```python
import pytest

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_my_workflow(e2e_project_factory):
    """Test description."""
    # Create test project
    project = await e2e_project_factory(
        name="test-my-feature",
        template="control_assistant",
        registry_style="extend"
    )

    # Initialize framework
    await project.initialize()

    # Execute query
    result = await project.query("Your test query")

    # Assert deterministic outcomes
    assert result.error is None
    assert len(result.artifacts) > 0
    # ... more assertions
```

### Best Practices

1. **Use deterministic assertions** - check files created, content present, no errors
2. **Grade in two layers** - when a test has to judge an agent's *work*, put the
   load-bearing grading in a deterministic structural floor over the tool trace
   ("did it actually do the thing?") and give an LLM judge only what a trace
   cannot see ("did it describe and interpret what it did, without inventing
   findings?"). Never let a judge carry a claim a floor could pin. Tell the
   judge what the floor already covered so it does not re-penalize it, and
   dry-verify both halves offline — the floor against hand-built traces, the
   judge against one passing conclusion plus one failing control per criterion —
   before you spend a live run. See `test_scan_stack_agentic.py`.
3. **Mark appropriately** - use `@pytest.mark.e2e`, `@pytest.mark.slow`, `@pytest.mark.requires_*`
4. **Clean validation** - verify actual outputs (files, code content) not just LLM responses

## CI/CD Integration

For CI pipelines, run e2e tests as a separate job:

```yaml
# .github/workflows/tests.yml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ -m "not e2e" -v

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/e2e/ -v
    env:
      ALS_APG_API_KEY: ${{ secrets.ALS_APG_API_KEY }}
```

## Troubleshooting

### "Python executor service not available in registry"

This occurs when tests run together and registry state leaks between tests. **Solution: Run e2e tests separately** as documented above.

### Tests pass individually but fail in batch

This is expected due to registry isolation issues. Each e2e test works individually because it gets a fresh registry. When run in batch, subsequent tests may fail. **Solution: This is acceptable** - e2e tests are meant to be run as their own test suite.

### Slow execution

E2E tests make real LLM API calls. Typical execution times:
- Single test: 20-40 seconds
- Full e2e suite: 2-5 minutes

Use `-k` to run specific tests during development:
```bash
pytest tests/e2e/ -k "basic_generator" -v
```
