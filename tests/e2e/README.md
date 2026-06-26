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
- LangGraph state management
- File system operations

Running e2e tests together with unit tests can cause:
- Registry state leakage between tests
- Service initialization conflicts
- Async fixture lifecycle issues

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

Each file's docstring lists the precise requirements; missing backends
yield a `skipped` not a `failed`. To run them locally, bring up the
relevant service via the OSPREY ``docker-compose`` services tree
(``services/postgres``, etc.) and re-run with ``pytest tests/e2e/<file> -v``.

## Per-PR LLM cost expectation

The ``e2e-tests`` job is now a hard-fail gate (was advisory through
2026-04). Per PR, expect ~90 channel-finder MCP queries from
``test_channel_finder_mcp_benchmarks.py`` (30 queries × 3 pipelines:
hierarchical, middle-layer, in-context) plus ~14 SDK-driven safety
scenario runs. At als-apg/haiku rates the channel-finder pass is the
dominant cost; the safety pass is cheap. Channel-finder thresholds were
re-tuned against als-apg/haiku in 2026-04 — if the model or provider
changes, re-tune (see test docstrings for the calibration date stamp).

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
2. **Don't use LLM judges** - they're slow, expensive, and non-deterministic
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
