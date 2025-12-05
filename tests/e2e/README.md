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

# Run specific e2e test file
pytest tests/e2e/test_code_generator_workflows.py -v

# Run specific e2e test
pytest tests/e2e/test_code_generator_workflows.py::test_basic_generator_simple_code_generation -v

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

### Code Generator Workflows (`test_code_generator_workflows.py`)

Tests different code generation strategies:
- **Basic Generator**: Simple prompt-to-code generation
- **Claude Code Generator**: Advanced codebase-aware generation with examples
- **Robust Profile**: Multi-phase workflow (scan → plan → implement)

All tests use **deterministic assertions** (no LLM judges) to verify:
- Code files are generated
- Required content (e.g., headers) appears in generated code
- PNG artifacts are created
- Workflows complete without errors

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

## Configuration

### API Keys

E2E tests require API access. Set the appropriate environment variable:

```bash
# For CBORG (default)
export CBORG_API_KEY="your-key"

# Or for Anthropic
export ANTHROPIC_API_KEY="your-key"
```

### Additional Dependencies

Some E2E tests require additional dependencies:

```bash
# For MCP capability generation tests
pip install fastmcp

# Claude Code generator is included in core (v0.9.6+)
# No additional installation needed
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
@pytest.mark.requires_cborg
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
      CBORG_API_KEY: ${{ secrets.CBORG_API_KEY }}
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
