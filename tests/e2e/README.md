# End-to-End Tests

This directory contains end-to-end tests that validate complete Osprey workflows using LLM-based evaluation.

## Overview

E2E tests validate the complete user experience by creating projects from templates, executing queries through the full system, and evaluating results using an LLM judge. Unlike traditional assertions, the LLM judge evaluates workflow outcomes against plain-text expectations, providing flexible validation for complex AI workflows.

## Test Structure

```
tests/e2e/
├── conftest.py          # Fixtures: project factory, LLM judge
├── judge.py             # LLM judge implementation
├── test_tutorials.py    # Tutorial workflow tests
└── README.md            # This file
```

## Requirements

### API Keys (Required)

E2E tests require valid LLM API access:

```bash
export CBORG_API_KEY=sk-your-key-here
# or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### Performance

Tests are slow (30-120 seconds each) due to complete workflow execution and LLM evaluation.

## Running Tests

### Run all E2E tests:

```bash
pytest tests/e2e/ -v -s
```

### Run with verbose judge output:

```bash
pytest tests/e2e/ -v -s --judge-verbose
```

### Run with real-time progress updates:

```bash
pytest tests/e2e/ -v -s --e2e-verbose
```

This shows real-time status updates during test execution so you can see what's actually happening:
- Query execution milestones
- Framework processing steps
- Completion time and response preview

### Run specific test:

```bash
pytest tests/e2e/test_tutorials.py::test_bpm_timeseries_and_correlation_tutorial -v -s
```

### Run channel finder benchmark tests:

```bash
# Run all benchmark tests (both pipelines)
pytest tests/e2e/test_channel_finder_benchmarks.py -v -s

# Run specific pipeline benchmark
pytest tests/e2e/test_channel_finder_benchmarks.py::test_in_context_pipeline_benchmark -v -s
pytest tests/e2e/test_channel_finder_benchmarks.py::test_hierarchical_pipeline_benchmark -v -s
```

### Command-line options:

| Option | Default | Description |
|--------|---------|-------------|
| `--judge-provider` | `cborg` | AI provider for judge |
| `--judge-model` | `anthropic/claude-sonnet` | Model for judge evaluation |
| `--judge-verbose` | `False` | Print detailed judge reasoning |
| `--e2e-verbose` | `False` | Show real-time progress updates during test execution |

## Test Coverage

### Tutorial Tests (`test_tutorials.py`)

- **`test_bpm_timeseries_and_correlation_tutorial`**: Validates channel finding, archiver retrieval, and plotting (~60-90s)
- **`test_simple_query_smoke_test`**: Quick validation of E2E infrastructure (~10-20s)

### Channel Finder Benchmark Tests (`test_channel_finder_benchmarks.py`)

These tests validate that the control-assistant channel finder pipelines meet production-quality benchmarks:

- **`test_in_context_pipeline_benchmark`**: Validates in-context pipeline achieves ≥90% accuracy
  - Tests the in-context pipeline as shown in the tutorial
  - 30 diverse natural language queries
  - Validates precision, recall, F1 score, and consistency

- **`test_hierarchical_pipeline_benchmark`**: Validates hierarchical pipeline achieves ≥90% accuracy
  - Tests the hierarchical pipeline as shown in the tutorial
  - 47 diverse natural language queries
  - Validates multi-level navigation and channel selection

**Success Criteria (per test):**
- ≥80% perfect match rate (F1 score = 1.0)
- ≥95% completion rate (no crashes)
- Overall F1 score ≥ 0.80
- High consistency across runs

**Note**: Run both tests to validate both pipelines, or run individually to test specific pipeline changes.

## Writing New Tests

### Basic template:

```python
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_your_workflow(e2e_project_factory, llm_judge):
    """Test description."""

    # Create and initialize project
    project = await e2e_project_factory(
        name="my-test-project",
        template="control_assistant",
        provider="cborg",
        model="anthropic/claude-haiku"
    )
    await project.initialize()

    # Execute query
    result = await project.query("Your test query")

    # Define expectations
    expectations = """
    The workflow should:
    - Find relevant channels
    - Retrieve data successfully
    - Generate appropriate response
    - Complete without errors
    """

    # Evaluate and assert
    evaluation = await llm_judge.evaluate(result, expectations)
    assert evaluation.passed, f"Failed: {evaluation.reasoning}"
```

### Writing expectations:

Expectations are plain-text descriptions evaluated by the LLM judge. Be specific about required outcomes but allow for acceptable variations:

```python
# Good: Specific but flexible
expectations = """
Should find BPM channels, retrieve 24h of archival data,
and generate time-series plots. Mock data is acceptable.
"""

# Avoid: Too vague
expectations = "Should work correctly"

# Avoid: Too brittle
expectations = "Must find exactly 17 channels named..."
```

### Test markers:

- `@pytest.mark.e2e` - All E2E tests
- `@pytest.mark.e2e_smoke` - Fast smoke tests (~10-20s)
- `@pytest.mark.e2e_tutorial` - Tutorial workflow validation
- `@pytest.mark.e2e_benchmark` - Channel finder benchmark validation (2-8 min per test)
- `@pytest.mark.requires_cborg` - Requires CBORG API key
- `@pytest.mark.slow` - Takes >30 seconds


## Debugging Failed Tests

### Enable verbose judge output:

```bash
pytest tests/e2e/test_tutorials.py::test_name -v -s --judge-verbose
```

Shows the complete judge evaluation including reasoning and confidence scores.

### Check execution trace:

Failed tests print the execution trace showing invoked capabilities, errors, and timing.

### Inspect artifacts:

```python
print(f"Artifacts: {result.artifacts}")
```


## CI/CD Integration

⚠️ E2E tests are not run by default in CI due to API key requirements, execution time, and cost.

To run in CI, store API keys as secrets and trigger via workflow dispatch or schedule:

```yaml
name: E2E Tests

on: workflow_dispatch

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run E2E Tests
        env:
          CBORG_API_KEY: ${{ secrets.CBORG_API_KEY }}
        run: pytest tests/e2e/ -v
```

## Test Reliability

### How Reliable Are LLM Judge Tests?

The LLM judge provides **high reliability** when used correctly:

**Strengths:**
- Flexible evaluation of complex, open-ended outputs
- Can assess intent and quality, not just exact matches
- Reduces brittleness compared to hard-coded assertions
- Provides detailed reasoning for failures

**Safeguards in Place:**
1. **Belt and Suspenders**: LLM judge is supplemented with hard assertions:
   ```python
   # LLM judge evaluation
   assert evaluation.passed, evaluation.reasoning

   # Hard assertion backup checks
   assert len(result.artifacts) >= 2, "Expected at least 2 artifacts"
   assert result.error is None, f"Workflow encountered error: {result.error}"
   ```

2. **Clear Expectations**: Tests use specific, unambiguous expectations rather than vague criteria

3. **Confidence Scoring**: Judge provides 0.0-1.0 confidence scores (we see 0.95+ for passing tests)

4. **Deterministic Core**: Framework execution is deterministic - LLM variance only affects evaluation

**When To Trust The Judge:**
- ✅ High confidence scores (>0.9)
- ✅ Detailed, specific reasoning
- ✅ Backed up by hard assertions
- ✅ Combined with `--e2e-verbose` to see actual execution

**When To Be Skeptical:**
- ⚠️ Low confidence (<0.7) but still passing
- ⚠️ Vague reasoning
- ⚠️ Frequent flapping between pass/fail

For critical workflows, use `--e2e-verbose` to watch execution in real-time and verify the test is actually doing what you expect.

## Troubleshooting

### Project creation failed
- Verify CLI templates are available
- Check `osprey.cli.init_cmd` works manually
- Verify tmp_path permissions

### Gateway initialization failed
- Check registry path configuration
- Verify model configuration
- Check for missing dependencies

### Judge evaluation timed out
- Increase timeout in judge configuration
- Use faster model for evaluation
- Check network connectivity

### Judge always passes or fails
- Review expectations (may be too vague or overly specific)
- Use `--judge-verbose` to see reasoning
- Check if expectations match what the workflow actually does
- Verify expectations are clear and measurable

### Benchmark test fails
- Check benchmark results files for detailed per-query analysis
- Review failed queries to identify patterns
- Verify database and prompts are up to date
- Consider if model temperature or other settings need adjustment
- Run individual benchmark queries manually to debug

## Additional Resources

- `tests/e2e/BENCHMARK_TESTS.md` - Detailed guide to channel finder benchmark tests
- `tests/integration/test_e2e_workflow.py` - Integration tests without LLM judge
- `TESTING_GUIDE.md` - General testing documentation

