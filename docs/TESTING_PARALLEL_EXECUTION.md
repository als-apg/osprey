# Testing Guide: Parallel Execution Feature

This guide explains how to test the parallel execution changes before submitting the PR.

## Quick Testing Options

### Option 1: Run the Test Suite (Recommended)

The parallel execution feature includes comprehensive tests. Run them to verify everything works:

```bash
cd osprey

# Run only the parallel execution tests (52 tests)
pytest tests/ -k parallel -v

# Or run specific test files
pytest tests/base/test_decorators_parallel_execution.py \
       tests/infrastructure/test_router_parallel_execution.py \
       tests/integration/test_parallel_execution_integration.py \
       tests/state/test_execution_step_results_reducer.py \
       tests/utils/test_config_parallel_execution.py -v

# Or run all tests to ensure no regressions (1000+ tests)
pytest tests/ -v
```

**Expected result**: All 52 parallel execution tests should pass ✅

### Option 2: Quick Pre-commit Check

Run the quick validation script:

```bash
cd osprey
./scripts/quick_check.sh
```

This will:
- Auto-fix code formatting
- Run fast unit tests
- Duration: < 30 seconds

### Option 3: Full CI Check (Before Pushing)

Replicate the entire GitHub Actions CI workflow locally:

```bash
cd osprey
./scripts/ci_check.sh
```

This runs:
1. Linting (ruff + mypy)
2. Full test suite with coverage
3. Documentation build
4. Package build validation

**Duration**: 2-3 minutes

### Option 4: Pre-merge Validation

Final comprehensive check before creating the PR:

```bash
cd osprey
./scripts/premerge_check.sh upstream/main
```

This checks for:
- Debug code (print, breakpoint, pdb)
- Commented-out code
- Hardcoded secrets
- CHANGELOG updates
- Type hints
- TODO/FIXME comments with issue links
- Code formatting
- Test suite

## Testing with a Real Agent

You can test the parallel execution feature with an actual Osprey agent:

### Step 1: Create a Test Agent

```bash
cd osprey

# Create a test agent (if you don't have one)
osprey create test-parallel-agent
cd test-parallel-agent
```

### Step 2: Enable Parallel Execution

Edit `config.yml` and enable the feature:

```yaml
execution_control:
  agent_control:
    parallel_execution_enabled: true  # Enable parallel execution
```

### Step 3: Run the Agent

```bash
# Run with a query that could benefit from parallel execution
osprey run --query "your test query here"

# Or use interactive mode
osprey chat
```

### Step 4: Verify Parallel Execution

Check the logs for parallel execution indicators:

```
# You should see logs like:
Capability channel_finding executing step 0 (key=step_1)
Capability time_range_parsing executing step 1 (key=step_2)
```

## What to Look For

### ✅ Success Indicators

1. **All tests pass**: No failures in the test suite
2. **No regressions**: Existing tests still pass
3. **Proper state updates**: `execution_step_results` is populated correctly
4. **Sequential mode works**: With `parallel_execution_enabled: false`, execution is sequential
5. **Parallel mode works**: With `parallel_execution_enabled: true`, independent steps can execute

### ❌ Failure Indicators

1. **Test failures**: Any test fails
2. **Import errors**: Missing dependencies
3. **Type errors**: mypy reports issues
4. **Linting errors**: ruff reports issues
5. **State corruption**: Execution plan doesn't complete properly

## Specific Test Cases to Verify

### Test 1: Configuration Loading

```bash
# Run the config test
pytest tests/utils/test_config_parallel_execution.py -v
```

**Expected**: Config properly loads `parallel_execution_enabled` flag

### Test 2: State Reducer

```bash
# Run the state reducer test
pytest tests/state/test_execution_step_results_reducer.py -v
```

**Expected**: State reducer properly merges step results

### Test 3: Router Logic

```bash
# Run the router test
pytest tests/infrastructure/test_router_parallel_execution.py -v
```

**Expected**: Router correctly handles both sequential and parallel modes

### Test 4: Decorator Behavior

```bash
# Run the decorator test
pytest tests/base/test_decorators_parallel_execution.py -v
```

**Expected**: Decorators handle parallel mode correctly (no step index increment)

### Test 5: Integration Test

```bash
# Run the integration test
pytest tests/integration/test_parallel_execution_integration.py -v
```

**Expected**: End-to-end workflow executes correctly in both modes

## Troubleshooting

### Issue: Tests fail with import errors

**Solution**: Install the package in development mode:
```bash
cd osprey
pip install -e ".[dev,docs]"
```

### Issue: "Virtual environment not found"

**Solution**: Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Issue: Linting errors

**Solution**: Auto-fix with ruff:
```bash
cd osprey
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Issue: Type checking errors

**Solution**: Run mypy to see specific issues:
```bash
cd osprey
mypy src/osprey
```

## Performance Testing (Optional)

To verify actual performance improvements:

1. Create a test agent with parallel-friendly workflow
2. Run with `parallel_execution_enabled: false` and time it
3. Run with `parallel_execution_enabled: true` and time it
4. Compare execution times

**Expected**: Parallel mode should be faster for workflows with independent steps

## Before Submitting PR

Run this final checklist:

```bash
cd osprey

# 1. Run all tests
pytest tests/ -v

# 2. Run linting
ruff check src/ tests/
ruff format --check src/ tests/

# 3. Run type checking
mypy src/osprey

# 4. Run pre-merge check
./scripts/premerge_check.sh upstream/main

# 5. Verify no debug code
grep -r "print(" src/osprey/  # Should find nothing
grep -r "breakpoint()" src/osprey/  # Should find nothing
```

All checks should pass ✅

## Summary

**Minimum testing before PR:**
1. ✅ Run parallel execution tests: `pytest tests/*parallel*.py -v`
2. ✅ Run quick check: `./scripts/quick_check.sh`
3. ✅ Run CI check: `./scripts/ci_check.sh`

**Recommended additional testing:**
4. ✅ Test with real agent (optional but recommended)
5. ✅ Run pre-merge check: `./scripts/premerge_check.sh upstream/main`

If all these pass, your PR is ready! 🚀