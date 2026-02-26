---
workflow: testing-workflow
category: code-quality
applies_when: [writing_code, adding_features, fixing_bugs, before_commit]
estimated_time: "varies (unit: seconds, e2e: minutes + cost)"
ai_ready: true
related: [pre-merge-cleanup, feature-development, bug-fix]
skill_description: >-
  Comprehensive testing guidance for Osprey projects. Use when the user wants
  to write tests, run tests, understand test strategy, or needs help with
  unit tests, integration tests, or e2e tests. Covers pytest patterns, markers,
  fixtures, mocking, async testing, and cost-effective test strategies.
---

# Testing Workflow

Comprehensive guide for testing in Osprey, with emphasis on cost-effective test strategy.

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant:**

```
Following @src/osprey/assist/tasks/testing-workflow/instructions.md, help me write tests for [COMPONENT/FEATURE].

Analyze the code and determine:
1. What type of tests are needed (unit, integration, or e2e)?
2. What are the critical paths that must be tested?
3. What can be mocked vs what needs real services?
4. Generate appropriate test cases with proper markers

Remember:
- Unit tests are free and fast - use liberally
- E2E tests cost $0.10-$0.25 and take 2-5 minutes - use sparingly
- Only use e2e tests for workflows that REQUIRE real LLMs

Show me the test file with appropriate pytest markers and fixtures.
```

**Related workflows**: [pre-merge-cleanup.md](pre-merge-cleanup.md)

---

## 💰 Test Economics: Choose Wisely

### The Cost of Testing

| Test Type | Speed | Cost | When to Use |
|-----------|-------|------|-------------|
| **Unit** | <5 seconds | Free | Always - test logic, functions, classes |
| **Integration** | <30 seconds | Free | Test component interactions (with mocks) |
| **E2E** | 2-5 minutes | **$0.10-$0.25** | **ONLY** for critical LLM workflows |

### 🚨 E2E Tests Are Precious

**E2E tests are expensive in time and money.** Use them only when:

✅ **Must test real LLM behavior** - No mocking can replicate LLM decision-making
✅ **Critical user workflows** - Tutorial validation, core orchestration paths
✅ **Integration between multiple LLM-driven components** - Capability chains, routing logic
✅ **LLM judge evaluation required** - Response quality verification

❌ **Do NOT use e2e tests for:**
- Pure Python logic (use unit tests)
- Data transformations (use unit tests)
- Error handling (use unit tests with mocks)
- Configuration validation (use unit tests)
- Service initialization (use integration tests with mocks)

**Rule of thumb**: If you can test it without calling OpenAI/Anthropic APIs, write a unit test.

---

## 🎯 Quick Start

### Running Tests

```bash
# Unit tests (fast, free) - run these constantly
pytest tests/ --ignore=tests/e2e -v

# E2E tests (slow, expensive) - run before releases
pytest tests/e2e/ -v

# Run specific test file
pytest tests/capabilities/test_my_capability.py -v

# Run specific test
pytest tests/capabilities/test_my_capability.py::test_my_function -v
```

### ⚠️ CRITICAL: Never Run Tests Together

```bash
# ❌ WRONG: Do NOT mix unit and e2e tests
pytest tests/  # Causes registry conflicts!

# ❌ WRONG: Do NOT use -m e2e marker
pytest -m e2e  # Causes collection issues!

# ✅ CORRECT: Always run separately
pytest tests/ --ignore=tests/e2e -v  # Unit tests
pytest tests/e2e/ -v                  # E2E tests separately
```

**Why?** E2E tests use real registry while unit tests use mocked registry. Running together causes state contamination.

---

## 📊 Test Types and Strategy

### 1. Unit Tests (Use Liberally)

**Purpose**: Test individual functions, classes, and methods in isolation

**Characteristics**:
- ✅ Fast (milliseconds to seconds)
- ✅ Free (no API calls)
- ✅ Deterministic (same input = same output)
- ✅ Easy to debug
- ✅ Run on every commit

**When to use**:
- Pure functions and business logic
- Data transformations
- Validation logic
- Error handling
- Configuration parsing
- Utility functions

**Example**:
```python
# tests/capabilities/test_channel_write.py
import pytest
from osprey.capabilities.channel_write import validate_pv_name

def test_validate_pv_name_accepts_valid():
    """Test PV name validation with valid input."""
    assert validate_pv_name("SR01:BPM:X") is True

def test_validate_pv_name_rejects_invalid():
    """Test PV name validation with invalid input."""
    with pytest.raises(ValueError, match="Invalid PV format"):
        validate_pv_name("invalid_name")
```

**Mocking**: Mock external dependencies (LLMs, APIs, file system)

```python
from unittest.mock import Mock, patch

@patch('osprey.services.llm.client.chat')
def test_capability_with_mocked_llm(mock_chat):
    """Test capability logic without real LLM calls."""
    mock_chat.return_value = {"content": "mocked response"}

    result = execute_capability(state)

    assert result.success is True
    assert "mocked response" in result.output
```

---

### 2. Integration Tests (Use Moderately)

**Purpose**: Test how components work together with mocked external services

**Characteristics**:
- ⚠️ Medium speed (seconds)
- ✅ Free (external services mocked)
- ✅ Tests component interactions
- ✅ Catches integration bugs

**When to use**:
- Service initialization
- Registry management
- State management
- Configuration loading
- Component coordination (with mocked LLMs)

**Example**:
```python
# tests/infrastructure/test_registry.py
import pytest

@pytest.mark.asyncio
async def test_registry_loads_capabilities(mock_registry):
    """Test registry successfully loads and registers capabilities."""
    from osprey.registry import get_registry

    registry = await get_registry()

    assert "channel_write" in registry.capabilities
    assert "channel_read" in registry.capabilities
```

---

### 3. End-to-End (E2E) Tests (Use Sparingly)

**Purpose**: Validate complete workflows with real LLMs

**Characteristics**:
- ❌ Slow (2-5 minutes for full suite)
- ❌ Expensive ($0.10-$0.25 per run)
- ⚠️ Non-deterministic (LLM variability)
- ✅ Tests real user experience
- ✅ Catches integration issues

**When to use** (only these scenarios):
1. **Tutorial validation** - Ensure tutorials work end-to-end
2. **Critical orchestration** - LLM-driven routing and planning
3. **Multi-capability workflows** - Real capability chains
4. **Response quality** - LLM judge evaluation required

**Example scenarios**:
- ✅ "Complete BPM timeseries tutorial workflow"
- ✅ "LLM correctly routes between capabilities"
- ✅ "Weather tutorial with mock API integration"
- ❌ "PV name validation works" (unit test instead!)
- ❌ "Configuration loads correctly" (integration test instead!)

**Example**:
```python
# tests/e2e/test_tutorials.py
import pytest

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_bpm_tutorial_workflow(e2e_project_factory):
    """Validate complete BPM analysis tutorial.

    This e2e test is justified because:
    - Tests LLM orchestration across multiple capabilities
    - Validates tutorial experience users will have
    - Requires real LLM decision-making (cannot be mocked)
    - Cost: ~$0.05, Time: ~60 seconds
    """
    project = await e2e_project_factory(
        name="test-bpm-tutorial",
        template="control_assistant"
    )

    await project.initialize()

    # Real LLM workflow
    result = await project.query(
        "Find BPM channels and plot their timeseries"
    )

    # Validate with LLM judge
    judge_result = await project.judge_response(
        query="Find BPM channels and plot timeseries",
        response=result,
        criteria=["channels found", "plot generated", "workflow completed"]
    )

    assert judge_result.passes
```

---

## 🔧 Writing Tests

### Test Structure

```python
"""tests/capabilities/test_my_capability.py"""
import pytest
from unittest.mock import Mock, patch

# ============================================================================
# UNIT TESTS
# ============================================================================

def test_my_function_basic_case():
    """Test basic functionality."""
    result = my_function("input")
    assert result == "expected"

def test_my_function_edge_case():
    """Test edge case handling."""
    result = my_function("")
    assert result is None

def test_my_function_error_handling():
    """Test error handling."""
    with pytest.raises(ValueError, match="Invalid input"):
        my_function(None)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_capability_initialization(mock_registry):
    """Test capability initializes correctly."""
    capability = MyCapability()
    await capability.initialize()
    assert capability.is_ready is True

# ============================================================================
# E2E TESTS (only if absolutely necessary)
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_critical_llm_workflow(e2e_project_factory):
    """Test workflow that REQUIRES real LLM.

    Justification: [Explain why this cannot be a unit test]
    Cost: $X.XX, Time: XX seconds
    """
    # Only write if you've exhausted unit/integration options
    pass
```

### Test Markers

```python
# Unit tests (no marker needed)
def test_unit():
    pass

# Integration tests
@pytest.mark.asyncio
async def test_integration():
    pass

# E2E tests (use ALL these markers)
@pytest.mark.e2e           # Marks as e2e test
@pytest.mark.slow          # Indicates slow test
@pytest.mark.requires_cborg  # or requires_anthropic, requires_openai
@pytest.mark.asyncio       # For async tests
async def test_e2e():
    pass
```

---

## 🧰 Test Fixtures and Utilities

Osprey provides shared fixtures in `tests/conftest.py`:

### Key Fixtures

```python
# Auto-reset (automatic - nothing to do!)
@pytest.fixture(autouse=True)
def reset_state_between_tests():
    """Resets registry/config before each test."""

# Create test states easily
from tests.conftest import create_test_state

def test_my_capability():
    state = create_test_state(
        user_message="What's the weather?",
        task_objective="Ask for location"
    )
    result = my_capability.execute(state)
    assert result["success"]

# Test with configuration
def test_with_config(test_config):
    os.environ['CONFIG_FILE'] = str(test_config)
    config = get_full_configuration()
    assert config is not None

# Mock code generator
def test_with_mock(mock_code_generator):
    mock_code_generator.set_code("results = {'value': 42}")
    code = await mock_code_generator.generate_code(request, [])
    assert "results" in code
```

### Available Fixtures

| Fixture | Purpose |
|---------|---------|
| `test_state` | Basic AgentState |
| `test_config` | Minimal test configuration |
| `mock_code_generator` | Mock code generator |
| `prompt_helpers` | Prompt testing utilities |
| `e2e_project_factory` | Create e2e test projects |
| `llm_judge` | LLM-based validation |

---

## 🔄 Parametrized Tests

Run the same test with different inputs to reduce duplication:

```python
import pytest

# Basic parametrization
@pytest.mark.parametrize("pv_name,expected", [
    ("SR01:BPM:X", True),
    ("SR01:BPM:Y", True),
    ("invalid", False),
    ("", False),
])
def test_validate_pv_name(pv_name, expected):
    """Test PV name validation with multiple inputs."""
    result = validate_pv_name(pv_name)
    assert result == expected

# With custom test IDs
@pytest.mark.parametrize(
    "value,limits,expected",
    [
        (5.0, (-10.0, 10.0), True),
        (15.0, (-10.0, 10.0), False),
    ],
    ids=["within_limits", "exceeds_max"]
)
def test_value_within_limits(value, limits, expected):
    result = check_limits(value, limits)
    assert result == expected

# Multiple parameters (creates 6 tests: 3 × 2)
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
@pytest.mark.parametrize("mode", ["read_only", "write_access"])
def test_code_generation(provider, mode):
    generator = create_generator(provider=provider, mode=mode)
    assert generator.generate() is not None
```

**When to use**: Testing same logic with different inputs, boundary testing, validation functions.

---

## 🎭 Mocking Strategies

**Mock external dependencies**: API calls, file I/O, database, network, time-dependent code.
**Don't mock**: Code you're testing, simple data structures, pure functions.

```python
from unittest.mock import Mock, patch, AsyncMock

# Basic mocking
def test_with_mock():
    mock_client = Mock()
    mock_client.get_data.return_value = {"value": 42}
    result = process_data(mock_client)
    assert result == {"value": 42}
    mock_client.get_data.assert_called_once()

# Patch decorator
@patch('osprey.services.llm.client.chat')
def test_with_patch(mock_chat):
    mock_chat.return_value = {"content": "mocked"}
    result = my_function()
    assert "mocked" in result

# Async mocking
@pytest.mark.asyncio
async def test_async():
    mock_client = AsyncMock()
    mock_client.fetch.return_value = {"data": [1, 2, 3]}
    result = await process_async(mock_client)
    assert len(result["data"]) == 3

# Side effects (sequence or exception)
def test_side_effects():
    mock_api = Mock()
    mock_api.get.side_effect = [{"status": "pending"}, {"status": "done"}]
    assert mock_api.get()["status"] == "pending"
    assert mock_api.get()["status"] == "done"

# Monkeypatch (recommended - auto cleanup)
def test_monkeypatch(monkeypatch):
    mock_get_config = Mock(return_value={"key": "value"})
    monkeypatch.setattr("osprey.utils.config.get_full_configuration", mock_get_config)
    monkeypatch.setenv("API_KEY", "test_key")
    result = my_function()
    assert result is not None
```

**Best practices**: Mock at boundaries, use real objects when possible, make mocks realistic, verify interactions.

---

## ⚡ Async Testing

**Always use `@pytest.mark.asyncio` for async tests.**

### Common Async Pitfalls

```python
# ❌ WRONG - Missing await
@pytest.mark.asyncio
async def test_wrong():
    result = my_async_function()  # Returns coroutine, doesn't execute!
    assert result  # Passes but didn't test anything!

# ✅ CORRECT
@pytest.mark.asyncio
async def test_correct():
    result = await my_async_function()
    assert result

# ❌ WRONG - Mock() doesn't work with async
mock_client = Mock()
result = await mock_client.fetch()  # Fails!

# ✅ CORRECT - Use AsyncMock
mock_client = AsyncMock()
mock_client.fetch.return_value = {"data": []}
result = await mock_client.fetch()  # Works!

# ❌ WRONG - Can't await in sync function
def test_wrong():
    result = await my_async_function()  # SyntaxError!

# ✅ CORRECT - Make test async
@pytest.mark.asyncio
async def test_correct():
    result = await my_async_function()
```

### Testing Concurrent Operations

```python
import asyncio

@pytest.mark.asyncio
async def test_concurrent():
    results = await asyncio.gather(task_one(), task_two(), task_three())
    assert len(results) == 3
```

**Note**: Osprey uses `asyncio_mode = auto` in `pytest.ini`, so async fixtures work automatically.

---

## 📋 Testing Checklist

### Before Writing Tests

- [ ] Can this be tested with a unit test? (default choice)
- [ ] Does this require component interaction? (integration test)
- [ ] Does this REQUIRE real LLM behavior? (consider e2e test)
- [ ] Have I exhausted mocking options before writing e2e test?

### Writing Unit Tests

- [ ] Test happy path
- [ ] Test edge cases (empty input, None, extremes)
- [ ] Test error conditions
- [ ] Mock external dependencies
- [ ] Keep tests fast (<100ms each)
- [ ] Make tests deterministic

### Writing E2E Tests

- [ ] **Justified**: Written justification in docstring explaining why unit tests insufficient
- [ ] **Cost-aware**: Documented estimated cost and time
- [ ] **Minimal**: Smallest possible test scope
- [ ] **Critical path**: Tests core user workflow
- [ ] **Markers**: All appropriate pytest markers applied
- [ ] **LLM judge**: Only if response quality needs verification

### Running Tests

- [ ] Unit tests pass: `pytest tests/ --ignore=tests/e2e -v`
- [ ] No warnings or deprecations
- [ ] Coverage adequate for new code (>80%)
- [ ] E2E tests pass (before release): `pytest tests/e2e/ -v`

---

## 🎛️ pytest Configuration

### Basic Usage

```bash
# Run unit tests (fast)
pytest tests/ --ignore=tests/e2e -v

# Run with coverage
pytest tests/ --ignore=tests/e2e --cov=src/osprey --cov-report=term-missing

# Run specific marker
pytest tests/ -m "not slow" -v

# Run with verbose output
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x
```

### E2E Test Options

```bash
# Run all e2e tests
pytest tests/e2e/ -v

# Run with e2e verbose progress
pytest tests/e2e/ -v -s --e2e-verbose

# Run with judge verbose reasoning
pytest tests/e2e/ --judge-verbose

# Run specific LLM judge
pytest tests/e2e/ --judge-provider=anthropic --judge-model=claude-sonnet-4

# Run only smoke tests (faster e2e subset)
pytest tests/e2e/ -m e2e_smoke -v
```

### CI/CD Integration

```yaml
# .github/workflows/tests.yml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run unit tests
        run: pytest tests/ --ignore=tests/e2e -v --cov=src/osprey

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run e2e tests
        run: pytest tests/e2e/ -v
        env:
          CBORG_API_KEY: ${{ secrets.CBORG_API_KEY }}
```

---

## 📁 Test Organization

**File naming**: `test_*.py` for files, `test_*` for functions, `Test*` for classes.

```python
# ✅ GOOD names
test_channel_write.py
def test_validate_pv_name_accepts_valid_input():
class TestChannelWriteValidation:

# ❌ BAD names
write_tests.py
def test_validation():
def test_1():
```

**Group related tests in classes**:

```python
class TestChannelWriteValidation:
    """Tests for channel write input validation."""
    def test_valid_pv_name(self):
        pass
    def test_invalid_pv_name(self):
        pass

class TestChannelWriteExecution:
    """Tests for channel write execution logic."""
    @pytest.mark.asyncio
    async def test_successful_write(self):
        pass
```

**Always add docstrings**. For e2e tests, include cost and justification:

```python
@pytest.mark.e2e
async def test_bpm_tutorial(e2e_project_factory):
    """Validate BPM tutorial workflow.

    Justification: Tests real LLM orchestration that cannot be mocked.
    Cost: ~$0.05, Time: ~60 seconds
    """
```

---

## 🐛 Troubleshooting

**"Registry contains no nodes"**
- Running unit and e2e tests together
- Fix: Always run separately: `pytest tests/ --ignore=tests/e2e`

**"Python executor service not available"**
- Registry state contamination
- Fix: Run e2e tests separately: `pytest tests/e2e/`

**Tests pass individually but fail in batch**
- State leakage between tests
- Fix: Check for global state that's not being reset
- For E2E: This is normal, run as separate suite

**Async tests hanging**
- Missing `await` or deadlock
- Fix: Add timeout `pytest --timeout=10`, run with `-s` to see output

**"Fixture not found"**
- Check `conftest.py` in test directory
- Run `pytest --fixtures` to see available fixtures

### Quick Debug Commands

```bash
# Drop into debugger on failure
pytest tests/test_my_test.py --pdb

# Show local variables on failure
pytest tests/test_my_test.py -l

# Show print statements
pytest tests/test_my_test.py -s

# Run specific test
pytest tests/test_file.py::test_function -v

# Show slowest tests
pytest tests/ --durations=10

# Coverage report
pytest tests/ --ignore=tests/e2e --cov=src/osprey --cov-report=term-missing
```

---

## 📊 Test Coverage Guidelines

### Coverage Expectations

- **New features**: >80% coverage
- **Bug fixes**: Include regression test
- **Refactoring**: Maintain existing coverage
- **Critical paths**: 100% coverage

### What to Cover

**High priority**:
- Public API functions
- Error handling paths
- Business logic
- Data transformations
- Edge cases

**Medium priority**:
- Internal utilities
- Configuration parsing
- Validation functions

**Low priority** (can skip):
- Simple getters/setters
- Trivial wrappers
- Obvious pass-through functions

---

## 🎓 Best Practices

### General Testing Principles

1. **Fast by default** - Unit tests should run in milliseconds
2. **Isolation** - Tests should not depend on each other
3. **Determinism** - Same input = same output
4. **Readability** - Test should document behavior
5. **Maintainability** - Tests should be easy to update

### E2E Test Principles

1. **Justify every e2e test** - Document why unit test insufficient
2. **Minimize scope** - Test smallest possible workflow
3. **Cost awareness** - Track and minimize API costs
4. **Real user workflows** - Test what users actually do
5. **LLM judge sparingly** - Only for response quality validation

### Anti-Patterns to Avoid

❌ **Testing implementation details** - Test behavior, not internals
❌ **Brittle tests** - Tests shouldn't break on refactoring
❌ **Slow unit tests** - If slow, it's probably integration
❌ **E2E tests for logic** - Logic belongs in unit tests
❌ **No assertions** - Every test must assert something
❌ **Unclear test names** - Name should describe what's tested

---

## 📚 Example: Complete Test File

```python
"""tests/capabilities/test_channel_write.py"""
import pytest
from unittest.mock import patch, AsyncMock
from osprey.capabilities.channel_write import validate_boundaries, execute_write
from tests.conftest import create_test_state

# ============================================================================
# UNIT TESTS - Fast, free, deterministic
# ============================================================================

class TestBoundaryValidation:
    """Test write value boundary validation."""

    @pytest.mark.parametrize("value,limits,expected", [
        (5.0, (-10.0, 10.0), True),
        (15.0, (-10.0, 10.0), False),
        (-15.0, (-10.0, 10.0), False),
    ], ids=["within_limits", "exceeds_max", "exceeds_min"])
    def test_validate_boundaries(self, value, limits, expected):
        """Test boundary validation with various values."""
        pv_limits = {"SR01:BPM:X": limits}
        result = validate_boundaries("SR01:BPM:X", value, pv_limits)
        assert result == expected

    def test_raises_error_when_no_limits_configured(self):
        """Test error raised when PV has no configured limits."""
        with pytest.raises(ValueError, match="No limits configured"):
            validate_boundaries("SR01:BPM:X", 5.0, {})

# ============================================================================
# INTEGRATION TESTS - Mocked external services
# ============================================================================

class TestChannelWriteExecution:
    """Test channel write execution with mocked LLM."""

    @pytest.mark.asyncio
    @patch('osprey.capabilities.channel_write.llm_client')
    async def test_execute_write_with_approval(self, mock_llm):
        """Test write execution when LLM approves."""
        mock_llm.chat.return_value = {"decision": "approved"}
        state = create_test_state(
            user_message="Set BPM X to 5.0",
            capability="channel_write",
        )
        result = await execute_write(state)
        assert result["success"] is True
        mock_llm.chat.assert_called_once()

# ============================================================================
# E2E TESTS - Only if real LLM required
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_write_approval_e2e(e2e_project_factory, llm_judge):
    """Test LLM-driven approval decision with real LLM.

    Justification: Approval logic requires real LLM reasoning about
    safety that cannot be effectively mocked.
    Cost: $0.02, Time: 10 seconds
    """
    project = await e2e_project_factory(
        name="test-write-approval",
        template="control_assistant"
    )
    await project.initialize()

    result = await project.query("Set SR01:BPM:X to 5.0")
    judge_result = await llm_judge.evaluate(
        query="Set SR01:BPM:X to 5.0",
        response=result,
        criteria=["write approved", "safety check passed"]
    )
    assert judge_result.passes
```


## 🔗 Additional Resources

- [tests/e2e/README.md](../../tests/e2e/README.md) - E2E test details
- [pytest.ini](../../pytest.ini) - pytest configuration
- [conftest.py](../../tests/conftest.py) - Shared fixtures
- [ai-code-review.md](ai-code-review.md) - Review AI code before testing
- [pre-merge-cleanup.md](pre-merge-cleanup.md) - Pre-commit checklist

---

## 📝 Quick Reference

```bash
# Unit tests (run constantly)
pytest tests/ --ignore=tests/e2e -v

# E2E tests (before releases)
pytest tests/e2e/ -v

# Specific test
pytest tests/test_file.py::test_function -v

# With coverage
pytest tests/ --ignore=tests/e2e --cov=src/osprey

# Debug
pytest tests/test_file.py --pdb -s
```

**Key markers**: `@pytest.mark.asyncio`, `@pytest.mark.parametrize()`, `@pytest.mark.e2e`, `@pytest.mark.requires_cborg`

**Key fixtures**: `test_state`, `test_config`, `mock_code_generator`, `e2e_project_factory`, `llm_judge`

---

**Remember**: Unit tests first (fast, free). E2E tests only for real LLM workflows (slow, expensive). When in doubt, write a unit test.
