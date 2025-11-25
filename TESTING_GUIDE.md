# Testing Guide

**Branch**: `fix/context-manager-extract-from-step`

**Requirements**: Python 3.11+

## Quick Setup

```bash
# 1. Clone and checkout this branch
git clone https://github.com/als-apg/osprey.git
cd osprey
git checkout fix/context-manager-extract-from-step

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Osprey in development mode
pip install -e .

# 4. Run the test suite
pytest

# 5. (Optional) Build documentation locally
pip install -e ".[docs]"
cd docs
python launch_docs.py
# Visit: http://localhost:8082
```

## What's New in This Branch

This branch includes several major improvements and one new provider:

1. **Context Manager Data Loss Fix** - Fixed critical bug where multiple contexts of the same type were silently lost
2. **Instance Method Pattern** - New recommended pattern for capabilities with helper methods
3. **Infrastructure Node Migration** - All 7 infrastructure nodes migrated to instance method pattern
4. **Argo Provider** - New AI provider for Argonne National Laboratory's proxy service

## Testing Priority Areas

### 1. Context Manager (Critical Fix)

**Test multiple same-type contexts:**

Create a capability that requires multiple contexts of the same type:

```python
class TestCapability(BaseCapability):
    name = "test_multi_context"
    requires = [('DATA', 'multiple')]  # Cardinality validation

    def execute(self, state, step):
        # Should return list of DATA contexts
        data_contexts = self.get_required_contexts()
        assert isinstance(data_contexts, list)
        assert len(data_contexts) > 1
```

**What to verify:**
- Multiple contexts of same type are preserved (previously lost)
- Single context returns object, multiple return list
- Cardinality constraints work: `('DATA', 'single')` or `('DATA', 'multiple')`

### 2. Instance Method Pattern for Capabilities

**Test the new helper methods:**

```python
class ExampleCapability(BaseCapability):
    name = "example"
    requires = [('DATA', 'single'), 'TIME']

    def execute(self, state, step):
        # New helper methods (easier than static pattern)
        data, time = self.get_required_contexts()  # Tuple unpacking
        task = self.get_task_objective()
        params = self.get_parameters()

        # Store outputs
        self.store_output_context(result_context)
```

**What to verify:**
- Helper methods work correctly
- Tuple unpacking matches order of `requires` field
- Backward compatibility with static method pattern
- `_state` and `_step` are properly injected

### 3. Infrastructure Nodes

**All 7 infrastructure nodes migrated:**
- Router
- Task Extraction
- Classification
- Clarify
- Respond
- Error
- Orchestration

**What to verify:**
- Nodes use `self._state` instead of `state` parameter
- Clarify and Respond have `self._step` injected
- Classification bypass mode still works
- Error node uses `StateManager.get_current_step_index()` (no `_step`)

### 4. Argo Provider (New)

**Test the new ANL provider:**

```bash
# Set up Argo (requires ANL affiliation)
export ARGO_API_KEY=your-key
```

Create a project and configure Argo:

```yaml
# In config.yml
api:
  providers:
    argo:
      api_key: ${ARGO_API_KEY}
      base_url: https://argo-bridge.cels.anl.gov

models:
  orchestrator:
    provider: argo
    model_id: claudesonnet45  # or claudehaiku45, gpt5mini, gemini25flash, etc.
```

**What to verify:**
- Argo provider loads correctly
- 8 models available: claudehaiku45, claudesonnet45, claudesonnet37, claudeopus41, gemini25flash, gemini25pro, gpt5, gpt5mini
- Authentication uses `$USER` environment variable
- Structured outputs work correctly
- Template generation includes Argo in environment variables

### 5. Interactive Menu

**Test project creation workflow:**

```bash
osprey  # Launch interactive menu
```

**What to verify:**
- Registry properly resets between projects (no contamination)
- Environment variable auto-detection includes STANFORD_API_KEY and ARGO_API_KEY
- All templates work (minimal, hello_world_weather, control_assistant)
- Version number displays in banner

## Running the Test Suite

```bash
# Run all tests (should see 300+ tests pass)
pytest

# Run specific test suites for new features
pytest tests/capabilities/          # Instance method pattern tests (12 tests)
pytest tests/infrastructure/        # Infrastructure node tests (15 tests)
pytest tests/context_manager/       # Context extraction tests (26 tests)

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=osprey --cov-report=html
```

**Expected test counts:**
- Total: ~300 tests
- New in this branch: 53 tests (capabilities + infrastructure + context manager)

## Documentation

Review updated documentation locally:
```bash
cd docs
python launch_docs.py
# Visit: http://localhost:8082
```

**Key sections to review:**
- **Developer Guides** → Migration Guide (Instance Methods)
- **Getting Started** → Hello World Tutorial (updated with Argo)
- **API Reference** → Configuration System (Argo provider examples)
- **Developer Guides** → Convention Over Configuration (Argo examples)

## Reporting Issues

Found a bug or have suggestions? https://github.com/als-apg/osprey/issues

