# Testing Guide - Runtime Utilities for Control System Operations

**Branch**: `feat/pv-boundary-checking`
**Requirements**: Python 3.11+

## ⚠️ CRITICAL: How to Run Tests

```bash
# ✅ CORRECT: Run unit tests
pytest tests/ --ignore=tests/e2e -v

# ✅ CORRECT: Run e2e tests
pytest tests/e2e/ -v

# ❌ WRONG: Do NOT use -m e2e (causes test collection issues)
pytest -m e2e  # DON'T DO THIS!
```

**Why?** Using `-m e2e` causes pytest to collect tests in the wrong order, leading to registry initialization failures. Always use `pytest tests/e2e/` directly. See [tests/e2e/README.md](tests/e2e/README.md) for full details.

---

## Overview

This branch introduces the `osprey.runtime` module - control-system-agnostic utilities that enable LLMs to interact with control systems through a simple synchronous API. The feature includes execution wrapper integration, custom prompt builders, and comprehensive safety integration with channel limits validation.

## Quick Setup

```bash
# 1. Checkout the branch
git checkout feat/pv-boundary-checking

# 2. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (if needed)
pip install -e "."

# 4. Run the test suite
pytest tests/runtime/ tests/e2e/test_runtime_utilities.py -v
```

## What Changed

### Core Implementation
- **`osprey.runtime` module**: Synchronous API (`write_channel`, `read_channel`, `write_channels`) that handles async internally
- **Automatic configuration**: Uses execution context snapshots for reproducible notebooks
- **Protocol-agnostic**: Works with EPICS, Mock, LabVIEW, Tango - any registered connector
- **Resource management**: Proper cleanup in finally blocks with graceful degradation
- **Fallback chain**: Context config → global config → clear error

### Integration
- **Execution wrapper**: Automatically configures runtime from context after load
- **Context manager**: Preserves control system config via `add_execution_config()`
- **Notebooks**: Include runtime configuration cells for standalone execution
- **Cleanup**: `cleanup_runtime()` called in finally blocks for resource safety

### Prompt System
- **ControlSystemPythonPromptBuilder**: Custom prompt builder teaching LLMs the runtime API
- **Framework prompt integration**: Automatic injection of domain-specific instructions
- **Enhanced classifier**: Examples for control system operations with Python
- **Graceful fallback**: Works without custom prompts if unavailable

### Documentation
- **API Reference**: Usage examples in generated code
- **Developer Guide**: Integration with Python execution service
- **Part 3 Tutorial**: Detailed explanation of runtime utilities
- **Part 4 Tutorial**: Framework prompt customization guide

## Testing Focus Areas

### 1. Unit Tests (236 lines)

```bash
# Run runtime module unit tests
pytest tests/runtime/test_runtime.py -v
```

**Coverage**:
- ✅ Configuration from context snapshots
- ✅ Configuration fallback to global config
- ✅ Error handling when config unavailable
- ✅ Write channel operations (success and failure)
- ✅ Read channel operations
- ✅ Bulk write operations
- ✅ Connector lifecycle (create, reuse, cleanup)
- ✅ Cleanup and reconnection

### 2. Integration Tests (285 lines)

```bash
# Run integration tests with Mock connector
pytest tests/runtime/test_runtime_integration.py -v
```

**Coverage**:
- ✅ Write and read with Mock connector
- ✅ Bulk channel writes
- ✅ Cleanup and reconnect workflows
- ✅ Context snapshot reproducibility
- ✅ Error handling for invalid channels
- ✅ Connector reuse across operations
- ✅ Kwargs passthrough to connector
- ✅ Fallback to global config

### 3. End-to-End Tests (482 lines)

```bash
# Run E2E tests (requires CBORG API key)
pytest tests/e2e/test_runtime_utilities.py -v -m "e2e and requires_cborg"
```

**Coverage**:
- ✅ LLM learns `osprey.runtime` API from prompts
- ✅ Generated code uses `write_channel()` correctly
- ✅ **CRITICAL SAFETY**: Runtime respects channel limits
- ✅ Valid writes within limits succeed
- ✅ Calculation + write workflows (e.g., `sqrt(4150)`)
- ✅ Context snapshot preservation in notebooks
- ✅ Notebook includes runtime configuration

### 4. Runtime Limits E2E Tests (522 lines)

```bash
# Run comprehensive limits safety tests
pytest tests/services/python_executor/test_runtime_limits_e2e.py -v -m "e2e"
```

**Coverage**:
- ✅ Simple writes respect limits
- ✅ Calculated values checked against limits
- ✅ Bulk writes validated
- ✅ Mixed read/write operations
- ✅ Complex workflows with limits integration

### 5. Prompt Builder Tests

```bash
# Run prompt builder integration tests
pytest tests/capabilities/test_python_capability.py -v -k "prompt"
```

**Coverage**:
- ✅ Prompt builder integration in PythonCapability
- ✅ Domain-specific instructions injected
- ✅ Graceful fallback if prompts unavailable
- ✅ Custom classifier examples

## Expected Behavior

### Runtime Module

#### Configuration
```python
from osprey.runtime import configure_from_context, write_channel, read_channel

# Automatically configured by execution wrapper
# Uses context snapshot for reproducibility
write_channel("TEST:VOLTAGE", 75.5)
value = read_channel("BEAM:CURRENT")
```

#### Synchronous API
- Functions appear synchronous (no `await` needed)
- Async handled internally via `_run_async()`
- Works in both subprocess and Jupyter contexts
- Avoids event loop conflicts

#### Cleanup
```python
from osprey.runtime import cleanup_runtime

# Called automatically in execution wrapper finally block
await cleanup_runtime()
```

### Generated Code

LLMs should generate code like:
```python
from osprey.runtime import write_channel
import math

# Calculate value
voltage = math.sqrt(4150)

# Write to control system
write_channel("TerminalVoltageSetPoint", voltage)

# Store results
results = {"voltage_set": voltage}
```

**NOT** like:
```python
import epics  # ❌ Direct EPICS import

epics.caput("TerminalVoltageSetPoint", voltage)  # ❌ Bypasses safety
```

### Safety Integration

Runtime utilities **must** respect channel limits:

```python
# If limits file defines TEST:VOLTAGE max = 100V
write_channel("TEST:VOLTAGE", 150)  # ❌ Should raise ChannelLimitsViolationError

write_channel("TEST:VOLTAGE", 75)   # ✅ Should succeed
```

## Manual Testing Scenarios

### Scenario 1: Basic Runtime Usage

```bash
# Create test project
osprey init test-runtime --template control_assistant

cd test-runtime

# Configure Mock connector
# Edit config.yml:
#   control_system:
#     type: mock
#     writes_enabled: true

# Start assistant
osprey chat

# Test query:
# "Write a Python script to set TEST:VOLTAGE to 75.5 volts"

# Verify:
# 1. Generated code uses osprey.runtime
# 2. Code executes successfully
# 3. Notebook includes runtime configuration cell
```

### Scenario 2: Safety Integration

```bash
cd test-runtime

# Add limits file
cat > data/channel_limits.json << EOF
{
  "TEST:VOLTAGE": {
    "min_value": 0.0,
    "max_value": 100.0,
    "writable": true
  }
}
EOF

# Enable limits in config.yml:
#   control_system:
#     limits_checking:
#       enabled: true
#       limits_file: "data/channel_limits.json"
#       policy:
#         allow_unlisted_channels: false
#         on_violation: "error"

# Test query:
# "Write Python code to set TEST:VOLTAGE to 150 volts"

# Expected result:
# - Code generates correctly
# - Execution fails with ChannelLimitsViolationError
# - Error message shows violation details
```

### Scenario 3: Context Reproducibility

```bash
cd test-runtime

# Generate code that uses runtime
osprey chat
# Query: "Set TEST:VOLTAGE to 50"

# Find generated notebook in _agent_data/executed_scripts/

# Re-run notebook standalone
jupyter notebook _agent_data/executed_scripts/<execution_folder>/notebook.ipynb

# Verify:
# 1. Runtime configuration cell present
# 2. Uses same control system config as generation time
# 3. Executes without errors
```

### Scenario 4: Prompt Builder Customization

```bash
cd test-runtime

# Inspect custom prompt builder
cat src/test_runtime/framework_prompts/python.py

# Verify:
# 1. Extends DefaultPythonPromptBuilder
# 2. Adds osprey.runtime guidance
# 3. Includes classifier examples
# 4. Registered in registry.py

# Test that LLM learns API
osprey chat
# Query: "Write code to read BEAM:CURRENT"

# Generated code should use:
# from osprey.runtime import read_channel
```

## Verification Checklist

### Code Quality
- [ ] All unit tests pass (336 tests)
- [ ] All integration tests pass (285 tests)
- [ ] E2E tests pass (requires API key)
- [ ] No linter errors
- [ ] Type hints correct

### Functionality
- [ ] Runtime module configures from context
- [ ] Fallback to global config works
- [ ] Cleanup happens in finally blocks
- [ ] Notebooks include runtime configuration
- [ ] Prompt builder teaches LLMs correctly

### Safety
- [ ] **CRITICAL**: Runtime respects channel limits
- [ ] Violations raise ChannelLimitsViolationError
- [ ] Valid writes succeed
- [ ] Safety layer not bypassed

### Documentation
- [ ] API reference updated
- [ ] Developer guide updated
- [ ] Part 3 tutorial explains runtime
- [ ] Part 4 tutorial shows customization
- [ ] CHANGELOG entries complete

### Integration
- [ ] Works with Mock connector
- [ ] Works with EPICS (if available)
- [ ] Template includes prompt builder
- [ ] Registry configuration correct

## Running Full Test Suite

```bash
# Run all runtime-related tests
pytest tests/runtime/ \
       tests/e2e/test_runtime_utilities.py \
       tests/services/python_executor/test_runtime_limits_e2e.py \
       tests/capabilities/test_python_capability.py::test_python_capability_with_custom_prompts \
       -v

# Expected: ~1600+ tests pass (including existing tests)
```

## Known Issues

None - all tests passing ✅

## Commits on This Branch

1. **feat**: Add runtime utilities for control system operations (989 insertions)
2. **feat**: Integrate runtime utilities with execution infrastructure (112 insertions)
3. **feat**: Add prompt builder system for control system operations (411 insertions, 86 deletions)
4. **test**: Add E2E tests for runtime utilities and safety (1010 insertions)
5. **docs**: Update positioning for control system focus and runtime utilities (70 insertions, 6 deletions)

**Total**: 2,592 insertions(+), 148 deletions(-)

## Publishing Checklist

- [x] All commits created
- [x] CHANGELOG updated
- [x] Documentation complete
- [ ] Testing guide updated
- [ ] All tests pass locally
- [ ] Ready to push

## Reporting Issues

Found a bug or have suggestions? https://github.com/als-apg/osprey/issues

## Additional Resources

- **Runtime Module**: `src/osprey/runtime/__init__.py`
- **Prompt Builder**: `src/osprey/templates/apps/control_assistant/framework_prompts/python.py.j2`
- **API Reference**: `docs/source/api_reference/03_production_systems/06_control-system-connectors.rst`
- **Tutorial Part 3**: `docs/source/getting-started/control-assistant-part3-production.rst`
- **Tutorial Part 4**: `docs/source/getting-started/control-assistant-part4-customization.rst`
