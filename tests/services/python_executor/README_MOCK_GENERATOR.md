# Mock Code Generator for Testing

This directory contains a comprehensive testing solution for the Python Executor Service using the `MockCodeGenerator` - a test-friendly code generator that requires no external dependencies.

## Overview

The `MockCodeGenerator` implements the `CodeGenerator` protocol and provides deterministic, fast, and reliable testing without requiring:
- LLM API access or credentials
- External service dependencies
- Container infrastructure (can use local execution)
- Network connectivity

## Quick Start

### Basic Usage

```python
from osprey.services.python_executor.generation import MockCodeGenerator
from osprey.services.python_executor import PythonExecutionRequest

# Create mock generator with predefined behavior
generator = MockCodeGenerator(behavior="success")

# Create a request
request = PythonExecutionRequest(
    user_query="Calculate test value",
    task_objective="Test computation",
    execution_folder_name="test"
)

# Generate code
code = await generator.generate_code(request, [])
```

### Predefined Behaviors

The mock generator supports several predefined behaviors for common test scenarios:

```python
# Successful execution
generator = MockCodeGenerator(behavior="success")

# Syntax error (for testing analyzer)
generator = MockCodeGenerator(behavior="syntax_error")

# Runtime error (for testing error handling)
generator = MockCodeGenerator(behavior="runtime_error")

# Missing results dictionary
generator = MockCodeGenerator(behavior="missing_results")

# Control system write operations (for testing approval)
generator = MockCodeGenerator(behavior="channel_write")

# Control system read operations (read-only, safe)
generator = MockCodeGenerator(behavior="channel_read")

# Security-sensitive operations
generator = MockCodeGenerator(behavior="security_risk")

# Error-aware generation (adapts to feedback)
generator = MockCodeGenerator(behavior="error_aware")
```

### Custom Code

You can configure the generator to return specific code:

```python
generator = MockCodeGenerator()

# Static code (same on every call)
generator.set_code("results = {'value': 42}")

# Code sequence (different on each call)
generator.set_code_sequence([
    "results = 1 / 0",  # Fails first time
    "results = {'value': 42}"  # Succeeds on retry
])
```

### Error-Aware Generation

The error-aware behavior simulates how a real LLM adapts to feedback:

```python
generator = MockCodeGenerator(behavior="error_aware")

# First attempt
code1 = await generator.generate_code(request, [])

# Second attempt with error feedback
code2 = await generator.generate_code(
    request,
    ["NameError: name 'numpy' is not defined"]
)
# code2 will have proper imports added
```

## Testing the Full Service

### Integration Test Example

```python
from unittest.mock import patch
from osprey.services.python_executor import PythonExecutorService

# Create mock generator
mock_gen = MockCodeGenerator(behavior="success")

# Patch the factory to use our mock
with patch('osprey.services.python_executor.generation.create_code_generator',
           return_value=mock_gen):
    service = PythonExecutorService()

    result = await service.ainvoke(request, config={
        "thread_id": "test",
        "configurable": {
            "execution": {"execution_method": "local"}
        }
    })

    # Verify results
    assert result.execution_result is not None
    assert mock_gen.call_count >= 1
```

## Test Organization

### Test Files

- `test_mock_generator.py` - Tests for the mock generator itself
  - Protocol compliance
  - Static code generation
  - Code sequences
  - Predefined behaviors
  - Error-aware generation
  - Call tracking

- `test_service_integration.py` - Full service integration tests
  - Complete workflow testing
  - Error handling and retry logic
  - Analysis and security checks
  - State management
  - Execution methods (local vs container)

- `conftest.py` - Shared fixtures for all tests
  - Common mock generators
  - Sample requests
  - Temporary directories
  - Code snippets

### Running Tests

```bash
# Run all mock generator tests
pytest tests/services/python_executor/test_mock_generator.py -v

# Run integration tests
pytest tests/services/python_executor/test_service_integration.py -v

# Run all python executor tests
pytest tests/services/python_executor/ -v

# Run only fast tests (skip slow integration tests)
pytest tests/services/python_executor/ -v -m "not slow"

# Run only integration tests
pytest tests/services/python_executor/ -v -m "integration"
```

## Call Tracking

The mock generator tracks all calls for test assertions:

```python
generator = MockCodeGenerator()
generator.set_code("results = {}")

code = await generator.generate_code(request, ["error1", "error2"])

# Verify calls
assert generator.call_count == 1
assert generator.last_request == request
assert generator.last_error_chain == ["error1", "error2"]

# Reset for next test
generator.reset()
assert generator.call_count == 0
```

## Advanced Usage

### Testing Retry Logic

```python
# Simulate: fail -> fail -> succeed
generator = MockCodeGenerator()
generator.set_code_sequence([
    "def broken(",  # Syntax error
    "results = 1 / 0",  # Runtime error
    "results = {'value': 42}"  # Success
])
```

### Testing Approval Workflows

```python
# Code that triggers approval
generator = MockCodeGenerator(behavior="channel_write")

# This will trigger approval interrupt
result = await service.ainvoke(request, config)
# Test approval handling...
```

### Testing Analysis

```python
# Test security detection
generator = MockCodeGenerator(behavior="security_risk")
# Generated code will have subprocess/os.system calls
# Static analyzer should flag these

# Test control system write detection
generator = MockCodeGenerator(behavior="channel_write")
# Analyzer should detect write_channel calls
```

## Benefits

### Compared to Real LLM Testing

| Aspect | Real LLM | Mock Generator |
|--------|----------|----------------|
| Speed | 1-5 seconds | < 1ms |
| Cost | API credits | Free |
| Reliability | Variable output | Deterministic |
| Network | Required | Not required |
| CI/CD | Complex setup | Works anywhere |
| Debugging | Hard to reproduce | Easy to reproduce |

### Compared to Manual Mocking

The mock generator is better than using generic mocks because:
- ✅ **Protocol Compliant** - Implements `CodeGenerator` correctly
- ✅ **Realistic Behaviors** - Simulates actual code generation patterns
- ✅ **Comprehensive** - Covers all common test scenarios
- ✅ **Documented** - Clear API and examples
- ✅ **Maintainable** - Single source of truth for test code
- ✅ **Reusable** - Same generator works across all tests

## Examples

See the test files for complete examples:
- `test_mock_generator.py` - 20+ examples of mock generator usage
- `test_service_integration.py` - 15+ examples of full service testing

## Contributing

When adding new test scenarios:
1. Add a new behavior to `MockCodeGenerator._apply_behavior()` if it's common
2. Add corresponding tests to `test_mock_generator.py`
3. Add integration tests to `test_service_integration.py` if relevant
4. Update this README with new behavior documentation

