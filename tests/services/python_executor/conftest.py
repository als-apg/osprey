"""Python executor-specific test fixtures and utilities.

This module provides fixtures and utilities specific to testing the
Python executor service. General-purpose fixtures are in tests/conftest.py.

Note:
    - Use mock_code_generator from tests/conftest.py for general mock generator needs
    - Use test_config from tests/conftest.py for general config needs
    - This file contains only python-executor-specific test helpers
"""

import pytest

from osprey.services.python_executor import PythonExecutionRequest
from osprey.services.python_executor.models import ExecutionError


@pytest.fixture
def python_execution_request():
    """Provide a standard PythonExecutionRequest for testing.

    Returns:
        PythonExecutionRequest with standard test values

    Examples:
        Basic usage::

            def test_request(python_execution_request):
                assert python_execution_request.user_query == "Test query"

        With modifications::

            def test_custom(python_execution_request):
                python_execution_request.user_query = "Custom query"
                # Use modified request
    """
    return PythonExecutionRequest(
        user_query="Test query",
        task_objective="Test objective",
        execution_folder_name="test_execution"
    )


@pytest.fixture
def sample_python_codes():
    """Provide a collection of sample Python code snippets for testing.

    Returns:
        Dictionary of code snippets keyed by scenario name

    Examples:
        Using code samples::

            def test_syntax_error(sample_python_codes):
                code = sample_python_codes['syntax_error']
                with pytest.raises(SyntaxError):
                    compile(code, '<string>', 'exec')
    """
    return {
        'success': """
import json
from datetime import datetime

results = {
    'value': 42,
    'status': 'success',
    'timestamp': datetime.now().isoformat()
}
""".strip(),

        'syntax_error': """
def broken_function(
    # Missing closing parenthesis
results = {}
""".strip(),

        'runtime_error': """
import json

value = 100 / 0  # This will raise ZeroDivisionError

results = {'value': value}
""".strip(),

        'channel_write': """
from osprey.runtime import read_channel, write_channel

current = read_channel('TEST:PV')
write_channel('TEST:PV', current * 1.1)

results = {'operation': 'write', 'channel': 'TEST:PV'}
""".strip(),

        'channel_read': """
from osprey.runtime import read_channel

value = read_channel('TEST:PV')

results = {'operation': 'read', 'value': value}
""".strip(),
    }


@pytest.fixture
def sample_execution_error():
    """Create a sample ExecutionError for testing.

    Returns:
        ExecutionError instance with default test values
    """
    return ExecutionError(
        error_type="execution",
        error_message="Test error message",
        attempt_number=1,
        stage="execution"
    )


@pytest.fixture
def sample_error_chain():
    """Create a sample error chain with ExecutionError objects.

    Returns:
        List of ExecutionError objects for testing error recovery
    """
    return [
        ExecutionError(
            error_type="execution",
            error_message="NameError: name 'undefined_var' is not defined",
            failed_code="x = undefined_var + 1",
            attempt_number=1,
            stage="execution"
        ),
        ExecutionError(
            error_type="syntax",
            error_message="SyntaxError: invalid syntax",
            failed_code="def broken(\n    print('hi')",
            attempt_number=2,
            stage="generation"
        )
    ]


# Pytest markers for Python executor tests
def pytest_configure(config):
    """Register Python executor-specific pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "requires_container: marks tests that require container infrastructure"
    )
    config.addinivalue_line(
        "markers",
        "requires_epics: marks tests that require EPICS infrastructure"
    )

