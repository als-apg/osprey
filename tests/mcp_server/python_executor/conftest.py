"""Fixtures for MCP python_executor adapter tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.mcp_server.python_executor.executor import ExecutionResult


@pytest.fixture
def mock_execution_result():
    """Factory for creating mock ExecutionResult objects."""

    def _make(
        success=True,
        stdout="42\n",
        stderr="",
        figures=None,
        execution_method_used="container",
        execution_time_seconds=1.5,
        error_message=None,
    ):
        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            figures=figures or [],
            execution_method_used=execution_method_used,
            execution_time_seconds=execution_time_seconds,
            error_message=error_message,
        )

    return _make


@pytest.fixture
def mock_container_executor():
    """Mock ContainerExecutor that returns configurable PythonExecutionEngineResult."""

    def _make(success=True, stdout="42\n", error_message=None, captured_figures=None):
        executor = AsyncMock()
        result = MagicMock()
        result.success = success
        result.stdout = stdout
        result.error_message = error_message
        result.captured_figures = captured_figures or []
        result.result_dict = None
        result.execution_time_seconds = 1.0
        executor.execute_code = AsyncMock(return_value=result)
        return executor

    return _make


@pytest.fixture
def execution_folder(tmp_path):
    """Create a temporary execution folder with standard structure."""
    folder = tmp_path / "execution_20240115_120000_abcd1234"
    folder.mkdir(parents=True)
    (folder / "figures").mkdir()
    return folder
