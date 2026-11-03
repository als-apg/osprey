"""A stand-in for the executor's ``execute_code``, for the execute-tool suites."""

from __future__ import annotations

from unittest.mock import AsyncMock

from osprey.mcp_server.python_executor.executor import ExecutionResult


def mock_execute_code(
    success=True,
    stdout="",
    stderr="",
    figures=None,
    execution_method_used="subprocess",
):
    """Build an AsyncMock for execute_code returning a configured ExecutionResult."""
    result = ExecutionResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        figures=figures or [],
        execution_method_used=execution_method_used,
        execution_time_seconds=0.1,
    )
    return AsyncMock(return_value=result)
