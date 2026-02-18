"""Tests for pre-execution safety checks in python_execute MCP tool.

Covers: syntax errors caught before execution, exec()/eval() flagged,
prohibited imports blocked, valid code passes through.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.mcp_server.conftest import get_tool_fn


def _get_python_execute():
    from osprey.mcp_server.python_executor.tools.python_execute import python_execute

    return get_tool_fn(python_execute)


@pytest.mark.unit
async def test_syntax_error_caught_before_execution(tmp_path, monkeypatch):
    """Code with syntax errors is rejected before execution."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    result = await fn(
        code="def foo(\n  x = ",
        description="syntax error",
        execution_mode="readonly",
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"
    assert any("Syntax error" in s for s in data["suggestions"])


@pytest.mark.unit
async def test_exec_call_flagged(tmp_path, monkeypatch):
    """Code containing exec() is flagged by safety checks."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    result = await fn(
        code="exec('print(1)')",
        description="exec call",
        execution_mode="readonly",
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"
    assert any("exec" in s.lower() for s in data["suggestions"])


@pytest.mark.unit
async def test_eval_call_flagged(tmp_path, monkeypatch):
    """Code containing eval() is flagged by safety checks."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    result = await fn(
        code="result = eval('2 + 2')",
        description="eval call",
        execution_mode="readonly",
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"
    assert any("eval" in s.lower() for s in data["suggestions"])


@pytest.mark.unit
async def test_prohibited_import_blocked(tmp_path, monkeypatch):
    """Code with prohibited imports is blocked."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    result = await fn(
        code="import subprocess\nsubprocess.run(['ls'])",
        description="prohibited import",
        execution_mode="readonly",
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"
    assert any("subprocess" in s.lower() for s in data["suggestions"])


@pytest.mark.unit
async def test_valid_code_passes_safety_checks(tmp_path, monkeypatch):
    """Valid code passes safety checks and executes normally."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.python_executor.executor import ExecutionResult

    mock_exec = AsyncMock(
        return_value=ExecutionResult(
            success=True, stdout="4\n", stderr="", execution_method_used="local"
        )
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection"
            ".detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="x = 2 + 2\nprint(x)",
            description="valid code",
            execution_mode="readonly",
        )

    data = json.loads(result)
    assert data.get("error") is not True
    assert "4" in data["summary"]["output"]


@pytest.mark.unit
async def test_dunder_import_flagged(tmp_path, monkeypatch):
    """Code containing __import__ is flagged."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    result = await fn(
        code="os = __import__('os')\nos.listdir('.')",
        description="dunder import",
        execution_mode="readonly",
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"


@pytest.mark.unit
def test_quick_safety_check_standalone():
    """Unit test for the quick_safety_check function directly."""
    from osprey.services.python_executor.analysis.safety_checks import quick_safety_check

    # Valid code
    passed, issues = quick_safety_check("x = 1 + 2")
    assert passed is True
    assert issues == []

    # Syntax error
    passed, issues = quick_safety_check("def (")
    assert passed is False
    assert any("Syntax error" in i for i in issues)

    # exec() call
    passed, issues = quick_safety_check("exec('code')")
    assert passed is False

    # open() — warning level, should NOT block
    passed, issues = quick_safety_check("f = open('file.txt')")
    assert passed is True
