"""Tests for pre-execution safety checks in python_execute MCP tool.

Covers: syntax errors caught before execution, exec()/eval() flagged,
prohibited imports blocked, valid code passes through.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn


def _get_python_execute():
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return get_tool_fn(execute)


@pytest.mark.unit
async def test_syntax_error_caught_before_execution(tmp_path, monkeypatch):
    """Code with syntax errors is rejected before execution."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    with assert_raises_error(error_type="safety_error") as _exc_ctx:
        await fn(
            code="def foo(\n  x = ",
            description="syntax error",
            execution_mode="readonly",
        )

    data = _exc_ctx["envelope"]
    assert any("Syntax error" in s for s in data["suggestions"])


@pytest.mark.unit
async def test_exec_call_flagged(tmp_path, monkeypatch):
    """Code containing exec() is flagged by safety checks."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    with assert_raises_error(error_type="safety_error") as _exc_ctx:
        await fn(
            code="exec('print(1)')",
            description="exec call",
            execution_mode="readonly",
        )

    data = _exc_ctx["envelope"]
    assert any("exec" in s.lower() for s in data["suggestions"])


@pytest.mark.unit
async def test_eval_call_flagged(tmp_path, monkeypatch):
    """Code containing eval() is flagged by safety checks."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    with assert_raises_error(error_type="safety_error") as _exc_ctx:
        await fn(
            code="result = eval('2 + 2')",
            description="eval call",
            execution_mode="readonly",
        )

    data = _exc_ctx["envelope"]
    assert any("eval" in s.lower() for s in data["suggestions"])


@pytest.mark.unit
async def test_prohibited_import_blocked(tmp_path, monkeypatch):
    """Code with prohibited imports is blocked."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    with assert_raises_error(error_type="safety_error") as _exc_ctx:
        await fn(
            code="import subprocess\nsubprocess.run(['ls'])",
            description="prohibited import",
            execution_mode="readonly",
        )

    data = _exc_ctx["envelope"]
    assert any("subprocess" in s.lower() for s in data["suggestions"])


@pytest.mark.unit
async def test_valid_code_passes_safety_checks(tmp_path, monkeypatch):
    """Valid code passes safety checks and executes normally."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.python_executor.executor import ExecutionResult

    mock_exec = AsyncMock(
        return_value=ExecutionResult(
            success=True, stdout="4\n", stderr="", execution_method_used="subprocess"
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

    data = extract_response_dict(result)
    assert data.get("error") is not True
    assert "4" in data["summary"]["output"]


@pytest.mark.unit
async def test_dunder_import_flagged(tmp_path, monkeypatch):
    """Code containing __import__ is flagged."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    with assert_raises_error(error_type="safety_error") as _exc_ctx:
        await fn(
            code="os = __import__('os')\nos.listdir('.')",
            description="dunder import",
            execution_mode="readonly",
        )

    _exc_ctx["envelope"]


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


# ============================================================================
# readonly import denylist — control-system client libraries
# ============================================================================


def _readonly_import_issues(code):
    from osprey.services.python_executor.analysis.safety_checks import check_readonly_imports

    return check_readonly_imports(code)


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        pytest.param("import epics", id="import"),
        pytest.param("import epics as e", id="import-as"),
        pytest.param("from epics import caput as _w", id="from-import-alias"),
        pytest.param("from epics.ca import put", id="from-submodule"),
        pytest.param("import p4p.client.thread", id="dotted-import"),
        pytest.param("from p4p.client.asyncio import Context", id="from-dotted"),
        pytest.param("import caproto", id="caproto"),
        pytest.param("import pvaccess", id="pvaccess"),
        pytest.param("import tango", id="tango"),
        pytest.param("from PyTango import DeviceProxy", id="pytango"),
        pytest.param("def f():\n    import epics\n    return epics", id="nested-in-function"),
    ],
)
def test_readonly_imports_denied(code):
    issues = _readonly_import_issues(code)
    assert issues, code
    assert all("readonly" in i for i in issues), issues


@pytest.mark.unit
@pytest.mark.parametrize(
    "code",
    [
        pytest.param("import numpy as np", id="numpy"),
        pytest.param("from osprey.runtime import read_channel", id="osprey-runtime"),
        pytest.param("import epicsarchiver_client", id="prefix-is-not-package"),
        pytest.param("# import epics\nprint('commented out')", id="comment-only"),
        pytest.param("s = 'import epics'", id="string-only"),
        pytest.param("x = epics.caput", id="no-import-statement"),
    ],
)
def test_readonly_imports_allowed(code):
    assert _readonly_import_issues(code) == []


@pytest.mark.unit
def test_readonly_imports_syntax_error_is_not_an_issue_here():
    """Syntax errors belong to check_syntax; this walker stays quiet."""
    assert _readonly_import_issues("def (:") == []


# ---------------------------------------------------------------------------
# Refused writes are audited
#
# "Log the attempt with the offending source for audit" is a requirement in its
# own right: an agent that tries to write while the session is readonly is a
# fact the operator needs after the fact, not only in the moment.
# ---------------------------------------------------------------------------


def _audit_records(root):
    import json

    from osprey.services.python_executor.refusal_audit import REFUSAL_LOG_FILENAME

    path = root / "var" / "audit" / REFUSAL_LOG_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.unit
async def test_denied_import_is_recorded_with_its_source(tmp_path, monkeypatch):
    """The import denylist refusal names the layer and keeps the offending code."""
    from osprey.services.python_executor import refusal_audit

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(refusal_audit, "audit_dir", lambda: tmp_path / "var" / "audit")

    code = "import epics\nepics.caput('SR:CORR:SP', 1.0)\n"
    fn = _get_python_execute()
    gates = "osprey.mcp_server.python_executor.tools._execution_gates"
    with patch(f"{gates}.notify_agent_activity_async", new=AsyncMock()):
        with assert_raises_error(error_type="safety_error"):
            await fn(code=code, description="import a client", execution_mode="readonly")

    (record,) = _audit_records(tmp_path)
    assert record["layer"] == "import_denylist"
    assert record["source"] == code
    assert record["description"] == "import a client"
    assert any("epics" in issue for issue in record["trigger"])


@pytest.mark.unit
async def test_a_clean_readonly_run_writes_no_audit_record(tmp_path, monkeypatch):
    """The log must stay a record of refusals, not of executions."""
    from osprey.services.python_executor import refusal_audit

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(refusal_audit, "audit_dir", lambda: tmp_path / "var" / "audit")

    fn = _get_python_execute()
    with patch(
        "osprey.mcp_server.python_executor.executor.execute_code", new=AsyncMock()
    ) as exec_code:
        exec_code.return_value = _clean_result()
        await fn(
            code="x = 1 + 1\nprint(x)",
            description="arithmetic",
            execution_mode="readonly",
            save_output=False,
        )

    assert _audit_records(tmp_path) == []


def _clean_result():
    from osprey.mcp_server.python_executor.executor import ExecutionResult

    return ExecutionResult(success=True, stdout="2\n", stderr="", execution_time_seconds=0.1)
