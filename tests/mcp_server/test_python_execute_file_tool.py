"""Tests for the execute_file MCP tool (python_execute_file module).

Covers: file validation, path resolution, containment checks, safety checks,
pattern detection, script args, encoding errors, and delegation to the
shared response builder.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.python_executor.executor import ExecutionResult
from tests.mcp_server.conftest import get_tool_fn


def _get_python_execute_file():
    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    return get_tool_fn(execute_file)


def _mock_execute_code(
    success=True,
    stdout="",
    stderr="",
    figures=None,
    execution_method_used="local",
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


def _no_writes():
    """Standard pattern detection return value with no writes."""
    return {"has_writes": False, "has_reads": False, "detected_patterns": {}}


@pytest.mark.unit
async def test_execute_file_basic(tmp_path, monkeypatch):
    """Basic execution reads file, injects preamble, and returns summary."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "hello.py"
    script.write_text("print('hello world')\n")

    mock_exec = _mock_execute_code(success=True, stdout="hello world\n")

    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value=_no_writes(),
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(script),
            description="hello world test",
            execution_mode="readonly",
        )

    # Verify execute_code was called with augmented code containing preamble
    call_args = mock_exec.call_args
    executed_code = call_args.kwargs["code"]
    assert "import sys" in executed_code
    assert f"__file__ = {str(script.resolve())!r}" in executed_code
    assert "print('hello world')" in executed_code

    data = json.loads(result)
    assert data["summary"]["status"] == "Success"
    assert "hello world" in data["summary"]["output"]


@pytest.mark.unit
async def test_execute_file_relative_path(tmp_path, monkeypatch):
    """Relative file paths resolve against project root."""
    monkeypatch.chdir(tmp_path)

    subdir = tmp_path / "scripts"
    subdir.mkdir()
    script = subdir / "run.py"
    script.write_text("x = 1\n")

    mock_exec = _mock_execute_code(success=True, stdout="")

    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value=_no_writes(),
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path="scripts/run.py",
            description="relative path test",
        )

    # Should have been called (path resolved successfully)
    mock_exec.assert_called_once()
    data = json.loads(result)
    assert data["summary"]["status"] == "Success"


@pytest.mark.unit
async def test_execute_file_not_found(tmp_path, monkeypatch):
    """Non-existent file returns file_not_found error."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=tmp_path,
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(tmp_path / "missing.py"),
            description="missing file test",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "file_not_found"


@pytest.mark.unit
async def test_execute_file_not_python(tmp_path, monkeypatch):
    """Non-.py file returns validation_error."""
    monkeypatch.chdir(tmp_path)

    txt_file = tmp_path / "data.txt"
    txt_file.write_text("not python\n")

    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=tmp_path,
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(txt_file),
            description="not python test",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "not a python file" in data["error_message"].lower()


@pytest.mark.unit
async def test_execute_file_outside_project_root(tmp_path, monkeypatch):
    """File outside project root returns validation_error (containment)."""
    monkeypatch.chdir(tmp_path)

    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("print('escaped')\n")

    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=project_root,
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(outside_file),
            description="containment test",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "outside" in data["error_message"].lower()


@pytest.mark.unit
async def test_execute_file_safety_check_blocks(tmp_path, monkeypatch):
    """File with dangerous code is blocked by safety checks."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "danger.py"
    # Use a benign-looking string; the mock controls whether it's blocked
    script.write_text("run_dangerous_code()\n")

    mock_exec = _mock_execute_code()

    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.safety_checks.quick_safety_check",
            return_value=(False, ["dangerous call detected"]),
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(script),
            description="safety check test",
        )

    mock_exec.assert_not_called()
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"


@pytest.mark.unit
async def test_execute_file_readonly_blocks_writes(tmp_path, monkeypatch):
    """File with write patterns in readonly mode is blocked."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "writer.py"
    script.write_text("epics.caput('PV', 1)\n")

    mock_exec = _mock_execute_code()

    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={
                "has_writes": True,
                "has_reads": False,
                "detected_patterns": {"writes": ["caput"], "reads": []},
            },
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(script),
            description="write pattern test",
            execution_mode="readonly",
        )

    mock_exec.assert_not_called()
    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "safety_error"


@pytest.mark.unit
async def test_execute_file_readwrite_allows_writes(tmp_path, monkeypatch):
    """Write patterns pass in readwrite mode."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "writer.py"
    script.write_text("epics.caput('PV', 1)\n")

    mock_exec = _mock_execute_code(success=True, stdout="done\n")

    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={
                "has_writes": True,
                "has_reads": False,
                "detected_patterns": {"writes": ["caput"], "reads": []},
            },
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(script),
            description="readwrite test",
            execution_mode="readwrite",
        )

    mock_exec.assert_called_once()
    data = json.loads(result)
    assert data["summary"]["status"] == "Success"


@pytest.mark.unit
async def test_execute_file_script_args(tmp_path, monkeypatch):
    """Script args are injected into sys.argv preamble."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "args.py"
    script.write_text("import sys; print(sys.argv)\n")

    mock_exec = _mock_execute_code(success=True, stdout="")

    with (
        patch(
            "osprey.mcp_server.python_executor.executor._resolve_project_root",
            return_value=tmp_path,
        ),
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value=_no_writes(),
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute_file()
        await fn(
            file_path=str(script),
            description="args test",
            script_args=["--verbose", "input.dat"],
        )

    call_args = mock_exec.call_args
    executed_code = call_args.kwargs["code"]
    assert "--verbose" in executed_code
    assert "input.dat" in executed_code
    # sys.argv should contain script path + args
    assert "sys.argv = [" in executed_code


@pytest.mark.unit
async def test_execute_file_empty(tmp_path, monkeypatch):
    """Empty file returns validation_error."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "empty.py"
    script.write_text("")

    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=tmp_path,
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(script),
            description="empty file test",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "empty" in data["error_message"].lower()


@pytest.mark.unit
async def test_execute_file_encoding_error(tmp_path, monkeypatch):
    """Binary file returns clean validation error."""
    monkeypatch.chdir(tmp_path)

    script = tmp_path / "binary.py"
    script.write_bytes(b"\x80\x81\x82\xff\xfe")

    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=tmp_path,
    ):
        fn = _get_python_execute_file()
        result = await fn(
            file_path=str(script),
            description="encoding error test",
        )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "utf-8" in data["error_message"].lower()


@pytest.mark.unit
async def test_execute_file_empty_path():
    """Empty file_path returns validation_error."""
    fn = _get_python_execute_file()
    result = await fn(
        file_path="",
        description="empty path test",
    )

    data = json.loads(result)
    assert data["error"] is True
    assert data["error_type"] == "validation_error"
    assert "no file path" in data["error_message"].lower()
