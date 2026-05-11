"""Tests for the execute MCP tool (python_execute module).

Covers: readonly execution, write-pattern detection, execution errors,
data context saving, error format compliance, and adapter integration.

The tool delegates execution to the adapter (executor.py) which tries
container -> local -> in-process fallback.  Most tests mock pattern
detection; the adapter falls back to in-process exec when no config.yml
or containers are available.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.python_executor.executor import ExecutionResult
from tests.mcp_server.conftest import assert_error, assert_raises_error, extract_response_dict, get_tool_fn


def _get_python_execute():
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return get_tool_fn(execute)


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


@pytest.mark.unit
async def test_python_execute_readonly(tmp_path, monkeypatch):
    """Readonly execution runs code and returns stdout in summary."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(success=True, stdout="42\n")

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print(6 * 7)",
            description="simple multiplication",
            execution_mode="readonly",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "42" in data["summary"]["output"]
    assert data["summary"]["status"] == "Success"


@pytest.mark.unit
async def test_python_execute_write_pattern_detection(tmp_path, monkeypatch):
    """Code containing EPICS write patterns in readonly mode returns safety_error."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
        return_value={
            "has_writes": True,
            "has_reads": False,
            "detected_patterns": {"caput": ["caput('TEST:PV', 42.0)"]},
        },
    ):
        fn = _get_python_execute()
        with assert_raises_error(error_type="safety_error") as _exc_ctx:
            await fn(
                code="caput('TEST:PV', 42.0)",
                description="write to PV",
                execution_mode="readonly",
            )

    data = _exc_ctx["envelope"]
    assert "suggestions" in data


@pytest.mark.unit
async def test_python_execute_execution_error(tmp_path, monkeypatch):
    """Code execution error raises ToolError with execution_error envelope."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(
        success=False,
        stderr="NameError: name 'undefined_var' is not defined",
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
        assert_raises_error(error_type="execution_error") as ctx,
    ):
        fn = _get_python_execute()
        await fn(
            code="print(undefined_var)",
            description="error case",
            execution_mode="readonly",
        )

    envelope = ctx["envelope"]
    assert "NameError" in envelope["error_message"]
    summary = envelope["details"]["summary"]
    assert summary["status"] == "Failed"
    assert summary["has_errors"] is True


@pytest.mark.unit
async def test_python_execute_data_file_saving(tmp_path, monkeypatch):
    """Execution saves output to artifact store data file."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(success=True, stdout="42\n")

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="x = 42\nprint(x)",
            description="test data file save",
            execution_mode="readonly",
            save_output=True,
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    assert "data_file" in data
    # data_file is a project-CWD-relative path the agent can open() directly
    assert data["data_file"].startswith("_agent_data/artifacts/")
    assert (tmp_path / data["data_file"]).exists()


@pytest.mark.unit
async def test_python_execute_no_saving(tmp_path, monkeypatch):
    """save_output=False returns inline result without data file."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(success=True, stdout="hello\n")

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('hello')",
            description="no save",
            execution_mode="readonly",
            save_output=False,
        )

    data = extract_response_dict(result)
    # When save_output=False, returns inline (no context entry)
    assert "data_file" not in data
    assert "stdout" in data
    assert "hello" in data["stdout"]


@pytest.mark.unit
async def test_python_execute_readwrite_mode(tmp_path, monkeypatch):
    """readwrite mode allows code with write patterns."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(success=True, stdout="write mode\n")

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={
                "has_writes": True,
                "has_reads": False,
                "detected_patterns": {"caput": ["caput('TEST:PV', 1.0)"]},
            },
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('write mode')",
            description="write operation",
            execution_mode="readwrite",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["summary"]["status"] == "Success"


@pytest.mark.unit
async def test_python_execute_empty_code(tmp_path, monkeypatch):
    """Empty code returns validation error."""
    monkeypatch.chdir(tmp_path)

    fn = _get_python_execute()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(code="", description="empty")

    data = _exc_ctx["envelope"]


@pytest.mark.unit
async def test_python_execute_pattern_detection_import_error(tmp_path, monkeypatch):
    """Pattern detection import failure falls back to allowing execution."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(success=True, stdout="fallback\n")

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            side_effect=ImportError("pattern detection unavailable"),
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('fallback')",
            description="import error fallback",
            execution_mode="readonly",
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "fallback" in data["summary"]["output"]


# ============================================================================
# Integration tests — adapter integration with the MCP tool
# ============================================================================


@pytest.mark.unit
async def test_python_execute_uses_adapter(tmp_path, monkeypatch):
    """Tool calls execute_code adapter instead of bare exec()."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(success=True, stdout="adapter called\n")

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('hello')",
            description="adapter test",
            execution_mode="readonly",
        )

    mock_exec.assert_called_once()
    call_kwargs = mock_exec.call_args.kwargs
    assert call_kwargs["code"] == "print('hello')"
    assert call_kwargs["execution_mode"] == "readonly"
    assert call_kwargs["description"] == "adapter test"
    assert "save_artifact_fn" not in call_kwargs

    data = extract_response_dict(result)
    assert data["summary"]["status"] == "Success"
    assert "adapter called" in data["summary"]["output"]


@pytest.mark.unit
async def test_safety_checks_run_before_adapter(tmp_path, monkeypatch):
    """quick_safety_check() blocks BEFORE adapter is called."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code()

    with (
        patch(
            "osprey.services.python_executor.analysis.safety_checks.quick_safety_check",
            return_value=(False, ["exec() call detected"]),
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        with assert_raises_error(error_type="safety_error") as _exc_ctx:
            await fn(
                code="exec('bad')",
                description="safety check test",
                execution_mode="readonly",
            )

    # Adapter should NOT have been called
    mock_exec.assert_not_called()

    data = _exc_ctx["envelope"]


@pytest.mark.unit
async def test_pattern_detection_runs_before_adapter(tmp_path, monkeypatch):
    """Write pattern detection blocks readonly mode BEFORE adapter is called."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code()

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={
                "has_writes": True,
                "has_reads": False,
                "detected_patterns": {"caput": ["caput('PV', 1)"]},
            },
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        with assert_raises_error(error_type="safety_error") as _exc_ctx:
            await fn(
                code="caput('TEST:PV', 1.0)",
                description="pattern detection test",
                execution_mode="readonly",
            )

    mock_exec.assert_not_called()

    data = _exc_ctx["envelope"]


@pytest.mark.unit
async def test_adapter_result_maps_to_tool_response(tmp_path, monkeypatch):
    """Adapter's ExecutionResult correctly maps to the JSON response."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(
        success=True,
        stdout="output line 1\noutput line 2\n",
        execution_method_used="container",
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('test')",
            description="mapping test",
            execution_mode="readonly",
        )

    data = extract_response_dict(result)
    assert data["summary"]["status"] == "Success"
    assert "output line 1" in data["summary"]["output"]
    assert data["summary"]["has_errors"] is False


@pytest.mark.unit
async def test_adapter_error_returns_tool_error(tmp_path, monkeypatch):
    """When adapter returns failed result, tool raises ToolError with envelope."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(
        success=False,
        stderr="RuntimeError: something broke",
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
        assert_raises_error(error_type="execution_error") as ctx,
    ):
        fn = _get_python_execute()
        await fn(
            code="raise RuntimeError('something broke')",
            description="error test",
            execution_mode="readonly",
        )

    envelope = ctx["envelope"]
    assert "RuntimeError" in envelope["error_message"]
    summary = envelope["details"]["summary"]
    assert summary["status"] == "Failed"
    assert summary["has_errors"] is True


@pytest.mark.unit
async def test_notebook_artifact_created_from_adapter_result(tmp_path, monkeypatch):
    """Post-execution notebook uses stdout/stderr from adapter result."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(
        success=True,
        stdout="notebook output\n",
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('notebook output')",
            description="notebook creation test",
            execution_mode="readonly",
        )

    data = extract_response_dict(result)
    # Notebook artifact should be created
    assert "notebook_artifact_id" in data or "artifact_ids" in data.get("summary", {})


@pytest.mark.unit
async def test_figure_artifacts_created_post_execution(tmp_path, monkeypatch):
    """Figures from adapter result are saved as gallery artifacts."""
    monkeypatch.chdir(tmp_path)

    # Create a figure file that the adapter "found"
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    fig_file = fig_dir / "plot.png"
    fig_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG header

    mock_exec = _mock_execute_code(
        success=True,
        stdout="plot created\n",
        figures=[fig_file],
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="import matplotlib; ...",
            description="figure artifact test",
            execution_mode="readonly",
        )

    data = extract_response_dict(result)
    # Figure should have been saved as an artifact
    artifact_ids = data.get("summary", {}).get("artifact_ids", [])
    # At least notebook artifact + figure artifact
    assert len(artifact_ids) >= 1


@pytest.mark.unit
async def test_executor_error_returns_failure_result(tmp_path, monkeypatch):
    """When execution setup fails, execute_code returns ExecutionResult(success=False)."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.python_executor.executor import execute_code

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor._create_execution_folder",
            side_effect=RuntimeError("disk full"),
        ),
    ):
        result = await execute_code(
            code="print('hello')",
            execution_mode="readonly",
            description="error test",
        )

    assert result.success is False
    assert "disk full" in result.error_message


@pytest.mark.unit
def test_executor_no_in_process_fallback():
    """Confirm the in-process fallback function has been removed."""
    from osprey.mcp_server.python_executor import executor

    assert not hasattr(executor, "_execute_in_process_fallback")


@pytest.mark.unit
async def test_data_context_saves_adapter_result(tmp_path, monkeypatch):
    """ArtifactStore.save_data() receives full result including execution_method."""
    monkeypatch.chdir(tmp_path)

    mock_exec = _mock_execute_code(
        success=True,
        stdout="context test\n",
        execution_method_used="local",
    )

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('context test')",
            description="data context test",
            execution_mode="readonly",
            save_output=True,
        )

    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert "artifact_id" in data
    assert "data_file" in data

    # data_file is a project-CWD-relative path the agent can open() directly
    assert data["data_file"].startswith("_agent_data/artifacts/")
    data_file = tmp_path / data["data_file"]
    assert data_file.exists()
    # ArtifactStore writes raw JSON (no envelope)
    saved_data = json.loads(data_file.read_text())
    assert saved_data["execution_method"] == "local"
    assert saved_data["stdout"] == "context test\n"


# ============================================================================
# save_artifact() in subprocess — wrapper injection + artifact collection
# ============================================================================


# ============================================================================
# Unit tests — codegen and collection helpers
# ============================================================================


@pytest.mark.unit
def test_save_artifact_injection_generates_code():
    """ExecutionWrapper._get_save_artifact_injection() generates valid Python."""
    import ast

    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(execution_mode="local")
    code = wrapper._get_save_artifact_injection()

    assert "def save_artifact(" in code
    assert "manifest.json" in code
    # Must be valid Python
    ast.parse(code)


@pytest.mark.unit
def test_save_artifact_in_full_wrapper():
    """create_wrapper() includes save_artifact() in the output."""
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(execution_mode="local")
    full_code = wrapper.create_wrapper("x = 1")

    assert "def save_artifact(" in full_code
    assert "artifacts/manifest.json" in full_code or "manifest.json" in full_code


@pytest.mark.unit
def test_collect_artifacts_reads_manifest(tmp_path):
    """_collect_artifacts() reads manifest.json and returns artifact dicts."""
    import json

    from osprey.mcp_server.python_executor.executor import _collect_artifacts

    # Set up an artifacts/ subdirectory with a manifest and file
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    art_file = art_dir / "abc123_test.json"
    art_file.write_bytes(b'{"key": "value"}')
    manifest = [
        {
            "id": "abc123",
            "filename": "abc123_test.json",
            "title": "Test JSON",
            "description": "A test artifact",
            "artifact_type": "json",
            "mime_type": "application/json",
            "size_bytes": 16,
        }
    ]
    (art_dir / "manifest.json").write_text(json.dumps(manifest))

    result = _collect_artifacts(tmp_path)
    assert len(result) == 1
    assert result[0]["title"] == "Test JSON"
    assert result[0]["artifact_type"] == "json"
    assert result[0]["path"] == art_file


@pytest.mark.unit
def test_collect_artifacts_empty_when_no_manifest(tmp_path):
    """_collect_artifacts() returns empty list when no manifest exists."""
    from osprey.mcp_server.python_executor.executor import _collect_artifacts

    assert _collect_artifacts(tmp_path) == []


@pytest.mark.unit
async def test_save_artifact_registered_in_gallery(tmp_path, monkeypatch):
    """Artifacts from exec_result.artifacts are saved to the gallery."""
    import json

    monkeypatch.chdir(tmp_path)

    # Create artifact file that _collect_artifacts would return
    art_dir = tmp_path / "collected_artifacts"
    art_dir.mkdir()
    art_file = art_dir / "abc123_test_data.json"
    art_file.write_bytes(b'{"key": "value"}')

    mock_exec = _mock_execute_code(
        success=True,
        stdout="ok\n",
    )
    # Inject artifacts into the result
    mock_exec.return_value.artifacts = [
        {
            "path": art_file,
            "title": "Test Data",
            "description": "From subprocess",
            "artifact_type": "json",
            "mime_type": "application/json",
        }
    ]

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.execute_code",
            mock_exec,
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="save_artifact({'key': 'value'}, title='Test Data')",
            description="subprocess artifact test",
            execution_mode="readonly",
        )

    data = extract_response_dict(result)
    # Should have: 1 subprocess artifact + 1 notebook artifact
    artifact_ids = data.get("summary", {}).get("artifact_ids", data.get("artifact_ids", []))
    assert len(artifact_ids) >= 2


# ============================================================================
# REGRESSION tests — real subprocess execution of save_artifact()
#
# These tests exercise the actual bug path: user code calling save_artifact()
# inside a subprocess spawned by _execute_via_local().  Before the fix, this
# produced NameError: name 'save_artifact' is not defined.
# ============================================================================


@pytest.mark.integration
async def test_save_artifact_no_name_error_in_subprocess(tmp_path):
    """REGRESSION: save_artifact() must not raise NameError in subprocess mode.

    This is the exact bug that was reported.  Before the fix, the wrapper
    template did not define save_artifact(), so any user code calling it
    in subprocess mode would crash with NameError.
    """
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    execution_folder = tmp_path / "exec"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    result = await _execute_via_local(
        code='save_artifact({"key": "value"}, title="Test JSON", description="regression test")',
        execution_mode="readonly",
        config=config,
        execution_folder=execution_folder,
        limits_validator=None,
    )

    # The core assertion: execution must succeed, not crash with NameError
    assert result.success, (
        f"save_artifact() failed in subprocess — likely NameError.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "NameError" not in result.stderr
    assert "save_artifact" not in result.stderr or "Artifact saved" in result.stdout


@pytest.mark.integration
async def test_save_artifact_creates_manifest_in_subprocess(tmp_path):
    """REGRESSION: save_artifact() in subprocess must create artifacts/manifest.json.

    Verifies the full file-based handoff: the subprocess writes the manifest,
    and the executor can find it.
    """
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    execution_folder = tmp_path / "exec"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    result = await _execute_via_local(
        code='save_artifact({"beam_current": 500.2}, title="Beam Data", description="test")',
        execution_mode="readonly",
        config=config,
        execution_folder=execution_folder,
        limits_validator=None,
    )

    assert result.success, f"Execution failed: {result.stderr}"

    # Manifest must exist
    manifest_path = execution_folder / "artifacts" / "manifest.json"
    assert manifest_path.exists(), "save_artifact() did not create artifacts/manifest.json"

    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 1
    assert manifest[0]["title"] == "Beam Data"
    assert manifest[0]["artifact_type"] == "json"

    # The actual artifact file must exist
    artifact_file = execution_folder / "artifacts" / manifest[0]["filename"]
    assert artifact_file.exists(), f"Artifact file missing: {manifest[0]['filename']}"

    # Content must be the serialized JSON
    content = json.loads(artifact_file.read_text())
    assert content["beam_current"] == 500.2


@pytest.mark.integration
async def test_save_artifact_collected_into_execution_result(tmp_path):
    """REGRESSION: _execute_via_local() must populate ExecutionResult.artifacts.

    End-to-end: subprocess writes artifact → executor collects it → result
    contains artifact metadata with valid file path.
    """
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    execution_folder = tmp_path / "exec"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    result = await _execute_via_local(
        code='save_artifact("# Analysis Report", title="Report", description="md test")',
        execution_mode="readonly",
        config=config,
        execution_folder=execution_folder,
        limits_validator=None,
    )

    assert result.success, f"Execution failed: {result.stderr}"

    # Artifacts must be collected into the result
    assert len(result.artifacts) == 1, (
        f"Expected 1 artifact in ExecutionResult, got {len(result.artifacts)}"
    )
    art = result.artifacts[0]
    assert art["title"] == "Report"
    assert art["artifact_type"] == "markdown"
    assert art["mime_type"] == "text/markdown"
    assert art["path"].exists(), f"Collected artifact path does not exist: {art['path']}"


@pytest.mark.integration
async def test_multiple_save_artifact_calls_in_subprocess(tmp_path):
    """REGRESSION: Multiple save_artifact() calls in one subprocess all persist."""
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    execution_folder = tmp_path / "exec"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    code = """
save_artifact({"key": "first"}, title="First Artifact", description="json")
save_artifact("# Second", title="Second Artifact", description="markdown")
save_artifact("<h1>Third</h1></h1>", title="Third Artifact", description="html")
"""

    result = await _execute_via_local(
        code=code,
        execution_mode="readonly",
        config=config,
        execution_folder=execution_folder,
        limits_validator=None,
    )

    assert result.success, f"Execution failed: {result.stderr}"

    # All three artifacts must be collected
    assert len(result.artifacts) == 3, (
        f"Expected 3 artifacts, got {len(result.artifacts)}: "
        f"{[a['title'] for a in result.artifacts]}"
    )
    titles = {a["title"] for a in result.artifacts}
    assert titles == {"First Artifact", "Second Artifact", "Third Artifact"}

    # Verify type detection worked correctly
    types = {a["title"]: a["artifact_type"] for a in result.artifacts}
    assert types["First Artifact"] == "json"
    assert types["Second Artifact"] == "markdown"
    assert types["Third Artifact"] == "html"


@pytest.mark.integration
async def test_save_artifact_string_type_detection_in_subprocess(tmp_path):
    """REGRESSION: Smart type detection works inside the subprocess save_artifact()."""
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    execution_folder = tmp_path / "exec"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    code = """
# Plain string → markdown
save_artifact("Just a plain string", title="Plain Text")
# HTML string → html
save_artifact("<html><body><p>Hello</p></body></html>", title="HTML Content")
# Dict → json
save_artifact({"x": [1, 2, 3]}, title="JSON Data")
# List → json
save_artifact([1, 2, 3], title="List Data")
"""

    result = await _execute_via_local(
        code=code,
        execution_mode="readonly",
        config=config,
        execution_folder=execution_folder,
        limits_validator=None,
    )

    assert result.success, f"Execution failed: {result.stderr}"
    assert len(result.artifacts) == 4

    type_map = {a["title"]: a["artifact_type"] for a in result.artifacts}
    assert type_map["Plain Text"] == "markdown"
    assert type_map["HTML Content"] == "html"
    assert type_map["JSON Data"] == "json"
    assert type_map["List Data"] == "json"


# ============================================================================
# REGRESSION: subprocess cwd must be project root, not execution sandbox
#
# User code receives file paths like "_agent_data/data/002_archiver_read.json"
# from other tools. These are relative to the project root. When the subprocess
# cwd is the execution sandbox, these paths don't resolve.
# ============================================================================


@pytest.mark.integration
async def test_subprocess_can_read_workspace_files_by_relative_path(tmp_path):
    """REGRESSION: User code must be able to open workspace files by relative path.

    Other MCP tools return paths like '_agent_data/data/002_archiver_read.json'.
    The subprocess must resolve these relative to the project root, not the
    execution sandbox directory.

    Before the fix: subprocess cwd was the execution sandbox dir, so relative
    paths to workspace files failed with FileNotFoundError.
    """
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    # Simulate the project layout: project_root/_agent_data/data/file.json
    project_root = tmp_path / "project"
    project_root.mkdir()

    workspace_root = project_root / "_agent_data"
    workspace_data = workspace_root / "data"
    workspace_data.mkdir(parents=True)
    data_file = workspace_data / "002_archiver_read.json"
    data_file.write_text('{"channels": ["SR:CURRENT"]}')

    # Execution folder is a subdirectory under workspace
    exec_base = workspace_data / "python_executions"
    exec_base.mkdir(parents=True)
    execution_folder = exec_base / "20260217_test"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    # User code tries to open a workspace file by the relative path
    # that another MCP tool returned
    code = """
import json
with open("_agent_data/data/002_archiver_read.json") as f:
    data = json.load(f)
print(f"Loaded: {data['channels']}")
"""

    # Mock _resolve_project_root to return our test project root
    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=project_root,
    ):
        result = await _execute_via_local(
            code=code,
            execution_mode="readonly",
            config=config,
            execution_folder=execution_folder,
            limits_validator=None,
        )

    assert result.success, (
        f"Failed to read workspace file by relative path.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Loaded:" in result.stdout
    assert "SR:CURRENT" in result.stdout


@pytest.mark.integration
async def test_subprocess_outputs_still_written_to_execution_folder(tmp_path):
    """Execution metadata and artifacts still go to execution_folder after cwd fix."""
    from osprey.mcp_server.python_executor.executor import _execute_via_local

    project_root = tmp_path / "project"
    project_root.mkdir()

    workspace_root = project_root / "_agent_data"
    exec_base = workspace_root / "data" / "python_executions"
    exec_base.mkdir(parents=True)
    execution_folder = exec_base / "20260217_test2"
    execution_folder.mkdir()
    (execution_folder / "figures").mkdir()

    config = {"timeout": 30, "python_env_path": None}

    with patch(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        return_value=project_root,
    ):
        result = await _execute_via_local(
            code='save_artifact({"test": True}, title="CWD Test")',
            execution_mode="readonly",
            config=config,
            execution_folder=execution_folder,
            limits_validator=None,
        )

    assert result.success, f"Execution failed: {result.stderr}"

    # Execution metadata must still be in execution_folder
    assert (execution_folder / "execution_metadata.json").exists(), (
        "execution_metadata.json not found in execution folder after cwd change"
    )

    # Artifacts must still be in execution_folder/artifacts/
    assert len(result.artifacts) == 1, "Artifact not collected after cwd change"
