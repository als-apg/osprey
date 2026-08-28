"""Tests for the shared python-executor response builder.

``build_execution_response`` turns an ``ExecutionResult`` into either a
``CallToolResult`` (success) or a raised ``ToolError`` carrying the OSPREY error
envelope (execution reported errors) — the fastmcp-safe error path. Tests run
against a real ArtifactStore in a tmp workspace and cover: the inline
(save_output=False) and persisted (save_output=True) branches, error handling in
each, notebook auto-save, and figure/artifact collection.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_SETUP,
    FAILURE_KIND_TIMEOUT,
    ExecutionResult,
)
from osprey.mcp_server.python_executor.tools._response_builder import build_execution_response
from tests.mcp_server.conftest import assert_error, extract_response_dict


@pytest.fixture(autouse=True)
def _store(tmp_path):
    """Initialize the ArtifactStore singleton in a throwaway workspace."""
    from osprey.stores.artifact_store import initialize_artifact_store

    initialize_artifact_store(workspace_root=tmp_path)
    yield


def _ok_result(stdout="hello\n", stderr="", **kw) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        stdout=stdout,
        stderr=stderr,
        execution_method_used="subprocess",
        **kw,
    )


def _err_result(stderr="Traceback: boom", **kw) -> ExecutionResult:
    return ExecutionResult(
        success=False,
        stdout="partial\n",
        stderr=stderr,
        execution_method_used="subprocess",
        **kw,
    )


def _setup_failed_result() -> ExecutionResult:
    """What ``execute_code`` returns when the sandbox never started."""
    return ExecutionResult(
        success=False,
        stdout="",
        stderr="Traceback (most recent call last):\n  ...\nOSError: disk full",
        execution_method_used="subprocess",
        error_message="Execution setup failed: disk full",
        failure_kind=FAILURE_KIND_SETUP,
    )


async def _build(exec_result, *, save_output, patterns=None):
    return await build_execution_response(
        code="print('hi')",
        description="demo run",
        execution_mode="readonly",
        exec_result=exec_result,
        patterns=patterns or {},
        save_output=save_output,
    )


# ---------------------------------------------------------------------------
# Inline path (save_output=False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_inline_success_returns_summary():
    result = await _build(_ok_result(), save_output=False)
    assert result.isError is False
    data = extract_response_dict(result)
    assert data["description"] == "demo run"
    assert data["execution_mode"] == "readonly"
    assert data["execution_method"] == "subprocess"
    assert data["stdout"] == "hello\n"
    assert data["has_errors"] is False


@pytest.mark.unit
async def test_inline_error_raises_execution_error():
    with pytest.raises(Exception) as exc_info:
        await _build(_err_result(stderr="ValueError: nope"), save_output=False)
    envelope = assert_error(str(exc_info.value), error_type="execution_error")
    assert "ValueError: nope" in envelope["error_message"]


# ---------------------------------------------------------------------------
# Who failed: the envelope names the subsystem and classes the cause
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("save_output", [False, True])
async def test_setup_failure_is_a_service_outage_not_a_code_error(save_output):
    """A sandbox that never started is ``service_unavailable`` (Connection class).

    Before this, a dead backend went out as ``execution_error`` and the
    error-guidance hook told the agent to help the user fix *their* code — the
    misattribution in #465. The submitted code never ran, so the envelope
    says so and names the subsystem.
    """
    with pytest.raises(Exception) as exc_info:
        await _build(_setup_failed_result(), save_output=save_output)
    envelope = assert_error(str(exc_info.value), error_type="service_unavailable")
    assert envelope["error_message"] == "Execution setup failed: disk full"
    assert envelope["details"]["subsystem"] == "python_executor"
    assert envelope["details"]["kind"] == "setup_failed"
    assert envelope["suggestions"], "an outage with no next step is what the hook mis-advised on"
    assert any("python_executor" in s for s in envelope["suggestions"])
    # The traceback still travels, in details rather than as the message.
    assert "disk full" in json.dumps(envelope["details"])


@pytest.mark.unit
@pytest.mark.parametrize("save_output", [False, True])
async def test_script_error_names_subsystem_and_suggests_a_fix(save_output):
    """The user's own traceback stays ``execution_error`` — with a next step."""
    with pytest.raises(Exception) as exc_info:
        await _build(_err_result(stderr="ValueError: nope"), save_output=save_output)
    envelope = assert_error(str(exc_info.value), error_type="execution_error")
    assert envelope["details"]["subsystem"] == "python_executor"
    assert envelope["details"]["kind"] == "script_error"
    assert envelope["suggestions"]


@pytest.mark.unit
async def test_timeout_is_an_execution_error_with_its_own_kind():
    """A run the sandbox killed is the script's problem, distinguished in details."""
    result = _err_result(
        stderr="Execution timed out after 30 seconds", failure_kind=FAILURE_KIND_TIMEOUT
    )
    with pytest.raises(Exception) as exc_info:
        await _build(result, save_output=False)
    envelope = assert_error(str(exc_info.value), error_type="execution_error")
    assert envelope["details"]["kind"] == "timeout"
    assert any("timeout" in s for s in envelope["suggestions"])


# ---------------------------------------------------------------------------
# Persisted path (save_output=True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_persisted_success_returns_tool_response_with_notebook():
    result = await _build(_ok_result(), save_output=True)
    assert result.isError is False
    data = extract_response_dict(result)
    # A notebook artifact is auto-saved for every execution and surfaced.
    assert "notebook_artifact_id" in data
    assert data["notebook_artifact_id"] in data.get("artifact_ids", [])


@pytest.mark.unit
async def test_execution_method_is_surfaced_honestly():
    """The response builder passes ``execution_method_used`` through verbatim —
    it is not the vocabulary choke point (that's ``resolve_execution_method``),
    so whatever the executor resolved is what callers see, in both response
    shapes. The persisted path surfaces it in the compact ``access_details``
    (alongside ``execution_mode``), not just buried in the full stored artifact.
    """
    result = await _build(_ok_result(), save_output=True)
    data = extract_response_dict(result)
    assert data["access_details"]["execution_method"] == "subprocess"

    result = await _build(_ok_result(), save_output=False)
    data = extract_response_dict(result)
    assert data["execution_method"] == "subprocess"


@pytest.mark.unit
async def test_persisted_error_raises_execution_error():
    with pytest.raises(Exception) as exc_info:
        await _build(_err_result(stderr="RuntimeError: kaput"), save_output=True)
    envelope = assert_error(str(exc_info.value), error_type="execution_error")
    assert "RuntimeError: kaput" in envelope["error_message"]
    # The error envelope carries the persisted response as structured details.
    assert isinstance(envelope["details"], dict)


# ---------------------------------------------------------------------------
# Figure / artifact collection
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_figures_are_saved_as_artifacts(tmp_path):
    fig = tmp_path / "plot.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    result = await _build(_ok_result(figures=[fig]), save_output=False)
    data = extract_response_dict(result)
    # The figure plus the always-on notebook yield at least one artifact id.
    assert data.get("artifact_ids")


@pytest.mark.unit
async def test_subprocess_artifacts_are_saved(tmp_path):
    art_path = tmp_path / "table.csv"
    art_path.write_text("a,b\n1,2\n")
    art = {
        "path": art_path,
        "artifact_type": "data",
        "title": "Table",
        "description": "a small table",
        "mime_type": "text/csv",
    }
    result = await _build(_ok_result(artifacts=[art]), save_output=False)
    data = extract_response_dict(result)
    assert data.get("artifact_ids")


@pytest.mark.unit
async def test_bad_figure_path_is_non_fatal():
    """A figure path that can't be read is logged and skipped, not raised."""
    result = await _build(
        _ok_result(figures=[Path("/nonexistent/does-not-exist.png")]),
        save_output=False,
    )
    assert result.isError is False


@pytest.mark.unit
async def test_bad_subprocess_artifact_is_non_fatal():
    """A subprocess artifact whose file is missing is logged and skipped."""
    art = {
        "path": Path("/nonexistent/missing.csv"),
        "artifact_type": "data",
        "title": "Missing",
        "description": "gone",
        "mime_type": "text/csv",
    }
    result = await _build(_ok_result(artifacts=[art]), save_output=False)
    assert result.isError is False


@pytest.mark.unit
async def test_notebook_creation_failure_is_non_fatal():
    """If notebook rendering fails, the response is still produced without it."""
    with patch(
        "osprey.stores.notebook_renderer.create_notebook_from_code",
        side_effect=RuntimeError("nbformat exploded"),
    ):
        result = await _build(_ok_result(), save_output=True)
    assert result.isError is False
    data = extract_response_dict(result)
    assert "notebook_artifact_id" not in data


@pytest.mark.unit
async def test_gallery_url_failure_is_swallowed(tmp_path):
    """A failing gallery_url lookup doesn't break the persisted response."""
    fig = tmp_path / "probe.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\nx")
    with patch(
        "osprey.mcp_server.http.gallery_url",
        side_effect=RuntimeError("no config"),
    ):
        # A saved artifact drives the gallery_url branch on the persisted path.
        result = await _build(_ok_result(figures=[fig]), save_output=True)
    assert result.isError is False
