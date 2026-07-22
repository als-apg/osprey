"""Tests for the shared python-executor response builder.

``build_execution_response`` turns an ``ExecutionResult`` into either a
``CallToolResult`` (success) or a raised ``ToolError`` carrying the OSPREY error
envelope (execution reported errors) — the fastmcp-safe error path. Tests run
against a real ArtifactStore in a tmp workspace and cover: the inline
(save_output=False) and persisted (save_output=True) branches, error handling in
each, notebook auto-save, and figure/artifact collection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.mcp_server.python_executor.executor import ExecutionResult
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
        execution_method_used="local",
        **kw,
    )


def _err_result(stderr="Traceback: boom") -> ExecutionResult:
    return ExecutionResult(
        success=False,
        stdout="partial\n",
        stderr=stderr,
        execution_method_used="local",
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
    assert data["stdout"] == "hello\n"
    assert data["has_errors"] is False


@pytest.mark.unit
async def test_inline_error_raises_execution_error():
    with pytest.raises(Exception) as exc_info:
        await _build(_err_result(stderr="ValueError: nope"), save_output=False)
    envelope = assert_error(str(exc_info.value), error_type="execution_error")
    assert "ValueError: nope" in envelope["error_message"]


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
