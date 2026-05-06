"""Tests for the artifact_focus MCP tool."""

import json

import pytest

from tests.mcp_server.conftest import (
    assert_error,
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)


def _get_artifact_focus():
    from osprey.mcp_server.workspace.tools.focus_tools import artifact_focus

    return get_tool_fn(artifact_focus)


def _get_artifact_save():
    from osprey.mcp_server.workspace.tools.artifact_save import artifact_save

    return get_tool_fn(artifact_save)


class TestArtifactFocusTool:
    """Tests for the artifact_focus MCP tool."""

    @pytest.mark.asyncio
    async def test_focus_unknown_artifact(self, tmp_path, monkeypatch):
        """Focusing an unknown artifact returns a not_found error."""
        monkeypatch.chdir(tmp_path)

        fn = _get_artifact_focus()
        with assert_raises_error(error_type="not_found") as _exc_ctx:
            await fn(artifact_id="nonexistent-id")

        data = _exc_ctx["envelope"]

    @pytest.mark.asyncio
    async def test_focus_valid_artifact(self, tmp_path, monkeypatch):
        """Focusing a valid artifact returns success (gallery POST is non-fatal)."""
        monkeypatch.chdir(tmp_path)

        # First save an artifact
        save_fn = _get_artifact_save()
        save_result = await save_fn(
            title="Test Artifact",
            content="# Hello",
            content_type="markdown",
        )
        save_data = extract_response_dict(save_result)
        artifact_id = save_data["artifact_id"]

        # Now focus on it
        fn = _get_artifact_focus()
        result = await fn(artifact_id=artifact_id)

        data = extract_response_dict(result)
        assert data["status"] == "success"
        assert data["artifact_id"] == artifact_id
        assert data["title"] == "Test Artifact"
        assert "gallery_url" in data
