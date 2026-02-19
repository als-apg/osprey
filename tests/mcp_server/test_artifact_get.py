"""Tests for the artifact_get MCP tool.

Covers:
  - Getting a valid artifact returns metadata + file_path
  - Missing artifact returns not_found error
  - File path is correct and accessible
"""

import json

import pytest

from osprey.mcp_server.artifact_store import ArtifactStore, initialize_artifact_store
from tests.mcp_server.conftest import get_tool_fn


@pytest.fixture
def store(tmp_path):
    """Initialize an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def get_tool():
    """Get the raw async function for artifact_get."""
    from osprey.mcp_server.workspace.tools.artifact_save import artifact_get

    return get_tool_fn(artifact_get)


def _save_artifact(store, content=b"<h1>Test</h1>", filename="test.html", title="Test Artifact"):
    """Helper to save an artifact with minimal boilerplate."""
    return store.save_file(
        file_content=content,
        filename=filename,
        artifact_type="html",
        title=title,
        description="A test artifact",
        mime_type="text/html",
        tool_source="test",
    )


class TestArtifactGet:
    """Tests for artifact_get."""

    @pytest.mark.asyncio
    async def test_get_valid_artifact(self, store, get_tool):
        entry = _save_artifact(store)

        result = json.loads(await get_tool(artifact_id=entry.id))

        assert result["artifact_id"] == entry.id
        assert result["title"] == "Test Artifact"
        assert result["description"] == "A test artifact"
        assert result["artifact_type"] == "html"
        assert result["mime_type"] == "text/html"
        assert result["size_bytes"] == len(b"<h1>Test</h1>")
        assert result["file_path"] is not None
        assert "gallery_url" in result

    @pytest.mark.asyncio
    async def test_get_missing_artifact(self, store, get_tool):
        result = json.loads(await get_tool(artifact_id="nonexistent-id"))

        assert result["error"] is True
        assert result["error_type"] == "not_found"
        assert "nonexistent-id" in result["error_message"]

    @pytest.mark.asyncio
    async def test_file_path_is_accessible(self, store, get_tool):
        entry = _save_artifact(store, content=b"PNG data", filename="plot.png")

        result = json.loads(await get_tool(artifact_id=entry.id))

        from pathlib import Path

        file_path = Path(result["file_path"])
        assert file_path.exists()
        assert file_path.read_bytes() == b"PNG data"

    @pytest.mark.asyncio
    async def test_get_returns_timestamp(self, store, get_tool):
        entry = _save_artifact(store)

        result = json.loads(await get_tool(artifact_id=entry.id))

        assert "timestamp" in result
        assert result["timestamp"] == entry.timestamp

    @pytest.mark.asyncio
    async def test_get_multiple_artifacts(self, store, get_tool):
        """Each artifact ID returns its own metadata."""
        e1 = _save_artifact(store, title="Plot A", filename="a.html")
        e2 = _save_artifact(store, title="Plot B", filename="b.html")

        r1 = json.loads(await get_tool(artifact_id=e1.id))
        r2 = json.loads(await get_tool(artifact_id=e2.id))

        assert r1["title"] == "Plot A"
        assert r2["title"] == "Plot B"
        assert r1["artifact_id"] != r2["artifact_id"]
