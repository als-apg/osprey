"""Tests for the data_read MCP tool.

Covers:
  - Reading a valid entry returns full JSON content (raw data, no envelope)
  - Missing entry returns not_found error
  - Missing data file returns file_not_found error
  - Oversized file returns file_too_large error
"""

import json

import pytest

from osprey.mcp_server.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import get_tool_fn


def _save_entry(store, tool="channel_read", category="channel_values", data=None):
    """Helper to save a data entry with minimal boilerplate."""
    return store.save_data(
        tool=tool,
        data=data or {"value": 42, "units": "mA"},
        title="test entry",
        description="test entry",
        summary={"count": 1},
        access_details={"format": "json"},
        category=category,
    )


@pytest.fixture
def store(tmp_path):
    """Initialize an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def read_tool():
    """Get the raw async function for data_read."""
    from osprey.mcp_server.workspace.tools.data_context_tools import data_read

    return get_tool_fn(data_read)


class TestDataRead:
    """Tests for data_read."""

    @pytest.mark.asyncio
    async def test_read_valid_entry(self, store, read_tool):
        entry = _save_entry(store)

        result = json.loads(await read_tool(entry_id=entry.id))

        assert result["value"] == 42
        assert result["units"] == "mA"

    @pytest.mark.asyncio
    async def test_read_missing_entry(self, store, read_tool):
        result = json.loads(await read_tool(entry_id="nonexistent_id"))

        assert result["error"] is True
        assert result["error_type"] == "not_found"
        assert "nonexistent_id" in result["error_message"]

    @pytest.mark.asyncio
    async def test_read_missing_data_file(self, store, read_tool):
        entry = _save_entry(store)

        # Delete the data file from disk
        store.get_file_path(entry.id).unlink()

        result = json.loads(await read_tool(entry_id=entry.id))

        assert result["error"] is True
        assert result["error_type"] == "file_not_found"

    @pytest.mark.asyncio
    async def test_read_oversized_file(self, store, read_tool):
        entry = _save_entry(store)

        # Overwrite data file with >5 MB content
        store.get_file_path(entry.id).write_text("x" * (6 * 1024 * 1024))

        result = json.loads(await read_tool(entry_id=entry.id))

        assert result["error"] is True
        assert result["error_type"] == "file_too_large"

    @pytest.mark.asyncio
    async def test_read_returns_raw_json_string(self, store, read_tool):
        """data_read returns the file content as-is (raw JSON, no envelope)."""
        entry = _save_entry(store, data={"channels": ["SR:C01:CURRENT"]})

        raw = await read_tool(entry_id=entry.id)
        parsed = json.loads(raw)

        # The raw content is the data itself — no envelope
        assert "channels" in parsed

    @pytest.mark.asyncio
    async def test_read_multiple_entries(self, store, read_tool):
        """Each entry ID returns its own data."""
        e1 = _save_entry(store, data={"sensor": "A"})
        e2 = _save_entry(store, data={"sensor": "B"})

        r1 = json.loads(await read_tool(entry_id=e1.id))
        r2 = json.loads(await read_tool(entry_id=e2.id))

        assert r1["sensor"] == "A"
        assert r2["sensor"] == "B"
