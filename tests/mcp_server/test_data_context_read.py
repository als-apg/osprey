"""Tests for the data_context_read MCP tool.

Covers:
  - Reading a valid entry returns full JSON content
  - Missing entry returns not_found error
  - Missing data file returns file_not_found error
  - Oversized file returns file_too_large error
"""

import json
from pathlib import Path

import pytest

from osprey.mcp_server.data_context import DataContext, initialize_data_context
from tests.mcp_server.conftest import get_tool_fn


def _save_entry(ctx, tool="channel_read", data_type="channel_values", data=None):
    """Helper to save a context entry with minimal boilerplate."""
    return ctx.save(
        tool=tool,
        data=data or {"value": 42, "units": "mA"},
        description="test entry",
        summary={"count": 1},
        access_details={"format": "json"},
        data_type=data_type,
    )


@pytest.fixture
def ctx(tmp_path):
    """Initialize a DataContext in a temporary workspace."""
    return initialize_data_context(workspace_root=tmp_path)


@pytest.fixture
def read_tool():
    """Get the raw async function for data_context_read."""
    from osprey.mcp_server.workspace.tools.data_context_tools import data_context_read

    return get_tool_fn(data_context_read)


class TestDataContextRead:
    """Tests for data_context_read."""

    @pytest.mark.asyncio
    async def test_read_valid_entry(self, ctx, read_tool):
        entry = _save_entry(ctx)

        result = json.loads(await read_tool(entry_id=entry.id))

        assert "data" in result
        assert result["data"]["value"] == 42
        assert result["data"]["units"] == "mA"
        assert result["_osprey_metadata"]["context_entry_id"] == entry.id

    @pytest.mark.asyncio
    async def test_read_missing_entry(self, ctx, read_tool):
        result = json.loads(await read_tool(entry_id=999))

        assert result["error"] is True
        assert result["error_type"] == "not_found"
        assert "999" in result["error_message"]

    @pytest.mark.asyncio
    async def test_read_missing_data_file(self, ctx, read_tool):
        entry = _save_entry(ctx)

        # Delete the data file from disk
        Path(entry.data_file).unlink()

        result = json.loads(await read_tool(entry_id=entry.id))

        assert result["error"] is True
        assert result["error_type"] == "file_not_found"

    @pytest.mark.asyncio
    async def test_read_oversized_file(self, ctx, read_tool):
        entry = _save_entry(ctx)

        # Overwrite data file with >5 MB content
        data_path = Path(entry.data_file)
        data_path.write_text("x" * (6 * 1024 * 1024))

        result = json.loads(await read_tool(entry_id=entry.id))

        assert result["error"] is True
        assert result["error_type"] == "file_too_large"

    @pytest.mark.asyncio
    async def test_read_returns_raw_json_string(self, ctx, read_tool):
        """data_context_read returns the file content as-is, not wrapped."""
        entry = _save_entry(ctx, data={"channels": ["SR:C01:CURRENT"]})

        raw = await read_tool(entry_id=entry.id)
        parsed = json.loads(raw)

        # The raw content is the data file — has _osprey_metadata and data keys
        assert "channels" in parsed["data"]

    @pytest.mark.asyncio
    async def test_read_multiple_entries(self, ctx, read_tool):
        """Each entry ID returns its own data."""
        e1 = _save_entry(ctx, data={"sensor": "A"})
        e2 = _save_entry(ctx, data={"sensor": "B"})

        r1 = json.loads(await read_tool(entry_id=e1.id))
        r2 = json.loads(await read_tool(entry_id=e2.id))

        assert r1["data"]["sensor"] == "A"
        assert r2["data"]["sensor"] == "B"
