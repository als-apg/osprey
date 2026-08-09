"""Tests for the data_list and data_delete MCP tools.

Covers:
  - Listing entries with and without filters, echoing the filters applied
  - ToolError passthrough vs internal_error wrapping on store failures
  - Deleting an entry (index + file), missing-entry not_found
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn


def _save_entry(store, tool="channel_read", category="channel_values"):
    """Helper to save a data entry with minimal boilerplate."""
    return store.save_data(
        tool=tool,
        data={"value": 42, "units": "mA"},
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
def list_tool():
    """Get the raw async function for data_list."""
    from osprey.mcp_server.workspace.tools.data_context_tools import data_list

    return get_tool_fn(data_list)


@pytest.fixture
def delete_tool():
    """Get the raw async function for data_delete."""
    from osprey.mcp_server.workspace.tools.data_context_tools import data_delete

    return get_tool_fn(data_delete)


@pytest.mark.asyncio
async def test_list_returns_entries_and_echoes_filters(store, list_tool):
    """No filters: every entry, null filters echoed. Filtered: only the match."""
    e1 = _save_entry(store)
    e2 = _save_entry(store, tool="archiver_read", category="archiver_data")

    unfiltered = json.loads(await list_tool())
    assert unfiltered["total_entries"] == 2
    assert {e["id"] for e in unfiltered["entries"]} == {e1.id, e2.id}
    assert unfiltered["filters_applied"] == {
        "tool": None,
        "category": None,
        "last_n": None,
        "source_agent": None,
    }

    filtered = json.loads(await list_tool(tool_filter="archiver_read", last_n=5))
    assert filtered["total_entries"] == 1
    assert filtered["entries"][0]["id"] == e2.id
    assert filtered["filters_applied"]["tool"] == "archiver_read"
    assert filtered["filters_applied"]["last_n"] == 5


@pytest.mark.asyncio
async def test_list_reraises_tool_error_unwrapped(store, list_tool, monkeypatch):
    """A ToolError from inside the store passes through, never re-wrapped as internal_error."""
    envelope = json.dumps(
        {"error": True, "error_type": "custom_error", "error_message": "x", "suggestions": []}
    )

    def _boom(**kwargs):
        raise ToolError(envelope)

    monkeypatch.setattr(store, "list_entries", _boom)

    with assert_raises_error(error_type="custom_error"):
        await list_tool()


@pytest.mark.asyncio
async def test_delete_removes_entry_and_file(store, delete_tool):
    entry = _save_entry(store)
    file_path = store.get_file_path(entry.id)

    result = json.loads(await delete_tool(entry_id=entry.id))

    assert result["status"] == "success"
    assert result["entry_id"] == entry.id
    assert store.get_entry(entry.id) is None
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_delete_missing_entry_is_not_found(store, delete_tool):
    with assert_raises_error(error_type="not_found") as ctx:
        await delete_tool(entry_id="nonexistent_id")
    assert "nonexistent_id" in ctx["envelope"]["error_message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_fixture", "store_method", "kwargs"),
    [
        ("list_tool", "list_entries", {}),
        ("delete_tool", "delete_entry", {"entry_id": "e1"}),
    ],
)
async def test_unexpected_store_failure_is_internal_error(
    store, request, monkeypatch, tool_fixture, store_method, kwargs
):
    """An unexpected store exception surfaces as internal_error, not a raw traceback."""

    def _boom(*args, **kw):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(store, store_method, _boom)

    with assert_raises_error(error_type="internal_error") as ctx:
        await request.getfixturevalue(tool_fixture)(**kwargs)
    assert "index unreadable" in ctx["envelope"]["error_message"]
