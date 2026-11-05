"""Tests for the artifact_list MCP tool.

Covers:
  - Listing artifacts with and without filters (form, format), echoing the filters applied
  - ToolError passthrough vs internal_error wrapping on store failures
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn


def _save_entry(store, tool="channel_read"):
    """Helper to save a data artifact with minimal boilerplate."""
    return store.save_data(
        tool=tool,
        data={"value": 42, "units": "mA"},
        title="test entry",
        description="test entry",
        summary={"count": 1},
        access_details={"format": "json"},
    )


@pytest.fixture
def store(tmp_path):
    """Initialize an ArtifactStore in a temporary workspace."""
    return initialize_artifact_store(workspace_root=tmp_path)


@pytest.fixture
def list_tool():
    """Get the raw async function for artifact_list."""
    from osprey.mcp_server.workspace.tools.artifact_query import artifact_list

    return get_tool_fn(artifact_list)


@pytest.mark.asyncio
async def test_list_returns_entries_and_echoes_filters(store, list_tool):
    """No filters: every artifact, null filters echoed. Filtered: only the match."""
    e1 = _save_entry(store)
    e2 = _save_entry(store, tool="archiver_read")

    unfiltered = json.loads(await list_tool())
    assert unfiltered["total_entries"] == 2
    assert {e["id"] for e in unfiltered["entries"]} == {e1.id, e2.id}
    assert unfiltered["filters_applied"] == {
        "tool": None,
        "form": None,
        "last_n": None,
        "source_agent": None,
        "artifact_type": None,
    }

    filtered = json.loads(await list_tool(tool="archiver_read", last_n=5))
    assert filtered["total_entries"] == 1
    assert filtered["entries"][0]["id"] == e2.id
    assert filtered["filters_applied"]["tool"] == "archiver_read"
    assert filtered["filters_applied"]["last_n"] == 5


def _save_doc(store, agent="pyat-specialist"):
    return store.save_file(
        file_content=b"# AR optics\n\nTunes computed from the design lattice.",
        filename=f"{agent}.md",
        artifact_type="markdown",
        title="AR optics",
        description=agent,
        mime_type="text/markdown",
        tool_source="submit_response",
    )


@pytest.mark.asyncio
async def test_list_narrows_to_a_form(store, list_tool):
    """``form`` is how a caller reaches one kind of artifact — not a category."""
    _save_entry(store)
    doc = _save_doc(store)

    result = json.loads(await list_tool(form="documents"))

    assert result["total_entries"] == 1
    assert result["entries"][0]["id"] == doc.id
    assert result["filters_applied"]["form"] == "documents"


@pytest.mark.asyncio
async def test_list_narrows_to_one_storage_format(store, list_tool):
    """``artifact_type`` separates the storage formats one agent produced.

    An agent's prose answer and the dataset it discusses share a source; a
    caller that intends to *load* the data asks for the format it can load.
    """
    dataset = _save_entry(store, tool="pyat-specialist")
    store.update_entry_metadata(dataset.id, source_agent="pyat-specialist")
    prose = _save_doc(store)
    store.update_entry_metadata(prose.id, source_agent="pyat-specialist")

    both = json.loads(await list_tool(source_agent="pyat-specialist"))
    assert both["total_entries"] == 2

    just_data = json.loads(await list_tool(source_agent="pyat-specialist", artifact_type="json"))
    assert just_data["total_entries"] == 1
    assert just_data["entries"][0]["id"] == dataset.id
    assert just_data["filters_applied"]["artifact_type"] == "json"


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
async def test_unexpected_store_failure_is_internal_error(store, list_tool, monkeypatch):
    """An unexpected store exception surfaces as internal_error, not a raw traceback."""

    def _boom(*args, **kw):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(store, "list_entries", _boom)

    with assert_raises_error(error_type="internal_error") as ctx:
        await list_tool()
    assert "index unreadable" in ctx["envelope"]["error_message"]
