"""Tests that submit_response groups results by source_agent in the ArtifactStore.

When source_agent is provided, artifact entries should carry the agent name in
the ``source_agent`` field so that ``list_entries(source_agent_filter=...)``
groups them by agent rather than lumping everything together.

The ``tool_source`` field is always ``"submit_response"`` (the actual MCP tool),
while ``source_agent`` carries the delegating agent name.
"""

import json

import pytest

from osprey.mcp_server.workspace.tools.submit_response import submit_response
from osprey.stores.artifact_store import (
    get_artifact_store,
    initialize_artifact_store,
    reset_artifact_store,
)

_fn = submit_response.fn if hasattr(submit_response, "fn") else submit_response


@pytest.fixture
def workspace(tmp_path):
    initialize_artifact_store(workspace_root=tmp_path)
    yield tmp_path
    reset_artifact_store()


class TestSubmitResponseGrouping:
    """Verify that agent results are grouped by source_agent in ArtifactStore."""

    @pytest.mark.asyncio
    async def test_source_agent_set_on_artifact_entry(self, workspace):
        """When source_agent is given, artifact entry.source_agent should be the agent name."""
        raw = await _fn(
            title="Beam Loss Analysis",
            content="Found 3 beam loss events.",
            source_agent="logbook-search",
        )
        data = json.loads(raw)
        assert data["status"] == "success"

        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        # source_agent carries the agent name for grouping
        assert entry.source_agent == "logbook-search"
        # tool_source is always "submit_response" (the actual MCP tool)
        assert entry.tool_source == "submit_response"

    @pytest.mark.asyncio
    async def test_source_agent_empty_without_agent(self, workspace):
        """When no source_agent, entry.source_agent should be empty."""
        raw = await _fn(
            title="Generic Result",
            content="Some result without an agent.",
        )
        data = json.loads(raw)
        assert data["status"] == "success"

        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.source_agent == ""
        assert entry.tool_source == "submit_response"

    @pytest.mark.asyncio
    async def test_artifact_filename_uses_agent_name(self, workspace):
        """Artifact file should be named with agent, e.g. {id}_wiki-search.md."""
        raw = await _fn(
            title="Wiki Result",
            content="Found wiki pages.",
            source_agent="wiki-search",
        )
        data = json.loads(raw)

        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert "wiki-search" in entry.filename
        assert "submit_response" not in entry.filename

    @pytest.mark.asyncio
    async def test_artifact_store_groups_by_source_agent(self, workspace):
        """Multiple agents should produce entries with different source_agent values."""
        await _fn(
            title="Logbook Result",
            content="Logbook findings.",
            source_agent="logbook-search",
            data_type="logbook_research",
        )
        await _fn(
            title="Wiki Result",
            content="Wiki findings.",
            source_agent="wiki-search",
            data_type="search_results",
        )
        await _fn(
            title="Channel Result",
            content="Channel addresses.",
            source_agent="channel-finder",
            data_type="channel_addresses",
        )

        store = get_artifact_store()
        all_entries = store.list_entries()
        agents = {e.source_agent for e in all_entries}

        # Each agent should have its own source_agent value
        assert "logbook-search" in agents
        assert "wiki-search" in agents
        assert "channel-finder" in agents

    @pytest.mark.asyncio
    async def test_source_agent_filter_works(self, workspace):
        """Should be able to filter by source_agent_filter=agent_name."""
        await _fn(
            title="Logbook Result",
            content="Logbook findings.",
            source_agent="logbook-search",
        )
        await _fn(
            title="Wiki Result",
            content="Wiki findings.",
            source_agent="wiki-search",
        )

        store = get_artifact_store()
        logbook_entries = store.list_entries(source_agent_filter="logbook-search")
        assert len(logbook_entries) == 1
        assert logbook_entries[0].title == "Logbook Result"

    @pytest.mark.asyncio
    async def test_source_agent_in_entry_and_metadata(self, workspace):
        """source_agent should be on both the entry field and in metadata."""
        raw = await _fn(
            title="Test Result",
            content="Test content.",
            source_agent="logbook-search",
        )
        data = json.loads(raw)

        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None

        # source_agent on the top-level entry field
        assert entry.source_agent == "logbook-search"
        # source_agent also in metadata dict
        assert entry.metadata["source_agent"] == "logbook-search"
