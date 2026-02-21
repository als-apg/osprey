"""Tests that submit_response groups results by agent name, not 'submit_response'.

When source_agent is provided, entries should use the agent name as the `tool` field
so that data_context_list groups them by agent rather than lumping everything under
"submit_response".
"""

import json

import pytest

from osprey.mcp_server.artifact_store import initialize_artifact_store, reset_artifact_store
from osprey.mcp_server.data_context import DataContext, initialize_data_context
from osprey.mcp_server.workspace.tools.submit_response import submit_response

_fn = submit_response.fn if hasattr(submit_response, "fn") else submit_response


@pytest.fixture
def workspace(tmp_path):
    initialize_data_context(workspace_root=tmp_path)
    initialize_artifact_store(workspace_root=tmp_path)
    yield tmp_path
    reset_artifact_store()


class TestSubmitResponseGrouping:
    """Verify that agent results are grouped by agent name, not 'submit_response'."""

    @pytest.mark.asyncio
    async def test_tool_field_uses_source_agent_when_provided(self, workspace):
        """When source_agent is given, entry.tool should be the agent name."""
        raw = await _fn(
            title="Beam Loss Analysis",
            content="Found 3 beam loss events.",
            source_agent="logbook-search",
        )
        data = json.loads(raw)
        assert data["status"] == "success"

        # Read the data file and check metadata
        with open(data["data_file"]) as f:
            payload = json.load(f)

        # The tool in metadata should be the agent name, NOT "submit_response"
        assert payload["_osprey_metadata"]["tool"] == "logbook-search"

    @pytest.mark.asyncio
    async def test_tool_field_stays_submit_response_without_agent(self, workspace):
        """When no source_agent, entry.tool should remain 'submit_response'."""
        raw = await _fn(
            title="Generic Result",
            content="Some result without an agent.",
        )
        data = json.loads(raw)
        assert data["status"] == "success"

        with open(data["data_file"]) as f:
            payload = json.load(f)

        assert payload["_osprey_metadata"]["tool"] == "submit_response"

    @pytest.mark.asyncio
    async def test_data_filename_uses_agent_name(self, workspace):
        """Data file should be named with agent, e.g. 001_logbook-search.json."""
        raw = await _fn(
            title="Wiki Result",
            content="Found wiki pages.",
            source_agent="wiki-search",
        )
        data = json.loads(raw)
        assert "wiki-search" in data["data_file"]
        assert "submit_response" not in data["data_file"]

    @pytest.mark.asyncio
    async def test_data_context_list_groups_by_agent(self, workspace):
        """Multiple agents should produce entries with different tool values."""
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

        ctx = DataContext(workspace_root=workspace)
        all_entries = ctx.list_entries()
        tools = {e.tool for e in all_entries}

        # Each agent should have its own tool name
        assert "logbook-search" in tools
        assert "wiki-search" in tools
        assert "channel-finder" in tools
        # None should be "submit_response"
        assert "submit_response" not in tools

    @pytest.mark.asyncio
    async def test_tool_filter_works_with_agent_name(self, workspace):
        """Should be able to filter by tool=agent_name."""
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

        ctx = DataContext(workspace_root=workspace)
        logbook_entries = ctx.list_entries(tool_filter="logbook-search")
        assert len(logbook_entries) == 1
        assert logbook_entries[0].description == "Logbook Result"

    @pytest.mark.asyncio
    async def test_source_agent_field_still_populated(self, workspace):
        """source_agent field should still be set independently of tool."""
        raw = await _fn(
            title="Test Result",
            content="Test content.",
            source_agent="logbook-search",
        )
        data = json.loads(raw)

        with open(data["data_file"]) as f:
            payload = json.load(f)

        # source_agent should still be in both data and metadata
        assert payload["data"]["source_agent"] == "logbook-search"
        assert payload["_osprey_metadata"]["source_agent"] == "logbook-search"
