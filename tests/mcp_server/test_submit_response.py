"""Tests for the submit_response MCP tool.

Covers:
  - Basic save to DataContext with context_entry_id returned
  - Automatic artifact creation in the gallery
  - entry_ids persisted in data payload
  - Validation: empty title, empty content
  - Custom data_type passed through
"""

import json

import pytest

from osprey.mcp_server.artifact_store import initialize_artifact_store, reset_artifact_store
from osprey.mcp_server.data_context import initialize_data_context
from osprey.mcp_server.workspace.tools.submit_response import submit_response

# Extract the raw async function from the FastMCP FunctionTool wrapper
_fn = submit_response.fn if hasattr(submit_response, "fn") else submit_response


@pytest.fixture
def workspace(tmp_path):
    """Initialize a temporary DataContext + ArtifactStore workspace."""
    initialize_data_context(workspace_root=tmp_path)
    initialize_artifact_store(workspace_root=tmp_path)
    yield tmp_path
    reset_artifact_store()


class TestSubmitResponse:
    """Tests for the submit_response tool."""

    @pytest.mark.asyncio
    async def test_submit_response_basic(self, workspace):
        raw = await _fn(title="Beam Loss Analysis", content="Found 3 beam loss events.")
        data = json.loads(raw)

        assert data["status"] == "success"
        assert "context_entry_id" in data
        assert data["context_entry_id"] == 1
        assert "data_file" in data
        assert data["summary"]["title"] == "Beam Loss Analysis"
        assert data["summary"]["content_length"] == len("Found 3 beam loss events.")
        assert data["summary"]["cited_entries"] == 0
        assert data["access_details"]["format"] == "markdown"
        assert data["access_details"]["data_type"] == "agent_response"

    @pytest.mark.asyncio
    async def test_submit_response_with_entry_ids(self, workspace):
        raw = await _fn(
            title="Vacuum Events",
            content="Analysis of vacuum events.",
            entry_ids=["e101", "e102", "e103"],
        )
        data = json.loads(raw)

        assert data["status"] == "success"
        assert data["summary"]["cited_entries"] == 3

        # Verify entry_ids are in the data file
        data_file = data["data_file"]
        with open(data_file) as f:
            payload = json.load(f)
        assert payload["data"]["entry_ids"] == ["e101", "e102", "e103"]

    @pytest.mark.asyncio
    async def test_submit_response_empty_title(self, workspace):
        raw = await _fn(title="", content="Some content.")
        data = json.loads(raw)

        assert data["error"] is True
        assert data["error_type"] == "validation_error"
        assert "title" in data["error_message"]

    @pytest.mark.asyncio
    async def test_submit_response_whitespace_title(self, workspace):
        raw = await _fn(title="   ", content="Some content.")
        data = json.loads(raw)

        assert data["error"] is True
        assert data["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_submit_response_empty_content(self, workspace):
        raw = await _fn(title="Valid Title", content="")
        data = json.loads(raw)

        assert data["error"] is True
        assert data["error_type"] == "validation_error"
        assert "content" in data["error_message"]

    @pytest.mark.asyncio
    async def test_submit_response_custom_data_type(self, workspace):
        raw = await _fn(
            title="BPM Channels",
            content="Found BPM channels.",
            data_type="channel_addresses",
        )
        data = json.loads(raw)

        assert data["status"] == "success"
        assert data["access_details"]["data_type"] == "channel_addresses"

        # Verify data_type in the data file
        data_file = data["data_file"]
        with open(data_file) as f:
            payload = json.load(f)
        assert payload["data"]["data_type"] == "channel_addresses"

    @pytest.mark.asyncio
    async def test_submit_response_with_source_agent(self, workspace):
        raw = await _fn(
            title="Beam Loss Analysis",
            content="Found 3 beam loss events.",
            source_agent="logbook-search",
        )
        data = json.loads(raw)

        assert data["status"] == "success"
        assert data["summary"]["source_agent"] == "logbook-search"

        # Verify source_agent in the data file and metadata
        data_file = data["data_file"]
        with open(data_file) as f:
            payload = json.load(f)
        assert payload["data"]["source_agent"] == "logbook-search"
        assert payload["_osprey_metadata"]["source_agent"] == "logbook-search"

    @pytest.mark.asyncio
    async def test_submit_response_without_source_agent(self, workspace):
        raw = await _fn(title="Basic Result", content="No agent specified.")
        data = json.loads(raw)

        assert data["status"] == "success"
        assert data["summary"]["source_agent"] == ""

        # Verify source_agent defaults to empty in data file
        data_file = data["data_file"]
        with open(data_file) as f:
            payload = json.load(f)
        assert payload["data"]["source_agent"] == ""
        # metadata should NOT have source_agent when empty
        assert "source_agent" not in payload["_osprey_metadata"]

    @pytest.mark.asyncio
    async def test_submit_response_data_file_content(self, workspace):
        raw = await _fn(
            title="Test Title",
            content="Test markdown content",
            data_type="logbook_research",
            entry_ids=["e1"],
        )
        data = json.loads(raw)

        data_file = data["data_file"]
        with open(data_file) as f:
            payload = json.load(f)

        assert payload["_osprey_metadata"]["tool"] == "submit_response"
        assert payload["data"]["title"] == "Test Title"
        assert payload["data"]["content"] == "Test markdown content"
        assert payload["data"]["data_type"] == "logbook_research"
        assert payload["data"]["entry_ids"] == ["e1"]

    # ---- Artifact auto-registration tests ----

    @pytest.mark.asyncio
    async def test_submit_response_creates_artifact(self, workspace):
        """submit_response should automatically create a gallery artifact."""
        raw = await _fn(title="Beam Loss Analysis", content="Found 3 beam loss events.")
        data = json.loads(raw)

        assert data["status"] == "success"
        assert "artifact_id" in data, "artifact_id missing from response"
        assert "gallery_url" in data, "gallery_url missing from response"
        assert len(data["artifact_id"]) == 12  # hex UUID prefix
        assert data["artifact_highlighted"] is True
        assert "already highlighted" in data["note"]

    @pytest.mark.asyncio
    async def test_artifact_is_markdown_file(self, workspace):
        """The auto-created artifact should be a markdown file on disk."""
        from osprey.mcp_server.artifact_store import get_artifact_store

        raw = await _fn(
            title="BPM Calibration Literature Review",
            content="## Summary\n\nFound 5 relevant papers.",
            data_type="literature_research",
            source_agent="literature-search",
        )
        data = json.loads(raw)
        artifact_id = data["artifact_id"]

        store = get_artifact_store()
        entry = store.get_entry(artifact_id)
        assert entry is not None
        assert entry.artifact_type == "markdown"
        assert entry.mime_type == "text/markdown"
        assert entry.title == "BPM Calibration Literature Review"
        assert entry.tool_source == "submit_response"

        # Verify file content matches
        file_path = store.get_file_path(artifact_id)
        assert file_path.exists()
        assert file_path.read_text() == "## Summary\n\nFound 5 relevant papers."

    @pytest.mark.asyncio
    async def test_artifact_metadata_links_to_data_context(self, workspace):
        """Artifact metadata should cross-reference the DataContext entry."""
        raw = await _fn(
            title="Orbit Correction Functions",
            content="Found 12 MML functions.",
            data_type="mml_research",
            source_agent="matlab-search",
            entry_ids=["getbpmresp", "setorbit"],
        )
        data = json.loads(raw)

        from osprey.mcp_server.artifact_store import get_artifact_store

        entry = get_artifact_store().get_entry(data["artifact_id"])
        assert entry.metadata["context_entry_id"] == data["context_entry_id"]
        assert entry.metadata["data_type"] == "mml_research"
        assert entry.metadata["source_agent"] == "matlab-search"
        assert entry.metadata["entry_ids"] == ["getbpmresp", "setorbit"]

    @pytest.mark.asyncio
    async def test_artifact_not_created_on_validation_error(self, workspace):
        """Validation errors should NOT create an artifact."""
        from osprey.mcp_server.artifact_store import get_artifact_store

        raw = await _fn(title="", content="Some content.")
        data = json.loads(raw)

        assert data["error"] is True
        store = get_artifact_store()
        assert len(store.list_entries()) == 0
