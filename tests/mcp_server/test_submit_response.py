"""Tests for the submit_response MCP tool.

Covers:
  - Basic save to ArtifactStore with artifact_id returned
  - Markdown artifact file created on disk with correct content
  - entry_ids persisted in artifact metadata
  - Validation: empty title, empty content
  - Custom data_type passed through
  - source_agent / category mapping
  - skip_artifact mode
"""

import json

import pytest

from osprey.mcp_server.workspace.tools.submit_response import submit_response
from osprey.stores.artifact_store import (
    get_artifact_store,
    initialize_artifact_store,
    reset_artifact_store,
)
from tests.mcp_server.conftest import (
    assert_error,
    assert_raises_error,
    extract_response_dict,
)

# Extract the raw async function from the FastMCP FunctionTool wrapper
_fn = submit_response.fn if hasattr(submit_response, "fn") else submit_response


@pytest.fixture
def workspace(tmp_path):
    """Initialize a temporary ArtifactStore workspace."""
    initialize_artifact_store(workspace_root=tmp_path)
    yield tmp_path
    reset_artifact_store()


class TestSubmitResponse:
    """Tests for the submit_response tool."""

    @pytest.mark.asyncio
    async def test_submit_response_basic(self, workspace):
        raw = await _fn(title="Beam Loss Analysis", content="Found 3 beam loss events.")
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert "artifact_id" in data
        assert len(data["artifact_id"]) == 12  # hex UUID prefix
        assert data["title"] == "Beam Loss Analysis"
        assert data["summary"]["title"] == "Beam Loss Analysis"
        assert data["summary"]["content_length"] == len("Found 3 beam loss events.")
        assert data["summary"]["cited_entries"] == 0
        # No context_entry_id or data_file — artifact IS the content
        assert "context_entry_id" not in data

    @pytest.mark.asyncio
    async def test_submit_response_with_entry_ids(self, workspace):
        raw = await _fn(
            title="Vacuum Events",
            content="Analysis of vacuum events.",
            entry_ids=["e101", "e102", "e103"],
        )
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert data["summary"]["cited_entries"] == 3

        # Verify entry_ids are persisted in artifact metadata
        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.metadata["entry_ids"] == ["e101", "e102", "e103"]

    @pytest.mark.asyncio
    async def test_submit_response_empty_title(self, workspace):
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await _fn(title="", content="Some content.")
        data = _exc_ctx["envelope"]
        assert "title" in data["error_message"]

    @pytest.mark.asyncio
    async def test_submit_response_whitespace_title(self, workspace):
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await _fn(title="   ", content="Some content.")
        data = _exc_ctx["envelope"]

    @pytest.mark.asyncio
    async def test_submit_response_empty_content(self, workspace):
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await _fn(title="Valid Title", content="")
        data = _exc_ctx["envelope"]
        assert "content" in data["error_message"]

    @pytest.mark.asyncio
    async def test_submit_response_custom_data_type(self, workspace):
        raw = await _fn(
            title="BPM Channels",
            content="Found BPM channels.",
            data_type="channel_addresses",
        )
        data = extract_response_dict(raw)

        assert data["status"] == "success"

        # Verify data_type is persisted in artifact metadata
        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.metadata["data_type"] == "channel_addresses"

    @pytest.mark.asyncio
    async def test_submit_response_with_source_agent(self, workspace):
        raw = await _fn(
            title="Beam Loss Analysis",
            content="Found 3 beam loss events.",
            source_agent="logbook-search",
        )
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert data["summary"]["source_agent"] == "logbook-search"

        # Verify source_agent is persisted on the artifact entry and metadata
        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.source_agent == "logbook-search"
        assert entry.metadata["source_agent"] == "logbook-search"

    @pytest.mark.asyncio
    async def test_submit_response_without_source_agent(self, workspace):
        raw = await _fn(title="Basic Result", content="No agent specified.")
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert data["summary"]["source_agent"] == ""

        # Verify source_agent defaults to empty on the artifact entry
        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.source_agent == ""
        # metadata stores "" for source_agent when not provided
        assert entry.metadata["source_agent"] == ""

    @pytest.mark.asyncio
    async def test_artifact_file_contains_markdown_content(self, workspace):
        """The artifact file on disk should contain the raw markdown content."""
        raw = await _fn(
            title="Test Title",
            content="Test markdown content",
            data_type="logbook_research",
            entry_ids=["e1"],
        )
        data = extract_response_dict(raw)

        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.tool_source == "submit_response"
        assert entry.title == "Test Title"
        assert entry.artifact_type == "markdown"

        # The artifact file IS the content (no JSON envelope)
        file_path = store.get_file_path(data["artifact_id"])
        assert file_path.exists()
        assert file_path.read_text() == "Test markdown content"

        # Structured metadata is on the entry, not in the file
        assert entry.metadata["data_type"] == "logbook_research"
        assert entry.metadata["entry_ids"] == ["e1"]

    # ---- Artifact auto-registration tests ----

    @pytest.mark.asyncio
    async def test_submit_response_creates_artifact(self, workspace):
        """submit_response should automatically create a gallery artifact."""
        raw = await _fn(title="Beam Loss Analysis", content="Found 3 beam loss events.")
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert "artifact_id" in data, "artifact_id missing from response"
        assert "gallery_url" in data, "gallery_url missing from response"
        assert len(data["artifact_id"]) == 12  # hex UUID prefix

    @pytest.mark.asyncio
    async def test_artifact_is_markdown_file(self, workspace):
        """The auto-created artifact should be a markdown file on disk."""
        raw = await _fn(
            title="BPM Calibration Logbook Review",
            content="## Summary\n\nFound 5 relevant entries.",
            data_type="logbook_research",
            source_agent="logbook-search",
        )
        data = extract_response_dict(raw)
        artifact_id = data["artifact_id"]

        store = get_artifact_store()
        entry = store.get_entry(artifact_id)
        assert entry is not None
        assert entry.artifact_type == "markdown"
        assert entry.mime_type == "text/markdown"
        assert entry.title == "BPM Calibration Logbook Review"
        assert entry.tool_source == "submit_response"

        # Verify file content matches
        file_path = store.get_file_path(artifact_id)
        assert file_path.exists()
        assert file_path.read_text() == "## Summary\n\nFound 5 relevant entries."

    @pytest.mark.asyncio
    async def test_artifact_metadata_has_source_agent_and_entry_ids(self, workspace):
        """Artifact metadata should carry source_agent and entry_ids."""
        raw = await _fn(
            title="BPM Channel Addresses",
            content="Found 12 BPM channels.",
            data_type="channel_addresses",
            source_agent="channel-finder",
            entry_ids=["SR:BPM:01", "SR:BPM:02"],
        )
        data = extract_response_dict(raw)

        entry = get_artifact_store().get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.metadata["data_type"] == "channel_addresses"
        assert entry.metadata["source_agent"] == "channel-finder"
        assert entry.metadata["entry_ids"] == ["SR:BPM:01", "SR:BPM:02"]
        # source_agent is also a top-level field on the entry
        assert entry.source_agent == "channel-finder"

    @pytest.mark.asyncio
    async def test_artifact_not_created_on_validation_error(self, workspace):
        """Validation errors should NOT create an artifact."""
        with assert_raises_error() as _exc_ctx:
            await _fn(title="", content="Some content.")
        data = _exc_ctx["envelope"]
        store = get_artifact_store()
        assert len(store.list_entries()) == 0
