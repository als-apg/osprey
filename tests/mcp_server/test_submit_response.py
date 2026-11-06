"""Tests for the submit_response MCP tool.

Covers:
  - Basic save to ArtifactStore with artifact_id returned
  - Markdown artifact file created on disk with correct content
  - entry_ids persisted in artifact metadata
  - Validation: empty title, empty content
  - no data_type / category: answers group by form and source_agent
  - skip_artifact mode
"""

import pytest

from osprey.mcp_server.workspace.tools.submit_response import submit_response
from osprey.stores.artifact_store import (
    get_artifact_store,
    initialize_artifact_store,
    reset_artifact_store,
)
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
)

# Extract the raw async function from the FastMCP FunctionTool wrapper
_fn = submit_response.fn if hasattr(submit_response, "fn") else submit_response


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Initialize a temporary ArtifactStore workspace.

    ``OSPREY_CONFIG`` is pinned into the same temp dir so the tool resolves its
    project there rather than wherever pytest happens to run.
    """
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    initialize_artifact_store(workspace_root=tmp_path)
    yield tmp_path
    reset_artifact_store()


class TestSubmitResponse:
    """Tests for the submit_response tool."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_basic(self):
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
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_with_entry_ids(self):
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
    @pytest.mark.usefixtures("workspace")
    @pytest.mark.parametrize(
        ("title", "content", "field"),
        [
            pytest.param("", "Some content.", "title", id="empty-title"),
            pytest.param("Valid Title", "", "content", id="empty-content"),
        ],
    )
    async def test_submit_response_empty_field_is_named(self, title, content, field):
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await _fn(title=title, content=content)
        data = _exc_ctx["envelope"]
        assert field in data["error_message"]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_an_entry_gone_before_its_metadata_is_reported_by_id(self, monkeypatch):
        from osprey.stores.artifact_store import ArtifactStore

        monkeypatch.setattr(ArtifactStore, "update_entry_metadata", lambda self, *a, **kw: None)
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await _fn(title="Orphan", content="Saved, then gone.")
        assert "left the index" in _exc_ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_whitespace_title(self):
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await _fn(title="   ", content="Some content.")
        _exc_ctx["envelope"]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_writes_no_data_type(self):
        raw = await _fn(
            title="BPM Channels",
            content="Found BPM channels.",
        )
        data = extract_response_dict(raw)

        assert data["status"] == "success"

        # Categories are retired: nothing records a data_type
        store = get_artifact_store()
        entry = store.get_entry(data["artifact_id"])
        assert entry is not None
        assert "data_type" not in entry.metadata

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_with_source_agent(self):
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
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_without_source_agent(self):
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
    @pytest.mark.usefixtures("workspace")
    async def test_artifact_file_contains_markdown_content(self):
        """The artifact file on disk should contain the raw markdown content."""
        raw = await _fn(
            title="Test Title",
            content="Test markdown content",
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
        assert "data_type" not in entry.metadata
        assert entry.metadata["entry_ids"] == ["e1"]

    # ---- Artifact auto-registration tests ----

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_submit_response_creates_artifact(self):
        """submit_response should automatically create a gallery artifact."""
        raw = await _fn(title="Beam Loss Analysis", content="Found 3 beam loss events.")
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert "artifact_id" in data, "artifact_id missing from response"
        assert "gallery_url" in data, "gallery_url missing from response"
        assert len(data["artifact_id"]) == 12  # hex UUID prefix

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_artifact_is_markdown_file(self):
        """The auto-created artifact should be a markdown file on disk."""
        raw = await _fn(
            title="BPM Calibration Logbook Review",
            content="## Summary\n\nFound 5 relevant entries.",
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
    @pytest.mark.usefixtures("workspace")
    async def test_artifact_metadata_has_source_agent_and_entry_ids(self):
        """Artifact metadata should carry source_agent and entry_ids."""
        raw = await _fn(
            title="BPM Channel Addresses",
            content="Found 12 BPM channels.",
            source_agent="channel-finder",
            entry_ids=["SR:BPM:01", "SR:BPM:02"],
        )
        data = extract_response_dict(raw)

        entry = get_artifact_store().get_entry(data["artifact_id"])
        assert entry is not None
        assert "data_type" not in entry.metadata
        assert entry.metadata["source_agent"] == "channel-finder"
        assert entry.metadata["entry_ids"] == ["SR:BPM:01", "SR:BPM:02"]
        # source_agent is also a top-level field on the entry
        assert entry.source_agent == "channel-finder"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_artifact_not_created_on_validation_error(self):
        """Validation errors should NOT create an artifact."""
        with assert_raises_error() as _exc_ctx:
            await _fn(title="", content="Some content.")
        _exc_ctx["envelope"]
        store = get_artifact_store()
        assert len(store.list_entries()) == 0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_a_hand_in_carries_no_category(self):
        """Categories are retired: an answer is a markdown document grouped by
        its form and its source_agent, never by a category."""
        raw = await _fn(
            title="AR Optics Summary",
            content="Tunes and beta functions computed from the design lattice.",
            source_agent="pyat-specialist",
        )
        data = extract_response_dict(raw)

        entry = get_artifact_store().get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.artifact_type == "markdown"
        assert entry.category == ""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_one_call_files_exactly_one_artifact(self):
        """Prose is the whole deliverable: one hand-in, one artifact.

        A computing agent used to owe a second JSON copy of its own numbers,
        re-typed by the model out of its own stdout. The values live in the
        answer; the executed code and its output are already saved as the
        notebook artifact of the run that produced them.
        """
        raw = await _fn(
            title="AR optics",
            content="| quantity | value |\n|---|---|\n| nu_x | 0.2345 |",
            source_agent="pyat-specialist",
        )
        data = extract_response_dict(raw)

        assert data["status"] == "success"
        assert "data_artifact_id" not in data

        entries = get_artifact_store().list_entries()
        assert len(entries) == 1
        assert entries[0].artifact_type == "markdown"
        assert entries[0].category == ""
        assert entries[0].source_agent == "pyat-specialist"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_description_names_the_answering_agent(self):
        """The description is operator-facing: the agent that answered."""
        raw = await _fn(
            title="BPM Channel Addresses",
            content="Found 12 BPM channels.",
            source_agent="channel-finder",
        )
        data = extract_response_dict(raw)

        entry = get_artifact_store().get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.description == "channel-finder"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_description_without_an_agent_is_generic(self):
        raw = await _fn(
            title="Ad-hoc note",
            content="Some prose.",
        )
        data = extract_response_dict(raw)

        entry = get_artifact_store().get_entry(data["artifact_id"])
        assert entry is not None
        assert entry.description == "Agent response"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("workspace")
    async def test_data_type_is_not_a_parameter(self):
        """The retired category parameter is refused, not silently dropped."""
        with pytest.raises(TypeError):
            await _fn(title="T", content="C", data_type="channel_addresses")
