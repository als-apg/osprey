"""Tests for the session_summary MCP tool.

Validates that session_summary produces correct inventory data from
the unified ArtifactStore.
"""

import json

import pytest

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import get_tool_fn


def _get_session_summary():
    from osprey.mcp_server.workspace.tools.session_summary import session_summary

    return get_tool_fn(session_summary)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Set up a temporary workspace with an artifacts dir."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "artifacts").mkdir()
    monkeypatch.setenv("OSPREY_WORKSPACE", str(ws))
    monkeypatch.setattr(
        "osprey.mcp_server.workspace.tools.session_summary.resolve_workspace_root",
        lambda: ws,
    )
    return ws


@pytest.fixture
def art_store(workspace):
    """Initialize an artifact store in the test workspace."""
    return initialize_artifact_store(workspace)


class TestSessionSummaryEmpty:
    """Empty workspace returns zero counts."""

    @pytest.mark.asyncio
    async def test_empty_workspace(self, workspace, art_store):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        assert result["totals"]["entry_count"] == 0
        assert result["entries"] == []

    @pytest.mark.asyncio
    async def test_totals_keys_present(self, workspace, art_store):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        totals = result["totals"]

        assert "entry_count" in totals
        assert "total_bytes" in totals
        assert "categories" in totals
        assert "tools_used" in totals
        assert "artifact_types" in totals


class TestSessionSummaryWithData:
    """Workspace with data entries and gallery artifacts."""

    @pytest.fixture
    def populated_workspace(self, workspace, art_store):
        """Add some entries to the unified artifact store."""
        # Data entries (via save_data, replacing old DataContext.save)
        art_store.save_data(
            tool="archiver_read",
            data={"dataframe": {"columns": ["SR:CURRENT", "SR:VOLTAGE"], "index": [], "data": []}},
            title="Beam current and voltage",
            description="Beam current and voltage",
            summary={"channels": ["SR:CURRENT", "SR:VOLTAGE"], "points": 100},
            access_details={"format": "split"},
            category="archiver_data",
            artifact_type="json",
        )
        art_store.save_data(
            tool="channel_read",
            data={"values": [{"channel": "SR:TEMP", "value": 25.3}]},
            title="Temperature reading",
            description="Temperature reading",
            summary={"channel_count": 1},
            access_details={"format": "list"},
            category="channel_values",
            artifact_type="json",
        )

        # Gallery artifact (via save_file, unchanged)
        art_store.save_file(
            file_content=b"<html>plot</html>",
            filename="beam_current.html",
            artifact_type="plot_html",
            title="Beam Current Plot",
            description="Line chart of beam current",
            mime_type="text/html",
            tool_source="create_static_plot",
        )

        return workspace

    @pytest.mark.asyncio
    async def test_entry_count(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        assert result["totals"]["entry_count"] == 3

    @pytest.mark.asyncio
    async def test_data_entries_in_unified_list(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        entries = result["entries"]
        assert len(entries) == 3

        # Check first entry (archiver_read data)
        archiver = entries[0]
        assert archiver["tool"] == "archiver_read"
        assert archiver["category"] == "archiver_data"
        assert archiver["artifact_type"] == "json"
        assert archiver["description"] == "Beam current and voltage"
        assert "size_bytes" in archiver
        assert archiver["size_bytes"] > 0
        assert set(archiver["channels"]) == {"SR:CURRENT", "SR:VOLTAGE"}

    @pytest.mark.asyncio
    async def test_gallery_artifact_in_unified_list(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        entries = result["entries"]
        # The gallery artifact is third (save order: archiver, channel_read, plot)
        plot = entries[2]
        assert plot["artifact_type"] == "plot_html"
        assert plot["title"] == "Beam Current Plot"
        assert plot["size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_category_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        cat_counts = result["totals"]["categories"]
        assert cat_counts["archiver_data"] == 1
        assert cat_counts["channel_values"] == 1
        # Gallery artifact has no category, so it doesn't appear here

    @pytest.mark.asyncio
    async def test_tool_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        tool_counts = result["totals"]["tools_used"]
        assert tool_counts["archiver_read"] == 1
        assert tool_counts["channel_read"] == 1
        assert tool_counts["create_static_plot"] == 1

    @pytest.mark.asyncio
    async def test_artifact_type_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        art_counts = result["totals"]["artifact_types"]
        assert art_counts["json"] == 2
        assert art_counts["plot_html"] == 1

    @pytest.mark.asyncio
    async def test_total_bytes(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        assert result["totals"]["total_bytes"] > 0

    @pytest.mark.asyncio
    async def test_entry_structure(self, populated_workspace):
        """Every entry has the expected keys."""
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        expected_keys = {
            "id",
            "tool",
            "category",
            "artifact_type",
            "title",
            "description",
            "size_bytes",
            "channels",
            "timestamp",
        }
        for entry in result["entries"]:
            assert set(entry.keys()) == expected_keys


class TestSessionSummaryChannelExtraction:
    """Channel extraction from various entry shapes."""

    @pytest.mark.asyncio
    async def test_channels_from_columns_key(self, workspace, art_store):
        art_store.save_data(
            tool="archiver_read",
            data={},
            title="test columns",
            description="test",
            summary={},
            access_details={"columns": ["PV:A", "PV:B"]},
            category="archiver_data",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["entries"][0]["channels"] == ["PV:A", "PV:B"]

    @pytest.mark.asyncio
    async def test_channels_deduplication(self, workspace, art_store):
        """Channels appearing in both summary and access_details are deduplicated."""
        art_store.save_data(
            tool="archiver_read",
            data={},
            title="test dedup",
            description="test",
            summary={"channels": ["PV:A", "PV:B"]},
            access_details={"channels": ["PV:B", "PV:C"]},
            category="archiver_data",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["entries"][0]["channels"] == ["PV:A", "PV:B", "PV:C"]

    @pytest.mark.asyncio
    async def test_no_channels(self, workspace, art_store):
        """Entries without channel info return empty list."""
        art_store.save_data(
            tool="execute",
            data={},
            title="script output",
            description="script output",
            summary={"result": "ok"},
            access_details={},
            category="code_output",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["entries"][0]["channels"] == []

    @pytest.mark.asyncio
    async def test_channels_from_pvs_key(self, workspace, art_store):
        """Channel extraction recognises the 'pvs' key."""
        art_store.save_data(
            tool="channel_read",
            data={},
            title="test pvs",
            description="test",
            summary={"pvs": ["PV:X", "PV:Y"]},
            access_details={},
            category="channel_values",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["entries"][0]["channels"] == ["PV:X", "PV:Y"]

    @pytest.mark.asyncio
    async def test_channels_from_channel_names_key(self, workspace, art_store):
        """Channel extraction recognises the 'channel_names' key."""
        art_store.save_data(
            tool="archiver_read",
            data={},
            title="test channel_names",
            description="test",
            summary={},
            access_details={"channel_names": ["PV:M", "PV:N"]},
            category="archiver_data",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["entries"][0]["channels"] == ["PV:M", "PV:N"]

    @pytest.mark.asyncio
    async def test_gallery_artifact_no_channels(self, workspace, art_store):
        """Gallery artifacts (save_file) have no summary/access_details, so no channels."""
        art_store.save_file(
            file_content=b"<html>chart</html>",
            filename="chart.html",
            artifact_type="plot_html",
            title="A Chart",
            description="Some chart",
            mime_type="text/html",
            tool_source="create_static_plot",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["entries"][0]["channels"] == []
