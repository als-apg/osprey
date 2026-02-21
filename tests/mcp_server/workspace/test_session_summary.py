"""Tests for the session_summary MCP tool.

Validates that session_summary produces correct inventory data from
the data context and artifact stores.
"""

import json
from pathlib import Path

import pytest

from osprey.mcp_server.artifact_store import ArtifactStore, initialize_artifact_store
from osprey.mcp_server.data_context import DataContext, initialize_data_context
from tests.mcp_server.conftest import get_tool_fn


def _get_session_summary():
    from osprey.mcp_server.workspace.tools.session_summary import session_summary

    return get_tool_fn(session_summary)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Set up a temporary workspace with data and artifact dirs."""
    ws = tmp_path / "osprey-workspace"
    ws.mkdir()
    (ws / "data").mkdir()
    (ws / "artifacts").mkdir()
    monkeypatch.setenv("OSPREY_WORKSPACE", str(ws))
    monkeypatch.setattr(
        "osprey.mcp_server.workspace.tools.session_summary.resolve_workspace_root",
        lambda: ws,
    )
    return ws


@pytest.fixture
def data_ctx(workspace):
    """Initialize a data context in the test workspace."""
    return initialize_data_context(workspace)


@pytest.fixture
def art_store(workspace):
    """Initialize an artifact store in the test workspace."""
    return initialize_artifact_store(workspace)


class TestSessionSummaryEmpty:
    """Empty workspace returns zero counts."""

    @pytest.mark.asyncio
    async def test_empty_workspace(self, workspace, data_ctx, art_store):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        assert result["totals"]["data_entry_count"] == 0
        assert result["totals"]["artifact_count"] == 0
        assert result["data_entries"] == []
        assert result["artifacts"] == []

    @pytest.mark.asyncio
    async def test_totals_keys_present(self, workspace, data_ctx, art_store):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        totals = result["totals"]

        assert "data_entry_count" in totals
        assert "artifact_count" in totals
        assert "total_data_bytes" in totals
        assert "total_artifact_bytes" in totals
        assert "data_types" in totals
        assert "tools_used" in totals
        assert "artifact_types" in totals


class TestSessionSummaryWithData:
    """Workspace with data entries and artifacts."""

    @pytest.fixture
    def populated_workspace(self, workspace, data_ctx, art_store):
        """Add some entries to both stores."""
        data_ctx.save(
            tool="archiver_read",
            data={"dataframe": {"columns": ["SR:CURRENT", "SR:VOLTAGE"], "index": [], "data": []}},
            description="Beam current and voltage",
            summary={"channels": ["SR:CURRENT", "SR:VOLTAGE"], "points": 100},
            access_details={"format": "split"},
            data_type="timeseries",
        )
        data_ctx.save(
            tool="channel_read",
            data={"values": [{"channel": "SR:TEMP", "value": 25.3}]},
            description="Temperature reading",
            summary={"channel_count": 1},
            access_details={"format": "list"},
            data_type="channel_values",
        )

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
    async def test_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        assert result["totals"]["data_entry_count"] == 2
        assert result["totals"]["artifact_count"] == 1

    @pytest.mark.asyncio
    async def test_data_entries_structure(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        de = result["data_entries"]
        assert len(de) == 2
        # Check first entry (archiver_read)
        archiver = de[0]
        assert archiver["tool"] == "archiver_read"
        assert archiver["data_type"] == "timeseries"
        assert archiver["description"] == "Beam current and voltage"
        assert "size_bytes" in archiver
        assert archiver["size_bytes"] > 0
        assert set(archiver["channels"]) == {"SR:CURRENT", "SR:VOLTAGE"}

    @pytest.mark.asyncio
    async def test_artifacts_structure(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        arts = result["artifacts"]
        assert len(arts) == 1
        art = arts[0]
        assert art["type"] == "plot_html"
        assert art["title"] == "Beam Current Plot"
        assert art["size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_data_type_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        dt_counts = result["totals"]["data_types"]
        assert dt_counts["timeseries"] == 1
        assert dt_counts["channel_values"] == 1

    @pytest.mark.asyncio
    async def test_tool_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        tool_counts = result["totals"]["tools_used"]
        assert tool_counts["archiver_read"] == 1
        assert tool_counts["channel_read"] == 1

    @pytest.mark.asyncio
    async def test_artifact_type_counts(self, populated_workspace):
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)

        art_counts = result["totals"]["artifact_types"]
        assert art_counts["plot_html"] == 1


class TestSessionSummaryChannelExtraction:
    """Channel extraction from various entry shapes."""

    @pytest.mark.asyncio
    async def test_channels_from_columns_key(self, workspace, data_ctx, art_store):
        data_ctx.save(
            tool="archiver_read",
            data={},
            description="test",
            summary={},
            access_details={"columns": ["PV:A", "PV:B"]},
            data_type="timeseries",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["data_entries"][0]["channels"] == ["PV:A", "PV:B"]

    @pytest.mark.asyncio
    async def test_channels_deduplication(self, workspace, data_ctx, art_store):
        """Channels appearing in both summary and access_details are deduplicated."""
        data_ctx.save(
            tool="archiver_read",
            data={},
            description="test",
            summary={"channels": ["PV:A", "PV:B"]},
            access_details={"channels": ["PV:B", "PV:C"]},
            data_type="timeseries",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["data_entries"][0]["channels"] == ["PV:A", "PV:B", "PV:C"]

    @pytest.mark.asyncio
    async def test_no_channels(self, workspace, data_ctx, art_store):
        """Entries without channel info return empty list."""
        data_ctx.save(
            tool="execute",
            data={},
            description="script output",
            summary={"result": "ok"},
            access_details={},
            data_type="code_output",
        )
        fn = _get_session_summary()
        raw = await fn()
        result = json.loads(raw)
        assert result["data_entries"][0]["channels"] == []
