"""Tests for the archiver_downsample MCP tool.

Validates LTTB downsampling of archiver_data artifact entries,
channel filtering, error handling for wrong entry categories, and
empty data edge cases.
"""

import json

import pytest

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import get_tool_fn


def _get_archiver_downsample():
    from osprey.mcp_server.workspace.tools.archiver_downsample import archiver_downsample

    return get_tool_fn(archiver_downsample)


def _make_timeseries_data(n_points: int, n_channels: int = 2):
    """Generate synthetic timeseries data in archiver format."""
    columns = [f"PV:CH{i}" for i in range(n_channels)]
    index = [f"2026-02-19T12:{i:02d}:00Z" for i in range(n_points)]
    data = [[float(ch * 10 + pt) for ch in range(n_channels)] for pt in range(n_points)]
    return {
        "dataframe": {"columns": columns, "index": index, "data": data},
        "query": {"start": index[0] if index else None, "end": index[-1] if index else None},
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "artifacts").mkdir()
    monkeypatch.setattr(
        "osprey.mcp_server.workspace.tools.archiver_downsample.resolve_workspace_root",
        lambda: ws,
    )
    return ws


@pytest.fixture
def art_store(workspace):
    return initialize_artifact_store(workspace)


class TestArchiverDownsampleBasic:
    """Basic downsampling of archiver_data entries."""

    @pytest.fixture
    def timeseries_entry(self, art_store):
        """Create an archiver_data entry with 50 points, 2 channels."""
        ts_data = _make_timeseries_data(50, 2)
        entry = art_store.save_data(
            tool="archiver_read",
            data=ts_data,
            title="Beam current and voltage",
            description="Beam current and voltage",
            summary={"channels": ["PV:CH0", "PV:CH1"], "points": 50},
            access_details={"format": "split"},
            category="archiver_data",
        )
        return entry

    @pytest.mark.asyncio
    async def test_downsample_returns_labels_and_datasets(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id)
        result = json.loads(raw)

        assert "labels" in result
        assert "datasets" in result
        assert "original_points" in result
        assert "downsampled_points" in result
        assert "time_range" in result

    @pytest.mark.asyncio
    async def test_downsample_reduces_points(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id, max_points=10)
        result = json.loads(raw)

        assert result["original_points"] == 50
        assert result["downsampled_points"] <= 10
        assert len(result["labels"]) == result["downsampled_points"]

    @pytest.mark.asyncio
    async def test_all_channels_included_by_default(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id)
        result = json.loads(raw)

        channel_names = [ds["channel"] for ds in result["datasets"]]
        assert "PV:CH0" in channel_names
        assert "PV:CH1" in channel_names

    @pytest.mark.asyncio
    async def test_time_range(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id)
        result = json.loads(raw)

        assert result["time_range"]["start"] is not None
        assert result["time_range"]["end"] is not None

    @pytest.mark.asyncio
    async def test_no_downsample_when_under_max(self, timeseries_entry):
        """When max_points >= data length, all points are returned."""
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id, max_points=1000)
        result = json.loads(raw)

        assert result["original_points"] == 50
        assert result["downsampled_points"] == 50


class TestArchiverDownsampleChannelFilter:
    """Filtering by specific channels."""

    @pytest.fixture
    def multi_channel_entry(self, art_store):
        ts_data = _make_timeseries_data(30, 4)
        entry = art_store.save_data(
            tool="archiver_read",
            data=ts_data,
            title="Four channels",
            description="Four channels",
            summary={"channels": ["PV:CH0", "PV:CH1", "PV:CH2", "PV:CH3"]},
            access_details={},
            category="archiver_data",
        )
        return entry

    @pytest.mark.asyncio
    async def test_filter_single_channel(self, multi_channel_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=multi_channel_entry.id, channels=["PV:CH2"])
        result = json.loads(raw)

        assert len(result["datasets"]) == 1
        assert result["datasets"][0]["channel"] == "PV:CH2"

    @pytest.mark.asyncio
    async def test_filter_multiple_channels(self, multi_channel_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=multi_channel_entry.id, channels=["PV:CH0", "PV:CH3"])
        result = json.loads(raw)

        channel_names = [ds["channel"] for ds in result["datasets"]]
        assert channel_names == ["PV:CH0", "PV:CH3"]

    @pytest.mark.asyncio
    async def test_filter_nonexistent_channel_errors(self, multi_channel_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=multi_channel_entry.id, channels=["NONEXISTENT"])
        result = json.loads(raw)

        assert result.get("error") is True
        assert "validation_error" in result.get("error_type", "")


class TestArchiverDownsampleErrors:
    """Error cases: wrong category, missing entry, etc."""

    @pytest.fixture
    def non_archiver_entry(self, art_store):
        return art_store.save_data(
            tool="channel_read",
            data={"values": [{"channel": "SR:TEMP", "value": 25.3}]},
            title="Temperature",
            description="Temperature",
            summary={},
            access_details={},
            category="channel_values",
        )

    @pytest.mark.asyncio
    async def test_wrong_category(self, workspace, non_archiver_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=non_archiver_entry.id)
        result = json.loads(raw)

        assert result.get("error") is True
        assert "archiver_data" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_nonexistent_entry(self, workspace, art_store):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id="deadbeef0000")
        result = json.loads(raw)

        assert result.get("error") is True
        assert "not found" in result.get("error_message", "").lower()


class TestArchiverDownsampleEmptyData:
    """Edge case: archiver_data entry with zero rows."""

    @pytest.fixture
    def empty_entry(self, art_store):
        ts_data = _make_timeseries_data(0)
        return art_store.save_data(
            tool="archiver_read",
            data=ts_data,
            title="Empty",
            description="Empty",
            summary={},
            access_details={},
            category="archiver_data",
        )

    @pytest.mark.asyncio
    async def test_empty_timeseries(self, workspace, empty_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=empty_entry.id)
        result = json.loads(raw)

        assert result["original_points"] == 0
        assert result["downsampled_points"] == 0
        assert result["labels"] == []
        assert result["datasets"] == []


class TestArchiverDownsampleFlatFormat:
    """Timeseries stored in flat format (no 'dataframe' wrapper)."""

    @pytest.fixture
    def flat_entry(self, art_store):
        """Flat format without 'dataframe' wrapper.

        The raw JSON on disk must still survive ``_extract_timeseries_frame``
        which does ``raw.get("data", raw)``.  We wrap the split-orient frame
        inside ``{"data": {...}}`` so the extractor finds a dict, not a list.
        """
        flat_data = {
            "data": {
                "columns": ["PV:FLAT"],
                "index": [f"2026-02-19T12:{i:02d}:00Z" for i in range(20)],
                "data": [[float(i)] for i in range(20)],
            }
        }
        return art_store.save_data(
            tool="archiver_read",
            data=flat_data,
            title="Flat format",
            description="Flat format",
            summary={},
            access_details={},
            category="archiver_data",
        )

    @pytest.mark.asyncio
    async def test_flat_format_works(self, workspace, flat_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=flat_entry.id, max_points=10)
        result = json.loads(raw)

        assert result["original_points"] == 20
        assert len(result["datasets"]) == 1
        assert result["datasets"][0]["channel"] == "PV:FLAT"
