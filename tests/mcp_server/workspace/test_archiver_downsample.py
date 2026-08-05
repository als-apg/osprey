"""Tests for the archiver_downsample MCP tool.

Validates LTTB downsampling of archiver_data artifact entries, channel
filtering, error handling for wrong entry categories, and empty data edge
cases.

Each channel now carries its own timestamps (independent sample cadences,
not aligned to a shared axis), so ``datasets`` entries are
``{"channel", "timestamps", "values", "original_points", "downsampled_points"}``
rather than a shared-``labels`` Chart.js config.
"""

import json

import pytest

from osprey.stores.artifact_store import initialize_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn


def _get_archiver_downsample():
    from osprey.mcp_server.workspace.tools.archiver_downsample import archiver_downsample

    return get_tool_fn(archiver_downsample)


def _make_timeseries_data(n_points: int, n_channels: int = 2):
    """Generate synthetic timeseries data in the legacy archiver split-orient format."""
    columns = [f"PV:CH{i}" for i in range(n_channels)]
    index = [f"2026-02-19T12:{i:02d}:00Z" for i in range(n_points)]
    data = [[float(ch * 10 + pt) for ch in range(n_channels)] for pt in range(n_points)]
    return {
        "dataframe": {"columns": columns, "index": index, "data": data},
        "query": {"start": index[0] if index else None, "end": index[-1] if index else None},
    }


def _make_new_format_series_data(channels_data: dict[str, tuple[list, list]]):
    """Generate synthetic timeseries data in the new per-channel 'series' format.

    ``channels_data`` maps channel name -> (timestamps, values).
    """
    return {
        "query": {"channels": list(channels_data.keys())},
        "series": {
            ch: {"timestamps": list(ts), "values": list(vals)}
            for ch, (ts, vals) in channels_data.items()
        },
    }


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "artifacts").mkdir()
    return ws


@pytest.fixture
def art_store(workspace):
    """Initialize the ArtifactStore singleton the tool reads via get_artifact_store()."""
    return initialize_artifact_store(workspace_root=workspace)


class TestArchiverDownsampleBasic:
    """Basic downsampling of archiver_data entries (legacy split-orient layout)."""

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
    async def test_downsample_returns_datasets(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id)
        result = json.loads(raw)

        assert "datasets" in result
        assert "original_points" in result
        assert "downsampled_points" in result
        assert "time_range" in result
        assert "labels" not in result  # no more shared-axis Chart.js config

    @pytest.mark.asyncio
    async def test_dataset_carries_its_own_timestamps(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id)
        result = json.loads(raw)

        for ds in result["datasets"]:
            assert "channel" in ds
            assert "timestamps" in ds
            assert "values" in ds
            assert len(ds["timestamps"]) == len(ds["values"])

    @pytest.mark.asyncio
    async def test_downsample_reduces_points(self, timeseries_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=timeseries_entry.id, max_points=10)
        result = json.loads(raw)

        # 2 channels x 50 points each -- summed across channels.
        assert result["original_points"] == 100
        assert result["downsampled_points"] == sum(
            ds["downsampled_points"] for ds in result["datasets"]
        )
        for ds in result["datasets"]:
            assert ds["original_points"] == 50
            assert ds["downsampled_points"] <= 10
            assert len(ds["timestamps"]) == ds["downsampled_points"]

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

        assert result["original_points"] == 100
        assert result["downsampled_points"] == 100


class TestArchiverDownsampleAggregates:
    """The top-level roll-ups: ``time_range`` and the two point totals.

    ``time_range`` is a min/max across every returned channel, not the first
    channel's span.
    """

    @pytest.fixture
    def staggered_entry(self, art_store):
        """Three channels whose spans are deliberately out of declaration order."""
        data = _make_new_format_series_data(
            {
                "PV:LATE": ([f"2026-02-19T12:{i:02d}:00Z" for i in range(10)], list(range(10))),
                "PV:EARLY": ([f"2026-02-19T11:{i:02d}:00Z" for i in range(5)], list(range(5))),
                "PV:MID": ([f"2026-02-19T12:{i:02d}:00Z" for i in range(30, 41)], list(range(11))),
            }
        )
        return art_store.save_data(
            tool="archiver_read",
            data=data,
            title="Staggered channels",
            description="Staggered channels",
            summary={},
            access_details={},
            category="archiver_data",
        )

    @pytest.mark.asyncio
    async def test_time_range_spans_every_channel_not_just_the_first(self, staggered_entry):
        fn = _get_archiver_downsample()
        result = json.loads(await fn(entry_id=staggered_entry.id))

        # Earliest sample belongs to the SECOND-declared channel, latest to the third.
        assert result["time_range"]["start"] == "2026-02-19T11:00:00Z"
        assert result["time_range"]["end"] == "2026-02-19T12:40:00Z"

    @pytest.mark.asyncio
    async def test_totals_are_summed_over_every_channel(self, staggered_entry):
        fn = _get_archiver_downsample()
        result = json.loads(await fn(entry_id=staggered_entry.id))

        assert result["original_points"] == 26  # 10 + 5 + 11
        assert result["downsampled_points"] == 26  # all under max_points
        assert result["original_points"] == sum(d["original_points"] for d in result["datasets"])
        assert result["downsampled_points"] == sum(
            d["downsampled_points"] for d in result["datasets"]
        )

    @pytest.mark.asyncio
    async def test_channel_with_no_samples_does_not_erase_the_span(self, art_store):
        """A requested-but-dead channel must not drag the payload-level span to ``None``."""
        data = _make_new_format_series_data(
            {
                "PV:DOWN": ([], []),
                "PV:UP": (["2026-02-19T12:00:00Z", "2026-02-19T12:05:00Z"], [1.0, 2.0]),
            }
        )
        entry = art_store.save_data(
            tool="archiver_read",
            data=data,
            title="One dead channel",
            description="One dead channel",
            summary={},
            access_details={},
            category="archiver_data",
        )

        fn = _get_archiver_downsample()
        result = json.loads(await fn(entry_id=entry.id))

        assert result["time_range"] == {
            "start": "2026-02-19T12:00:00Z",
            "end": "2026-02-19T12:05:00Z",
        }
        assert result["original_points"] == 2
        by_channel = {d["channel"]: d for d in result["datasets"]}
        assert by_channel["PV:DOWN"]["timestamps"] == []
        assert by_channel["PV:DOWN"]["original_points"] == 0

    @pytest.mark.asyncio
    async def test_all_none_channel_still_contributes_its_span(self, art_store):
        """Gap values are not missing samples: the timestamps are real, so they
        count toward the totals and the span.
        """
        stamps = [f"2026-02-19T09:{i:02d}:00Z" for i in range(40)]
        data = _make_new_format_series_data({"PV:DEAD": (stamps, [None] * 40)})
        entry = art_store.save_data(
            tool="archiver_read",
            data=data,
            title="All gaps",
            description="All gaps",
            summary={},
            access_details={},
            category="archiver_data",
        )

        fn = _get_archiver_downsample()
        result = json.loads(await fn(entry_id=entry.id, max_points=10))

        assert result["original_points"] == 40
        assert result["downsampled_points"] == 10
        assert result["time_range"] == {
            "start": "2026-02-19T09:00:00Z",
            "end": "2026-02-19T09:39:00Z",
        }
        # The gaps survive as gaps -- never a manufactured 0.0.
        assert result["datasets"][0]["values"] == [None] * 10

    @pytest.mark.asyncio
    async def test_no_datasets_selected_yields_null_span(self, art_store):
        """Zero channels at all: the span is ``None``/``None``, not an error."""
        data = _make_new_format_series_data({})
        entry = art_store.save_data(
            tool="archiver_read",
            data=data,
            title="No channels",
            description="No channels",
            summary={},
            access_details={},
            category="archiver_data",
        )

        fn = _get_archiver_downsample()
        result = json.loads(await fn(entry_id=entry.id))

        assert result["datasets"] == []
        assert result["original_points"] == 0
        assert result["downsampled_points"] == 0
        assert result["time_range"] == {"start": None, "end": None}


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
    async def test_filter_preserves_the_callers_requested_order(self, multi_channel_entry):
        """``datasets`` order follows the caller's ``channels`` argument, not
        the artifact's column order.
        """
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=multi_channel_entry.id, channels=["PV:CH3", "PV:CH0", "PV:CH1"])
        result = json.loads(raw)

        channel_names = [ds["channel"] for ds in result["datasets"]]
        assert channel_names == ["PV:CH3", "PV:CH0", "PV:CH1"]

    @pytest.mark.asyncio
    async def test_filter_nonexistent_channel_errors(self, multi_channel_entry):
        fn = _get_archiver_downsample()
        with assert_raises_error(error_type="validation_error") as _exc_ctx:
            await fn(entry_id=multi_channel_entry.id, channels=["NONEXISTENT"])
        result = _exc_ctx["envelope"]
        assert result["error"] is True


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
        with assert_raises_error() as _exc_ctx:
            await fn(entry_id=non_archiver_entry.id)
        result = _exc_ctx["envelope"]
        assert "archiver_data" in result["error_message"].lower()

    @pytest.mark.asyncio
    async def test_nonexistent_entry(self, workspace, art_store):
        fn = _get_archiver_downsample()
        with assert_raises_error() as _exc_ctx:
            await fn(entry_id="deadbeef0000")
        result = _exc_ctx["envelope"]
        assert "not found" in result["error_message"].lower()


class TestArchiverDownsampleEmptyData:
    """Edge case: archiver_data entry with zero rows.

    Every queried channel is still declared, just with empty arrays.
    """

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
        assert len(result["datasets"]) == 2  # PV:CH0, PV:CH1 -- both declared, both empty
        for ds in result["datasets"]:
            assert ds["timestamps"] == []
            assert ds["values"] == []
        assert result["time_range"] == {"start": None, "end": None}


class TestArchiverDownsampleFlatFormat:
    """Timeseries stored in flat format (no 'dataframe' wrapper)."""

    @pytest.fixture
    def flat_entry(self, art_store):
        """Flat format without 'dataframe' wrapper.

        The raw JSON on disk must still survive ``extract_channel_series``,
        which does ``raw.get("data", raw)``. We wrap the split-orient frame
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
        assert (
            len(result["datasets"][0]["timestamps"]) == result["datasets"][0]["downsampled_points"]
        )


class TestArchiverDownsampleNewSeriesFormat:
    """The new long-format archiver_read payload: {"query", "series"}."""

    @pytest.fixture
    def new_format_entry(self, art_store):
        n = 100
        ts_data = _make_new_format_series_data(
            {
                "PV:A": (
                    [f"2026-02-19T12:{i:02d}:00Z" for i in range(n)],
                    [float(i) for i in range(n)],
                ),
                # A different (sparser) cadence -- independent from PV:A.
                "PV:B": (
                    [f"2026-02-19T12:{i:02d}:00Z" for i in range(0, n, 4)],
                    [float(i) * 2 for i in range(0, n, 4)],
                ),
            }
        )
        return art_store.save_data(
            tool="archiver_read",
            data=ts_data,
            title="New format",
            description="New format",
            summary={},
            access_details={},
            category="archiver_data",
        )

    @pytest.mark.asyncio
    async def test_new_format_is_not_rendered_empty(self, new_format_entry):
        """Reproduces + verifies the fix for the reported empty-render bug."""
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=new_format_entry.id, max_points=1000)
        result = json.loads(raw)

        assert result["original_points"] == 125  # 100 + 25
        assert len(result["datasets"]) == 2
        by_channel = {ds["channel"]: ds for ds in result["datasets"]}
        assert len(by_channel["PV:A"]["timestamps"]) == 100
        assert len(by_channel["PV:B"]["timestamps"]) == 25

    @pytest.mark.asyncio
    async def test_channels_downsampled_independently(self, new_format_entry):
        """Each channel is downsampled against its OWN point count, not a shared one."""
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=new_format_entry.id, max_points=10)
        result = json.loads(raw)

        by_channel = {ds["channel"]: ds for ds in result["datasets"]}
        assert len(by_channel["PV:A"]["timestamps"]) == 10  # 100 -> 10
        assert len(by_channel["PV:B"]["timestamps"]) == 10  # 25 -> 10
        # Timestamps are NOT shared across channels.
        assert by_channel["PV:A"]["timestamps"] != by_channel["PV:B"]["timestamps"]


class TestArchiverDownsampleNonNumericChannel:
    """Enum/status channels (machine mode, interlock state, RF state) carry
    strings, not numbers. LTTB's triangle-area math is meaningless for these
    -- values must survive untouched, never coerced toward 0.0.
    """

    @pytest.fixture
    def status_entry(self, art_store):
        n = 200
        timestamps = [f"2026-02-19T12:{i:02d}:00Z" for i in range(n)]
        statuses = ["CW" if i % 2 == 0 else "STANDBY" for i in range(n)]
        ts_data = _make_new_format_series_data({"T:MODE": (timestamps, statuses)})
        return art_store.save_data(
            tool="archiver_read",
            data=ts_data,
            title="Status channel",
            description="Status channel",
            summary={},
            access_details={},
            category="archiver_data",
        )

    @pytest.mark.asyncio
    async def test_status_channel_values_never_coerced(self, status_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=status_entry.id, max_points=20)
        result = json.loads(raw)

        assert len(result["datasets"]) == 1
        ds = result["datasets"][0]
        assert ds["channel"] == "T:MODE"
        assert len(ds["values"]) <= 20
        assert all(v in ("CW", "STANDBY") for v in ds["values"])
        # First and last real samples are preserved even for the even-subsample path.
        assert ds["values"][0] == "CW"
        # The front-end needs this flag -- an enum channel cannot share a numeric y-axis.
        assert ds["numeric"] is False


class TestArchiverDownsampleNumericFlag:
    """Every dataset carries a ``numeric`` flag so callers don't re-scan values."""

    @pytest.fixture
    def mixed_entry(self, art_store):
        n = 50
        ts_data = _make_new_format_series_data(
            {
                "PV:NUMERIC": (
                    [f"2026-02-19T12:{i:02d}:00Z" for i in range(n)],
                    [float(i) for i in range(n)],
                ),
                "T:STATUS": (
                    [f"2026-02-19T12:{i:02d}:00Z" for i in range(n)],
                    ["CW" if i % 2 == 0 else "STANDBY" for i in range(n)],
                ),
            }
        )
        return art_store.save_data(
            tool="archiver_read",
            data=ts_data,
            title="Mixed dtype",
            description="Mixed dtype",
            summary={},
            access_details={},
            category="archiver_data",
        )

    @pytest.mark.asyncio
    async def test_numeric_and_non_numeric_channels_flagged_correctly(self, mixed_entry):
        fn = _get_archiver_downsample()
        raw = await fn(entry_id=mixed_entry.id, max_points=200)
        result = json.loads(raw)

        by_channel = {ds["channel"]: ds for ds in result["datasets"]}
        assert by_channel["PV:NUMERIC"]["numeric"] is True
        assert by_channel["T:STATUS"]["numeric"] is False
