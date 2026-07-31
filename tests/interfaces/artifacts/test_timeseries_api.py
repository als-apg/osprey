"""Tests for the timeseries artifact data API and LTTB downsampling.

Covers:
  - GET /api/artifacts/{id}/data (no format, chart, table)
  - format param on non-timeseries → 400
  - LTTB algorithm properties
"""

import json
import math

import pytest


def _make_timeseries_artifact(store, workspace_root, n_rows=100, n_channels=2, channels=None):
    """Create an artifact with associated timeseries data file."""
    if channels is None:
        channels = [f"PV:CH{i}" for i in range(n_channels)]
    else:
        n_channels = len(channels)

    columns = channels
    index = list(range(n_rows))
    data = [
        [math.sin(2 * math.pi * r / n_rows + c) for c in range(n_channels)] for r in range(n_rows)
    ]

    payload = {"columns": columns, "index": index, "data": data}

    # Write data file
    data_dir = workspace_root / "data"
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / f"ts_{n_rows}.json"
    data_file.write_text(json.dumps({"data": payload}))

    entry = store.save_file(
        file_content=b"timeseries placeholder",
        filename="ts.txt",
        artifact_type="text",
        title=f"Timeseries ({n_rows} rows)",
        description="test timeseries artifact",
        mime_type="text/plain",
        tool_source="archiver_read",
        metadata={
            "data_type": "timeseries",
            "data_file": str(data_file),
        },
    )
    return entry, payload


def _make_new_format_artifact(store, workspace_root, series, filename="series_artifact.json"):
    """Create an artifact using the new long-format archiver_read payload:

    ``{"query": {...}, "series": {channel: {"timestamps": [...], "values": [...]}}}``.

    ``series`` maps channel name -> (timestamps, values); channels may have
    independent lengths/cadences (unlike the legacy split-orient layout,
    where every column shares one index).
    """
    payload = {
        "query": {"channels": list(series.keys())},
        "series": {
            ch: {"timestamps": list(ts), "values": list(vals)} for ch, (ts, vals) in series.items()
        },
    }

    data_dir = workspace_root / "data"
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / filename
    data_file.write_text(json.dumps(payload))

    entry = store.save_file(
        file_content=b"timeseries placeholder",
        filename="ts.txt",
        artifact_type="text",
        title="New format timeseries",
        description="test new-format timeseries artifact",
        mime_type="text/plain",
        tool_source="archiver_read",
        metadata={
            "data_type": "timeseries",
            "data_file": str(data_file),
        },
    )
    return entry, payload


def _make_legacy_dataframe_artifact(store, workspace_root, columns, index, data):
    """Create an artifact using the legacy archiver split-orient wrapper layout:

    ``{"dataframe": {"columns": [...], "index": [...], "data": [...]}, "query": {...}}``
    -- this is the exact wrapper shape ``archiver_read`` wrote to disk before the
    long-format migration, and is exactly what is already on disk for artifacts
    created before this task.
    """
    payload = {
        "dataframe": {"columns": columns, "index": index, "data": data},
        "query": {"channels": columns},
    }

    data_dir = workspace_root / "data"
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / "dataframe_wrapper.json"
    data_file.write_text(json.dumps(payload))

    entry = store.save_file(
        file_content=b"timeseries placeholder",
        filename="ts.txt",
        artifact_type="text",
        title="Legacy dataframe-wrapper timeseries",
        description="test legacy dataframe-wrapper artifact",
        mime_type="text/plain",
        tool_source="archiver_read",
        metadata={
            "data_type": "timeseries",
            "data_file": str(data_file),
        },
    )
    return entry, payload


def _make_raw_series_artifact(store, workspace_root, series_payload, filename="raw_series.json"):
    """Create an artifact from an already-built ``series`` dict, with no
    coercion of the timestamps/values it's handed -- unlike
    ``_make_new_format_artifact``, which always builds well-formed equal-length
    (timestamps, values) pairs. Used to construct deliberately malformed
    payloads (duplicate timestamps, mismatched lengths, mixed types) that a
    tuple-based helper can't represent.
    """
    payload = {"query": {}, "series": series_payload}

    data_dir = workspace_root / "data"
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / filename
    data_file.write_text(json.dumps(payload))

    entry = store.save_file(
        file_content=b"timeseries placeholder",
        filename="ts.txt",
        artifact_type="text",
        title="Raw series timeseries",
        description="test raw-series artifact",
        mime_type="text/plain",
        tool_source="archiver_read",
        metadata={
            "data_type": "timeseries",
            "data_file": str(data_file),
        },
    )
    return entry, payload


def _make_non_timeseries_artifact(store):
    """Create an artifact without timeseries data."""
    return store.save_file(
        file_content=b"hello",
        filename="test.txt",
        artifact_type="text",
        title="Non-timeseries",
        description="no data file",
        mime_type="text/plain",
        tool_source="test",
    )


class TestArtifactDataAPI:
    """Tests for GET /api/artifacts/{id}/data endpoint."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app), tmp_path

    @pytest.mark.unit
    def test_no_format_returns_full_json(self, app_client):
        """No format param → full JSON data file."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, payload = _make_timeseries_artifact(store, workspace, n_rows=50)

        resp = client.get(f"/api/artifacts/{entry.id}/data")
        assert resp.status_code == 200
        raw = resp.json()
        data = raw["data"]
        assert data["columns"] == payload["columns"]
        assert len(data["index"]) == 50

    @pytest.mark.unit
    def test_chart_format_returns_downsampled(self, app_client):
        """?format=chart with large dataset returns per-channel downsampled series."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, payload = _make_timeseries_artifact(store, workspace, n_rows=5000)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=100")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["channels"]) == len(payload["columns"])
        for ch in data["channels"]:
            assert ch["total_points"] == 5000
            assert ch["downsampled"] is True
            assert ch["returned_points"] <= 100
            assert len(ch["timestamps"]) == ch["returned_points"]
            assert len(ch["values"]) == ch["returned_points"]

    @pytest.mark.unit
    def test_chart_small_dataset_not_downsampled(self, app_client):
        """Small dataset passes through without downsampling."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_timeseries_artifact(store, workspace, n_rows=50)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=2000")
        assert resp.status_code == 200
        data = resp.json()
        for ch in data["channels"]:
            assert ch["total_points"] == 50
            assert ch["downsampled"] is False
            assert ch["returned_points"] == 50

    @pytest.mark.unit
    def test_chart_format_new_layout_channels_have_independent_timestamps(self, app_client):
        """A new-format archiver artifact renders per-channel series, not empty.

        Reproduces + verifies the fix for the reported bug: before this task,
        ``extract_timeseries_frame`` had no branch for the new
        ``{"query", "series"}`` payload, so this endpoint returned empty
        ``columns``/``index``/``data`` for it.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_new_format_artifact(
            store,
            workspace,
            {
                "PV:A": ([f"t{i}" for i in range(100)], [float(i) for i in range(100)]),
                # Sparser, independent cadence.
                "PV:B": ([f"t{i}" for i in range(0, 100, 5)], [float(i) for i in range(0, 100, 5)]),
            },
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=1000")
        assert resp.status_code == 200
        data = resp.json()
        by_channel = {ch["channel"]: ch for ch in data["channels"]}
        assert len(by_channel["PV:A"]["timestamps"]) == 100
        assert len(by_channel["PV:B"]["timestamps"]) == 20
        assert by_channel["PV:A"]["timestamps"] != by_channel["PV:B"]["timestamps"]

    @pytest.mark.unit
    def test_chart_format_non_numeric_channel_not_coerced(self, app_client):
        """An enum/status channel's string values survive the chart path untouched."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        n = 200
        timestamps = [f"t{i}" for i in range(n)]
        statuses = ["CW" if i % 2 == 0 else "STANDBY" for i in range(n)]
        entry, _ = _make_new_format_artifact(store, workspace, {"T:MODE": (timestamps, statuses)})

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=20")
        assert resp.status_code == 200
        data = resp.json()
        ch = data["channels"][0]
        assert ch["channel"] == "T:MODE"
        assert len(ch["values"]) <= 20
        assert all(v in ("CW", "STANDBY") for v in ch["values"])
        # Task 6 (front-end) needs this to pick a trace type/axis per channel --
        # an enum channel cannot share a numeric y-axis with the others.
        assert ch["numeric"] is False

    @pytest.mark.unit
    def test_chart_format_numeric_flag_on_a_numeric_channel(self, app_client):
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_timeseries_artifact(store, workspace, n_rows=50)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        assert resp.status_code == 200
        data = resp.json()
        assert all(ch["numeric"] is True for ch in data["channels"])

    @pytest.mark.unit
    def test_table_format_returns_correct_slice(self, app_client):
        """?format=table returns paginated slice."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, payload = _make_timeseries_artifact(store, workspace, n_rows=200)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table&offset=50&limit=25")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 200
        assert data["offset"] == 50
        assert data["limit"] == 25
        assert data["returned_rows"] == 25
        assert data["index"] == payload["index"][50:75]

    @pytest.mark.unit
    def test_table_format_new_layout_unions_timestamps_with_null_fill(self, app_client):
        """Channels at different cadences are pivoted for display with no fill --
        a missing sample at a shared timestamp becomes null, never interpolated.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_new_format_artifact(
            store,
            workspace,
            {
                "PV:A": (["t0", "t1", "t2"], [1.0, 2.0, 3.0]),
                "PV:B": (["t0", "t2"], [10.0, 30.0]),  # no sample at t1
            },
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 3  # union of t0, t1, t2
        assert data["columns"] == ["PV:A", "PV:B"]
        by_index = dict(zip(data["index"], data["data"], strict=True))
        assert by_index["t1"] == [2.0, None]  # PV:B has no sample at t1 -- null, not filled

    @pytest.mark.unit
    def test_legacy_dataframe_wrapper_layout_chart_and_table(self, app_client):
        """The legacy archiver split-orient WRAPPER layout (``{"dataframe":
        ..., "query": ...}``) is exactly what is already on disk for artifacts
        written before this task -- both formats must still render it.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        columns = ["PV:A", "PV:B"]
        # Zero-padded so lexicographic (string) sort matches insertion order --
        # real archiver timestamps are fixed-width ISO-8601 and sort the same way.
        index = [f"t{i:02d}" for i in range(50)]
        data = [[float(i), float(i) * 2] for i in range(50)]
        entry, _ = _make_legacy_dataframe_artifact(store, workspace, columns, index, data)

        chart_resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=2000")
        assert chart_resp.status_code == 200
        chart = chart_resp.json()
        by_channel = {ch["channel"]: ch for ch in chart["channels"]}
        assert by_channel["PV:A"]["total_points"] == 50
        assert by_channel["PV:B"]["values"][0] == 0.0

        table_resp = client.get(f"/api/artifacts/{entry.id}/data?format=table&limit=50")
        assert table_resp.status_code == 200
        table = table_resp.json()
        assert table["total_rows"] == 50
        assert table["columns"] == columns
        assert table["index"] == index
        assert table["data"] == data

    @pytest.mark.unit
    def test_legacy_wide_layout_with_nulls_round_trips_through_table(self, app_client):
        """A legacy wide layout with a null cell (channel had no sample at that
        shared timestamp) drops the null while transposing
        (``extract_channel_series``) and then re-inserts it while pivoting
        back for display (``_pivot_channel_series_to_table``) -- the
        drop-then-pivot round trip must reproduce the original wide frame
        exactly, null for null.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        columns = ["PV:A", "PV:B"]
        index = ["t0", "t1", "t2"]
        data = [[1.0, None], [None, 20.0], [3.0, 30.0]]
        entry, _ = _make_legacy_dataframe_artifact(store, workspace, columns, index, data)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 200
        table = resp.json()
        assert table["total_rows"] == 3
        assert table["index"] == index
        assert table["data"] == data  # exact round trip, including null positions

    @pytest.mark.unit
    def test_no_data_file_returns_400(self, app_client):
        """Artifact without data_file metadata returns 400."""
        client, _ = app_client
        store = client.app.state.artifact_store
        entry = _make_non_timeseries_artifact(store)

        resp = client.get(f"/api/artifacts/{entry.id}/data")
        assert resp.status_code == 400

    @pytest.mark.unit
    def test_format_on_non_timeseries_returns_400(self, app_client):
        """format param on artifact with non-timeseries data_type → 400."""
        client, workspace = app_client
        store = client.app.state.artifact_store

        # Create artifact with data file but non-timeseries data type
        data_dir = workspace / "data"
        data_dir.mkdir(exist_ok=True)
        data_file = data_dir / "scalar.json"
        data_file.write_text(json.dumps({"data": {"value": 42}}))

        entry = store.save_file(
            file_content=b"scalar",
            filename="scalar.txt",
            artifact_type="text",
            title="Scalar data",
            description="not timeseries",
            mime_type="text/plain",
            tool_source="test",
            metadata={"data_type": "scalar", "data_file": str(data_file)},
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        assert resp.status_code == 400
        assert "timeseries" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_missing_artifact_returns_404(self, app_client):
        """Nonexistent artifact → 404."""
        client, _ = app_client
        resp = client.get("/api/artifacts/nonexistent/data?format=chart")
        assert resp.status_code == 404


class TestArtifactTablePivotRobustness:
    """Edge cases in ``_pivot_channel_series_to_table``, the union-and-fill
    pivot behind ``format=table``.

    A columns/index/data table has exactly one cell per (timestamp, channel)
    pair, so a channel with more than one sample at the same timestamp label
    cannot be represented without silently discarding one of them -- that must
    be a visible failure, not a quiet drop. A length mismatch between one
    channel's own timestamps/values (a malformed artifact, not a shape
    collision) must NOT crash the table path when the chart path tolerates it.
    A union of mutually-incomparable timestamp types must not crash ``sorted()``.
    """

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app), tmp_path

    @pytest.mark.unit
    def test_duplicate_timestamp_within_a_channel_is_a_visible_error_not_silent_loss(
        self, app_client
    ):
        """Reproduces the reviewer's exact repro: a legacy shared index with a
        duplicated label (``["t0", "t0", "t1"]``) used to silently collapse to
        2 rows, discarding the first t0 sample (1.0) with no signal. It must
        now fail loudly instead of returning a corrupted row count.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_legacy_dataframe_artifact(
            store,
            workspace,
            columns=["PV:A"],
            index=["t0", "t0", "t1"],
            data=[[1.0], [2.0], [3.0]],
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code >= 400  # a visible error, not a silently-wrong 200
        assert "t0" in resp.json()["detail"]

    @pytest.mark.unit
    def test_mismatched_channel_length_does_not_500_in_table(self, app_client):
        """A malformed channel (more timestamps than values) must not crash the
        table path when the chart path already tolerates it -- table should be
        at least as tolerant as chart on the same input.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {
                "PV:A": {
                    "timestamps": ["t0", "t1", "t2", "t3", "t4"],
                    "values": [1.0, 2.0, 3.0],  # 2 short
                }
            },
        )

        chart_resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        table_resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert chart_resp.status_code == 200
        assert table_resp.status_code == 200

    @pytest.mark.unit
    def test_mixed_type_timestamps_across_channels_do_not_crash_the_sort(self, app_client):
        """Timestamps of mutually-incomparable types across channels (e.g. an
        int from one channel, a str from another) can't be sorted directly by
        Python's default ``<`` -- this must fall back to a deterministic order
        rather than raising ``TypeError`` out of the endpoint.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {
                "PV:A": {"timestamps": [0, 1], "values": [1.0, 2.0]},
                "PV:B": {"timestamps": ["t1"], "values": ["CW"]},
            },
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 200
        assert resp.json()["total_rows"] == 3


class TestArtifactPinHighlightAPI:
    """Tests for POST /api/artifacts/{id}/pin and /highlight endpoints."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_pin_artifact(self, app_client):
        store = app_client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"test",
            filename="test.txt",
            artifact_type="text",
            title="Pin Test",
            description="",
            mime_type="text/plain",
            tool_source="test",
        )

        resp = app_client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": True})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

        # Verify via list
        resp = app_client.get("/api/artifacts?pinned=true")
        assert resp.json()["count"] == 1

    @pytest.mark.unit
    def test_unpin_artifact(self, app_client):
        store = app_client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"test",
            filename="test.txt",
            artifact_type="text",
            title="Unpin Test",
            description="",
            mime_type="text/plain",
            tool_source="test",
        )
        store.set_pinned(entry.id, True)

        resp = app_client.post(f"/api/artifacts/{entry.id}/pin", json={"pinned": False})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is False

    @pytest.mark.unit
    def test_pin_not_found(self, app_client):
        resp = app_client.post("/api/artifacts/nonexistent/pin", json={"pinned": True})
        assert resp.status_code == 404


class TestLTTBAlgorithm:
    """Unit tests for the LTTB downsampling function."""

    @pytest.mark.unit
    def test_preserves_endpoints(self):
        """First and last points always preserved."""
        from osprey.utils.timeseries import lttb_downsample

        index = list(range(100))
        data = [[float(i)] for i in range(100)]

        new_idx, new_data = lttb_downsample(index, data, 10)
        assert new_idx[0] == 0
        assert new_idx[-1] == 99

    @pytest.mark.unit
    def test_passthrough_small_data(self):
        """Data smaller than max_points passes through unchanged."""
        from osprey.utils.timeseries import lttb_downsample

        index = list(range(5))
        data = [[float(i)] for i in range(5)]

        new_idx, new_data = lttb_downsample(index, data, 100)
        assert new_idx == index
        assert new_data == data

    @pytest.mark.unit
    def test_preserves_extrema(self):
        """LTTB should preserve clear peaks and valleys."""
        from osprey.utils.timeseries import lttb_downsample

        # Create data with a clear spike at index 50
        n = 200
        index = list(range(n))
        data = [[0.0] for _ in range(n)]
        data[50] = [100.0]  # big spike
        data[150] = [-100.0]  # big valley

        new_idx, new_data = lttb_downsample(index, data, 20)
        # The spike and valley should be preserved
        assert 50 in new_idx
        assert 150 in new_idx

    @pytest.mark.unit
    def test_multi_channel_shared_indices(self):
        """All channels use the same selected indices."""
        from osprey.utils.timeseries import lttb_downsample

        n = 500
        index = list(range(n))
        # 3 channels with different patterns
        data = [
            [math.sin(2 * math.pi * r / n), math.cos(2 * math.pi * r / n), float(r)]
            for r in range(n)
        ]

        new_idx, new_data = lttb_downsample(index, data, 50)
        assert len(new_idx) == 50
        assert len(new_data) == 50
        # Each row still has 3 columns
        for row in new_data:
            assert len(row) == 3

    @pytest.mark.unit
    def test_output_size_matches_max_points(self):
        """Output has exactly max_points entries."""
        from osprey.utils.timeseries import lttb_downsample

        index = list(range(1000))
        data = [[float(i)] for i in range(1000)]

        new_idx, new_data = lttb_downsample(index, data, 100)
        assert len(new_idx) == 100
        assert len(new_data) == 100

    @pytest.mark.unit
    def test_handles_none_gap_values_without_crash(self):
        """None gap values (archiver disconnects) must not crash the area math."""
        from osprey.utils.timeseries import lttb_downsample

        n = 200
        index = list(range(n))
        data = [[float(i)] for i in range(n)]
        # Punch a contiguous gap, as an archiver disconnect would produce.
        data[50:60] = [[None] for _ in range(10)]

        new_idx, new_data = lttb_downsample(index, data, 20)
        assert len(new_data) == 20

    @pytest.mark.unit
    def test_preserves_none_gaps_in_output(self):
        """A None gap on a kept point survives into the output (not replaced with 0.0)."""
        from osprey.utils.timeseries import lttb_downsample

        n = 200
        index = list(range(n))
        data = [[float(i)] for i in range(n)]
        # LTTB always keeps the first point (selected = [0]).
        data[0] = [None]

        _new_idx, new_data = lttb_downsample(index, data, 20)
        assert new_data[0] == [None]

    @pytest.mark.unit
    def test_none_in_secondary_column(self):
        """None in a non-representative column is preserved; column 0 drives selection.

        Column 1 is never read by the triangle-area math (only column 0 is), so this
        locks in None-preservation for secondary channels and guards the defensive
        all-column sanitization in the working copy.
        """
        from osprey.utils.timeseries import lttb_downsample

        n = 200
        index = list(range(n))
        data = [[float(i), None] for i in range(n)]

        _new_idx, new_data = lttb_downsample(index, data, 20)
        # First point is always kept and still carries the None in column 1.
        assert new_data[0] == [0.0, None]
        for row in new_data:
            assert row[1] is None
