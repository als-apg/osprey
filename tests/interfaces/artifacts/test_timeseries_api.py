"""Tests for the timeseries artifact data API and LTTB downsampling.

Covers:
  - GET /api/artifacts/{id}/data (no format, chart, table)
  - format param on non-timeseries → 400
  - LTTB algorithm properties
"""

import json
import math

import pytest


def _write_artifact(store, workspace_root, payload, filename, title):
    """Write ``payload`` as an artifact's data file and register the entry.

    The half every builder below shares: the entry itself is always the same
    placeholder text file whose ``metadata.data_file`` points at the JSON the
    endpoint actually reads. Only the payload shape differs between builders,
    so only the payload is built by them.
    """
    data_dir = workspace_root / "data"
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / filename
    data_file.write_text(json.dumps(payload))

    return store.save_file(
        file_content=b"timeseries placeholder",
        filename="ts.txt",
        artifact_type="text",
        title=title,
        description=f"test {title} artifact",
        mime_type="text/plain",
        tool_source="archiver_read",
        metadata={
            "data_type": "timeseries",
            "data_file": str(data_file),
        },
    )


def _make_timeseries_artifact(store, workspace_root, n_rows=100, n_channels=2, channels=None):
    """Create an artifact with associated timeseries data file."""
    if channels is None:
        channels = [f"PV:CH{i}" for i in range(n_channels)]
    else:
        n_channels = len(channels)

    payload = {
        "columns": channels,
        "index": list(range(n_rows)),
        "data": [
            [math.sin(2 * math.pi * r / n_rows + c) for c in range(n_channels)]
            for r in range(n_rows)
        ],
    }
    entry = _write_artifact(
        store, workspace_root, {"data": payload}, f"ts_{n_rows}.json", f"Timeseries ({n_rows} rows)"
    )
    return entry, payload


def _make_new_format_artifact(store, workspace_root, series, filename="series_artifact.json"):
    """Create an artifact using the new long-format archiver_read payload:

    ``{"query": {...}, "series": {channel: {"timestamps": [...], "values": [...]}}}``.

    ``series`` maps channel name -> (timestamps, values); channels may have
    independent lengths/cadences (unlike the legacy split-orient layout,
    where every column shares one index). Pairs are always built well-formed
    and equal-length -- see ``_make_raw_series_artifact`` for the deliberately
    malformed cases this shape cannot express.
    """
    payload = {
        "query": {"channels": list(series.keys())},
        "series": {
            ch: {"timestamps": list(ts), "values": list(vals)} for ch, (ts, vals) in series.items()
        },
    }
    return _write_artifact(
        store, workspace_root, payload, filename, "New format timeseries"
    ), payload


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
    entry = _write_artifact(
        store,
        workspace_root,
        payload,
        "dataframe_wrapper.json",
        "Legacy dataframe-wrapper timeseries",
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
    return _write_artifact(
        store, workspace_root, payload, filename, "Raw series timeseries"
    ), payload


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
            assert ch["returned_points"] <= 100
            assert len(ch["timestamps"]) == ch["returned_points"]
            assert len(ch["values"]) == ch["returned_points"]
        # "Was anything reduced" is published once, on the summary, derived from
        # the counts above rather than repeated as a per-channel flag.
        assert data["summary"]["downsampled"] is True

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
            assert ch["returned_points"] == 50
        assert data["summary"]["downsampled"] is False

    @pytest.mark.unit
    def test_chart_format_new_layout_channels_have_independent_timestamps(self, app_client):
        """A new-format archiver artifact renders per-channel series, not empty.

        Reproduces + verifies the fix for the reported bug: the split-orient-only
        extractor this endpoint used to call had no branch for the
        ``{"query", "series"}`` payload, so it returned empty
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
    def test_chart_payload_carries_the_server_computed_info_bar_aggregates(self, app_client):
        """The chart response owns its own cross-channel aggregates.

        The info bar needs three numbers the client would otherwise re-derive by
        summing ``channels`` itself, plus one it *cannot* derive at all: the
        unioned row count. Two channels on independent cadences make the two
        quantities differ (5 samples across 3 distinct timestamps), so a
        ``row_count`` that merely echoed the per-channel sum would fail here.
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

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=2000")
        assert resp.status_code == 200
        summary = resp.json()["summary"]
        assert summary["total_points"] == 5  # 3 + 2, the cross-channel sum
        assert summary["returned_points"] == 5  # nothing downsampled at max_points=2000
        assert summary["downsampled"] is False
        assert summary["row_count"] == 3  # union of t0, t1, t2 -- not 5

        # The number the badge shows must be the one the table's pagination
        # footer shows; that reconciliation is only possible server-side.
        table = client.get(f"/api/artifacts/{entry.id}/data?format=table").json()
        assert summary["row_count"] == table["total_rows"]

    @pytest.mark.unit
    def test_chart_summary_reports_downsampling_across_channels(self, app_client):
        """``downsampled`` is an any(), and ``returned_points`` the post-LTTB sum."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_new_format_artifact(
            store,
            workspace,
            {
                # Only PV:A exceeds max_points, so `downsampled` must be an
                # any() over channels, not an all().
                "PV:A": ([f"t{i:04d}" for i in range(500)], [float(i) for i in range(500)]),
                "PV:B": ([f"t{i:04d}" for i in range(10)], [float(i) for i in range(10)]),
            },
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=100")
        assert resp.status_code == 200
        data = resp.json()
        summary = data["summary"]
        channels = data["channels"]
        assert summary["downsampled"] is True
        assert summary["total_points"] == 510
        assert summary["returned_points"] == sum(ch["returned_points"] for ch in channels)
        assert summary["returned_points"] < summary["total_points"]
        assert summary["row_count"] == 500  # t0000..t0499, PV:B's stamps are a subset

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
    def test_table_columns_are_the_columns_its_own_rows_were_built_from(self, app_client):
        """``format=table`` is a self-consistent payload: header *and* cells.

        The table view's header used to be taken from the ``format=chart``
        response while its cells came from ``format=table`` -- two separate
        requests against a file the gallery treats as live, so a channel
        appearing or disappearing between them silently shifts correct numbers
        under the wrong PV names. The table response therefore carries the
        column list the pivot actually indexed each row with: same length as
        every row, same order as the values in it.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_new_format_artifact(
            store,
            workspace,
            {
                # Distinct value ranges per channel, and a series order that is
                # NOT alphabetical, so a header built from anything but the
                # rows' own column list is detectable from the cells alone.
                "PV:C": (["t0", "t1"], [100.0, 200.0]),
                "PV:A": (["t0", "t1"], [1.0, 2.0]),
                "PV:B": (["t0", "t1"], [10.0, 20.0]),
            },
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 200
        table = resp.json()
        columns = table["columns"]
        assert all(len(row) == len(columns) for row in table["data"])
        by_ts = dict(zip(table["index"], table["data"], strict=True))
        for ts, scale in (("t0", 1.0), ("t1", 2.0)):
            cells = dict(zip(columns, by_ts[ts], strict=True))
            assert cells == {"PV:A": scale, "PV:B": scale * 10, "PV:C": scale * 100}

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
    def test_duplicate_detection_names_the_first_repeat_by_position(self, app_client):
        """Which duplicate the 500 names is part of the contract.

        ``["b", "a", "a", "b"]`` has two repeats: ``"a"`` at index 2 and ``"b"``
        at index 3. The error must name ``"a"`` -- the first repeat reached
        scanning left to right -- not ``"b"``, whose *first occurrence* comes
        earlier. A rewrite that detects duplicates in bulk and then re-derives
        the offender (e.g. from the first duplicated key by insertion order)
        flips this.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {"PV:A": {"timestamps": ["b", "a", "a", "b"], "values": [1.0, 2.0, 3.0, 4.0]}},
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 500
        assert resp.json()["detail"] == (
            "Channel 'PV:A' has more than one sample at timestamp 'a'; "
            "a table view has exactly one cell per channel per timestamp "
            "and cannot represent both without silently discarding one."
        )

    @pytest.mark.unit
    def test_duplicate_beyond_the_shorter_values_list_is_not_an_error(self, app_client):
        """Pairs are zipped, so timestamps past the end of ``values`` don't exist.

        ``timestamps`` repeats ``"t0"`` at index 3, but ``values`` has only 3
        entries -- that pair is never formed, so there is no second cell to
        collide with and the table must render.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {"PV:A": {"timestamps": ["t0", "t1", "t2", "t0"], "values": [1.0, 2.0, 3.0]}},
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 200
        assert resp.json()["total_rows"] == 3

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

    @pytest.mark.unit
    def test_unhashable_timestamps_still_pivot_into_a_table(self, app_client):
        """A timestamp that is itself a JSON array cannot key a dict, which is
        how each channel's samples are looked up per row.

        Nothing in this repo writes such an artifact, but the endpoint renders
        whatever JSON an entry's ``data_file`` points at, and the cells are
        perfectly displayable — the same reasoning that keeps the
        incomparable-types sort fallback. Falls back to matching timestamps by
        equality instead of by hash.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {
                "PV:A": {"timestamps": [["t0"], ["t1"]], "values": [1.0, 2.0]},
                "PV:B": {"timestamps": [["t1"]], "values": ["CW"]},
            },
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_rows"] == 2
        assert body["columns"] == ["PV:A", "PV:B"]
        # PV:B has no sample at ["t0"] -- that cell is a gap, not a value.
        assert body["data"] == [[1.0, None], [2.0, "CW"]]

    @pytest.mark.unit
    def test_unhashable_duplicate_timestamps_are_still_a_visible_error(self, app_client):
        """The equality fallback must not quietly become laxer than the hash
        path it replaces: two samples at the same unhashable label still cannot
        share one cell, so this raises rather than dropping one of them.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {"PV:A": {"timestamps": [["t0"], ["t0"]], "values": [1.0, 2.0]}},
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=table")
        assert resp.status_code == 500
        assert "more than one sample at timestamp" in resp.json()["detail"]
        assert "PV:A" in resp.json()["detail"]

    @pytest.mark.unit
    def test_unhashable_timestamps_do_not_crash_the_chart_row_count(self, app_client):
        """A timestamp that is itself a JSON array is unhashable, so the row axis
        cannot be deduplicated through a ``set``.

        ``format=chart`` renders such an artifact fine -- it draws each channel
        on its own timestamps and never needs a shared axis. It only acquired a
        row count (for the info bar) once both formats started sharing one
        row-axis helper, and a number shown in a corner must not be able to take
        down a chart that would otherwise draw. Same rule as the
        incomparable-types case above.

        ``format=table`` is deliberately NOT covered: a table has one cell per
        (timestamp, channel) pair, so it must key a lookup by the timestamp
        itself. An unhashable label cannot be represented at all, and that path
        raised before this row-count work and still does.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_raw_series_artifact(
            store,
            workspace,
            {"PV:A": {"timestamps": [["t0"], ["t1"], ["t0"]], "values": [1.0, 2.0, 3.0]}},
        )

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        assert resp.status_code == 200
        assert resp.json()["summary"]["row_count"] == 2


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


class TestTablePivotPagination:
    """``format=table`` must build only the requested page, not the whole file.

    The pivot used to materialize one cell per (timestamp, channel) pair across
    the entire artifact and slice afterwards -- redone from scratch on every
    page click, with a 50-row default page and a 200 MB file cap. The shared
    axis still has to be unioned and sorted in full to know where the page
    starts, but that is one entry per timestamp rather than one per timestamp
    *and* channel.
    """

    @staticmethod
    def _series(n_rows: int, channels: list[str]) -> dict[str, dict]:
        stamps = [f"2026-07-30T00:00:{i:02d}+00:00" for i in range(n_rows)]
        return {
            ch: {"timestamps": stamps, "values": [float(i + c) for i in range(n_rows)]}
            for c, ch in enumerate(channels)
        }

    @pytest.mark.unit
    def test_only_the_requested_page_is_materialized(self):
        """Counts the actual row-building work rather than trusting the shape.

        Every cell built does one ``value_by_channel[ch]`` lookup, i.e. one hash
        of a channel name, so channel names that count their own hashing measure
        cells materialized directly: ``n_channels`` for the one-time
        ``value_by_channel`` insert, then ``n_channels`` per row built.
        Asserting only ``len(rows) == limit`` would still pass a version that
        builds every row and slices inside the function -- which is the
        regression this guards.
        """
        from osprey.interfaces.artifacts.app import _pivot_channel_series_to_table

        class CountingName(str):
            hashes = 0

            def __hash__(self):
                type(self).hashes += 1
                return super().__hash__()

        channels = [CountingName(n) for n in ("PV:A", "PV:B", "PV:C")]
        series = self._series(1000, channels)
        CountingName.hashes = 0

        columns, page, rows, total = _pivot_channel_series_to_table(series, offset=0, limit=10)

        assert total == 1000
        assert len(page) == len(rows) == 10
        # 3 inserts + 3 lookups x 10 rows == 33; a whole-file build would be
        # 3 + 3 x 1000. Bounded rather than exact so an extra bookkeeping hash
        # is not a test failure, while a full materialization still is.
        assert CountingName.hashes <= len(channels) * (10 + 5)
        assert list(columns) == list(channels)

    @pytest.mark.unit
    def test_page_contents_match_the_full_pivot(self):
        """Pagination must not change which values land in which cell."""
        from osprey.interfaces.artifacts.app import _pivot_channel_series_to_table

        channels = ["PV:A", "PV:B"]
        series = self._series(50, channels)

        _cols, full_index, full_rows, full_total = _pivot_channel_series_to_table(
            series, offset=0, limit=50
        )
        _cols, page_index, page_rows, page_total = _pivot_channel_series_to_table(
            series, offset=20, limit=5
        )

        assert full_total == page_total == 50
        assert len(full_rows) == 50
        assert page_index == full_index[20:25]
        assert page_rows == full_rows[20:25]

    @pytest.mark.unit
    def test_columns_are_derived_from_the_series_and_index_the_rows(self):
        """The pivot reports the columns it built the rows against.

        Nothing outside can hand it a column list any more, so the header the
        endpoint returns cannot drift from the cells: the same list that indexes
        each row is the one returned. Channels carry disjoint value ranges, so a
        mutant returning e.g. ``sorted(series)`` while building rows in
        insertion order is caught by the cells, not just by the header's order.
        """
        from osprey.interfaces.artifacts.app import _pivot_channel_series_to_table

        series = {
            "PV:Z": {"timestamps": ["t0", "t1"], "values": [1.0, 2.0]},
            "PV:A": {"timestamps": ["t0", "t1"], "values": [10.0, 20.0]},
        }

        columns, index, rows, total = _pivot_channel_series_to_table(series, offset=0, limit=10)

        assert columns == ["PV:Z", "PV:A"]  # series order, not sorted
        assert total == 2
        assert index == ["t0", "t1"]
        assert [dict(zip(columns, row, strict=True)) for row in rows] == [
            {"PV:Z": 1.0, "PV:A": 10.0},
            {"PV:Z": 2.0, "PV:A": 20.0},
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("offset", "limit", "expected"),
        [
            (0, 5, 5),
            (8, 5, 2),  # page runs past the end
            (10, 5, 0),  # offset at the end
            (99, 5, 0),  # offset past the end
        ],
    )
    def test_page_boundaries(self, offset, limit, expected):
        from osprey.interfaces.artifacts.app import _pivot_channel_series_to_table

        series = self._series(10, ["PV:A"])
        _columns, page, rows, total = _pivot_channel_series_to_table(
            series, offset=offset, limit=limit
        )

        assert total == 10
        assert len(page) == len(rows) == expected
