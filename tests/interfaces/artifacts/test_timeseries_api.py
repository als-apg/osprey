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
    """Write ``payload`` as an artifact's data file and register the entry."""
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
    independent lengths/cadences.
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
    -- the shape ``archiver_read`` wrote to disk before the long-format migration.
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
    """Create an artifact from an already-built ``series`` dict with no coercion,
    for deliberately malformed payloads (duplicate timestamps, mismatched
    lengths, mixed types).
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
        """A new-format archiver artifact renders per-channel series, not empty."""
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
        # The front-end needs this flag -- an enum channel cannot share the numeric y-axis.
        assert ch["numeric"] is False

    @pytest.mark.unit
    def test_chart_payload_carries_the_server_computed_info_bar_aggregates(self, app_client):
        """The chart response owns its own cross-channel aggregates.

        ``row_count`` counts the unioned timestamp axis and is not derivable
        client-side: two channels on independent cadences give 5 samples
        across 3 distinct timestamps.
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

        # The badge number must match the table's pagination total.
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

        The table response carries the column list the pivot actually indexed
        each row with -- same length as every row, same order as its values.
        """
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_new_format_artifact(
            store,
            workspace,
            {
                # Non-alphabetical order and distinct value ranges per channel
                # make a wrong header detectable from the cells alone.
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
        """The legacy wrapper layout already on disk must still render in both formats."""
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
        """A null cell in a legacy wide layout survives the drop-then-pivot
        round trip exactly, null for null.
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


class TestDataFileResolution:
    """How ``/api/artifacts/{id}/data`` locates the entry's ``data_file`` on
    disk: absolute paths are used as-is; relative paths are tried against the
    workspace parent, the artifact store dir, and the workspace root in turn.
    """

    _PAYLOAD = {
        "query": {"channels": ["PV:A"]},
        "series": {"PV:A": {"timestamps": ["t0", "t1"], "values": [1.0, 2.0]}},
    }

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app), tmp_path

    @staticmethod
    def _save_with_data_file(store, data_file: str):
        """Register a timeseries artifact whose ``data_file`` is the given string."""
        return store.save_file(
            file_content=b"timeseries placeholder",
            filename="ts.txt",
            artifact_type="text",
            title="Relative data file",
            description="data_file resolution test",
            mime_type="text/plain",
            tool_source="archiver_read",
            metadata={"data_type": "timeseries", "data_file": data_file},
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "data_file",
        ["data/rel_series.json", "bare_series.json"],
        ids=["workspace-relative", "bare-filename"],
    )
    def test_relative_data_file_resolves(self, app_client, data_file):
        """A workspace-relative path (the last candidate tried) and a bare
        filename (tried against the store dir) both still resolve — legacy
        entries wrote such paths."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry = self._save_with_data_file(store, data_file)
        # save_file created the store dir; a bare name lands beside the index.
        base = store.artifact_dir if data_file == "bare_series.json" else workspace
        target = base / data_file
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(self._PAYLOAD))

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        assert resp.status_code == 200
        assert resp.json()["channels"][0]["channel"] == "PV:A"

    @pytest.mark.unit
    @pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
    def test_missing_data_file_returns_404(self, app_client, absolute):
        """A data_file that resolves nowhere — absolute, or relative and
        matching no candidate location — is a 404, not a 500."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        data_file = str(workspace / "vanished.json") if absolute else "data/ghost_series.json"
        entry = self._save_with_data_file(store, data_file)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        assert resp.status_code == 404
        assert "not found on disk" in resp.json()["detail"]

    @pytest.mark.unit
    def test_oversized_data_file_returns_413(self, app_client, monkeypatch):
        """A data file over the size cap is refused with 413 rather than
        parsed; the cap protects the gallery process, not the client."""
        monkeypatch.setattr("osprey.interfaces.artifacts.app.MAX_TIMESERIES_FILE_BYTES", 64)
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_timeseries_artifact(store, workspace, n_rows=50)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart")
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

        # The cap applies only to the parsed formats: the raw passthrough
        # (no format param) still streams the file bytes.
        raw = client.get(f"/api/artifacts/{entry.id}/data")
        assert raw.status_code == 200


class TestArtifactTablePivotRobustness:
    """Edge cases in ``_pivot_channel_series_to_table``, the union-and-fill
    pivot behind ``format=table``.
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
        """A legacy shared index with a duplicated label must fail loudly, not
        silently collapse rows and discard a sample.
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
        table path when the chart path already tolerates it.
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

        ``["b", "a", "a", "b"]`` has two repeats; the error must name ``"a"``,
        the first repeat reached scanning left to right -- not ``"b"``, whose
        *first occurrence* comes earlier.
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
        """Pairs are zipped, so timestamps past the end of ``values`` don't exist."""
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
        """A timestamp that is itself a JSON array cannot key a dict; the pivot
        falls back to matching timestamps by equality instead of by hash.
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
        """An unhashable timestamp cannot be deduplicated through a ``set``; the
        chart's row count must not take down a chart that would otherwise draw.

        ``format=table`` is deliberately NOT covered: a table must key a lookup
        by the timestamp itself, so an unhashable label cannot be represented.
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

    The shared axis still has to be unioned and sorted in full to know where
    the page starts, but that is one entry per timestamp rather than one per
    timestamp *and* channel.
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

        Every cell built does one ``value_by_channel[ch]`` lookup, so channel
        names that count their own hashing measure cells materialized directly.
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
        # 3 inserts + 3 lookups x 10 rows == 33; bounded rather than exact so
        # an extra bookkeeping hash is not a failure, but a whole-file build is.
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
        """The pivot reports the columns it built the rows against."""
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
