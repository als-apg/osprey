"""Tests for the timeseries context data API.

Covers:
  - Default endpoint returns full JSON (backward compatibility)
  - ?format=chart returns downsampled structure
  - Small datasets pass through without downsampling
  - ?format=table returns correct paginated slice
  - Table offset bounds handling
  - format param on non-timeseries → 400
  - LTTB preserves extrema
  - Multi-channel shared indices
"""

import math

import pytest


def _make_timeseries_entry(context_store, n_rows=100, n_channels=2, channels=None):
    """Create a timeseries data context entry with synthetic data."""
    if channels is None:
        channels = [f"PV:CH{i}" for i in range(n_channels)]
    else:
        n_channels = len(channels)

    columns = channels
    index = list(range(n_rows))
    data = [
        [math.sin(2 * math.pi * r / n_rows + c) for c in range(n_channels)]
        for r in range(n_rows)
    ]

    payload = {"columns": columns, "index": index, "data": data}

    entry = context_store.save(
        tool="archiver_read",
        description=f"Test timeseries ({n_rows} rows)",
        summary={"channels": channels, "time_range": "test"},
        access_details={"format": "split-orient DataFrame"},
        data=payload,
        data_type="timeseries",
    )
    return entry, payload


def _make_json_entry(context_store):
    """Create a non-timeseries data context entry."""
    payload = {"status": "ok", "value": 42}
    entry = context_store.save(
        tool="channel_read",
        description="Test channel read",
        summary={"channel": "PV:TEST"},
        access_details={"format": "scalar"},
        data=payload,
        data_type="scalar",
    )
    return entry, payload


class TestTimeseriesAPI:
    """Tests for the timeseries format endpoints."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_default_returns_full_json(self, app_client):
        """No format param → full JSON file (backward compatible)."""
        ctx = app_client.app.state.context_store
        entry, payload = _make_timeseries_entry(ctx, n_rows=50)

        resp = app_client.get(f"/api/context/{entry.id}/data")
        assert resp.status_code == 200
        raw = resp.json()
        # File has envelope: {"_osprey_metadata": ..., "data": {...}}
        data = raw["data"]
        assert data["columns"] == payload["columns"]
        assert len(data["index"]) == 50
        assert len(data["data"]) == 50

    @pytest.mark.unit
    def test_chart_format_returns_downsampled(self, app_client):
        """?format=chart with large dataset returns downsampled structure."""
        ctx = app_client.app.state.context_store
        entry, _ = _make_timeseries_entry(ctx, n_rows=5000)

        resp = app_client.get(
            f"/api/context/{entry.id}/data?format=chart&max_points=100"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 5000
        assert data["downsampled"] is True
        assert data["returned_points"] <= 100
        assert len(data["index"]) == data["returned_points"]
        assert len(data["data"]) == data["returned_points"]
        # All columns preserved
        assert data["columns"] == ["PV:CH0", "PV:CH1"]

    @pytest.mark.unit
    def test_chart_small_dataset_not_downsampled(self, app_client):
        """Small dataset passes through without downsampling."""
        ctx = app_client.app.state.context_store
        entry, _ = _make_timeseries_entry(ctx, n_rows=50)

        resp = app_client.get(
            f"/api/context/{entry.id}/data?format=chart&max_points=2000"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 50
        assert data["downsampled"] is False
        assert data["returned_points"] == 50

    @pytest.mark.unit
    def test_table_format_returns_correct_slice(self, app_client):
        """?format=table returns paginated slice."""
        ctx = app_client.app.state.context_store
        entry, payload = _make_timeseries_entry(ctx, n_rows=200)

        resp = app_client.get(
            f"/api/context/{entry.id}/data?format=table&offset=50&limit=25"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 200
        assert data["offset"] == 50
        assert data["limit"] == 25
        assert data["returned_rows"] == 25
        assert data["index"] == payload["index"][50:75]

    @pytest.mark.unit
    def test_table_offset_beyond_bounds(self, app_client):
        """Offset past end returns empty result."""
        ctx = app_client.app.state.context_store
        entry, _ = _make_timeseries_entry(ctx, n_rows=50)

        resp = app_client.get(
            f"/api/context/{entry.id}/data?format=table&offset=100&limit=25"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_rows"] == 0
        assert data["index"] == []
        assert data["data"] == []

    @pytest.mark.unit
    def test_table_partial_last_page(self, app_client):
        """Last page with fewer rows than limit."""
        ctx = app_client.app.state.context_store
        entry, _ = _make_timeseries_entry(ctx, n_rows=75)

        resp = app_client.get(
            f"/api/context/{entry.id}/data?format=table&offset=50&limit=50"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_rows"] == 25  # only 25 remaining

    @pytest.mark.unit
    def test_format_on_non_timeseries_returns_400(self, app_client):
        """format param on non-timeseries entry → 400."""
        ctx = app_client.app.state.context_store
        entry, _ = _make_json_entry(ctx)

        resp = app_client.get(f"/api/context/{entry.id}/data?format=chart")
        assert resp.status_code == 400
        assert "timeseries" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_format_on_missing_entry_returns_404(self, app_client):
        """format param on nonexistent entry → 404."""
        resp = app_client.get("/api/context/9999/data?format=chart")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_invalid_format_value_returns_422(self, app_client):
        """Invalid format value → 422 validation error."""
        ctx = app_client.app.state.context_store
        entry, _ = _make_timeseries_entry(ctx, n_rows=10)

        resp = app_client.get(f"/api/context/{entry.id}/data?format=invalid")
        assert resp.status_code == 422


class TestLTTBAlgorithm:
    """Unit tests for the LTTB downsampling function."""

    @pytest.mark.unit
    def test_preserves_endpoints(self):
        """First and last points always preserved."""
        from osprey.interfaces.artifacts.app import lttb_downsample

        index = list(range(100))
        data = [[float(i)] for i in range(100)]

        new_idx, new_data = lttb_downsample(index, data, 10)
        assert new_idx[0] == 0
        assert new_idx[-1] == 99

    @pytest.mark.unit
    def test_passthrough_small_data(self):
        """Data smaller than max_points passes through unchanged."""
        from osprey.interfaces.artifacts.app import lttb_downsample

        index = list(range(5))
        data = [[float(i)] for i in range(5)]

        new_idx, new_data = lttb_downsample(index, data, 100)
        assert new_idx == index
        assert new_data == data

    @pytest.mark.unit
    def test_preserves_extrema(self):
        """LTTB should preserve clear peaks and valleys."""
        from osprey.interfaces.artifacts.app import lttb_downsample

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
        from osprey.interfaces.artifacts.app import lttb_downsample

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
        from osprey.interfaces.artifacts.app import lttb_downsample

        index = list(range(1000))
        data = [[float(i)] for i in range(1000)]

        new_idx, new_data = lttb_downsample(index, data, 100)
        assert len(new_idx) == 100
        assert len(new_data) == 100
