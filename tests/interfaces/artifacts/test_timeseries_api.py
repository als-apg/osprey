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
        """?format=chart with large dataset returns downsampled structure."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_timeseries_artifact(store, workspace, n_rows=5000)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 5000
        assert data["downsampled"] is True
        assert data["returned_points"] <= 100

    @pytest.mark.unit
    def test_chart_small_dataset_not_downsampled(self, app_client):
        """Small dataset passes through without downsampling."""
        client, workspace = app_client
        store = client.app.state.artifact_store
        entry, _ = _make_timeseries_artifact(store, workspace, n_rows=50)

        resp = client.get(f"/api/artifacts/{entry.id}/data?format=chart&max_points=2000")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 50
        assert data["downsampled"] is False
        assert data["returned_points"] == 50

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
    def test_highlight_artifact(self, app_client):
        store = app_client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"test",
            filename="test.txt",
            artifact_type="text",
            title="Highlight Test",
            description="",
            mime_type="text/plain",
            tool_source="test",
            highlighted=False,
        )

        resp = app_client.post(f"/api/artifacts/{entry.id}/highlight", json={"highlighted": True})
        assert resp.status_code == 200
        assert resp.json()["highlighted"] is True

    @pytest.mark.unit
    def test_pin_not_found(self, app_client):
        resp = app_client.post("/api/artifacts/nonexistent/pin", json={"pinned": True})
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_highlight_not_found(self, app_client):
        resp = app_client.post("/api/artifacts/nonexistent/highlight", json={"highlighted": True})
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_list_filter_highlighted(self, app_client):
        """GET /api/artifacts?highlighted=true filters correctly."""
        store = app_client.app.state.artifact_store
        store.save_file(
            file_content=b"a", filename="a.txt", artifact_type="text",
            title="A", description="", mime_type="text/plain",
            tool_source="test", highlighted=True,
        )
        store.save_file(
            file_content=b"b", filename="b.txt", artifact_type="text",
            title="B", description="", mime_type="text/plain",
            tool_source="test", highlighted=False,
        )

        resp = app_client.get("/api/artifacts?highlighted=true")
        assert resp.json()["count"] == 1
        assert resp.json()["artifacts"][0]["title"] == "A"

        resp = app_client.get("/api/artifacts?highlighted=false")
        assert resp.json()["count"] == 1
        assert resp.json()["artifacts"][0]["title"] == "B"


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
