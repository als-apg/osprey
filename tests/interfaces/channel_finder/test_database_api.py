"""Tests for Channel Finder database REST API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

_DB_PATCH = "osprey.interfaces.channel_finder.database_api._get_database"
_FACILITY_PATCH = "osprey.interfaces.channel_finder.database_api._get_facility_name"


class TestInfoEndpoint:
    """Tests for GET /api/info."""

    def test_info_returns_pipeline_type(self, client):
        mock_db = MagicMock()
        mock_db.get_statistics.return_value = {"total_channels": 100}
        mock_db.chunk_database.return_value = [[]] * 2
        mock_db.db_path = "/tmp/test.json"
        with (
            patch(_DB_PATCH, return_value=mock_db),
            patch(_FACILITY_PATCH, return_value="TEST"),
        ):
            resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_type"] == "in_context"
        assert data["metadata"]["total_channels"] == 100
        assert data["metadata"]["total_chunks_at_50"] == 2
        assert data["metadata"]["facility_name"] == "TEST"


class TestStatisticsEndpoint:
    """Tests for GET /api/statistics."""

    def test_statistics_returns_data(self, client):
        mock_db = MagicMock()
        mock_db.get_statistics.return_value = {"total_channels": 250}
        mock_db.chunk_database.return_value = [[]] * 5
        with (
            patch(_DB_PATCH, return_value=mock_db),
            patch(_FACILITY_PATCH, return_value="ALS"),
        ):
            resp = client.get("/api/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_channels"] == 250
        assert data["total_chunks_at_50"] == 5
        assert data["facility_name"] == "ALS"

    def test_statistics_error_returns_500(self, client):
        mock_db = MagicMock()
        mock_db.get_statistics.side_effect = RuntimeError("Database not loaded")
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/statistics")
        assert resp.status_code == 500


class TestValidateEndpoint:
    """Tests for POST /api/validate."""

    def test_validate_channels(self, client):
        mock_db = MagicMock()
        mock_db.validate_channels.return_value = [
            {"channel": "SR:BPM:01:X", "valid": True},
            {"channel": "INVALID", "valid": False},
        ]
        mock_db.get_valid_channels.return_value = ["SR:BPM:01:X"]
        mock_db.get_invalid_channels.return_value = ["INVALID"]
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/validate", json={"channels": ["SR:BPM:01:X", "INVALID"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid_count"] == 1
        assert data["invalid_count"] == 1


class TestChannelsEndpoint:
    """Tests for GET /api/channels (in-context)."""

    def test_get_channels_chunk(self, client):
        mock_db = MagicMock()
        mock_db.chunk_database.return_value = [[{"name": "ch1", "description": "test"}]]
        mock_db.format_chunk_for_prompt.return_value = "ch1 - test"
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/channels?chunk_idx=0&chunk_size=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_idx"] == 0
        assert len(data["channels"]) == 1
        assert data["formatted"] == "ch1 - test"


class TestPipelineGating:
    """Tests that pipeline-specific endpoints return 404 for wrong pipeline."""

    def test_hierarchical_endpoint_returns_404_on_in_context(self, client):
        resp = client.get("/api/explore/options?level=system")
        assert resp.status_code == 404

    def test_middle_layer_endpoint_returns_404_on_in_context(self, client):
        resp = client.get("/api/explore/systems")
        assert resp.status_code == 404

    def test_tree_crud_returns_404_on_in_context(self, client):
        resp = client.post(
            "/api/tree/node",
            json={"level": "system", "name": "TEST"},
        )
        assert resp.status_code == 404

    def test_structure_crud_returns_404_on_in_context(self, client):
        resp = client.post(
            "/api/structure/family",
            json={"system": "SR", "family": "TEST"},
        )
        assert resp.status_code == 404


class TestCrudEndpoints:
    """Tests for CRUD endpoints with mocked database instances."""

    def test_ic_create_channel(self, client):
        mock_db = MagicMock()
        mock_db.add_channel.return_value = {"success": True, "channel": "TEST:CH"}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post(
                "/api/channels",
                json={"channel_name": "TEST:CH", "address": "TEST:CH"},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_ic_delete_channel(self, client):
        mock_db = MagicMock()
        mock_db.delete_channel.return_value = {"success": True, "channel": "TEST:CH"}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.delete("/api/channels/TEST:CH")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_ic_update_channel(self, client):
        mock_db = MagicMock()
        mock_db.update_channel.return_value = {"success": True, "channel": "TEST:CH"}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.put(
                "/api/channels/TEST:CH",
                json={"description": "Updated"},
            )
        assert resp.status_code == 200

    def test_ic_create_crud_error_returns_400(self, client):
        from osprey.services.channel_finder.core.base_database import DatabaseWriteError

        mock_db = MagicMock()
        mock_db.add_channel.side_effect = DatabaseWriteError("Channel already exists", "duplicate")
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post(
                "/api/channels",
                json={"channel_name": "TEST:CH"},
            )
        assert resp.status_code == 400

    def test_ic_create_unexpected_error_returns_500(self, client):
        mock_db = MagicMock()
        mock_db.add_channel.side_effect = RuntimeError("Unexpected")
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post(
                "/api/channels",
                json={"channel_name": "TEST:CH"},
            )
        assert resp.status_code == 500
