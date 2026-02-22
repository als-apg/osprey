"""Tests for Channel Finder database REST API."""

from __future__ import annotations

import json
from unittest.mock import patch


class TestInfoEndpoint:
    """Tests for GET /api/info."""

    def test_info_returns_pipeline_type(self, client):
        stats_result = json.dumps({"total_channels": 100, "facility_name": "TEST"})
        with patch(
            "osprey.services.channel_finder.mcp.in_context.tools.statistics.statistics"
        ) as mock_stats:
            mock_stats.fn.return_value = stats_result
            resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_type"] == "in_context"


class TestStatisticsEndpoint:
    """Tests for GET /api/statistics."""

    def test_statistics_returns_data(self, client):
        stats_result = json.dumps(
            {
                "total_channels": 250,
                "total_chunks_at_50": 5,
                "facility_name": "ALS",
            }
        )
        with patch(
            "osprey.services.channel_finder.mcp.in_context.tools.statistics.statistics"
        ) as mock_stats:
            mock_stats.fn.return_value = stats_result
            resp = client.get("/api/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_channels"] == 250

    def test_statistics_error_envelope(self, client):
        error_result = json.dumps(
            {
                "error": True,
                "error_type": "internal_error",
                "error_message": "Database not loaded",
                "suggestions": [],
            }
        )
        with patch(
            "osprey.services.channel_finder.mcp.in_context.tools.statistics.statistics"
        ) as mock_stats:
            mock_stats.fn.return_value = error_result
            resp = client.get("/api/statistics")
        assert resp.status_code == 500


class TestValidateEndpoint:
    """Tests for POST /api/validate."""

    def test_validate_channels(self, client):
        validate_result = json.dumps(
            {
                "results": [
                    {"channel": "SR:BPM:01:X", "valid": True},
                    {"channel": "INVALID", "valid": False},
                ],
                "valid_count": 1,
                "invalid_count": 1,
                "total": 2,
            }
        )
        with patch(
            "osprey.services.channel_finder.mcp.in_context.tools.validate.validate"
        ) as mock_val:
            mock_val.fn.return_value = validate_result
            resp = client.post("/api/validate", json={"channels": ["SR:BPM:01:X", "INVALID"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid_count"] == 1
        assert data["invalid_count"] == 1


class TestChannelsEndpoint:
    """Tests for GET /api/channels (in-context)."""

    def test_get_channels_chunk(self, client):
        chunk_result = json.dumps(
            {
                "chunk_idx": 0,
                "total_chunks": 3,
                "chunk_size": 50,
                "channels": [{"name": "ch1", "description": "test"}],
            }
        )
        with patch(
            "osprey.services.channel_finder.mcp.in_context.tools.get_channels.get_channels"
        ) as mock_ch:
            mock_ch.fn.return_value = chunk_result
            resp = client.get("/api/channels?chunk_idx=0&chunk_size=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_idx"] == 0
        assert len(data["channels"]) == 1


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
    """Tests for CRUD endpoints with mocked database_crud functions."""

    def test_ic_create_channel(self, client):
        with (
            patch(
                "osprey.interfaces.channel_finder.database_crud.ic_add_channel",
                return_value={"success": True, "channel": "TEST:CH"},
            ),
            patch(
                "osprey.interfaces.channel_finder.database_crud._reload_registry",
            ),
        ):
            # Mock _get_db_path
            with patch(
                "osprey.interfaces.channel_finder.database_api._get_db_path",
                return_value="/tmp/test.json",
            ):
                resp = client.post(
                    "/api/channels",
                    json={"channel_name": "TEST:CH", "address": "TEST:CH"},
                )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_ic_delete_channel(self, client):
        with (
            patch(
                "osprey.interfaces.channel_finder.database_crud.ic_delete_channel",
                return_value={"success": True, "channel": "TEST:CH"},
            ),
            patch(
                "osprey.interfaces.channel_finder.database_api._get_db_path",
                return_value="/tmp/test.json",
            ),
        ):
            resp = client.delete("/api/channels/TEST:CH")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_ic_update_channel(self, client):
        with (
            patch(
                "osprey.interfaces.channel_finder.database_crud.ic_update_channel",
                return_value={"success": True, "channel": "TEST:CH"},
            ),
            patch(
                "osprey.interfaces.channel_finder.database_api._get_db_path",
                return_value="/tmp/test.json",
            ),
        ):
            resp = client.put(
                "/api/channels/TEST:CH",
                json={"description": "Updated"},
            )
        assert resp.status_code == 200

    def test_ic_create_crud_error_returns_400(self, client):
        from osprey.interfaces.channel_finder.database_crud import CrudError

        with (
            patch(
                "osprey.interfaces.channel_finder.database_crud.ic_add_channel",
                side_effect=CrudError("Channel already exists", "duplicate"),
            ),
            patch(
                "osprey.interfaces.channel_finder.database_api._get_db_path",
                return_value="/tmp/test.json",
            ),
        ):
            resp = client.post(
                "/api/channels",
                json={"channel_name": "TEST:CH"},
            )
        assert resp.status_code == 400

    def test_ic_create_unexpected_error_returns_500(self, client):
        with (
            patch(
                "osprey.interfaces.channel_finder.database_crud.ic_add_channel",
                side_effect=RuntimeError("Unexpected"),
            ),
            patch(
                "osprey.interfaces.channel_finder.database_api._get_db_path",
                return_value="/tmp/test.json",
            ),
        ):
            resp = client.post(
                "/api/channels",
                json={"channel_name": "TEST:CH"},
            )
        assert resp.status_code == 500
