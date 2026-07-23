"""Tests for Channel Finder database_api pipeline-specific routes.

Complements ``test_database_api.py`` (in-context happy paths + 404 gating) by
covering the hierarchical and middle-layer route bodies, the runtime pipeline
switch, JSON-parameter validation (422), chunk bounds, and the
database-unavailable (503) path — the branches those routes gate behind a
non-default ``pipeline_type``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

_DB_PATCH = "osprey.interfaces.channel_finder.database_api._get_database"
_FACILITY_PATCH = "osprey.interfaces.channel_finder.database_api._get_facility_name"


def _set_pipeline(client, pt: str) -> None:
    client.app.state.pipeline_type = pt


class TestSwitchPipeline:
    def test_switch_to_available_pipeline(self, client):
        client.app.state.available_pipelines = ["in_context", "hierarchical"]
        resp = client.put("/api/pipeline", json={"pipeline_type": "hierarchical"})
        assert resp.status_code == 200
        assert resp.json()["pipeline_type"] == "hierarchical"
        assert client.app.state.pipeline_type == "hierarchical"

    def test_switch_to_unavailable_pipeline_400(self, client):
        client.app.state.available_pipelines = ["in_context"]
        resp = client.put("/api/pipeline", json={"pipeline_type": "middle_layer"})
        assert resp.status_code == 400


class TestDatabaseUnavailable:
    def test_statistics_503_when_no_database(self, client):
        _set_pipeline(client, "in_context")
        client.app.state.databases = {}
        resp = client.get("/api/statistics")
        assert resp.status_code == 503


class TestInfoMetadata:
    def test_hierarchical_metadata(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.db_path = "/tmp/h.json"
        mock_db.hierarchy_levels = ["system", "device"]
        mock_db.hierarchy_config = {"system": {}}
        mock_db.naming_pattern = "{system}:{device}"
        with (
            patch(_DB_PATCH, return_value=mock_db),
            patch(_FACILITY_PATCH, return_value="HIER"),
        ):
            resp = client.get("/api/info")
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["hierarchy_levels"] == ["system", "device"]
        assert meta["facility_name"] == "HIER"

    def test_middle_layer_metadata_counts_systems(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.db_path = "/tmp/ml.json"
        mock_db.list_systems.return_value = ["SR", "BR", "LN"]
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/info")
        assert resp.status_code == 200
        assert resp.json()["metadata"]["system_count"] == 3


class TestStatisticsNonInContext:
    def test_middle_layer_statistics_passthrough(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.get_statistics.return_value = {"families": 12}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/statistics")
        assert resp.status_code == 200
        # Non-in-context branch returns the raw stats without chunk augmentation.
        assert resp.json() == {"families": 12}
        mock_db.chunk_database.assert_not_called()


class TestValidateNonInContext:
    def test_hierarchical_validate_per_channel(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.validate_channel.side_effect = lambda ch: ch != "BAD"
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/validate", json={"channels": ["OK1", "BAD", "OK2"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid_count"] == 2
        assert data["invalid_count"] == 1
        assert data["total"] == 3


class TestHierarchicalExplore:
    def test_explore_options_success(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.get_options_at_level.return_value = ["SR", "BR"]
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get('/api/explore/options?level=system&selections={"a":"b"}')
        assert resp.status_code == 200
        data = resp.json()
        assert data["options"] == ["SR", "BR"]
        assert data["total"] == 2
        mock_db.get_options_at_level.assert_called_once_with("system", {"a": "b"})

    def test_explore_options_invalid_json_422(self, client):
        _set_pipeline(client, "hierarchical")
        with patch(_DB_PATCH, return_value=MagicMock()):
            resp = client.get("/api/explore/options?level=system&selections={bad")
        assert resp.status_code == 422

    def test_explore_build_partitions_valid_invalid(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.build_channels_from_selections.return_value = ["A", "B"]
        mock_db.validate_channel.side_effect = lambda ch: ch == "A"
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get('/api/explore/build?selections={"x":"y"}')
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] == ["A"]
        assert data["invalid"] == ["B"]

    def test_explore_build_invalid_json_422(self, client):
        _set_pipeline(client, "hierarchical")
        with patch(_DB_PATCH, return_value=MagicMock()):
            resp = client.get("/api/explore/build?selections={bad")
        assert resp.status_code == 422

    def test_hierarchy_info(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.hierarchy_levels = ["system"]
        mock_db.hierarchy_config = {}
        mock_db.naming_pattern = "{system}"
        with (
            patch(_DB_PATCH, return_value=mock_db),
            patch(_FACILITY_PATCH, return_value="ALS"),
        ):
            resp = client.get("/api/explore/hierarchy-info")
        assert resp.status_code == 200
        assert resp.json()["facility_name"] == "ALS"


class TestHierarchicalCrud:
    def test_add_tree_node_success(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.add_node.return_value = {"status": "added"}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post(
                "/api/tree/node",
                json={"level": "system", "name": "SR", "description": "ring"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "added"

    def test_add_tree_node_write_error_400(self, client):
        _set_pipeline(client, "hierarchical")
        from osprey.services.channel_finder.core.base_database import DatabaseWriteError

        mock_db = MagicMock()
        mock_db.add_node.side_effect = DatabaseWriteError("duplicate", "dup")
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/tree/node", json={"level": "system", "name": "SR"})
        assert resp.status_code == 400

    def test_add_tree_node_unexpected_error_500(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.add_node.side_effect = RuntimeError("boom")
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/tree/node", json={"level": "system", "name": "SR"})
        assert resp.status_code == 500

    def test_tree_impact_reports_breakdown(self, client):
        _set_pipeline(client, "hierarchical")
        mock_db = MagicMock()
        mock_db.count_descendants.return_value = {"channels": 40, "devices": 5}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/tree/impact", json={"level": "system", "name": "SR"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["affected_channels"] == 40
        assert data["breakdown"] == {"devices": 5}

    def test_get_tree_expansion_invalid_json_422(self, client):
        _set_pipeline(client, "hierarchical")
        with patch(_DB_PATCH, return_value=MagicMock()):
            resp = client.get("/api/tree/expansion?level=device&selections={bad")
        assert resp.status_code == 422


class TestMiddleLayerExplore:
    def test_explore_systems(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.list_systems.return_value = ["SR", "BR"]
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/explore/systems")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_explore_families(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.list_families.return_value = ["BPM", "HCM"]
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/explore/families?system=SR")
        assert resp.status_code == 200
        assert resp.json()["families"] == ["BPM", "HCM"]

    def test_explore_fields(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.inspect_fields.return_value = {"Monitor": {}}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/explore/fields?system=SR&family=BPM")
        assert resp.status_code == 200
        assert resp.json()["fields"] == {"Monitor": {}}

    def test_explore_channels_success(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.list_channel_names.return_value = ["SR:BPM:01:X"]
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get(
                "/api/explore/channels?system=SR&family=BPM&field=Monitor&sectors=[1,2]"
            )
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        mock_db.list_channel_names.assert_called_once_with(
            "SR", "BPM", "Monitor", None, [1, 2], None
        )

    def test_explore_channels_invalid_json_422(self, client):
        _set_pipeline(client, "middle_layer")
        with patch(_DB_PATCH, return_value=MagicMock()):
            resp = client.get(
                "/api/explore/channels?system=SR&family=BPM&field=Monitor&sectors=[bad"
            )
        assert resp.status_code == 422

    def test_explore_device_info(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.get_device_info.return_value = {"count": 8}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/explore/device-info?system=SR&family=BPM")
        assert resp.status_code == 200
        assert resp.json() == {"count": 8}


class TestMiddleLayerCrud:
    def test_add_family_success(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.add_family.return_value = {"status": "added"}
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post(
                "/api/structure/family",
                json={"system": "SR", "family": "BPM", "description": "monitors"},
            )
        assert resp.status_code == 200

    def test_add_family_write_error_400(self, client):
        _set_pipeline(client, "middle_layer")
        from osprey.services.channel_finder.core.base_database import DatabaseWriteError

        mock_db = MagicMock()
        mock_db.add_family.side_effect = DatabaseWriteError("dup", "dup")
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/structure/family", json={"system": "SR", "family": "BPM"})
        assert resp.status_code == 400

    def test_structure_impact(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.count_family_channels.return_value = 24
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.post("/api/structure/impact", json={"system": "SR", "family": "BPM"})
        assert resp.status_code == 200
        assert resp.json()["affected_channels"] == 24


class TestInContextChunkBounds:
    def test_chunk_idx_out_of_range_422(self, client):
        _set_pipeline(client, "in_context")
        mock_db = MagicMock()
        mock_db.chunk_database.return_value = []  # zero chunks
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/channels?chunk_idx=0")
        assert resp.status_code == 422
