"""Tests for Channel Finder database_api pipeline-specific routes.

Complements ``test_database_api.py`` (in-context happy paths + 404 gating) by
covering the hierarchical and middle-layer route bodies, the runtime pipeline
switch, JSON-parameter validation (422), chunk bounds, and the
database-unavailable (503) path — the branches those routes gate behind a
non-default ``pipeline_type``.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from osprey.mcp_server.graph.server_context import GraphUnreachable
from tests.interfaces.channel_finder.graph_fixture import (
    DEMO_STATISTICS,
    DEMO_STORE_URI,
    demo_context,
    install_graph_paradigm,
)

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


class TestGraphParadigmRoutes:
    """The graph paradigm has no database file, and the routes say so plainly."""

    def test_info_reports_the_paradigm_and_its_tools(self, client):
        install_graph_paradigm(client)
        client.app.state.facility_name = "ALS"
        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_type"] == "graph"
        assert data["available_pipelines"] == ["graph"]
        assert data["graph_backed"] is True
        assert data["db_path"] is None
        assert "read_cypher" in data["tools"]
        assert "get_schema" in data["tools"]
        assert data["metadata"]["facility_name"] == "ALS"

    @staticmethod
    def _set_ttl_filename(client, name: str | None) -> None:
        """Set the seeded TTL basename on the app, or remove it entirely.

        A project whose config names no TTL leaves the attribute unset rather
        than ``None``, so "not configured" is tested by removing it.
        """
        if name is None:
            if hasattr(client.app.state, "graph_ttl_filename"):
                delattr(client.app.state, "graph_ttl_filename")
        else:
            client.app.state.graph_ttl_filename = name

    def test_info_names_where_the_store_lives(self, client):
        # The graph paradigm's answer to "which database am I looking at?" —
        # the file-backed payload's ``db_path``, told as a store URI.
        install_graph_paradigm(client, demo_context())
        self._set_ttl_filename(client, None)

        resp = client.get("/api/info")

        assert resp.status_code == 200
        assert resp.json()["graph_store"]["uri"] == DEMO_STORE_URI

    def test_info_names_the_seeded_corpus_when_one_is_configured(self, client):
        install_graph_paradigm(client, demo_context())
        self._set_ttl_filename(client, "facility.ttl")

        resp = client.get("/api/info")

        assert resp.status_code == 200
        assert resp.json()["graph_store"]["ttl_filename"] == "facility.ttl"

    def test_info_reports_no_corpus_when_none_is_configured(self, client):
        install_graph_paradigm(client, demo_context())
        self._set_ttl_filename(client, None)

        resp = client.get("/api/info")

        assert resp.status_code == 200
        # Reported as null rather than omitted: the panel distinguishes "no
        # corpus configured" from a payload it failed to understand.
        assert resp.json()["graph_store"]["ttl_filename"] is None

    def test_info_answers_even_when_there_is_no_store(self, client):
        # The panel must boot against a store that is down or was never
        # configured, so the absent context is a reportable value, not a 503.
        install_graph_paradigm(client)
        self._set_ttl_filename(client, None)

        resp = client.get("/api/info")

        assert resp.status_code == 200
        assert resp.json()["graph_store"] == {"uri": None, "ttl_filename": None}

    def test_info_omits_the_store_block_for_file_backed_paradigms(self, client):
        # ``graph_store`` is the graph paradigm's ``db_path``; a file-backed
        # payload already has the real one and must not grow a second answer.
        _set_pipeline(client, "in_context")
        client.app.state.graph_ttl_filename = "facility.ttl"
        mock_db = MagicMock()
        mock_db.db_path = "/tmp/ic.json"
        mock_db.get_statistics.return_value = {"total_channels": 3}
        mock_db.chunk_database.return_value = [[], []]
        with (
            patch(_DB_PATCH, return_value=mock_db),
            patch(_FACILITY_PATCH, return_value="ALS"),
        ):
            resp = client.get("/api/info")

        assert resp.status_code == 200
        assert "graph_store" not in resp.json()

    def test_info_marks_file_backed_paradigms_as_not_graph_backed(self, client):
        _set_pipeline(client, "middle_layer")
        mock_db = MagicMock()
        mock_db.db_path = "/tmp/ml.json"
        mock_db.list_systems.return_value = []
        with patch(_DB_PATCH, return_value=mock_db):
            resp = client.get("/api/info")
        assert resp.status_code == 200
        assert resp.json()["graph_backed"] is False

    def test_statistics_counts_the_store(self, client):
        install_graph_paradigm(client, demo_context())

        resp = client.get("/api/statistics")

        assert resp.status_code == 200
        assert resp.json() == {
            "total_devices": DEMO_STATISTICS["devices"],
            "total_channels": DEMO_STATISTICS["channels"],
            # The class count is the taxonomy the explorer draws, not the raw
            # class tree: the store's non-device classes are pruned out first.
            "total_classes": DEMO_STATISTICS["classes"],
            "total_signals": DEMO_STATISTICS["signals"],
            "total_sections": DEMO_STATISTICS["sections"],
        }

    def test_statistics_asks_the_store_once_per_population(self, client):
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        client.get("/api/statistics")

        # Four censuses plus the class tree, each asked exactly once.
        assert len(ctx.calls) == 5
        assert len({cypher for cypher, _ in ctx.calls}) == 5
        # Every read carries an explicit bound rather than the store's default.
        assert all(max_rows is not None for _, max_rows in ctx.calls)
        # Statistics never needs to tell an empty store from a broken one: a
        # store that answers zero has answered.
        assert ctx.empty_checks == 0

    def test_statistics_reads_run_off_the_event_loop(self, client):
        # The store's driver is synchronous; awaiting it inline would stall
        # every other request the app is serving.
        ctx = demo_context()
        install_graph_paradigm(client, ctx)

        client.get("/api/statistics")

        assert ctx.saw_running_loop == [False] * 5

    def test_statistics_503_when_the_store_is_unreachable(self, client):
        install_graph_paradigm(
            client,
            demo_context(
                raises=GraphUnreachable(
                    "Graph store at bolt://localhost:7687 is unreachable.",
                    ["Start the graphdb service."],
                )
            ),
        )

        resp = client.get("/api/statistics")

        assert resp.status_code == 503
        body = resp.json()
        # Not nested under FastAPI's "detail" envelope: the web UI reads all
        # three keys off the body it is handed.
        assert "unreachable" in body["detail"]
        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"] == ["Start the graphdb service."]

    def test_statistics_503_when_the_app_has_no_store_at_all(self, client):
        install_graph_paradigm(client)

        resp = client.get("/api/statistics")

        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "service_unavailable"
        assert "graphdb" in " ".join(body["suggestions"])

    def test_validate_501_naming_the_graph_tools(self, client):
        install_graph_paradigm(client)
        resp = client.post("/api/validate", json={"channels": ["SR:BPM:01:X"]})
        assert resp.status_code == 501
        detail = resp.json()["detail"]
        assert "read_cypher" in detail
        assert "get_schema" in detail

    def test_switch_pipeline_400_names_the_paradigm_and_read_cypher(self, client):
        install_graph_paradigm(client)
        resp = client.put("/api/pipeline", json={"pipeline_type": "in_context"})
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "graph paradigm" in detail
        assert "read_cypher" in detail
        # Not the generic "not available, available: []" message.
        assert "Available:" not in detail
        assert client.app.state.pipeline_type == "graph"

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            # One route per pipeline gate: hierarchical, middle_layer,
            # in_context, plus the two write gates that share those checks.
            ("get", "/api/explore/options?level=system", None),
            ("get", "/api/explore/systems", None),
            ("get", "/api/channels", None),
            ("post", "/api/tree/node", {"level": "system", "name": "SR"}),
            ("post", "/api/structure/family", {"system": "SR", "family": "BPM"}),
        ],
    )
    def test_explorer_routes_404_under_graph(self, client, method, path, body):
        install_graph_paradigm(client)
        kwargs = {"json": body} if body is not None else {}
        resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code == 404


class TestGraphImportClosure:
    """What the web app pays to import its own routes.

    The graph routes share their Cypher with tooling that is far heavier than a
    web app — the benchmark harness runs an agent, the seeder opens a driver.
    Sharing a constant must not drag either of those into the app's startup, so
    the closure is asserted in a fresh interpreter rather than in this one,
    where the test session has already imported half the framework.
    """

    def test_importing_the_routes_loads_neither_the_agent_sdk_nor_a_driver(self) -> None:
        source = (
            "import sys\n"
            "from osprey.interfaces.channel_finder import database_api\n"
            "assert database_api.GRAPH_CHANNEL_COUNT_CYPHER\n"
            "assert 'claude_agent_sdk' not in sys.modules, 'agent SDK in the web import closure'\n"
            "assert 'neo4j' not in sys.modules, 'neo4j imported at module scope'\n"
        )
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
