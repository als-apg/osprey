"""Tests for Channel Finder FastAPI app factory."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from tests.interfaces.channel_finder.graph_fixture import FakeGraphContext

_APP_LOGGER = "osprey.interfaces.channel_finder.app"
_IC_INIT = "osprey.mcp_server.channel_finder_in_context.server_context.initialize_cf_ic_context"


def _start_app(config):
    """Run the app's lifespan against ``config`` and return the started app."""
    with patch("osprey.utils.workspace.load_osprey_config", return_value=config):
        from osprey.interfaces.channel_finder.app import create_app

        application = create_app(project_cwd="/tmp/test-project")
        with TestClient(application):
            return application


def _records(caplog, level):
    return [r for r in caplog.records if r.name == _APP_LOGGER and r.levelno == level]


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "channel-finder"

    def test_health_includes_pipeline_type(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["pipeline_type"] == "in_context"


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


class TestPipelineResolution:
    """Tests for how the app resolves and reports the active pipeline."""

    def test_missing_pipeline_mode_warns_naming_the_key(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            app = _start_app({"channel_finder": {}})

        warnings = [r.getMessage() for r in _records(caplog, logging.WARNING)]
        assert any("channel_finder.pipeline_mode" in m for m in warnings), warnings
        # Nothing to detect either, so no paradigm is invented.
        assert app.state.pipeline_type is None

    def test_configured_pipeline_mode_is_not_warned_about(self, caplog, mock_config, mock_registry):
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            app = _start_app(mock_config)

        warnings = [r.getMessage() for r in _records(caplog, logging.WARNING)]
        assert not any("channel_finder.pipeline_mode" in m for m in warnings), warnings
        assert app.state.pipeline_type == "in_context"

    def test_absent_pipeline_block_skips_at_debug_without_warning(
        self, caplog, mock_config, mock_registry
    ):
        # mock_config configures in_context only.
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            app = _start_app(mock_config)

        warnings = [r.getMessage() for r in _records(caplog, logging.WARNING)]
        for absent in ("hierarchical", "middle_layer"):
            assert not any(absent in m for m in warnings), (absent, warnings)

        debug = [r.getMessage() for r in _records(caplog, logging.DEBUG)]
        assert any("channel_finder.pipelines.hierarchical" in m for m in debug), debug
        assert any("channel_finder.pipelines.middle_layer" in m for m in debug), debug
        assert app.state.available_pipelines == ["in_context"]

    def test_configured_pipeline_that_fails_warns_with_traceback(self, caplog, mock_config):
        with (
            patch(_IC_INIT, side_effect=RuntimeError("database file is missing")),
            caplog.at_level(logging.DEBUG, logger=_APP_LOGGER),
        ):
            app = _start_app(mock_config)

        failures = [r for r in _records(caplog, logging.WARNING) if "in_context" in r.getMessage()]
        assert failures, [r.getMessage() for r in _records(caplog, logging.WARNING)]
        assert failures[0].exc_info is not None
        assert app.state.available_pipelines == []

    def test_explicit_pipeline_mode_is_served_from_its_own_registry(self, caplog):
        """The paradigm detection resolves is the one whose registry is served."""
        config = {
            "channel_finder": {
                "pipeline_mode": "in_context",
                "pipelines": {"in_context": {"database": {"path": "/tmp/test_db.json"}}},
            },
        }
        registry = MagicMock()
        registry.database = MagicMock()
        registry.facility_name = "TEST"
        with (
            patch(_IC_INIT, return_value=registry),
            caplog.at_level(logging.DEBUG, logger=_APP_LOGGER),
        ):
            app = _start_app(config)

        assert app.state.pipeline_type == "in_context"
        assert app.state.databases["in_context"] is registry.database
        assert app.state.facility_names["in_context"] == "TEST"


class TestGraphParadigmState:
    """The graph paradigm is served from the mode, without a channel database."""

    @staticmethod
    def _graph_config(pipelines=None, graphdb=None):
        graphdb_block = {"uri": "bolt://localhost:7687"}
        if graphdb is not None:
            graphdb_block.update(graphdb)
        return {
            "channel_finder": {"pipeline_mode": "graph", "pipelines": pipelines},
            "services": {"graphdb": graphdb_block},
        }

    @staticmethod
    def _fake_context(monkeypatch, error=None):
        """Serve a fake store context from the lifespan's factory.

        The real context reads config and resolves a connection; none of that
        is what the lifespan is on the hook for, so the tests hand the app the
        shared fake and assert only that it is stored and closed.

        Args:
            monkeypatch: The pytest fixture the factory is replaced through.
            error: Raise this instead of returning a context, standing in for a
                store whose ``initialize()`` failed.

        Returns:
            The context the lifespan will be handed.
        """
        from osprey.interfaces.channel_finder import app as app_module

        context = FakeGraphContext()

        def _make():
            if error is not None:
                raise error
            return context

        monkeypatch.setattr(app_module, "_make_graph_context", _make)
        return context

    def test_graph_mode_is_served_with_no_database(self, caplog):
        # A graph render writes the ``pipelines`` key with nothing under it.
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            app = _start_app(self._graph_config())

        assert app.state.pipeline_type == "graph"
        assert app.state.graph_backed is True
        assert app.state.available_pipelines == ["graph"]
        assert app.state.databases == {}

    def test_graph_mode_says_so_at_info_naming_its_tools(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            _start_app(self._graph_config())

        infos = [r.getMessage() for r in _records(caplog, logging.INFO)]
        assert any("graph" in m and "read_cypher" in m for m in infos), infos

    def test_graph_mode_warns_about_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            _start_app(self._graph_config())

        # No missing pipeline_mode, no absent-block chatter, no failed registry.
        assert [r.getMessage() for r in _records(caplog, logging.WARNING)] == []

    def test_graph_mode_never_falls_back_to_a_file_paradigm(self, caplog):
        """A stray in_context block does not displace the resolved paradigm."""
        config = self._graph_config(
            {"in_context": {"database": {"path": "/tmp/test_db.json"}}},
        )
        registry = MagicMock()
        registry.database = MagicMock()
        registry.facility_name = "TEST"
        with (
            patch(_IC_INIT, return_value=registry) as init_mock,
            caplog.at_level(logging.DEBUG, logger=_APP_LOGGER),
        ):
            app = _start_app(config)

        assert app.state.pipeline_type == "graph"
        assert app.state.available_pipelines == ["graph"]
        assert app.state.databases == {}
        init_mock.assert_not_called()

    def test_graph_mode_puts_the_store_context_on_state(self, monkeypatch):
        context = self._fake_context(monkeypatch)
        app = _start_app(self._graph_config())
        assert app.state.graph_context is context

    def test_the_store_context_is_shut_down_with_the_app(self, monkeypatch):
        context = self._fake_context(monkeypatch)
        _start_app(self._graph_config())
        assert context.shutdowns == 1

    def test_a_store_that_fails_to_come_up_leaves_no_context(self, monkeypatch, caplog):
        """A store that is down is a 503 at the routes, not a dead app."""
        from osprey.mcp_server.graph.server_context import GraphUnreachable

        self._fake_context(monkeypatch, error=GraphUnreachable("store did not answer"))
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            app = _start_app(self._graph_config())

        assert app.state.pipeline_type == "graph"
        assert getattr(app.state, "graph_context", None) is None
        warnings = [r.getMessage() for r in _records(caplog, logging.WARNING)]
        assert any("graph" in m.lower() for m in warnings), warnings

    def test_the_seeded_corpus_is_named_by_its_filename(self, monkeypatch):
        self._fake_context(monkeypatch)
        app = _start_app(self._graph_config(graphdb={"ttl_path": "corpora/als/facility.ttl"}))
        assert app.state.graph_ttl_filename == "facility.ttl"

    def test_no_configured_corpus_names_no_file(self, monkeypatch):
        self._fake_context(monkeypatch)
        app = _start_app(self._graph_config())
        assert app.state.graph_ttl_filename is None

    def test_an_absent_graphdb_block_names_no_file(self, monkeypatch):
        self._fake_context(monkeypatch)
        app = _start_app({"channel_finder": {"pipeline_mode": "graph", "pipelines": None}})
        assert app.state.graph_ttl_filename is None

    def test_a_malformed_graphdb_block_names_no_file_and_still_serves(self, monkeypatch, caplog):
        self._fake_context(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=_APP_LOGGER):
            app = _start_app(self._graph_config(graphdb={"port_host": "not-a-port"}))

        assert app.state.pipeline_type == "graph"
        assert app.state.graph_ttl_filename is None
        warnings = [r.getMessage() for r in _records(caplog, logging.WARNING)]
        assert any("graphdb" in m for m in warnings), warnings

    def test_file_backed_mode_reports_not_graph_backed(self, mock_config, mock_registry):
        app = _start_app(mock_config)
        assert app.state.graph_backed is False

    def test_graph_app_answers_its_routes_from_the_state_it_started_with(self):
        """End to end: the routes read what a real graph lifespan put on state."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value=self._graph_config(),
        ):
            from osprey.interfaces.channel_finder.app import create_app

            with TestClient(create_app(project_cwd="/tmp/test-project")) as c:
                assert c.get("/health").json()["pipeline_type"] == "graph"

                info = c.get("/api/info").json()
                assert info["pipeline_type"] == "graph"
                assert info["graph_backed"] is True
                assert info["db_path"] is None
                assert "read_cypher" in info["tools"]

                # Statistics is served from the store, and no store answers in
                # a test process: the route reports that rather than pretending
                # the paradigm has no implementation.
                stats = c.get("/api/statistics")
                assert stats.status_code == 503
                assert stats.json()["error_type"]
                assert stats.json()["suggestions"]

                assert c.post("/api/validate", json={"channels": []}).status_code == 501
                switched = c.put("/api/pipeline", json={"pipeline_type": "in_context"})
                assert switched.status_code == 400
                assert "read_cypher" in switched.json()["detail"]

                # Every explorer gate is closed, one route per gate.
                assert c.get("/api/explore/options?level=system").status_code == 404
                assert c.get("/api/explore/systems").status_code == 404
                assert c.get("/api/channels").status_code == 404
