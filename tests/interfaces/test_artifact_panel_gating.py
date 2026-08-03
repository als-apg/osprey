"""The WORKSPACE tab must follow ``artifact_server.auto_launch``.

Panel availability is computed from ``app.state.artifact_server_url`` alone, so
publishing a URL for a server that was never started hands the operator an
enabled tab whose iframe returns a bare 502. These tests pin the honest
behaviour: the URL is set only when the artifact server is actually going to be
launched, and nothing — not a disabled auto-launch, not a config-load failure —
leaves a hardcoded fallback URL behind.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal.app import _launch_artifact_server
from osprey.interfaces.web_terminal.routes import panels
from osprey.utils import workspace


@pytest.fixture
def config(monkeypatch) -> dict:
    """A mutable config seen by both the launcher and the web terminal."""
    cfg: dict = {}
    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: cfg)
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: cfg)
    # The port resolver honours this override; an inherited value would decide
    # the port out from under the config these tests set.
    monkeypatch.delenv("OSPREY_ARTIFACT_SERVER_PORT", raising=False)
    return cfg


@pytest.fixture
def ensure_artifact_server(monkeypatch) -> MagicMock:
    """Stub out the real launch so no server is started."""
    mock = MagicMock()
    monkeypatch.setattr(server_launcher, "ensure_artifact_server", mock)
    return mock


def _stub_app() -> SimpleNamespace:
    """A stand-in for the FastAPI app — ``_launch_artifact_server`` only uses ``.state``."""
    return SimpleNamespace(state=SimpleNamespace())


def _artifact_url(app) -> str | None:
    return getattr(app.state, "artifact_server_url", None)


class TestLaunchGating:
    def test_auto_launch_false_leaves_url_unset(self, config, ensure_artifact_server):
        config["artifact_server"] = {"auto_launch": False, "port": 8086}
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) is None
        ensure_artifact_server.assert_not_called()

    def test_auto_launch_true_sets_url_and_launches(self, config, ensure_artifact_server):
        config["artifact_server"] = {"auto_launch": True, "host": "127.0.0.1", "port": 8099}
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) == "http://127.0.0.1:8099"
        ensure_artifact_server.assert_called_once_with()

    def test_absent_section_still_launches(self, config, ensure_artifact_server):
        """``auto_launch`` defaults to on — omitting the section must not disable the tab."""
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) == "http://127.0.0.1:8086"
        ensure_artifact_server.assert_called_once_with()

    def test_port_env_override_wins(self, config, monkeypatch, ensure_artifact_server):
        """The published URL comes from the shared resolver, env override included."""
        config["artifact_server"] = {"auto_launch": True, "port": 8086}
        monkeypatch.setenv("OSPREY_ARTIFACT_SERVER_PORT", "9291")
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) == "http://127.0.0.1:9291"

    def test_empty_port_env_override_falls_back_to_config(
        self, config, monkeypatch, ensure_artifact_server
    ):
        """An exported-but-empty override must not kill the launch (compose `VAR=`)."""
        config["artifact_server"] = {"auto_launch": True, "port": 8086}
        monkeypatch.setenv("OSPREY_ARTIFACT_SERVER_PORT", "")
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) == "http://127.0.0.1:8086"

    def test_config_failure_leaves_no_fallback_url(self, monkeypatch, ensure_artifact_server):
        def _boom() -> dict:
            raise RuntimeError("config.yml is unreadable")

        monkeypatch.setattr(server_launcher, "load_osprey_config", _boom)
        monkeypatch.setattr(workspace, "load_osprey_config", _boom)
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) is None
        ensure_artifact_server.assert_not_called()

    def test_launch_failure_retracts_the_url(self, config, monkeypatch):
        """A URL already assigned when the launch blows up must not survive."""
        config["artifact_server"] = {"auto_launch": True}
        monkeypatch.setattr(
            server_launcher,
            "ensure_artifact_server",
            MagicMock(side_effect=OSError("port unusable")),
        )
        app = _stub_app()

        _launch_artifact_server(app)

        assert _artifact_url(app) is None


class TestPanelEndpoint:
    """``/api/artifact-server`` is what disables the tab in the front end."""

    @staticmethod
    def _api() -> FastAPI:
        api = FastAPI()
        api.include_router(panels.router)
        return api

    def test_reports_unavailable_when_auto_launch_is_false(self, config, ensure_artifact_server):
        config["artifact_server"] = {"auto_launch": False}
        api = self._api()
        _launch_artifact_server(api)

        body = TestClient(api).get("/api/artifact-server").json()

        assert body == {"url": None, "available": False}

    def test_reports_available_when_auto_launch_is_on(self, config, ensure_artifact_server):
        config["artifact_server"] = {"auto_launch": True}
        api = self._api()
        _launch_artifact_server(api)

        body = TestClient(api).get("/api/artifact-server").json()

        assert body["available"] is True
        assert body["url"] == "/panel/artifacts"
