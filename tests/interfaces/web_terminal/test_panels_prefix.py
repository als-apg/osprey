"""Tests for server-side prefix-awareness of panel URL producers.

Multi-user deployments mount each user's Web Terminal container at
``/u/<user>/`` behind a shared nginx front door (see ``compute_url_prefix()``
in ``osprey.interfaces.web_terminal.app``). Every panel URL the server hands
to the browser — the five ``*_server_config`` endpoints, ``_browser_panel_url``
(and therefore ``GET /api/panels``), the ``panel_register`` broadcast, and the
``panel_focus`` broadcast's optional ``url`` — must be prefixed with that same
constant so iframes and SSE-driven navigation resolve inside the user's own
mount. An empty prefix (no ``OSPREY_TERMINAL_USER``) must reproduce the
pre-prefix root-absolute paths byte-for-byte, and an already-absolute URL
(``http://``, ``https://``, protocol-relative ``//``) must pass through
unprefixed so an external panel URL is never corrupted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import _browser_panel_url, router

_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"


def _make_app(**state) -> FastAPI:
    """A bare FastAPI app with only the panels router and the given app.state."""
    app = FastAPI()
    app.include_router(router)
    for key, value in state.items():
        setattr(app.state, key, value)
    return app


# ---- _browser_panel_url ----


class TestBrowserPanelUrlPrefix:
    def test_url_backed_panel_prefixed(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        assert _browser_panel_url({"id": "grafana"}) == "/u/alice/panel/grafana"

    def test_discovered_panel_prefixed(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        cp = {"id": "demo", "discovered": True, "url": "/panel-static/demo/"}
        assert _browser_panel_url(cp) == "/u/alice/panel-static/demo/"

    def test_discovered_panel_missing_url_falls_back_prefixed(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        cp = {"id": "demo", "discovered": True}
        assert _browser_panel_url(cp) == "/u/alice/panel-static/demo/"

    def test_url_backed_panel_empty_prefix_unchanged(self, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        assert _browser_panel_url({"id": "grafana"}) == "/panel/grafana"

    def test_discovered_panel_empty_prefix_unchanged(self, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        cp = {"id": "demo", "discovered": True, "url": "/panel-static/demo/"}
        assert _browser_panel_url(cp) == "/panel-static/demo/"


# ---- Five *_server_config endpoints ----


@pytest.mark.parametrize(
    ("path", "state_key", "panel_id"),
    [
        ("/api/artifact-server", "artifact_server_url", "artifacts"),
        ("/api/ariel-server", "ariel_server_url", "ariel"),
        ("/api/channel-finder-server", "channel_finder_server_url", "channel-finder"),
        ("/api/lattice-server", "lattice_dashboard_server_url", "lattice"),
        ("/api/okf-server", "okf_server_url", "okf"),
    ],
)
class TestServerConfigEndpointsPrefix:
    def test_prefixed_under_user(self, monkeypatch, path, state_key, panel_id):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        app = _make_app(**{state_key: "http://127.0.0.1:9000"})
        client = TestClient(app)

        resp = client.get(path)

        assert resp.status_code == 200
        assert resp.json()["url"] == f"/u/alice/panel/{panel_id}"

    def test_empty_prefix_unchanged(self, monkeypatch, path, state_key, panel_id):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        app = _make_app(**{state_key: "http://127.0.0.1:9000"})
        client = TestClient(app)

        resp = client.get(path)

        assert resp.status_code == 200
        assert resp.json()["url"] == f"/panel/{panel_id}"


# ---- GET /api/panels ----


class TestGetPanelsPrefix:
    def _client(self):
        app = _make_app(
            enabled_panels=set(),
            custom_panels=[
                {"id": "grafana", "label": "Grafana", "url": "http://grafana.lan:3000/"},
                {
                    "id": "demo",
                    "label": "Demo",
                    "url": "/panel-static/demo/",
                    "discovered": True,
                },
            ],
            default_panel=None,
            visible_panels=["grafana", "demo"],
            active_panel=None,
        )
        return TestClient(app)

    def test_custom_urls_prefixed_under_user(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        resp = self._client().get("/api/panels")

        assert resp.status_code == 200
        by_id = {cp["id"]: cp for cp in resp.json()["custom"]}
        assert by_id["grafana"]["url"] == "/u/alice/panel/grafana"
        assert by_id["demo"]["url"] == "/u/alice/panel-static/demo/"

    def test_custom_urls_empty_prefix_unchanged(self, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        resp = self._client().get("/api/panels")

        assert resp.status_code == 200
        by_id = {cp["id"]: cp for cp in resp.json()["custom"]}
        assert by_id["grafana"]["url"] == "/panel/grafana"
        assert by_id["demo"]["url"] == "/panel-static/demo/"


# ---- POST /api/panels/register broadcast + response ----


class TestRegisterPanelPrefix:
    def _client(self):
        app = _make_app(
            allow_runtime_panels=True,
            custom_panels=[],
            visible_panels=[],
            runtime_panel_allowlist=None,
            broadcaster=MagicMock(),
        )
        return TestClient(app)

    def test_register_response_and_broadcast_prefixed_under_user(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        client = self._client()

        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
            )

        assert resp.status_code == 200
        assert resp.json()["url"] == "/u/alice/panel/grafana"
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "/u/alice/panel/grafana"

    def test_register_response_and_broadcast_empty_prefix_unchanged(self, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        client = self._client()

        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
            )

        assert resp.status_code == 200
        assert resp.json()["url"] == "/panel/grafana"
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "/panel/grafana"


# ---- POST /api/panel-focus broadcast ----


class TestSetPanelFocusPrefix:
    def _client(self):
        app = _make_app(
            enabled_panels={"ariel"},
            custom_panels=[],
            active_panel=None,
            broadcaster=MagicMock(),
        )
        return TestClient(app)

    def test_root_absolute_url_prefixed_under_user(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        client = self._client()

        resp = client.post("/api/panel-focus", json={"panel": "ariel", "url": "/panel/ariel"})

        assert resp.status_code == 200
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "/u/alice/panel/ariel"

    def test_absolute_url_passed_through_unchanged(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        client = self._client()

        resp = client.post(
            "/api/panel-focus",
            json={"panel": "ariel", "url": "https://grafana.lan:3000/d/abc"},
        )

        assert resp.status_code == 200
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "https://grafana.lan:3000/d/abc"

    def test_protocol_relative_url_passed_through_unchanged(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        client = self._client()

        resp = client.post("/api/panel-focus", json={"panel": "ariel", "url": "//evil.example/x"})

        assert resp.status_code == 200
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "//evil.example/x"

    def test_root_absolute_url_empty_prefix_unchanged(self, monkeypatch):
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
        client = self._client()

        resp = client.post("/api/panel-focus", json={"panel": "ariel", "url": "/panel/ariel"})

        assert resp.status_code == 200
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "/panel/ariel"
