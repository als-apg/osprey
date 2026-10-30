"""Tests for server-side prefix-awareness of panel URL producers.

Multi-user deployments mount each user's Web Terminal container at
``/u/<user>/`` behind a shared nginx front door (see ``compute_url_prefix()``
in ``osprey.interfaces.web_terminal.app``). Every panel URL the server hands
to the browser — the seven ``*_server_config`` endpoints, the custom panel URLs in
``GET /api/panels``, the ``panel_register`` broadcast, and the
``panel_focus`` broadcast's optional ``url`` — must be prefixed with that same
constant so iframes and SSE-driven navigation resolve inside the user's own
mount. An empty prefix (no ``OSPREY_TERMINAL_USER``) must reproduce the
pre-prefix root-absolute paths byte-for-byte, and an already-absolute URL
(``http://``, ``https://``, protocol-relative ``//``) must pass through
unprefixed so an external panel URL is never corrupted.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app

_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"


def _make_app(**state) -> FastAPI:
    """A bare FastAPI app with only the panels router and the given app.state."""
    return bare_route_app(router, **state)


#: ``(OSPREY_TERMINAL_USER, the prefix it bakes in)``: under a user, and unset.
_PREFIXES = [
    pytest.param("alice", "/u/alice", id="under_user"),
    pytest.param(None, "", id="empty_prefix"),
]


def _set_user(monkeypatch, user: str | None) -> None:
    if user is None:
        monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    else:
        monkeypatch.setenv("OSPREY_TERMINAL_USER", user)


# ---- The *_server_config endpoints ----


@pytest.mark.parametrize(
    ("path", "state_key", "panel_id"),
    [
        ("/api/artifact-server", "artifact_server_url", "artifacts"),
        ("/api/ariel-server", "ariel_server_url", "ariel"),
        ("/api/channel-finder-server", "channel_finder_server_url", "channel-finder"),
        ("/api/lattice-server", "lattice_dashboard_server_url", "lattice"),
        ("/api/okf-server", "okf_server_url", "okf"),
        ("/api/jupyter-server", "jupyter_server_url", "jupyter"),
        ("/api/system-health-server", "system_health_server_url", "system-health"),
    ],
)
class TestServerConfigEndpointsPrefix:
    @pytest.mark.parametrize(("user", "prefix"), _PREFIXES)
    def test_url_carries_the_prefix(self, monkeypatch, path, state_key, panel_id, user, prefix):
        _set_user(monkeypatch, user)
        app = _make_app(**{state_key: "http://127.0.0.1:9000"})
        client = TestClient(app)

        resp = client.get(path)

        assert resp.status_code == 200
        assert resp.json()["url"] == f"{prefix}/panel/{panel_id}"


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


# ---- POST /api/panels/register broadcast + response ----


class TestRegisterPanelPrefix:
    def _client(self):
        app = _make_app(
            allow_runtime_panels=True,
            custom_panels=[],
            visible_panels=[],
            runtime_panel_allowlist=None,
        )
        return TestClient(app)

    @pytest.mark.parametrize(("user", "prefix"), _PREFIXES)
    def test_register_response_and_broadcast_carry_the_prefix(self, monkeypatch, user, prefix):
        _set_user(monkeypatch, user)
        client = self._client()

        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
            )

        assert resp.status_code == 200
        assert resp.json()["url"] == f"{prefix}/panel/grafana"
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == f"{prefix}/panel/grafana"


# ---- POST /api/panel-focus broadcast ----


class TestSetPanelFocusPrefix:
    def _client(self):
        app = _make_app(
            enabled_panels={"ariel"},
            custom_panels=[],
            active_panel=None,
        )
        return TestClient(app)

    @pytest.mark.parametrize(
        ("user", "url", "broadcast"),
        [
            # A root-absolute URL is prefixed under a user ...
            pytest.param("alice", "/panel/ariel", "/u/alice/panel/ariel", id="root_absolute"),
            # ... and an already-absolute one is never touched.
            pytest.param(
                "alice",
                "https://grafana.lan:3000/d/abc",
                "https://grafana.lan:3000/d/abc",
                id="absolute",
            ),
            pytest.param("alice", "//evil.example/x", "//evil.example/x", id="protocol_relative"),
            pytest.param(None, "/panel/ariel", "/panel/ariel", id="root_absolute_empty_prefix"),
        ],
    )
    def test_broadcast_url(self, monkeypatch, user, url, broadcast):
        _set_user(monkeypatch, user)
        client = self._client()

        resp = client.post(
            "/api/panel-focus",
            json={"panel": "ariel", "url": url, "source": "agent"},
        )

        assert resp.status_code == 200
        event = client.app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == broadcast
