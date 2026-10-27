"""Application panels on the panel proxy: declared agent routes and write arming.

An application panel is a config-defined panel whose ``web.panels.<id>`` block
names the application it serves (``app``). Two facts ride on that block:

* ``agent_routes`` — the exact ``{method, path}`` pairs (backend-root-relative)
  the terminal admits on the panel token, composed as ``/panel/<id><path>``;
* ``arm_writes`` — whether this login arms the application. Missing reads as
  false. On a login that does not arm it, the panel hop forwards only safe
  methods and refuses every other method and every websocket upgrade, naming
  the login and the application, before anything leaves the terminal.

A panel without ``app`` — a built-in panel, or a config-defined panel of the
non-application shape — keeps every method and its sockets.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces import web_auth
from osprey.interfaces.web_auth import WebCredentials
from osprey.interfaces.web_terminal.app import _load_panel_config
from osprey.interfaces.web_terminal.routes.proxy import (
    _PANEL_STATE_MAP,
    configured_agent_routes,
)
from tests.interfaces.web_terminal._proxy_fakes import _FakeConnect, _patch_connect, panel_app
from tests.interfaces.web_terminal._started_app import started_client

#: The application the panel serves, and the panel id it is published under.
APP_NAME = "knob"
PANEL_ID = "knob"

#: The declared read route the approval hook fetches (``state_path``).
STATE_PATH = "/api/state"

#: A write path on the same backend — never declared on an unarmed render.
WRITE_PATH = "/api/set"

#: One built-in panel, taken from the registry-derived map rather than named.
BUILTIN_PANEL_ID, BUILTIN_STATE_ATTR = sorted(_PANEL_STATE_MAP.items())[0]

PANEL_TOKEN = "panel-token-value"
OPERATOR_SECRET = "operator-secret-value"


def _app_panel(*, arm_writes: bool | None, panel_id: str = PANEL_ID) -> dict:
    """A config-defined application panel as ``_load_panel_config`` forwards it."""
    panel = {
        "id": panel_id,
        "label": panel_id.upper(),
        "url": "http://127.0.0.1:9300",
        "configDefined": True,
        "app": APP_NAME,
        "agentRoutes": [{"method": "GET", "path": STATE_PATH}],
    }
    if arm_writes is not None:
        panel["armWrites"] = arm_writes
    return panel


#: A config-defined panel of the non-application shape: no ``app``, no arming.
PLAIN_PANEL = {
    "id": "plain",
    "label": "PLAIN",
    "url": "http://127.0.0.1:9301",
    "configDefined": True,
}


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _boot(workspace_dir, *, arm_writes):
    gen = panel_app(workspace_dir, [_app_panel(arm_writes=arm_writes), dict(PLAIN_PANEL)])
    for app, client in gen:
        setattr(app.state, BUILTIN_STATE_ATTR, "http://127.0.0.1:9200")
        yield app, client


@pytest.fixture
def unarmed(workspace_dir):
    """An app whose application panel declares no ``arm_writes`` (read as false)."""
    yield from _boot(workspace_dir, arm_writes=None)


@pytest.fixture
def armed(workspace_dir):
    """An app on a login that arms the application."""
    yield from _boot(workspace_dir, arm_writes=True)


def _capture(app) -> list[tuple[str, str]]:
    """Stub the upstream HTTP client; return the ``(method, url)`` pairs it was sent."""
    sent: list[tuple[str, str]] = []

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        sent.append((method, url))
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return sent


# ---------------------------------------------------------------------------
# The panel spec the terminal reads
# ---------------------------------------------------------------------------


def _load(panels: dict) -> list[dict]:
    with patch(
        "osprey_connectors.workspace.load_osprey_config",
        return_value={"web": {"panels": panels}},
    ):
        _enabled, custom, _default = _load_panel_config()
    return custom


class TestPanelSpec:
    def test_application_fields_are_forwarded(self):
        (panel,) = _load(
            {
                PANEL_ID: {
                    "url": "http://127.0.0.1:9300",
                    "app": APP_NAME,
                    "arm_writes": True,
                    "agent_routes": [
                        {"method": "get", "path": STATE_PATH},
                        {"method": "POST", "path": WRITE_PATH},
                    ],
                }
            }
        )

        assert panel["app"] == APP_NAME
        assert panel["armWrites"] is True
        assert panel["agentRoutes"] == [
            {"method": "GET", "path": STATE_PATH},
            {"method": "POST", "path": WRITE_PATH},
        ]

    def test_missing_fields_read_as_unarmed_and_undeclared(self):
        (panel,) = _load({"plain": {"url": "http://127.0.0.1:9301"}})

        assert panel["app"] is None
        assert panel["armWrites"] is False
        assert panel["agentRoutes"] == []

    @pytest.mark.parametrize(
        "routes",
        [
            "GET /api/state",
            [{"method": "GET"}],
            [{"path": STATE_PATH}],
            [{"method": "GET", "path": "api/state"}],
            [{"method": "GET", "path": "/panel/knob/api/state?x=1"}],
            [{"method": "GET", "path": STATE_PATH, "extra": 1}],
        ],
    )
    def test_malformed_agent_routes_refuse_at_boot(self, routes):
        with pytest.raises(ValueError, match=f"web.panels.{PANEL_ID}.agent_routes"):
            _load({PANEL_ID: {"url": "http://127.0.0.1:9300", "agent_routes": routes}})


# ---------------------------------------------------------------------------
# Declared agent routes on the panel credential
# ---------------------------------------------------------------------------


class TestAgentRoutes:
    def test_composed_under_the_panel_prefix(self):
        routes = configured_agent_routes(
            [
                _app_panel(arm_writes=False),
                {
                    **_app_panel(arm_writes=False, panel_id="rooted"),
                    "agentRoutes": [{"method": "GET", "path": "/"}],
                },
            ]
        )

        assert routes == {("GET", f"/panel/{PANEL_ID}{STATE_PATH}"), ("GET", "/panel/rooted/")}

    def test_only_config_defined_loopback_panels_declare(self):
        declared = {"agentRoutes": [{"method": "GET", "path": STATE_PATH}]}
        routes = configured_agent_routes(
            [
                {"id": "registered", "url": "http://127.0.0.1:9001", **declared},
                {
                    "id": "facility",
                    "url": "http://grafana.facility.lan:3000",
                    "configDefined": True,
                    **declared,
                },
            ]
        )

        assert routes == frozenset()

    def test_startup_admits_the_hook_state_url(self, workspace_dir):
        """Booted over config, the hook's exact P5 URL classifies as panel-tier."""
        web = {
            "panels": {
                PANEL_ID: {
                    "url": "http://127.0.0.1:9300",
                    "app": APP_NAME,
                    "agent_routes": [{"method": "GET", "path": STATE_PATH}],
                },
                "plain": {"url": "http://127.0.0.1:9301"},
            }
        }
        with started_client(workspace_dir, web=web):
            hook_url = f"/panel/{PANEL_ID}{STATE_PATH}"
            assert web_auth.classify("GET", hook_url, False) is web_auth.Tier.PANEL
            assert web_auth.classify("POST", hook_url, False) is web_auth.Tier.OPERATOR
            assert web_auth.classify("GET", "/panel/plain/", False) is web_auth.Tier.OPERATOR

    def test_unconfigured_panel_stays_operator_tier(self, workspace_dir):
        with started_client(
            workspace_dir, web={"panels": {"plain": {"url": "http://127.0.0.1:1"}}}
        ):
            assert (
                web_auth.classify("GET", f"/panel/{PANEL_ID}{STATE_PATH}", False)
                is web_auth.Tier.OPERATOR
            )

    @pytest.mark.no_auth_seam
    def test_panel_token_reaches_the_declared_route_only(self, unarmed):
        app, _client = unarmed
        app.state.web_credentials = WebCredentials(
            operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
        )
        sent = _capture(app)
        bearer = {"authorization": f"Bearer {PANEL_TOKEN}"}
        client = TestClient(app, client=("127.0.0.1", 54321))

        declared = client.get(f"/panel/{PANEL_ID}{STATE_PATH}", headers=bearer)
        undeclared = client.get(f"/panel/{PANEL_ID}/api/other", headers=bearer)

        assert declared.status_code == 200
        assert undeclared.status_code == 401
        assert sent == [("GET", f"http://127.0.0.1:9300{STATE_PATH}")]


# ---------------------------------------------------------------------------
# Write arming on the HTTP hop
# ---------------------------------------------------------------------------

WRITE_METHODS = ["POST", "PUT", "PATCH", "DELETE"]


class TestUnarmedHttp:
    @pytest.mark.parametrize("method", WRITE_METHODS)
    @pytest.mark.parametrize("path", [f"/panel/{PANEL_ID}{WRITE_PATH}", f"/panel/{PANEL_ID}"])
    def test_write_method_refused(self, unarmed, method, path):
        app, client = unarmed
        sent = _capture(app)

        resp = client.request(method, path)

        assert resp.status_code == 403
        assert APP_NAME in resp.text
        assert "this login" in resp.text
        assert sent == []

    def test_refusal_names_the_login(self, unarmed):
        app, client = unarmed
        app.state.terminal_user = "alice"
        _capture(app)

        resp = client.post(f"/panel/{PANEL_ID}{WRITE_PATH}")

        assert resp.status_code == 403
        assert "'alice'" in resp.text
        assert f"'{APP_NAME}'" in resp.text

    def test_unarmed_explicitly_is_refused_too(self, workspace_dir):
        for app, client in panel_app(workspace_dir, [_app_panel(arm_writes=False)]):
            sent = _capture(app)
            assert client.post(f"/panel/{PANEL_ID}{WRITE_PATH}").status_code == 403
            assert sent == []

    @pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
    def test_safe_methods_forwarded(self, unarmed, method):
        app, client = unarmed
        sent = _capture(app)

        resp = client.request(method, f"/panel/{PANEL_ID}{STATE_PATH}")

        assert resp.status_code == 200
        assert sent == [(method, f"http://127.0.0.1:9300{STATE_PATH}")]

    @pytest.mark.parametrize("method", WRITE_METHODS)
    def test_non_application_panel_unaffected(self, unarmed, method):
        app, client = unarmed
        sent = _capture(app)

        resp = client.request(method, "/panel/plain/api/thing")

        assert resp.status_code == 200
        assert sent == [(method, "http://127.0.0.1:9301/api/thing")]

    def test_builtin_panel_unaffected(self, unarmed):
        app, client = unarmed
        sent = _capture(app)

        resp = client.post(f"/panel/{BUILTIN_PANEL_ID}/api/contents")

        assert resp.status_code == 200
        assert sent == [("POST", "http://127.0.0.1:9200/api/contents")]


class TestArmedHttp:
    @pytest.mark.parametrize("method", WRITE_METHODS)
    def test_write_method_forwarded(self, armed, method):
        app, client = armed
        sent = _capture(app)

        resp = client.request(method, f"/panel/{PANEL_ID}{WRITE_PATH}")

        assert resp.status_code == 200
        assert sent == [(method, f"http://127.0.0.1:9300{WRITE_PATH}")]


# ---------------------------------------------------------------------------
# Websocket upgrades
# ---------------------------------------------------------------------------


def _open(client, path) -> _FakeConnect:
    fake = _FakeConnect()
    with _patch_connect(fake):
        with client.websocket_connect(path):
            pass
    return fake


class TestWebsocket:
    def test_unarmed_application_socket_refused(self, unarmed):
        app, client = unarmed
        app.state.terminal_user = "alice"
        fake = _FakeConnect()

        with _patch_connect(fake):
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect(f"/panel/{PANEL_ID}/ws"):
                    pass

        assert closed.value.code == 4003
        assert "'alice'" in closed.value.reason
        assert f"'{APP_NAME}'" in closed.value.reason
        assert fake.target is None

    def test_armed_application_socket_forwarded(self, armed):
        _app, client = armed

        fake = _open(client, f"/panel/{PANEL_ID}/ws")

        assert fake.target == "ws://127.0.0.1:9300/ws"

    def test_non_application_socket_unaffected(self, unarmed):
        _app, client = unarmed

        fake = _open(client, "/panel/plain/ws")

        assert fake.target == "ws://127.0.0.1:9301/ws"

    def test_builtin_socket_unaffected(self, unarmed):
        _app, client = unarmed

        fake = _open(client, f"/panel/{BUILTIN_PANEL_ID}/api/kernels/k/channels")

        assert fake.target == "ws://127.0.0.1:9200/api/kernels/k/channels"
