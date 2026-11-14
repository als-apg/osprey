"""The terminal proxy tells an application backend who is calling.

Every request and WebSocket upgrade the proxy forwards to a panel carries
``X-Osprey-Caller``, stamped by the proxy from what it admitted: ``agent`` when
the terminal's gate admitted the request on the panel token, ``panel`` for
everything else. An inbound value never crosses the hop, in either spelling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_auth import WebCredentials
from osprey.interfaces.web_terminal.routes.proxy import _is_stripped_header
from osprey.utils.owner_header import CALLER_HEADER
from tests.interfaces._asgi import OPERATOR_SECRET, PANEL_TOKEN

from ._proxy_fakes import _FakeConnect, _lower, _patch_connect, panel_app

APP_PANEL = "knob"
STATE_PATH = "/api/state"
PLAIN_PANEL = "plain"

CALLER_WIRE_NAME = CALLER_HEADER.lower()
BEARER = {"authorization": f"Bearer {PANEL_TOKEN}"}


@pytest.fixture
def app_and_client(workspace_dir):
    """An application panel declaring one agent route, beside a plain panel."""
    custom = [
        {
            "id": APP_PANEL,
            "label": "KNOB",
            "url": "http://127.0.0.1:9300",
            "configDefined": True,
            "app": APP_PANEL,
            "agentRoutes": [{"method": "GET", "path": STATE_PATH}],
        },
        {
            "id": PLAIN_PANEL,
            "label": "PLAIN",
            "url": "http://127.0.0.1:9301",
            "configDefined": True,
        },
    ]
    for app, client in panel_app(workspace_dir, custom):
        app.state.config_path = None
        app.state.web_credentials = WebCredentials(
            operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
        )
        yield app, client


def _capture(app) -> list[dict[str, str]]:
    """Stub the upstream HTTP client; return the header sets it was sent."""
    sent: list[dict[str, str]] = []

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        sent.append(dict(headers))
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return sent


def _callers(headers: dict[str, str]) -> list[str]:
    return [v for k, v in headers.items() if k.lower().replace("_", "-") == CALLER_WIRE_NAME]


def test_the_caller_header_is_stripped_in_both_spellings():
    assert _is_stripped_header(CALLER_HEADER)
    assert _is_stripped_header("X_Osprey_Caller")


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("forged", ["X-Osprey-Caller", "X_Osprey_Caller"])
def test_a_browser_claiming_agent_arrives_as_panel(app_and_client, forged):
    app, _client = app_and_client
    sent = _capture(app)
    client = TestClient(app, client=("127.0.0.1", 54321))

    resp = client.get(
        f"/panel/{APP_PANEL}{STATE_PATH}",
        headers={"X-Osprey-Terminal-Secret": OPERATOR_SECRET, forged: "agent"},
    )

    assert resp.status_code == 200
    (headers,) = sent
    assert _callers(headers) == ["panel"]


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("forged", [None, "panel"])
def test_a_panel_token_call_on_an_agent_route_arrives_as_agent(app_and_client, forged):
    app, _client = app_and_client
    sent = _capture(app)
    client = TestClient(app, client=("127.0.0.1", 54321))
    headers = dict(BEARER)
    if forged is not None:
        headers[CALLER_HEADER] = forged

    resp = client.get(f"/panel/{APP_PANEL}{STATE_PATH}", headers=headers)

    assert resp.status_code == 200
    (forwarded,) = sent
    assert _callers(forwarded) == ["agent"]


def test_a_plain_panel_request_carries_panel(app_and_client):
    app, client = app_and_client
    sent = _capture(app)

    assert client.get(f"/panel/{PLAIN_PANEL}/api/thing").status_code == 200

    (headers,) = sent
    assert _callers(headers) == ["panel"]


def test_the_websocket_upgrade_carries_panel(app_and_client):
    _app, client = app_and_client
    fake = _FakeConnect()

    with _patch_connect(fake):
        with client.websocket_connect(f"/panel/{PLAIN_PANEL}/ws", headers={CALLER_HEADER: "agent"}):
            pass

    assert _lower(fake.kwargs["additional_headers"] or {})[CALLER_WIRE_NAME] == "panel"
