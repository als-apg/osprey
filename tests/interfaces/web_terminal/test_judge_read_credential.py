"""A watchdog judge reads its applications' routes through the terminal's door.

Each judge run carries its own read credential: admitted on the safe-method
agent routes of exactly the applications whose tool servers the run is given,
refused (``403``) everywhere else, and unknown (``401``) once the run has ended.
The proxy forwards an admitted call exactly as it forwards the agent's.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from osprey.interfaces.web_auth import JudgeGrants, WebCredentials
from osprey.utils.owner_header import CALLER_HEADER
from tests.interfaces._asgi import OPERATOR_SECRET, PANEL_TOKEN

from ._proxy_fakes import panel_app

pytestmark = pytest.mark.no_auth_seam

APP = "kb"
PANEL = "kb-panel"
OTHER_APP = "other"
OTHER_PANEL = "other-panel"
READ_PATH = f"/panel/{PANEL}/api/search"
POST_PATH = f"/panel/{PANEL}/api/draft"
OTHER_READ_PATH = f"/panel/{OTHER_PANEL}/api/state"


def _app_panel(panel_id: str, app: str, port: int, routes: list[tuple[str, str]]) -> dict:
    return {
        "id": panel_id,
        "label": panel_id.upper(),
        "url": f"http://127.0.0.1:{port}",
        "configDefined": True,
        "app": app,
        "agentRoutes": [{"method": method, "path": path} for method, path in routes],
    }


@pytest.fixture
def terminal(workspace_dir):
    """The real terminal, serving a fixture app with a GET and a POST agent route."""
    custom = [
        _app_panel(PANEL, APP, 9310, [("GET", "/api/search"), ("POST", "/api/draft")]),
        _app_panel(OTHER_PANEL, OTHER_APP, 9311, [("GET", "/api/state")]),
    ]
    for app, _client in panel_app(workspace_dir, custom):
        app.state.config_path = None
        app.state.web_credentials = WebCredentials(
            operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
        )
        yield app


@pytest.fixture
def forwarded(terminal) -> list[tuple[str, str, dict[str, str]]]:
    """Stub the upstream client; collect each forwarded ``(method, url, headers)``."""
    sent: list[tuple[str, str, dict[str, str]]] = []

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        sent.append((method, url, dict(headers)))
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    terminal.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return sent


def _client(app) -> TestClient:
    """A client on the running app; not entered, so the lifespan is not run a second time."""
    return TestClient(app, client=("127.0.0.1", 54321))


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_the_terminal_holds_judge_grants_over_its_own_registry(terminal):
    grants = terminal.state.judge_grants
    assert isinstance(grants, JudgeGrants)
    assert grants.panels_serving([APP]) == frozenset({PANEL})


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", READ_PATH, 200),
        ("POST", POST_PATH, 403),
        ("GET", OTHER_READ_PATH, 403),
        ("GET", "/api/panels", 403),
        ("POST", "/api/panel-op/open", 403),
        ("GET", "/api/config", 403),
        ("GET", f"/panel/{PANEL}/index.html", 403),
    ],
)
def test_a_judge_token_reaches_only_its_apps_read_routes(terminal, forwarded, method, path, status):
    token = terminal.state.judge_grants.mint([PANEL])
    client = _client(terminal)
    resp = client.request(method, path, headers=_bearer(token), json=None)
    assert resp.status_code == status
    assert bool(forwarded) is (status == 200)


def test_a_revoked_judge_token_is_unknown(terminal, forwarded):
    grants = terminal.state.judge_grants
    token = grants.mint([PANEL])
    grants.revoke(token)
    client = _client(terminal)
    resp = client.get(READ_PATH, headers=_bearer(token))
    assert resp.status_code == 401
    assert forwarded == []


def test_a_judge_token_opens_no_websocket(terminal):
    token = terminal.state.judge_grants.mint([PANEL])
    client = _client(terminal)
    with pytest.raises(WebSocketDenialResponse) as refused:
        with client.websocket_connect(f"/panel/{PANEL}/ws", headers=_bearer(token)):
            pass
    assert refused.value.status_code == 403


def test_an_admitted_judge_call_is_forwarded_as_the_agents(terminal, forwarded):
    token = terminal.state.judge_grants.mint([PANEL])
    client = _client(terminal)
    assert client.get(READ_PATH, headers=_bearer(token)).status_code == 200
    ((_method, url, headers),) = forwarded
    assert url.endswith("/api/search")
    callers = [v for k, v in headers.items() if k.lower() == CALLER_HEADER.lower()]
    assert callers == ["agent"]
