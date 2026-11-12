"""A watchdog judge reads its applications' routes through the terminal's door.

Each judge run carries its own read credential: admitted on the safe-method
agent routes of exactly the applications whose tool servers the run is given,
refused (``403``) everywhere else, and unknown (``401``) once the run has ended.
The proxy forwards an admitted call exactly as it forwards the agent's.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from osprey.agent_runner import ResultEvent
from osprey.agent_runner import launcher as agent_launcher
from osprey.apps.projection import APP_NAME_ENV
from osprey.apps.sdk import agent_panel_headers
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, JudgeGrants, WebCredentials
from osprey.mcp_server import http as mcp_http
from osprey.services.watchdog.config import WatchdogDefinition
from osprey.services.watchdog.engines import agent as agent_module
from osprey.services.watchdog.engines.agent import AgentEngine
from osprey.services.watchdog.history import SessionNodes
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
ANSWER = json.dumps({"flagged": False, "explanation": "read the knowledge"})


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


# --------------------------------------------------------------------------- #
# A judge run, end to end
# --------------------------------------------------------------------------- #


@pytest.fixture
def build_dir(tmp_path: Path, monkeypatch) -> Path:
    """A built project with two applications' tool servers beside the controls server."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("control_system:\n  writes: allowed\n")
    (build / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "controls": {"command": "controls-mcp"},
                    APP: {"command": "kb-mcp", "env": {APP_NAME_ENV: APP}},
                    OTHER_APP: {"command": "other-mcp", "env": {APP_NAME_ENV: OTHER_APP}},
                }
            }
        )
    )
    monkeypatch.setattr(
        agent_launcher, "project_agent_cli_path", lambda _dir: "/opt/bin/claude", raising=True
    )
    monkeypatch.setattr(
        "osprey.agent_runner.primitives.sdk_env", lambda *a, **k: {"ANTHROPIC_API_KEY": "k"}
    )
    monkeypatch.setattr(
        "osprey.agent_runner.primitives.resolve_default_model", lambda *a, **k: "provider-default"
    )
    monkeypatch.setattr(
        "osprey.agent_runner.primitives._resolve_project_spec", lambda *a, **k: None
    )
    return build


def _result() -> ResultEvent:
    return ResultEvent(
        subtype="success",
        is_error=False,
        num_turns=1,
        duration_ms=1,
        session_id="judge-run",
        total_cost_usd=0.0,
        usage=None,
        result=ANSWER,
        api_error_status=None,
    )


def test_a_judge_runs_tool_call_reads_through_the_door_and_cannot_post(
    terminal, forwarded, build_dir, monkeypatch
):
    """The app tool server's own header recipe, on the run's env, reaches GET and not POST."""
    monkeypatch.setattr(mcp_http, "_PANEL_TOKEN_LATCH", None)
    statuses: dict[str, int] = {}
    seen: dict[str, str] = {}

    def stream(_project_dir, _prompt, **kwargs):
        server_env = kwargs["mcp_servers"][APP]["env"]
        seen["token"] = server_env[PANEL_TOKEN_ENV]
        # What the app's tool server does on a call: its SDK builds the header
        # from the environment the run handed it.
        monkeypatch.setenv(PANEL_TOKEN_ENV, server_env[PANEL_TOKEN_ENV])
        headers = agent_panel_headers()
        client = _client(terminal)
        statuses["get"] = client.get(READ_PATH, headers=headers).status_code
        statuses["post"] = client.post(POST_PATH, headers=headers, json={}).status_code

        async def events():
            yield _result()

        return events()

    monkeypatch.setattr(agent_module, "stream_query", stream)
    defn = WatchdogDefinition(
        name="w",
        engine="agent",
        instructions="read the knowledge",
        tools=(f"mcp__{APP}__search",),
    )
    engine = AgentEngine(build_dir, judge_grants=terminal.state.judge_grants)
    verdict = asyncio.run(
        engine.judge(
            defn,
            SessionNodes(),
            "raise the current",
            deadline=time.monotonic() + 60,
            record_session_id="rec",
            submission_id="sub",
        )
    )

    assert verdict.status == "ok"
    assert statuses == {"get": 200, "post": 403}
    assert [method for method, _url, _headers in forwarded] == ["GET"]
    # The run is over: its credential no longer opens anything.
    assert terminal.state.judge_grants.panels_of(seen["token"]) is None
