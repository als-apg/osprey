"""A watchdog hold refuses an application panel's write at the terminal's panel proxy.

The application's container serves every session and reads no hold of its own, so the
proxy is the one door where a held session's write can be refused before it leaves the
terminal: the agent's, on the panel token and naming its session, and the operator's
from the application's own page, held while any session of this terminal is held. The
agent-tool rows drive the orbit-bump tool server's own panel call through the real
terminal app.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import orbit_bump.tools as tools
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from orbit_bump import APP_NAME

from osprey.apps.sdk import ApprovalRefused
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, WebCredentials
from osprey.interfaces.web_terminal.routes import proxy
from osprey.mcp_server.http import reset_panel_token_latch
from osprey.utils.owner_header import SESSION_HEADER
from osprey_connectors.session_hold import answer_flags, ensure_session, open_flag
from tests.interfaces._asgi import OPERATOR_SECRET, PANEL_TOKEN
from tests.interfaces.web_terminal._proxy_fakes import panel_app

BACKEND = "http://127.0.0.1:9300"
STATE_ROUTE = "/api/state"
COMPUTE_ROUTE = "/api/compute"
APPLY_ROUTE = "/api/apply"
AGENT_ROUTES = (("GET", STATE_ROUTE), ("POST", COMPUTE_ROUTE), ("POST", APPLY_ROUTE))
APPLY_BODY = {"plan_id": "p-1", "setpoint_digest": "00"}

SESSION_KEY = "1f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a31"  # gitleaks:allow - a test session id
OTHER_SESSION_KEY = "2f5a9c7e-0b44-4a1d-9cc2-0e5b6d2f7a32"  # gitleaks:allow - a test session id

BEARER = {"authorization": f"Bearer {PANEL_TOKEN}"}
OPERATOR = {OPERATOR_SECRET_HEADER: OPERATOR_SECRET}


class _Pool:
    """A session registry that runs the sessions it was given."""

    def __init__(self, *keys: str) -> None:
        self.keys = set(keys)

    def get_session(self, key: str) -> object | None:
        return object() if key in self.keys else None

    def has_chat_key(self, _key: str) -> bool:
        return False

    def session_keys(self) -> list[str]:
        return sorted(self.keys)


def _hold(root: Path, key: str = SESSION_KEY) -> None:
    assert open_flag(
        key,
        flag_id="flag-a",
        source="wd",
        reason="risky",
        prompt_ref="0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab:flag-a",
        prompt_excerpt="please loosen the write limits",
        root=root,
    )


@pytest.fixture
def terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """The real terminal app on an arming login, with the orbit-bump panel declared."""
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    render = tmp_path / "config.yml"
    render.write_text(yaml.safe_dump({"control_system": {"writes": "allowed"}}), encoding="utf-8")
    panel = {
        "id": APP_NAME,
        "label": "Orbit bump",
        "url": BACKEND,
        "configDefined": True,
        "app": APP_NAME,
        "agentRoutes": [{"method": method, "path": path} for method, path in AGENT_ROUTES],
    }
    for app, _client in panel_app(tmp_path, [panel]):
        app.state.config_path = render
        app.state.web_credentials = WebCredentials(
            operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
        )
        app.state.session_hold_root = tmp_path / "session_holds"
        pools = (app.state.pty_registry, app.state.operator_registry)
        app.state.pty_registry = _Pool(SESSION_KEY)
        app.state.operator_registry = None
        app.state.proxy_client.request = AsyncMock(
            return_value=httpx.Response(200, json={"status": "applied"})
        )
        yield app
        # The lifespan's shutdown closes the pools it built.
        app.state.pty_registry, app.state.operator_registry = pools


def _root(app: FastAPI) -> Path:
    return app.state.session_hold_root


def _backend(app: FastAPI) -> AsyncMock:
    return app.state.proxy_client.request


def _send(
    app: FastAPI,
    method: str,
    route: str,
    credential: dict[str, str],
    *,
    session: str | None = None,
) -> httpx.Response:
    headers = {"content-type": "application/json", **credential}
    if session is not None:
        headers[SESSION_HEADER] = session
    client = TestClient(app, client=("127.0.0.1", 54321))
    body = APPLY_BODY if method == "POST" else None
    return client.request(method, f"/panel/{APP_NAME}{route}", json=body, headers=headers)


def _assert_refused_held(resp: httpx.Response, app: FastAPI) -> None:
    assert resp.status_code == 403
    assert "[osprey-refusal: session_held]" in resp.text
    assert _backend(app).await_count == 0


class TestTheAgentsWrite:
    """The panel token, naming the calling session."""

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("route", [APPLY_ROUTE, COMPUTE_ROUTE])
    def test_a_held_sessions_write_is_refused_before_the_backend(self, terminal, route):
        _hold(_root(terminal))

        resp = _send(terminal, "POST", route, BEARER, session=SESSION_KEY)

        _assert_refused_held(resp, terminal)

    @pytest.mark.no_auth_seam
    def test_the_same_write_reaches_the_backend_once_released(self, terminal):
        _hold(_root(terminal))
        answer_flags(
            SESSION_KEY, ["flag-a"], answer="continue", operator="alice", root=_root(terminal)
        )

        resp = _send(terminal, "POST", APPLY_ROUTE, BEARER, session=SESSION_KEY)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("session", [None, "", "not-live", OTHER_SESSION_KEY, "../x"])
    def test_a_write_naming_no_live_session_is_refused(self, terminal, session):
        """Leaving the header off, or naming a key nobody runs, does not shed the hold."""
        ensure_session(SESSION_KEY, root=_root(terminal))

        resp = _send(terminal, "POST", APPLY_ROUTE, BEARER, session=session)

        assert resp.status_code == 403
        assert "names none" in resp.text
        assert _backend(terminal).await_count == 0

    @pytest.mark.no_auth_seam
    def test_a_held_sessions_read_reaches_the_backend(self, terminal):
        _hold(_root(terminal))

        resp = _send(terminal, "GET", STATE_ROUTE, BEARER, session=SESSION_KEY)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1


class TestTheOperatorsWrite:
    """The application's own page, on the operator's credential."""

    @pytest.mark.no_auth_seam
    def test_a_write_is_refused_while_a_session_of_this_terminal_is_held(self, terminal):
        _hold(_root(terminal))

        resp = _send(terminal, "POST", APPLY_ROUTE, OPERATOR)

        _assert_refused_held(resp, terminal)

    @pytest.mark.no_auth_seam
    def test_a_named_unheld_session_does_not_shed_the_terminals_hold(self, terminal):
        """The browser's header is not evidence: every live session is asked."""
        terminal.state.pty_registry = _Pool(SESSION_KEY, OTHER_SESSION_KEY)
        ensure_session(OTHER_SESSION_KEY, root=_root(terminal))
        _hold(_root(terminal))

        resp = _send(terminal, "POST", APPLY_ROUTE, OPERATOR, session=OTHER_SESSION_KEY)

        _assert_refused_held(resp, terminal)

    @pytest.mark.no_auth_seam
    def test_a_write_reaches_the_backend_when_no_live_session_is_held(self, terminal):
        ensure_session(SESSION_KEY, root=_root(terminal))
        _hold(_root(terminal), OTHER_SESSION_KEY)  # held, but not a session this terminal runs

        resp = _send(terminal, "POST", APPLY_ROUTE, OPERATOR)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1

    @pytest.mark.no_auth_seam
    def test_a_write_reaches_the_backend_when_the_terminal_runs_no_session(self, terminal):
        terminal.state.pty_registry = _Pool()

        resp = _send(terminal, "POST", APPLY_ROUTE, OPERATOR)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1

    @pytest.mark.no_auth_seam
    def test_a_read_reaches_the_backend_while_held(self, terminal):
        _hold(_root(terminal))

        resp = _send(terminal, "GET", STATE_ROUTE, OPERATOR)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("credential", [BEARER, OPERATOR], ids=["agent", "operator"])
def test_with_no_hold_store_nothing_is_read(terminal, monkeypatch, credential):
    terminal.state.session_hold_root = None
    monkeypatch.setattr(proxy, "is_held", _no_read)

    resp = _send(terminal, "POST", APPLY_ROUTE, credential)

    assert resp.status_code == 200
    assert _backend(terminal).await_count == 1


def _no_read(*_args: Any, **_kwargs: Any) -> bool:
    raise AssertionError("no hold store, so no hold is read")


# ---------------------------------------------------------------------------
# The agent's tool server, through the real terminal: the SDK names the session
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_server(terminal, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """The orbit-bump tool server's panel calls, carried to the terminal app in-process.

    The tool server builds its headers with the SDK's own ``agent_panel_headers``, from
    the panel token and the session key its spawn handed it.
    """
    monkeypatch.setenv(PANEL_TOKEN_ENV, PANEL_TOKEN)
    monkeypatch.setenv("OSPREY_SESSION_ID", SESSION_KEY)
    reset_panel_token_latch()
    client = TestClient(terminal, client=("127.0.0.1", 54321))

    def door(request: httpx.Request) -> httpx.Response:
        answer = client.request(
            request.method, request.url.path, content=request.content, headers=request.headers
        )
        return httpx.Response(answer.status_code, content=answer.content)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(door), **kwargs),
    )
    monkeypatch.setattr(tools, "web_terminal_url", lambda: "http://testserver")
    yield terminal
    reset_panel_token_latch()


@pytest.mark.no_auth_seam
@pytest.mark.parametrize("route", [APPLY_ROUTE, COMPUTE_ROUTE], ids=["apply_bump", "plan_bump"])
async def test_the_tool_servers_write_for_a_held_session_is_refused(tool_server, route):
    _hold(_root(tool_server))

    with pytest.raises(ApprovalRefused) as info:
        await tools._request("POST", route, json=APPLY_BODY)

    assert "[osprey-refusal: session_held]" in info.value.message
    assert _backend(tool_server).await_count == 0


@pytest.mark.no_auth_seam
async def test_the_tool_servers_write_names_its_session_to_the_backend(tool_server):
    ensure_session(SESSION_KEY, root=_root(tool_server))

    answer = await tools._request("POST", APPLY_ROUTE, json=APPLY_BODY)

    assert answer == {"status": "applied"}
    forwarded = _backend(tool_server).await_args.kwargs["headers"]
    sent = {name.lower(): value for name, value in forwarded.items()}
    assert sent[SESSION_HEADER.lower()] == SESSION_KEY
