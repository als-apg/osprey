"""A watchdog hold refuses an application panel's write at the terminal's panel proxy.

The application's container serves every session and reads no hold of its own, so the
proxy is the one door where a held session's write can be refused before it leaves the
terminal: the agent's, on the panel token and naming its session, and the operator's
from the application's own page, held while any session of this terminal is held. The
agent-tool rows drive the scaffold application's tool server's own panel call through
the real terminal app.

The websocket hop asks the same rule at every frame the browser sends: the upgrade and
the application's own frames pass as a read does, and a frame sent under a hold closes
the socket with the refusal, whether the hold began before or after the socket opened.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import osprey.apps
from osprey.apps.sdk import ApprovalRefused
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, WebCredentials
from osprey.interfaces.web_terminal.routes import proxy
from osprey.mcp_server.http import reset_panel_token_latch
from osprey.utils.owner_header import SESSION_HEADER
from osprey_connectors.session_hold import answer_flags, ensure_session, open_flag
from tests.interfaces._asgi import OPERATOR_SECRET, PANEL_TOKEN
from tests.interfaces.web_terminal._proxy_fakes import _FakeConnect, _patch_connect, panel_app

BACKEND = "http://127.0.0.1:9300"
STATE_ROUTE = "/api/state"
#: A second POST route the panel declares: the proxy's rule is by method, not by route.
RESET_ROUTE = "/api/reset"
WRITE_ROUTE = "/api/write"
AGENT_ROUTES = (("GET", STATE_ROUTE), ("POST", RESET_ROUTE), ("POST", WRITE_ROUTE))
WRITE_BODY = {"value": 1.0, "target": "simulation", "readback": 0.0}

#: The scaffold application, whose tool server the agent-tool rows drive.
TEMPLATE = Path(osprey.apps.__file__).resolve().parent / "scaffold-app"
APP_NAME = "scaffold-app"

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
    """The real terminal app on an arming login, with the application's panel declared."""
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    render = tmp_path / "config.yml"
    render.write_text(yaml.safe_dump({"control_system": {"writes": "allowed"}}), encoding="utf-8")
    panel = {
        "id": APP_NAME,
        "label": "Scaffold",
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
    body = WRITE_BODY if method == "POST" else None
    return client.request(method, f"/panel/{APP_NAME}{route}", json=body, headers=headers)


def _assert_refused_held(resp: httpx.Response, app: FastAPI) -> None:
    assert resp.status_code == 403
    assert "[osprey-refusal: session_held]" in resp.text
    assert _backend(app).await_count == 0


class TestTheAgentsWrite:
    """The panel token, naming the calling session."""

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("route", [WRITE_ROUTE, RESET_ROUTE])
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

        resp = _send(terminal, "POST", WRITE_ROUTE, BEARER, session=SESSION_KEY)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("session", [None, "", "not-live", OTHER_SESSION_KEY, "../x"])
    def test_a_write_naming_no_live_session_is_refused(self, terminal, session):
        """Leaving the header off, or naming a key nobody runs, does not shed the hold."""
        ensure_session(SESSION_KEY, root=_root(terminal))

        resp = _send(terminal, "POST", WRITE_ROUTE, BEARER, session=session)

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

        resp = _send(terminal, "POST", WRITE_ROUTE, OPERATOR)

        _assert_refused_held(resp, terminal)

    @pytest.mark.no_auth_seam
    def test_a_named_unheld_session_does_not_shed_the_terminals_hold(self, terminal):
        """The browser's header is not evidence: every live session is asked."""
        terminal.state.pty_registry = _Pool(SESSION_KEY, OTHER_SESSION_KEY)
        ensure_session(OTHER_SESSION_KEY, root=_root(terminal))
        _hold(_root(terminal))

        resp = _send(terminal, "POST", WRITE_ROUTE, OPERATOR, session=OTHER_SESSION_KEY)

        _assert_refused_held(resp, terminal)

    @pytest.mark.no_auth_seam
    def test_a_write_reaches_the_backend_when_no_live_session_is_held(self, terminal):
        ensure_session(SESSION_KEY, root=_root(terminal))
        _hold(_root(terminal), OTHER_SESSION_KEY)  # held, but not a session this terminal runs

        resp = _send(terminal, "POST", WRITE_ROUTE, OPERATOR)

        assert resp.status_code == 200
        assert _backend(terminal).await_count == 1

    @pytest.mark.no_auth_seam
    def test_a_write_reaches_the_backend_when_the_terminal_runs_no_session(self, terminal):
        terminal.state.pty_registry = _Pool()

        resp = _send(terminal, "POST", WRITE_ROUTE, OPERATOR)

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

    resp = _send(terminal, "POST", WRITE_ROUTE, credential)

    assert resp.status_code == 200
    assert _backend(terminal).await_count == 1


def _no_read(*_args: Any, **_kwargs: Any) -> bool:
    raise AssertionError("no hold store, so no hold is read")


# ---------------------------------------------------------------------------
# The websocket hop: every frame the browser sends is a write
# ---------------------------------------------------------------------------

SOCKET = f"/panel/{APP_NAME}/ws"


class _Application:
    """The application's end of a socket: it echoes each frame, and can push one of its own."""

    def __init__(self) -> None:
        self.received: list[object] = []
        self._frames: asyncio.Queue[str] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()

    async def send(self, data: object) -> None:
        self.received.append(data)
        self._frames.put_nowait(f"echo:{data}")

    def push(self, frame: str) -> None:
        """Send the browser a frame, from the test's thread."""
        self._loop.call_soon_threadsafe(self._frames.put_nowait, frame)

    def __aiter__(self) -> _Application:
        return self

    async def __anext__(self) -> str:
        return await self._frames.get()


class _Connect(_FakeConnect):
    """A connect that keeps the application end it handed the relay."""

    application: _Application

    async def __aenter__(self) -> _Application:
        self.application = _Application()
        return self.application


def _assert_closed_held(closed: pytest.ExceptionInfo[WebSocketDisconnect]) -> None:
    assert closed.value.code == 4003
    assert "[osprey-refusal: session_held]" in closed.value.reason
    assert len(closed.value.reason.encode()) <= 123  # the wire's cap on a close reason


class TestTheApplicationsSocket:
    """The operator's page; the gate opens no websocket to the panel token."""

    @pytest.mark.no_auth_seam
    def test_a_frame_reaches_the_application_when_no_session_is_held(self, terminal):
        ensure_session(SESSION_KEY, root=_root(terminal))
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), client.websocket_connect(SOCKET, headers=OPERATOR) as ws:
            ws.send_text("set")
            assert ws.receive_text() == "echo:set"

        assert connect.application.received == ["set"]

    @pytest.mark.no_auth_seam
    def test_a_held_terminals_socket_opens_and_reads_but_its_frame_closes_it(self, terminal):
        _hold(_root(terminal))
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), client.websocket_connect(SOCKET, headers=OPERATOR) as ws:
            connect.application.push("reading")
            assert ws.receive_text() == "reading"
            ws.send_text("set")
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()

        _assert_closed_held(closed)
        assert connect.application.received == []

    @pytest.mark.no_auth_seam
    @pytest.mark.parametrize("frame", ["set", b"\x01set"], ids=["text", "bytes"])
    def test_a_hold_that_starts_after_the_socket_opened_refuses_the_next_frame(
        self, terminal, frame
    ):
        ensure_session(SESSION_KEY, root=_root(terminal))
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), client.websocket_connect(SOCKET, headers=OPERATOR) as ws:
            ws.send_text("before")
            assert ws.receive_text() == "echo:before"
            _hold(_root(terminal))
            connect.application.push("reading")
            assert ws.receive_text() == "reading"
            if isinstance(frame, bytes):
                ws.send_bytes(frame)
            else:
                ws.send_text(frame)
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()

        _assert_closed_held(closed)
        assert connect.application.received == ["before"]

    @pytest.mark.no_auth_seam
    def test_a_frame_reaches_the_application_once_released(self, terminal):
        _hold(_root(terminal))
        answer_flags(
            SESSION_KEY, ["flag-a"], answer="continue", operator="alice", root=_root(terminal)
        )
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), client.websocket_connect(SOCKET, headers=OPERATOR) as ws:
            ws.send_text("set")
            assert ws.receive_text() == "echo:set"

    @pytest.mark.no_auth_seam
    def test_a_frame_reaches_the_application_when_no_live_session_is_held(self, terminal):
        ensure_session(SESSION_KEY, root=_root(terminal))
        _hold(_root(terminal), OTHER_SESSION_KEY)  # held, but not a session this terminal runs
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), client.websocket_connect(SOCKET, headers=OPERATOR) as ws:
            ws.send_text("set")
            assert ws.receive_text() == "echo:set"

    @pytest.mark.no_auth_seam
    def test_with_no_hold_store_nothing_is_read(self, terminal, monkeypatch):
        terminal.state.session_hold_root = None
        monkeypatch.setattr(proxy, "is_held", _no_read)
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), client.websocket_connect(SOCKET, headers=OPERATOR) as ws:
            ws.send_text("set")
            assert ws.receive_text() == "echo:set"

    @pytest.mark.no_auth_seam
    def test_the_panel_token_opens_no_socket_to_the_application(self, terminal):
        """Why no row here names a session: the agent's credential never reaches this hop."""
        ensure_session(SESSION_KEY, root=_root(terminal))
        client = TestClient(terminal, client=("127.0.0.1", 54321))
        connect = _Connect()

        with _patch_connect(connect), pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(SOCKET, headers={**BEARER, SESSION_HEADER: SESSION_KEY}):
                pass  # pragma: no cover - the handshake is refused

        assert connect.target is None


# ---------------------------------------------------------------------------
# The agent's tool server, through the real terminal: the SDK names the session
# ---------------------------------------------------------------------------


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The scaffold's tool-server module, imported from its source tree."""
    monkeypatch.syspath_prepend(str(TEMPLATE))
    import scaffold_app.tools

    assert scaffold_app.APP_NAME == APP_NAME and scaffold_app.WRITE_ROUTE == WRITE_ROUTE
    return scaffold_app.tools


@pytest.fixture
def tool_server(terminal, tools: ModuleType, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """The scaffold tool server's panel calls, carried to the terminal app in-process.

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
async def test_the_tool_servers_write_for_a_held_session_is_refused(tool_server, tools):
    _hold(_root(tool_server))

    with pytest.raises(ApprovalRefused) as info:
        await tools._panel_call("POST", WRITE_ROUTE, json=WRITE_BODY)

    assert "[osprey-refusal: session_held]" in info.value.message
    assert _backend(tool_server).await_count == 0


@pytest.mark.no_auth_seam
async def test_the_tool_servers_write_names_its_session_to_the_backend(tool_server, tools):
    ensure_session(SESSION_KEY, root=_root(tool_server))

    answer = await tools._panel_call("POST", WRITE_ROUTE, json=WRITE_BODY)

    assert answer == {"status": "applied"}
    forwarded = _backend(tool_server).await_args.kwargs["headers"]
    sent = {name.lower(): value for name, value in forwarded.items()}
    assert sent[SESSION_HEADER.lower()] == SESSION_KEY
