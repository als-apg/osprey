"""Every moved reader says "record unavailable" or "incomplete" — never an empty, complete answer.

FR17. Each reader is driven through its own production entry point, with its
own record client resolved from the environment the way it is deployed:

* **agent-record stopped.** Every record client aims at a port nothing
  listens on. The history page, ``session_log``, the session list, chat
  replay, ``/api/audit/recent``, the hook debug panel, dispatch status and
  bridge replay each answer "record unavailable" (dispatch status: ``pending``
  with ``record: "unavailable"``) within the reader's bound — its client
  timeout plus its wait bound — and none answers an empty result as complete.
* **``osprey feedback list``** reads through a host-local record when no URL is
  set. Its outage is the record's loopback port held by a server that is not
  the record: no host-local record can start there, so the command refuses,
  naming the record, within the ensure's health bound, and spawns nothing.
* **A stalled consumer, the service up.** A real agent-record over the
  embedded backend serves on a real port, but its spool consumer commits
  nothing and its drain waits one second. Every reader that waits on a drain —
  ``session_log``, the session list, chat replay, dispatch status, bridge
  replay and the feedback read — answers ``incomplete`` within its bound, with a
  line still pending in the spool its drain covers.
"""

from __future__ import annotations

import base64
import functools
import json
import time
from collections.abc import Iterator
from typing import Any

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record import contract as c
from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record import identities as ids
from osprey.agent_record import reader as rd
from osprey.agent_record import service, writer
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import DEFAULT_TIMEOUT, URL_ENV, RecordClient
from osprey.agent_record.ingest.spool import ConsumeReport, SpoolConsumer
from osprey.audit import call as audit_call
from osprey.bridges.core.history import RECORD_INCOMPLETE as BRIDGE_INCOMPLETE
from osprey.bridges.core.history import RECORD_UNAVAILABLE as BRIDGE_UNAVAILABLE
from osprey.bridges.core.history import ConversationHistory
from osprey.cli import feedback_cmd
from osprey.cli.feedback_cmd import feedback
from osprey.history.app import create_app as create_history_app
from osprey.interfaces._serving import run_app_server
from osprey.interfaces.web_terminal.routes import audit as audit_routes
from osprey.interfaces.web_terminal.routes import config as config_routes
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.mcp_server.dispatch_worker import dispatch_api
from osprey.mcp_server.session import DISPATCH_RUN_ENV
from osprey_connectors_tests._harness._ports import reserved_port
from tests.agent_record.test_identities import _config
from tests.agent_record.test_spool_consumer import _line, _sealed
from tests.cli.test_feedback_cmd import _make_repo
from tests.history.test_history_app import _headers as _history_headers
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

TERMINAL = "alice"
DISPATCH = "dispatch-worker-1"
BRIDGE = ids.BRIDGE_IDENTITIES["teams_bridge"]

SESSION = "3f8a1c02-0000-4000-8000-abcdefabcdef"
RUN_ID = "run-7e1d0c5a"
CONVERSATION = "teams:conv-1"

#: The longest any reader whose client is resolved from the environment may
#: take to answer: the client's own timeout plus the record's drain wait.
READER_BOUND_S = DEFAULT_TIMEOUT + rd.WAIT_SECONDS

#: The drain wait the stalled service runs with, and the bound a drained read
#: under it must stay inside (its wait plus generous room for the read itself).
STALLED_WAIT_S = 1.0
STALLED_BOUND_S = STALLED_WAIT_S + 10.0

#: How long ``osprey feedback list`` waits for a usable record on the port
#: before it refuses. Shortened from ``HEALTH_WAIT_S`` so the test does not
#: spend a minute; the bound stays the same shape (wait plus one probe).
FEEDBACK_HEALTH_WAIT_S = 2.0
FEEDBACK_BOUND_S = FEEDBACK_HEALTH_WAIT_S + hr._PROBE_TIMEOUT_S + 5.0


# --- fixtures -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR", raising=False)
    yield
    writer.seal_all()


@pytest.fixture(autouse=True)
def _fresh_dispatch_state(monkeypatch):
    """No run is in flight in, or settled by, this process."""
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_settled", dispatch_api.OrderedDict())


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _login_pair(identity: str) -> str:
    login = ids.login_for(identity)
    return f"{login}:{_token(login)}"


def _aim(monkeypatch, url: str, identity: str = TERMINAL) -> None:
    """Every record client this process builds from its environment reads *url* as *identity*."""
    monkeypatch.setenv(URL_ENV, url)
    monkeypatch.setenv(creds.LOGIN_ENV, _login_pair(identity))


@pytest.fixture
def stopped_url() -> Iterator[str]:
    """A loopback URL nothing listens on: agent-record is stopped. The port is
    reserved for the test and never bound."""
    with reserved_port() as port:
        yield f"http://127.0.0.1:{port}"


@pytest.fixture
def web_terminal(tmp_path) -> TestClient:
    """The web terminal's record-reading routes, on an app whose state they rely on."""
    app = FastAPI()
    app.include_router(session_routes.router)
    app.include_router(audit_routes.router)
    app.include_router(config_routes.router)
    app.state.terminal_user = TERMINAL
    app.state.config_panel_enabled = True
    app.state.transcript_map = {}
    app.state.transcript_map_provisional = False
    app.state.project_cwd = str(tmp_path)
    return TestClient(app)


@pytest.fixture
def session_log_tool(tmp_path, monkeypatch):
    """The ``session_log`` MCP tool, serving *SESSION* from a render at *tmp_path*."""
    from osprey.mcp_server.workspace.tools import session_log as module

    monkeypatch.delenv(audit_call._OSPREY_CONVERSATION_ENV, raising=False)
    monkeypatch.setenv(audit_call._HARNESS_CONVERSATION_ENV, SESSION)
    monkeypatch.delenv(DISPATCH_RUN_ENV, raising=False)
    monkeypatch.setattr(module, "deployed_render_dir", lambda: tmp_path)
    return get_tool_fn(module.session_log)


def _timed(call):
    started = time.monotonic()
    value = call()
    return value, time.monotonic() - started


async def _timed_async(call):
    started = time.monotonic()
    value = await call()
    return value, time.monotonic() - started


def _not_complete(body: dict[str, Any], state: str) -> None:
    """*body* names *state* in its ``record`` field and a reason for it."""
    assert body.get("record") == state, body
    assert body.get("record_reason") or body.get("reason"), body


# --- agent-record stopped -----------------------------------------------------------


def test_the_history_page_says_record_unavailable(stopped_url):
    env = {URL_ENV: stopped_url, creds.LOGIN_ENV: _login_pair(ids.HISTORY_IDENTITY)}
    page = TestClient(create_history_app(env=env, max_days=30))

    for path in ("/", f"/session/{SESSION}"):
        response, elapsed = _timed(lambda path=path: page.get(path, headers=_history_headers()))
        assert response.status_code == 503, (path, response.text)
        assert "record unavailable" in response.text, path
        assert elapsed < READER_BOUND_S, (path, elapsed)


async def test_session_log_says_record_unavailable(stopped_url, monkeypatch, session_log_tool):
    _aim(monkeypatch, stopped_url)

    started = time.monotonic()
    with assert_raises_error(error_type="record_unavailable") as raised:
        await session_log_tool()
    elapsed = time.monotonic() - started

    assert "unavailable" in raised["envelope"]["error_message"]
    assert elapsed < READER_BOUND_S


def test_the_session_list_says_record_unavailable(stopped_url, monkeypatch, web_terminal):
    _aim(monkeypatch, stopped_url)

    response, elapsed = _timed(lambda: web_terminal.get("/api/sessions"))

    assert response.status_code == 200
    body = response.json()
    assert body["sessions"] == []
    _not_complete(body, "unavailable")
    assert elapsed < READER_BOUND_S


def test_simple_view_replay_says_record_unavailable(stopped_url, monkeypatch, web_terminal):
    _aim(monkeypatch, stopped_url)

    response, elapsed = _timed(
        lambda: web_terminal.get("/api/session-chat", params={"session_id": SESSION, "events": 1})
    )

    assert response.status_code == 200
    body = response.json()
    assert body["turns"] == [] and body["count"] == 0
    _not_complete(body, "unavailable")
    assert elapsed < READER_BOUND_S


def test_recent_audit_says_record_unavailable(stopped_url, monkeypatch, web_terminal):
    _aim(monkeypatch, stopped_url)

    response, elapsed = _timed(lambda: web_terminal.get("/api/audit/recent"))

    assert response.status_code == 200
    body = response.json()
    assert body["records"] == []
    _not_complete(body, audit_routes.UNAVAILABLE)
    assert elapsed < READER_BOUND_S


def test_the_hook_debug_panel_says_record_unavailable(stopped_url, monkeypatch, web_terminal):
    _aim(monkeypatch, stopped_url)

    response, elapsed = _timed(lambda: web_terminal.get("/api/hooks/debug-log"))

    assert response.status_code == 200
    body = response.json()
    assert body["entries"] == []
    _not_complete(body, "unavailable")
    assert elapsed < READER_BOUND_S


async def test_dispatch_status_is_pending_with_the_record_unavailable(stopped_url, monkeypatch):
    _aim(monkeypatch, stopped_url, DISPATCH)

    body, elapsed = await _timed_async(lambda: dispatch_api.get_dispatch_result(RUN_ID))

    assert body["run_id"] == RUN_ID
    assert body["status"] == "pending"
    assert body["record"] == dispatch_api.RECORD_UNAVAILABLE
    assert elapsed < READER_BOUND_S


def test_bridge_replay_says_record_unavailable(stopped_url, monkeypatch):
    _aim(monkeypatch, stopped_url, BRIDGE)

    read, elapsed = _timed(lambda: ConversationHistory().read(CONVERSATION))

    assert read.record == BRIDGE_UNAVAILABLE
    assert read.turns == [] and read.reason
    assert elapsed < READER_BOUND_S


# --- osprey feedback list: the record's port held by something else -----------------


@pytest.fixture
def not_the_record() -> Iterator[int]:
    """A server on a loopback port that answers, but is not an agent-record."""
    with run_app_server(FastAPI()) as url:
        yield int(url.rsplit(":", 1)[1])


def test_feedback_list_refuses_when_the_records_port_is_held_by_another_server(
    tmp_path, monkeypatch, not_the_record
):
    claude = tmp_path / "claude"
    claude.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.setattr(
        hr,
        "ensure_host_record",
        functools.partial(hr.ensure_host_record, wait_s=FEEDBACK_HEALTH_WAIT_S),
    )
    root, config = _make_repo(tmp_path, port=not_the_record)

    result, elapsed = _timed(lambda: CliRunner().invoke(feedback, ["list", "--repo", str(root)]))

    assert result.exit_code != 0, result.output
    assert "agent record is not available" in result.output
    assert "No feedback has been sent" not in result.output
    assert elapsed < FEEDBACK_BOUND_S
    # It never spawned over the live port.
    assert hr._running_process(hr.host_state_dir(root, config)) is None


# --- a stalled consumer, the service up ---------------------------------------------


@pytest.fixture
def stalled_record(tmp_path, monkeypatch) -> Iterator[str]:
    """A real agent-record whose consumer commits nothing, with a line pending per reader.

    Each drain-waiting reader's own spool directory (and so its drain) holds a
    sealed line the consumer never gets to, and the service's drain waits
    :data:`STALLED_WAIT_S` before it answers incomplete.
    """
    monkeypatch.setattr(SpoolConsumer, "consume", lambda self: ConsumeReport())
    monkeypatch.setattr(
        service, "RecordReader", functools.partial(rd.RecordReader, wait_seconds=STALLED_WAIT_S)
    )
    config = _config(workers=1, bridges=("teams_bridge",))
    config["project_root"] = str(tmp_path / "project")
    spool = tmp_path / "spool"
    _sealed(spool / TERMINAL, _line(c.KIND_AUDIT, {"what": "stuck"}, session_id=SESSION))
    _sealed(
        spool / DISPATCH,
        _line(
            c.KIND_RUN,
            {"phase": c.RUN_ACCEPT, "run_id": RUN_ID, "session_id": SESSION, "requester": "bob"},
            run_id=RUN_ID,
            session_id=SESSION,
        ),
    )
    _sealed(spool / BRIDGE, _line(c.KIND_AUDIT, {"what": "stuck"}))
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    env = {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps(["s3cr3t-token-value-0123456789"])),
        service.SPOOL_ROOT_ENV: str(spool),
        service.RECORD_DIR_ENV: str(tmp_path / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "transcripts"),
    }
    with EmbeddedBackend(tmp_path / "record.duckdb") as backend:
        app = service.create_app(config=config, backend=backend, env=env)
        with run_app_server(app) as url:
            yield url


async def test_session_log_answers_incomplete(stalled_record, monkeypatch, session_log_tool):
    _aim(monkeypatch, stalled_record)

    raw, elapsed = await _timed_async(session_log_tool)

    body = json.loads(raw)
    _not_complete(body, "incomplete")
    assert STALLED_WAIT_S <= elapsed < STALLED_BOUND_S


def test_the_session_list_answers_incomplete(stalled_record, monkeypatch, web_terminal):
    _aim(monkeypatch, stalled_record)

    response, elapsed = _timed(lambda: web_terminal.get("/api/sessions"))

    _not_complete(response.json(), "incomplete")
    assert STALLED_WAIT_S <= elapsed < STALLED_BOUND_S


def test_chat_replay_answers_incomplete(stalled_record, monkeypatch, web_terminal):
    _aim(monkeypatch, stalled_record)

    response, elapsed = _timed(
        lambda: web_terminal.get("/api/session-chat", params={"session_id": SESSION, "events": 1})
    )

    _not_complete(response.json(), "incomplete")
    assert STALLED_WAIT_S <= elapsed < STALLED_BOUND_S


async def test_dispatch_status_is_pending_with_the_record_incomplete(stalled_record, monkeypatch):
    _aim(monkeypatch, stalled_record, DISPATCH)

    body, elapsed = await _timed_async(lambda: dispatch_api.get_dispatch_result(RUN_ID))

    assert body["status"] == "pending"
    assert body["record"] == dispatch_api.RECORD_INCOMPLETE
    assert STALLED_WAIT_S <= elapsed < STALLED_BOUND_S


def test_bridge_replay_answers_incomplete(stalled_record, monkeypatch):
    _aim(monkeypatch, stalled_record, BRIDGE)

    read, elapsed = _timed(lambda: ConversationHistory().read(CONVERSATION))

    assert read.record == BRIDGE_INCOMPLETE
    assert read.turns == [] and read.reason
    assert STALLED_WAIT_S <= elapsed < STALLED_BOUND_S


def test_the_feedback_read_warns_it_is_incomplete(stalled_record, monkeypatch):
    warned: list[str] = []
    monkeypatch.setattr(feedback_cmd, "warn", warned.append)
    login = ids.login_for(ids.OPERATOR_IDENTITY)
    client = RecordClient(stalled_record, login, _token(login))

    _, elapsed = _timed(lambda: feedback_cmd._read_feedback(client))

    assert warned and "before every pending submission" in warned[0]
    assert STALLED_WAIT_S <= elapsed < STALLED_BOUND_S
