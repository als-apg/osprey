"""The Simple view's hand-off endpoint: ``POST /api/session/{key}/handoff``.

The route is a thin translation of one call — ``acquire_surface`` for the
Simple surface — into HTTP, and everything worth pinning is in that
translation:

- what the operator gets back when the flip works, with no turn taken on the
  chat the flip leaves pooled;
- every refusal carrying the status *and* the ``detail.error`` slug the client
  branches on, rather than a sentence it would have to match;
- a caller that goes away mid-wait costing nothing: no teardown, no spawn, no
  body.

The route is driven over HTTP throughout — through the router, the body model
and the exception handlers — rather than by calling the handler. The pools
behind it are the phase (a)/(b) fakes, and the wait's clock is faked with
them, so a two-second attach grace and a five-second interrupt grace are
exercised without being spent. The disconnect test drives the ASGI app
directly because no HTTP test client can hang up mid-request: its ``receive``
answers the hand-off's own disconnect polls.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.chat_session_pool import (
    ChatCapacityError,
    ChatSessionTerminatedError,
)
from osprey.interfaces.web_terminal.operator_session import POSTURE_SOURCE_LIVE
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes.session_handoff import router
from osprey.interfaces.web_terminal.session_handoff import (
    ERROR_CHAT_CAPACITY,
    ERROR_HANDOFF_NEEDS_INTERRUPT,
    ERROR_HANDOFF_SUPERSEDED,
    ERROR_OUTGOING_STILL_RUNNING,
    ERROR_SESSION_ATTACHED_ELSEWHERE,
    ERROR_SPAWN_NOT_POOLED,
    HandoffState,
    get_state,
)
from osprey.interfaces.web_terminal.turn_state import BUSY, IDLE
from osprey_connectors import posture_store
from tests.interfaces.web_terminal._fakes import FakeChatPool, FakeChatSession, FakeClock
from tests.interfaces.web_terminal.test_handoff_phase_a import KEY
from tests.interfaces.web_terminal.test_handoff_phase_b import RecordingPty, pool_pty, set_store

HANDOFF_PATH = f"/api/session/{KEY}/handoff"


@pytest.fixture(autouse=True)
def shared_root(tmp_path, monkeypatch):
    """Pin the agent-data root so a spawned child's environment resolves in ``tmp_path``.

    The environment builder stamps the root it resolves. Without a pinned one
    the resolution walks up to the repository and the suite's leak guard
    fires.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


# ---------------------------------------------------------------------------
# The app under test
# ---------------------------------------------------------------------------


class SurvivingPty(RecordingPty):
    """A terminal whose child outlives its kill: ``terminate`` returns, it does not."""

    def terminate(self) -> None:
        self.terminates += 1


class Chat(FakeChatSession):
    """A pooled chat that refuses to have a turn taken on it.

    The hand-off leaves the chat idle and free; ``POST /api/chat`` is what
    mints the per-turn guard. A route that took the turn here would strand it,
    so the guard is made loud rather than merely unasserted.
    """

    def acquire_turn(self) -> int:
        raise AssertionError("the hand-off must not take a turn on the chat")


class FakeChatRegistry:
    """The slice of ``OperatorRegistry`` the route and the acquire read.

    ``get_or_create_chat_session`` is the route's spawn callback in one hop:
    it records what it was asked for, calls the environment *builder* the way
    the real pool does, and pools the session it returns. The knobs are the
    three ways that can go wrong — an exception from the pool, and a spawn
    whose session never reaches the pool.
    """

    def __init__(self, chats: FakeChatPool) -> None:
        self.chats = chats
        self.spawns: list[tuple[str, str, str | None]] = []
        self.envs: list[dict[str, str]] = []
        self.raises: Exception | None = None
        self.pools_the_spawn = True

    async def get_or_create_chat_session(
        self,
        chat_id: str,
        cwd: str,
        env: Any = None,
        *,
        resume_id: str | None = None,
    ) -> tuple[Chat, bool]:
        self.spawns.append((chat_id, cwd, resume_id))
        if self.raises is not None:
            raise self.raises
        self.envs.append(env() if callable(env) else env)
        session = Chat()
        if self.pools_the_spawn:
            self.chats.sessions[chat_id] = session
        return session, False


class PacedClock(FakeClock):
    """The fake clock, spending a real millisecond per look.

    The plain fake yields once per look, which makes a two-second attach grace
    forty bare yields — gone before a worker thread the other request is
    waiting on can even return. A real millisecond per look keeps fake time
    fake and gives the two concurrent requests room to interleave the way
    they do behind a real server.
    """

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        for hook in list(self.on_sleep):
            hook(len(self.sleeps))
        await asyncio.sleep(0.001)


def make_app(tmp_path, *, hook: bool = True, paced: bool = False) -> FastAPI:
    """A FastAPI app exposing the hand-off route over the fake pools."""
    clock = PacedClock() if paced else FakeClock()
    app = FastAPI()
    app.include_router(router)
    app.state.project_cwd = str(tmp_path)
    app.state.pty_registry = PtyRegistry(max_background=5)
    app.state.operator_registry = FakeChatRegistry(FakeChatPool())
    app.state.handoff = HandoffState(clock=clock, sleep=clock.sleep)
    app.state.turn_hook_present = hook
    app.state.turn_state = {}
    app.state.transcript_map = {}
    app.state.transcript_map_provisional = False
    app.state.clock = clock
    return app


def chats(app: FastAPI) -> FakeChatPool:
    return app.state.operator_registry.chats


def registry(app: FastAPI) -> PtyRegistry:
    return app.state.pty_registry


def post(app: FastAPI, **body) -> Any:
    """POST a client-shaped hand-off body and return the response."""
    payload = {"to": "simple"}
    payload.update(body)
    with TestClient(app) as client:
        return client.post(HANDOFF_PATH, json=payload)


def assert_released(app: FastAPI) -> None:
    """The pending slot and the reservation an ended acquire leaves behind."""
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)


# ---------------------------------------------------------------------------
# Registration and the body model
# ---------------------------------------------------------------------------


def test_the_route_is_reachable_on_the_composite_router():
    """The web terminal's own router serves the path the client posts to.

    The composite router keeps its sub-routers unflattened until an app
    resolves them, so the question is put to a request rather than to a list
    of routes: a refused key is the handler answering, and a path nobody
    registered would be a 404.
    """
    from osprey.interfaces.web_terminal.routes import router as composite

    app = FastAPI()
    app.include_router(composite)

    resp = TestClient(app).post("/api/session/not-a-uuid/handoff", json={"to": "simple"})

    assert resp.status_code == 400


def test_only_the_simple_surface_can_be_asked_for(tmp_path):
    """The Expert view acquires on its socket; there is no HTTP form of it."""
    app = make_app(tmp_path)
    resp = post(app, to="expert")
    assert resp.status_code == 422
    assert app.state.operator_registry.spawns == []


def test_the_key_must_be_a_session_uuid(tmp_path):
    """A key outside the grammar never reaches the acquire."""
    app = make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.post("/api/session/not-a-uuid/handoff", json={"to": "simple"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "invalid_session_id"
    assert app.state.operator_registry.spawns == []


# ---------------------------------------------------------------------------
# The flip itself
# ---------------------------------------------------------------------------


def test_an_unheld_key_starts_the_chat_and_answers_with_the_surface(tmp_path):
    """Nothing live under the key: the chat is started fresh under it."""
    app = make_app(tmp_path)

    resp = post(app)

    assert resp.status_code == 200
    assert resp.json() == {"state": "simple", "session_id": KEY}
    assert app.state.operator_registry.spawns == [(KEY, str(tmp_path), None)]
    assert isinstance(chats(app).sessions[KEY], Chat)
    assert_released(app)


def test_the_child_is_stamped_with_the_key_and_a_live_posture(tmp_path):
    """The spawned chat carries the session key and the posture surface can address it."""
    app = make_app(tmp_path)

    assert post(app).status_code == 200

    env = app.state.operator_registry.envs[0]
    assert env["OSPREY_SESSION_ID"] == KEY
    assert env["OSPREY_POSTURE_SESSION"] == KEY
    assert env["OSPREY_POSTURE_SOURCE"] == POSTURE_SOURCE_LIVE


def test_an_idle_chat_already_under_the_key_is_handed_back_untouched(tmp_path):
    """The Simple view's own pooled chat is the answer; nothing is spawned or claimed."""
    app = make_app(tmp_path)
    pooled = Chat()
    chats(app).sessions[KEY] = pooled

    resp = post(app)

    assert resp.status_code == 200
    assert resp.json() == {"state": "simple", "session_id": KEY}
    assert app.state.operator_registry.spawns == []
    assert chats(app).sessions[KEY] is pooled
    assert pooled.teardowns == 0


def test_a_terminal_holding_the_key_is_torn_down_before_the_chat_starts(tmp_path):
    """The ordinary flip: the idle terminal dies, then the chat resumes the key."""
    app = make_app(tmp_path)
    pty = pool_pty(app)
    set_store(app, IDLE, app.state.clock.now)

    resp = post(app)

    assert resp.status_code == 200
    assert pty.terminates == 1
    assert registry(app).get_session(KEY) is None
    assert app.state.operator_registry.spawns == [(KEY, str(tmp_path), None)]
    assert_released(app)


def test_a_terminal_whose_turns_are_invisible_is_refused_until_the_caller_interrupts(tmp_path):
    """No turn-state hook: waiting is refused, and *Stop and switch now* is honoured."""
    app = make_app(tmp_path, hook=False)
    pty = pool_pty(app)

    refused = post(app)

    assert refused.status_code == 409
    assert refused.json()["detail"]["error"] == ERROR_HANDOFF_NEEDS_INTERRUPT
    assert pty.terminates == 0
    assert app.state.operator_registry.spawns == []
    assert_released(app)

    cut = post(app, interrupt=True)

    assert cut.status_code == 200
    assert pty.writes == [b"\x1b"]
    assert registry(app).get_session(KEY) is None
    assert app.state.operator_registry.spawns == [(KEY, str(tmp_path), None)]


# ---------------------------------------------------------------------------
# The refusals, one status and slug each
# ---------------------------------------------------------------------------


def test_a_key_another_view_is_consuming_is_a_409(tmp_path):
    """An attached terminal is re-looked at for the grace, then refused."""
    app = make_app(tmp_path)
    pool_pty(app)
    assert registry(app).attach_session(KEY, object())

    resp = post(app)

    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == ERROR_SESSION_ATTACHED_ELSEWHERE
    assert app.state.operator_registry.spawns == []
    assert_released(app)


def test_a_full_chat_pool_is_a_429(tmp_path):
    """Capacity is the pool's word, and it reaches the client as a retryable slug."""
    app = make_app(tmp_path)
    app.state.operator_registry.raises = ChatCapacityError("full")

    resp = post(app)

    assert resp.status_code == 429
    assert resp.json()["detail"]["error"] == ERROR_CHAT_CAPACITY
    assert_released(app)


def test_a_chat_torn_down_while_it_started_is_a_409(tmp_path):
    """A posture flip or a DELETE landing inside the spawn is retryable, not a 500."""
    app = make_app(tmp_path)
    app.state.operator_registry.raises = ChatSessionTerminatedError("gone")

    resp = post(app)

    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "chat_terminated"
    assert_released(app)


def test_a_terminal_that_survives_its_kill_is_a_503(tmp_path):
    """The previous agent is still shutting down; retrying re-runs the kill."""
    app = make_app(tmp_path)
    pty = pool_pty(app, SurvivingPty())
    set_store(app, IDLE, app.state.clock.now)

    resp = post(app)

    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == ERROR_OUTGOING_STILL_RUNNING
    assert pty.terminates == 1
    assert registry(app).get_session(KEY) is pty
    assert app.state.operator_registry.spawns == []
    assert_released(app)


def test_a_spawn_the_pool_does_not_hold_is_a_503(tmp_path):
    """A started child nobody pooled is a broken premise, not a hand-off."""
    app = make_app(tmp_path)
    app.state.operator_registry.pools_the_spawn = False

    resp = post(app)

    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == ERROR_SPAWN_NOT_POOLED
    assert_released(app)


# ---------------------------------------------------------------------------
# The caller that goes away
# ---------------------------------------------------------------------------


async def call_asgi(app: FastAPI, path: str, body: dict, *, polls_before_hangup: int) -> list[dict]:
    """POST to *app* over raw ASGI, hanging up after *polls_before_hangup* looks.

    No HTTP client can disconnect mid-request — both test transports hold
    their disconnect back until the response is complete — so the hand-off's
    own ``request.is_disconnected()`` polls are answered here: the body first,
    then that many messages that are not a disconnect, then the hang-up.
    """
    raw = json.dumps(body).encode()
    messages: list[dict] = [{"type": "http.request", "body": raw, "more_body": False}]
    messages += [{"type": "http.request", "body": b"", "more_body": False}] * polls_before_hangup

    async def receive() -> dict:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw)).encode()),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "state": {},
        },
        receive,
        send,
    )
    return sent


@pytest.mark.asyncio
async def test_a_caller_that_hangs_up_mid_wait_gets_no_body_and_costs_nothing(tmp_path):
    """The flip is abandoned where it stood: nothing torn down, nothing started."""
    app = make_app(tmp_path)
    pty = pool_pty(app)
    set_store(app, BUSY, app.state.clock.now)

    sent = await call_asgi(app, HANDOFF_PATH, {"to": "simple"}, polls_before_hangup=2)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 204
    assert all(not m.get("body") for m in sent if m["type"] == "http.response.body")
    assert app.state.clock.sleeps, "the hang-up must land while the wait is running"
    assert pty.terminates == 0
    assert pty.writes == []
    assert registry(app).get_session(KEY) is pty
    assert app.state.operator_registry.spawns == []
    assert_released(app)


# ---------------------------------------------------------------------------
# Two requests for one key: the newer one, interrupting, takes over
# ---------------------------------------------------------------------------


def test_a_second_request_with_an_interrupt_takes_the_key_from_a_waiting_one(tmp_path):
    """*Stop and switch now* from the transitional state, through the whole stack.

    The first request waits on the terminal's turn; the second, made while it
    waits and carrying the interrupt, is what the console sends when the
    operator presses the button. The first is answered ``handoff_superseded``
    and the second completes the flip — nothing depends on the server seeing
    the first request abandoned, which under a browser it does not.
    """
    app = make_app(tmp_path, paced=True)
    pty = pool_pty(app)
    set_store(app, BUSY, app.state.clock.now)

    def idle_once_escaped(_count: int) -> None:
        if pty.writes:
            set_store(app, IDLE, app.state.clock.now)

    app.state.clock.on_sleep.append(idle_once_escaped)

    # Every wait here is bounded, and the executor is never joined on a request
    # that may still be waiting: should either POST fail to answer, the
    # terminal is reported idle so the wait ends on its own, and the failure
    # is a timeout in seconds rather than a join that never returns.
    threads = ThreadPoolExecutor(max_workers=2)
    try:
        with TestClient(app) as client:
            first = threads.submit(client.post, HANDOFF_PATH, json={"to": "simple"})
            deadline = time.monotonic() + 5.0
            while KEY not in get_state(app).pending:
                assert time.monotonic() < deadline, "the first request never reached its wait"
                time.sleep(0.005)
            second = threads.submit(
                client.post, HANDOFF_PATH, json={"to": "simple", "interrupt": True}
            )
            try:
                superseded = first.result(timeout=30.0)
                cut = second.result(timeout=30.0)
            finally:
                set_store(app, IDLE, app.state.clock.now)
    finally:
        threads.shutdown(wait=True)

    assert superseded.status_code == 409
    assert superseded.json()["detail"]["error"] == ERROR_HANDOFF_SUPERSEDED
    assert cut.status_code == 200
    assert cut.json() == {"state": "simple", "session_id": KEY}
    assert pty.writes == [b"\x1b"]
    assert pty.terminates == 1
    assert registry(app).get_session(KEY) is None
    assert app.state.operator_registry.spawns == [(KEY, str(tmp_path), None)]
    assert_released(app)
