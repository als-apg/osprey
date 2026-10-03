"""Tests for the REST chat endpoint (`routes/chat.py`).

Task 1.4 (strip-for-chat): `_strip_for_chat` is the wire-hygiene filter applied
to every event before it reaches the chat API client. It drops heavy or
sensitive payloads (tool arguments, tool-result bodies, thinking text, and the
cost/duration/turn metadata on `result`) while preserving each event's identity
and light metadata.

The table below is anchored to the real event shapes produced by
`_event_to_wire` in ``operator_session.py`` (text / thinking / tool_use /
result / system / error), a ``tool_result`` shape, plus the ``session_reset`` marker the
control routes emit — so a change to those shapes that this filter should react
to will surface here.

Later tasks (chat-stream-route, chat-control-routes, integration) append their
own route-level test classes to this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from claude_agent_sdk import (
    CLIConnectionError,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import osprey.interfaces.web_terminal.routes.chat as chat_module
import osprey.interfaces.web_terminal.session_handoff as handoff_module
from osprey.audit.envelope import POSTURE_SOURCE_LIVE, POSTURE_SOURCE_PROCESS
from osprey.audit.posture import posture_source
from osprey.interfaces.web_terminal.chat_session_pool import ChatCapacityError
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    OperatorRegistry,
    OperatorSession,
)
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes.chat import _strip_for_chat
from tests.interfaces.web_terminal._fakes import (
    assistant_message,
    result_message,
    sdk_seam,
    user_message,
)


@pytest.fixture(autouse=True)
def transcripts_on_disk():
    """The transcripts the hand-off door believes exist, empty by default.

    Every turn now goes through ``acquire_surface``, which decides what a new
    child resumes by listing the transcripts under the app's project. These
    tests run against whatever that project directory really holds, so the
    listing is pinned here instead. A test that wants a key to resume adds its
    id to the yielded set.
    """
    ids: set[str] = set()
    with patch.object(handoff_module, "_transcripts_on_disk", lambda app: set(ids)):
        yield ids


# ---- Representative events, one per type `_event_to_wire` can emit. ----
# Each entry: (label, input_event, expected_output_event).
_STRIP_CASES = [
    (
        "text_passes_through",
        {"type": "text", "content": "hello world"},
        {"type": "text", "content": "hello world"},
    ),
    (
        "thinking_drops_content_keeps_marker",
        {"type": "thinking", "content": "internal reasoning"},
        {"type": "thinking"},
    ),
    (
        "tool_use_drops_input_keeps_name",
        {
            "type": "tool_use",
            "tool_name": "Channel Read",
            "tool_name_raw": "mcp__osprey__channel_read",
            "tool_use_id": "tu_1",
            "input": {"channel": "SR:BPM:1", "secret": "value"},
        },
        {
            "type": "tool_use",
            "tool_name": "Channel Read",
            "tool_name_raw": "mcp__osprey__channel_read",
            "tool_use_id": "tu_1",
        },
    ),
    (
        "tool_result_drops_content_keeps_is_error",
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "large result body",
            "is_error": False,
        },
        {"type": "tool_result", "tool_use_id": "tu_1", "is_error": False},
    ),
    (
        "result_reduces_to_type_and_is_error",
        {
            "type": "result",
            "is_error": False,
            "total_cost_usd": 0.0123,
            "duration_ms": 4567,
            "num_turns": 3,
        },
        {"type": "result", "is_error": False},
    ),
    (
        "result_error_preserves_is_error_true",
        {
            "type": "result",
            "is_error": True,
            "total_cost_usd": 0.5,
            "duration_ms": 10,
            "num_turns": 1,
        },
        {"type": "result", "is_error": True},
    ),
    (
        "system_passes_through",
        {"type": "system", "subtype": "init"},
        {"type": "system", "subtype": "init"},
    ),
    (
        "error_passes_through",
        {
            "type": "error",
            "message": "boom",
            "error_type": "ClaudeSDKError",
        },
        {
            "type": "error",
            "message": "boom",
            "error_type": "ClaudeSDKError",
        },
    ),
    (
        "session_reset_passes_through",
        {"type": "session_reset"},
        {"type": "session_reset"},
    ),
    (
        "tool_result_list_content_error_keeps_is_error",
        {
            "type": "tool_result",
            "tool_use_id": "tu_x",
            "content": [{"type": "text", "text": "..."}],
            "is_error": True,
        },
        {"type": "tool_result", "tool_use_id": "tu_x", "is_error": True},
    ),
    (
        "unknown_type_passes_through",
        {"type": "keepalive", "extra": 1},
        {"type": "keepalive", "extra": 1},
    ),
]


class TestStripForChat:
    """`_strip_for_chat`: per-type key survival + no-mutation contract."""

    @pytest.mark.parametrize(
        "input_event,expected",
        [(inp, exp) for _label, inp, exp in _STRIP_CASES],
        ids=[label for label, _inp, _exp in _STRIP_CASES],
    )
    def test_exact_surviving_keys(self, input_event, expected):
        """Each event type keeps exactly the expected keys and values."""
        assert _strip_for_chat(input_event) == expected

    @pytest.mark.parametrize(
        "input_event",
        [inp for _label, inp, _exp in _STRIP_CASES],
        ids=[label for label, _inp, _exp in _STRIP_CASES],
    )
    def test_does_not_mutate_input(self, input_event):
        """The filter never mutates its argument."""
        import copy

        before = copy.deepcopy(input_event)
        _strip_for_chat(input_event)
        assert input_event == before


# ---- Route-level smoke tests for the SSE branch (task chat-stream-route) ----
#
# These are intentionally light — they exercise the wire contract of the SSE
# branch (session_reset gating, per-event strip, guard release, and the
# 409/429/422/503 status map). The full scenario matrix lives in task
# chat-route-integration-tests.


class _FakeAgent:
    """Stands in for the runner's session; records signal-only `interrupt()` calls."""

    def __init__(self):
        self.interrupts = 0

    async def interrupt(self):
        self.interrupts += 1


class _FakeChatSession(OperatorSession):
    """OperatorSession with a preloaded queue instead of an agent runner.

    Inherits the real turn guard and ``run_turn`` machine; only the transport
    (``send_prompt``) and the quiesce side effect are replaced, with counters.
    """

    def __init__(self, events):
        super().__init__(cwd="/tmp")
        # A session the pools consider live: ``is_active`` is what the
        # hand-off door reads to tell a pooled chat from a corpse.
        self._started = True
        self._events = list(events)
        self.quiesce_calls = 0
        self.release_calls = 0
        self.prompts: list[str] = []
        # An open runner scope over the stand-in: the session reads as live
        # and forwards interrupts to it.
        self._agent = _FakeAgent()
        self._agent_scope = AsyncExitStack()

    def release_turn(self, token: int) -> bool:
        self.release_calls += 1
        return super().release_turn(token)

    async def send_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)
        for event in self._events:
            await self._queue.put(event)

    def spawn_quiesce(self):
        self.quiesce_calls += 1
        return asyncio.get_event_loop().create_task(asyncio.sleep(0))


class _FakeChatPool:
    """The pool face the hand-off door inspects, backed by ``_FakeRegistry``."""

    def __init__(self, registry):
        self._registry = registry

    def get(self, chat_id):
        return self._registry.pooled.get(chat_id)

    def has_key(self, chat_id):
        return chat_id in self._registry.pooled

    async def terminate(self, chat_id):
        return self._registry.pooled.pop(chat_id, None)


class _FakeRegistry:
    """A chat registry double, plus the pool view ``acquire_surface`` reads.

    ``pooled_key`` seeds the pool with *session* under that key, which is how
    a test says "this chat is already live": the hand-off hands it back and
    nothing is spawned. Left unset, the key holds nothing and the route's
    spawn callback creates the session — the only path that still reaches
    ``get_or_create_chat_session``.
    """

    def __init__(self, session=None, pooled_key=None, capacity=False):
        self._session = session
        self._capacity = capacity
        self.calls: list[str] = []
        self.terminated: list[str] = []
        self.resume_ids: list[str | None] = []
        self.pooled: dict[str, object] = {}
        if pooled_key is not None:
            self.pooled[pooled_key] = session
        self.chats = _FakeChatPool(self)

    async def get_or_create_chat_session(self, chat_id, _cwd, _env, *, resume_id=None):
        self.calls.append(chat_id)
        self.resume_ids.append(resume_id)
        if self._capacity:
            raise ChatCapacityError("all busy")
        self.pooled[chat_id] = self._session
        return self._session, False

    def get_chat_session(self, _chat_id):
        return self._session

    async def terminate_chat_session(self, chat_id):
        self.terminated.append(chat_id)


def _make_chat_app(registry, turn_timeout_s=5) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_module.router)
    app.state.project_cwd = "/tmp"
    app.state.operator_registry = registry
    app.state.chat_turn_timeout_s = turn_timeout_s
    _seed_handoff_state(app.state)
    return app


def _seed_handoff_state(state) -> None:
    """The app state ``acquire_surface`` reads beyond the chat pool.

    An empty PTY registry (no terminal holds these keys), and a transcript map
    that is already loaded so no key resolution reaches the agent-data root.
    """
    state.pty_registry = PtyRegistry()
    state.transcript_map = {}
    state.transcript_map_provisional = False


async def _connected() -> bool:
    """A client that is still there; the channel probe every request needs."""
    return False


def _data_frames(text: str) -> list[dict]:
    """Parse the JSON payloads of `data:` SSE frames from a buffered response."""
    return [
        _json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


class TestChatStreamRoute:
    """SSE branch: wire contract + status map."""

    def test_new_session_emits_session_reset_then_release(self):
        session = _FakeChatSession(
            [
                {"type": "text", "content": "hi"},
                {"type": "result", "is_error": False, "total_cost_usd": 0.1},
            ]
        )
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "hello", "chat_id": "c1"})
        assert resp.status_code == 200
        frames = _data_frames(resp.text)

        assert frames[0] == {"type": "session_reset"}
        assert {"type": "text", "content": "hi"} in frames
        # result reduced by _strip_for_chat — cost never reaches the client.
        assert {"type": "result", "is_error": False} in frames
        assert all("total_cost_usd" not in f for f in frames)
        # Terminal exit releases the guard and does not quiesce.
        assert session.release_calls >= 1
        assert session.in_flight is False
        assert session.quiesce_calls == 0

    def test_turn_in_progress_returns_409(self):
        session = _FakeChatSession([{"type": "result", "is_error": False}])
        session.acquire_turn()  # a turn is already held
        registry = _FakeRegistry(session=session, pooled_key="c1")
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "turn_in_progress"

    def test_empty_prompt_returns_422(self):
        registry = _FakeRegistry(session=_FakeChatSession([]))
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "   ", "chat_id": "c1"})
        assert resp.status_code == 422

    def test_sdk_unavailable_returns_503(self, monkeypatch):
        monkeypatch.setattr(chat_module, "HAS_SDK", False)
        registry = _FakeRegistry(session=_FakeChatSession([]))
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})
        assert resp.status_code == 503

    async def test_a_failed_submit_names_the_agent_sdk_error_on_the_wire(self):
        with _seam(_clean_responder(), query_error=CLIConnectionError("closed")):
            registry = OperatorRegistry()
            req = _req(registry)

            resp = await chat_module.chat(req, chat_module.ChatRequest(prompt="p", chat_id="c"))
            frames = await _collect_sse(resp)

            errors = [f for f in frames if f.get("type") == "error"]
            assert errors
            assert errors[-1]["error_type"] == "CLIConnectionError"
            assert errors[-1]["message"] == "closed"
            await registry.cleanup_all()


class TestChatBufferedRoute:
    """Buffered branch (`stream=false`): reduced payload + guard discipline."""

    def test_new_session_prepends_session_reset_and_reduces_payload(self):
        session = _FakeChatSession(
            [
                {"type": "text", "content": "hello "},
                {"type": "text", "content": "world"},
                {"type": "result", "is_error": False, "total_cost_usd": 0.9, "num_turns": 2},
            ]
        )
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        resp = client.post(
            "/api/chat", params={"stream": "false"}, json={"prompt": "hi", "chat_id": "c1"}
        )
        assert resp.status_code == 200
        payload = resp.json()

        # Top-level keys reduced to exactly {text, events, is_error} (no error).
        assert set(payload) == {"text", "events", "is_error"}
        assert payload["text"] == "hello world"
        assert payload["is_error"] is False
        # session_reset prepended; result reduced by _strip_for_chat.
        assert payload["events"][0] == {"type": "session_reset"}
        assert {"type": "result", "is_error": False} in payload["events"]
        for event in payload["events"]:
            assert "total_cost_usd" not in event and "num_turns" not in event
        # Terminal exit released the guard, no quiesce.
        assert session.in_flight is False
        assert session.quiesce_calls == 0

    def test_terminal_error_returns_500_with_error_key(self):
        session = _FakeChatSession(
            [{"type": "error", "message": "boom", "error_type": "ClaudeSDKError"}]
        )
        registry = _FakeRegistry(session=session, pooled_key="c1")
        client = TestClient(_make_chat_app(registry))

        resp = client.post(
            "/api/chat", params={"stream": "false"}, json={"prompt": "hi", "chat_id": "c1"}
        )
        assert resp.status_code == 500
        payload = resp.json()
        assert set(payload) == {"text", "events", "is_error", "error"}
        assert payload["is_error"] is True
        assert payload["error"] == "boom"


class TestChatHandoffMapping:
    """What the hand-off door raises, and what the client is told.

    Every refusal reaches the client as a status plus a machine-readable
    ``error`` slug, because that slug is what ``chat.js`` branches on to
    choose its copy and whether to offer a retry.
    """

    def _refusing_app(self, exc):
        """A chat app whose acquire always fails with *exc*."""
        registry = _FakeRegistry(session=_FakeChatSession([]))
        app = _make_chat_app(registry)

        async def refuse(*args, **kwargs):
            raise exc

        return app, patch.object(chat_module, "acquire_surface", refuse)

    @pytest.mark.parametrize(
        ("refusal", "status", "slug"),
        [
            (handoff_module.HandoffRefused.needs_interrupt, 409, "handoff_needs_interrupt"),
            (handoff_module.HandoffRefused.outgoing_still_running, 503, "outgoing_still_running"),
        ],
        ids=["needs-interrupt", "outgoing-still-running"],
    )
    def test_a_refusal_maps_to_its_status_and_slug(self, refusal, status, slug):
        app, patched = self._refusing_app(refusal("c1"))
        with patched, TestClient(app) as client:
            resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})

        assert resp.status_code == status
        assert resp.json()["detail"]["error"] == slug

    def test_a_vanished_premise_is_503_with_its_slug(self):
        app, patched = self._refusing_app(handoff_module.HandoffError.vanished("c1", "expert"))
        with patched, TestClient(app) as client:
            resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})

        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "outgoing_vanished"

    def test_a_client_that_hung_up_gets_no_stream(self):
        """``ChannelClosed`` is not an error to report — there is nobody to report it to."""
        app, patched = self._refusing_app(handoff_module.ChannelClosed())
        with patched, TestClient(app) as client:
            resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})

        assert resp.status_code == 204
        assert resp.content == b""


class TestChatResumesTheKeysTranscript:
    """The transcript the key points at is what the new chat child opens."""

    def test_the_resume_id_reaches_the_pool_and_suppresses_the_reset(self, transcripts_on_disk):
        transcripts_on_disk.add("c1")
        session = _FakeChatSession([{"type": "result", "is_error": False}])
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        frames = _data_frames(client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"}).text)

        assert registry.resume_ids == ["c1"]
        assert all(f.get("type") != "session_reset" for f in frames)

    def test_a_key_with_no_transcript_starts_fresh_under_the_key(self):
        session = _FakeChatSession([{"type": "result", "is_error": False}])
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        frames = _data_frames(client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"}).text)

        assert registry.resume_ids == [None]
        assert frames[0] == {"type": "session_reset"}


class TestInterruptEndpoint:
    """POST /api/chat/{chat_id}/interrupt — signal-only, never releases."""

    def test_unknown_chat_is_noop_204(self):
        registry = _FakeRegistry(session=None)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat/nope/interrupt")
        assert resp.status_code == 204


# ===========================================================================
# Full route-level scenario matrix (task chat-route-integration-tests)
# ===========================================================================
#
# Unlike the smoke classes above (which fake the whole session/registry), these
# tests run a REAL ``OperatorRegistry`` + ``OperatorSession`` and patch only the
# SDK client the agent runner constructs: ``ClaudeSDKClient`` becomes a
# controllable ``_ScriptedSdkClient`` that yields real SDK messages, built by the
# shared helpers in ``_fakes``. That makes session reuse, the one-creation double-submit,
# the awaited interrupt, and guard release provable *at the fake seam* — the
# same fake client instance is observed receiving both prompts, etc.


class _ScriptedSdkClient:
    """Controllable ``ClaudeSDKClient`` double, patched where the agent runner builds it.

    A real :class:`OperatorSession` / :class:`OperatorRegistry` runs on top of
    this. Each turn's behaviour is supplied by ``responder`` — an async-generator
    function ``responder(client, prompt)`` that yields SDK messages. The
    instance records prompts and interrupt/aenter/aexit counts so a route test
    can prove session reuse, the awaited interrupt, and client teardown.

    ``interrupt`` is a coroutine whose body only runs when awaited, so a caller
    that forgot to ``await`` it would leave ``interrupt_calls`` at zero. Per-turn
    ``interrupted`` / ``reached_hold`` events let a test sequence a turn precisely.
    """

    def __init__(
        self, responder, *, aenter_delay: float = 0.0, query_error: Exception | None = None
    ) -> None:
        self.responder = responder
        self.aenter_delay = aenter_delay
        self.query_error = query_error
        self.prompts: list[str] = []
        self.query_calls = 0
        self.interrupt_calls = 0
        self.aenter_calls = 0
        self.aexit_calls = 0
        self._prompt: str | None = None
        self.interrupted = asyncio.Event()
        self.reached_hold = asyncio.Event()

    async def __aenter__(self):
        self.aenter_calls += 1
        if self.aenter_delay:
            await asyncio.sleep(self.aenter_delay)
        return self

    async def __aexit__(self, *exc):
        self.aexit_calls += 1
        return False

    async def query(self, prompt: str) -> None:
        self.query_calls += 1
        if self.query_error is not None:
            raise self.query_error
        self.prompts.append(prompt)
        self._prompt = prompt
        # Reset (never replace) the per-turn events so a test that captured a
        # reference before the turn started still observes the current turn, and
        # a later turn is not pre-interrupted by an earlier turn's signal.
        self.interrupted.clear()
        self.reached_hold.clear()

    async def interrupt(self) -> None:
        self.interrupt_calls += 1
        self.interrupted.set()

    async def receive_response(self):
        async for message in self.responder(self, self._prompt):
            yield message


# ---- Responder factories (async-generator functions) ---- #


def _clean_responder(text: str = "ok"):
    """One text block then a terminal result — a clean, prompt turn."""

    async def responder(_client, _prompt):
        yield assistant_message([TextBlock(text)])
        yield result_message(is_error=False)

    return responder


def _rich_responder():
    """A turn carrying every heavy/sensitive payload the strip filter must drop."""

    async def responder(_client, _prompt):
        yield assistant_message(
            [
                ThinkingBlock("secret chain of thought", "sig"),
                ToolUseBlock(
                    "tu_1", "mcp__osprey__channel_read", {"channel": "SR:BPM", "secret": "x"}
                ),
            ]
        )
        # The CLI sends a tool's result back in a user message.
        yield user_message([ToolResultBlock("tu_1", "large result body", is_error=False)])
        yield assistant_message([TextBlock("done")])
        yield result_message(is_error=False, total_cost_usd=0.9, duration_ms=42, num_turns=3)

    return responder


def _stall_responder():
    """Never emits — models an SDK that goes silent until interrupted.

    Yields nothing (so the turn times out / stays in-flight) and, crucially,
    yields nothing *after* the interrupt either, so no stale terminal event is
    left in the queue for a subsequent turn on the same session to misread.
    """

    async def responder(client, _prompt):
        client.reached_hold.set()
        await client.interrupted.wait()
        if False:  # pragma: no cover - present only to make this an async generator
            yield

    return responder


def _partial_then_hold_responder(text: str = "partial"):
    """Emit one partial event, then park until interrupted (no terminal)."""

    async def responder(client, _prompt):
        yield assistant_message([TextBlock(text)])
        client.reached_hold.set()
        await client.interrupted.wait()

    return responder


@contextlib.contextmanager
def _seam(responder, *, aenter_delay: float = 0.0, query_error: Exception | None = None):
    """Patch the runner's SDK client; yield a client factory with ``.created``.

    Every ``OperatorSession.start()`` builds a ``_ScriptedSdkClient(responder)`` via
    the factory; ``factory.created`` is the ordered list of every client made,
    so a test can assert exactly-one-creation and reach into the live client.
    """
    created: list[_ScriptedSdkClient] = []

    # ``ClaudeSDKClient``'s constructor, which the session calls with ``options`` by name.
    def factory(options=None):  # noqa: ARG001
        client = _ScriptedSdkClient(responder, aenter_delay=aenter_delay, query_error=query_error)
        created.append(client)
        return client

    factory.created = created  # type: ignore[attr-defined]

    with (
        patch.object(chat_module, "HAS_SDK", True),
        sdk_seam(factory),
    ):
        yield factory


def _req(registry, *, cwd: str = "/tmp", turn_timeout_s: float = 5.0):
    """A minimal ``Request`` double carrying the app.state the routes read."""
    state = SimpleNamespace(
        project_cwd=cwd, operator_registry=registry, chat_turn_timeout_s=turn_timeout_s
    )
    _seed_handoff_state(state)
    return SimpleNamespace(app=SimpleNamespace(state=state), is_disconnected=_connected)


async def _collect_sse(resp) -> list[dict]:
    """Drive a StreamingResponse's body iterator and parse its data frames."""
    chunks: list[str] = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return _data_frames("".join(chunks))


def _fast_handoff_clock(app):
    """Spend the hand-off's graces in fake time.

    ``HandoffState`` takes its clock and its sleep by injection precisely so a
    test need not sit out the two-second attach grace for real. Each sleep
    advances the clock by what it was asked to wait and yields instead.
    """
    state = handoff_module.get_state(app)
    now = [0.0]

    async def sleep(seconds):
        now[0] += seconds
        await asyncio.sleep(0)

    state.clock = lambda: now[0]
    state.sleep = sleep


async def _settled(task, *, ticks: int = 60, tick: float = 0.05):
    """Await *task*, which is expected to finish on its own within the ticks.

    Polls rather than ``wait_for`` so a task that overruns is reported as a
    failed assertion instead of being cancelled mid hand-off.
    """
    for _ in range(ticks):
        if task.done():
            break
        await asyncio.sleep(tick)
    assert task.done(), "the acquire never completed"
    return await task


async def _inflight_session(req, chat_id: str, make):
    """Create a real chat session and park it mid-turn on the stall responder.

    Returns ``(session, token)`` with a genuinely in-flight turn: the guard is
    held and the reader is running (blocked in ``receive_response``), so the
    registry reports the session busy. *make* is the :func:`_seam` factory the
    session's client was built by.

    Built through the route's own ``_acquire_chat_turn`` rather than by handing
    the pool a bare ``{}``. The pool now compares the environment a live entry
    was built from before reusing it, so a session seeded with an environment
    no route would produce is not reusable by the route — it would be torn
    down and rebuilt, and a test meaning to park one session would quietly get
    two.
    """
    session, token, _reused = await chat_module._acquire_chat_turn(req, chat_id)
    await session.send_prompt("hold")
    await asyncio.wait_for(make.created[-1].reached_hold.wait(), timeout=1.0)
    return session, token


class TestChatAtomicity:
    """Matrix 2: concurrent first requests for one chat_id create ONE session."""

    async def test_double_submit_shares_one_creation(self):
        with _seam(_clean_responder(), aenter_delay=0.03) as make:
            registry = OperatorRegistry()
            req = _req(registry)

            results = await asyncio.gather(
                chat_module.chat(req, chat_module.ChatRequest(prompt="p1", chat_id="c")),
                chat_module.chat(req, chat_module.ChatRequest(prompt="p2", chat_id="c")),
                return_exceptions=True,
            )

            # Exactly one SDK client was ever built — both joined one creation.
            assert len(make.created) == 1
            streaming = [r for r in results if not isinstance(r, Exception)]
            conflicts = [r for r in results if isinstance(r, HTTPException)]
            # The guard then serialises them: one turn runs, the other 409s.
            assert len(streaming) == 1
            assert len(conflicts) == 1 and conflicts[0].status_code == 409
            assert conflicts[0].detail["error"] == "turn_in_progress"

            await _collect_sse(streaming[0])  # drain the winner to release its guard
            await registry.cleanup_all()


class TestChatStatusMapIntegration:
    """Matrix 3 & 4: the wait on a turn in flight, and 429 (all busy) on real sessions."""

    async def test_second_prompt_waits_for_the_turn_in_flight(self):
        """The second prompt is queued behind the first, not refused.

        Taking the key for the Simple view when the Simple view already holds
        it and is working is a wait: the hand-off polls the chat until it is
        idle and then hands back the very same child. This replaces the
        immediate 409 the route used to answer — that refusal survives only as
        the guard's backstop for a turn that starts inside the gap between the
        wait ending and the guard being taken (``test_turn_in_progress_returns_409``).
        """
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            session, token = await _inflight_session(req, "c", make)

            second = asyncio.ensure_future(
                chat_module.chat(req, chat_module.ChatRequest(prompt="second", chat_id="c"))
            )
            for _ in range(5):
                await asyncio.sleep(0.05)
            assert not second.done()  # still waiting on the turn in flight
            assert len(make.created) == 1  # and waiting on the pooled chat, not a new one

            # End the parked turn: the guard goes first, so the wait sees idle.
            session.release_turn(token)
            make.created[0].interrupted.set()

            resp = await _settled(second)
            assert resp.media_type == "text/event-stream"
            assert len(make.created) == 1  # the same child answers the second prompt

            await registry.cleanup_all()

    async def test_a_third_prompt_arriving_mid_wait_is_refused(self):
        """Two overlapping acquires of one key: the newcomer is not queued.

        The first prompt is working and the second is inside the hand-off's
        wait, holding the key's pending slot. A third arrival has no way in —
        the slot names another connection — so the attach grace runs out and
        it is refused, rather than stacking a second unbounded wait on one
        key. The refusal is the same one the terminal answers with a 4409
        close, so the client already knows the slug.
        """
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            _fast_handoff_clock(req.app)
            session, token = await _inflight_session(req, "c", make)

            waiting = asyncio.ensure_future(
                chat_module.chat(req, chat_module.ChatRequest(prompt="second", chat_id="c"))
            )
            for _ in range(5):  # let the second request register its slot
                await asyncio.sleep(0)
            assert not waiting.done()

            with pytest.raises(HTTPException) as ei:
                await chat_module.chat(req, chat_module.ChatRequest(prompt="third", chat_id="c"))

            assert ei.value.status_code == 409
            assert ei.value.detail["error"] == "session_attached_elsewhere"
            assert len(make.created) == 1  # neither latecomer built anything

            waiting.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiting
            session.release_turn(token)
            make.created[0].interrupted.set()
            await registry.cleanup_all()

    async def test_all_busy_returns_429(self):
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry(chat_max_sessions=1)
            req = _req(registry)
            a, token = await _inflight_session(req, "A", make)

            with pytest.raises(HTTPException) as ei:
                await chat_module.chat(req, chat_module.ChatRequest(prompt="x", chat_id="B"))

            assert ei.value.status_code == 429
            assert ei.value.detail["error"] == "chat_capacity"
            assert len(make.created) == 1  # 'B' was never created

            a.release_turn(token)
            make.created[0].interrupted.set()
            await registry.cleanup_all()


class TestChatSessionResetContract:
    """Matrix 5: session_reset placement on both paths, eviction, and the negative."""

    def test_sse_new_session_resets_once_at_index_0_reused_omits(self):
        with _seam(_clean_responder("hi")) as make:
            app = _make_chat_app(OperatorRegistry())
            with TestClient(app) as client:
                f1 = _data_frames(
                    client.post("/api/chat", json={"prompt": "p1", "chat_id": "c"}).text
                )
                f2 = _data_frames(
                    client.post("/api/chat", json={"prompt": "p2", "chat_id": "c"}).text
                )

            # Fresh conversation: exactly one reset, and it is the FIRST frame.
            assert f1[0] == {"type": "session_reset"}
            assert sum(1 for f in f1 if f.get("type") == "session_reset") == 1
            # Negative: the reused turn carries no reset marker to misrender.
            assert all(f.get("type") != "session_reset" for f in f2)
            # Same client served both — persistence at the fake seam.
            assert len(make.created) == 1
            assert make.created[0].prompts == ["p1", "p2"]

    def test_buffered_new_session_resets_at_events0_reused_omits(self):
        with _seam(_clean_responder("hi")):
            app = _make_chat_app(OperatorRegistry())
            with TestClient(app) as client:
                p1 = client.post(
                    "/api/chat", params={"stream": "false"}, json={"prompt": "p1", "chat_id": "c"}
                ).json()
                p2 = client.post(
                    "/api/chat", params={"stream": "false"}, json={"prompt": "p2", "chat_id": "c"}
                ).json()

            assert p1["events"][0] == {"type": "session_reset"}
            assert sum(1 for e in p1["events"] if e.get("type") == "session_reset") == 1
            assert all(e.get("type") != "session_reset" for e in p2["events"])

    def test_eviction_makes_a_recreated_chat_fresh_again(self):
        """A re-create with no transcript to resume starts the conversation over."""
        with _seam(_clean_responder()) as make:
            app = _make_chat_app(OperatorRegistry(chat_max_sessions=1))
            with TestClient(app) as client:
                client.post("/api/chat", json={"prompt": "a", "chat_id": "A"})
                # 'A' is idle (its turn finished) → creating 'B' at cap 1 evicts it.
                client.post("/api/chat", json={"prompt": "b", "chat_id": "B"})
                fA2 = _data_frames(
                    client.post("/api/chat", json={"prompt": "a2", "chat_id": "A"}).text
                )

            # 'A' was rebuilt from scratch → a fresh reset leads its stream.
            assert fA2[0] == {"type": "session_reset"}
            # Three distinct creations: A#1, B, A#2.
            assert len(make.created) == 3

    def test_a_recreated_chat_that_resumes_carries_no_reset(self, transcripts_on_disk):
        """A child started on an existing transcript continues the conversation.

        The marker says "this conversation starts here", and after a resume it
        does not: the operator is looking at the same exchange, whether they
        left it in this view or in the terminal. So the re-created 'A' — a new
        process by every other measure — sends none.
        """
        transcripts_on_disk.add("A")
        with _seam(_clean_responder()) as make:
            app = _make_chat_app(OperatorRegistry(chat_max_sessions=1))
            with TestClient(app) as client:
                client.post("/api/chat", json={"prompt": "a", "chat_id": "A"})
                client.post("/api/chat", json={"prompt": "b", "chat_id": "B"})
                fA2 = _data_frames(
                    client.post("/api/chat", json={"prompt": "a2", "chat_id": "A"}).text
                )

            assert all(f.get("type") != "session_reset" for f in fA2)
            assert len(make.created) == 3  # still a new process, just not a new conversation


#: A key the posture route can address: a canonical, lowercase, bare UUID.
_ADDRESSABLE_CHAT_ID = "cccccccc-1111-2222-3333-444444444444"


async def _spawned_child_env(chat_id: str) -> dict[str, str]:
    """The environment the chat route hands the child it spawns for *chat_id*."""

    def make_client(options=None):  # noqa: ARG001
        return _ScriptedSdkClient(_clean_responder())

    with patch.object(chat_module, "HAS_SDK", True), sdk_seam(make_client) as captured:
        registry = OperatorRegistry()
        try:
            resp = await chat_module.chat(
                _req(registry), chat_module.ChatRequest(prompt="p", chat_id=chat_id)
            )
            await _collect_sse(resp)
        finally:
            await registry.cleanup_all()

    assert len(captured) == 1, "one turn on an empty key spawns exactly one child"
    return dict(captured[0].env)


class TestChatPostureSource:
    """The chat route stamps the child's posture source from the key grammar.

    The route stamps ``live`` only for a key the posture route can address and
    ``process`` for any other, and an auditor reads that marker off every
    envelope the child files. These tests read it from the options the spawned
    child was built with, so they see what the conditional computes rather than
    how it is spelled.
    """

    async def test_an_addressable_key_is_stamped_live(self, monkeypatch):
        """A canonical bare UUID is a key the posture route answers for, so the
        child says ``live`` and names that key."""
        env = await _spawned_child_env(_ADDRESSABLE_CHAT_ID)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == _ADDRESSABLE_CHAT_ID
        monkeypatch.setenv(POSTURE_SOURCE_ENV, env[POSTURE_SOURCE_ENV])
        assert posture_source() == POSTURE_SOURCE_LIVE

    @pytest.mark.parametrize(
        "chat_id",
        [
            pytest.param("user-42-chat-3", id="embedder-chosen"),
            pytest.param("CCCCCCCC-1111-2222-3333-444444444444", id="uppercase-uuid"),
            pytest.param("operator-deadbeef", id="operator-shaped"),
        ],
    )
    async def test_an_unaddressable_key_is_stamped_process(self, monkeypatch, chat_id):
        """The posture route refuses each of these keys, so no toggle can govern
        the child and it says ``process``; it still names the key, because the
        session marker is the ledger's join and is stamped whatever the source."""
        env = await _spawned_child_env(chat_id)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_PROCESS
        assert env[POSTURE_SESSION_ENV] == chat_id
        monkeypatch.setenv(POSTURE_SOURCE_ENV, env[POSTURE_SOURCE_ENV])
        assert posture_source() == POSTURE_SOURCE_PROCESS


class TestChatDeleteEndpointIntegration:
    """Matrix 6 & 7: DELETE on idle (prompt return) and busy (interrupt + teardown)."""

    async def test_delete_idle_returns_promptly_without_interrupt(self):
        with _seam(_clean_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            session, _ = await registry.get_or_create_chat_session("c", "/tmp", {})

            resp = await asyncio.wait_for(chat_module.delete_chat("c", req), timeout=2.0)

            assert resp.status_code == 204
            assert registry.get_chat_session("c") is None
            assert make.created[0].aexit_calls == 1  # client closed
            assert make.created[0].interrupt_calls == 0  # idle cancel short-circuits

    async def test_delete_busy_interrupts_bounded_and_tears_down(self):
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            session, _token = await _inflight_session(req, "c", make)

            resp = await asyncio.wait_for(chat_module.delete_chat("c", req), timeout=3.0)

            assert resp.status_code == 204
            assert registry.get_chat_session("c") is None
            assert make.created[0].interrupt_calls >= 1  # interrupt-signalled on teardown
            assert make.created[0].aexit_calls == 1  # client closed


class TestChatInterruptEndpointIntegration:
    """Matrix 8: signal-only interrupt awaits the client; no-op when nothing is running."""

    async def test_in_flight_turn_awaits_client_interrupt_signal_only(self):
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            session, token = await _inflight_session(req, "c", make)

            resp = await chat_module.interrupt_chat("c", req)

            assert resp.status_code == 204
            # interrupt() body ran → it was actually awaited, not discarded.
            assert make.created[0].interrupt_calls == 1
            # Signal-only: the guard is NOT released by the interrupt endpoint.
            assert session.in_flight is True

            session.release_turn(token)
            await registry.cleanup_all()

    async def test_idle_session_interrupt_is_noop_204(self):
        with _seam(_clean_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            await registry.get_or_create_chat_session("c", "/tmp", {})

            resp = await chat_module.interrupt_chat("c", req)

            assert resp.status_code == 204
            assert make.created[0].interrupt_calls == 0
            await registry.cleanup_all()


class TestChatTurnRelease:
    """Matrix 9 & 10: guard release on SDK-silence timeout and on cancelled-scope disconnect."""

    async def test_timeout_releases_guard_and_next_prompt_succeeds(self):
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry, turn_timeout_s=0.05)

            r1 = await chat_module.chat(req, chat_module.ChatRequest(prompt="p1", chat_id="c"))
            f1 = await _collect_sse(r1)

            # Turn 1 timed out: a TimeoutError frame, no result.
            assert any(
                f.get("type") == "error" and f.get("error_type") == "TimeoutError" for f in f1
            )
            assert all(f.get("type") != "result" for f in f1)

            session = registry.get_chat_session("c")
            assert session is not None
            assert session.in_flight is False  # guard released on the non-terminal exit

            # Let the detached quiesce finish before reusing the session.
            if session._quiesce_task is not None:
                await asyncio.wait_for(asyncio.shield(session._quiesce_task), timeout=2.0)

            # Turn 2 on the SAME session succeeds once the SDK responds.
            make.created[0].responder = _clean_responder("recovered")
            r2 = await chat_module.chat(req, chat_module.ChatRequest(prompt="p2", chat_id="c"))
            f2 = await _collect_sse(r2)
            assert {"type": "text", "content": "recovered"} in f2
            assert {"type": "result", "is_error": False} in f2

            await registry.cleanup_all()

    async def test_release_under_cancelled_scope_then_reuse(self):
        with _seam(_partial_then_hold_responder("partial")) as make:
            registry = OperatorRegistry()
            session, token = await registry.get_or_create_chat_session("c", "/tmp", {})
            token = session.acquire_turn()
            client = make.created[0]

            # Drive the real buffered handler as a task, then cancel it mid-turn.
            task = asyncio.create_task(
                chat_module._buffered_response(
                    session, token, "p", fresh_conversation=False, turn_timeout_s=30
                )
            )
            await asyncio.wait_for(client.reached_hold.wait(), timeout=1.0)
            # Ensure the handler has consumed the partial and re-parked on the queue.
            for _ in range(20):
                if session._queue.empty():
                    break
                await asyncio.sleep(0)
            assert session.in_flight is True

            # anyio-level cancellation of the handler's scope.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # Guard released in the finally despite the cancelled-scope exit.
            assert session.in_flight is False

            # The session is reusable for a fresh, clean turn.
            if session._quiesce_task is not None:
                await asyncio.wait_for(asyncio.shield(session._quiesce_task), timeout=2.0)
            client.responder = _clean_responder("after-cancel")
            token2 = session.acquire_turn()
            resp = await chat_module._buffered_response(
                session, token2, "p2", fresh_conversation=False, turn_timeout_s=5
            )
            payload = _json.loads(resp.body)
            assert payload["text"] == "after-cancel"
            assert payload["is_error"] is False

            await registry.cleanup_all()


class TestChatStripOnTheWire:
    """Matrix 11: heavy/sensitive payloads are stripped on BOTH transport paths."""

    def test_strip_applied_on_sse_path(self):
        with _seam(_rich_responder()):
            app = _make_chat_app(OperatorRegistry())
            client = TestClient(app)
            frames = _data_frames(
                client.post("/api/chat", json={"prompt": "x", "chat_id": "c"}).text
            )

        tool_use = [f for f in frames if f.get("type") == "tool_use"]
        thinking = [f for f in frames if f.get("type") == "thinking"]
        result = [f for f in frames if f.get("type") == "result"]

        assert tool_use and all("input" not in f for f in tool_use)
        assert not any(f.get("type") == "tool_result" for f in frames)
        assert thinking and all("content" not in f for f in thinking)
        assert result and all(set(f) == {"type", "is_error"} for f in result)
        # Cost/duration/turn metadata never reaches the wire on any frame.
        for f in frames:
            assert "total_cost_usd" not in f
            assert "duration_ms" not in f
            assert "num_turns" not in f

    def test_strip_applied_on_buffered_path_including_top_level(self):
        with _seam(_rich_responder()):
            app = _make_chat_app(OperatorRegistry())
            client = TestClient(app)
            payload = client.post(
                "/api/chat", params={"stream": "false"}, json={"prompt": "x", "chat_id": "c"}
            ).json()

        # Top-level reduced to exactly {text, events, is_error} (no error on success).
        assert set(payload) == {"text", "events", "is_error"}
        assert payload["text"] == "done"
        assert payload["is_error"] is False
        # Cost/duration/turn counts ABSENT (not present as None) at the top level.
        for key in ("total_cost_usd", "duration_ms", "num_turns", "cost", "duration"):
            assert key not in payload

        events = payload["events"]
        assert any(e.get("type") == "tool_use" and "input" not in e for e in events)
        assert not any(e.get("type") == "tool_result" for e in events)
        assert any(e.get("type") == "thinking" and "content" not in e for e in events)
        result = [e for e in events if e.get("type") == "result"]
        assert result and all(set(e) == {"type", "is_error"} for e in result)
        for e in events:
            assert "total_cost_usd" not in e
            assert "duration_ms" not in e
            assert "num_turns" not in e


# ---------------------------------------------------------------------------
# Pure-helper contracts (terminal detection + SSE wire format)
# ---------------------------------------------------------------------------


class TestIsTerminal:
    """is_terminal_event: which chat events end a turn's event stream."""

    @pytest.mark.parametrize(
        ("event", "terminal"),
        [({"type": "result"}, True), ({"type": "text", "content": "hi"}, False)],
        ids=["result", "text"],
    )
    def test_result_is_terminal(self, event, terminal):
        assert chat_module.is_terminal_event(event) is terminal

    def test_fatal_error_is_terminal(self):
        assert chat_module.is_terminal_event({"type": "error", "error_type": "Boom"}) is True

    def test_assistant_message_error_is_not_terminal(self):
        """A per-message SDK error is recoverable — the stream must continue."""
        assert (
            chat_module.is_terminal_event({"type": "error", "error_type": "AssistantMessageError"})
            is False
        )


class TestSseFormatting:
    def test_formats_as_sse_data_line(self):
        assert chat_module._sse({"type": "text"}) == 'data: {"type": "text"}\n\n'
