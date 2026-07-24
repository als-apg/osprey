"""Tests for the REST chat endpoint (`routes/chat.py`).

Task 1.4 (strip-for-chat): `_strip_for_chat` is the wire-hygiene filter applied
to every event before it reaches the chat API client. It drops heavy or
sensitive payloads (tool arguments, tool-result bodies, thinking text, and the
cost/duration/turn metadata on `result`) while preserving each event's identity
and light metadata.

The table below is anchored to the real event shapes produced by
`_message_to_events` in ``operator_session.py`` (text / thinking / tool_use /
tool_result / result / system / error) plus the ``session_reset`` marker the
control routes emit — so a change to those shapes that this filter should react
to will surface here.

Later tasks (chat-stream-route, chat-control-routes, integration) append their
own route-level test classes to this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import osprey.interfaces.web_terminal.routes.chat as chat_module
from osprey.interfaces.web_terminal.chat_session_pool import ChatCapacityError
from osprey.interfaces.web_terminal.operator_session import (
    OperatorRegistry,
    OperatorSession,
)
from osprey.interfaces.web_terminal.routes.chat import _strip_for_chat
from tests.interfaces.web_terminal.test_operator_session import (
    FakeAssistantMessage,
    FakeResultMessage,
    FakeSystemMessage,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeToolResultBlock,
    FakeToolUseBlock,
)

# ---- Representative events, one per type `_message_to_events` can emit. ----
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

    def test_tool_use_input_is_gone_entirely(self):
        """`input` (arguments) must not survive on a stripped tool_use."""
        out = _strip_for_chat(
            {
                "type": "tool_use",
                "tool_name": "Bash",
                "tool_name_raw": "Bash",
                "tool_use_id": "tu_x",
                "input": {"command": "rm -rf /"},
            }
        )
        assert "input" not in out
        assert out["tool_name"] == "Bash"

    def test_tool_result_content_is_gone_entirely(self):
        """`content` (result body) must not survive on a stripped tool_result."""
        out = _strip_for_chat(
            {
                "type": "tool_result",
                "tool_use_id": "tu_x",
                "content": [{"type": "text", "text": "..."}],
                "is_error": True,
            }
        )
        assert "content" not in out
        assert out["is_error"] is True

    def test_result_drops_cost_duration_turns(self):
        """Cost / duration / turn metadata must never reach the chat client."""
        out = _strip_for_chat(
            {
                "type": "result",
                "is_error": False,
                "total_cost_usd": 1.5,
                "duration_ms": 999,
                "num_turns": 7,
            }
        )
        assert set(out) == {"type", "is_error"}
        for leaked in ("total_cost_usd", "duration_ms", "num_turns"):
            assert leaked not in out

    def test_thinking_marker_survives_without_content(self):
        """A bare `{type: 'thinking'}` marker survives; the text does not."""
        out = _strip_for_chat({"type": "thinking", "content": "chain of thought"})
        assert out == {"type": "thinking"}

    def test_unknown_type_passes_through_unchanged(self):
        """An unrecognized event type is passed through untouched."""
        event = {"type": "keepalive", "extra": 1}
        assert _strip_for_chat(event) == event


# ---- Route-level smoke tests for the SSE branch (task chat-stream-route) ----
#
# These are intentionally light — they exercise the wire contract of the SSE
# branch (session_reset gating, per-event strip, guard release, and the
# 409/429/422/503 status map). The full scenario matrix lives in task
# chat-route-integration-tests.


class _FakeSdkClient:
    """Records signal-only `interrupt()` calls."""

    def __init__(self):
        self.interrupts = 0

    async def interrupt(self):
        self.interrupts += 1


class _FakeChatSession(OperatorSession):
    """OperatorSession with a preloaded queue instead of an SDK transport.

    Inherits the real turn guard and ``run_turn`` machine; only the transport
    (``send_prompt``) and the quiesce side effect are replaced, with counters.
    """

    def __init__(self, events):
        super().__init__(cwd="/tmp")
        self._events = list(events)
        self.quiesce_calls = 0
        self.release_calls = 0
        self.prompts: list[str] = []
        self._client = _FakeSdkClient()

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


class _FakeRegistry:
    def __init__(self, session=None, was_reused=False, capacity=False):
        self._session = session
        self._was_reused = was_reused
        self._capacity = capacity
        self.calls: list[str] = []
        self.terminated: list[str] = []

    async def get_or_create_chat_session(self, chat_id, cwd, env):
        self.calls.append(chat_id)
        if self._capacity:
            raise ChatCapacityError("all busy")
        return self._session, self._was_reused

    def get_chat_session(self, chat_id):
        return self._session

    async def terminate_chat_session(self, chat_id):
        self.terminated.append(chat_id)


def _make_chat_app(registry, turn_timeout_s=5) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_module.router)
    app.state.project_cwd = "/tmp"
    app.state.operator_registry = registry
    app.state.chat_turn_timeout_s = turn_timeout_s
    return app


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
        registry = _FakeRegistry(session=session, was_reused=False)
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

    def test_reused_session_has_no_session_reset(self):
        session = _FakeChatSession([{"type": "result", "is_error": False}])
        registry = _FakeRegistry(session=session, was_reused=True)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "again", "chat_id": "c1"})
        frames = _data_frames(resp.text)
        assert all(f.get("type") != "session_reset" for f in frames)

    def test_strip_applied_on_the_wire(self):
        session = _FakeChatSession(
            [
                {
                    "type": "tool_use",
                    "tool_name": "Bash",
                    "tool_name_raw": "Bash",
                    "tool_use_id": "tu_1",
                    "input": {"command": "rm -rf /"},
                },
                {"type": "result", "is_error": False},
            ]
        )
        registry = _FakeRegistry(session=session, was_reused=True)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})
        tool_frames = [f for f in _data_frames(resp.text) if f.get("type") == "tool_use"]
        assert tool_frames and all("input" not in f for f in tool_frames)

    def test_turn_in_progress_returns_409(self):
        session = _FakeChatSession([{"type": "result", "is_error": False}])
        session.acquire_turn()  # a turn is already held
        registry = _FakeRegistry(session=session, was_reused=True)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "turn_in_progress"

    def test_capacity_returns_429(self):
        registry = _FakeRegistry(capacity=True)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})
        assert resp.status_code == 429
        assert resp.json()["detail"]["error"] == "chat_capacity"

    def test_empty_prompt_returns_422(self):
        registry = _FakeRegistry(session=_FakeChatSession([]))
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "   ", "chat_id": "c1"})
        assert resp.status_code == 422

    def test_sdk_unavailable_returns_503(self, monkeypatch):
        monkeypatch.setattr(chat_module, "CLAUDE_SDK_AVAILABLE", False)
        registry = _FakeRegistry(session=_FakeChatSession([]))
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat", json={"prompt": "x", "chat_id": "c1"})
        assert resp.status_code == 503


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
        registry = _FakeRegistry(session=session, was_reused=False)
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

    def test_reused_session_omits_session_reset(self):
        session = _FakeChatSession([{"type": "result", "is_error": False}])
        registry = _FakeRegistry(session=session, was_reused=True)
        client = TestClient(_make_chat_app(registry))

        resp = client.post(
            "/api/chat", params={"stream": "false"}, json={"prompt": "hi", "chat_id": "c1"}
        )
        payload = resp.json()
        assert all(e.get("type") != "session_reset" for e in payload["events"])

    def test_terminal_error_returns_500_with_error_key(self):
        session = _FakeChatSession(
            [{"type": "error", "message": "boom", "error_type": "ClaudeSDKError"}]
        )
        registry = _FakeRegistry(session=session, was_reused=True)
        client = TestClient(_make_chat_app(registry))

        resp = client.post(
            "/api/chat", params={"stream": "false"}, json={"prompt": "hi", "chat_id": "c1"}
        )
        assert resp.status_code == 500
        payload = resp.json()
        assert set(payload) == {"text", "events", "is_error", "error"}
        assert payload["is_error"] is True
        assert payload["error"] == "boom"


class TestInterruptEndpoint:
    """POST /api/chat/{chat_id}/interrupt — signal-only, never releases."""

    def test_in_flight_turn_gets_interrupt_signal(self):
        session = _FakeChatSession([])
        session.acquire_turn()  # a turn is in flight
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat/c1/interrupt")
        assert resp.status_code == 204
        assert session._client.interrupts == 1
        # Signal-only: the guard is NOT released here.
        assert session.in_flight is True
        assert session.release_calls == 0

    def test_idle_session_is_noop_204(self):
        session = _FakeChatSession([])  # no turn in flight
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat/c1/interrupt")
        assert resp.status_code == 204
        assert session._client.interrupts == 0

    def test_unknown_chat_is_noop_204(self):
        registry = _FakeRegistry(session=None)
        client = TestClient(_make_chat_app(registry))

        resp = client.post("/api/chat/nope/interrupt")
        assert resp.status_code == 204


class TestDeleteEndpoint:
    """DELETE /api/chat/{chat_id} — delegates to terminate_chat_session."""

    def test_delete_delegates_and_returns_204(self):
        session = _FakeChatSession([])
        registry = _FakeRegistry(session=session)
        client = TestClient(_make_chat_app(registry))

        resp = client.delete("/api/chat/c1")
        assert resp.status_code == 204
        assert registry.terminated == ["c1"]

    def test_delete_unknown_chat_is_idempotent_204(self):
        registry = _FakeRegistry(session=None)
        client = TestClient(_make_chat_app(registry))

        resp = client.delete("/api/chat/nope")
        assert resp.status_code == 204
        assert registry.terminated == ["nope"]


class TestRouteRegistration:
    """Routes are flattened into the OpenAPI schema (include_router convention)."""

    def test_control_routes_registered(self):
        registry = _FakeRegistry(session=_FakeChatSession([]))
        paths = _make_chat_app(registry).openapi()["paths"]
        assert "/api/chat" in paths
        assert "/api/chat/{chat_id}/interrupt" in paths
        assert "/api/chat/{chat_id}" in paths
        assert "post" in paths["/api/chat/{chat_id}/interrupt"]
        assert "delete" in paths["/api/chat/{chat_id}"]


# ===========================================================================
# Full route-level scenario matrix (task chat-route-integration-tests)
# ===========================================================================
#
# Unlike the smoke classes above (which fake the whole session/registry), these
# tests run a REAL ``OperatorRegistry`` + ``OperatorSession`` and patch only the
# SDK seam inside ``operator_session``: ``ClaudeSDKClient`` becomes a
# controllable ``_ScriptedSdkClient`` and the SDK message/block types become the
# ``Fake*`` doubles reused from ``test_operator_session`` so ``_message_to_events``
# converts our fakes. That makes session reuse, the one-creation double-submit,
# the awaited interrupt, and guard release provable *at the fake seam* — the
# same fake client instance is observed receiving both prompts, etc.

_OS = "osprey.interfaces.web_terminal.operator_session."


class _ScriptedSdkClient:
    """Controllable ``ClaudeSDKClient`` double, patched at the operator_session seam.

    A real :class:`OperatorSession` / :class:`OperatorRegistry` runs on top of
    this. Each turn's behaviour is supplied by ``responder`` — an async-generator
    function ``responder(client, prompt)`` that yields SDK-message doubles. The
    instance records prompts and interrupt/aenter/aexit counts so a route test
    can prove session reuse, the awaited interrupt, and client teardown.

    ``interrupt`` is a coroutine whose body only runs when awaited, so a caller
    that forgot to ``await`` it would leave ``interrupt_calls`` at zero. Per-turn
    ``interrupted`` / ``reached_hold`` events let a test sequence a turn precisely.
    """

    def __init__(self, responder, *, aenter_delay: float = 0.0) -> None:
        self.responder = responder
        self.aenter_delay = aenter_delay
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

    async def responder(client, prompt):
        yield FakeAssistantMessage([FakeTextBlock(text)])
        yield FakeResultMessage(is_error=False)

    return responder


def _rich_responder():
    """A turn carrying every heavy/sensitive payload the strip filter must drop."""

    async def responder(client, prompt):
        yield FakeAssistantMessage(
            [
                FakeThinkingBlock("secret chain of thought"),
                FakeToolUseBlock(
                    "mcp__osprey__channel_read", "tu_1", {"channel": "SR:BPM", "secret": "x"}
                ),
                FakeToolResultBlock("tu_1", "large result body", is_error=False),
                FakeTextBlock("done"),
            ]
        )
        yield FakeResultMessage(is_error=False, total_cost_usd=0.9, duration_ms=42, num_turns=3)

    return responder


def _stall_responder():
    """Never emits — models an SDK that goes silent until interrupted.

    Yields nothing (so the turn times out / stays in-flight) and, crucially,
    yields nothing *after* the interrupt either, so no stale terminal event is
    left in the queue for a subsequent turn on the same session to misread.
    """

    async def responder(client, prompt):
        client.reached_hold.set()
        await client.interrupted.wait()
        if False:  # pragma: no cover - present only to make this an async generator
            yield

    return responder


def _partial_then_hold_responder(text: str = "partial"):
    """Emit one partial event, then park until interrupted (no terminal)."""

    async def responder(client, prompt):
        yield FakeAssistantMessage([FakeTextBlock(text)])
        client.reached_hold.set()
        await client.interrupted.wait()

    return responder


@contextlib.contextmanager
def _seam(responder, *, aenter_delay: float = 0.0):
    """Patch the operator_session SDK seam; yield a client factory with ``.created``.

    Every ``OperatorSession.start()`` builds a ``_ScriptedSdkClient(responder)`` via
    the factory; ``factory.created`` is the ordered list of every client made,
    so a test can assert exactly-one-creation and reach into the live client.
    """
    created: list[_ScriptedSdkClient] = []

    def factory(options=None):
        client = _ScriptedSdkClient(responder, aenter_delay=aenter_delay)
        created.append(client)
        return client

    factory.created = created  # type: ignore[attr-defined]

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(chat_module, "CLAUDE_SDK_AVAILABLE", True))
        stack.enter_context(patch(_OS + "CLAUDE_SDK_AVAILABLE", True))
        stack.enter_context(patch(_OS + "ClaudeSDKClient", factory))
        stack.enter_context(patch(_OS + "AssistantMessage", FakeAssistantMessage))
        stack.enter_context(patch(_OS + "ResultMessage", FakeResultMessage))
        stack.enter_context(patch(_OS + "SystemMessage", FakeSystemMessage))
        stack.enter_context(patch(_OS + "TextBlock", FakeTextBlock))
        stack.enter_context(patch(_OS + "ThinkingBlock", FakeThinkingBlock))
        stack.enter_context(patch(_OS + "ToolUseBlock", FakeToolUseBlock))
        stack.enter_context(patch(_OS + "ToolResultBlock", FakeToolResultBlock))
        stack.enter_context(patch(_OS + "validate_project_directory", lambda cwd: []))
        stack.enter_context(
            patch(
                _OS + "build_system_prompt", lambda tz: {"type": "preset", "preset": "claude_code"}
            )
        )
        stack.enter_context(patch(_OS + "get_facility_timezone", lambda: None))
        yield factory


def _req(registry, *, cwd: str = "/tmp", turn_timeout_s: float = 5.0):
    """A minimal ``Request`` double carrying the app.state the routes read."""
    state = SimpleNamespace(
        project_cwd=cwd, operator_registry=registry, chat_turn_timeout_s=turn_timeout_s
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


async def _collect_sse(resp) -> list[dict]:
    """Drive a StreamingResponse's body iterator and parse its data frames."""
    chunks: list[str] = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return _data_frames("".join(chunks))


async def _inflight_session(registry, chat_id: str, *, cwd: str = "/tmp"):
    """Create a real chat session and park it mid-turn on the stall responder.

    Returns ``(session, token)`` with a genuinely in-flight turn: the guard is
    held and the reader is running (blocked in ``receive_response``), so the
    registry reports the session busy.
    """
    session, _ = await registry.get_or_create_chat_session(chat_id, cwd, {})
    token = session.acquire_turn()
    await session.send_prompt("hold")
    await asyncio.wait_for(session._client.reached_hold.wait(), timeout=1.0)
    return session, token


class TestChatMultiTurnPersistence:
    """Matrix 1: two prompts on one chat_id reach the SAME fake client."""

    async def test_same_client_receives_both_prompts(self):
        with _seam(_clean_responder("hi")) as make:
            registry = OperatorRegistry()
            req = _req(registry)

            r1 = await chat_module.chat(req, chat_module.ChatRequest(prompt="p1", chat_id="c"))
            f1 = await _collect_sse(r1)
            r2 = await chat_module.chat(req, chat_module.ChatRequest(prompt="p2", chat_id="c"))
            f2 = await _collect_sse(r2)

            # Exactly one session/client — reuse proven at the fake seam.
            assert len(make.created) == 1
            assert make.created[0].prompts == ["p1", "p2"]
            assert make.created[0].query_calls == 2
            # Both turns completed with a (stripped) result.
            assert {"type": "result", "is_error": False} in f1
            assert {"type": "result", "is_error": False} in f2
            await registry.cleanup_all()


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
    """Matrix 3 & 4: 409 (turn in flight) and 429 (all busy) on real sessions."""

    async def test_second_prompt_while_in_flight_returns_409(self):
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry()
            req = _req(registry)
            session, token = await _inflight_session(registry, "c")

            with pytest.raises(HTTPException) as ei:
                await chat_module.chat(req, chat_module.ChatRequest(prompt="second", chat_id="c"))

            assert ei.value.status_code == 409
            assert ei.value.detail["error"] == "turn_in_progress"
            assert len(make.created) == 1  # no new session for the rejected prompt

            session.release_turn(token)
            session._client.interrupted.set()
            await registry.cleanup_all()

    async def test_all_busy_returns_429(self):
        with _seam(_stall_responder()) as make:
            registry = OperatorRegistry(chat_max_sessions=1)
            req = _req(registry)
            a, token = await _inflight_session(registry, "A")

            with pytest.raises(HTTPException) as ei:
                await chat_module.chat(req, chat_module.ChatRequest(prompt="x", chat_id="B"))

            assert ei.value.status_code == 429
            assert ei.value.detail["error"] == "chat_capacity"
            assert len(make.created) == 1  # 'B' was never created

            a.release_turn(token)
            a._client.interrupted.set()
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
            session, _token = await _inflight_session(registry, "c")

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
            session, token = await _inflight_session(registry, "c")

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
                    session, token, "p", was_reused=True, turn_timeout_s=30
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
                session, token2, "p2", was_reused=True, turn_timeout_s=5
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
        tool_result = [f for f in frames if f.get("type") == "tool_result"]
        thinking = [f for f in frames if f.get("type") == "thinking"]
        result = [f for f in frames if f.get("type") == "result"]

        assert tool_use and all("input" not in f for f in tool_use)
        assert tool_result and all("content" not in f for f in tool_result)
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
        assert any(e.get("type") == "tool_result" and "content" not in e for e in events)
        assert any(e.get("type") == "thinking" and "content" not in e for e in events)
        result = [e for e in events if e.get("type") == "result"]
        assert result and all(set(e) == {"type", "is_error"} for e in result)
        for e in events:
            assert "total_cost_usd" not in e
            assert "duration_ms" not in e
            assert "num_turns" not in e
