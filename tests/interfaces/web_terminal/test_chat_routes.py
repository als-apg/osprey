"""Tests for the REST chat endpoint (POST /api/chat).

Drives the route against a fake OperatorSession so the SSE / buffered event
loops, terminal-event detection, and error translation are exercised without
spawning a real Claude Agent SDK subprocess.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes import chat as chat_mod
from osprey.interfaces.web_terminal.routes.chat import _is_terminal, _sse, router


@pytest.fixture
def app(tmp_path):
    application = FastAPI()
    application.include_router(router)
    application.state.project_cwd = str(tmp_path)
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


class _FakeSession:
    """Minimal OperatorSession stand-in feeding a fixed event list via _queue."""

    def __init__(self, events, *, send_error: Exception | None = None):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._events = events
        self._send_error = send_error
        self.stopped = False

    async def start(self):
        pass

    async def send_prompt(self, prompt):
        if self._send_error is not None:
            raise self._send_error
        for e in self._events:
            await self._queue.put(e)

    async def stop(self):
        self.stopped = True


def _install_session(session):
    """Patch OperatorSession + build_clean_env + SDK flag for the chat module."""
    return (
        patch.object(chat_mod, "CLAUDE_SDK_AVAILABLE", True),
        patch.object(chat_mod, "build_clean_env", MagicMock(return_value={})),
        patch.object(chat_mod, "OperatorSession", lambda cwd, env: session),
    )


def _run_with(session, fn):
    p1, p2, p3 = _install_session(session)
    with p1, p2, p3:
        return fn()


class TestIsTerminal:
    def test_result_is_terminal(self):
        assert _is_terminal({"type": "result"}) is True

    def test_fatal_error_is_terminal(self):
        assert _is_terminal({"type": "error", "error_type": "Boom"}) is True

    def test_assistant_message_error_is_not_terminal(self):
        assert _is_terminal({"type": "error", "error_type": "AssistantMessageError"}) is False

    def test_text_is_not_terminal(self):
        assert _is_terminal({"type": "text", "content": "hi"}) is False


class TestSseFormatting:
    def test_formats_as_sse_data_line(self):
        assert _sse({"type": "text"}) == 'data: {"type": "text"}\n\n'


class TestChatGuards:
    def test_sdk_unavailable_returns_503(self, client):
        with patch.object(chat_mod, "CLAUDE_SDK_AVAILABLE", False):
            resp = client.post("/api/chat?stream=false", json={"prompt": "hi"})
        assert resp.status_code == 503

    def test_empty_prompt_returns_422(self, client):
        with patch.object(chat_mod, "CLAUDE_SDK_AVAILABLE", True):
            resp = client.post("/api/chat?stream=false", json={"prompt": "   "})
        assert resp.status_code == 422


class TestBufferedMode:
    def test_collects_text_and_result(self, client):
        session = _FakeSession(
            [
                {"type": "text", "content": "Hello "},
                {"type": "text", "content": "world"},
                {
                    "type": "result",
                    "is_error": False,
                    "total_cost_usd": 0.02,
                    "duration_ms": 120,
                    "num_turns": 1,
                },
            ]
        )
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=false", json={"prompt": "hi"})
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        assert data["is_error"] is False
        assert data["total_cost_usd"] == 0.02
        assert data["num_turns"] == 1
        assert session.stopped is True

    def test_keepalive_events_are_skipped(self, client):
        session = _FakeSession(
            [
                {"type": "keepalive"},
                {"type": "text", "content": "ok"},
                {"type": "result", "is_error": False},
            ]
        )
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=false", json={"prompt": "hi"})
        )
        data = resp.json()
        assert data["text"] == "ok"
        # keepalive is not appended to the events list.
        assert all(e.get("type") != "keepalive" for e in data["events"])

    def test_fatal_error_event_returns_500(self, client):
        session = _FakeSession(
            [{"type": "error", "error_type": "ProviderError", "message": "backend down"}]
        )
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=false", json={"prompt": "hi"})
        )
        assert resp.status_code == 500
        data = resp.json()
        assert data["is_error"] is True
        assert data["error"] == "backend down"

    def test_assistant_message_error_does_not_terminate(self, client):
        """A non-fatal AssistantMessageError is buffered but the loop keeps
        going until the result event — it must not 500."""
        session = _FakeSession(
            [
                {"type": "error", "error_type": "AssistantMessageError", "message": "partial"},
                {"type": "text", "content": "recovered"},
                {"type": "result", "is_error": False},
            ]
        )
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=false", json={"prompt": "hi"})
        )
        assert resp.status_code == 200
        assert resp.json()["text"] == "recovered"

    def test_timeout_returns_504(self, client, monkeypatch):
        monkeypatch.setattr(chat_mod, "EVENT_TIMEOUT_S", 0.01)
        session = _FakeSession([])  # queue stays empty → wait_for times out
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=false", json={"prompt": "hi"})
        )
        assert resp.status_code == 504
        assert resp.json()["error"] == "Timeout waiting for response"


class TestStreamingMode:
    def _events(self, text: str) -> list[dict]:
        lines = [ln[len("data: ") :] for ln in text.splitlines() if ln.startswith("data: ")]
        return [json.loads(ln) for ln in lines]

    def test_streams_events_until_terminal(self, client):
        session = _FakeSession(
            [
                {"type": "keepalive"},
                {"type": "text", "content": "streamed"},
                {"type": "result", "is_error": False},
            ]
        )
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=true", json={"prompt": "hi"})
        )
        assert resp.status_code == 200
        events = self._events(resp.text)
        types = [e["type"] for e in events]
        # keepalive filtered out; text + terminal result delivered.
        assert "keepalive" not in types
        assert types == ["text", "result"]

    def test_timeout_yields_error_event(self, client, monkeypatch):
        monkeypatch.setattr(chat_mod, "EVENT_TIMEOUT_S", 0.01)
        session = _FakeSession([])
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=true", json={"prompt": "hi"})
        )
        events = self._events(resp.text)
        assert events[-1]["error_type"] == "TimeoutError"

    def test_send_prompt_exception_yields_error_event(self, client):
        session = _FakeSession([], send_error=RuntimeError("spawn failed"))
        resp = _run_with(
            session, lambda: client.post("/api/chat?stream=true", json={"prompt": "hi"})
        )
        events = self._events(resp.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["message"] == "spawn failed"
        assert session.stopped is True
