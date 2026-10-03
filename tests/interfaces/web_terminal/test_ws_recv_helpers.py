"""The deadline-bounded websocket readers fail fast and say what they saw."""

from __future__ import annotations

import time

import anyio
import pytest
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket

from tests.interfaces.web_terminal._ws import json_frames_until, recv_frame, recv_json

SHORT = 0.3


async def _quiet(ws: WebSocket) -> None:
    await ws.accept()
    await anyio.sleep_forever()


async def _one_then_quiet(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "output", "data": "x"})
    await anyio.sleep_forever()


async def _mixed(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_bytes(b"\x1b[2J")
    await ws.send_json({"type": "output", "data": "x"})
    await ws.send_json({"type": "session_info", "session_id": "s-1"})
    await anyio.sleep_forever()


async def _chatty(ws: WebSocket) -> None:
    await ws.accept()
    for _ in range(40):
        await ws.send_json({"type": "output"})
    await anyio.sleep_forever()


async def _closes(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "handoff_pending", "busy": False})
    await ws.close(code=4409)


@pytest.fixture()
def client():
    app = Starlette(
        routes=[
            WebSocketRoute("/quiet", _quiet),
            WebSocketRoute("/one-then-quiet", _one_then_quiet),
            WebSocketRoute("/mixed", _mixed),
            WebSocketRoute("/chatty", _chatty),
            WebSocketRoute("/closes", _closes),
        ]
    )
    with TestClient(app) as c:
        yield c


def test_a_quiet_handler_fails_within_the_deadline_and_names_the_frame(client):
    started = time.monotonic()
    with client.websocket_connect("/quiet") as ws:
        with pytest.raises(AssertionError) as failure:
            recv_json(ws, "session_info", within=SHORT)
        waited = time.monotonic() - started
    assert str(failure.value) == "no 'session_info' frame within 0.3 s; JSON types seen: []"
    assert waited < 3.0
    assert time.monotonic() - started < 3.0  # leaving the session did not hang either


def test_the_deadline_message_lists_the_json_types_that_did_arrive(client):
    with client.websocket_connect("/one-then-quiet") as ws:
        with pytest.raises(AssertionError, match=r"within 0\.3 s; JSON types seen: \['output'\]$"):
            recv_json(ws, "session_info", within=SHORT)


def test_the_wanted_frame_is_returned_and_binary_frames_are_skipped(client):
    with client.websocket_connect("/mixed") as ws:
        assert recv_json(ws, "session_info") == {"type": "session_info", "session_id": "s-1"}
    with client.websocket_connect("/mixed") as ws:
        frames = json_frames_until(ws, "session_info")
    assert [f["type"] for f in frames] == ["output", "session_info"]


def test_too_many_frames_without_the_wanted_one_fails_and_lists_them(client):
    with client.websocket_connect("/chatty") as ws:
        with pytest.raises(AssertionError) as failure:
            recv_json(ws, "session_info", max_frames=3)
    assert str(failure.value) == (
        "no 'session_info' frame within 3 frames; JSON types seen: ['output', 'output', 'output']"
    )


def test_a_close_before_the_wanted_frame_names_the_code(client):
    with client.websocket_connect("/closes") as ws:
        with pytest.raises(AssertionError) as failure:
            recv_json(ws, "session_info", within=SHORT)
    assert str(failure.value) == (
        "socket closed (4409) before 'session_info'; JSON types seen: ['handoff_pending']"
    )


def test_recv_frame_hands_back_the_close_frame_and_fails_on_silence(client):
    with client.websocket_connect("/closes") as ws:
        assert recv_json(ws, "handoff_pending")["busy"] is False
        closed = recv_frame(ws)
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 4409
    with client.websocket_connect("/quiet") as ws:
        with pytest.raises(AssertionError, match=r"^no frame within 0\.3 s$"):
            recv_frame(ws, within=SHORT)
