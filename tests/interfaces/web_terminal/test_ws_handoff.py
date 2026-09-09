"""``terminal_ws`` through the hand-off door.

Every PTY the terminal handler serves now comes from ``acquire_surface``:
the session key is one session whichever view shows it, so a terminal that
takes a key the chat surface holds is a hand-off, not a second process. What
the handler owes the client around that door:

- ``handoff_pending`` before the wait whenever the chat surface holds the
  key, ``session_info`` once the terminal has it;
- a busy chat is waited on, and ``interrupt=1`` on the resume URL cuts its
  turn short instead;
- a client that leaves during the wait gets nothing, and the chat it was
  waiting on is left alone;
- a newer terminal on the key displaces the older one, which is closed
  with 4409; an outgoing child that survives its kill refuses the terminal
  with 4503; a hand-off error is an ``error`` frame and a close;
- a resize sent while waiting sizes the spawn, one that lands after the
  spawn is applied once the door returns, and a reused PTY is resized to
  the client's size by the handler itself;
- the spawn resumes the key's current transcript when one is on disk and
  starts fresh under the key otherwise;
- ``switch_session`` to a chat-held key hands off the same way.

Harness as in ``test_ws_resume_confirm.py``: a real app and ``PtyRegistry``
with ``_spawn_session`` patched to a fake, and a stand-in operator registry
whose chat pool is the phase (c) fake.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid as uuid_mod
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from osprey.interfaces.web_terminal.session_handoff import (
    WS_CLOSE_OUTGOING_RUNNING,
    WS_CLOSE_SESSION_ATTACHED,
    HandoffError,
    HandoffState,
    get_state,
)
from tests.interfaces.web_terminal._fakes import FakeClock, FakePtySession
from tests.interfaces.web_terminal.test_handoff_phase_c import Chat, ChatPool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ObservedPty(FakePtySession):
    """The resume-confirm fake, counting the output loops reading it."""

    def __init__(self):
        super().__init__()
        self.readers = 0

    async def read_output(self):
        self.readers += 1
        try:
            async for chunk in super().read_output():
                yield chunk
        finally:
            self.readers -= 1


@dataclass
class Spawn:
    """One call the registry made to spawn a PTY, and what it got."""

    command: list[str]
    rows: int
    cols: int
    session: ObservedPty


class ChatRegistry:
    """An ``operator_registry`` whose chat pool the door inspects and tears down."""

    def __init__(self) -> None:
        self.chats = ChatPool()

    def hold(self, key: str, chat: Chat | None = None) -> Chat:
        chat = chat or Chat()
        self.chats.sessions[key] = chat
        return chat

    def has_chat_key(self, session_id: str) -> bool:
        return self.chats.has_key(session_id)

    async def cleanup_all(self) -> None:
        pass


@pytest.fixture()
def app(tmp_path):
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


@pytest.fixture()
def sessions_dir(tmp_path):
    d = tmp_path / "claude_sessions"
    d.mkdir()
    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: d):
        yield d


def _patch_spawn(app, *, failing: str | None = None) -> list[Spawn]:
    """Replace ``_spawn_session`` so no real PTY is created; record each call.

    A spawn whose command names *failing* raises ``OSError`` instead, the way
    a CLI that cannot be executed does.
    """
    spawns: list[Spawn] = []

    def tracked_spawn(command, rows, cols, extra_env, cwd=None):
        if failing is not None and failing in command:
            raise OSError("cannot spawn")
        session = ObservedPty()
        spawns.append(Spawn(list(command), rows, cols, session))
        return session

    app.state.pty_registry._spawn_session = tracked_spawn
    return spawns


def _chats(app) -> ChatRegistry:
    registry = ChatRegistry()
    app.state.operator_registry = registry
    return registry


def _uuid() -> str:
    return str(uuid_mod.uuid4())


def _resume_url(session_id: str, *, interrupt: bool = False) -> str:
    url = f"/ws/terminal?session_id={session_id}&mode=resume"
    return f"{url}&interrupt=1" if interrupt else url


def _send_resize(ws, cols: int = 80, rows: int = 24) -> None:
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


def _json_frames_until(ws, msg_type: str, max_frames: int = 30) -> list[dict]:
    """Every JSON frame up to and including the first of *msg_type*."""
    collected: list[dict] = []
    for _ in range(max_frames):
        raw = ws.receive()
        if raw.get("type") == "websocket.close":
            raise AssertionError(f"socket closed ({raw.get('code')}) before '{msg_type}'")
        if "text" in raw:
            data = json.loads(raw["text"])
            collected.append(data)
            if data.get("type") == msg_type:
                return collected
    raise AssertionError(f"'{msg_type}' not received; got {[d.get('type') for d in collected]}")


def _recv_json(ws, msg_type: str) -> dict:
    return _json_frames_until(ws, msg_type)[-1]


def _sync(ws, session_id: str) -> None:
    """Round-trip through the main loop: a switch to the current key is a no-op answer."""
    ws.send_json({"type": "switch_session", "session_id": session_id})
    _recv_json(ws, "session_switched")


def _until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate(), "condition not met in time"


def _fake_clock(app) -> FakeClock:
    """Run the door's grace and death waits on a clock that only moves when slept."""
    clock = FakeClock()
    app.state.handoff = HandoffState(clock=clock, sleep=clock.sleep)
    return clock


# ---------------------------------------------------------------------------
# A chat-held key: the frames and the spawn
# ---------------------------------------------------------------------------


def test_a_chat_held_key_is_handed_off_with_the_pending_frame_first(app, sessions_dir):
    """``handoff_pending`` goes out before the wait, ``session_info`` after the door."""
    sid = _uuid()
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        chat = _chats(app).hold(sid)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            frames = _json_frames_until(ws, "session_info")

    assert [f["type"] for f in frames] == ["handoff_pending", "session_info"]
    # An idle chat has no turn to finish: the frame says so, and the client
    # shows a restart rather than a wait.
    assert frames[0] == {"type": "handoff_pending", "busy": False}
    assert frames[-1]["session_id"] == sid
    assert chat.teardowns == 1
    assert app.state.operator_registry.chats.get(sid) is None
    assert len(spawns) == 1


def test_a_key_with_no_transcript_starts_fresh_under_the_key(app, sessions_dir):
    """A chat that never wrote a transcript hands off to ``--session-id <key>``."""
    sid = _uuid()
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        _chats(app).hold(sid)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            _recv_json(ws, "session_info")

    command = spawns[0].command
    assert command[command.index("--session-id") + 1] == sid
    assert "--resume" not in command


def test_the_spawn_resumes_the_keys_current_transcript(app, sessions_dir):
    """The key's transcript may have moved (``/clear``); the spawn resumes the current one."""
    sid, transcript = _uuid(), _uuid()
    (sessions_dir / f"{transcript}.jsonl").write_text("")
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        _chats(app).hold(sid)
        app.state.transcript_map = {sid: transcript}
        app.state.transcript_map_provisional = False
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            _recv_json(ws, "session_info")

    command = spawns[0].command
    assert command[command.index("--resume") + 1] == transcript
    assert "--session-id" not in command


def test_a_spawn_at_capacity_evicts_the_oldest_background_pty_off_the_loop(app, sessions_dir):
    """The pool's eviction kill blocks; the handler takes the victim out and kills it in a thread."""
    sid, background = _uuid(), _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")

    class EvictedPty(ObservedPty):
        killed_on_loop: bool | None = None

        def terminate(self):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                self.killed_on_loop = False
            else:
                self.killed_on_loop = True
            super().terminate()

    victim = EvictedPty()
    with TestClient(app) as client:
        # The lifespan installs the app's registry; a one-slot pool replaces it.
        registry = PtyRegistry(max_background=1)
        app.state.pty_registry = registry
        registry._sessions[background] = victim
        spawns = _patch_spawn(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            _recv_json(ws, "session_info")
            assert background not in registry._sessions
            assert len(spawns) == 1 and registry.get_session(sid) is spawns[0].session

    assert victim.killed_on_loop is False


def test_a_free_key_gets_no_pending_frame(app, sessions_dir):
    """Nothing to wait on, nothing to announce: the first frame is the confirmation."""
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    with TestClient(app) as client:
        _patch_spawn(app)
        _chats(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            frames = _json_frames_until(ws, "session_info")

    assert [f["type"] for f in frames] == ["session_info"]


# ---------------------------------------------------------------------------
# A busy chat: wait, interrupt, leave
# ---------------------------------------------------------------------------


def test_a_busy_chat_is_waited_on(app, sessions_dir):
    """The terminal is confirmed once the chat's turn ends, and not before."""
    sid = _uuid()
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        chat = _chats(app).hold(sid, Chat(busy=True))
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            # Mid-turn: the frame says the wait is real, so the client shows
            # it as one from the first moment.
            assert _recv_json(ws, "handoff_pending")["busy"] is True
            time.sleep(0.3)
            assert spawns == []
            assert app.state.operator_registry.chats.get(sid) is chat

            chat.is_busy = False
            assert _recv_json(ws, "session_info")["session_id"] == sid

    assert chat.teardowns == 1
    assert len(spawns) == 1


def test_interrupt_cuts_the_chat_turn_short(app, sessions_dir):
    """``interrupt=1`` on the resume URL is the "stop and switch now" path."""
    sid = _uuid()
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        chat = _chats(app).hold(sid, Chat(busy=True))
        with client.websocket_connect(_resume_url(sid, interrupt=True)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid

    assert chat.teardowns == 1
    assert len(spawns) == 1


def test_leaving_during_the_wait_sends_nothing_and_spares_the_chat(app, sessions_dir):
    """A closed socket ends the wait: no spawn, no frame, the chat untouched, the key clean."""
    sid = _uuid()
    with TestClient(app) as client:
        registry = app.state.pty_registry
        spawns = _patch_spawn(app)
        chat = _chats(app).hold(sid, Chat(busy=True))
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            _recv_json(ws, "handoff_pending")

        _until(lambda: sid not in get_state(app).pending)
        assert not registry.is_reserved(sid)
        assert app.state.operator_registry.chats.get(sid) is chat
        assert chat.teardowns == 0
        assert spawns == []
        assert registry.get_session(sid) is None


# ---------------------------------------------------------------------------
# Refusals and errors
# ---------------------------------------------------------------------------


def test_a_newer_terminal_displaces_the_older_one_with_4409(app, sessions_dir):
    """Same key, second socket: the first is closed with 4409, the PTY is reused."""
    with TestClient(app) as client, ExitStack() as stack:
        spawns = _patch_spawn(app)
        _chats(app)
        first = stack.enter_context(client.websocket_connect("/ws/terminal"))
        _send_resize(first)
        sid = _recv_json(first, "session_info")["session_id"]

        second = stack.enter_context(client.websocket_connect(_resume_url(sid)))
        _send_resize(second)
        assert _recv_json(second, "session_info")["session_id"] == sid

        closed = first.receive()
        assert closed["type"] == "websocket.close"
        assert closed["code"] == WS_CLOSE_SESSION_ATTACHED
        # The displaced handler's output loop is stopped before the close
        # handshake completes: one reader on the PTY, the new terminal's.
        _until(lambda: spawns[0].session.readers == 1)

    assert len(spawns) == 1


def test_an_outgoing_child_that_survives_its_kill_is_refused_with_4503(app, sessions_dir):
    """The chat child would not die: nothing is spawned, the chat is pooled again, 4503."""
    sid = _uuid()
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        _fake_clock(app)
        chat = _chats(app).hold(sid, Chat(exits=False))
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            _recv_json(ws, "handoff_pending")
            closed = ws.receive()

    assert closed["type"] == "websocket.close"
    assert closed["code"] == WS_CLOSE_OUTGOING_RUNNING
    assert spawns == []
    assert app.state.operator_registry.chats.get(sid) is chat
    assert app.state.pty_registry.get_session(sid) is None


def test_a_handoff_error_is_an_error_frame_and_a_close(app, sessions_dir):
    """An error from inside the door reaches the client as ``error``, then the socket closes."""
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    failing = AsyncMock(side_effect=HandoffError.vanished(sid, "simple"))
    with (
        patch("osprey.interfaces.web_terminal.session_handoff.acquire_surface", failing),
        TestClient(app) as client,
    ):
        spawns = _patch_spawn(app)
        _chats(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            error = _recv_json(ws, "error")
            assert ws.receive()["type"] == "websocket.close"

    assert sid in error["message"]
    assert spawns == []


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_a_resize_sent_while_waiting_sizes_the_spawn(app, sessions_dir):
    """The side reader records the resize; the spawn starts at that size."""
    sid = _uuid()
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        chat = _chats(app).hold(sid, Chat(busy=True))
        with client.websocket_connect(_resume_url(sid)) as ws:
            _recv_json(ws, "handoff_pending")
            _send_resize(ws, cols=132, rows=40)
            time.sleep(0.2)
            chat.is_busy = False
            _recv_json(ws, "session_info")

    assert (spawns[0].rows, spawns[0].cols) == (40, 132)


def test_a_resize_landing_after_the_spawn_is_applied_after_the_door(app, sessions_dir):
    """The spawn read the size before the resize arrived; the handler applies it after."""
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")

    async def spawn_then_linger(app_, key, surface, channel, *, interrupt=False, spawn=None):
        session = await spawn(SimpleNamespace(key=key, resume_id=None, transcript_id=key))
        app_.state.pty_registry.attach_session(key, channel)
        # The door is still busy after the spawn; the client's resize lands now.
        await asyncio.sleep(0.4)
        return SimpleNamespace(session=session, spawned=True, resume_id=None)

    with (
        patch("osprey.interfaces.web_terminal.session_handoff.acquire_surface", spawn_then_linger),
        TestClient(app) as client,
    ):
        spawns = _patch_spawn(app)
        _chats(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            time.sleep(0.15)
            assert (spawns[0].rows, spawns[0].cols) == (24, 80)
            _send_resize(ws, cols=132, rows=40)
            _recv_json(ws, "session_info")
            session = spawns[0].session
            assert (session._last_rows, session._last_cols) == (40, 132)


def test_a_reused_pty_is_resized_to_the_clients_size(app, sessions_dir):
    """The door hands a pooled PTY back as is; the handler applies the client's size."""
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        _chats(app)
        with client.websocket_connect("/ws/terminal") as ws:
            _send_resize(ws, cols=80, rows=24)
            sid = _recv_json(ws, "session_info")["session_id"]

        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws, cols=100, rows=30)
            _recv_json(ws, "session_info")
            _sync(ws, sid)
            session = spawns[0].session
            assert (session._last_rows, session._last_cols) == (30, 100)

    assert len(spawns) == 1


# ---------------------------------------------------------------------------
# The switch path
# ---------------------------------------------------------------------------


def test_switching_to_a_chat_held_key_hands_off(app, sessions_dir):
    """``switch_session`` to a key the chat holds: pending frame, then ``session_switched``."""
    initial, target = _uuid(), _uuid()
    (sessions_dir / f"{initial}.jsonl").write_text("")
    with TestClient(app) as client:
        spawns = _patch_spawn(app)
        chat = _chats(app).hold(target)
        with client.websocket_connect(_resume_url(initial)) as ws:
            _send_resize(ws)
            _recv_json(ws, "session_info")

            ws.send_json({"type": "switch_session", "session_id": target})
            frames = _json_frames_until(ws, "session_switched")

    assert [f["type"] for f in frames] == ["handoff_pending", "session_switched"]
    assert frames[-1]["session_id"] == target
    assert chat.teardowns == 1
    assert len(spawns) == 2
    assert spawns[1].command[spawns[1].command.index("--session-id") + 1] == target


def test_a_switch_whose_spawn_fails_is_an_error_frame_and_a_close(app, sessions_dir):
    """The old session is already let go; a spawn that fails ends the connection cleanly."""
    initial, target = _uuid(), _uuid()
    (sessions_dir / f"{initial}.jsonl").write_text("")
    (sessions_dir / f"{target}.jsonl").write_text("")
    with TestClient(app) as client:
        registry = app.state.pty_registry
        spawns = _patch_spawn(app, failing=target)
        _chats(app)
        with client.websocket_connect(_resume_url(initial)) as ws:
            _send_resize(ws)
            _recv_json(ws, "session_info")

            ws.send_json({"type": "switch_session", "session_id": target})
            error = _recv_json(ws, "error")
            assert ws.receive()["type"] == "websocket.close"

        _until(lambda: target not in get_state(app).pending)
        assert "switch" in error["message"].lower()
        assert len(spawns) == 1
        assert registry.get_session(target) is None
        assert not registry.is_reserved(target)
        assert not registry.is_attached(initial)
        assert registry.get_session(initial) is spawns[0].session
