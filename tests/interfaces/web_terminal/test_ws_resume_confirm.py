"""WebSocket resume-boundary tests.

``terminal_ws`` confirms every resume with a ``session_info`` frame carrying
the id actually attached, and refuses to resume an id whose transcript is not
on disk. The outcomes covered here:

- A reused warm session confirms synchronously with the requested id — with
  or without a transcript, because a session that was opened and never
  prompted has a live PTY and no ``.jsonl`` yet.
- A cold resume whose transcript already exists on disk confirms
  synchronously with the requested id.
- A cold resume of an id with no transcript spawns nothing: the server
  answers ``transcript_missing`` and closes, and the client renders that
  state instead of a dead PTY.
- A cold resume of an id the chat pool answers to is served: both views are
  windows onto one session key, so a conversation the operator started in
  Simple is a session here even before a transcript names it. Serving it is
  a hand-off — the chat is torn down and the terminal spawned in its place.
- A ``--resume`` child that the pre-spawn check let through but which still
  reports ``No conversation found with session ID`` and exits gets the same
  ``transcript_missing`` frame in place of a bare ``exit``.

Mirrors the harness in ``test_session_switching.py``: real ``PtyRegistry``
with ``_spawn_session`` patched to a ``FakePtySession``.
"""

from __future__ import annotations

import json
import sys
import time
import uuid as uuid_mod
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from tests.interfaces.web_terminal._fakes import FakePtySession

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


def _recv_json(ws, msg_type: str, max_frames: int = 30):
    """Receive frames until a JSON message with the given ``type`` arrives.

    Skips binary frames. Raises ``AssertionError`` if ``msg_type`` is not
    found within *max_frames* frames.
    """
    collected = []
    for _ in range(max_frames):
        raw = ws.receive()
        if "text" in raw:
            data = json.loads(raw["text"])
            collected.append(data)
            if data.get("type") == msg_type:
                return data
        # binary frames are silently skipped
    types = [d.get("type") for d in collected]
    raise AssertionError(
        f"Expected JSON type '{msg_type}' not received within {max_frames} frames. "
        f"Got types: {types}"
    )


def _collect_json_until(ws, msg_type: str, max_frames: int = 30) -> list[dict]:
    """Every JSON message up to and including the first *msg_type*.

    The twin of :func:`_recv_json` for asserting what did NOT arrive: read to a
    frame the server is guaranteed to send, then inspect the whole run.
    """
    collected = []
    for _ in range(max_frames):
        raw = ws.receive()
        if "text" in raw:
            data = json.loads(raw["text"])
            collected.append(data)
            if data.get("type") == msg_type:
                return collected
    types = [d.get("type") for d in collected]
    raise AssertionError(f"'{msg_type}' not received within {max_frames} frames. Got: {types}")


def _uuid() -> str:
    return str(uuid_mod.uuid4())


def _resume_url(session_id: str) -> str:
    return f"/ws/terminal?session_id={session_id}&mode=resume"


def _send_resize(ws, cols: int = 80, rows: int = 24):
    """Send the initial resize the handler waits for before spawning."""
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


def _wait_for_spawn(spawned: list, timeout: float = 5.0) -> FakePtySession:
    deadline = time.monotonic() + timeout
    while not spawned and time.monotonic() < deadline:
        time.sleep(0.01)
    assert spawned, "handler never spawned a PTY"
    return spawned[-1]


@pytest.fixture()
def app(tmp_path):
    """Create a web terminal app pointed at a temp project dir."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


@pytest.fixture()
def sessions_dir(tmp_path):
    """Point ``SessionDiscovery`` at an empty transcript directory."""
    d = tmp_path / "claude_sessions"
    d.mkdir()
    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: d):
        yield d


class _IdleChat:
    """The slice of ``OperatorSession`` the hand-off door reads for an idle chat."""

    is_busy = False

    def __init__(self):
        self._active = True
        self.process_exited: bool | None = None
        self.teardowns = 0

    @property
    def is_active(self) -> bool:
        return self._active

    async def teardown(self) -> None:
        self.teardowns += 1
        self._active = False
        self.process_exited = True


class _ChatPool:
    """``ChatSessionPool`` as the door sees it: ``get``, ``has_key``, ``terminate``."""

    def __init__(self):
        self.sessions: dict[str, _IdleChat] = {}

    def get(self, chat_id: str) -> _IdleChat | None:
        return self.sessions.get(chat_id)

    def has_key(self, chat_id: str) -> bool:
        return chat_id in self.sessions

    async def terminate(self, chat_id: str) -> _IdleChat | None:
        session = self.sessions.pop(chat_id, None)
        if session is not None:
            await session.teardown()
        return session

    async def reinsert(self, chat_id: str, session: _IdleChat) -> bool:
        self.sessions[chat_id] = session
        return True


class _ChatHoldingRegistry:
    """An ``operator_registry`` whose chat pool holds exactly one idle chat.

    The addressability facade the gate asks for
    (:func:`~osprey.interfaces.web_terminal.routes.websocket._chat_pool_answers_to`
    reaches ``has_chat_key``) and the ``chats`` pool the hand-off door
    inspects and tears down; ``cleanup_all`` is the app's own shutdown hook.
    """

    def __init__(self, key: str):
        self.key = key
        self.chats = _ChatPool()
        self.chat = _IdleChat()
        self.chats.sessions[key] = self.chat

    def has_chat_key(self, session_id: str) -> bool:
        return self.chats.has_key(session_id)

    async def cleanup_all(self) -> None:
        pass


def _patch_spawn(app):
    """Replace ``_spawn_session`` so no real PTY is created."""
    reg = app.state.pty_registry
    spawned: list[FakePtySession] = []

    def tracked_spawn(*_args, **_kwargs):
        s = FakePtySession()
        spawned.append(s)
        return s

    reg._spawn_session = tracked_spawn
    return reg, spawned


# ---------------------------------------------------------------------------
# Warm session — confirmed synchronously
# ---------------------------------------------------------------------------


def test_reused_warm_session_confirms_immediately(app, sessions_dir):
    """Reconnecting to an already-warm session confirms the requested id."""
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    with TestClient(app) as client:
        _, spawned = _patch_spawn(app)

        # First connect: fresh spawn, keeps the session warm in the pool.
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid

        # Second connect: reuses the warm session from the pool.
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid

    assert len(spawned) == 1


def test_warm_session_without_transcript_is_reattached(app, sessions_dir):
    """A session opened and never prompted has a PTY but no transcript yet.

    Claude Code writes ``projects/<encoded>/<id>.jsonl`` on the first prompt,
    so a reload straight after opening a terminal resumes an id with no file
    on disk. The warm PTY is the session; the missing file must not be read
    as a missing transcript.
    """
    with TestClient(app) as client:
        _, spawned = _patch_spawn(app)
        with client.websocket_connect("/ws/terminal") as ws:
            _send_resize(ws)
            sid = _recv_json(ws, "session_info")["session_id"]

        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid

    assert len(spawned) == 1


# ---------------------------------------------------------------------------
# Cold resume, transcript on disk — confirmed synchronously
# ---------------------------------------------------------------------------


def test_cold_resume_with_existing_file_confirms_immediately(app, sessions_dir):
    """A not-currently-warm resume whose transcript exists is trusted at once."""
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    with TestClient(app) as client:
        _, spawned = _patch_spawn(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid
    assert len(spawned) == 1


# ---------------------------------------------------------------------------
# Cold resume, no transcript — surfaced, never spawned
# ---------------------------------------------------------------------------


def test_cold_resume_without_transcript_is_surfaced_not_spawned(app, sessions_dir):
    """No warm PTY and no ``.jsonl``: nothing to resume, so nothing is spawned.

    ``claude --resume <id>`` on such an id prints ``No conversation found``
    and exits 1, which left the operator on a dead PTY. The server already
    knows before spawning; it says so and closes, and the client renders the
    state (terminal.js) instead of a doomed child.
    """
    sid = _uuid()
    with TestClient(app) as client:
        reg, spawned = _patch_spawn(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            msg = _recv_json(ws, "transcript_missing")
            assert msg["session_id"] == sid
            # The server closes rather than holding a socket with no PTY.
            assert ws.receive()["type"] == "websocket.close"

        assert spawned == []
        assert reg.get_session(sid) is None


# ---------------------------------------------------------------------------
# Belt and braces — the child itself reports the transcript is gone
# ---------------------------------------------------------------------------


def test_resume_child_that_finds_no_conversation_is_surfaced(app, sessions_dir):
    """A ``--resume`` child that exits on ``No conversation found`` is surfaced.

    The pre-spawn check can be beaten (a transcript removed after the check,
    or one the CLI refuses to load). The child's own verdict is then the
    signal: its output names the failure and it exits non-zero. The final
    frame is ``transcript_missing`` — the same state the pre-spawn refusal
    produces — not a bare ``exit`` the client would read as the operator
    ending a session.
    """
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    with TestClient(app) as client:
        _, spawned = _patch_spawn(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid
            child = _wait_for_spawn(spawned)
            child.emit(b"No conversation found with session ID:\r\n")
            child.emit(f"{sid}\r\n".encode())
            child.exit(1)

            seen = _collect_json_until(ws, "transcript_missing")

    final = seen[-1]
    assert final == {"type": "transcript_missing", "session_id": sid, "code": 1}
    assert "exit" not in [m["type"] for m in seen]


def test_resume_child_exiting_for_another_reason_still_sends_exit(app, sessions_dir):
    """Only the missing-transcript verdict is rewritten; other exits stay exits."""
    sid = _uuid()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    with TestClient(app) as client:
        _, spawned = _patch_spawn(app)
        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid
            child = _wait_for_spawn(spawned)
            child.emit(b"Resumed. Goodbye.\r\n")
            child.exit(0)

            seen = _collect_json_until(ws, "exit")

    assert seen[-1] == {"type": "exit", "code": 0}
    assert "transcript_missing" not in [m["type"] for m in seen]


# ---------------------------------------------------------------------------
# A chat-held key is a session
# ---------------------------------------------------------------------------


def test_cold_resume_of_a_chat_held_key_is_not_refused(app, sessions_dir):
    """An id only the chat pool holds resumes instead of being refused.

    The Simple view keys its pool on the same session key the terminal does,
    so a conversation started there names a session on this surface too — and
    it can name one before any ``.jsonl`` exists, because the transcript is
    written on the first turn. Refusing it would tell the operator the
    conversation they are looking at is gone. What happens instead is the
    hand-off: the chat is torn down and the terminal takes the key.
    """
    sid = _uuid()
    with TestClient(app) as client:
        _, spawned = _patch_spawn(app)
        holder = _ChatHoldingRegistry(sid)
        app.state.operator_registry = holder

        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "session_info")["session_id"] == sid

    assert len(spawned) == 1
    assert holder.chats.get(sid) is None
    assert holder.chat.teardowns == 1


def test_cold_resume_of_a_key_no_surface_holds_is_refused(app, sessions_dir):
    """With a chat pool answering to some OTHER key, the refusal still stands."""
    sid = _uuid()
    with TestClient(app) as client:
        reg, spawned = _patch_spawn(app)
        app.state.operator_registry = _ChatHoldingRegistry(_uuid())

        with client.websocket_connect(_resume_url(sid)) as ws:
            _send_resize(ws)
            assert _recv_json(ws, "transcript_missing")["session_id"] == sid
            assert ws.receive()["type"] == "websocket.close"

        assert spawned == []
        assert reg.get_session(sid) is None
