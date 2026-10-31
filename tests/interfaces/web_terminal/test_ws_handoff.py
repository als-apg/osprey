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
- a reconnect reattaches the pooled PTY while the server would launch it
  under the same overlay, and replaces it once that overlay has changed;
- a resize sent while waiting sizes the spawn, one that lands after the
  spawn is applied once the door returns, and a reused PTY is resized to
  the client's size by the handler itself;
- the spawn resumes the key's current transcript when one is on disk and
  starts fresh under the key otherwise;
- ``switch_session`` to a chat-held key hands off the same way;
- a chat whose agent started commands still running refuses the terminal
  with a ``handoff_refused`` frame naming them and a 4428 close, unless the
  resume URL carries ``end_started=1``.

Harness as in ``test_ws_resume_confirm.py``: a real app and ``PtyRegistry``
with ``_spawn_session`` patched to a fake, and a stand-in operator registry
whose chat pool is the phase (c) fake.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_auth import PANEL_TOKEN_ENV
from osprey.interfaces.web_terminal.process_tree import ProcessGroup
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry, env_fingerprint
from osprey.interfaces.web_terminal.routes import websocket as websocket_route
from osprey.interfaces.web_terminal.routes.websocket import _spawn_pooled_pty
from osprey.interfaces.web_terminal.session_handoff import (
    ERROR_HANDOFF_STARTED_COMMANDS,
    WS_CLOSE_OUTGOING_RUNNING,
    WS_CLOSE_SESSION_ATTACHED,
    WS_CLOSE_STARTED_COMMANDS,
    HandoffError,
    HandoffState,
    get_state,
)
from tests._live import wait_until
from tests.interfaces.web_terminal._fakes import FakeClock, FakePtySession
from tests.interfaces.web_terminal._handoff_harness import Chat, ChatPool
from tests.interfaces.web_terminal._ws import (
    json_frames_until,
    new_session_id,
    recv_frame,
    recv_json,
    resume_url,
    send_resize,
    sync_session,
)

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
    cwd: str | None = None


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


def _patch_spawn(ws_app, *, failing: str | None = None) -> list[Spawn]:
    """Replace ``_spawn_session`` so no real PTY is created; record each call.

    A spawn whose command names *failing* raises ``OSError`` instead, the way
    a CLI that cannot be executed does.
    """
    spawns: list[Spawn] = []

    def tracked_spawn(command, rows, cols, _extra_env, cwd=None):
        if failing is not None and failing in command:
            raise OSError("cannot spawn")
        session = ObservedPty()
        spawns.append(Spawn(list(command), rows, cols, session, cwd))
        return session

    ws_app.state.pty_registry._spawn_session = tracked_spawn
    return spawns


def _chats(ws_app) -> ChatRegistry:
    registry = ChatRegistry()
    ws_app.state.operator_registry = registry
    return registry


def _with_started_command(chat: Chat) -> Chat:
    """*chat* with one command its agent started still running."""
    chat.running_commands = [
        ProcessGroup(
            pgid=4242,
            members=((4242, 0.0),),
            label="magnet_scan.py",
            command="python magnet_scan.py",
        )
    ]
    return chat


def _fake_clock(ws_app) -> FakeClock:
    """Run the door's grace and death waits on a clock that only moves when slept."""
    clock = FakeClock()
    ws_app.state.handoff = HandoffState(clock=clock, sleep=clock.sleep)
    return clock


# ---------------------------------------------------------------------------
# A chat-held key: the frames and the spawn
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("sessions_dir")
def test_a_chat_held_key_is_handed_off_with_the_pending_frame_first(ws_app):
    """``handoff_pending`` goes out before the wait, ``session_info`` after the door."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            frames = json_frames_until(ws, "session_info")

    assert [f["type"] for f in frames] == ["handoff_pending", "session_info"]
    # An idle chat has no turn to finish: the frame says so, and the client
    # shows a restart rather than a wait.
    assert frames[0] == {"type": "handoff_pending", "busy": False}
    assert frames[-1]["session_id"] == sid
    assert chat.teardowns == 1
    assert ws_app.state.operator_registry.chats.get(sid) is None
    assert len(spawns) == 1


@pytest.mark.usefixtures("sessions_dir")
def test_a_key_with_no_transcript_starts_fresh_under_the_key(ws_app):
    """A chat that never wrote a transcript hands off to ``--session-id <key>``."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        _chats(ws_app).hold(sid)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")

    command = spawns[0].command
    assert command[command.index("--session-id") + 1] == sid
    assert "--resume" not in command


@pytest.mark.usefixtures("sessions_dir")
def test_the_spawn_runs_in_the_project_directory(ws_app):
    """The launch argv names ``.mcp.json`` relative to the working directory."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        _chats(ws_app).hold(sid)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")

    assert spawns[0].cwd == ws_app.state.project_cwd


def test_the_spawn_resumes_the_keys_current_transcript(ws_app, sessions_dir):
    """The key's transcript may have moved (``/clear``); the spawn resumes the current one."""
    sid, transcript = new_session_id(), new_session_id()
    (sessions_dir / f"{transcript}.jsonl").write_text("")
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        _chats(ws_app).hold(sid)
        ws_app.state.transcript_map = {sid: transcript}
        ws_app.state.transcript_map_provisional = False
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")

    command = spawns[0].command
    assert command[command.index("--resume") + 1] == transcript
    assert "--session-id" not in command


def test_a_spawn_at_capacity_evicts_the_oldest_background_pty_off_the_loop(ws_app, sessions_dir):
    """The pool's eviction kill blocks; the handler takes the victim out and kills it in a thread."""
    sid, background = new_session_id(), new_session_id()
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
    with TestClient(ws_app) as client:
        # The lifespan installs the app's registry; a one-slot pool replaces it.
        registry = PtyRegistry(max_background=1)
        ws_app.state.pty_registry = registry
        registry._sessions[background] = victim
        spawns = _patch_spawn(ws_app)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")
            assert background not in registry._sessions
            assert len(spawns) == 1 and registry.get_session(sid) is spawns[0].session

    assert victim.killed_on_loop is False


async def test_a_launch_env_change_kills_the_stale_pty_off_the_loop():
    """The stale child's kill blocks for seconds, so the spawn path pops it and kills it in a
    worker thread while the loop keeps serving."""

    class SlowStalePty(ObservedPty):
        def __init__(self):
            super().__init__()
            self.kill_started = threading.Event()
            self.loop_moved = threading.Event()
            self.kills = 0
            self.loop_was_free: bool | None = None

        def terminate(self):
            self.kills += 1
            self.kill_started.set()
            # Blocks until a coroutine on the loop gets to run, or gives up.
            self.loop_was_free = self.loop_moved.wait(timeout=1.0)
            super().terminate()

    registry = PtyRegistry(max_background=5)
    stale = SlowStalePty()
    registry._sessions["k"] = stale
    registry._env_fingerprints["k"] = env_fingerprint({"OSPREY_PANEL_TOKEN": "before"})
    fresh = ObservedPty()
    registry._spawn_session = lambda *_args, **_kwargs: fresh

    async def note_the_loop_moving():
        while not stale.kill_started.is_set():
            await asyncio.sleep(0.005)
        stale.loop_moved.set()

    watcher = asyncio.create_task(note_the_loop_moving())

    spawned = await _spawn_pooled_pty(
        registry,
        "k",
        ["fake"],
        rows=24,
        cols=80,
        extra_env={"OSPREY_PANEL_TOKEN": "after"},
        cwd=None,
    )
    await asyncio.wait_for(watcher, 2.0)

    assert stale.loop_was_free is True
    assert stale.kills == 1
    assert stale.is_alive is False
    assert spawned is fresh
    assert registry.get_session("k") is fresh
    assert registry._env_fingerprints["k"] == env_fingerprint({"OSPREY_PANEL_TOKEN": "after"})


def test_a_free_key_gets_no_pending_frame(ws_app, sessions_dir):
    """Nothing to wait on, nothing to announce: the first frame is the confirmation."""
    sid = new_session_id()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    with TestClient(ws_app) as client:
        _patch_spawn(ws_app)
        _chats(ws_app)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            frames = json_frames_until(ws, "session_info")

    assert [f["type"] for f in frames] == ["session_info"]


# ---------------------------------------------------------------------------
# A busy chat: wait, interrupt, leave
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("sessions_dir")
def test_a_busy_chat_is_waited_on(ws_app):
    """The terminal is confirmed once the chat's turn ends, and not before."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid, Chat(busy=True))
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            # Mid-turn: the frame says the wait is real, so the client shows
            # it as one from the first moment.
            assert recv_json(ws, "handoff_pending")["busy"] is True
            time.sleep(0.3)
            assert spawns == []
            assert ws_app.state.operator_registry.chats.get(sid) is chat

            chat.is_busy = False
            assert recv_json(ws, "session_info")["session_id"] == sid

    assert chat.teardowns == 1
    assert len(spawns) == 1


@pytest.mark.usefixtures("sessions_dir")
def test_interrupt_cuts_the_chat_turn_short(ws_app):
    """``interrupt=1`` on the resume URL is the "stop and switch now" path."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid, Chat(busy=True))
        with client.websocket_connect(resume_url(sid, interrupt=True)) as ws:
            send_resize(ws)
            assert recv_json(ws, "session_info")["session_id"] == sid

    assert chat.teardowns == 1
    assert len(spawns) == 1


@pytest.mark.usefixtures("sessions_dir")
def test_leaving_during_the_wait_sends_nothing_and_spares_the_chat(ws_app):
    """A closed socket ends the wait: no spawn, no frame, the chat untouched, the key clean."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        registry = ws_app.state.pty_registry
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid, Chat(busy=True))
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            recv_json(ws, "handoff_pending")

        wait_until(
            lambda: sid not in get_state(ws_app).pending,
            timeout=3.0,
            what="the pending entry cleared",
        )
        assert not registry.is_reserved(sid)
        assert ws_app.state.operator_registry.chats.get(sid) is chat
        assert chat.teardowns == 0
        assert spawns == []
        assert registry.get_session(sid) is None


# ---------------------------------------------------------------------------
# Refusals and errors
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("sessions_dir")
def test_a_newer_terminal_displaces_the_older_one_with_4409(ws_app):
    """Same key, second socket: the first is closed with 4409, the PTY is reused."""
    with TestClient(ws_app) as client, ExitStack() as stack:
        spawns = _patch_spawn(ws_app)
        _chats(ws_app)
        first = stack.enter_context(client.websocket_connect("/ws/terminal"))
        send_resize(first)
        sid = recv_json(first, "session_info")["session_id"]

        second = stack.enter_context(client.websocket_connect(resume_url(sid)))
        send_resize(second)
        assert recv_json(second, "session_info")["session_id"] == sid

        closed = first.receive()
        assert closed["type"] == "websocket.close"
        assert closed["code"] == WS_CLOSE_SESSION_ATTACHED
        # The displaced handler's output loop is stopped before the close
        # handshake completes: one reader on the PTY, the new terminal's.
        wait_until(
            lambda: spawns[0].session.readers == 1, timeout=3.0, what="one reader on the PTY"
        )

    assert len(spawns) == 1


@pytest.mark.usefixtures("sessions_dir")
def test_an_outgoing_child_that_survives_its_kill_is_refused_with_4503(ws_app):
    """The chat child would not die: nothing is spawned, the chat is pooled again, 4503."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        _fake_clock(ws_app)
        chat = _chats(ws_app).hold(sid, Chat(exits=False))
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            recv_json(ws, "handoff_pending")
            closed = recv_frame(ws)

    assert closed["type"] == "websocket.close"
    assert closed["code"] == WS_CLOSE_OUTGOING_RUNNING
    assert spawns == []
    assert ws_app.state.operator_registry.chats.get(sid) is chat
    assert ws_app.state.pty_registry.get_session(sid) is None


@pytest.mark.usefixtures("sessions_dir")
def test_a_chat_with_started_commands_refuses_the_terminal_with_the_list(ws_app):
    """The list rides on a ``handoff_refused`` frame; the close code is the question's."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid, _with_started_command(Chat()))
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            frames = json_frames_until(ws, "handoff_refused")
            closed = recv_frame(ws)

    assert [f["type"] for f in frames] == ["handoff_pending", "handoff_refused"]
    assert frames[-1] == {
        "type": "handoff_refused",
        "error": ERROR_HANDOFF_STARTED_COMMANDS,
        "session_id": sid,
        "commands": [{"label": "magnet_scan.py", "command": "python magnet_scan.py"}],
        "unread": False,
    }
    assert closed["type"] == "websocket.close"
    assert closed["code"] == WS_CLOSE_STARTED_COMMANDS
    assert chat.teardowns == 0
    assert spawns == []


@pytest.mark.usefixtures("sessions_dir")
def test_a_chat_whose_started_commands_cannot_be_read_refuses_the_terminal(ws_app):
    """A process table that cannot be read is asked about, never taken as nothing running."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        unread = Chat()
        unread.running_commands = None
        chat = _chats(ws_app).hold(sid, unread)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            frames = json_frames_until(ws, "handoff_refused")
            closed = recv_frame(ws)

    assert frames[-1] == {
        "type": "handoff_refused",
        "error": ERROR_HANDOFF_STARTED_COMMANDS,
        "session_id": sid,
        "commands": [],
        "unread": True,
    }
    assert closed["code"] == WS_CLOSE_STARTED_COMMANDS
    assert chat.teardowns == 0
    assert spawns == []


@pytest.mark.usefixtures("sessions_dir")
def test_end_started_on_the_resume_url_hands_the_key_off(ws_app):
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid, _with_started_command(Chat()))
        with client.websocket_connect(resume_url(sid, end_started=True)) as ws:
            send_resize(ws)
            info = recv_json(ws, "session_info")

    assert info["session_id"] == sid
    assert chat.teardowns == 1
    assert len(spawns) == 1


def test_a_handoff_error_is_an_error_frame_and_a_close(ws_app, sessions_dir):
    """An error from inside the door reaches the client as ``error``, then the socket closes."""
    sid = new_session_id()
    (sessions_dir / f"{sid}.jsonl").write_text("")
    failing = AsyncMock(side_effect=HandoffError.vanished(sid, "simple"))
    with (
        patch("osprey.interfaces.web_terminal.session_handoff.acquire_surface", failing),
        TestClient(ws_app) as client,
    ):
        spawns = _patch_spawn(ws_app)
        _chats(ws_app)
        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws)
            error = recv_json(ws, "error")
            assert recv_frame(ws)["type"] == "websocket.close"

    assert sid in error["message"]
    assert spawns == []


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("sessions_dir")
def test_a_resize_sent_while_waiting_sizes_the_spawn(ws_app):
    """The side reader records the resize; the spawn starts at that size."""
    sid = new_session_id()
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(sid, Chat(busy=True))
        with client.websocket_connect(resume_url(sid)) as ws:
            recv_json(ws, "handoff_pending")
            send_resize(ws, cols=132, rows=40)
            time.sleep(0.2)
            chat.is_busy = False
            recv_json(ws, "session_info")

    assert (spawns[0].rows, spawns[0].cols) == (40, 132)


def test_a_resize_landing_after_the_spawn_is_applied_after_the_door(ws_app, sessions_dir):
    """The spawn read the size before the resize arrived; the handler applies it after."""
    sid = new_session_id()
    (sessions_dir / f"{sid}.jsonl").write_text("")

    # ``acquire_surface``'s signature: the websocket route names ``interrupt``,
    # ``end_started`` and ``launch_env``.
    async def spawn_then_linger(
        app_,
        key,
        _surface,
        channel,
        *,
        interrupt=False,  # noqa: ARG001
        spawn=None,
        end_started=False,  # noqa: ARG001
        launch_env=None,
    ):
        session = await spawn(
            SimpleNamespace(key=key, resume_id=None, transcript_id=key, launch_env=launch_env(key))
        )
        app_.state.pty_registry.attach_session(key, channel)
        # The door is still busy after the spawn; the client's resize lands now.
        await asyncio.sleep(0.4)
        return SimpleNamespace(session=session, spawned=True, resume_id=None)

    with (
        patch("osprey.interfaces.web_terminal.session_handoff.acquire_surface", spawn_then_linger),
        TestClient(ws_app) as client,
    ):
        spawns = _patch_spawn(ws_app)
        _chats(ws_app)
        with client.websocket_connect(resume_url(sid)) as ws:
            time.sleep(0.15)
            assert (spawns[0].rows, spawns[0].cols) == (24, 80)
            send_resize(ws, cols=132, rows=40)
            recv_json(ws, "session_info")
            session = spawns[0].session
            assert (session._last_rows, session._last_cols) == (40, 132)


@pytest.mark.usefixtures("sessions_dir")
def test_a_reused_pty_is_resized_to_the_clients_size(ws_app):
    """The door hands a pooled PTY back as is; the handler applies the client's size."""
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        _chats(ws_app)
        with client.websocket_connect("/ws/terminal") as ws:
            send_resize(ws, cols=80, rows=24)
            sid = recv_json(ws, "session_info")["session_id"]

        with client.websocket_connect(resume_url(sid)) as ws:
            send_resize(ws, cols=100, rows=30)
            recv_json(ws, "session_info")
            sync_session(ws, sid)
            session = spawns[0].session
            assert (session._last_rows, session._last_cols) == (30, 100)

    assert len(spawns) == 1


@pytest.mark.usefixtures("sessions_dir")
def test_a_reconnect_after_the_launch_env_changed_gets_a_fresh_pty(ws_app):
    """A pooled PTY launched under an older overlay is replaced, not reattached."""
    real_build = websocket_route._build_extra_env
    changed: dict[str, dict[str, str]] = {}

    def build_with_a_new_token(websocket, claude_session_id, telemetry_session_id=None):
        overlay = real_build(websocket, claude_session_id, telemetry_session_id)
        overlay[PANEL_TOKEN_ENV] = "rotated-" + overlay[PANEL_TOKEN_ENV]
        changed["overlay"] = overlay
        return overlay

    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        _chats(ws_app)
        with client.websocket_connect("/ws/terminal") as ws:
            send_resize(ws)
            sid = recv_json(ws, "session_info")["session_id"]

        with (
            patch.object(websocket_route, "_build_extra_env", build_with_a_new_token),
            client.websocket_connect(resume_url(sid)) as ws,
        ):
            send_resize(ws)
            recv_json(ws, "session_info")
            sync_session(ws, sid)

        registry = ws_app.state.pty_registry
        assert len(spawns) == 2
        assert spawns[0].session.is_alive is False
        assert registry.get_session(sid) is spawns[1].session
        assert registry._env_fingerprints[sid] == env_fingerprint(changed["overlay"])


# ---------------------------------------------------------------------------
# The switch path
# ---------------------------------------------------------------------------


def test_switching_to_a_chat_held_key_hands_off(ws_app, sessions_dir):
    """``switch_session`` to a key the chat holds: pending frame, then ``session_switched``."""
    initial, target = new_session_id(), new_session_id()
    (sessions_dir / f"{initial}.jsonl").write_text("")
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(target)
        with client.websocket_connect(resume_url(initial)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")

            ws.send_json({"type": "switch_session", "session_id": target})
            frames = json_frames_until(ws, "session_switched")

    assert [f["type"] for f in frames] == ["handoff_pending", "session_switched"]
    assert frames[-1]["session_id"] == target
    assert chat.teardowns == 1
    assert len(spawns) == 2
    assert spawns[1].command[spawns[1].command.index("--session-id") + 1] == target


def test_a_switch_to_a_chat_with_started_commands_is_refused_with_the_list(ws_app, sessions_dir):
    initial, target = new_session_id(), new_session_id()
    (sessions_dir / f"{initial}.jsonl").write_text("")
    with TestClient(ws_app) as client:
        spawns = _patch_spawn(ws_app)
        chat = _chats(ws_app).hold(target, _with_started_command(Chat()))
        with client.websocket_connect(resume_url(initial)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")

            ws.send_json({"type": "switch_session", "session_id": target})
            frames = json_frames_until(ws, "handoff_refused")
            closed = recv_frame(ws)

    assert [f["type"] for f in frames] == ["handoff_pending", "handoff_refused"]
    assert frames[-1]["session_id"] == target
    assert frames[-1]["commands"] == [
        {"label": "magnet_scan.py", "command": "python magnet_scan.py"}
    ]
    assert closed["type"] == "websocket.close"
    assert closed["code"] == WS_CLOSE_STARTED_COMMANDS
    assert chat.teardowns == 0
    assert len(spawns) == 1


def test_a_switch_whose_spawn_fails_is_an_error_frame_and_a_close(ws_app, sessions_dir):
    """The old session is already let go; a spawn that fails ends the connection cleanly."""
    initial, target = new_session_id(), new_session_id()
    (sessions_dir / f"{initial}.jsonl").write_text("")
    (sessions_dir / f"{target}.jsonl").write_text("")
    with TestClient(ws_app) as client:
        registry = ws_app.state.pty_registry
        spawns = _patch_spawn(ws_app, failing=target)
        _chats(ws_app)
        with client.websocket_connect(resume_url(initial)) as ws:
            send_resize(ws)
            recv_json(ws, "session_info")

            ws.send_json({"type": "switch_session", "session_id": target})
            error = recv_json(ws, "error")
            assert recv_frame(ws)["type"] == "websocket.close"

        wait_until(
            lambda: target not in get_state(ws_app).pending,
            timeout=3.0,
            what="the pending entry cleared",
        )
        assert "switch" in error["message"].lower()
        assert len(spawns) == 1
        assert registry.get_session(target) is None
        assert not registry.is_reserved(target)
        assert not registry.is_attached(initial)
        assert registry.get_session(initial) is spawns[0].session
