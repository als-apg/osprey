"""WebSocket new-session confirmation tests.

``session_info`` is the client's only source for ``currentSessionId``, and a
new terminal's id is not something the server has to find out — it is dictated
on the CLI's own command line (``claude --session-id <uuid>``, built in
``terminal_ws``). These tests pin that the handler confirms that id straight
away, on its own, from nothing but what it already knows.

The regression they guard is a race that the old implementation lost as a
matter of course. It used to poll the transcript directory for a
newly-appeared ``<uuid>.jsonl`` and confirm whatever turned up. But Claude Code
writes a session's transcript once that session first has content, not when
the process starts, so a terminal that sat idle past the poll window never got
a ``session_info`` at all — and never got one later either, because there is no
retry. That tab then ran on ``currentSessionId === null`` for as long as it
lived, however much work the operator went on to do in it: feedback filed with
no session context, nothing stored to resume from, and a fresh PTY spawned on
every reload. Hence :func:`test_idle_terminal_is_confirmed_without_a_transcript`,
which is the same assertion made against an empty transcript directory to say
so out loud.

Harness mirrors ``test_ws_resume_confirm.py``: a real ``PtyRegistry`` with
``_spawn_session`` patched to a ``FakePtySession``, so no PTY is ever created.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


class FakePtySession:
    """Minimal PtySession substitute — stays alive, produces no output."""

    def __init__(self):
        self._alive = True

    @property
    def is_alive(self):
        return self._alive

    @property
    def exit_code(self):
        return None if self._alive else 0

    def start(self, initial_rows=24, initial_cols=80, extra_env=None, cwd=None):
        pass

    def resize(self, rows, cols):
        pass

    def write_input(self, data):
        pass

    def terminate(self):
        self._alive = False

    async def read_output(self):
        """Async generator that blocks quietly until the session dies."""
        try:
            while self._alive:
                await asyncio.sleep(0.05)
        except (asyncio.CancelledError, GeneratorExit):
            return
        # Unreachable yield — makes this function an async generator.
        if False:
            yield b""  # pragma: no cover


def _recv_json(ws, msg_type: str, max_frames: int = 30):
    """Receive frames until a JSON message with the given ``type`` arrives."""
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


def _send_resize(ws, cols: int = 80, rows: int = 24):
    """Send the initial resize the handler waits for before spawning."""
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


def _forced_session_id(command: list[str]) -> str:
    """Return the id the handler put after ``--session-id`` in *command*."""
    assert "--session-id" in command, f"no --session-id in spawned command: {command}"
    return command[command.index("--session-id") + 1]


@pytest.fixture()
def app(tmp_path):
    """Create a web terminal app pointed at a temp project dir."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


def _patch_spawn(app):
    """Replace ``_spawn_session`` so no real PTY is created.

    Returns the registry and the list of commands it was asked to spawn, which
    is where the forced ``--session-id`` can be read back from.
    """
    reg = app.state.pty_registry
    commands: list[list[str]] = []

    def tracked_spawn(command, *_args, **_kwargs):
        commands.append(list(command))
        return FakePtySession()

    reg._spawn_session = tracked_spawn
    return reg, commands


def _patch_spawn_tracking_sessions(app):
    """As :func:`_patch_spawn`, but hands back the sessions themselves."""
    reg = app.state.pty_registry
    spawned: list[FakePtySession] = []

    def tracked_spawn(*_args, **_kwargs):
        s = FakePtySession()
        spawned.append(s)
        return s

    reg._spawn_session = tracked_spawn
    return reg, spawned


# ---------------------------------------------------------------------------
# The confirmation itself
# ---------------------------------------------------------------------------


def test_new_session_confirms_the_forced_id(app):
    """A new terminal is confirmed with the id spawned on its command line."""
    with TestClient(app) as client:
        _, commands = _patch_spawn(app)
        with client.websocket_connect("/ws/terminal") as ws:
            _send_resize(ws)
            msg = _recv_json(ws, "session_info")

        assert len(commands) == 1
        assert msg["session_id"] == _forced_session_id(commands[0])


def test_new_session_confirmation_does_not_consult_discovery(app):
    """The id is known, so nothing polls the transcript directory for it."""
    with patch(
        "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session"
    ) as mock_discover:
        with TestClient(app) as client:
            _patch_spawn(app)
            with client.websocket_connect("/ws/terminal") as ws:
                _send_resize(ws)
                _recv_json(ws, "session_info")

        mock_discover.assert_not_called()


def test_idle_terminal_is_confirmed_without_a_transcript(app, tmp_path):
    """The regression: a terminal nobody has typed into is still confirmed.

    Claude Code has written no ``<uuid>.jsonl`` at this point and may never
    write one. Under the old discovery-based implementation that meant no
    ``session_info`` and a client stuck on a null session id for the life of
    the tab; the confirmation must not depend on the transcript existing.
    """
    sessions_dir = tmp_path / "claude_sessions"
    sessions_dir.mkdir()

    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: sessions_dir):
        with TestClient(app) as client:
            _, commands = _patch_spawn(app)
            with client.websocket_connect("/ws/terminal") as ws:
                _send_resize(ws)
                msg = _recv_json(ws, "session_info")

            session_id = msg["session_id"]
            assert session_id == _forced_session_id(commands[0])
            # Nothing on disk backs that id — the confirmation stands alone.
            assert list(sessions_dir.glob("*.jsonl")) == []


# ---------------------------------------------------------------------------
# The pool key
# ---------------------------------------------------------------------------


def test_new_session_is_pooled_under_its_real_id(app):
    """The pool is keyed by the session id, not a placeholder needing a rekey.

    A placeholder key is what let a reconnect miss the warm PTY it should have
    found and spawn a second one against the same session.
    """
    with TestClient(app) as client:
        reg, commands = _patch_spawn(app)
        with client.websocket_connect("/ws/terminal") as ws:
            _send_resize(ws)
            msg = _recv_json(ws, "session_info")

        session_id = msg["session_id"]
        assert reg.get_session(session_id) is not None
        assert list(reg._sessions) == [session_id]
        assert len(commands) == 1


def test_resuming_the_confirmed_id_reuses_the_warm_session(app):
    """Reconnecting with the confirmed id finds the pooled PTY, not a new one.

    This is the round trip the fix restores end to end: the client stores the
    id it was given and resumes it on the next page load.
    """
    with patch(
        "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session",
        return_value=None,
    ):
        with TestClient(app) as client:
            reg, commands = _patch_spawn(app)

            with client.websocket_connect("/ws/terminal") as ws:
                _send_resize(ws)
                session_id = _recv_json(ws, "session_info")["session_id"]

            with client.websocket_connect(
                f"/ws/terminal?session_id={session_id}&mode=resume"
            ) as ws:
                _send_resize(ws)
                msg = _recv_json(ws, "session_info")
                assert msg["session_id"] == session_id

            # Warm reuse: the second connection spawned nothing.
            assert len(commands) == 1
            assert list(reg._sessions) == [session_id]


def test_stale_handler_teardown_spares_the_replacement_session(app):
    """A first tab closing must not kill the PTY a second tab is using.

    Handing every new session an id the client stores and resumes makes two
    handlers on one pool key ordinary rather than accidental, so teardown has
    to check ownership: this handler's PTY has died and been replaced under
    the same key, and terminating by key alone would take the live
    replacement — and the operator's terminal — down with it.
    """
    with patch(
        "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session",
        return_value=None,
    ):
        with TestClient(app) as client, ExitStack() as stack:
            reg, spawned = _patch_spawn_tracking_sessions(app)

            # Tab one: a new session, confirmed with the forced id.
            first = stack.enter_context(client.websocket_connect("/ws/terminal"))
            _send_resize(first)
            session_id = _recv_json(first, "session_info")["session_id"]

            # Its CLI exits. The socket stays open, as it does in the browser
            # — the tab just shows "[Process exited]".
            spawned[0].terminate()

            # Tab two resumes the id tab one stored. The dead PTY is dropped
            # and a replacement spawned under the same key.
            with client.websocket_connect(
                f"/ws/terminal?session_id={session_id}&mode=resume"
            ) as second:
                _send_resize(second)
                assert _recv_json(second, "session_info")["session_id"] == session_id
                assert len(spawned) == 2
                assert reg.get_session(session_id) is spawned[1]

                # Tab one goes away WHILE tab two is still open, so its
                # teardown runs against a session that is no longer the one
                # the pool holds under this key.
                stack.close()

                assert spawned[1].is_alive, "tab one's teardown killed tab two's PTY"
                assert reg.get_session(session_id) is spawned[1]
