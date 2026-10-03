"""WebSocket new-session confirmation tests.

``session_info`` is the client's only source for ``currentSessionId``, and a
new terminal's id is not something the server has to find out — it is dictated
on the CLI's own command line (``claude --session-id <uuid>``, built in
``terminal_ws``). These tests pin that the handler confirms that id straight
away, on its own, from nothing but what it already knows.

Claude Code writes a session's transcript only once the session has content,
so a confirmation that waited for ``<uuid>.jsonl`` would never reach a tab
nobody has typed into, and that tab would run on a null session id: feedback
filed with no session context, nothing stored to resume from, and a fresh PTY
spawned on every reload. The confirmation is therefore pinned against an
empty transcript directory.

Harness mirrors ``test_ws_resume_confirm.py``: a real ``PtyRegistry`` with
``_spawn_session`` patched to a ``FakePtySession``, so no PTY is ever created.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from tests.interfaces.web_terminal._fakes import FakePtySession
from tests.interfaces.web_terminal._ws import recv_json

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


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


def test_new_session_confirms_the_forced_id(app, tmp_path):
    """A new terminal is confirmed with the id spawned on its command line.

    The confirmation must not depend on a transcript: Claude Code writes
    ``<uuid>.jsonl`` only once the session has content, so a terminal nobody
    has typed into has none, and a discovery-based confirmation would leave the
    client on a null session id for the life of the tab.
    """
    sessions_dir = tmp_path / "claude_sessions"
    sessions_dir.mkdir()

    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: sessions_dir):
        with TestClient(app) as client:
            _, commands = _patch_spawn(app)
            with client.websocket_connect("/ws/terminal") as ws:
                _send_resize(ws)
                msg = recv_json(ws, "session_info")

    assert len(commands) == 1
    assert msg["session_id"] == _forced_session_id(commands[0])
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
            msg = recv_json(ws, "session_info")

        session_id = msg["session_id"]
        assert reg.get_session(session_id) is not None
        assert list(reg._sessions) == [session_id]
        assert len(commands) == 1


def test_stale_handler_teardown_spares_the_replacement_session(app, tmp_path):
    """A first tab closing must not kill the PTY a second tab is using.

    Handing every new session an id the client stores and resumes makes two
    handlers on one pool key ordinary rather than accidental, so teardown has
    to check ownership: this handler's PTY has died and been replaced under
    the same key, and terminating by key alone would take the live
    replacement — and the operator's terminal — down with it.

    The session was prompted before its CLI exited, so its transcript is on
    disk: with a dead PTY that is what makes the id resumable at all.
    """
    sessions_dir = tmp_path / "claude_sessions"
    sessions_dir.mkdir()

    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: sessions_dir):
        with TestClient(app) as client, ExitStack() as stack:
            reg, spawned = _patch_spawn_tracking_sessions(app)

            # Tab one: a new session, confirmed with the forced id.
            first = stack.enter_context(client.websocket_connect("/ws/terminal"))
            _send_resize(first)
            session_id = recv_json(first, "session_info")["session_id"]

            # Its CLI exits after a prompt was sent. The socket stays open, as
            # it does in the browser — the tab just shows "[Process exited]".
            (sessions_dir / f"{session_id}.jsonl").write_text("")
            spawned[0].terminate()

            # Tab two resumes the id tab one stored. The dead PTY is dropped
            # and a replacement spawned under the same key.
            with client.websocket_connect(
                f"/ws/terminal?session_id={session_id}&mode=resume"
            ) as second:
                _send_resize(second)
                assert recv_json(second, "session_info")["session_id"] == session_id
                assert len(spawned) == 2
                assert reg.get_session(session_id) is spawned[1]

                # Tab one goes away WHILE tab two is still open, so its
                # teardown runs against a session that is no longer the one
                # the pool holds under this key.
                stack.close()

                assert spawned[1].is_alive, "tab one's teardown killed tab two's PTY"
                assert reg.get_session(session_id) is spawned[1]
