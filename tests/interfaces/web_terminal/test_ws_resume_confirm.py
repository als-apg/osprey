"""WebSocket resume-confirmation tests.

Covers the `mode=resume` confirmation added in `terminal_ws`: today
`session_info` is only emitted on the new-session discovery path, so a
client resuming a stale/absent `--resume-id` has no way to tell a live
resume from a silently-fresh PTY. These tests exercise all three outcomes:

- A reused warm session confirms synchronously with the requested id.
- A cold resume whose session file already exists on disk confirms
  synchronously with the requested id (no discovery needed — the id was
  genuinely valid).
- A freshly-spawned PTY for an id with no file on disk (stale/absent)
  confirms via the discovery mechanism — either the requested id (nothing
  new appeared) or a newly-discovered id (the CLI started a fresh session
  instead).

Mirrors the harness in ``test_session_switching.py``: real ``PtyRegistry``
with ``_spawn_session`` patched to a ``FakePtySession``. ``discover_new_session``
is patched per-test to avoid the real filesystem poll.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import uuid as uuid_mod
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
        self._last_rows = 24
        self._last_cols = 80
        self._command_list = ["fake"]

    @property
    def is_alive(self):
        return self._alive

    @property
    def exit_code(self):
        return None if self._alive else 0

    def start(self, initial_rows=24, initial_cols=80, extra_env=None, cwd=None):
        self._last_rows = initial_rows
        self._last_cols = initial_cols

    def resize(self, rows, cols):
        self._last_rows = rows
        self._last_cols = cols

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


def _uuid() -> str:
    return str(uuid_mod.uuid4())


def _resume_url(session_id: str) -> str:
    return f"/ws/terminal?session_id={session_id}&mode=resume"


def _send_resize(ws, cols: int = 80, rows: int = 24):
    """Send the initial resize the handler waits for before spawning."""
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


@pytest.fixture()
def app(tmp_path):
    """Create a web terminal app pointed at a temp project dir."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


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
# Fresh spawn — no new session file appears (resume genuinely succeeded)
# ---------------------------------------------------------------------------


def test_fresh_resume_confirms_requested_id(app):
    """A fresh PTY spawned for a genuinely-valid --resume-id confirms that id."""
    sid = _uuid()
    with patch(
        "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session",
        return_value=None,
    ):
        with TestClient(app) as client:
            _patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                msg = _recv_json(ws, "session_info")
                assert msg["session_id"] == sid


# ---------------------------------------------------------------------------
# Fresh spawn — a NEW session file appears (requested id was stale/absent)
# ---------------------------------------------------------------------------


def test_stale_resume_confirms_discovered_id(app):
    """A stale/absent --resume-id: the CLI starts fresh under a new id, and
    the confirmation carries that discovered id, not the requested one."""
    stale_sid = _uuid()
    discovered_sid = _uuid()
    with patch(
        "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session",
        return_value=discovered_sid,
    ):
        with TestClient(app) as client:
            reg, _ = _patch_spawn(app)
            with client.websocket_connect(_resume_url(stale_sid)) as ws:
                _send_resize(ws)
                msg = _recv_json(ws, "session_info")
                assert msg["session_id"] == discovered_sid
                assert msg["session_id"] != stale_sid

            # The registry pool entry was rekeyed to the discovered id.
            assert reg.get_session(discovered_sid) is not None
            assert reg.get_session(stale_sid) is None


# ---------------------------------------------------------------------------
# Reused warm session — confirmed synchronously, no discovery needed
# ---------------------------------------------------------------------------


def test_reused_warm_session_confirms_immediately(app):
    """Reconnecting to an already-warm session confirms the requested id
    without going through discovery (was_reused=True)."""
    sid = _uuid()
    with patch(
        "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session",
        return_value=None,
    ) as mock_discover:
        with TestClient(app) as client:
            _patch_spawn(app)

            # First connect: fresh spawn, keeps the session warm in the pool.
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                msg = _recv_json(ws, "session_info")
                assert msg["session_id"] == sid

            mock_discover.reset_mock()

            # Second connect: reuses the warm session from the pool.
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                msg = _recv_json(ws, "session_info")
                assert msg["session_id"] == sid

        # The reused-session confirmation never touched discovery.
        mock_discover.assert_not_called()


# ---------------------------------------------------------------------------
# Cold resume, session file already on disk — confirmed immediately
# ---------------------------------------------------------------------------


def test_cold_resume_with_existing_file_confirms_immediately(app, tmp_path):
    """A not-currently-warm resume whose session file already exists on disk
    is trusted immediately — no discovery poll needed."""
    sid = _uuid()
    sessions_dir = tmp_path / "claude_sessions"
    sessions_dir.mkdir()
    (sessions_dir / f"{sid}.jsonl").write_text("")

    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: sessions_dir):
        with patch(
            "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session"
        ) as mock_discover:
            with TestClient(app) as client:
                _patch_spawn(app)
                with client.websocket_connect(_resume_url(sid)) as ws:
                    _send_resize(ws)
                    msg = _recv_json(ws, "session_info")
                    assert msg["session_id"] == sid

            mock_discover.assert_not_called()


# ---------------------------------------------------------------------------
# Stale id, file appears mid-poll — discovery must use the full window
# ---------------------------------------------------------------------------


def test_stale_resume_with_delayed_file_confirms_discovered_id(app, tmp_path):
    """A stale/absent --resume-id whose replacement session file only shows
    up partway through the discovery window is still caught, and the window
    used is the full one (not a shortened, easy-to-miss one).

    Every entry into the "no file on disk" branch is racing the CLI's own
    startup to write its new session file — a too-short window would let
    that race lose by default and silently confirm the stale requested id
    instead (defeating the whole point of this confirmation).
    """
    stale_sid = _uuid()
    discovered_sid = _uuid()
    sessions_dir = tmp_path / "claude_sessions"
    sessions_dir.mkdir()

    real_discover_new_session = SessionDiscovery.discover_new_session
    captured_timeouts: list[float] = []

    def _delayed_discover(self, before, timeout=15.0):
        captured_timeouts.append(timeout)
        # The file doesn't exist yet when polling starts — it shows up
        # partway through, simulating real (slow, MCP-heavy) CLI startup.
        # A too-short timeout would give up before this fires.
        threading.Timer(
            0.6, lambda: (sessions_dir / f"{discovered_sid}.jsonl").write_text("")
        ).start()
        return real_discover_new_session(self, before, timeout=timeout)

    with patch.object(SessionDiscovery, "_resolve_sessions_dir", lambda self: sessions_dir):
        with patch.object(SessionDiscovery, "discover_new_session", _delayed_discover):
            with TestClient(app) as client:
                reg, _ = _patch_spawn(app)
                with client.websocket_connect(_resume_url(stale_sid)) as ws:
                    _send_resize(ws)
                    msg = _recv_json(ws, "session_info", max_frames=60)
                    assert msg["session_id"] == discovered_sid
                    assert msg["session_id"] != stale_sid

                assert reg.get_session(discovered_sid) is not None
                assert reg.get_session(stale_sid) is None

    # The stale branch must use the same (full) window as the new-session
    # discovery path — not a shortened one that races the CLI and loses.
    assert captured_timeouts == [15.0]
