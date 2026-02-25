"""WebSocket session switching tests — L1 integration + L2 contract.

L1 (Integration): Real PtyRegistry, fake PtySession via patched _spawn_session.
Tests the WebSocket message protocol end-to-end through the real handler.

L2 (Contract): Fully mocked PtyRegistry.
Tests that terminal_ws calls registry methods in the correct sequence.

All tests connect in ``mode=resume`` with a pre-set UUID to avoid the
5-second session-discovery poll that fires for new sessions.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid as uuid_mod
from unittest.mock import MagicMock, call, patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    def start(self, initial_rows=24, initial_cols=80, extra_env=None):
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

    Skips binary frames.  Raises ``AssertionError`` if ``msg_type`` is not
    found within *max_frames* frames.

    .. note::

       If the server sends **nothing**, ``ws.receive()`` blocks indefinitely.
       Use pytest-timeout or a CI-level timeout as a safety net.
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
    """Build WS URL in resume mode (avoids session-discovery task)."""
    return f"/ws/terminal?session_id={session_id}&mode=resume"


def _send_resize(ws, cols: int = 80, rows: int = 24):
    """Send the initial resize the handler waits for before spawning."""
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


def _sync_after_connect(ws, session_id: str):
    """Round-trip to ensure the handler finished initial connect processing."""
    ws.send_json({"type": "switch_session", "session_id": session_id})
    _recv_json(ws, "session_switched")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app(tmp_path):
    """Create a web terminal app pointed at a temp project dir."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# L1: Integration — real PtyRegistry, fake PtySession
# ---------------------------------------------------------------------------


class TestSessionSwitchingProtocol:
    """L1 — tests the WebSocket message protocol end-to-end."""

    @staticmethod
    def _patch_spawn(app):
        """Replace ``_spawn_session`` so no real PTY is created.

        Returns ``(registry, spawned_list)`` where *spawned_list* tracks
        every FakePtySession created by the registry.
        """
        reg = app.state.pty_registry
        spawned: list[FakePtySession] = []

        def tracked_spawn(*_args, **_kwargs):
            s = FakePtySession()
            spawned.append(s)
            return s

        reg._spawn_session = tracked_spawn
        return reg, spawned

    # -- basic connectivity --

    def test_connect_and_resize(self, app):
        """Connecting + sending resize completes without error."""
        sid = _uuid()
        with TestClient(app) as client:
            self._patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)

    # -- switch_session happy path --

    def test_switch_returns_session_switched(self, app):
        """Switching to a new UUID returns ``session_switched``."""
        initial, target = _uuid(), _uuid()
        with TestClient(app) as client:
            self._patch_spawn(app)
            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": target})
                msg = _recv_json(ws, "session_switched")
                assert msg["session_id"] == target

    def test_switch_same_session_is_noop(self, app):
        """Switching to the current session confirms without respawning."""
        sid = _uuid()
        with TestClient(app) as client:
            _, spawned = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": sid})
                msg = _recv_json(ws, "session_switched")
                assert msg["session_id"] == sid
                assert len(spawned) == 1  # no extra spawn

    # -- switch_session error paths --

    def test_switch_invalid_uuid_returns_error(self, app):
        """Non-UUID session_id returns an error message."""
        sid = _uuid()
        with TestClient(app) as client:
            self._patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": "not-a-uuid"})
                msg = _recv_json(ws, "error")
                assert "Invalid" in msg["message"]

    # -- warm session reuse --

    def test_switch_back_reuses_warm_session(self, app):
        """A → B → A reuses session A from the pool (no extra spawn)."""
        a, b = _uuid(), _uuid()
        with TestClient(app) as client:
            _, spawned = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(a)) as ws:
                _send_resize(ws)
                _sync_after_connect(ws, a)
                assert len(spawned) == 1  # initial A

                ws.send_json({"type": "switch_session", "session_id": b})
                _recv_json(ws, "session_switched")
                assert len(spawned) == 2  # A + B

                ws.send_json({"type": "switch_session", "session_id": a})
                _recv_json(ws, "session_switched")
                assert len(spawned) == 2  # still 2 — A was reused from pool

    def test_pool_contains_both_after_switch(self, app):
        """After A → B, both sessions remain in the registry pool."""
        a, b = _uuid(), _uuid()
        with TestClient(app) as client:
            reg, _ = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(a)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": b})
                _recv_json(ws, "session_switched")

                assert reg.get_session(a) is not None
                assert reg.get_session(b) is not None

    def test_triple_switch_spawns_three(self, app):
        """A → B → C creates three sessions total."""
        a, b, c = _uuid(), _uuid(), _uuid()
        with TestClient(app) as client:
            _, spawned = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(a)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": b})
                _recv_json(ws, "session_switched")
                ws.send_json({"type": "switch_session", "session_id": c})
                _recv_json(ws, "session_switched")
                assert len(spawned) == 3


# ---------------------------------------------------------------------------
# L2: Contract — mocked PtyRegistry
# ---------------------------------------------------------------------------


class TestSessionSwitchingContract:
    """L2 — verifies handler → registry call sequences."""

    @staticmethod
    def _mock_registry(app, fake_session=None):
        """Replace the registry with a MagicMock that returns *fake_session*."""
        if fake_session is None:
            fake_session = FakePtySession()
        mock_reg = MagicMock(spec=PtyRegistry)
        mock_reg.get_or_create_session.return_value = (fake_session, False)
        mock_reg.attach_session.return_value = True
        app.state.pty_registry = mock_reg
        return mock_reg, fake_session

    # -- initial connection contract --

    def test_connect_calls_get_or_create_then_attach(self, app):
        """On connect: get_or_create_session(key, ...) then attach_session(key)."""
        sid = _uuid()
        with TestClient(app) as client:
            mock_reg, _ = self._mock_registry(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)

        # get_or_create_session was called with the session UUID as key
        mock_reg.get_or_create_session.assert_called_once()
        key_used = mock_reg.get_or_create_session.call_args[0][0]
        assert key_used == sid

        # attach_session was called (at least once) with the same key
        attach_calls = [
            c for c in mock_reg.attach_session.call_args_list if c == call(sid)
        ]
        assert len(attach_calls) >= 1

    # -- switch_session contract --

    def test_switch_calls_detach_create_attach_in_order(self, app):
        """Switch: detach(old) → get_or_create(new) → attach(new), in order."""
        initial, target = _uuid(), _uuid()
        with TestClient(app) as client:
            mock_reg, _ = self._mock_registry(app)
            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                _sync_after_connect(ws, initial)

                # Reset to isolate switch calls from initial-connect calls
                mock_reg.reset_mock()
                new_fake = FakePtySession()
                mock_reg.get_or_create_session.return_value = (new_fake, False)
                mock_reg.attach_session.return_value = True

                ws.send_json({"type": "switch_session", "session_id": target})
                _recv_json(ws, "session_switched")

        # Reconstruct ordered method calls (ignoring cleanup from finally)
        names = [c[0] for c in mock_reg.method_calls]

        # detach must come before get_or_create which must come before attach
        assert "detach_session" in names
        assert "get_or_create_session" in names

        detach_i = names.index("detach_session")
        create_i = names.index("get_or_create_session")
        # Find the attach_session AFTER get_or_create (not the cleanup one)
        attach_indices = [i for i, n in enumerate(names) if n == "attach_session"]
        attach_after_create = [i for i in attach_indices if i > create_i]
        assert attach_after_create, "No attach_session after get_or_create_session"

        assert detach_i < create_i < attach_after_create[0]

        # Verify args
        assert mock_reg.method_calls[detach_i] == call.detach_session(initial)
        create_call = mock_reg.method_calls[create_i]
        assert create_call[1][0] == target  # first positional arg = target UUID

    def test_invalid_uuid_skips_registry(self, app):
        """Invalid UUID is rejected before any switch-related registry call."""
        sid = _uuid()
        with TestClient(app) as client:
            mock_reg, _ = self._mock_registry(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                _sync_after_connect(ws, sid)
                mock_reg.reset_mock()

                ws.send_json({"type": "switch_session", "session_id": "bad"})
                _recv_json(ws, "error")

        # No switch-path calls should have been made
        for c in mock_reg.method_calls:
            name = c[0]
            # Cleanup (detach in finally) is OK — switch-path calls are not
            if name == "get_or_create_session":
                pytest.fail("get_or_create_session called for invalid UUID")

    # -- disconnect contract --

    def test_disconnect_detaches_live_session(self, app):
        """On WS close with live session: detach but do NOT terminate."""
        sid = _uuid()
        with TestClient(app) as client:
            mock_reg, fake = self._mock_registry(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                mock_reg.reset_mock()

        # Handler's finally: detach(current_key)
        mock_reg.detach_session.assert_called_with(sid)
        # Session is alive → terminate_session should NOT be called
        mock_reg.terminate_session.assert_not_called()

    def test_disconnect_terminates_dead_session(self, app):
        """On WS close with dead session: detach AND terminate."""
        sid = _uuid()
        import time

        with TestClient(app) as client:
            dead = FakePtySession()
            mock_reg, _ = self._mock_registry(app, fake_session=dead)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                mock_reg.reset_mock()
                # Kill the session while connected
                dead._alive = False
                time.sleep(0.2)  # let output loop notice and exit

        mock_reg.detach_session.assert_called_with(sid)
        mock_reg.terminate_session.assert_called_with(sid)
