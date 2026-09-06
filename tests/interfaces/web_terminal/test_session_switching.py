"""WebSocket session switching tests — L1 integration + L2 contract.

L1 (Integration): Real PtyRegistry, fake PtySession via patched _spawn_session.
Tests the WebSocket message protocol end-to-end through the real handler.

L2 (Contract): Mocked PtyRegistry with a stateful pool behind it.
Tests that terminal_ws calls registry methods in the correct sequence. The
handler takes every key through ``acquire_surface``, which inspects the pool
before spawning and checks the spawn was pooled afterwards, so the mock keeps
a real dict of what sits under each key rather than a fixed return value.

All tests connect in ``mode=resume`` with a pre-set UUID to avoid the
5-second session-discovery poll that fires for new sessions. The
``sessions_dir`` fixture below additionally seeds an on-disk session file
for every id these tests connect to or switch to: ``terminal_ws`` resumes
only ids whose transcript exists (or whose PTY is still warm), and these
tests exercise registry/pool behavior, not that boundary — which has its own
cases at the end of the L1 class and in ``test_ws_resume_confirm.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid as uuid_mod
from unittest.mock import MagicMock, patch

import anyio
import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes.websocket import _TerminalChannel
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from tests.interfaces.web_terminal._fakes import FakePtySession

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: How long one frame may take to arrive before the test fails. A handler
#: that answers nothing is a regression, not a stall.
RECV_TIMEOUT_S = 10.0


async def _receive_within(rx, seconds: float):
    with anyio.fail_after(seconds):
        return await rx.receive()


def _recv(ws):
    """One frame off the socket, or ``TimeoutError`` after ``RECV_TIMEOUT_S``.

    ``ws.receive()`` waits on the portal with no deadline, and the test thread
    then sits in a lock wait that pytest-timeout's signal cannot interrupt.
    The deadline runs inside the portal, on the stream the session reads. A
    server close surfaces at once as ``WebSocketDisconnect``.
    """
    message = ws.portal.call(_receive_within, ws._send_rx, RECV_TIMEOUT_S)
    ws._raise_on_close(message)
    return message


def _recv_json(ws, msg_type: str, max_frames: int = 30):
    """Receive frames until a JSON message with the given ``type`` arrives.

    Skips binary frames.  Raises ``AssertionError`` if ``msg_type`` is not
    found within *max_frames* frames, ``TimeoutError`` if a frame is late.
    """
    collected = []
    for _ in range(max_frames):
        raw = _recv(ws)
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


@pytest.fixture()
def sessions_dir(tmp_path, monkeypatch):
    """Point ``SessionDiscovery`` at a tmp dir and return it.

    Seed a session id's file here (see ``_seed_session_file``) before an
    initial resume connect so ``terminal_ws``'s resume confirmation takes
    its trusted synchronous path — the id already exists on disk — instead
    of polling discovery, which these tests have no interest in exercising.
    """
    d = tmp_path / "claude_sessions"
    d.mkdir()
    monkeypatch.setattr(SessionDiscovery, "_resolve_sessions_dir", lambda self: d)
    return d


def _seed_session_file(sessions_dir, session_id: str) -> None:
    """Create an (empty) on-disk session file for *session_id*."""
    (sessions_dir / f"{session_id}.jsonl").write_text("")


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

    def test_connect_and_resize(self, app, sessions_dir):
        """Connecting + sending resize completes without error."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            self._patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)

    # -- switch_session happy path --

    def test_switch_returns_session_switched(self, app, sessions_dir):
        """Switching to a new UUID returns ``session_switched``."""
        initial, target = _uuid(), _uuid()
        _seed_session_file(sessions_dir, initial)
        _seed_session_file(sessions_dir, target)
        with TestClient(app) as client:
            self._patch_spawn(app)
            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": target})
                msg = _recv_json(ws, "session_switched")
                assert msg["session_id"] == target

    def test_switch_same_session_is_noop(self, app, sessions_dir):
        """Switching to the current session confirms without respawning."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            _, spawned = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": sid})
                msg = _recv_json(ws, "session_switched")
                assert msg["session_id"] == sid
                assert len(spawned) == 1  # no extra spawn

    # -- switch_session error paths --

    def test_switch_invalid_uuid_returns_error(self, app, sessions_dir):
        """Non-UUID session_id returns an error message."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            self._patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": "not-a-uuid"})
                msg = _recv_json(ws, "error")
                assert "Invalid" in msg["message"]

    # -- warm session reuse --

    def test_switch_back_reuses_warm_session(self, app, sessions_dir):
        """A → B → A reuses session A from the pool (no extra spawn)."""
        a, b = _uuid(), _uuid()
        _seed_session_file(sessions_dir, a)
        _seed_session_file(sessions_dir, b)
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

    def test_pool_contains_both_after_switch(self, app, sessions_dir):
        """After A → B, both sessions remain in the registry pool."""
        a, b = _uuid(), _uuid()
        _seed_session_file(sessions_dir, a)
        _seed_session_file(sessions_dir, b)
        with TestClient(app) as client:
            reg, _ = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(a)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": b})
                _recv_json(ws, "session_switched")

                assert reg.get_session(a) is not None
                assert reg.get_session(b) is not None

    def test_triple_switch_spawns_three(self, app, sessions_dir):
        """A → B → C creates three sessions total."""
        a, b, c = _uuid(), _uuid(), _uuid()
        for sid in (a, b, c):
            _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            _, spawned = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(a)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": b})
                _recv_json(ws, "session_switched")
                ws.send_json({"type": "switch_session", "session_id": c})
                _recv_json(ws, "session_switched")
                assert len(spawned) == 3

    # -- the resume boundary on the switch path --

    def test_switch_to_an_id_without_transcript_is_refused_in_place(self, app, sessions_dir):
        """No warm PTY and no transcript for the target: refused, nothing torn down.

        The picker lists only ids with a transcript, so this is the race where
        one vanished between listing and clicking, or a stale pointer replayed
        through the cold-resume fallback. Spawning ``--resume`` there produced
        a child that printed ``No conversation found`` and died. The handler
        answers ``transcript_missing`` for the target and keeps the operator on
        the session they were on.
        """
        initial, missing = _uuid(), _uuid()
        _seed_session_file(sessions_dir, initial)
        with TestClient(app) as client:
            reg, spawned = self._patch_spawn(app)
            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                _sync_after_connect(ws, initial)

                ws.send_json({"type": "switch_session", "session_id": missing})
                msg = _recv_json(ws, "transcript_missing")
                assert msg["session_id"] == missing

                # Still on the initial session: a switch to it is the no-op
                # answer, and the pool holds only what it held before.
                ws.send_json({"type": "switch_session", "session_id": initial})
                assert _recv_json(ws, "session_switched")["session_id"] == initial
                assert len(spawned) == 1
                assert reg.get_session(missing) is None
                assert reg.get_session(initial) is spawned[0]

    def test_switch_to_a_warm_session_without_transcript_is_allowed(self, app, sessions_dir):
        """A pooled live PTY is a session whether or not its transcript exists yet."""
        a, never_prompted = _uuid(), _uuid()
        _seed_session_file(sessions_dir, a)
        with TestClient(app) as client:
            _, spawned = self._patch_spawn(app)
            # Open a fresh session (no transcript until the first prompt) and
            # leave it warm in the pool.
            with client.websocket_connect("/ws/terminal") as ws:
                _send_resize(ws)
                never_prompted = _recv_json(ws, "session_info")["session_id"]

            with client.websocket_connect(_resume_url(a)) as ws:
                _send_resize(ws)
                ws.send_json({"type": "switch_session", "session_id": never_prompted})
                assert _recv_json(ws, "session_switched")["session_id"] == never_prompted
                assert len(spawned) == 2  # a + the warm one, reused


# ---------------------------------------------------------------------------
# L2: Contract — mocked PtyRegistry
# ---------------------------------------------------------------------------


class TestSessionSwitchingContract:
    """L2 — verifies handler → registry call sequences."""

    @staticmethod
    def _mock_registry(app, fake_session=None):
        """Replace the registry with a MagicMock over a real pool dict.

        ``get_or_create_session`` hands out *fake_session* first, then
        whatever a test appends to ``mock_reg.hand_out``, and pools it under
        the key; ``get_session`` reads that pool, so the hand-off door sees an
        empty key before the spawn and the spawned session after it — and the
        handler's teardown asks it who currently owns the key.
        test_disconnect_leaves_a_replacement_session_alone below replaces the
        pool entry to model a newer handler's session under the same key.
        """
        if fake_session is None:
            fake_session = FakePtySession()
        pool: dict[str, FakePtySession] = {}
        hand_out: list[FakePtySession] = [fake_session]
        mock_reg = MagicMock(spec=PtyRegistry)

        def get_or_create(key, *_args, **_kwargs):
            pooled = pool.get(key)
            if pooled is not None and pooled.is_alive:
                return pooled, True
            session = hand_out.pop(0) if hand_out else FakePtySession()
            pool[key] = session
            return session, False

        mock_reg.get_or_create_session.side_effect = get_or_create
        mock_reg.get_session.side_effect = pool.get
        mock_reg.pop_session.side_effect = lambda key: pool.pop(key, None)
        mock_reg.pop_lru_victim.return_value = None  # a pool below capacity evicts nothing
        mock_reg.attach_session.return_value = True
        mock_reg.is_attached.return_value = False
        mock_reg.attached_owner.return_value = None
        mock_reg.pool = pool
        mock_reg.hand_out = hand_out
        app.state.pty_registry = mock_reg
        return mock_reg, fake_session

    # -- initial connection contract --

    def test_connect_calls_get_or_create_then_attach(self, app, sessions_dir):
        """On connect: get_or_create_session(key, ...) then attach_session(key)."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            mock_reg, _ = self._mock_registry(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)

        # get_or_create_session was called with the session UUID as key
        mock_reg.get_or_create_session.assert_called_once()
        key_used = mock_reg.get_or_create_session.call_args[0][0]
        assert key_used == sid

        # attach_session was called (at least once) with the same key, and
        # with the handler's attachment token alongside it.
        attach_calls = [c for c in mock_reg.attach_session.call_args_list if c.args[0] == sid]
        assert len(attach_calls) >= 1
        assert all(len(c.args) == 2 for c in attach_calls)

    # -- switch_session contract --

    def test_switch_calls_detach_create_attach_in_order(self, app, sessions_dir):
        """Switch: detach(old) → get_or_create(new) → attach(new), in order."""
        initial, target = _uuid(), _uuid()
        _seed_session_file(sessions_dir, initial)
        _seed_session_file(sessions_dir, target)
        with TestClient(app) as client:
            mock_reg, _ = self._mock_registry(app)
            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                _sync_after_connect(ws, initial)

                # Reset to isolate switch calls from initial-connect calls
                mock_reg.reset_mock()
                mock_reg.hand_out.append(FakePtySession())

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

        # Verify args. The detach names the old key and the token the attach
        # that follows it hands to the new one — one token per handler.
        detach_call = mock_reg.method_calls[detach_i]
        assert detach_call[0] == "detach_session"
        assert detach_call[1][0] == initial
        token = detach_call[1][1]
        assert mock_reg.method_calls[attach_after_create[0]][1] == (target, token)
        create_call = mock_reg.method_calls[create_i]
        assert create_call[1][0] == target  # first positional arg = target UUID

    def test_invalid_uuid_skips_registry(self, app, sessions_dir):
        """Invalid UUID is rejected before any switch-related registry call."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
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

    def test_disconnect_detaches_live_session(self, app, sessions_dir):
        """On WS close with live session: detach but do NOT terminate."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            mock_reg, fake = self._mock_registry(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                mock_reg.reset_mock()

        # Handler's finally: detach(current_key, token)
        assert mock_reg.detach_session.call_args.args[0] == sid
        # Session is alive → nothing is terminated
        mock_reg.terminate_session.assert_not_called()
        mock_reg.terminate_session_if_owner.assert_not_called()

    def test_disconnect_terminates_dead_session(self, app, sessions_dir):
        """On WS close with dead session: detach AND terminate."""
        sid = _uuid()
        import time

        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            dead = FakePtySession()
            mock_reg, _ = self._mock_registry(app, fake_session=dead)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                mock_reg.reset_mock()
                # Kill the session while connected
                dead._alive = False
                time.sleep(0.2)  # let output loop notice and exit

        assert mock_reg.detach_session.call_args.args[0] == sid
        # Terminated through the owner-checked entry point, which takes the
        # session this handler owns as well as the key — see
        # test_disconnect_leaves_a_replacement_session_alone.
        mock_reg.terminate_session_if_owner.assert_called_with(sid, dead)

    def test_disconnect_leaves_a_replacement_session_alone(self, app, sessions_dir):
        """A stale handler's teardown must not touch a newer session.

        Two handlers meeting on one pool key is ordinary: a second tab (or a
        reload whose disconnect the server sees late) resumes the id, finds
        this PTY dead, and gets a replacement spawned under the same key.
        Teardown keyed on the id alone would then terminate the replacement
        and clear its attachment, killing a terminal someone is looking at.
        """
        sid = _uuid()
        import time

        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            dead = FakePtySession()
            mock_reg, _ = self._mock_registry(app, fake_session=dead)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                mock_reg.reset_mock()
                dead._alive = False
                time.sleep(0.2)  # let output loop notice and exit
                # A newer handler has since put its own session under this key.
                mock_reg.pool[sid] = FakePtySession()

        # The newer handler's attachment is left intact...
        mock_reg.detach_session.assert_not_called()
        # ...and the registry entry is not terminated by key. The dead PTY is
        # still handed to the owner-checked call, which terminates the process
        # this handler owns without disturbing the pool.
        mock_reg.terminate_session.assert_not_called()
        mock_reg.terminate_session_if_owner.assert_called_with(sid, dead)


# ---------------------------------------------------------------------------
# L0: the socket reader under the hand-off door
# ---------------------------------------------------------------------------


class _MemoryWebSocket:
    """A socket that hands a frame straight to a parked receiver.

    Starlette's test client feeds the app from an anyio memory stream, which
    delivers a frame to a waiting receiver in the sender's own loop turn; a
    receiver cancelled before it runs then drops that frame.
    """

    def __init__(self) -> None:
        self.tx, self.rx = anyio.create_memory_object_stream[dict](10)

    async def receive(self) -> dict:
        return await self.rx.receive()


async def _ticks_until(condition, what: str) -> None:
    """Run the loop tick by tick until *condition* holds; a bounded wait."""
    for _ in range(50):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"never reached: {what}")


async def _park(ws: _MemoryWebSocket, task: asyncio.Task) -> None:
    """Let *task* run until its socket read is a parked receiver on the stream."""
    await _ticks_until(lambda: bool(ws.rx._state.waiting_receivers), "a parked receiver")
    assert not task.done()


def _switch_frame() -> dict:
    return {
        "type": "websocket.receive",
        "text": json.dumps({"type": "switch_session", "session_id": _uuid()}),
    }


class TestTerminalChannelReader:
    """The frame that lands as the door returns must reach the main loop."""

    async def test_frame_delivered_as_the_side_reader_is_cancelled_is_kept(self):
        ws = _MemoryWebSocket()
        channel = _TerminalChannel(ws)
        reader = asyncio.ensure_future(channel._read_while_acquiring())
        await _park(ws, reader)

        frame = _switch_frame()
        ws.tx.send_nowait(frame)  # handed straight to the parked receiver
        reader.cancel()  # before any wakeup runs
        with pytest.raises(asyncio.CancelledError):
            await reader

        assert await asyncio.wait_for(channel.receive(), 1.0) == frame
        channel.release()

    async def test_frame_read_before_the_side_reader_is_cancelled_is_kept(self):
        """The read has completed, the reader has not run: the cancel lands
        on its wakeup, and the completed read must still be collected."""
        ws = _MemoryWebSocket()
        channel = _TerminalChannel(ws)
        reader = asyncio.ensure_future(channel._read_while_acquiring())
        await _park(ws, reader)

        frame = _switch_frame()
        ws.tx.send_nowait(frame)
        await _ticks_until(lambda: channel._read is not None and channel._read.done(), "the read")
        assert not channel.deferred  # the reader has not been woken yet
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader

        assert await asyncio.wait_for(channel.receive(), 1.0) == frame
        channel.release()

    async def test_release_cancels_a_read_left_in_flight(self):
        ws = _MemoryWebSocket()
        channel = _TerminalChannel(ws)
        waiter = asyncio.ensure_future(channel.receive())
        await _park(ws, waiter)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        pending = channel._read
        assert pending is not None and not pending.done()
        channel.release()
        await asyncio.sleep(0)
        assert pending.cancelled()
        assert channel._read is None
