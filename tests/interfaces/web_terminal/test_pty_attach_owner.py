"""Attachment ownership in the PTY pool.

A pool key can be reached by more than one consumer — a second tab, a reload
whose disconnect the server sees late, a view flip. Attachment is therefore
owned by a *token* the consumer holds, not by the key: only the holder can
release it, and a key that is already attached is refused rather than shared,
because two readers on one PTY file descriptor split the child's output
between them.

Two levels here:

- The registry contract: what ``attach_session``/``detach_session`` do with a
  token, and how the token survives a rekey, a terminate and a respawn.
- The handler contract: ``terminal_ws`` honours a refused attach by closing
  with 4409, and releases every key it took by the time the socket is gone.

The pool's LRU behaviour around attachment lives in
``test_pty_registry_pool.py``; the fake PTY here mirrors the harness in
``test_ws_resume_confirm.py``.
"""

from __future__ import annotations

import json
import sys
import uuid as uuid_mod
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from tests.interfaces.web_terminal._fakes import FakePtySession

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session(alive: bool = True) -> MagicMock:
    """A pool entry that records termination without owning a process."""
    s = MagicMock()
    s.is_alive = alive
    s.resize = MagicMock()
    s.terminate = MagicMock()
    return s


def _uuid() -> str:
    return str(uuid_mod.uuid4())


def _resume_url(session_id: str) -> str:
    return f"/ws/terminal?session_id={session_id}&mode=resume"


def _send_resize(ws, cols: int = 80, rows: int = 24):
    """Send the initial resize the handler waits for before spawning."""
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


def _recv_type(ws, msg_type: str, max_frames: int = 30):
    """Receive frames until a JSON message with the given ``type`` arrives."""
    seen = []
    for _ in range(max_frames):
        raw = ws.receive()
        if "text" in raw:
            data = json.loads(raw["text"])
            seen.append(data.get("type"))
            if data.get("type") == msg_type:
                return data
    raise AssertionError(f"'{msg_type}' not received within {max_frames} frames. Got: {seen}")


def _recv_close(ws, max_frames: int = 30) -> dict:
    """The close frame that ends the socket, skipping anything sent first."""
    seen = []
    for _ in range(max_frames):
        raw = ws.receive()
        if raw["type"] == "websocket.close":
            return raw
        seen.append(raw.get("text") or raw["type"])
    raise AssertionError(f"No close frame within {max_frames} frames. Got: {seen}")


@pytest.fixture()
def app(tmp_path):
    """A web terminal app pointed at a temp project dir."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


@pytest.fixture()
def sessions_dir(tmp_path, monkeypatch):
    """Point ``SessionDiscovery`` at a tmp transcript directory."""
    d = tmp_path / "claude_sessions"
    d.mkdir()
    monkeypatch.setattr(SessionDiscovery, "_resolve_sessions_dir", lambda self: d)
    return d


def _seed_session_file(sessions_dir, session_id: str) -> None:
    (sessions_dir / f"{session_id}.jsonl").write_text("")


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
# Registry contract
# ---------------------------------------------------------------------------


class TestAttachmentOwnership:
    """``attach_session``/``detach_session`` answer to a token, not a key."""

    def test_attach_records_the_owner(self):
        """A taken attachment names the token that took it."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["x"] = _mock_session()
        owner = object()

        assert registry.attach_session("x", owner) is True
        assert registry.is_attached("x") is True
        assert registry.attached_owner("x") is owner

    def test_a_free_key_has_no_owner(self):
        """An unattached key — pooled or not — reports neither."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["x"] = _mock_session()

        assert registry.is_attached("x") is False
        assert registry.attached_owner("x") is None
        assert registry.is_attached("never-pooled") is False
        assert registry.attached_owner("never-pooled") is None

    def test_a_refused_attach_leaves_the_owner_alone(self):
        """The holder keeps the key when a second consumer is turned away."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["x"] = _mock_session()
        holder, latecomer = object(), object()
        registry.attach_session("x", holder)

        assert registry.attach_session("x", latecomer) is False
        assert registry.attached_owner("x") is holder

    def test_attach_on_an_unpooled_key_records_nothing(self):
        """A key with no session behind it cannot be attached."""
        registry = PtyRegistry(max_background=3)

        assert registry.attach_session("unknown", object()) is False
        assert registry.is_attached("unknown") is False

    def test_detach_from_a_non_owner_is_a_no_op(self):
        """A departing consumer cannot release a key someone else holds.

        The case this exists for: a handler whose PTY died is torn down after
        a newer one has taken the key over. Releasing the attachment on the
        way out would leave the live consumer's terminal evictable.
        """
        registry = PtyRegistry(max_background=3)
        session = _mock_session()
        registry._sessions["x"] = session
        holder = object()
        registry.attach_session("x", holder)

        registry.detach_session("x", object())

        assert registry.is_attached("x") is True
        assert registry.attached_owner("x") is holder
        session.terminate.assert_not_called()

    def test_detach_from_the_owner_releases_the_key(self):
        """The holder's detach frees the key without killing the session."""
        registry = PtyRegistry(max_background=3)
        session = _mock_session()
        registry._sessions["x"] = session
        owner = object()
        registry.attach_session("x", owner)

        registry.detach_session("x", owner)

        assert registry.is_attached("x") is False
        assert registry.attached_owner("x") is None
        assert "x" in registry._sessions
        session.terminate.assert_not_called()

    def test_detach_of_an_unattached_key_is_a_no_op(self):
        """Nothing to release, nothing to raise — and no LRU bump either."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["a"] = _mock_session()
        registry._sessions["b"] = _mock_session()

        registry.detach_session("a", object())

        assert list(registry._sessions) == ["a", "b"]

    def test_detach_lru_bumps_the_released_session(self):
        """A released session moves to the back of the eviction queue."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["a"] = _mock_session()
        registry._sessions["b"] = _mock_session()
        owner = object()
        registry.attach_session("a", owner)

        registry.detach_session("a", owner)

        assert list(registry._sessions) == ["b", "a"]

    def test_the_owner_survives_a_rekey(self):
        """A session renamed under the holder keeps its holder."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["temp-key"] = _mock_session()
        owner = object()
        registry.attach_session("temp-key", owner)

        registry.rekey_session("temp-key", "real-uuid")

        assert registry.is_attached("temp-key") is False
        assert registry.attached_owner("real-uuid") is owner
        # And the holder can still release it under its new name.
        registry.detach_session("real-uuid", owner)
        assert registry.is_attached("real-uuid") is False

    def test_terminate_clears_the_attachment(self):
        """A terminated session takes its attachment with it."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["x"] = _mock_session()
        registry.attach_session("x", object())

        registry.terminate_session("x")

        assert registry.is_attached("x") is False
        assert registry.attached_owner("x") is None

    def test_respawning_a_dead_entry_clears_the_stale_attachment(self):
        """A key whose child died is free for the consumer that respawns it."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["x"] = _mock_session(alive=False)
        registry.attach_session("x", object())

        with patch.object(registry, "_spawn_session") as mock_spawn:
            mock_spawn.return_value = _mock_session()
            registry.get_or_create_session("x", ["cmd"], 24, 80)

        assert registry.is_attached("x") is False
        assert registry.attach_session("x", object()) is True

    def test_cleanup_all_releases_every_attachment(self):
        """Shutdown leaves no key claimed."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["a"] = _mock_session()
        registry._sessions["b"] = _mock_session()
        registry.attach_session("a", object())

        registry.cleanup_all()

        assert registry.is_attached("a") is False
        assert registry.attached_owner("a") is None


# ---------------------------------------------------------------------------
# Handler contract
# ---------------------------------------------------------------------------


class TestTerminalWsAttachment:
    """``terminal_ws`` holds one token and honours a refused attach."""

    def test_connect_holds_the_key_and_releases_it_on_disconnect(self, app, sessions_dir):
        """The key is attached while the socket is open and free after it."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            reg, _ = _patch_spawn(app)
            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                _recv_type(ws, "session_info")
                assert reg.is_attached(sid) is True

            assert reg.is_attached(sid) is False

    def test_a_switch_moves_the_attachment_and_releases_both_keys(self, app, sessions_dir):
        """One token per handler: it detaches the old key and takes the new.

        Both keys are free after the socket closes, which is only true if the
        token the switch detached with is the token the connect attached with.
        """
        initial, target = _uuid(), _uuid()
        _seed_session_file(sessions_dir, initial)
        _seed_session_file(sessions_dir, target)
        with TestClient(app) as client:
            reg, _ = _patch_spawn(app)
            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                _recv_type(ws, "session_info")
                ws.send_json({"type": "switch_session", "session_id": target})
                _recv_type(ws, "session_switched")

                assert reg.is_attached(initial) is False
                assert reg.is_attached(target) is True

            assert reg.is_attached(initial) is False
            assert reg.is_attached(target) is False

    def test_a_refused_attach_closes_the_socket_with_4409(self, app, sessions_dir):
        """A key someone else is reading is not served half a terminal."""
        sid = _uuid()
        _seed_session_file(sessions_dir, sid)
        with TestClient(app) as client:
            reg, _ = _patch_spawn(app)
            reg.attach_session = lambda key, owner: False

            with client.websocket_connect(_resume_url(sid)) as ws:
                _send_resize(ws)
                message = _recv_close(ws)

        assert message["code"] == 4409

    def test_a_refused_switch_closes_the_socket_with_4409(self, app, sessions_dir):
        """Same refusal on the switch path, once the old key is released."""
        initial, target = _uuid(), _uuid()
        _seed_session_file(sessions_dir, initial)
        _seed_session_file(sessions_dir, target)
        with TestClient(app) as client:
            reg, _ = _patch_spawn(app)

            with client.websocket_connect(_resume_url(initial)) as ws:
                _send_resize(ws)
                _recv_type(ws, "session_info")
                reg.attach_session = lambda key, owner: False

                ws.send_json({"type": "switch_session", "session_id": target})
                message = _recv_close(ws)

        assert message["code"] == 4409
