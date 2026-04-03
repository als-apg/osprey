"""Tests for PtyRegistry LRU session pool behavior.

Uses mock PtySession objects (no real PTY spawning) to verify pool
semantics: reuse, eviction, attach/detach, and rekey.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osprey.interfaces.web_terminal.pty_manager import PtyRegistry


def _mock_session(alive: bool = True) -> MagicMock:
    """Create a mock PtySession with configurable is_alive."""
    s = MagicMock()
    s.is_alive = alive
    s.resize = MagicMock()
    s.terminate = MagicMock()
    return s


class TestPtyRegistryPool:
    """Unit tests for LRU pool methods on PtyRegistry."""

    def test_get_or_create_spawns_new_session(self):
        """When key doesn't exist, a new session is created."""
        registry = PtyRegistry(max_background=3)

        with patch.object(registry, "_spawn_session") as mock_spawn:
            mock_spawn.return_value = _mock_session()
            session, was_reused = registry.get_or_create_session(
                "sess-1", ["claude", "--resume", "sess-1"], 24, 80
            )

        assert not was_reused
        mock_spawn.assert_called_once()
        assert session is not None

    def test_get_or_create_reuses_live_session(self):
        """When key exists and session is alive, reuse it."""
        registry = PtyRegistry(max_background=3)
        live = _mock_session(alive=True)
        registry._sessions["sess-1"] = live

        session, was_reused = registry.get_or_create_session(
            "sess-1", ["claude", "--resume", "sess-1"], 24, 80
        )

        assert was_reused
        assert session is live
        # Should resize to current dimensions
        live.resize.assert_called_once_with(24, 80)

    def test_get_or_create_respawns_dead_session(self):
        """When key exists but session is dead, remove and respawn."""
        registry = PtyRegistry(max_background=3)
        dead = _mock_session(alive=False)
        registry._sessions["sess-1"] = dead

        with patch.object(registry, "_spawn_session") as mock_spawn:
            fresh = _mock_session()
            mock_spawn.return_value = fresh
            session, was_reused = registry.get_or_create_session(
                "sess-1", ["claude", "--resume", "sess-1"], 24, 80
            )

        assert not was_reused
        assert session is fresh

    def test_lru_eviction_at_capacity(self):
        """When pool is full, oldest non-attached session is evicted."""
        registry = PtyRegistry(max_background=2)

        # Fill pool with 2 sessions
        s1 = _mock_session()
        s2 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2

        with patch.object(registry, "_spawn_session") as mock_spawn:
            s3 = _mock_session()
            mock_spawn.return_value = s3
            registry.get_or_create_session("c", ["cmd"], 24, 80)

        # s1 (oldest) should have been evicted and terminated
        s1.terminate.assert_called_once()
        assert "a" not in registry._sessions
        assert "b" in registry._sessions
        assert "c" in registry._sessions

    def test_attached_session_not_evicted(self):
        """Attached sessions must not be evicted, even if oldest."""
        registry = PtyRegistry(max_background=2)

        s1 = _mock_session()
        s2 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry.attach_session("a")  # protect s1

        with patch.object(registry, "_spawn_session") as mock_spawn:
            s3 = _mock_session()
            mock_spawn.return_value = s3
            registry.get_or_create_session("c", ["cmd"], 24, 80)

        # s1 is attached — must NOT be evicted. s2 (next oldest) evicted instead.
        s1.terminate.assert_not_called()
        s2.terminate.assert_called_once()
        assert "a" in registry._sessions
        assert "b" not in registry._sessions
        assert "c" in registry._sessions

    def test_detach_does_not_terminate(self):
        """Detaching a session removes from _attached but doesn't kill it."""
        registry = PtyRegistry(max_background=3)
        s = _mock_session()
        registry._sessions["x"] = s
        registry.attach_session("x")

        registry.detach_session("x")

        s.terminate.assert_not_called()
        assert "x" not in registry._attached
        assert "x" in registry._sessions  # still in pool

    def test_rekey_session(self):
        """rekey_session moves entry from old key to new key."""
        registry = PtyRegistry(max_background=3)
        s = _mock_session()
        registry._sessions["temp-key"] = s
        registry.attach_session("temp-key")

        registry.rekey_session("temp-key", "real-uuid")

        assert "temp-key" not in registry._sessions
        assert registry._sessions["real-uuid"] is s
        # _attached should also be updated
        assert "temp-key" not in registry._attached
        assert "real-uuid" in registry._attached

    def test_rekey_noop_when_old_key_missing(self):
        """rekey_session does nothing if old key doesn't exist."""
        registry = PtyRegistry(max_background=3)
        registry.rekey_session("nonexistent", "new-key")
        assert "new-key" not in registry._sessions

    def test_cleanup_all_terminates_pool_sessions(self):
        """cleanup_all terminates all sessions, including detached pool entries."""
        registry = PtyRegistry(max_background=3)
        s1 = _mock_session()
        s2 = _mock_session()
        s3 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry._sessions["c"] = s3
        registry.attach_session("a")

        registry.cleanup_all()

        s1.terminate.assert_called_once()
        s2.terminate.assert_called_once()
        s3.terminate.assert_called_once()
        assert len(registry._sessions) == 0
        assert len(registry._attached) == 0

    def test_attach_returns_false_if_already_attached(self):
        """attach_session rejects double-attach (prevents concurrent fd reads)."""
        registry = PtyRegistry(max_background=3)
        s = _mock_session()
        registry._sessions["x"] = s

        assert registry.attach_session("x") is True
        assert registry.attach_session("x") is False

    def test_attach_returns_false_if_not_in_pool(self):
        """attach_session returns False for unknown session keys."""
        registry = PtyRegistry(max_background=3)
        assert registry.attach_session("unknown") is False

    def test_detach_noop_for_unknown_key(self):
        """detach_session is safe to call with unknown keys."""
        registry = PtyRegistry(max_background=3)
        registry.detach_session("nonexistent")  # should not raise

    def test_lru_ordering_after_reuse(self):
        """Reusing a session LRU-bumps it (moves to end)."""
        registry = PtyRegistry(max_background=3)
        s1 = _mock_session()
        s2 = _mock_session()
        s3 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry._sessions["c"] = s3

        # Access "a" — should bump it to newest
        registry.get_or_create_session("a", ["cmd"], 24, 80)

        # Now add "d" — "b" should be evicted (oldest after bump)
        with patch.object(registry, "_spawn_session") as mock_spawn:
            s4 = _mock_session()
            mock_spawn.return_value = s4
            registry.get_or_create_session("d", ["cmd"], 24, 80)

        s2.terminate.assert_called_once()
        assert "b" not in registry._sessions
        assert "a" in registry._sessions
