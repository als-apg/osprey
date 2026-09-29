"""Tests for PtyRegistry's pool, over mock PtySession objects.

No real PTY is spawned. Covered here: reuse, respawn and LRU eviction on the
production spawn path; ``pop_session``, the bookkeeping half of
``terminate_session`` that leaves the kill to the caller; ``pop_lru_victim``,
the eviction pass without the kill; hand-off reservations, which hold a key
over the gap where it is attached to nobody; and ``reinsert``, which puts a
popped session back when its kill did not take. Attachment ownership rules
live in ``test_pty_attach_owner.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.pty_manager import PtyRegistry

#: Stand-in for a consumer's attachment token. Attachment is owned by a token,
#: not by a key, so every attach/detach in these tests names one.
OWNER = object()


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
        live.terminate.assert_not_called()

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

    @pytest.mark.parametrize("hold", ["attach", "reserve"])
    def test_held_session_not_evicted(self, hold):
        """An attached or reserved session is never evicted, even when oldest."""
        registry = PtyRegistry(max_background=2)

        s1 = _mock_session()
        s2 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        if hold == "attach":
            registry.attach_session("a", OWNER)
        else:
            registry.reserve("a")

        with patch.object(registry, "_spawn_session") as mock_spawn:
            mock_spawn.return_value = _mock_session()
            registry.get_or_create_session("c", ["cmd"], 24, 80)

        # s1 is held — must NOT be evicted. s2 (next oldest) evicted instead.
        s1.terminate.assert_not_called()
        s2.terminate.assert_called_once()
        assert "a" in registry._sessions
        assert "b" not in registry._sessions
        assert "c" in registry._sessions

    def test_cleanup_all_terminates_pool_sessions(self):
        """cleanup_all terminates all sessions, including detached pool entries."""
        registry = PtyRegistry(max_background=3)
        s1 = _mock_session()
        s2 = _mock_session()
        s3 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry._sessions["c"] = s3
        registry._env_fingerprints.update({"a": "fp", "b": "fp", "c": "fp"})
        registry.attach_session("a", OWNER)

        registry.cleanup_all()

        s1.terminate.assert_called_once()
        s2.terminate.assert_called_once()
        s3.terminate.assert_called_once()
        assert len(registry._sessions) == 0
        assert len(registry._attached) == 0
        assert registry._env_fingerprints == {}

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


class TestPopSession:
    """``pop_session`` empties the key and returns the live session."""

    def test_returns_the_session_without_terminating_it(self):
        """The child is handed to the caller alive — the kill is theirs."""
        registry = PtyRegistry(max_background=3)
        session = _mock_session()
        registry._sessions["k"] = session

        popped = registry.pop_session("k")

        assert popped is session
        session.terminate.assert_not_called()
        assert popped.is_alive

    def test_forgets_the_key_completely(self):
        """Pool entry, fingerprint and attachment all go."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["k"] = _mock_session()
        registry._env_fingerprints["k"] = "fp"
        registry.attach_session("k", OWNER)

        registry.pop_session("k")

        assert "k" not in registry._sessions
        assert "k" not in registry._env_fingerprints
        assert not registry.is_attached("k")

    def test_unknown_key_returns_none(self):
        """No pooled session, no error — the bookkeeping is a no-op."""
        registry = PtyRegistry(max_background=3)

        assert registry.pop_session("absent") is None

    def test_reservation_survives_the_pop(self):
        """A hand-off pops and re-fills under one reservation."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["k"] = _mock_session()
        registry.reserve("k")

        registry.pop_session("k")

        assert registry.is_reserved("k")

    def test_reinsert_puts_a_popped_session_back_unheld(self):
        """A survivor of a failed kill is re-pooled as an ordinary background entry."""
        registry = PtyRegistry(max_background=5)
        pty = _mock_session()
        with patch.object(registry, "_spawn_session", return_value=pty):
            registry.get_or_create_session("k", ["fake"])
        registry.attach_session("k", object())
        assert registry.pop_session("k") is pty

        assert registry.reinsert("k", pty) is True
        assert registry.get_session("k") is pty
        assert not registry.is_attached("k")
        assert not registry.is_reserved("k")
        # An occupied key is left alone.
        assert registry.reinsert("k", _mock_session()) is False
        assert registry.get_session("k") is pty


class TestTerminateSessionDelegates:
    """``terminate_session`` is ``pop_session`` plus the kill."""

    def test_pops_and_terminates(self):
        """Same observable teardown as before the split."""
        registry = PtyRegistry(max_background=3)
        session = _mock_session()
        registry._sessions["k"] = session
        registry._env_fingerprints["k"] = "fp"
        registry.attach_session("k", OWNER)

        registry.terminate_session("k")

        session.terminate.assert_called_once()
        assert "k" not in registry._sessions
        assert "k" not in registry._env_fingerprints
        assert not registry.is_attached("k")

    def test_unknown_key_kills_nothing(self):
        """No session under the key means nothing to terminate."""
        registry = PtyRegistry(max_background=3)
        other = _mock_session()
        registry._sessions["k"] = other

        registry.terminate_session("absent")

        other.terminate.assert_not_called()
        assert "k" in registry._sessions


class TestPopLruVictim:
    """``pop_lru_victim`` is the eviction pass without the kill."""

    def test_nothing_is_popped_below_capacity(self):
        registry = PtyRegistry(max_background=3)
        registry._sessions["a"] = _mock_session()

        assert registry.pop_lru_victim() is None
        assert "a" in registry._sessions

    def test_the_oldest_unheld_session_is_returned_alive_and_forgotten(self):
        """The victim leaves the pool with its bookkeeping; the kill is the caller's."""
        registry = PtyRegistry(max_background=2)
        s1 = _mock_session()
        s2 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry._env_fingerprints["a"] = "fp"

        victim = registry.pop_lru_victim()

        assert victim is s1
        s1.terminate.assert_not_called()
        assert "a" not in registry._sessions
        assert "a" not in registry._env_fingerprints
        assert "b" in registry._sessions

    def test_held_sessions_are_stepped_over(self):
        """An attached or reserved entry is never the victim, even when oldest."""
        registry = PtyRegistry(max_background=3)
        s1 = _mock_session()
        s2 = _mock_session()
        s3 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry._sessions["c"] = s3
        registry.attach_session("a", OWNER)
        registry.reserve("b")

        assert registry.pop_lru_victim() is s3
        assert registry.pop_lru_victim() is None
        assert "a" in registry._sessions and "b" in registry._sessions


class TestReservations:
    """Reservations keep the eviction pass off a key mid-hand-off."""

    def test_unreserve_restores_eviction_eligibility(self):
        """Once the hand-off releases the key it is a background session again."""
        registry = PtyRegistry(max_background=2)
        s1 = _mock_session()
        s2 = _mock_session()
        registry._sessions["a"] = s1
        registry._sessions["b"] = s2
        registry.reserve("a")
        registry.unreserve("a")

        with patch.object(registry, "_spawn_session") as mock_spawn:
            mock_spawn.return_value = _mock_session()
            registry.get_or_create_session("c", ["cmd"], 24, 80)

        s1.terminate.assert_called_once()
        assert "a" not in registry._sessions
        assert "b" in registry._sessions

    def test_reserve_is_idempotent_and_release_is_total(self):
        """Reserving twice is reserving once; one unreserve clears it."""
        registry = PtyRegistry(max_background=3)

        registry.reserve("k")
        registry.reserve("k")
        assert registry.is_reserved("k")

        registry.unreserve("k")
        assert not registry.is_reserved("k")

    def test_unknown_key_is_unreserved(self):
        """Never-reserved keys report false and release without error."""
        registry = PtyRegistry(max_background=3)

        assert not registry.is_reserved("absent")
        registry.unreserve("absent")
        assert not registry.is_reserved("absent")

    def test_reserving_a_key_with_no_session_is_allowed(self):
        """A hand-off may reserve the key before the entry exists."""
        registry = PtyRegistry(max_background=3)

        registry.reserve("k")

        assert registry.is_reserved("k")
        assert registry.get_session("k") is None

    def test_reservation_is_not_an_attachment(self):
        """A reserved key is still free for the incoming surface to attach."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["k"] = _mock_session()
        registry.reserve("k")

        assert not registry.is_attached("k")
        assert registry.attach_session("k", OWNER)

    def test_cleanup_all_drops_reservations(self):
        """Shutdown leaves no reservation behind to block a later pool."""
        registry = PtyRegistry(max_background=3)
        registry._sessions["k"] = _mock_session()
        registry.reserve("k")

        registry.cleanup_all()

        assert not registry.is_reserved("k")
