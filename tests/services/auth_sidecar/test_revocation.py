"""Tests for the auth sidecar's in-memory revocation store."""

from __future__ import annotations

import pytest

from osprey.services.auth_sidecar.revocation import RevocationStore


class FakeClock:
    """A hand-advanced stand-in for ``time.time`` (absolute epoch seconds)."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock: FakeClock) -> RevocationStore:
    return RevocationStore(clock=clock)


def test_unknown_session_is_not_revoked(store: RevocationStore) -> None:
    assert store.is_revoked("never-seen") is False


def test_revoked_session_is_reported_revoked(store: RevocationStore, clock: FakeClock) -> None:
    store.revoke("sess-a", expires_at=clock.now + 3600)

    assert store.is_revoked("sess-a") is True


def test_revocation_is_scoped_to_the_revoked_id(store: RevocationStore, clock: FakeClock) -> None:
    store.revoke("sess-a", expires_at=clock.now + 3600)

    assert store.is_revoked("sess-b") is False


def test_revocation_holds_right_up_to_the_expiry(store: RevocationStore, clock: FakeClock) -> None:
    store.revoke("sess-a", expires_at=clock.now + 3600)

    clock.advance(3599)

    assert store.is_revoked("sess-a") is True


def test_revocation_lapses_at_the_expiry(store: RevocationStore, clock: FakeClock) -> None:
    store.revoke("sess-a", expires_at=clock.now + 3600)

    clock.advance(3600)

    assert store.is_revoked("sess-a") is False


def test_entry_is_dropped_when_its_expiry_is_observed(
    store: RevocationStore, clock: FakeClock
) -> None:
    """Reading a lapsed entry evicts it, so lookups alone bound memory."""
    store.revoke("sess-a", expires_at=clock.now + 60)
    clock.advance(60)

    store.is_revoked("sess-a")

    assert len(store) == 0


def test_revoking_sweeps_entries_that_have_expired(
    store: RevocationStore, clock: FakeClock
) -> None:
    """Memory is bounded by logouts-per-lifetime: old entries go on each logout."""
    store.revoke("old-1", expires_at=clock.now + 60)
    store.revoke("old-2", expires_at=clock.now + 60)
    clock.advance(120)

    store.revoke("fresh", expires_at=clock.now + 3600)

    assert len(store) == 1
    assert store.is_revoked("fresh") is True


def test_live_entries_survive_a_sweep(store: RevocationStore, clock: FakeClock) -> None:
    store.revoke("short", expires_at=clock.now + 60)
    store.revoke("long", expires_at=clock.now + 3600)
    clock.advance(120)

    assert store.purge_expired() == 1
    assert store.is_revoked("long") is True
    assert len(store) == 1


def test_purge_expired_reports_zero_when_nothing_is_stale(
    store: RevocationStore, clock: FakeClock
) -> None:
    store.revoke("sess-a", expires_at=clock.now + 3600)

    assert store.purge_expired() == 0
    assert len(store) == 1


def test_revoking_an_already_expired_session_is_inert(
    store: RevocationStore, clock: FakeClock
) -> None:
    """The session is already dead on its own expiry check; nothing accumulates."""
    store.revoke("stale", expires_at=clock.now - 1)

    assert store.is_revoked("stale") is False
    assert store.purge_expired() == 0
    assert len(store) == 0


def test_re_revoking_keeps_the_later_expiry(store: RevocationStore, clock: FakeClock) -> None:
    """A shorter-lived cookie for the same id must not cut the record short."""
    store.revoke("sess-a", expires_at=clock.now + 3600)
    store.revoke("sess-a", expires_at=clock.now + 60)

    clock.advance(120)

    assert store.is_revoked("sess-a") is True


def test_re_revoking_extends_to_a_later_expiry(store: RevocationStore, clock: FakeClock) -> None:
    store.revoke("sess-a", expires_at=clock.now + 60)
    store.revoke("sess-a", expires_at=clock.now + 3600)

    clock.advance(120)

    assert store.is_revoked("sess-a") is True
    assert len(store) == 1


def test_clear_forgets_every_revocation(store: RevocationStore, clock: FakeClock) -> None:
    """Stands in for what a container force-recreate does to the store."""
    store.revoke("sess-a", expires_at=clock.now + 3600)
    store.revoke("sess-b", expires_at=clock.now + 3600)

    store.clear()

    assert len(store) == 0
    assert store.is_revoked("sess-a") is False


def test_many_logouts_within_one_lifetime_stay_bounded(
    store: RevocationStore, clock: FakeClock
) -> None:
    """Sessions revoked in earlier lifetimes do not accumulate across uptime."""
    lifetime = 12 * 3600
    for round_index in range(5):
        for user_index in range(20):
            store.revoke(f"sess-{round_index}-{user_index}", expires_at=clock.now + lifetime)
        clock.advance(lifetime + 1)

    store.revoke("final", expires_at=clock.now + lifetime)

    assert len(store) == 1


def test_defaults_to_wall_clock_time() -> None:
    """Expiries are absolute epoch seconds, matching the cookie's own values."""
    import time

    store = RevocationStore()
    store.revoke("sess-a", expires_at=time.time() + 3600)

    assert store.is_revoked("sess-a") is True

    store.revoke("sess-b", expires_at=time.time() - 1)

    assert store.is_revoked("sess-b") is False
