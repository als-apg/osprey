"""Tests for the auth sidecar's on-disk revocation store."""

from __future__ import annotations

import hashlib
import json
import logging
import stat
from pathlib import Path

import pytest

from osprey.services.auth_sidecar import revocation
from osprey.services.auth_sidecar.revocation import REVOCATION_FILE_NAME, RevocationStore


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


@pytest.fixture
def disk_store(tmp_path: Path, clock: FakeClock) -> RevocationStore:
    return RevocationStore(tmp_path, clock=clock)


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


def _digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _write(directory: Path, payload: object) -> Path:
    path = directory / REVOCATION_FILE_NAME
    path.write_text(json.dumps(payload))
    return path


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.levelno >= logging.WARNING and r.name == revocation.__name__
    ]


@pytest.fixture(autouse=True)
def _fresh_warning_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(revocation, "_warned", False)


class TestOnDisk:
    def test_a_revocation_survives_a_new_store_on_the_same_directory(
        self, disk_store: RevocationStore, tmp_path: Path, clock: FakeClock
    ) -> None:
        disk_store.revoke("sess-a", expires_at=clock.now + 3600)

        reborn = RevocationStore(tmp_path, clock=clock)

        assert reborn.is_revoked("sess-a") is True
        assert reborn.is_revoked("sess-b") is False

    def test_the_file_holds_the_digest_and_never_the_session_id(
        self, disk_store: RevocationStore, tmp_path: Path, clock: FakeClock
    ) -> None:
        session_id = "Zm9vYmFyLXNlc3Npb24taWQ"
        disk_store.revoke(session_id, expires_at=clock.now + 3600)

        raw = (tmp_path / REVOCATION_FILE_NAME).read_bytes()

        assert session_id.encode() not in raw
        assert _digest(session_id).encode() in raw

    def test_the_file_is_owner_only(
        self, disk_store: RevocationStore, tmp_path: Path, clock: FakeClock
    ) -> None:
        disk_store.revoke("sess-a", expires_at=clock.now + 3600)

        mode = (tmp_path / REVOCATION_FILE_NAME).stat().st_mode

        assert stat.S_IMODE(mode) == 0o600

    def test_the_file_is_versioned_and_sorted(
        self, disk_store: RevocationStore, tmp_path: Path, clock: FakeClock
    ) -> None:
        disk_store.revoke("sess-b", expires_at=clock.now + 3600)
        disk_store.revoke("sess-a", expires_at=clock.now + 7200)

        text = (tmp_path / REVOCATION_FILE_NAME).read_text()
        payload = json.loads(text)

        assert payload["v"] == 1
        assert isinstance(payload["revoked"], dict)
        assert list(payload["revoked"]) == sorted(payload["revoked"])
        assert payload["revoked"][_digest("sess-a")] == clock.now + 7200

    def test_expired_entries_are_dropped_at_load(self, tmp_path: Path, clock: FakeClock) -> None:
        _write(
            tmp_path,
            {
                "v": 1,
                "revoked": {
                    _digest("dead"): clock.now - 1,
                    _digest("edge"): clock.now,
                    _digest("live"): clock.now + 60,
                },
            },
        )

        store = RevocationStore(tmp_path, clock=clock)

        assert len(store) == 1
        assert store.is_revoked("live") is True

    def test_a_revoke_rewrites_the_file_without_expired_entries(
        self, disk_store: RevocationStore, tmp_path: Path, clock: FakeClock
    ) -> None:
        disk_store.revoke("old", expires_at=clock.now + 60)
        clock.advance(120)

        disk_store.revoke("fresh", expires_at=clock.now + 3600)

        payload = json.loads((tmp_path / REVOCATION_FILE_NAME).read_text())
        assert list(payload["revoked"]) == [_digest("fresh")]

    def test_a_missing_file_is_a_silent_fresh_start(
        self, tmp_path: Path, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)

        store = RevocationStore(tmp_path, clock=clock)

        assert len(store) == 0
        assert store.path == tmp_path / REVOCATION_FILE_NAME
        assert _warnings(caplog) == []

    @pytest.mark.parametrize(
        "content",
        [
            b"not json {",
            b"\xff\xfe\x00garbage",
            b"[1, 2, 3]",
            b'{"v": 2, "revoked": {}}',
            b'{"v": 1, "revoked": []}',
        ],
        ids=["not-json", "not-utf8", "a-list", "version-2", "revoked-not-an-object"],
    )
    def test_an_unusable_file_starts_empty_and_warns_once(
        self,
        tmp_path: Path,
        clock: FakeClock,
        caplog: pytest.LogCaptureFixture,
        content: bytes,
    ) -> None:
        (tmp_path / REVOCATION_FILE_NAME).write_bytes(content)
        caplog.set_level(logging.WARNING)

        first = RevocationStore(tmp_path, clock=clock)
        RevocationStore(tmp_path, clock=clock)

        assert len(first) == 0
        assert len(_warnings(caplog)) == 1

    def test_malformed_entries_are_skipped_and_well_formed_ones_kept(
        self, tmp_path: Path, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(
            tmp_path,
            {
                "v": 1,
                "revoked": {
                    _digest("kept"): clock.now + 60,
                    "not-hex": clock.now + 60,
                    _digest("upper").upper(): clock.now + 60,
                    _digest("string-expiry"): "soon",
                    _digest("bool-expiry"): True,
                    _digest("null-expiry"): None,
                },
            },
        )
        caplog.set_level(logging.WARNING)

        store = RevocationStore(tmp_path, clock=clock)

        assert len(store) == 1
        assert store.is_revoked("kept") is True
        assert len(_warnings(caplog)) == 1

    def test_a_missing_directory_is_not_created_and_the_revocation_holds_in_memory(
        self, tmp_path: Path, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        directory = tmp_path / "unbound"
        caplog.set_level(logging.WARNING)
        store = RevocationStore(directory, clock=clock)

        store.revoke("sess-a", expires_at=clock.now + 3600)

        assert not directory.exists()
        assert store.is_revoked("sess-a") is True
        assert len(_warnings(caplog)) == 1

    def test_a_failed_write_warns_once_across_many_logouts(
        self, tmp_path: Path, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = RevocationStore(tmp_path / "unbound", clock=clock)
        caplog.set_level(logging.WARNING)

        for index in range(5):
            store.revoke(f"sess-{index}", expires_at=clock.now + 3600)

        assert len(store) == 5
        assert len(_warnings(caplog)) == 1

    def test_a_failed_write_leaves_no_temporary_file(
        self,
        disk_store: RevocationStore,
        tmp_path: Path,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _refuse(*_args: object, **_kwargs: object) -> None:
            raise OSError("no space left on device")

        monkeypatch.setattr(revocation.os, "replace", _refuse)

        disk_store.revoke("sess-a", expires_at=clock.now + 3600)

        assert disk_store.is_revoked("sess-a") is True
        assert list(tmp_path.glob("*.tmp")) == []
        assert list(tmp_path.glob(".*.tmp")) == []

    def test_a_memory_only_store_has_no_path_and_writes_nothing(
        self,
        store: RevocationStore,
        tmp_path: Path,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        store.revoke("sess-a", expires_at=clock.now + 3600)

        assert store.path is None
        assert list(tmp_path.iterdir()) == []
