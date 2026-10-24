"""The revoked-session file moves to the state directory, once, without losing a logout.

The sidecar keeps its revocations in ``OSPREY_STATE_DIR``. A deployment that
recorded logouts before that directory existed has them in the audit
directory; the store copies that file into the state directory the first time
it is built there, never touches the original, and from then on reads only its
own file.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar import revocation
from osprey.services.auth_sidecar.revocation import (
    REVOCATION_FILE_NAME,
    STATE_DIR_ENV,
    RevocationStore,
    state_directory,
)

from .test_logout import PASSWORD_ENV, _app, _logout, _mint, _unlocked, _verify
from .test_revocation import FakeClock


@pytest.fixture(autouse=True)
def _fresh_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The once-per-process warning latch, reset so each test sees its own."""
    monkeypatch.setattr(revocation, "_warned", False)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def zones(tmp_path: Path) -> tuple[Path, Path]:
    audit = tmp_path / "audit" / "sidecar"
    state = tmp_path / "state" / "sidecar"
    audit.mkdir(parents=True)
    state.mkdir(parents=True)
    return audit, state


def test_a_session_revoked_before_the_move_is_still_revoked_after_it(
    zones: tuple[Path, Path], clock: FakeClock
) -> None:
    audit, state = zones
    RevocationStore(audit, clock=clock).revoke("sess-a", expires_at=clock.now + 3600)

    moved = RevocationStore(state, clock=clock, legacy_directory=audit)

    assert moved.is_revoked("sess-a")
    assert moved.path == state / REVOCATION_FILE_NAME
    assert (state / REVOCATION_FILE_NAME).is_file()


def test_the_carried_file_is_owner_only(zones: tuple[Path, Path], clock: FakeClock) -> None:
    audit, state = zones
    RevocationStore(audit, clock=clock).revoke("sess-a", expires_at=clock.now + 3600)

    RevocationStore(state, clock=clock, legacy_directory=audit)

    assert stat.S_IMODE((state / REVOCATION_FILE_NAME).stat().st_mode) == 0o600


def test_the_legacy_file_is_left_untouched(zones: tuple[Path, Path], clock: FakeClock) -> None:
    audit, state = zones
    RevocationStore(audit, clock=clock).revoke("sess-a", expires_at=clock.now + 3600)
    legacy = audit / REVOCATION_FILE_NAME
    before = legacy.read_bytes()

    moved = RevocationStore(state, clock=clock, legacy_directory=audit)
    moved.revoke("sess-b", expires_at=clock.now + 3600)

    assert legacy.read_bytes() == before


def test_new_logouts_go_to_the_state_dir(zones: tuple[Path, Path], clock: FakeClock) -> None:
    audit, state = zones
    RevocationStore(audit, clock=clock).revoke("sess-a", expires_at=clock.now + 3600)

    RevocationStore(state, clock=clock, legacy_directory=audit).revoke(
        "sess-b", expires_at=clock.now + 3600
    )
    restarted = RevocationStore(state, clock=clock)

    assert restarted.is_revoked("sess-a")
    assert restarted.is_revoked("sess-b")


def test_an_existing_state_file_is_never_overwritten_by_the_legacy_one(
    zones: tuple[Path, Path], clock: FakeClock
) -> None:
    """The carry-over is one-time: once the state file exists it is the only one read."""
    audit, state = zones
    RevocationStore(state, clock=clock).revoke("sess-new", expires_at=clock.now + 3600)
    RevocationStore(audit, clock=clock).revoke("sess-old", expires_at=clock.now + 3600)

    store = RevocationStore(state, clock=clock, legacy_directory=audit)

    assert store.is_revoked("sess-new")
    assert not store.is_revoked("sess-old")


def test_no_legacy_file_is_a_fresh_start(zones: tuple[Path, Path], clock: FakeClock) -> None:
    audit, state = zones

    store = RevocationStore(state, clock=clock, legacy_directory=audit)

    assert len(store) == 0
    assert not (state / REVOCATION_FILE_NAME).exists()


def test_a_failed_copy_still_honours_the_legacy_revocations(
    zones: tuple[Path, Path], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit, state = zones
    RevocationStore(audit, clock=clock).revoke("sess-a", expires_at=clock.now + 3600)

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(revocation, "atomic_replace", refuse)
    store = RevocationStore(state, clock=clock, legacy_directory=audit)

    assert store.is_revoked("sess-a")
    assert not (state / REVOCATION_FILE_NAME).exists()


def test_expired_legacy_entries_are_not_carried_into_memory(
    zones: tuple[Path, Path], clock: FakeClock
) -> None:
    audit, state = zones
    (audit / REVOCATION_FILE_NAME).write_text(
        json.dumps({"v": 1, "revoked": {"a" * 64: clock.now - 1}})
    )

    store = RevocationStore(state, clock=clock, legacy_directory=audit)

    assert len(store) == 0


@pytest.mark.parametrize("value", ["", "  ", "relative/dir"])
def test_the_state_directory_must_be_absolute(value: str) -> None:
    assert state_directory({STATE_DIR_ENV: value}) is None


def test_the_state_directory_is_read_from_the_mapping(tmp_path: Path) -> None:
    assert state_directory({STATE_DIR_ENV: str(tmp_path)}) == tmp_path
    assert state_directory({}) is None


def test_the_app_keeps_revocations_in_the_state_dir(zones: tuple[Path, Path]) -> None:
    audit, state = zones
    env = {**PASSWORD_ENV, "OSPREY_AUTH_AUDIT_DIR": str(audit), STATE_DIR_ENV: str(state)}
    cookie = _mint(_unlocked("alice"))

    with TestClient(_app(env)) as client:
        _logout(client, "alice", cookie)

    assert (state / REVOCATION_FILE_NAME).is_file()
    assert not (audit / REVOCATION_FILE_NAME).exists()
    with TestClient(_app(env)) as restarted:
        assert _verify(restarted, "alice", cookie) == 401


def test_a_logout_recorded_by_the_old_wiring_survives_the_upgrade(
    zones: tuple[Path, Path],
) -> None:
    """The file an older sidecar kept in its audit directory refuses the cookie
    after the upgrade; the same cookie with no such file is still honoured."""
    audit, state = zones
    cookie = _mint(_unlocked("alice"))
    with TestClient(_app(PASSWORD_ENV)) as old:
        # The old wiring kept the store in the audit directory.
        old.app.state.revocation_store = RevocationStore(audit)
        _logout(old, "alice", cookie)
    assert (audit / REVOCATION_FILE_NAME).is_file()

    upgraded_env = {**PASSWORD_ENV, "OSPREY_AUTH_AUDIT_DIR": str(audit), STATE_DIR_ENV: str(state)}
    with TestClient(_app(upgraded_env)) as upgraded:
        assert _verify(upgraded, "alice", cookie) == 401

    fresh_state = state.parent / "fresh"
    fresh_state.mkdir()
    control_env = {**PASSWORD_ENV, STATE_DIR_ENV: str(fresh_state)}
    with TestClient(_app(control_env)) as control:
        assert _verify(control, "alice", cookie) == 200
    assert os.listdir(fresh_state) == []
