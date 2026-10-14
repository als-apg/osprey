"""The lock is a regular file with one name, under its own directory.

``var/guarded_run/`` and each ``<target>/`` under it are writable by every
account of the deployment, so anything can be waiting at ``run.lock``, and
either directory can have been swapped for a link. Every opener of the lock
in the runtime is held to one table here
(:data:`tests._guarded_run.PLANTED_REFUSED`): what is planted is refused with
its path named, the call returns, and nothing is read, written, truncated or
created through it. No control system is involved.
"""

from __future__ import annotations

import os
import pathlib
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime.guarded_run as guarded_run
from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    LOCK_FILE_NAME,
    GuardedRunDirError,
    guarded_run_dir,
    lock,
)
from tests._guarded_run import (
    ASKING,
    LINKED_DIRECTORY_REFUSAL,
    PLANTED_REFUSAL,
    PLANTED_REFUSED,
    dangling_name,
    link_directory,
    lock_path,
    plant_kind,
    reset_run_state,
    returns_within,
    write_config,
)

#: The two guarded-run levels another account can swap for a link.
LINKED_LEVELS = ("linked_target", "linked_guarded_run")


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, ASKING)
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    reset_run_state(monkeypatch)
    return root


def _lock_path() -> Path:
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _listing(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


class _Planted:
    """One planted name and what it names, checked untouched after a refusal."""

    def __init__(self, path: Path, kind: str, victim: Path) -> None:
        self.path = path
        self.kind = kind
        self.victim = victim
        plant_kind(path, kind, victim)
        self._bytes = victim.read_bytes()
        self._stat = victim.stat()
        self._planted = os.lstat(path)
        self._beside = {d: _listing(d) for d in {path.parent, victim.parent}}

    @property
    def refusal(self) -> str:
        return PLANTED_REFUSAL[self.kind]

    def call(self, opener: Callable[[], Any]) -> Any:
        """Run *opener*, failing the test when a planted FIFO holds it."""
        return returns_within(opener, release=self.path if self.kind == "fifo" else None)

    def refuses(self, opener: Callable[[], Any]) -> None:
        """*opener* is refused naming the planted path, and nothing was touched."""
        with pytest.raises(GuardedRunDirError) as refused:
            self.call(opener)
        assert str(refused.value) == f"{self.path} {self.refusal}"
        self.untouched()

    def untouched(self) -> None:
        """The victim, the planted name and both directories are as the planter left them."""
        after = self.victim.stat()
        assert self.victim.read_bytes() == self._bytes
        assert after.st_mode == self._stat.st_mode
        assert after.st_nlink == self._stat.st_nlink
        planted = os.lstat(self.path)
        assert (planted.st_ino, planted.st_mode) == (self._planted.st_ino, self._planted.st_mode)
        assert not dangling_name(self.victim).exists()
        for directory, names in self._beside.items():
            assert _listing(directory) == names


def _victim(tmp_path: Path, content: bytes = b"keep\n") -> Path:
    """A file outside the guarded-run directory, on the same filesystem."""
    victim = tmp_path / "victim"
    victim.write_bytes(content)
    return victim


def _link_level(level: str) -> tuple[Path, Path]:
    """Swap one guarded-run level for a link; ``(the linked level, the real target directory)``.

    The real directory holds an empty lock.
    """
    target = guarded_run_dir("live")
    (target / LOCK_FILE_NAME).touch()
    if level == "linked_target":
        return target, link_directory(target)
    real = link_directory(target.parent)
    return target.parent, real / "live"


class _Real:
    """The real directory behind a linked level, checked untouched after a refusal."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._names = _listing(directory)

    def untouched(self) -> None:
        assert (self.directory / LOCK_FILE_NAME).read_bytes() == b""
        assert _listing(self.directory) == self._names


# ---------------------------------------------------------------------------
# The table, opener by opener
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
def test_a_planted_name_refuses_the_lock(kind: str, tmp_path: Path) -> None:
    planted = _Planted(_lock_path(), kind, _victim(tmp_path))

    def take() -> None:
        with lock("live"):
            pytest.fail("a run whose lock is planted never starts")

    planted.refuses(take)


@pytest.mark.parametrize("level", LINKED_LEVELS)
def test_a_linked_directory_refuses_every_runtime_opener(level: str) -> None:
    linked, real_directory = _link_level(level)
    real = _Real(real_directory)

    def take() -> None:
        with lock("live"):
            pytest.fail("a run under a linked directory never starts")

    for opener in (lambda: guarded_run_dir("live"), take):
        with pytest.raises(GuardedRunDirError) as refused:
            returns_within(opener)
        assert str(refused.value) == f"{linked} {LINKED_DIRECTORY_REFUSAL}"

    assert linked.is_symlink()
    real.untouched()


def test_a_regular_file_is_accepted_by_every_runtime_opener() -> None:
    _lock_path().touch()

    with lock("live"):
        assert b'"pid"' in _lock_path().read_bytes()
    assert _lock_path().read_bytes() == b""


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_no_name_is_checked_before_the_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether a file was created is the open's answer, never a look at its name."""

    def refusing(real: Callable[..., bool]) -> Callable[..., bool]:
        def asked(path: Any, *args: Any, **kwargs: Any) -> bool:
            if os.path.basename(os.fspath(path)) == LOCK_FILE_NAME:
                pytest.fail(f"{path} was checked by name")
            return real(path, *args, **kwargs)

        return asked

    monkeypatch.setattr(pathlib.Path, "exists", refusing(pathlib.Path.exists))
    monkeypatch.setattr(os.path, "exists", refusing(os.path.exists))
    monkeypatch.setattr(os.path, "isdir", refusing(os.path.isdir))

    with lock("live"):
        pass

    assert _lock_path().read_bytes() == b""


def test_an_existing_lock_keeps_its_mode() -> None:
    lock_path = _lock_path()
    previous = os.umask(0o022)
    try:
        with lock("live"):
            pass
        created_mode = stat.S_IMODE(lock_path.stat().st_mode)
        os.chmod(lock_path, 0o600)
        with lock("live"):
            pass
    finally:
        os.umask(previous)

    assert created_mode == 0o664
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def _open_descriptors() -> set[str]:
    return set(os.listdir("/dev/fd"))


def test_a_released_lock_leaves_no_descriptor_open(tmp_path: Path) -> None:
    with lock("live"):
        pass  # everything a first run imports and caches is loaded
    before = _open_descriptors()

    with lock("live"):
        assert _open_descriptors() != before
    assert _open_descriptors() == before

    victim = _victim(tmp_path)
    for kind in PLANTED_REFUSED:
        path = guarded_run_dir("live") / LOCK_FILE_NAME
        path.unlink(missing_ok=True)
        plant_kind(path, kind, victim)
        with pytest.raises(GuardedRunDirError):
            with lock("live"):
                pytest.fail("a run whose lock is planted never starts")
        assert _open_descriptors() == before, kind
        if kind == "directory":
            path.rmdir()
        else:
            path.unlink()


def test_a_lock_replaced_while_it_is_opened_is_opened_once_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rename over the name between the open and the check leaves a file no name holds."""
    path = _lock_path()
    path.touch()
    real_fstat = os.fstat

    def nameless(times: int) -> list[int]:
        opened: list[int] = []

        def fstat(fd: int) -> Any:
            st = real_fstat(fd)
            opened.append(fd)
            if len(opened) > times:
                return st
            return os.stat_result((*st[:3], 0, *st[4:10]))

        monkeypatch.setattr(guarded_run.os, "fstat", fstat)
        return opened

    opened = nameless(1)
    with lock("live"):
        pass
    assert len(opened) == 2

    nameless(2)
    with pytest.raises(GuardedRunDirError) as refused:
        with lock("live"):
            pytest.fail("a lock replaced twice while it was opened is refused")
    assert str(refused.value) == f"{path} was replaced while it was opened"
