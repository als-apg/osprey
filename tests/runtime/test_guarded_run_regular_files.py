"""The lock and the journal are regular files with one name, under their own directory.

``var/guarded_run/`` and each ``<target>/`` under it are writable by every
account of the deployment, so anything can be waiting at ``run.lock`` or
``run.journal``, and either directory can have been swapped for a link. Every
opener of the two files is held to one table here
(:data:`tests.runtime._guarded_run_fixtures.PLANTED_REFUSED`): what is planted
is refused with its path named, the call returns, and nothing is read, written,
truncated or created through it. Channels are a dict; no control system is
involved.
"""

from __future__ import annotations

import os
import pathlib
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
import osprey.runtime.journal as journal_module
from osprey.errors import ChannelLimitsViolationError
from osprey.mcp_server.python_executor import executor
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import (
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
    GUARDED_RUN_DIR,
    JOURNAL_FILE_NAME,
    LOCK_FILE_NAME,
    GuardedRunDirError,
    guarded_run_dir,
    journaled_run,
    lock,
    read_run_journal,
)
from osprey.runtime.journal import (
    DurableJournal,
    guarded_write,
    read_journal_bytes,
    read_pending_journal,
)
from tests.runtime._guarded_run_fixtures import (
    ASKING,
    GENERATION,
    LINKED_DIRECTORY_REFUSAL,
    PLANTED_REFUSAL,
    PLANTED_REFUSED,
    dangling_name,
    journal_path,
    link_directory,
    plant,
    plant_kind,
    reset_run_state,
    returns_within,
    write_config,
)

#: The two guarded-run levels another account can swap for a link.
LINKED_LEVELS = ("linked_target", "linked_guarded_run")


class _Channels:
    """A dict of channel values; every write is recorded and lands."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, Any]] = []

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        return [self.values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.writes.append((address, value))
        self.values[address] = value

    def channel_limits(self, _address: str) -> None:
        return None


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, ASKING)
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, str(GENERATION))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(ENV_APPROVED_JOURNAL_SHA256, raising=False)
    monkeypatch.delenv(ENV_APPROVED_TARGET, raising=False)
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    reset_run_state(monkeypatch)
    return root


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> _Channels:
    """``Q`` at 3.0 and ``S`` at 5.0, both writable."""
    fake = _Channels({"Q": 3.0, "S": 5.0})
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


def _lock_path() -> Path:
    return guarded_run_dir("live") / LOCK_FILE_NAME


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


def _journal_victim(tmp_path: Path) -> Path:
    """A well-formed journal holding ``Q``, written by the product and moved aside."""
    victim = tmp_path / "victim"
    plant({"Q": 0.0}).rename(victim)
    return victim


def _link_level(level: str) -> tuple[Path, Path]:
    """Swap one guarded-run level for a link; ``(the linked level, the real target directory)``.

    The real directory holds a well-formed journal and an empty lock.
    """
    target = guarded_run_dir("live")
    plant({"Q": 0.0})
    (target / LOCK_FILE_NAME).touch()
    if level == "linked_target":
        return target, link_directory(target)
    real = link_directory(target.parent)
    return target.parent, real / "live"


class _Real:
    """The real directory behind a linked level, checked untouched after a refusal."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._journal = (directory / JOURNAL_FILE_NAME).read_bytes()
        self._names = _listing(directory)

    def untouched(self) -> None:
        assert (self.directory / JOURNAL_FILE_NAME).read_bytes() == self._journal
        assert (self.directory / LOCK_FILE_NAME).read_bytes() == b""
        assert _listing(self.directory) == self._names


# ---------------------------------------------------------------------------
# The table, opener by opener
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
@pytest.mark.parametrize("relative", [False, True], ids=["by-name", "under-a-directory"])
def test_a_planted_name_refuses_the_journal_open(kind: str, relative: bool, tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    planted = _Planted(folder / JOURNAL_FILE_NAME, kind, _victim(tmp_path))
    dir_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY) if relative else None
    try:
        planted.refuses(lambda: DurableJournal(planted.path, dir_fd=dir_fd))
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
def test_a_planted_name_refuses_the_lock(kind: str, tmp_path: Path, channels: _Channels) -> None:
    planted = _Planted(_lock_path(), kind, _victim(tmp_path))

    def take() -> None:
        with lock("live"):
            pytest.fail("a run whose lock is planted never starts")

    planted.refuses(take)

    assert channels.writes == []
    assert not journal_path().exists()


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
@pytest.mark.parametrize("enter", [lock, journaled_run], ids=["lock", "journaled-run"])
def test_a_planted_name_refuses_the_restore_read(
    kind: str, enter: Callable[..., Any], tmp_path: Path, channels: _Channels
) -> None:
    victim = _journal_victim(tmp_path)
    with lock("live"):
        pass  # the lock file is there, so a refused run adds nothing beside the journal
    planted = _Planted(journal_path(), kind, victim)

    def start() -> None:
        with enter("live"):
            pytest.fail("a run whose journal is planted never starts")

    planted.refuses(start)

    assert channels.writes == []
    assert _lock_path().read_bytes() == b""


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
@pytest.mark.parametrize("read", [read_pending_journal, read_journal_bytes])
def test_a_planted_name_refuses_the_pending_read(
    kind: str, read: Callable[[Path], Any], tmp_path: Path
) -> None:
    planted = _Planted(journal_path(), kind, _journal_victim(tmp_path))

    planted.refuses(lambda: read(planted.path))


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
def test_a_planted_name_is_unreadable_to_the_executor(kind: str, tmp_path: Path) -> None:
    planted = _Planted(journal_path(), kind, _journal_victim(tmp_path))

    observed = planted.call(lambda: executor._observe_journal("live"))

    assert observed == {"journal": str(planted.path), "state": "unreadable", "values": {}}
    planted.untouched()


@pytest.mark.parametrize("level", LINKED_LEVELS)
def test_a_linked_directory_refuses_every_runtime_opener(
    level: str, repo: Path, channels: _Channels
) -> None:
    linked, real_directory = _link_level(level)
    real = _Real(real_directory)
    directory = repo / "var" / GUARDED_RUN_DIR / "live"

    def take() -> None:
        with lock("live"):
            pytest.fail("a run under a linked directory never starts")

    def start() -> None:
        with journaled_run("live"):
            pytest.fail("a run under a linked directory never starts")

    for opener in (
        lambda: guarded_run_dir("live"),
        take,
        start,
        lambda: read_run_journal(directory),
    ):
        with pytest.raises(GuardedRunDirError) as refused:
            returns_within(opener)
        assert str(refused.value) == f"{linked} {LINKED_DIRECTORY_REFUSAL}"
    observed = executor._observe_journal("live")

    assert observed["state"] == "unreadable" and observed["values"] == {}
    assert channels.writes == []
    assert linked.is_symlink()
    real.untouched()


def test_a_regular_file_is_accepted_by_every_runtime_opener() -> None:
    path = plant({"Q": 0.0})
    planted_bytes = path.read_bytes()

    assert read_journal_bytes(path) == planted_bytes
    assert read_run_journal(path.parent) == planted_bytes
    pending = read_pending_journal(path)
    assert pending is not None and pending.values == {"Q": 0.0}
    observed = executor._observe_journal("live")
    assert observed["state"] == "entries" and observed["values"] == {"Q": 0.0}

    durable = DurableJournal(path)
    try:
        assert durable.read() == planted_bytes
        durable.record("S", 1.0)
        assert durable.read() == path.read_bytes() != planted_bytes
        durable.clear()
    finally:
        durable.close()
    assert path.read_bytes() == b""

    with lock("live"):
        assert b'"pid"' in _lock_path().read_bytes()
    assert _lock_path().read_bytes() == b""


def test_a_missing_directory_or_journal_reads_as_none(repo: Path) -> None:
    directory = repo / "var" / GUARDED_RUN_DIR / "live"

    assert read_run_journal(directory) is None
    assert not (repo / "var").exists()

    guarded_run_dir("live")

    assert read_run_journal(directory) is None
    assert read_journal_bytes(directory / JOURNAL_FILE_NAME) is None
    assert _listing(directory) == []


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_no_name_is_checked_before_the_open(
    tmp_path: Path, channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether a file was created is the open's answer, never a look at its name."""
    guarded = (LOCK_FILE_NAME, JOURNAL_FILE_NAME)

    def refusing(real: Callable[..., bool]) -> Callable[..., bool]:
        def asked(path: Any, *args: Any, **kwargs: Any) -> bool:
            if os.path.basename(os.fspath(path)) in guarded:
                pytest.fail(f"{path} was checked by name")
            return real(path, *args, **kwargs)

        return asked

    monkeypatch.setattr(pathlib.Path, "exists", refusing(pathlib.Path.exists))
    monkeypatch.setattr(os.path, "exists", refusing(os.path.exists))
    monkeypatch.setattr(os.path, "isdir", refusing(os.path.isdir))

    with lock("live"):
        pass
    with journaled_run("live"):
        guarded_write(["Q"], lambda: osprey.runtime.write_channel("Q", 4.0), lambda e: e)
    folder = tmp_path / "folder"
    folder.mkdir()
    DurableJournal(folder / JOURNAL_FILE_NAME).close()

    assert channels.writes == [("Q", 4.0)]


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
    for name in (LOCK_FILE_NAME, JOURNAL_FILE_NAME):
        for kind in PLANTED_REFUSED:
            path = guarded_run_dir("live") / name
            path.unlink(missing_ok=True)
            plant_kind(path, kind, victim)
            with pytest.raises(GuardedRunDirError):
                with lock("live"):
                    pytest.fail("a run whose lock or journal is planted never starts")
            assert _open_descriptors() == before, (name, kind)
            if kind == "directory":
                path.rmdir()
            else:
                path.unlink()


def test_an_incomplete_restore_rewrites_under_the_held_directory(
    tmp_path: Path, channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = guarded_run_dir("live")
    aside = directory.with_name("live.aside")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def refuse_q(address: str, value: Any, **kwargs: Any) -> None:
        if address == "Q":
            raise ChannelLimitsViolationError(address, value, "MAX_VALUE", "outside the band")
        channels.write_channel(address, value, **kwargs)

    with pytest.raises(RuntimeError, match="the run's own error"):
        with journaled_run("live"):
            for address in ("Q", "S"):
                guarded_write(
                    [address],
                    lambda address=address: osprey.runtime.write_channel(address, 9.0),
                    lambda e: e,
                )
            monkeypatch.setattr(osprey.runtime, "write_channel", refuse_q)
            directory.rename(aside)
            os.symlink(elsewhere, directory, target_is_directory=True)
            raise RuntimeError("the run's own error")

    pending = read_pending_journal(aside / JOURNAL_FILE_NAME)
    assert pending is not None and pending.values == {"Q": 3.0}
    assert _listing(aside) == sorted([JOURNAL_FILE_NAME, LOCK_FILE_NAME])
    assert _listing(elsewhere) == []
    assert channels.values == {"Q": 9.0, "S": 5.0}


def test_a_journal_replaced_while_it_is_opened_is_read_once_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rename over the name between the open and the check leaves a file no name holds."""
    path = plant({"Q": 0.0})
    real_fstat = os.fstat

    def nameless(times: int) -> list[int]:
        opened: list[int] = []

        def fstat(fd: int) -> Any:
            st = real_fstat(fd)
            opened.append(fd)
            if len(opened) > times:
                return st
            return os.stat_result((*st[:3], 0, *st[4:10]))

        monkeypatch.setattr(journal_module.os, "fstat", fstat)
        return opened

    opened = nameless(1)
    assert read_journal_bytes(path) == path.read_bytes()
    assert len(opened) == 2

    nameless(2)
    with pytest.raises(GuardedRunDirError) as refused:
        read_journal_bytes(path)
    assert str(refused.value) == f"{path} was replaced while it was opened"
