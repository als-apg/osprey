"""Shared fixtures of every test that touches a guarded run's files: paths, journals, a bound call.

Every test that binds a call or reads the reports starts from a fresh run state
(:func:`reset_run_state`), the one place the process's guarded run is kept.

A planted journal is written by the product's own writer
(:class:`~osprey.runtime.journal.DurableJournal`), so its bytes are the bytes a
killed run leaves; :func:`plant_raw` is only for content that writer never
produces.

:data:`PLANTED_REFUSED` is the one table of what another account can leave at
the lock's or the journal's name; every opener of the two files, in the
runtime, the executor and the approval hook, is held to it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import pytest
import yaml

import osprey.runtime._run_state as _run_state
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime.guarded_run import (
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
    GUARDED_RUN_DIR,
    JOURNAL_FILE_NAME,
    LOCK_FILE_NAME,
    guarded_run_dir,
    lock,
)
from osprey.runtime.journal import DurableJournal
from osprey_connectors.workspace import STATE_DIR_NAME, reset_config_cache

#: The generation this process is stamped with.
GENERATION = 4

#: The start time every planted journal's header records.
STARTED = "2026-01-01T00:00:00Z"

#: An ``approval`` block that puts every tool's call to a human.
ASKING = {"enabled": True, "default_policy": "always"}

#: A deployment on a real control system, whose own target is ``live``.
LIVE_BASELINE = {"type": "epics", "connector": {"epics": {"gateways": {}}}}

_T = TypeVar("_T")

#: What every refusal of a planted name ends with.
FILES_ONLY = "the guarded-run directory holds files only"

#: Everything but a regular file with one name that can sit at the lock's or
#: the journal's name, and the refusal it meets.
PLANTED_REFUSAL = {
    "symlink": f"is a link; {FILES_ONLY}",
    "dangling_symlink": f"is a link; {FILES_ONLY}",
    "hard_link": f"has 2 hard links; {FILES_ONLY}",
    "fifo": f"is not a regular file; {FILES_ONLY}",
    "directory": f"is not a regular file; {FILES_ONLY}",
}

#: The planted kinds, in table order.
PLANTED_REFUSED = ("symlink", "dangling_symlink", "hard_link", "fifo", "directory")

#: The refusal a link at ``guarded_run`` or at the target's directory meets.
LINKED_DIRECTORY_REFUSAL = "is a link; a guarded-run directory is never a link"


def reset_run_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this test a fresh run state and lock, put back when it ends."""
    monkeypatch.setattr(_run_state, "LOCK", threading.RLock())
    monkeypatch.setattr(_run_state, "STATE", _run_state.RunState())


@pytest.fixture
def fresh_run_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh run state for the test: no call bound, nothing held, no report filed."""
    reset_run_state(monkeypatch)


def write_config(root: Path, approval: dict[str, Any] | None = ASKING) -> None:
    """Write ``build/config.yml`` on :data:`LIVE_BASELINE` with ``approval``, and drop the cache."""
    config: dict[str, Any] = {"control_system": LIVE_BASELINE}
    if approval is not None:
        config["approval"] = approval
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()


def journal_path(target: str = "live") -> Path:
    """The journal file of ``target``'s guarded run."""
    return guarded_run_dir(target) / JOURNAL_FILE_NAME


def guarded_root(root: Path) -> Path:
    """The guarded-run directory under the project ``root``; creates nothing."""
    return root / STATE_DIR_NAME / GUARDED_RUN_DIR


def guarded_dir(root: Path, target: str) -> Path:
    """``target``'s guarded-run directory under the project ``root``, created."""
    level = guarded_root(root) / target
    level.mkdir(parents=True, exist_ok=True)
    return level


def journal_at(root: Path, target: str = "live") -> Path:
    """The journal file of ``target``'s guarded run under the project ``root``; creates nothing."""
    return guarded_root(root) / target / JOURNAL_FILE_NAME


def lock_at(root: Path, target: str = "live") -> Path:
    """The lock file of ``target``'s guarded run under the project ``root``; creates nothing."""
    return guarded_root(root) / target / LOCK_FILE_NAME


def _write_journal(
    path: Path,
    target: str,
    entries: Iterable[tuple[str, Any]],
    *,
    identity: str,
    pid: int,
    generation: int,
    started: str,
) -> Path:
    """Write at ``path`` what a killed run on ``target`` leaves, with the product's writer."""
    journal = DurableJournal(path)
    try:
        journal.start(
            target=target,
            generation=generation,
            identity=identity,
            pid=pid,
            started=started,
        )
        for address, value in entries:
            journal.record(address, value)
    finally:
        journal.close()
    return journal.path


def plant(
    values: dict[str, Any],
    *,
    target: str = "live",
    generation: int = GENERATION,
    identity: str = "alice",
) -> Path:
    """A journal a killed run on ``target`` left, holding ``values``."""
    return _write_journal(
        journal_path(target),
        target,
        values.items(),
        identity=identity,
        pid=4242,
        generation=generation,
        started=STARTED,
    )


def plant_at(
    root: Path,
    target: str,
    entries: Iterable[tuple[str, Any]],
    *,
    identity: str = "alice",
    pid: int = 4242,
    generation: int = GENERATION,
    started: str = STARTED,
) -> Path:
    """A journal a killed run on ``target`` under the project ``root`` left, holding ``entries``.

    Args:
        root: The project root whose state zone holds the guarded-run directory.
        target: The control target the run was on.
        entries: ``(address, value)`` pairs, in the order the run displaced them.
        identity: The account the run's header records.
        pid: The process id the run's header records.
        generation: The generation the run's header records.
        started: The start time the run's header records.

    Returns:
        The journal's path.
    """
    guarded_dir(root, target)
    return _write_journal(
        journal_at(root, target),
        target,
        entries,
        identity=identity,
        pid=pid,
        generation=generation,
        started=started,
    )


def plant_raw(lines: Iterable[str], *, target: str = "live") -> Path:
    """A journal on ``target`` holding exactly ``lines``, for content the writer never produces."""
    path = journal_path(target)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def plant_raw_at(root: Path, target: str, raw: bytes) -> Path:
    """A journal on ``target`` under ``root`` holding exactly ``raw``, for content the writer never produces."""
    guarded_dir(root, target)
    path = journal_at(root, target)
    path.write_bytes(raw)
    return path


def digest(path: Path) -> str:
    """The sha256 hex digest of the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approve(
    monkeypatch: pytest.MonkeyPatch,
    digest: str | None,
    target: str | None = "live",
    tool: str = "execute",
    *,
    enter_lock: bool = False,
) -> None:
    """Bind a fresh approved ``tool`` call, as the wrapper does before user code.

    ``None`` hands the sandbox no such field. With ``enter_lock`` the call's lock
    is entered once, as the wrapper does before user code runs.
    """
    reset_run_state(monkeypatch)
    if digest is not None:
        monkeypatch.setenv(ENV_APPROVED_JOURNAL_SHA256, digest)
    if target is not None:
        monkeypatch.setenv(ENV_APPROVED_TARGET, target)
    guarded_run._open_approved_call(tool)
    if enter_lock:
        with lock(None):
            pass


def plant_kind(path: Path, kind: str, victim: Path) -> None:
    """Plant one :data:`PLANTED_REFUSED` *kind* at *path*.

    A link and a hard link name *victim*, an existing file; a dangling link
    names :func:`dangling_name` of it, which does not exist. A FIFO and a
    directory name nothing.
    """
    if kind == "symlink":
        os.symlink(victim, path)
    elif kind == "dangling_symlink":
        os.symlink(dangling_name(victim), path)
    elif kind == "hard_link":
        os.link(victim, path)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        raise ValueError(f"unknown planted kind {kind!r}")


def dangling_name(victim: Path) -> Path:
    """The path a ``dangling_symlink`` planted beside *victim* names."""
    return victim.with_name(victim.name + ".absent")


def returns_within(
    call: Callable[[], _T], seconds: float = 10.0, *, release: Path | None = None
) -> _T:
    """The result of *call*, or what it raised, failing the test when it has not returned.

    *call* runs on a daemon thread. When it is still running after *seconds*,
    the other end of the FIFO *release* is opened without blocking, so the
    thread ends, and the test fails: an open of a planted FIFO has to return.
    """
    outcome: list[Any] = []

    def run() -> None:
        try:
            outcome.append((True, call()))
        except BaseException as exc:  # handed to the calling thread
            outcome.append((False, exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        if release is not None:
            try:
                os.close(os.open(release, os.O_RDWR | os.O_NONBLOCK))
            except OSError:
                pass
            thread.join(seconds)
        pytest.fail("the open blocked")
    returned, value = outcome[0]
    if not returned:
        raise value
    return value


def link_directory(level: Path) -> Path:
    """Put a link at the directory *level*, naming the real directory moved beside it.

    Returns:
        The real directory, holding whatever *level* held.
    """
    real = level.with_name(level.name + ".real")
    level.rename(real)
    os.symlink(real, level, target_is_directory=True)
    return real
