"""Shared fixtures of the run-lock tests: a deployment config, the lock's path, a held lock.

Every test that takes the lock in process starts from a fresh run state
(:func:`reset_run_state`), the one place the process's hold is kept.

:data:`PLANTED_REFUSED` is the one table of what another account can leave at
the lock's name; every opener of the lock file in the runtime is held to it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest
import yaml

import osprey.runtime._run_state as _run_state
from osprey.runtime.guarded_run import GUARDED_RUN_DIR, LOCK_FILE_NAME
from osprey_connectors.workspace import STATE_DIR_NAME, reset_config_cache

#: An ``approval`` block that puts every tool's call to a human.
ASKING = {"enabled": True, "default_policy": "always"}

#: A deployment on a real control system, whose own target is ``live``.
LIVE_BASELINE = {"type": "epics", "connector": {"epics": {"gateways": {}}}}

_T = TypeVar("_T")

#: What every refusal of a planted name ends with.
FILES_ONLY = "the guarded-run directory holds files only"

#: Everything but a regular file with one name that can sit at the lock's name,
#: and the refusal it meets.
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

#: The ``started`` a lock held by :func:`held_lock` records.
HELD_STARTED = "2026-01-01T00:00:00Z"

#: A helper process that holds the run lock at ``argv[1]`` with ``argv[2]`` in
#: it until its stdin closes, the way a live run holds it; an empty ``argv[3]``
#: writes ``argv[2]`` as given, else a holder record with the helper's own pid.
_LOCK_HOLDER = """
import fcntl, json, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o664)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
content = sys.argv[2]
if sys.argv[3]:
    content = json.dumps({"pid": os.getpid(), "started": sys.argv[3]})
os.ftruncate(fd, 0)
os.pwrite(fd, content.encode("utf-8"), 0)
print("held", flush=True)
sys.stdin.read()
"""


def reset_run_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this test a fresh run state and lock, put back when it ends."""
    monkeypatch.setattr(_run_state, "LOCK", threading.Lock())
    monkeypatch.setattr(_run_state, "STATE", _run_state.RunState())


@pytest.fixture
def fresh_run_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh run state for the test: nothing held."""
    reset_run_state(monkeypatch)


def write_config(root: Path, approval: dict[str, Any] | None = ASKING) -> None:
    """Write ``build/config.yml`` on :data:`LIVE_BASELINE` with ``approval``, and drop the cache."""
    config: dict[str, Any] = {"control_system": LIVE_BASELINE}
    if approval is not None:
        config["approval"] = approval
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()


def lock_path(target: str = "live", root: Path | None = None) -> Path:
    """The run lock file of *target* under the repo *root*, the working directory when not given."""
    base = Path.cwd() if root is None else root
    return base / STATE_DIR_NAME / GUARDED_RUN_DIR / target / LOCK_FILE_NAME


@contextlib.contextmanager
def held_lock(
    target: str = "live", content: str | None = None, *, root: Path | None = None
) -> Iterator[subprocess.Popen[str]]:
    """Hold *target*'s run lock from a child process for the body.

    The lock file records *content* as given, or, when it is ``None``, a holder
    record naming the child's pid and :data:`HELD_STARTED`. The child releases
    the lock and exits when the body ends.

    Yields:
        The child holding the lock.
    """
    path = lock_path(target, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = HELD_STARTED if content is None else ""
    child = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, str(path), content or "", started],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        yield child
    finally:
        if child.stdin is not None:
            child.stdin.close()
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        if child.stdout is not None:
            child.stdout.close()


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
            with contextlib.suppress(OSError):
                os.close(os.open(release, os.O_RDWR | os.O_NONBLOCK))
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
