"""Shared fixtures of the guarded-run tests: a deployment config, a planted journal, a bound call.

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

import contextlib
import hashlib
import os
import queue
import signal
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest
import yaml

import osprey.runtime._run_state as _run_state
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime.guarded_run import (
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
    JOURNAL_FILE_NAME,
    guarded_run_dir,
    lock,
)
from osprey.runtime.journal import DurableJournal
from osprey_connectors.workspace import reset_config_cache

#: The generation this process is stamped with.
GENERATION = 4

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
    monkeypatch.setattr(_run_state, "ENTRY", threading.Lock())
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


def plant(
    values: dict[str, Any],
    *,
    target: str = "live",
    generation: int = GENERATION,
    identity: str = "alice",
) -> Path:
    """A journal a killed run on ``target`` left, holding ``values``."""
    journal = DurableJournal(journal_path(target))
    try:
        journal.start(
            target=target,
            generation=generation,
            identity=identity,
            pid=4242,
            started="2026-01-01T00:00:00Z",
        )
        for address, value in values.items():
            journal.record(address, value)
    finally:
        journal.close()
    return journal.path


def plant_raw(lines: Iterable[str], *, target: str = "live") -> Path:
    """A journal on ``target`` holding exactly ``lines``, for content the writer never produces."""
    path = journal_path(target)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
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


#: How long a holder script may take to reach a step or answer; a hang guard,
#: never a statement about speed.
FORK_HANG_S = 60.0

#: The holder script's common part. ``_run_state.ENTRY`` is a probe that frees
#: the blocked step only when a fork is found waiting for it, so a fork that
#: waits for the bracket and one that does not are both deterministic. The
#: child reports whether it holds a descriptor of the watched file and whether
#: its entry lock is free.
_FORK_PROBE = """
import os, sys, threading, time, warnings
warnings.simplefilter("ignore", DeprecationWarning)
import osprey.runtime._run_state as _run_state
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime.journal import DurableJournal

WATCHED, HANG_S = sys.argv[1], float(sys.argv[2])
go, in_window = threading.Event(), threading.Event()


class Probe:
    def __init__(self):
        self.real = threading.Lock()

    def acquire(self, blocking=True):
        if self.real.acquire(blocking=False):
            return True
        if not blocking:
            return False
        go.set()
        return self.real.acquire()

    def release(self):
        self.real.release()

    def __enter__(self):
        self.real.acquire()
        return self

    def __exit__(self, *exc):
        self.real.release()


_run_state.ENTRY = Probe()


def same(fd, path):
    try:
        st, want = os.fstat(fd), os.stat(path)
    except OSError:
        return False
    return (st.st_dev, st.st_ino) == (want.st_dev, want.st_ino)


def blocking(real, when=lambda *args, **kwargs: True):
    def blocked(*args, **kwargs):
        if not in_window.is_set() and when(*args, **kwargs):
            in_window.set()
            assert go.wait(HANG_S)
        return real(*args, **kwargs)
    return blocked


def fork_and_report(published):
    child = os.fork()
    if child == 0:
        held = [fd for fd in range(3, 1024) if same(fd, WATCHED)]
        print("child", os.getpid(), held == [], _run_state.ENTRY.acquire(blocking=False), flush=True)
        time.sleep(600)
        os._exit(0)
    print("parent", published(), flush=True)
    go.set()
"""

_FORK_TAIL = """
threading.Thread(target=run, daemon=True).start()
assert in_window.wait(HANG_S)
forker = threading.Thread(target=fork_and_report, args=(published,))
forker.start()
forker.join(HANG_S)
time.sleep(600)
"""


@contextlib.contextmanager
def fork_in_a_window(
    repo: Path, watched: Path, *, block: str, run: str, published: str
) -> Iterator[tuple[subprocess.Popen[str], list[str], list[str]]]:
    """Fork a holder process from a second thread while one step of its run is blocked.

    The holder script runs *block* (which wraps one step in ``blocking(...)``),
    starts ``run()`` (the body *run*) on a daemon thread, waits until the step
    is reached, forks from a second thread and prints the parent's
    ``published()`` (the expression *published*) once the fork returned.

    Yields:
        ``(holder, parent, child)``: the holder process, the parent's line
        ``["parent", <published>]`` and the child's line ``["child", <pid>,
        <holds no descriptor of watched>, <entry lock free>]``. The holder and
        the child are killed on exit.
    """
    script = "\n".join(
        [
            _FORK_PROBE,
            textwrap.dedent(block),
            "def run():\n" + textwrap.indent(textwrap.dedent(run), "    "),
            f"published = lambda: {published}",
            _FORK_TAIL,
        ]
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(watched), str(FORK_HANG_S)],
        cwd=repo,
        stdout=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        assert holder.stdout is not None
        for line in holder.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    child: int | None = None
    try:
        reported: list[list[str]] = []
        for _ in range(2):
            try:
                line = lines.get(timeout=FORK_HANG_S)
            except queue.Empty:
                line = None
            if line is None:
                pytest.fail(f"the holder reported {reported} (exit {holder.poll()})")
            reported.append(line.split())
        forked, parent = sorted(reported)
        child = int(forked[1])
        yield holder, parent, forked
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=FORK_HANG_S)
        if child is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child, signal.SIGKILL)
