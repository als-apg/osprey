"""The run lock: one read-write run per control target.

``lock(target)`` takes the target's lock without blocking and records its
holder in the lock file; a second run on the target, from any thread or
process, is busy. A forked child never holds the lock, at any moment. No
control system is involved.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    LOCK_FILE_NAME,
    GuardedRunDirError,
    OspreyRunBusy,
    OspreyWriteRefused,
    guarded_run_dir,
    lock,
)
from tests._guarded_run import lock_path, write_config

#: Seconds a child process may take to import the runtime and try the lock.
CHILD_TIMEOUT_S = 120.0


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline that asks nobody, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    return root


def _holder(directory: Path) -> dict[str, Any] | None:
    text = (directory / LOCK_FILE_NAME).read_text(encoding="utf-8")
    return json.loads(text) if text else None


def test_osprey_runtime_reexports_the_directory_and_its_names() -> None:
    assert osprey.runtime.guarded_run_dir is guarded_run.guarded_run_dir
    assert osprey.runtime.GUARDED_RUN_DIR is guarded_run.GUARDED_RUN_DIR
    assert osprey.runtime.LOCK_FILE_NAME is guarded_run.LOCK_FILE_NAME
    assert {"guarded_run_dir", "GUARDED_RUN_DIR", "LOCK_FILE_NAME"} <= set(osprey.runtime.__all__)


def test_the_lock_records_its_holder_and_releases() -> None:
    with lock("live") as directory:
        assert directory == guarded_run_dir("live")
        holder = _holder(directory)
        assert holder is not None and holder["pid"] == os.getpid()
        assert isinstance(holder["started"], str)

    assert _holder(directory) is None, "a released lock records no holder"
    with lock("live"):
        pass


def test_a_created_lock_is_group_writable_under_umask_022() -> None:
    previous = os.umask(0o022)
    try:
        with lock("live") as directory:
            pass
    finally:
        os.umask(previous)

    assert (directory / LOCK_FILE_NAME).stat().st_mode & 0o060 == 0o060


def test_the_lock_file_is_group_writable(repo: Path) -> None:
    """Every identity of the deployment's group can take the lock."""
    assert not (repo / "var").exists(), "the run creates its directory"
    previous = os.umask(0o077)
    try:
        with lock("live") as directory:
            pass
    finally:
        os.umask(previous)

    assert stat.S_IMODE(os.stat(directory / LOCK_FILE_NAME).st_mode) == 0o664
    # A process outside the directory's group cannot set its setgid bit.
    in_group = os.geteuid() == 0 or os.stat(directory).st_gid in os.getgroups()
    assert stat.S_IMODE(os.stat(directory).st_mode) == (0o2775 if in_group else 0o775)


def test_a_second_lock_is_busy() -> None:
    with lock("live") as directory:
        with pytest.raises(OspreyRunBusy) as caught:
            with lock(None):
                pytest.fail("a busy run never starts")

    assert caught.value.root == str(directory)
    assert caught.value.pid == os.getpid()
    assert f"pid {os.getpid()}" in str(caught.value)


def test_another_process_is_busy_while_the_lock_is_held(repo: Path) -> None:
    child = textwrap.dedent(
        """
        import sys
        from osprey.runtime.guarded_run import OspreyRunBusy, lock

        try:
            with lock("live"):
                pass
        except OspreyRunBusy as exc:
            print(exc.pid, flush=True)
            sys.exit(3)
        """
    )
    env = {**os.environ}

    with lock("live"):
        held = subprocess.run(
            [sys.executable, "-c", child],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    free = subprocess.run(
        [sys.executable, "-c", child],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )

    assert held.returncode == 3, held.stderr
    assert held.stdout.strip() == str(os.getpid())
    assert free.returncode == 0, free.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_a_forked_child_neither_holds_nor_keeps_the_lock(repo: Path) -> None:
    """A child forked under the hold drops the lock; killing the holder frees it at once."""
    child = textwrap.dedent(
        """
        import os, time
        import osprey.runtime._run_state as _run_state
        from osprey.runtime.guarded_run import lock

        with lock("live"):
            if os.fork() == 0:
                print(os.getpid(), _run_state.STATE.hold is None, flush=True)
                time.sleep(60)
                os._exit(0)
            time.sleep(60)
        """
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", child], cwd=repo, stdout=subprocess.PIPE, text=True
    )
    grandchild: int | None = None
    try:
        assert holder.stdout is not None
        line = holder.stdout.readline().split()
        assert len(line) == 2, f"the forked child reported nothing (exit {holder.poll()})"
        grandchild = int(line[0])
        assert line[1] == "True", "a forked child sees the target as held"
        holder.kill()
        holder.wait(timeout=CHILD_TIMEOUT_S)

        with lock("live"):
            pass
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=CHILD_TIMEOUT_S)
        if grandchild is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)


def test_a_readonly_run_raises_before_the_lock(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with pytest.raises(OspreyWriteRefused, match="readonly execution mode"):
        with lock("live"):
            pytest.fail("a readonly run never takes the lock")

    assert not (repo / "var").exists()


def test_no_guarded_run_directory_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(bare)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    with pytest.raises(GuardedRunDirError, match="guarded runs need var/guarded_run"):
        with lock("live"):
            pytest.fail("a run with no directory never starts")


#: The refusal a link planted in the guarded-run directory meets.
LINK_REFUSED = "is a link; the guarded-run directory holds files only"


def test_a_link_planted_as_the_lock_is_refused(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep\n")
    path = lock_path()
    path.parent.mkdir(parents=True)
    os.symlink(victim, path)

    with pytest.raises(GuardedRunDirError, match=LINK_REFUSED) as refused:
        with lock("live"):
            pytest.fail("a run whose lock is a link never starts")

    assert str(path) in str(refused.value)
    assert victim.read_bytes() == b"keep\n"


def test_a_join_inside_the_lock_runs_under_it_and_releases_nothing() -> None:
    with lock("live") as directory:
        hold = osprey.runtime._run_state.STATE.hold
        with lock(None, join=True) as joined:
            assert joined == directory
            assert osprey.runtime._run_state.STATE.hold is hold
        assert osprey.runtime._run_state.STATE.hold is hold, "a join releases nothing"
        assert _holder(directory) == {"pid": os.getpid(), "started": hold.started}

    assert osprey.runtime._run_state.STATE.hold is None
    with lock(None, join=True):
        assert osprey.runtime._run_state.STATE.hold is not None, "a join with no hold takes one"
    assert osprey.runtime._run_state.STATE.hold is None


def test_a_join_on_another_target_is_refused() -> None:
    with lock("live"):
        with pytest.raises(
            RuntimeError, match="holds the run lock on live; it cannot take one on va"
        ):
            with lock("va", join=True):
                pytest.fail("a join on another target never runs")


def test_the_target_directory_holds_only_the_lock_after_a_run() -> None:
    with lock("live") as directory:
        pass

    assert sorted(path.name for path in directory.iterdir()) == [LOCK_FILE_NAME]


#: The script that holds ``live``'s lock and forks from a second thread while
#: one step of the lock's entry or release is blocked. ``_run_state.LOCK`` is a
#: probe lock: the blocked step goes on once the fork is found waiting on that
#: lock, or once the fork has returned without waiting, so neither outcome
#: rests on a sleep. The parent prints whether the hold was published at the
#: fork; the child prints whether it holds no descriptor of ``run.lock`` and
#: whether its run-state lock is free, then stays alive.
_FORK_DURING = textwrap.dedent(
    """
    import fcntl, os, sys, threading, time, warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import osprey.runtime._run_state as _run_state
    import osprey.runtime.guarded_run as guarded_run
    from osprey.runtime.guarded_run import lock

    STEP = sys.argv[1]
    go = threading.Event()
    blocked = threading.Event()
    stay = threading.Event()
    published = []


    class ProbeLock:
        def __init__(self):
            self._lock = threading.Lock()

        def acquire(self, blocking=True, timeout=-1):
            if self._lock.acquire(blocking=False):
                return True
            if not blocking:
                return False
            go.set()
            return self._lock.acquire(blocking, timeout)

        def release(self):
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_exc):
            self.release()


    def note_published():
        published.append(_run_state.STATE.hold is not None)


    # Registered before the runtime's hooks, so it runs after them at a fork.
    os.register_at_fork(before=note_published)
    _run_state.LOCK = ProbeLock()
    lock_fd = []


    def wait_here():
        if not blocked.is_set():
            blocked.set()
            go.wait()


    if STEP == "write_holder":
        write_holder = guarded_run._write_holder

        def blocking_write_holder(fd):
            wait_here()
            return write_holder(fd)

        guarded_run._write_holder = blocking_write_holder
    elif STEP == "flock":
        flock = fcntl.flock

        def blocking_flock(fd, operation):
            if operation & fcntl.LOCK_EX:
                wait_here()
            return flock(fd, operation)

        fcntl.flock = blocking_flock
    else:
        close = os.close

        def blocking_close(fd):
            if lock_fd and fd == lock_fd[0]:
                wait_here()
            return close(fd)

        os.close = blocking_close


    def descriptor_free(path):
        target = os.stat(path)
        for fd in range(os.sysconf("SC_OPEN_MAX") if hasattr(os, "sysconf") else 1024):
            try:
                st = os.fstat(fd)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
                return False
        return True


    def forker():
        blocked.wait()
        pid = os.fork()
        if pid == 0:
            path = guarded_run.guarded_run_dir("live") / guarded_run.LOCK_FILE_NAME
            free = _run_state.LOCK.acquire(blocking=False)
            print("child", descriptor_free(path), free, flush=True)
            time.sleep(600)
            os._exit(0)
        go.set()
        print("parent", published[0] if published else None, os.getpid(), pid, flush=True)


    def locker():
        with lock("live"):
            lock_fd.append(_run_state.STATE.hold.fd)
            if STEP != "close":
                stay.wait()


    thread = threading.Thread(target=forker)
    thread.start()
    held = threading.Thread(target=locker, daemon=True)
    held.start()
    thread.join()
    sys.stdin.read()
    """
)


def _fork_during(repo: Path, step: str) -> tuple[list[str], list[str]]:
    """Run :data:`_FORK_DURING` blocking *step*; kill its holder and return what both printed.

    The forked child is left alive for :func:`forked_children` to kill.
    """
    holder = subprocess.Popen(
        [sys.executable, "-c", _FORK_DURING, step],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    lines: dict[str, list[str]] = {}
    try:
        assert holder.stdout is not None
        while len(lines) < 2:
            line = holder.stdout.readline().split()
            assert line, f"the script ended early (exit {holder.poll()})"
            lines[line[0]] = line[1:]
        child_pid = int(lines["parent"][2])
        holder.kill()
        holder.wait(timeout=CHILD_TIMEOUT_S)
        _FORKED.append(child_pid)
        os.kill(child_pid, 0)  # the child outlives its parent
        return lines["parent"], lines["child"]
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=CHILD_TIMEOUT_S)


#: Forked children the fork tests leave alive until they end.
_FORKED: list[int] = []


@pytest.fixture
def forked_children():
    """Kill every child :func:`_fork_during` left alive."""
    yield
    while _FORKED:
        with contextlib.suppress(ProcessLookupError):
            os.kill(_FORKED.pop(), signal.SIGKILL)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
@pytest.mark.usefixtures("forked_children")
def test_a_fork_during_lock_entry_waits_for_the_hold(repo: Path) -> None:
    parent, child = _fork_during(repo, "write_holder")

    assert parent[0] == "True", "the fork waited until the hold was published"
    assert child == ["True", "True"], "the child holds no descriptor of the lock"
    with lock("live"):
        pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
@pytest.mark.usefixtures("forked_children")
def test_a_fork_between_the_open_and_the_flock_waits_too(repo: Path) -> None:
    parent, child = _fork_during(repo, "flock")

    assert parent[0] == "True", "the fork waited until the hold was published"
    assert child == ["True", "True"], "the child holds no descriptor of the lock"
    with lock("live"):
        pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
@pytest.mark.usefixtures("forked_children")
def test_a_fork_after_release_holds_nothing(repo: Path) -> None:
    _parent, child = _fork_during(repo, "close")

    assert child == ["True", "True"], "the child holds no descriptor of the lock"
    with lock("live"):
        pass
