"""The held listening socket and the supervised child behind it.

Every case runs a real child: a small Python script that receives the held
socket's descriptor, optionally answers one connection with its pid and the
generation its environment names, and exits with the status the case asks for.
"""

from __future__ import annotations

import errno
import itertools
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from osprey.interfaces.web_terminal import supervised_child
from osprey.interfaces.web_terminal.supervised_child import (
    FAILURE_GAVE_UP,
    FAILURE_REFUSED,
    PREVIOUS_EXIT_ENV,
    HeldSocket,
    SupervisedChild,
)
from tests._processes import pid_alive

BINDING_MOVED = 75
TRANSPORT_STALE = 76
START_REFUSED = 78

CHILD = textwrap.dedent(
    """
    import os, signal, socket, sys, time

    fd, mode, code = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])

    def answer():
        listener = socket.socket(fileno=fd)
        conn, _ = listener.accept()
        conn.recv(64)
        conn.sendall(f"{os.getpid()} {os.environ.get('GEN', '')}\\n".encode())
        conn.close()
        listener.detach()

    if mode == "exit":
        pass
    elif mode == "linger":
        time.sleep(float(sys.argv[4]))
    elif mode == "answer":
        answer()
    elif mode == "serve":
        while True:
            answer()
    elif mode == "deaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            answer()
    elif mode == "record":
        # One line per incarnation: the predecessor's exit status it was given.
        with open(sys.argv[4], "a") as out:
            out.write(os.environ.get("OSPREY_SIDECAR_PREVIOUS_EXIT", "-") + "\\n")
        time.sleep(float(sys.argv[5]))
    sys.exit(code)
    """
)


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _ask(port: int, timeout: float = 10.0) -> tuple[int, str]:
    """Connect to *port*, send one request, return the answering (pid, generation)."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.sendall(b"hello\n")
        conn.settimeout(timeout)
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(64)
            if not chunk:
                break
            data += chunk
    pid, _, generation = data.decode().strip().partition(" ")
    return int(pid), generation


def _bind_errno(port: int) -> int | None:
    """Try to bind *port* on loopback; return the errno, or ``None`` on success."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        return exc.errno
    finally:
        probe.close()
    return None


def _watcher_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "supervised-child-exit"]


@pytest.fixture
def held() -> Iterator[HeldSocket]:
    sock = HeldSocket()
    yield sock
    sock.close()


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "child.py"
    path.write_text(CHILD, encoding="utf-8")
    return path


class Harness:
    """Builds supervisors over one script and one held socket; stops them all."""

    def __init__(self, script: Path, held: HeldSocket) -> None:
        self.script = script
        self.held = held
        self.give_ups = 0
        self.spawn_times: list[float] = []
        self.generation = 0
        self._children: list[SupervisedChild] = []

    def argv(self, mode: str, code: int = 0, *extra: str) -> list[str]:
        return [sys.executable, str(self.script), str(self.held.fd), mode, str(code), *extra]

    def supervise(self, mode: str, code: int = 0, *extra: str, **kwargs: object) -> SupervisedChild:
        def argv_for() -> list[str]:
            self.spawn_times.append(time.monotonic())
            return self.argv(mode, code, *extra)

        def env_for() -> dict[str, str]:
            self.generation += 1
            return {**os.environ, "GEN": str(self.generation)}

        def on_give_up() -> None:
            self.give_ups += 1

        options: dict[str, object] = {
            "grace_s": 5.0,
            "respawn_now_codes": {BINDING_MOVED},
            "never_respawn_codes": {START_REFUSED},
            "on_give_up": on_give_up,
            "pass_fds": (self.held.fd,),
        }
        options.update(kwargs)
        child = SupervisedChild(argv_for, env_for, **options)  # type: ignore[arg-type]
        self._children.append(child)
        return child

    def stop_all(self) -> None:
        for child in self._children:
            child.stop()


@pytest.fixture
def harness(script: Path, held: HeldSocket) -> Iterator[Harness]:
    h = Harness(script, held)
    yield h
    h.stop_all()


class TestHeldSocket:
    def test_binds_a_loopback_port_and_listens_in_the_parent(self, held: HeldSocket) -> None:
        assert held.port > 0
        # Nothing accepts, yet the connection completes: the parent listens.
        with socket.create_connection(("127.0.0.1", held.port), timeout=5) as conn:
            assert conn.getpeername() == ("127.0.0.1", held.port)

    def test_the_port_is_taken_until_close(self) -> None:
        sock = HeldSocket()
        port = sock.port
        try:
            assert _bind_errno(port) == errno.EADDRINUSE
        finally:
            sock.close()
        assert sock.closed
        assert sock.port == port
        assert _bind_errno(port) is None

    def test_close_is_idempotent_and_retires_the_fd(self) -> None:
        sock = HeldSocket()
        sock.close()
        sock.close()
        with pytest.raises(RuntimeError):
            _ = sock.fd

    def test_context_manager_closes(self) -> None:
        with HeldSocket() as sock:
            port = sock.port
            assert not sock.closed
        assert sock.closed
        assert _bind_errno(port) is None

    def test_the_fd_is_not_inheritable(self, held: HeldSocket) -> None:
        assert os.get_inheritable(held.fd) is False

    def test_the_fd_reaches_a_child_only_through_pass_fds(self, held: HeldSocket) -> None:
        probe = (
            "import socket, sys\n"
            "try:\n"
            "    s = socket.socket(fileno=int(sys.argv[1]))\n"
            "    s.getsockname()\n"
            "except OSError:\n"
            "    sys.exit(3)\n"
            "sys.exit(0)\n"
        )
        argv = [sys.executable, "-c", probe, str(held.fd)]
        # close_fds=False is the widest inheritance Popen offers: the fd still
        # does not cross, because it is not inheritable.
        assert subprocess.run(argv, close_fds=False, check=False).returncode == 3
        assert subprocess.run(argv, check=False).returncode == 3
        assert subprocess.run(argv, pass_fds=(held.fd,), check=False).returncode == 0


class TestRespawn:
    def test_port_is_stable_over_twenty_respawns(self, harness: Harness) -> None:
        port = harness.held.port
        child = harness.supervise("answer", BINDING_MOVED, quick_exit_s=0.0, give_up_after=22)
        child.start()

        seen = [_ask(port) for _ in range(21)]

        assert harness.held.port == port
        pids = [pid for pid, _ in seen]
        assert len(set(pids)) == 21, "every request was answered by a new child"
        # The environment is rendered again for every start.
        assert [generation for _, generation in seen] == [str(n) for n in range(1, 22)]
        assert _wait_for(lambda: child.starts == 22)
        assert harness.give_ups == 0
        assert child.running

    def test_one_watcher_thread_per_incarnation(self, harness: Harness) -> None:
        before = len(_watcher_threads())
        child = harness.supervise("answer", BINDING_MOVED, quick_exit_s=0.0, give_up_after=6)
        child.start()
        for _ in range(5):
            _ask(harness.held.port)
        assert _wait_for(lambda: child.starts == 6 and child.pid is not None)
        # Five incarnations are gone and so are their threads.
        assert _wait_for(lambda: len(_watcher_threads()) == before + 1)

    def test_a_request_queued_while_no_child_lives_is_answered_by_the_next(
        self, harness: Harness
    ) -> None:
        port = harness.held.port
        child = harness.supervise("answer", 3, backoff_s=(1.0,))
        child.start()
        first, _ = _ask(port)
        assert _wait_for(lambda: not pid_alive(first) and child.pid is None)
        assert child.starts == 1, "the crash is waiting out its backoff"

        # No child is alive: the connection is completed by the parent's
        # listen() and the request waits in the backlog.
        second, generation = _ask(port)

        assert second != first
        assert generation == "2"
        assert child.exit_status == 3

    def test_respawn_now_past_the_quick_bound_skips_the_backoff(self, harness: Harness) -> None:
        # Every lifetime on the step clock is 1.5 s, past the quick-exit bound,
        # and eight exits fit well inside the window: each 75 is respawned at
        # once and counted, so the eighth ends supervision.
        ticks = itertools.count()
        child = harness.supervise(
            "exit",
            BINDING_MOVED,
            backoff_s=(30.0,),
            give_up_after=8,
            clock=lambda: next(ticks) * 1.5,
        )
        child.start()
        assert _wait_for(lambda: harness.give_ups == 1, timeout=20)
        assert child.starts == 8
        assert child.failure == FAILURE_GAVE_UP
        gaps = [b - a for a, b in itertools.pairwise(harness.spawn_times)]
        assert len(gaps) == 7
        assert max(gaps) < 10.0, "no backoff was waited"

    def test_a_crash_is_respawned_after_the_backoff(self, harness: Harness) -> None:
        child = harness.supervise("exit", 1, backoff_s=(0.3,), give_up_after=3)
        child.start()
        assert _wait_for(lambda: harness.give_ups == 1)
        assert child.starts == 3
        gaps = [b - a for a, b in itertools.pairwise(harness.spawn_times)]
        assert all(gap >= 0.3 for gap in gaps)

    def test_crashes_outside_the_window_are_forgotten(self, harness: Harness) -> None:
        child = harness.supervise(
            "exit", 1, backoff_s=(0.2,), give_up_after=2, give_up_window_s=0.1
        )
        child.start()
        # Each crash is alone in its window, so the count never reaches two.
        assert _wait_for(lambda: child.starts >= 5)
        assert harness.give_ups == 0

    def test_a_child_that_cannot_be_started_counts_as_a_crash(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        calls = 0

        def argv_for() -> list[str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return harness.argv("exit", 1)
            return [str(tmp_path / "no-such-program")]

        child = SupervisedChild(
            argv_for,
            lambda: dict(os.environ),
            grace_s=1.0,
            respawn_now_codes={BINDING_MOVED},
            never_respawn_codes={START_REFUSED},
            on_give_up=lambda: None,
            backoff_s=(0.05,),
            give_up_after=4,
        )
        child.start()
        try:
            assert _wait_for(lambda: child.failure == FAILURE_GAVE_UP)
            assert child.starts == 1
            assert calls == 4
        finally:
            child.stop()

    def test_start_raises_when_the_first_child_cannot_be_started(self, tmp_path: Path) -> None:
        child = SupervisedChild(
            lambda: [str(tmp_path / "no-such-program")],
            lambda: dict(os.environ),
            grace_s=1.0,
            respawn_now_codes=(),
            never_respawn_codes=(),
            on_give_up=None,
        )
        with pytest.raises(OSError):
            child.start()
        assert not child.running


class TestContractSchedule:
    def test_the_defaults_are_the_contract(self) -> None:
        assert supervised_child.BACKOFF_S == (0.5, 1.0, 2.0, 4.0)
        assert supervised_child.GIVE_UP_AFTER == 5
        assert supervised_child.GIVE_UP_WINDOW_S == 60.0
        assert supervised_child.QUICK_EXIT_S == 1.0

    def test_a_quick_75_is_counted_backs_off_and_gives_up_on_the_fifth(
        self, harness: Harness
    ) -> None:
        """Untuned supervisor, child that exits 75 at once.

        A 75 from a child younger than a second waits the backoff, so the
        loop is bounded: waits of 0.5, 1, 2 and 4 s, then the fifth exit ends
        it, and the port is still bound. The supervisor's clock is frozen, so
        every lifetime is zero and every exit falls in the window whatever the
        host's speed.
        """
        port = harness.held.port
        child = harness.supervise("exit", BINDING_MOVED, clock=lambda: 0.0)
        child.start()

        assert _wait_for(lambda: harness.give_ups == 1, timeout=40)

        assert child.starts == 5
        assert child.failure == FAILURE_GAVE_UP
        assert child.exit_status == BINDING_MOVED
        assert not child.running
        assert child.pid is None
        gaps = [b - a for a, b in itertools.pairwise(harness.spawn_times)]
        assert len(gaps) == 4
        for gap, wait in zip(gaps, (0.5, 1.0, 2.0, 4.0), strict=True):
            # The gap is the wait plus one short child lifetime.
            assert wait <= gap < wait + 5.0

        time.sleep(0.3)
        assert harness.give_ups == 1, "called once"
        assert child.starts == 5, "nothing is started after giving up"
        assert _bind_errno(port) == errno.EADDRINUSE
        harness.held.close()
        assert _bind_errno(port) is None

    def test_a_slow_75_is_respawned_at_once_and_still_gives_up_on_the_fifth(
        self, harness: Harness
    ) -> None:
        """Untuned window and count, child that exits 75 past the quick bound.

        The step clock makes every lifetime 1.5 s and keeps five exits inside
        60 s: each 75 skips the 30 s backoff, and the fifth still ends it.
        """
        ticks = itertools.count()
        child = harness.supervise(
            "exit", BINDING_MOVED, backoff_s=(30.0,), clock=lambda: next(ticks) * 1.5
        )
        child.start()

        # Five starts inside 20 s: no 30 s backoff was taken.
        assert _wait_for(lambda: harness.give_ups == 1, timeout=20)
        assert child.starts == 5
        assert child.failure == FAILURE_GAVE_UP


class TestStartRefused:
    def test_78_is_never_respawned_and_calls_on_give_up_once(self, harness: Harness) -> None:
        port = harness.held.port
        child = harness.supervise("exit", START_REFUSED, backoff_s=(0.05,))
        child.start()

        assert _wait_for(lambda: harness.give_ups == 1)
        time.sleep(0.5)

        assert harness.give_ups == 1
        assert child.starts == 1
        assert child.failure == FAILURE_REFUSED
        assert child.exit_status == START_REFUSED
        assert not child.running
        assert _bind_errno(port) == errno.EADDRINUSE
        harness.held.close()
        assert _bind_errno(port) is None

    def test_stderr_tail_says_why(self) -> None:
        argv = [
            sys.executable,
            "-c",
            "import sys; print('refusing: no context dir', file=sys.stderr); sys.exit(78)",
        ]
        gave_up = threading.Event()
        child = SupervisedChild(
            lambda: argv,
            lambda: dict(os.environ),
            grace_s=1.0,
            respawn_now_codes={BINDING_MOVED},
            never_respawn_codes={START_REFUSED},
            on_give_up=gave_up.set,
        )
        child.start()
        try:
            assert gave_up.wait(10)
            assert "refusing: no context dir" in child.stderr_tail
        finally:
            child.stop()

    def test_stop_may_be_called_from_on_give_up(self, harness: Harness) -> None:
        done = threading.Event()
        holder: list[SupervisedChild] = []

        def on_give_up() -> None:
            holder[0].stop()
            done.set()

        child = harness.supervise("exit", START_REFUSED, on_give_up=on_give_up)
        holder.append(child)
        child.start()
        assert done.wait(10)
        assert child.failure == FAILURE_REFUSED

    def test_start_again_after_a_refusal(self, harness: Harness) -> None:
        child = harness.supervise("exit", START_REFUSED)
        child.start()
        assert _wait_for(lambda: harness.give_ups == 1)
        child.start()
        assert _wait_for(lambda: harness.give_ups == 2)
        assert child.starts == 1


class TestStop:
    def test_stop_terminates_and_never_respawns(self, harness: Harness) -> None:
        child = harness.supervise("serve", 0, backoff_s=(0.05,))
        child.start()
        pid, _ = _ask(harness.held.port)
        assert child.pid == pid

        child.stop()

        assert not pid_alive(pid)
        assert child.pid is None
        assert not child.running
        time.sleep(0.5)
        assert child.starts == 1
        assert harness.give_ups == 0

    def test_stop_kills_a_child_that_ignores_sigterm_after_the_grace(
        self, harness: Harness
    ) -> None:
        child = harness.supervise("deaf", 0, grace_s=0.5, backoff_s=(0.05,))
        child.start()
        pid, _ = _ask(harness.held.port)  # answered only once SIGTERM is ignored

        began = time.monotonic()
        child.stop()
        took = time.monotonic() - began

        assert not pid_alive(pid)
        assert took >= 0.5, "the grace was waited before SIGKILL"
        time.sleep(0.3)
        assert child.starts == 1

    def test_stop_during_a_backoff_cancels_the_respawn(self, harness: Harness) -> None:
        child = harness.supervise("exit", 1, backoff_s=(1.0,))
        child.start()
        assert _wait_for(lambda: child.exit_status == 1 and child.pid is None)

        began = time.monotonic()
        child.stop()
        assert time.monotonic() - began < 0.9, "stop does not wait the backoff out"

        time.sleep(1.5)
        assert child.starts == 1
        assert harness.give_ups == 0
        assert not child.running

    def test_stop_is_idempotent_and_safe_before_start(self, harness: Harness) -> None:
        child = harness.supervise("serve", 0)
        child.stop()
        child.start()
        child.stop()
        child.stop()
        assert child.pid is None

    def test_start_twice_is_refused_and_restart_after_stop_works(self, harness: Harness) -> None:
        child = harness.supervise("serve", 0)
        child.start()
        with pytest.raises(RuntimeError):
            child.start()
        first, _ = _ask(harness.held.port)
        child.stop()
        child.start()
        second, generation = _ask(harness.held.port)
        assert second != first
        assert generation == "2"

    def test_stop_signals_the_childs_whole_process_group(self, harness: Harness) -> None:
        child = harness.supervise("serve", 0)
        child.start()
        pid, _ = _ask(harness.held.port)
        assert os.getpgid(pid) == pid, "the child leads its own group"
        assert os.getpgid(pid) != os.getpgid(0)
        child.stop()
        assert not pid_alive(pid)


def test_the_held_socket_outlives_a_killed_child(harness: Harness) -> None:
    """``kill -9`` on the child: same port, next child answers."""
    port = harness.held.port
    child = harness.supervise("serve", 0, backoff_s=(0.05,))
    child.start()
    first, _ = _ask(port)

    os.kill(first, signal.SIGKILL)
    # A child blocked in accept() can take the next connection in the kernel
    # and die holding it, which resets the client; ask only once the
    # supervisor has reaped it and the next child is up.
    assert _wait_for(
        lambda: child.exit_status == -signal.SIGKILL and child.pid not in (None, first)
    )

    second, _ = _ask(port)
    assert second != first
    assert child.exit_status == -signal.SIGKILL
    assert harness.held.port == port


class TestPreviousExit:
    """``_spawn`` hands each child its predecessor's exit status, and only that."""

    @staticmethod
    def _seen(path: Path) -> list[str]:
        return path.read_text().split() if path.exists() else []

    def test_a_child_replacing_a_stale_exit_sees_76_and_the_first_sees_none(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The parent's environment carries the variable: the first child must
        # still see none, because _spawn pops what env_for supplied.
        monkeypatch.setenv(PREVIOUS_EXIT_ENV, "75")
        out = tmp_path / "seen.txt"
        child = harness.supervise(
            "record",
            TRANSPORT_STALE,
            str(out),
            "0.15",
            respawn_now_codes={BINDING_MOVED, TRANSPORT_STALE},
            quick_exit_s=0.1,
            backoff_s=(30.0,),
            give_up_after=10,
        )
        child.start()
        assert _wait_for(lambda: len(self._seen(out)) >= 3, timeout=20)
        seen = self._seen(out)
        assert seen[0] == "-"
        assert set(seen[1:]) == {"76"}
        # 76 takes 75's path: respawned at once.
        assert harness.give_ups == 0
        assert child.failure is None

    def test_a_child_replacing_a_crash_sees_the_crash_status(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        out = tmp_path / "seen.txt"
        child = harness.supervise("record", 1, str(out), "0", backoff_s=(0.05,), give_up_after=3)
        child.start()
        assert _wait_for(lambda: harness.give_ups == 1)
        assert self._seen(out) == ["-", "1", "1"]

    def test_a_new_start_forgets_the_previous_run(self, harness: Harness, tmp_path: Path) -> None:
        out = tmp_path / "seen.txt"
        child = harness.supervise("record", 1, str(out), "0", backoff_s=(0.05,), give_up_after=2)
        child.start()
        assert _wait_for(lambda: harness.give_ups == 1)
        assert child.exit_status == 1
        assert self._seen(out) == ["-", "1"]

        child.start()
        assert _wait_for(lambda: len(self._seen(out)) >= 3)
        assert self._seen(out)[2] == "-", "start() begins a run with no predecessor"
