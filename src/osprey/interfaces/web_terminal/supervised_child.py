"""A child process kept alive behind a listening socket its parent holds.

Two pieces, usable separately:

:class:`HeldSocket` binds a loopback port through the one serving bind and
holds it listening in the parent. Because the kernel completes connections on a
listening socket whether or not anything is accepting, a request that arrives
while no child is alive waits in the backlog and is answered by the next child;
the port never changes and never goes away between children.

:class:`SupervisedChild` starts a process, watches it, and decides what an exit
means:

==========  ================================  ====================================
Exit        Meaning                           Action
==========  ================================  ====================================
respawn-now the child asked to be replaced    respawn at once, counted
never       the child refused to start        no respawn; ``on_give_up`` is called
any other   crash                             respawn after 0.5, 1, 2, 4 s
==========  ================================  ====================================

Every exit that is not never-respawn counts: five of them within 60 s give up,
whatever their status. A respawn-now exit from a child that lived under one
second waits the backoff like a crash, which keeps a child that asks to be
replaced in a loop from spending its five exits at once.

The argv and the environment are produced by callables and evaluated again for
every start, so a replacement child sees the world as it is when it starts
rather than as it was at the first launch. One variable is always the
supervisor's own: a replacement child finds its predecessor's exit status in
``OSPREY_SIDECAR_PREVIOUS_EXIT`` (the first child of a start has none, and the
variable is removed even when the environment callable supplies it).

POSIX only: descriptor passing and process-group signals.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import IO

logger = logging.getLogger(__name__)

#: Waits before respawning after the first, second, third and fourth crash.
BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

#: This many exits inside :data:`GIVE_UP_WINDOW_S` end the supervision.
GIVE_UP_AFTER = 5

#: The window the exit count is taken over, in seconds.
GIVE_UP_WINDOW_S = 60.0

#: A respawn-now exit from a child younger than this waits the backoff.
QUICK_EXIT_S = 1.0

#: Why supervision ended on its own: the child exited with a never-respawn code.
FAILURE_REFUSED = "refused"

#: Why supervision ended on its own: too many exits in the window.
FAILURE_GAVE_UP = "gave_up"

#: The environment variable a replacement child finds its predecessor's exit
#: status in (a negative number is the signal that ended it).
PREVIOUS_EXIT_ENV = "OSPREY_SIDECAR_PREVIOUS_EXIT"

_LOOPBACK = "127.0.0.1"
_TAIL_LINES = 20


class HeldSocket:
    """A loopback TCP socket, bound and listening, owned by the parent.

    The port is chosen by the kernel. The descriptor is not inheritable: a
    child gets it only when it is named in ``pass_fds``, so no unrelated
    subprocess of the parent can accept on it.

    The socket stays bound until :meth:`close`, whatever happens to the
    children that accept on it. A child that received the descriptor holds its
    own duplicate, so the port is released once this socket is closed *and* no
    such child is alive.
    """

    def __init__(self) -> None:
        from osprey.interfaces._serving import listening_socket

        sock = listening_socket(_LOOPBACK, 0)
        try:
            sock.set_inheritable(False)
        except BaseException:
            sock.close()
            raise
        self._sock: socket.socket | None = sock
        self._port: int = sock.getsockname()[1]
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        """The bound port. Still readable after :meth:`close`."""
        return self._port

    @property
    def fd(self) -> int:
        """The listening descriptor, for ``pass_fds`` and the child's argv.

        Raises:
            RuntimeError: the socket was closed.
        """
        with self._lock:
            if self._sock is None:
                raise RuntimeError("the held socket is closed")
            return self._sock.fileno()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._sock is None

    def close(self) -> None:
        """Release the port. Idempotent."""
        with self._lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()

    def __enter__(self) -> HeldSocket:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _StderrTail(threading.Thread):
    """Drain a child's stderr, keeping its last lines."""

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(daemon=True, name="supervised-child-stderr")
        self._stream = stream
        self._lines: collections.deque[str] = collections.deque(maxlen=_TAIL_LINES)
        self._lines_lock = threading.Lock()

    def run(self) -> None:
        try:
            for raw in self._stream:
                line = raw.decode("utf-8", errors="replace").rstrip()
                with self._lines_lock:
                    self._lines.append(line)
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                self._stream.close()

    @property
    def text(self) -> str:
        with self._lines_lock:
            return "\n".join(self._lines)


class SupervisedChild:
    """Run a child process and respawn it according to how it exits.

    Args:
        argv_for: Returns the child's argv. Called for every (re)start.
        env_for: Returns the child's complete environment. Called for every
            (re)start, so a replacement is launched with a freshly rendered
            environment. Both callables run under the supervisor's lock and
            must not call back into this supervisor (its properties,
            :meth:`stop`), or they deadlock it.
        grace_s: How long :meth:`stop` waits after SIGTERM before SIGKILL.
        respawn_now_codes: Exit statuses that mean "replace me at once".
            Respawned without a wait, unless the child lived under
            *quick_exit_s*; counted toward giving up like every exit.
        never_respawn_codes: Exit statuses after which the child is not
            started again.
        on_give_up: Called once, from the watcher thread, when supervision
            ends on its own (a never-respawn exit, or too many exits).
            :attr:`failure`, :attr:`exit_status` and :attr:`stderr_tail`
            describe why. Not called for :meth:`stop`.
        pass_fds: Descriptors the child inherits, typically a
            :attr:`HeldSocket.fd`. Nothing else of the parent's is inherited.
        name: Used in log lines and thread names.
        backoff_s: Waits before the respawn that follows the first, second,
            ... exit in the window; the last entry repeats. A respawn-now exit
            past *quick_exit_s* waits none.
        give_up_after: Exits other than never-respawn within
            *give_up_window_s* that end supervision.
        give_up_window_s: The window the count is taken over.
        quick_exit_s: A respawn-now exit from a younger child waits the
            backoff.
        clock: The time source for lifetimes and the give-up window.
    """

    def __init__(
        self,
        argv_for: Callable[[], Sequence[str]],
        env_for: Callable[[], Mapping[str, str]],
        *,
        grace_s: float,
        respawn_now_codes: Collection[int],
        never_respawn_codes: Collection[int],
        on_give_up: Callable[[], None] | None,
        pass_fds: Collection[int] = (),
        name: str = "supervised child",
        backoff_s: Sequence[float] = BACKOFF_S,
        give_up_after: int = GIVE_UP_AFTER,
        give_up_window_s: float = GIVE_UP_WINDOW_S,
        quick_exit_s: float = QUICK_EXIT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not backoff_s:
            raise ValueError("backoff_s needs at least one entry")
        self._argv_for = argv_for
        self._env_for = env_for
        self._grace_s = grace_s
        self._respawn_now = frozenset(respawn_now_codes)
        self._never_respawn = frozenset(never_respawn_codes)
        self._on_give_up = on_give_up
        self._pass_fds = tuple(pass_fds)
        self._name = name
        self._backoff_s = tuple(backoff_s)
        self._give_up_after = give_up_after
        self._give_up_window_s = give_up_window_s
        self._quick_exit_s = quick_exit_s
        self._clock = clock

        # Guards every field below, and makes "has stop() run?" and "spawn"
        # one step: a start that takes the lock after stop() finds the stop
        # event set and does nothing.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = False
        self._process: subprocess.Popen[bytes] | None = None
        self._watcher: threading.Thread | None = None
        self._tail: _StderrTail | None = None
        self._crashes: list[float] = []
        self._starts = 0
        self._exit_status: int | None = None
        self._failure: str | None = None

    # What the supervisor knows about its child

    @property
    def pid(self) -> int | None:
        """The live child's pid, or ``None`` between children or when stopped."""
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return None
        return process.pid

    @property
    def running(self) -> bool:
        """Whether supervision is on: a child is alive or a respawn is pending."""
        with self._lock:
            return self._active

    @property
    def starts(self) -> int:
        """How many children have been started since :meth:`start`."""
        with self._lock:
            return self._starts

    @property
    def exit_status(self) -> int | None:
        """The exit status of the most recent child that exited on its own."""
        with self._lock:
            return self._exit_status

    @property
    def failure(self) -> str | None:
        """:data:`FAILURE_REFUSED`, :data:`FAILURE_GAVE_UP`, or ``None``."""
        with self._lock:
            return self._failure

    @property
    def stderr_tail(self) -> str:
        """The last stderr lines of the most recent child."""
        with self._lock:
            tail = self._tail
        return tail.text if tail is not None else ""

    # Starting and stopping the child

    def start(self) -> None:
        """Start the first child. Blocks for the ``Popen`` only.

        May be called again after :meth:`stop`, or after supervision ended on
        its own; the crash history starts empty.

        Raises:
            RuntimeError: supervision is already on.
            OSError: the child could not be started.
        """
        with self._lock:
            if self._active:
                raise RuntimeError(f"{self._name} is already supervised")
            stop = threading.Event()
            self._stop = stop
            self._active = True
            self._crashes = []
            self._starts = 0
            self._exit_status = None
            self._failure = None
        try:
            self._spawn(stop)
        except BaseException:
            with self._lock:
                if self._stop is stop:
                    self._active = False
            raise

    def stop(self) -> None:
        """SIGTERM the child, wait ``grace_s``, then SIGKILL. Idempotent.

        After this returns no child is alive and none will be started, also
        when the call lands while a respawn is waiting out its backoff.
        """
        with self._lock:
            stop = self._stop
            stop.set()
            self._active = False
            process, self._process = self._process, None
            watcher, self._watcher = self._watcher, None
            tail = self._tail
        if process is not None:
            if process.poll() is None:
                _signal_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(self._grace_s)
                except subprocess.TimeoutExpired:
                    _signal_group(process.pid, signal.SIGKILL)
                    process.wait()
            logger.info("%s stopped (pid %s)", self._name, process.pid)
        current = threading.current_thread()
        if watcher is not None and watcher is not current:
            watcher.join(timeout=1)
        if tail is not None and tail is not current:
            tail.join(timeout=1)

    # How a child is started, watched and replaced

    def _spawn(self, stop: threading.Event) -> bool:
        """Start one child and its watcher, unless *stop* is set.

        Returns:
            Whether a child was started.
        """
        with self._lock:
            if stop.is_set():
                return False
            argv = list(self._argv_for())
            env = dict(self._env_for())
            if self._exit_status is not None:
                env[PREVIOUS_EXIT_ENV] = str(self._exit_status)
            else:
                env.pop(PREVIOUS_EXIT_ENV, None)
            process = subprocess.Popen(
                argv,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=self._pass_fds,
                start_new_session=True,
            )
            started = self._clock()
            assert process.stderr is not None
            tail = _StderrTail(process.stderr)
            tail.start()
            watcher = threading.Thread(
                target=self._watch,
                args=(process, tail, started, stop),
                daemon=True,
                name="supervised-child-exit",
            )
            self._process = process
            self._tail = tail
            self._watcher = watcher
            self._starts += 1
            watcher.start()
        logger.info("%s launched (pid %s)", self._name, process.pid)
        return True

    def _watch(
        self,
        process: subprocess.Popen[bytes],
        tail: _StderrTail,
        started: float,
        stop: threading.Event,
    ) -> None:
        """Reap one child, then decide what follows it.

        This thread belongs to *process* alone. When it respawns, the new
        child gets a thread of its own and this one ends.
        """
        status: int | None = process.wait()
        lifetime = self._clock() - started
        if stop.is_set():
            return
        tail.join(timeout=1)
        with self._lock:
            if stop.is_set():
                return
            if self._process is process:
                self._process = None
            self._exit_status = status
        text = tail.text
        logger.warning(
            "%s exited (rc=%s) after %.1f s%s",
            self._name,
            status,
            lifetime,
            f": {text}" if text else "",
        )

        while True:
            if status in self._never_respawn:
                self._give_up(stop, FAILURE_REFUSED)
                return
            now = self._clock()
            with self._lock:
                self._crashes = [t for t in self._crashes if now - t < self._give_up_window_s]
                self._crashes.append(now)
                count = len(self._crashes)
            if count >= self._give_up_after:
                self._give_up(stop, FAILURE_GAVE_UP)
                return
            if status in self._respawn_now and lifetime >= self._quick_exit_s:
                delay = 0.0
            else:
                delay = self._backoff_s[min(count, len(self._backoff_s)) - 1]
            if delay > 0 and stop.wait(delay):
                return
            try:
                self._spawn(stop)
            except Exception:
                # A child that cannot be started is a crash like any other:
                # it is retried on the same schedule and counted.
                logger.exception("%s could not be started", self._name)
                status = None
                lifetime = 0.0
                continue
            return

    def _give_up(self, stop: threading.Event, failure: str) -> None:
        with self._lock:
            if stop.is_set():
                return
            self._active = False
            self._failure = failure
            callback = self._on_give_up
        logger.error("%s will not be restarted (%s)", self._name, failure)
        if callback is not None:
            callback()


def _signal_group(pid: int, signum: signal.Signals) -> None:
    """Send *signum* to the process group led by *pid*; a gone group is fine."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signum)
