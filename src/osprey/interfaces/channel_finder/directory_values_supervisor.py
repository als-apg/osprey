"""The Channels panel's value child, started and kept by the panel's lifespan.

The panel process never opens a control session. Every current value it shows
is read by :mod:`.directory_values_child`, which this module starts, watches and
replaces, and which it asks through :meth:`ValueSupervisor.read`.

**The socket.** An AF_UNIX listening socket, bound in a directory of its own
(mode 0700) made at :meth:`~ValueSupervisor.start`, so no other local account
can read under the operator's audit name. This process holds it across every
replacement and the child inherits it by descriptor, so a read that arrives
between two children waits in the backlog and is answered by the next one.
:meth:`~ValueSupervisor.stop` closes it and removes the directory.

**How an exit is answered** (:class:`~osprey.interfaces.web_terminal.supervised_child.SupervisedChild`):

======  =========================  =========================================
Exit    Meaning                    What follows
======  =========================  =========================================
75      the control target moved   a replacement at once, rendered again
76      the transport went stale   a replacement at once, rendered again
78      the start was refused      no replacement; values are unavailable
other   crash                      replacements after 0.5, 1, 2, 4 s
======  =========================  =========================================

Any five exits other than 78 in 60 s also make values unavailable. Unavailable
lasts until the panel restarts, and every row then carries the reason: for 78
the child's own (its last stderr line after
:data:`~.directory_values_ipc.START_REFUSED_PREFIX`).

**The child's environment**, rendered again for every (re)start:

* the panel's environment without the credential set and the terminal's names
  (:func:`~osprey.mcp_server.sandbox_env.drop_sandbox_denied_env`);
* ``OSPREY_CONFIG`` and ``CONFIG_FILE``: the panel's config file;
* ``OSPREY_CONTROL_CONTEXT_DIR``: the control-context directory this process
  resolves, so the child reads the record the panel follows;
* ``OSPREY_AUDIT_IDENTITY``: the identity this process acts as, since the
  terminal's rung of the ladder is among the names dropped.

The target is not rendered: the child reads the record when its session
opens. The variables a session refuses (``session_env.FORBIDDEN_ENV`` and the
``OSPREY_CONTROL_TARGET`` prefix) are passed through, not scrubbed: a wrong
environment ends in exit 78 with its reason. No secret is in the environment or
the argv.

**Lifetime.** Channels runs under ``uvicorn.run`` in a daemon thread, so a
process exit skips the lifespan, and the child runs in its own session: it
leaves by itself once its parent is gone. :meth:`~ValueSupervisor.stop` serves
clean shutdowns only.

POSIX only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from osprey.interfaces._serving import unix_listening_socket
from osprey.interfaces.channel_finder import directory_values_ipc as ipc
from osprey.interfaces.channel_finder.directory_values_ipc import ValueRow
from osprey.interfaces.web_terminal.supervised_child import (
    BACKOFF_S,
    FAILURE_REFUSED,
    GIVE_UP_AFTER,
    GIVE_UP_WINDOW_S,
    QUICK_EXIT_S,
    SupervisedChild,
)
from osprey.mcp_server.sandbox_env import drop_sandbox_denied_env
from osprey.runtime.session import EXIT_BINDING_MOVED, EXIT_START_REFUSED, EXIT_TRANSPORT_STALE
from osprey_connectors import posture_store
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, acting_identity

logger = logging.getLogger(__name__)

__all__ = ["READ_WAIT_S", "ValueSupervisor"]

#: Exits answered by a replacement at once.
RESPAWN_NOW_CODES = frozenset({EXIT_BINDING_MOVED, EXIT_TRANSPORT_STALE})

#: Exits after which no child is started again.
NEVER_RESPAWN_CODES = frozenset({EXIT_START_REFUSED})

#: How long a read waits for the child's answer. Above the child's row bound
#: (``PV_READ_BOUND_S`` + ``ROW_MARGIN_S``, 1.5 s), so a slow channel returns its
#: own row rather than costing the batch.
READ_WAIT_S = 3.0

#: How long a stopping child gets after SIGTERM: one batch's bound and more.
STOP_GRACE_S = 3.0

#: The socket's file name in its directory.
SOCKET_NAME = "values.sock"

_NAME = "Channels value process"


def _child_argv(fd: int) -> list[str]:
    return [sys.executable, "-m", ipc.CHILD_MODULE, ipc.SOCKET_FD_ARG, str(fd)]


def _refusal_reason(stderr_tail: str) -> str | None:
    """The reason on the child's last start-refused line, if it wrote one."""
    for line in reversed(stderr_tail.splitlines()):
        if line.startswith(ipc.START_REFUSED_PREFIX):
            return line[len(ipc.START_REFUSED_PREFIX) :].strip() or None
    return None


def _unavailable(names: Sequence[str], reason: str) -> list[ValueRow]:
    return [ValueRow(name=name, outcome=ipc.OUTCOME_UNAVAILABLE, reason=reason) for name in names]


class ValueSupervisor:
    """The value child behind its held socket, and the read the route calls.

    Args:
        config_path: The panel's config file, rendered into the child's
            environment.
        argv_for: The child's argv for the socket's descriptor; the real child
            unless a test supplies another.
        backoff_s: Waits before the replacements after a crash.
        quick_exit_s: A 75 or 76 from a child younger than this waits the
            backoff, so a child that asks to be replaced in a loop does not
            spend its five exits at once.
        socket_parent: Where the socket's directory is made; the system's
            temporary directory when not given.
    """

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        *,
        argv_for: Callable[[int], Sequence[str]] = _child_argv,
        backoff_s: Sequence[float] = BACKOFF_S,
        quick_exit_s: float = QUICK_EXIT_S,
        socket_parent: str | os.PathLike[str] | None = None,
    ) -> None:
        self._config_path = str(Path(config_path).expanduser().resolve())
        self._argv_for = argv_for
        self._backoff_s = tuple(backoff_s)
        self._quick_exit_s = quick_exit_s
        self._socket_parent = socket_parent
        self._lock = threading.Lock()
        self._directory: Path | None = None
        self._listener: socket.socket | None = None
        self._child: SupervisedChild | None = None
        self._reason: str | None = "the value process is not running"

    # -- state ----------------------------------------------------------------

    @property
    def socket_path(self) -> Path:
        """The listening socket's path.

        Raises:
            RuntimeError: Not started.
        """
        with self._lock:
            directory = self._directory
        if directory is None:
            raise RuntimeError(f"{_NAME} was not started")
        return directory / SOCKET_NAME

    @property
    def unavailable_reason(self) -> str | None:
        """Why every row reads unavailable; ``None`` while values are served."""
        with self._lock:
            return self._reason

    @property
    def pid(self) -> int | None:
        """The live child's pid; ``None`` between children or when stopped."""
        with self._lock:
            child = self._child
        return child.pid if child is not None else None

    def render_environment(self) -> dict[str, str]:
        """The child's complete environment, as it is now (module docstring)."""
        env = drop_sandbox_denied_env(os.environ)
        env["OSPREY_CONFIG"] = env["CONFIG_FILE"] = self._config_path
        state_dir = posture_store.state_dir()
        if state_dir is not None:
            env[posture_store.CONTROL_CONTEXT_DIR_ENV_VAR] = str(state_dir)
        env[AUDIT_IDENTITY_ENV] = acting_identity()
        return env

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Bind the socket and start the first child. Blocks for the fork only.

        A child that cannot be started makes values unavailable with the reason;
        nothing is raised to the lifespan.

        Raises:
            RuntimeError: Already started.
        """
        with self._lock:
            if self._directory is not None:
                raise RuntimeError(f"{_NAME} is already running")
            directory = Path(tempfile.mkdtemp(prefix="osprey-values-", dir=self._socket_parent))
            self._directory = directory
        try:
            os.chmod(directory, 0o700)
            listener = unix_listening_socket(str(directory / SOCKET_NAME))
            try:
                listener.set_inheritable(False)
            except BaseException:
                listener.close()
                raise
        except OSError as exc:
            self._fail(f"the value socket could not be bound: {exc}")
            return
        fd = listener.fileno()
        child = SupervisedChild(
            lambda: self._argv_for(fd),
            self.render_environment,
            grace_s=STOP_GRACE_S,
            respawn_now_codes=RESPAWN_NOW_CODES,
            never_respawn_codes=NEVER_RESPAWN_CODES,
            on_give_up=self._gave_up,
            pass_fds=(fd,),
            name=_NAME,
            backoff_s=self._backoff_s,
            quick_exit_s=self._quick_exit_s,
        )
        with self._lock:
            self._listener = listener
            self._child = child
            self._reason = None
        try:
            child.start()
        except OSError as exc:
            logger.error("%s could not be started", _NAME, exc_info=True)
            self._fail(f"the value process could not be started: {exc}")

    def stop(self) -> None:
        """Stop the child, close the socket and remove its directory. Idempotent."""
        with self._lock:
            child, self._child = self._child, None
            listener, self._listener = self._listener, None
            directory, self._directory = self._directory, None
            self._reason = "the value process is not running"
        if child is not None:
            child.stop()
        if listener is not None:
            listener.close()
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)

    def _fail(self, reason: str) -> None:
        """Make values unavailable with *reason* and release what start took."""
        with self._lock:
            listener, self._listener = self._listener, None
            self._reason = reason
        if listener is not None:
            listener.close()

    def _gave_up(self) -> None:
        """Supervision ended on its own; runs on the child's watcher thread."""
        with self._lock:
            child = self._child
        if child is None:
            return
        status = child.exit_status
        if child.failure == FAILURE_REFUSED:
            reason = _refusal_reason(child.stderr_tail) or (
                f"the value process refused to start (status {status})"
            )
        else:
            reason = (
                f"the value process exited {GIVE_UP_AFTER} times within "
                f"{GIVE_UP_WINDOW_S:g} s, last with status {status}"
            )
        logger.error("Current values are unavailable: %s", reason)
        self._fail(reason)

    # -- reads ----------------------------------------------------------------

    async def read(self, names: Sequence[str]) -> list[ValueRow]:
        """One row per name, in order, read by the child.

        Whatever keeps the child from answering costs the batch rows that read
        unavailable with the reason, never an exception.

        Raises:
            ValueError: More than :data:`~.directory_values_ipc.MAX_NAMES` names.
        """
        names = list(names)
        if len(names) > ipc.MAX_NAMES:
            raise ValueError(f"{len(names)} names is over the batch limit of {ipc.MAX_NAMES}")
        if not names:
            return []
        with self._lock:
            reason = self._reason
            directory = self._directory
        if reason is not None or directory is None:
            return _unavailable(names, reason or "the value process is not running")
        try:
            async with asyncio.timeout(READ_WAIT_S):
                rows = await self._ask(directory / SOCKET_NAME, names)
        except TimeoutError:
            return _unavailable(names, f"the value process did not answer within {READ_WAIT_S:g} s")
        except (OSError, ipc.FrameError) as exc:
            return _unavailable(names, f"the value process could not be read: {exc}")
        if [row.name for row in rows] != names:
            return _unavailable(names, "the value process answered for other channels")
        return rows

    @staticmethod
    async def _ask(path: Path, names: list[str]) -> list[ValueRow]:
        reader, writer = await asyncio.open_unix_connection(str(path))
        try:
            writer.write(ipc.encode_request(names))
            await writer.drain()
            message = await ipc.read_frame(reader)
            if message is None:
                raise ipc.FrameError("the connection closed before an answer")
            return ipc.decode_response(message)
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
