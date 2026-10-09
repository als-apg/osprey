"""Starting, stopping and listening to connector-host children.

Every supervisor of :mod:`osprey_connectors.ipc.host` children needs the same
four things — how the child is launched, what environment it is handed, how it
is put down, and how the requests still in flight on a child that was put down
learn why — and this module is where they are spelled once. The controls MCP
server's single-child manager and the library's multi-target
:class:`~osprey_connectors.ipc.pool.ConnectorHostPool` both use them.

Nothing here imports a control-system client library, and nothing here sets an
``EPICS_*`` variable: a supervisor is by definition a process that talks to the
control system only through its children.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Mapping
from typing import Any

from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV
from osprey_connectors.ipc.host import EPICS_ENV_PREFIXES

__all__ = [
    "CHILD_MODULE",
    "DEFAULT_TERMINATE_GRACE_S",
    "AttributedReader",
    "host_env",
    "spawn_host",
    "terminate_host",
]

#: The child is always this module, run with ``-m``. No arguments: everything
#: the child needs arrives on the wire, so nothing about a deployment shows up
#: in ``ps``.
CHILD_MODULE = "osprey_connectors.ipc.host"

#: How long a child gets between ``SIGTERM`` and ``SIGKILL``.
DEFAULT_TERMINATE_GRACE_S = 2.0


def host_env() -> dict[str, str]:
    """The environment a child is launched with: :data:`os.environ`, minus EPICS.

    The child scrubs ``EPICS_CA_*``/``EPICS_PVA_*`` again on its own first
    line, and that is the scrub the design depends on. This one is the
    defense-in-depth half: an ambient gateway never reaches the process that
    could act on it, so no window exists between exec and scrub.

    The child is stamped with
    :data:`~osprey_connectors.dotenv.ENV_CHAIN_APPLIED_ENV`. Reading its config
    file would otherwise load the project ``.env`` from its working directory
    after the scrub, and an ``EPICS_*`` line there would put back what both
    scrubs took out.
    """
    env = {
        name: value for name, value in os.environ.items() if not name.startswith(EPICS_ENV_PREFIXES)
    }
    env[ENV_CHAIN_APPLIED_ENV] = "1"
    return env


async def spawn_host(python: str, env: Mapping[str, str]) -> Any:
    """Launch one connector-host child with its frame pipes attached.

    Args:
        python: The interpreter to run the child under.
        env: The child's environment, normally :func:`host_env`.

    Returns:
        The ``asyncio.subprocess.Process``. stderr is inherited, so the child's
        diagnostics land wherever the supervisor's own do.

    Raises:
        OSError: The interpreter could not be executed. Callers wrap this in
            whatever their own failure type is.
    """
    return await asyncio.create_subprocess_exec(
        python,
        "-m",
        CHILD_MODULE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=dict(env),
    )


async def terminate_host(process: Any, grace_s: float) -> None:
    """``SIGTERM``, then ``SIGKILL`` after the grace period. Never raises.

    Only the event loop's child watcher ever reaps the child, so its own exit
    code survives; see :func:`_signal_unreaped`.
    """
    if process.returncode is not None:
        return
    _signal_unreaped(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), grace_s)
        return
    except TimeoutError:
        pass
    _signal_unreaped(process.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), grace_s)


def _signal_unreaped(pid: int, sig: int) -> None:
    """Send *sig* to the child *pid* only while it is still running, never reaping it.

    ``Process.terminate()`` and ``kill()`` go through ``Popen.send_signal``,
    which polls first; a poll that finds the child exited reaps it behind the
    event loop's child watcher, and asyncio then reports exit code 255 in place
    of the child's own. So the signal goes to the pid directly, and only after
    a ``waitid(WNOWAIT)`` look, which reports an exited child without reaping
    it.

    The look covers both kinds of watcher. Where the loop reaps the child
    itself (the pidfd watcher), an exited child stays a zombie until the loop
    gets to it: the look finds it exited and nothing is sent. Where a watcher
    thread reaps it (macOS, Linux without pidfd), the child may already be
    reaped, and its pid free for reuse, while ``returncode`` still reads
    ``None``: the look raises ``ChildProcessError`` and nothing is sent. A
    thread watcher can still reap the child between the look and the signal,
    so the window is narrowed, not closed. Without ``waitid`` (macOS before
    Python 3.13) the signal goes out unchecked.
    """
    if sys.platform != "darwin" or sys.version_info >= (3, 13):
        try:
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                return
        except ChildProcessError:
            return
    with contextlib.suppress(OSError):
        os.kill(pid, sig)


class AttributedReader:
    """A child's stdout, able to say why reading it stopped.

    The proxy fails every outstanding request with whatever ended its read
    stream. Left to itself that is an anonymous end-of-pipe, which tells an
    operator nothing about the decision that caused it — so the supervisor
    names the reason here before it kills the child, and the proxy's
    :class:`ConnectionError` carries it.
    """

    def __init__(self, stream: asyncio.StreamReader) -> None:
        self._stream = stream
        self._reason: str | None = None

    def retire(self, reason: str) -> None:
        """Name the reason the stream is about to end. Call before the kill."""
        self._reason = reason

    async def read(self, count: int) -> bytes:
        if self._reason is not None:
            raise ConnectionError(self._reason)
        chunk: bytes = await self._stream.read(count)
        # The usual path: the reader was already blocked here when the child was
        # retired, and the kill it was told about is what ends the read.
        if not chunk and self._reason is not None:
            raise ConnectionError(self._reason)
        return chunk
