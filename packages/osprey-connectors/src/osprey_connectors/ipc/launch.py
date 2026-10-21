"""Starting, stopping and listening to connector-host children.

Every supervisor of :mod:`osprey_connectors.ipc.host` children needs the same
four things — how the child is launched, what environment it is handed, how it
is put down, and how the requests still in flight on a child that was put down
learn why — and this module is where they are spelled once. Ending the process
itself is :mod:`osprey_connectors.process`'s, which spells it for every child a
supervisor owns; this module wraps it with what a connector-host child's proxy
needs around it. The controls MCP server's single-child manager and the
library's multi-target :class:`~osprey_connectors.ipc.pool.ConnectorHostPool`
both use them.

Nothing here imports a control-system client library, and nothing here sets an
``EPICS_*`` variable: a supervisor is by definition a process that talks to the
control system only through its children.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV
from osprey_connectors.ipc.host import EPICS_ENV_PREFIXES, START_MARKS_FD_ENV
from osprey_connectors.process import terminate

__all__ = [
    "CHILD_MODULE",
    "SETTLE_TIMEOUT_S",
    "AttributedReader",
    "host_env",
    "kill_host",
    "spawn_host",
    "stop_host",
]

#: The child is always this module, run with ``-m``. No arguments: everything
#: the child needs arrives on the wire, so nothing about a deployment shows up
#: in ``ps``.
CHILD_MODULE = "osprey_connectors.ipc.host"


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

    :data:`~osprey_connectors.ipc.host.START_MARKS_FD_ENV` is dropped too: a
    supervisor that is itself running under one never passes on a descriptor
    number that names something else in its child.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(EPICS_ENV_PREFIXES) and name != START_MARKS_FD_ENV
    }
    env[ENV_CHAIN_APPLIED_ENV] = "1"
    return env


async def spawn_host(
    python: str, env: Mapping[str, str], *, start_marks_fd: int | None = None
) -> Any:
    """Launch one connector-host child with its frame pipes attached.

    Args:
        python: The interpreter to run the child under.
        env: The child's environment, normally :func:`host_env`.
        start_marks_fd: The write end of a pipe the child writes its start
            marks to (see :mod:`osprey_connectors.ipc.host`), or ``None`` for
            a child that writes none. The child inherits the descriptor under
            the same number; the caller closes its own copy after the spawn.

    Returns:
        The ``asyncio.subprocess.Process``. stderr is inherited, so the child's
        diagnostics land wherever the supervisor's own do.

    Raises:
        OSError: The interpreter could not be executed. Callers wrap this in
            whatever their own failure type is.
    """
    child_env = dict(env)
    pass_fds: tuple[int, ...] = ()
    if start_marks_fd is not None:
        child_env[START_MARKS_FD_ENV] = str(start_marks_fd)
        pass_fds = (start_marks_fd,)
    return await asyncio.create_subprocess_exec(
        python,
        "-m",
        CHILD_MODULE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=child_env,
        pass_fds=pass_fds,
    )


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


#: How long, after a child is killed, the proxy's reader gets to turn the dead
#: pipe into failures on the calls that were in flight.
SETTLE_TIMEOUT_S = 2.0


async def kill_host(
    process: Any, proxy: Any, reader: AttributedReader, *, reason: str | None, grace_s: float
) -> None:
    """Put down a child that failed, and let its proxy settle. Never raises.

    Args:
        process: The child's ``asyncio.subprocess.Process``.
        proxy: The :class:`~osprey_connectors.ipc.proxy.ConnectorHostProxy`
            over its pipes.
        reader: The proxy's :class:`AttributedReader`.
        reason: Why the child is put down, named on the reader before the kill
            so every request in flight fails with it. ``None`` when the caller
            has already retired the reader.
        grace_s: How long the child gets to exit after SIGTERM before SIGKILL.
    """
    if reason is not None:
        reader.retire(reason)
    await terminate(process, grace_s)
    await proxy.drain(SETTLE_TIMEOUT_S)
    await proxy.disconnect(ack_timeout=0.0)


async def stop_host(process: Any, proxy: Any, *, grace_s: float) -> None:
    """Stop a healthy child in order: acknowledge, then make sure it is gone. Never raises.

    Args:
        process: The child's ``asyncio.subprocess.Process``.
        proxy: The :class:`~osprey_connectors.ipc.proxy.ConnectorHostProxy`
            over its pipes.
        grace_s: How long the child gets to exit after SIGTERM before SIGKILL.
    """
    await proxy.disconnect()
    await terminate(process, grace_s)
