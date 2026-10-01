"""A notebook kernel's control-system access, one child process per target.

A notebook kernel is the one process that outlives a control-target switch: its
``pre_run_cell`` re-stamps the target before every cell, and the next cell is
promised the new machine without a restart. The EPICS connector cannot follow
that in-process — pvapy reads ``EPICS_CA_*`` / ``EPICS_PVA_*`` once per process,
at its first channel, so a second connector for another gateway is refused with
:class:`~osprey_connectors.errors.ClientEndpointConflictError`. In a kernel the
Channel Access connector is therefore never built in this process at all: it is
served by a :class:`~osprey_connectors.ipc.pool.ConnectorHostPool` child, which
binds its own pvapy to its own target's gateway and is verified against what
this process derives before a call reaches it.

Which processes this is
-----------------------
Exactly one: the kernel process :mod:`osprey.jupyter_kernel` launched. Its
launcher stamps :data:`ENV_NOTEBOOK_KERNEL_PID` with its own pid, and
:func:`in_notebook_kernel` compares that against :func:`os.getpid` — so a
subprocess a cell starts, or a fork, inherits the variable and is still not a
kernel. An executor sandbox is stamped once and dies with its run; it keeps the
in-process connector. Within a kernel only the Channel Access types
(:data:`~osprey_connectors.types.CHANNEL_ACCESS_TYPES`) are pooled: the mock
and every other connector bind nothing per process and stay in-process, where a
mock keeps its simulated values across cells.

The loop
--------
The pool is bound to the event loop it is first used on, so a kernel gets one
for its whole life: a daemon thread running a dedicated loop, started on first
use. Every runtime coroutine in a kernel — pooled or not — runs on that loop,
marshalled with :func:`asyncio.run_coroutine_threadsafe`, so the runtime's lock,
the pool's locks and the children's pipes all belong to one loop. Nothing on
it ever calls pvapy: this process never imports it.

Lifecycle
---------
The runtime holds one connector at a time and rebuilds it when the cell's stamp
moves; a pooled connector is retired the same way, which stops its child. A
kernel therefore runs at most one child, and a switch never leaves the old
target's child behind. A child that dies is replaced by the pool on the next
call. :func:`shutdown` — run from the runtime's ``atexit`` hook — closes the
pool and stops the loop; a kernel killed outright leaves its children to exit
on their stdin's end-of-file.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey_connectors.errors import ChannelWriteFailedError
    from osprey_connectors.ipc.pool import ConnectorHostPool, PooledConnector

logger = get_logger("runtime")

#: The pid of the notebook kernel process, stamped by :mod:`osprey.jupyter_kernel`
#: (its only writer) and re-spelled there; ``tests/runtime/test_jupyter_kernel.py``
#: pins the two spellings equal.
ENV_NOTEBOOK_KERNEL_PID = "OSPREY_NOTEBOOK_KERNEL_PID"

#: How long :func:`shutdown` waits for the children to stop.
SHUTDOWN_TIMEOUT_S = 15.0


def in_notebook_kernel() -> bool:
    """Whether THIS process is a notebook kernel, and so outlives a switch."""
    return os.environ.get(ENV_NOTEBOOK_KERNEL_PID, "").strip() == str(os.getpid())


@dataclass(frozen=True)
class PoolRoute:
    """Where a kernel's pooled connector goes: one pool key, and what it was built under.

    ``launch_pin`` is part of the route because the child reads it from the
    environment it was spawned with: a child kept across a change of pin would
    answer writes from the pin it started under.
    """

    target: str
    execution_mode: str | None
    launch_pin: str | None


def pool_route(stamped_target: str | None) -> PoolRoute | None:
    """The pool key this kernel's next connector is served from, or ``None``.

    ``None`` means "build in-process": the selected connector is not a Channel
    Access one, so nothing about it is bound per process.

    An unstamped cell reads the deployment baseline, as the in-process build
    does; for a Channel Access baseline that is the target the baseline type
    is the baseline OF, which resolves back to the same type.

    Raises:
        ValueError: From :func:`~osprey_connectors.types.resolve_target`, for a
            target this deployment cannot build — as the in-process build raises.
        RuntimeError: For a baseline whose target resolves elsewhere. Building
            it in-process instead would bind pvapy here.
    """
    from osprey_connectors import posture_store
    from osprey_connectors.config import get_config_value
    from osprey_connectors.control_system.base import is_readonly_run
    from osprey_connectors.ipc.pool import READONLY
    from osprey_connectors.types import (
        CHANNEL_ACCESS_TYPES,
        baseline_target,
        resolve_control_system_type,
        resolve_target,
    )

    section = get_config_value("control_system", {})
    if not isinstance(section, dict):
        section = {}
    if stamped_target is None:
        connector_type = resolve_control_system_type(section)
        if connector_type not in CHANNEL_ACCESS_TYPES:
            return None
        target = baseline_target(section)
        if resolve_target(section, target) != connector_type:
            raise RuntimeError(
                f"The deployment baseline {connector_type!r} does not resolve back from "
                f"target {target!r}, so this kernel has no connector-host child to read it "
                "through."
            )
    else:
        target = stamped_target
        if resolve_target(section, target) not in CHANNEL_ACCESS_TYPES:
            return None
    return PoolRoute(
        target=target,
        execution_mode=READONLY if is_readonly_run() else None,
        launch_pin=os.environ.get(posture_store.LAUNCH_POSTURE_ENV_VAR),
    )


class _OwnerLoop:
    """A daemon thread running one event loop for the life of the kernel."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="osprey-runtime-kernel-loop", daemon=True
        )
        self.thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def current(self) -> bool:
        return threading.current_thread() is self.thread


_owner: _OwnerLoop | None = None
_owner_lock = threading.Lock()
#: Created on the owner loop, by the first pooled connector.
_pool: ConnectorHostPool | None = None
#: Where the children's stderr goes; ``None`` inherits this process's. Set by
#: the kernel launcher, whose descriptor 2 is published into the running cell.
_child_stderr: int | None = None


def set_child_stderr(fd: int | None) -> None:
    """Send every connector-host child's diagnostics to *fd* rather than the cell.

    Called by :mod:`osprey.jupyter_kernel` with the duplicate of the original
    stderr its own log records go to, so a child's log lines reach the terminal
    log rather than the output of the cell that happened to spawn it.
    """
    global _child_stderr
    _child_stderr = fd


def _owner_loop() -> _OwnerLoop:
    global _owner
    with _owner_lock:
        if _owner is None:
            _owner = _OwnerLoop()
        return _owner


def owner_started() -> bool:
    """Whether the kernel's loop exists — i.e. the runtime has run anything here."""
    return _owner is not None


def on_owner_loop() -> bool:
    return _owner is not None and _owner.current()


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run *coro* on the kernel's loop and wait for it, from any other thread.

    An interrupt while waiting (the operator stopping the cell) cancels the
    coroutine on the loop before it propagates.
    """
    owner = _owner_loop()
    if owner.current():
        coro.close()
        raise RuntimeError(
            "osprey.runtime's synchronous API was called from its own kernel loop thread"
        )
    future = asyncio.run_coroutine_threadsafe(coro, owner.loop)
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise


async def run_from_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    """Await *coro* on the kernel's loop from a coroutine running on another one."""
    owner = _owner_loop()
    if owner.current():
        return await coro
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, owner.loop))


def _config_file() -> str | None:
    """The config file the parent's section came from, for the child to read too.

    The child reads its write posture and limits from this file, and the pool
    refuses a child whose posture disagrees with the section it was handed —
    so both must be the one file this process loaded.
    """
    from osprey_connectors.config import default_config_path

    return default_config_path() or os.environ.get("CONFIG_FILE") or None


async def pooled_connector(route: PoolRoute) -> PooledConnector:
    """The connector for *route*, served by a connector-host child. Owner loop only."""
    global _pool
    if not on_owner_loop():
        raise RuntimeError("the kernel's connector-host pool is driven from its own loop only")
    if _pool is None:
        from osprey_connectors.config import get_config_value
        from osprey_connectors.ipc.pool import ConnectorHostPool

        section = get_config_value("control_system", {})
        _pool = ConnectorHostPool(
            section if isinstance(section, dict) else {},
            config_file=_config_file(),
            child_stderr=_child_stderr,
        )
    return await _pool.connector(route.target, execution_mode=route.execution_mode)


def is_pooled(connector: Any) -> bool:
    """Whether *connector* is served by a child. Costs no import when nothing is pooled."""
    module = sys.modules.get("osprey_connectors.ipc.pool")
    return module is not None and isinstance(connector, module.PooledConnector)


def lost_write(
    channel_address: str, value: Any, exc: BaseException
) -> ChannelWriteFailedError | None:
    """The runtime's error for a write whose child did not answer it, or ``None``.

    A child lost with the write in flight, or one that missed its deadline, may
    or may not have put the value on the wire. In-process that is a write that
    was sent and not confirmed, so it raises as one: ``UNCONFIRMED``. ``None``
    for anything the child's own connector raised — that crosses typed and is
    already the in-process error.
    """
    from osprey_connectors.control_system.base import WriteOutcome
    from osprey_connectors.errors import ChannelWriteFailedError
    from osprey_connectors.ipc.pool import ConnectorHostLostError
    from osprey_connectors.ipc.proxy import raised_by_child

    if not isinstance(exc, ConnectorHostLostError | TimeoutError) or raised_by_child(exc):
        return None
    return ChannelWriteFailedError(
        channel_address,
        "UNCONFIRMED",
        f"Write to '{channel_address}' not confirmed (UNCONFIRMED): {exc}",
        outcome=WriteOutcome.UNCONFIRMED,
        value_written=value,
    )


def shutdown(cleanup: Coroutine[Any, Any, Any] | None = None) -> None:
    """Run *cleanup*, close the pool and stop the kernel's loop. Never raises.

    Called at interpreter exit. The loop's thread is a daemon that never
    touched pvapy, so joining it cannot hang.
    """
    global _owner, _pool
    owner = _owner
    if owner is None:
        if cleanup is not None:
            cleanup.close()
        return

    async def _close() -> None:
        global _pool
        try:
            if cleanup is not None:
                await cleanup
        finally:
            pool, _pool = _pool, None
            if pool is not None:
                await pool.close()

    try:
        future = asyncio.run_coroutine_threadsafe(_close(), owner.loop)
        future.result(timeout=SHUTDOWN_TIMEOUT_S)
    except Exception:
        logger.warning("Could not stop the kernel's connector-host children", exc_info=True)
    try:
        owner.loop.call_soon_threadsafe(owner.loop.stop)
        owner.thread.join(timeout=SHUTDOWN_TIMEOUT_S)
    except Exception:
        logger.debug("Kernel loop did not stop cleanly", exc_info=True)
    _owner = None
