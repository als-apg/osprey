"""Several connector-host children at once, one per target, behind one object.

A process that talks to more than one control-system target — production, a
read-only view of it, and a simulator IOC serving the *same* channel names, say
— cannot hold them in one address space: a Channel Access client reads its
environment once, process-wide, and pins the process to one gateway for its
lifetime. :class:`ConnectorHostPool` gives each target a
:mod:`connector-host child <osprey_connectors.ipc.host>` of its own and hands
the caller a connector-shaped object per target, so a caller can drive any
number of targets from one process without knowing any of that.

::

    pool = ConnectorHostPool(control_system_section, config_file="config.yml")
    live = await pool.connector("live")
    live_ro = await pool.connector("live", execution_mode="readonly")
    sim = await pool.connector("standin")
    value = await sim.read_channel("SR:DCCT")
    await pool.close()

Keys and children
-----------------
A child is keyed by ``(target, execution_mode)``: ``live`` and ``live`` in
``readonly`` mode are two children, each with its own gateway selection.
:meth:`ConnectorHostPool.connector` brings the key's child up if there is none
and returns a :class:`PooledConnector` bound to the *key*, not to the process —
so a handle a caller keeps goes on working across a replacement child. Calls to
different keys never wait on each other; the only lock is per key, around
bringing a child up.

Bringing a child up
-------------------
The ``control_system`` section is resolved against the environment
(``${VAR}`` placeholders, via :func:`~osprey_connectors.config.resolve_env_vars`)
**at every spawn**, in this process, so a port chosen after the pool was built
is the one a fresh child dials. A placeholder that stays unresolved in the
target's connector block refuses the spawn: a child sent the literal text
``${EPICS_TESTING_PORT}`` as a port would configure an endpoint nobody meant.

The child's post-connect report is then verified against what this process
derives the child should have done
(:func:`~osprey_connectors.ipc.verification.derive_endpoints` and
:func:`~osprey_connectors.ipc.verification.verify_host_report`, the same
check the controls MCP server's target switch runs), and the report must name
the target and connector type this process resolved. A child that fails any of
it is stopped before the caller sees anything.

Write posture is never granted by the pool. The child reads
``control_system.connector.<type>.writes_enabled`` from the project config at
``config_file`` — not from the section it was sent — and with no config file
reachable its writes stay off. ``execution_mode="readonly"`` can only take
writes away. Verification compares the write posture the child reports with
the one this process derives from the section it was given, for every
connector type, gateways or none: a child armed where the section says it is
not — or unarmed where the section arms it — is refused, so the section a
caller hands the pool and ``config_file`` must agree.

Failure
-------
Every error the pool raises of its own is a :class:`ConnectorHostError`, which
is a :class:`ConnectionError`, so a handler that already reads that as "the
connector is gone" keeps doing so. Its message is the operator-facing sentence.

* :class:`ConnectorHostStartError` — the child could not be brought up, and
  nothing was left running: unresolved config, spawn failure, no answer to its
  init frame within ``start_timeout_s``, exit before answering, or a report that
  failed verification. ``stage`` says which.
* A child that answers its init frame with an **error of its own** — the
  connector's ``connect()`` failing, say — has that error raised as-is, in the
  class the child raised it as, with a note naming the child and target.
* :class:`ConnectorHostLostError` — the child went away with the call in
  flight: it exited (``cause="exited"``), or the pool stopped it because it was
  closed or :meth:`PooledConnector.disconnect` was called
  (``cause="stopped"``). The call fails; the child is dropped; the **next**
  call to that key brings up a fresh one.
* :class:`ConnectorHostUnresponsiveError` — a ``ConnectorHostLostError`` that
  is also a :class:`TimeoutError` (``cause="unresponsive"``): the child let a
  call's own ``timeout`` plus ``timeout_grace_s`` pass — or ``call_deadline_s``
  for a call with no timeout — without replying, **and then failed to answer a
  ping** within ``ping_timeout_s``. It is killed, and every other call in flight
  on it fails the same way. A ``TimeoutError`` because that is what the call
  would have raised in-process: the write may or may not have landed.

A failed call is **never retried**: a write may already have reached the IOC,
and only the caller can decide whether to send it again.

A child that misses a call's deadline but **does** answer the ping is alive and
merely slow — a batched or confirmed write can legitimately take several of its
call's timeouts — so it is left running, and the call raises a plain
:class:`TimeoutError`; it may still complete in the child. An error the
*connector* raised in a healthy child — a :class:`ConnectionError` for a channel
that cannot be reached, say — is passed through unchanged and leaves the child
in place. The two are told apart by origin
(:func:`~osprey_connectors.ipc.proxy.raised_by_child`), never by timing.

Event-loop affinity
-------------------
The pool is bound to the event loop it is first used on. Its children's pipes,
its locks and every in-flight future belong to that loop, and a call from any
other loop raises :class:`RuntimeError`. A synchronous caller drives it from
one dedicated loop thread (``asyncio.run_coroutine_threadsafe`` onto that
loop); nothing here is thread-safe on its own.

Lifetime
--------
:meth:`ConnectorHostPool.close` stops every child, waiting for any that is
still starting and for any teardown already under way. A child also never
outlives this process: it exits when its stdin reaches end-of-file — which the
kernel delivers when this process dies, however it dies, because the pipe
descriptors are non-inheritable (PEP 446) and no sibling child holds another's
pipe open — and its watchdog exits it if it is ever reparented to init.

Teardown is shielded from the caller's cancellation: a call cancelled while its
child is being killed does not leave the kill half-done.

Nothing in this module imports a control-system client library, and nothing
in it sets an ``EPICS_*`` variable.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from osprey_connectors import posture_store
from osprey_connectors.config import resolve_env_vars
from osprey_connectors.control_system.base import (
    ChannelValue,
    ChannelWriteResult,
    is_readonly_run,
)
from osprey_connectors.ipc.launch import (
    DEFAULT_TERMINATE_GRACE_S,
    AttributedReader,
    host_env,
    spawn_host,
    terminate_host,
)
from osprey_connectors.ipc.proxy import (
    ChildUnresponsiveError,
    ConnectorHostProxy,
    raised_by_child,
)
from osprey_connectors.ipc.verification import derive_endpoints, verify_host_report
from osprey_connectors.types import resolve_target

__all__ = [
    "READONLY",
    "ConnectorHostError",
    "ConnectorHostLostError",
    "ConnectorHostPool",
    "ConnectorHostStartError",
    "ConnectorHostUnresponsiveError",
    "PooledConnector",
]

logger = logging.getLogger("osprey_connectors.ipc.pool")

#: The one execution mode a key may carry besides ``None``.
READONLY = "readonly"

#: Bound on "process started and answered its init frame". Generous, because it
#: covers a cold import of a control-system client library.
DEFAULT_START_TIMEOUT_S = 30.0

#: Local deadline for a call that names no timeout of its own.
DEFAULT_CALL_DEADLINE_S = 60.0

#: How long a child that missed a call's deadline gets to answer a ping before
#: it is judged wedged and killed.
DEFAULT_PING_TIMEOUT_S = 2.0

#: How long, after a child is killed, the proxy's reader gets to turn the dead
#: pipe into failures on the calls that were in flight.
_SETTLE_TIMEOUT_S = 2.0

#: The placeholder shapes :func:`resolve_env_vars` substitutes; one still
#: present after resolution named a variable the environment does not have.
_PLACEHOLDER = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")

#: Stages of :class:`ConnectorHostStartError`.
STAGE_CONFIG = "config"
STAGE_SPAWN = "spawn"
STAGE_INIT = "init"
STAGE_VERIFY = "verify"

#: Causes of :class:`ConnectorHostLostError`.
CAUSE_EXITED = "exited"
CAUSE_UNRESPONSIVE = "unresponsive"
CAUSE_STOPPED = "stopped"

_Key = tuple[str, str | None]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConnectorHostError(ConnectionError):
    """A connector-host child could not serve a call (see the module docstring).

    Attributes:
        target: The control target the child serves.
        execution_mode: ``"readonly"`` or ``None``.
        pid: The child's process id, or ``None`` when none was started.
    """

    def __init__(
        self, message: str, *, target: str, execution_mode: str | None, pid: int | None
    ) -> None:
        super().__init__(message)
        self.target = target
        self.execution_mode = execution_mode
        self.pid = pid


class ConnectorHostStartError(ConnectorHostError):
    """A child could not be brought up. Nothing was left running.

    Attributes:
        stage: ``"config"`` (unresolved placeholder), ``"spawn"`` (the
            interpreter could not be started), ``"init"`` (no answer within the
            start timeout, an exit before answering, or an unusable answer) or
            ``"verify"`` (the report did not match the derivation).
    """

    def __init__(
        self,
        message: str,
        *,
        target: str,
        execution_mode: str | None,
        pid: int | None,
        stage: str,
    ) -> None:
        super().__init__(message, target=target, execution_mode=execution_mode, pid=pid)
        self.stage = stage


class ConnectorHostLostError(ConnectorHostError):
    """The child went away with this call in flight; the call was not retried.

    Attributes:
        cause: ``"exited"`` (the child died), ``"stopped"`` (the pool was closed
            or the key disconnected) or ``"unresponsive"`` (see
            :class:`ConnectorHostUnresponsiveError`).
        returncode: The child's exit status once reaped, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        target: str,
        execution_mode: str | None,
        pid: int | None,
        cause: str,
        returncode: int | None,
    ) -> None:
        super().__init__(message, target=target, execution_mode=execution_mode, pid=pid)
        self.cause = cause
        self.returncode = returncode


class ConnectorHostUnresponsiveError(ConnectorHostLostError, TimeoutError):
    """The child missed a call's deadline and then a ping, and was killed."""


# ---------------------------------------------------------------------------
# One child
# ---------------------------------------------------------------------------


@dataclass
class _PoolChild:
    key: _Key
    process: Any
    proxy: ConnectorHostProxy
    reader: AttributedReader
    report: dict[str, Any]
    #: Set once the pool has decided to drop this child, to the cause it named.
    retired: str | None = None
    #: The one teardown of this child, shared by everyone who asks for it.
    teardown: asyncio.Task[None] | None = None

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def usable(self) -> bool:
        return (
            self.retired is None
            and self.process.returncode is None
            and self.proxy.dead_reason is None
        )


def _label(key: _Key) -> str:
    target, mode = key
    return f"target {target!r}" + (f" ({mode})" if mode else "")


def _unresolved(block: Any) -> list[str]:
    """Every placeholder left in *block*, depth-first."""
    if isinstance(block, str):
        return _PLACEHOLDER.findall(block)
    if isinstance(block, Mapping):
        return [found for value in block.values() for found in _unresolved(value)]
    if isinstance(block, list | tuple):
        return [found for value in block for found in _unresolved(value)]
    return []


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class ConnectorHostPool:
    """One connector-host child per ``(target, execution_mode)``, on demand.

    Args:
        control_system: The ``control_system`` config section, as loaded.
            Copied; ``${VAR}`` placeholders in it are resolved at each spawn.
        config_file: The project config the children read their write posture
            and limits from. Without one, a child finds whatever
            ``CONFIG_FILE`` (or ``./config.yml``) this process's environment
            leads it to, and with none its writes stay off.
        python: The interpreter children run under.
        start_timeout_s: Bound on a child starting and answering its init frame.
        call_deadline_s: Local deadline for a call that names no ``timeout``;
            a child silent that long is pinged.
        timeout_grace_s: How long past a call's own ``timeout`` the child may
            take to answer before it is pinged. The child applies the timeout
            to the control-system call and reports its own
            :class:`TimeoutError` first, which is the better error.
        ping_timeout_s: How long a child that missed a deadline gets to answer
            a ping before it is judged wedged and killed.
        terminate_grace_s: Time between ``SIGTERM`` and ``SIGKILL``.
    """

    def __init__(
        self,
        control_system: Mapping[str, Any],
        *,
        config_file: str | PathLike[str] | None = None,
        python: str = sys.executable,
        start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
        call_deadline_s: float = DEFAULT_CALL_DEADLINE_S,
        timeout_grace_s: float = 1.0,
        ping_timeout_s: float = DEFAULT_PING_TIMEOUT_S,
        terminate_grace_s: float = DEFAULT_TERMINATE_GRACE_S,
    ) -> None:
        self._section = copy.deepcopy(dict(control_system))
        self._config_file = str(Path(config_file).resolve()) if config_file else None
        self._python = python
        self._start_timeout_s = start_timeout_s
        self._call_deadline_s = call_deadline_s
        self._timeout_grace_s = timeout_grace_s
        self._ping_timeout_s = ping_timeout_s
        self._terminate_grace_s = terminate_grace_s
        self._teardowns: set[asyncio.Task[None]] = set()
        self._children: dict[_Key, _PoolChild] = {}
        self._locks: dict[_Key, asyncio.Lock] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    # -- public surface --------------------------------------------------

    async def connector(self, target: str, *, execution_mode: str | None = None) -> PooledConnector:
        """The connector for *target*, bringing its child up if there is none.

        Args:
            target: ``"live"``, ``"va"`` or ``"standin"``, resolved per
                :func:`~osprey_connectors.types.resolve_target`.
            execution_mode: ``None``, or ``"readonly"`` for a child that
                refuses every write whatever the config arms.

        Raises:
            ValueError: *target* does not resolve on this deployment, or
                *execution_mode* is not one of the two accepted values.
            ConnectorHostStartError: The child could not be brought up.
            Exception: The child's own typed error from ``connect()``.
        """
        if execution_mode not in (None, READONLY):
            raise ValueError(f"execution_mode must be None or {READONLY!r}, got {execution_mode!r}")
        key: _Key = (target, execution_mode)
        await self._ensure(key)
        return PooledConnector(self, key)

    def pids(self) -> dict[_Key, int]:
        """The live children's process ids, by key."""
        return {key: child.pid for key, child in self._children.items() if child.usable()}

    async def close(self) -> None:
        """Stop every child. Idempotent; the pool refuses calls afterwards.

        Waits for a child still starting (it is stopped as soon as it is up)
        and for every teardown already under way, so nothing this pool started
        is running when this returns.
        """
        if self._loop is not None:
            self._check_loop()
        self._closed = True
        # A start in progress holds its key's lock; taking each lock in turn
        # waits it out, and _ensure stops what it started once it sees _closed.
        for lock in list(self._locks.values()):
            async with lock:
                pass
        children = list(self._children.values())
        self._children.clear()
        await asyncio.gather(*(self._stop(child) for child in children))
        if self._teardowns:
            await asyncio.gather(*list(self._teardowns), return_exceptions=True)

    async def __aenter__(self) -> ConnectorHostPool:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # -- calls -------------------------------------------------------------

    async def _invoke(self, key: _Key, method: str, *args: Any, **kwargs: Any) -> Any:
        """Run one call on *key*'s child. Never retried."""
        child = await self._ensure(key)
        try:
            return await getattr(child.proxy, method)(*args, **kwargs)
        except ChildUnresponsiveError as exc:
            if child.retired is not None:
                # Another call already retired this child; its cause stands.
                reason = (
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} was "
                    f"retired ({child.retired}) while {method!r} was in flight; the call was "
                    "not retried."
                )
                await self._discard(child, reason, child.retired)
                raise self._lost(child, reason, child.retired) from exc
            if await self._answers_ping(child):
                raise TimeoutError(
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} did not "
                    f"answer {method!r} in time ({exc}), but it still answers a ping, so it was "
                    "left running. The call may yet complete in the child; it was not retried."
                ) from exc
            reason = (
                f"The connector-host child (pid {child.pid}) serving {_label(key)} did not "
                f"answer {method!r} in time, nor a ping afterwards, and was killed; the call "
                "was not retried, and the next call starts a fresh child."
            )
            await self._discard(child, reason, CAUSE_UNRESPONSIVE)
            raise self._lost(child, reason, CAUSE_UNRESPONSIVE) from exc
        except ConnectionError as exc:
            if raised_by_child(exc):
                # The connector's own error, sent by a child that answered.
                raise
            # Made by the proxy: the pipe is gone. If the pool already retired
            # this child, its cause is the true one; otherwise it exited.
            cause = child.retired or CAUSE_EXITED
            if cause == CAUSE_STOPPED:
                reason = (
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} was "
                    f"stopped — the pool was closed, or the target disconnected — while "
                    f"{method!r} was in flight; the call was not retried."
                )
            else:
                reason = (
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} was lost "
                    f"while {method!r} was in flight ({exc}); the call was not retried, and "
                    "the next call starts a fresh child."
                )
            await self._discard(child, reason, CAUSE_EXITED)
            raise self._lost(child, reason, cause) from exc

    async def _answers_ping(self, child: _PoolChild) -> bool:
        """Whether a child that missed a deadline is still alive enough to answer."""
        if not child.usable():
            return False
        try:
            await child.proxy.supervisor_request("ping", {}, self._ping_timeout_s)
        except Exception:
            return False
        return True

    def _lost(self, child: _PoolChild, detail: str, cause: str) -> ConnectorHostLostError:
        target, mode = child.key
        error_class = (
            ConnectorHostUnresponsiveError
            if cause == CAUSE_UNRESPONSIVE
            else ConnectorHostLostError
        )
        return error_class(
            detail,
            target=target,
            execution_mode=mode,
            pid=child.pid,
            cause=cause,
            returncode=child.process.returncode,
        )

    async def _retire_key(self, key: _Key) -> None:
        """Stop *key*'s child in an orderly way; the next call starts another."""
        self._check_loop()
        child = self._children.pop(key, None)
        if child is not None:
            await self._stop(child)

    # -- children ------------------------------------------------------------

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError(
                "ConnectorHostPool is bound to the event loop it was first used on; "
                "drive it from that loop only"
            )

    async def _ensure(self, key: _Key) -> _PoolChild:
        self._check_loop()
        if self._closed:
            raise RuntimeError("ConnectorHostPool is closed")
        child = self._children.get(key)
        if child is not None and child.usable():
            return child
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if self._closed:
                raise RuntimeError("ConnectorHostPool is closed")
            child = self._children.get(key)
            if child is not None and child.usable():
                return child
            if child is not None:
                await self._discard(
                    child,
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} "
                    "had exited; a fresh one is starting.",
                    CAUSE_EXITED,
                )
            child = await self._start(key)
            if self._closed:
                # close() ran while this child was starting; it waits on this
                # lock, so stopping the child here is what it is waiting for.
                await self._stop(child)
                raise RuntimeError("ConnectorHostPool was closed while this child was starting")
            self._children[key] = child
            return child

    async def _start(self, key: _Key) -> _PoolChild:
        target, mode = key
        readonly = mode == READONLY
        section = resolve_env_vars(copy.deepcopy(self._section))
        connector_type = resolve_target(section, target)

        connectors = section.get("connector")
        block = connectors.get(connector_type) if isinstance(connectors, dict) else None
        unresolved = _unresolved(block)
        if unresolved:
            raise ConnectorHostStartError(
                f"Refusing to start a connector-host child for {_label(key)}: "
                f"'control_system.connector.{connector_type}' still carries "
                f"{', '.join(sorted(set(unresolved)))} after environment resolution. "
                "Set the variable before the first call to this target.",
                target=target,
                execution_mode=mode,
                pid=None,
                stage=STAGE_CONFIG,
            )

        try:
            process = await spawn_host(self._python, host_env())
        except OSError as exc:
            raise ConnectorHostStartError(
                f"Could not spawn a connector-host child for {_label(key)}: {exc}",
                target=target,
                execution_mode=mode,
                pid=None,
                stage=STAGE_SPAWN,
            ) from exc

        reader = AttributedReader(process.stdout)
        proxy = ConnectorHostProxy(
            reader,
            process.stdin,
            timeout_grace_s=self._timeout_grace_s,
            deadline_s=self._call_deadline_s,
        )
        child = _PoolChild(key=key, process=process, proxy=proxy, reader=reader, report={})
        pid = child.pid
        init: dict[str, Any] = {"control_system": section, "target": target}
        if self._config_file:
            init["config_file"] = self._config_file
        if readonly:
            init["execution_mode"] = READONLY

        def failure(stage: str, detail: str) -> ConnectorHostStartError:
            return ConnectorHostStartError(
                f"The connector-host child (pid {pid}) for {_label(key)} {detail}",
                target=target,
                execution_mode=mode,
                pid=pid,
                stage=stage,
            )

        try:
            try:
                report = await proxy.supervisor_request("init", init, self._start_timeout_s)
            except ChildUnresponsiveError as exc:
                raise failure(
                    STAGE_INIT, f"did not answer its init frame within {self._start_timeout_s}s."
                ) from exc
            except Exception as exc:
                if isinstance(exc, ConnectionError) and not raised_by_child(exc):
                    await terminate_host(process, self._terminate_grace_s)
                    raise failure(
                        STAGE_INIT,
                        f"exited before answering its init frame (exit code "
                        f"{process.returncode}): {exc}",
                    ) from exc
                exc.add_note(f"raised by the connector-host child (pid {pid}) for {_label(key)}")
                raise

            if not isinstance(report, dict):
                raise failure(
                    STAGE_INIT,
                    f"answered its init frame with {type(report).__name__}, "
                    "not the post-connect report.",
                )
            child.report = report
            self._verify(key, section, connector_type, report, failure)
        except BaseException:
            await self._discard(
                child, f"The connector-host child for {_label(key)} failed to start.", CAUSE_EXITED
            )
            raise

        logger.info(
            "connector host pool: child pid %s serving %s as %r (role=%r %s:%s)",
            pid,
            _label(key),
            connector_type,
            report.get("selected_role"),
            report.get("host"),
            report.get("port"),
        )
        return child

    def _verify(
        self,
        key: _Key,
        section: dict[str, Any],
        connector_type: str,
        report: dict[str, Any],
        failure: Any,
    ) -> None:
        """Refuse a child whose report is not what this process derives."""
        target, mode = key
        for field, expected in (("target", target), ("connector_type", connector_type)):
            if report.get(field) != expected:
                raise failure(
                    STAGE_VERIFY,
                    f"reports {field} {report.get(field)!r} where {expected!r} was asked for.",
                )
        readonly_run = mode == READONLY or is_readonly_run()
        writes = False if readonly_run else posture_store.effective_writes(section, target)

        # Posture first, for every connector type: the endpoint check below
        # only sees posture through the gateway role, which a connector with no
        # gateways, or with a read_only row alone, never varies.
        if readonly_run:
            if report.get("readonly_run") is not True:
                raise failure(
                    STAGE_VERIFY,
                    "was asked to run readonly but reports it is not in a readonly run.",
                )
        else:
            child_writes = report.get("writes_enabled") is True
            if child_writes != writes:
                source = self._config_file or "none given, so CONFIG_FILE or ./config.yml"
                raise failure(
                    STAGE_VERIFY,
                    f"reports writes {'armed' if child_writes else 'off'} where the section "
                    f"given to the pool has them {'armed' if writes else 'off'}. The child "
                    f"reads its write posture from config_file ({source}), not from the "
                    "section; the two must agree.",
                )

        derivation = derive_endpoints(
            {"control_system": section},
            target,
            writes_enabled=writes,
            readonly_run=readonly_run,
        )
        verification = verify_host_report(derivation, report)
        if not verification.ok:
            raise failure(
                STAGE_VERIFY,
                f"came up somewhere other than derived: {verification.detail}",
            )

    async def _discard(self, child: _PoolChild, reason: str, cause: str) -> None:
        """Drop a child that failed: attribute, kill, and let the proxy settle.

        Only the key's *current* child is removed from the pool — a stale
        failure processed after a respawn must not drop the replacement. The
        first caller starts the teardown; every caller waits for it.
        """
        if self._children.get(child.key) is child:
            del self._children[child.key]
        if child.teardown is None:
            child.retired = cause
            child.reader.retire(reason)
            child.teardown = self._track(self._kill(child))
        await asyncio.shield(child.teardown)

    async def _stop(self, child: _PoolChild) -> None:
        """Stop a healthy child in order: acknowledge, then make sure it is gone."""
        if child.teardown is None:
            child.retired = CAUSE_STOPPED
            child.teardown = self._track(self._orderly_stop(child))
        await asyncio.shield(child.teardown)

    def _track(self, teardown: Any) -> asyncio.Task[None]:
        """Run a teardown as its own task, so a cancelled caller cannot cut it short."""
        task: asyncio.Task[None] = asyncio.get_running_loop().create_task(teardown)
        self._teardowns.add(task)
        task.add_done_callback(self._teardowns.discard)
        return task

    async def _kill(self, child: _PoolChild) -> None:
        await terminate_host(child.process, self._terminate_grace_s)
        await child.proxy.drain(_SETTLE_TIMEOUT_S)
        await child.proxy.disconnect(ack_timeout=0.0)

    async def _orderly_stop(self, child: _PoolChild) -> None:
        await child.proxy.disconnect()
        await terminate_host(child.process, self._terminate_grace_s)


# ---------------------------------------------------------------------------
# The caller's handle
# ---------------------------------------------------------------------------


class PooledConnector:
    """A target's connector, as the pool serves it.

    Bound to a pool key, not to a process: every call goes to the key's current
    child, starting a fresh one if the last one was lost. The call surface
    mirrors :class:`~osprey_connectors.control_system.base.ControlSystemConnector`
    for the methods it offers; writes are guarded and limit-checked in the child,
    exactly as they would be in-process.
    """

    def __init__(self, pool: ConnectorHostPool, key: _Key) -> None:
        self._pool = pool
        self._key = key

    @property
    def target(self) -> str:
        return self._key[0]

    @property
    def execution_mode(self) -> str | None:
        return self._key[1]

    @property
    def pid(self) -> int | None:
        """The current child's pid, or ``None`` when none is running."""
        child = self._pool._children.get(self._key)
        return child.pid if child is not None and child.usable() else None

    @property
    def report(self) -> dict[str, Any]:
        """The current child's post-connect report, empty when none is running."""
        child = self._pool._children.get(self._key)
        return dict(child.report) if child is not None and child.usable() else {}

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        result: ChannelValue = await self._pool._invoke(
            self._key, "read_channel", channel_address, timeout
        )
        return result

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        result: dict[str, ChannelValue] = await self._pool._invoke(
            self._key, "read_multiple_channels", channel_addresses, timeout
        )
        return result

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        result: ChannelWriteResult = await self._pool._invoke(
            self._key, "write_channel", channel_address, value, timeout, confirm
        )
        return result

    async def write_multiple_channels(
        self,
        operations: list[tuple[str, Any]],
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> list[ChannelWriteResult]:
        result: list[ChannelWriteResult] = await self._pool._invoke(
            self._key, "write_multiple_channels", operations, timeout, confirm
        )
        return result

    async def write_channel_checked(
        self, channel_address: str, value: Any, **kwargs: Any
    ) -> ChannelWriteResult:
        """Write, and raise unless the write came back confirmed or unrequested.

        ``ChannelWriteBlockedError`` for a refusal — writes disabled, a
        readonly child, limits — and ``ChannelWriteFailedError`` for a write
        that was sent but not confirmed. ``confirm`` and ``timeout`` pass
        through to :meth:`write_channel`.
        """
        result: ChannelWriteResult = await self._pool._invoke(
            self._key, "write_channel_checked", channel_address, value, **kwargs
        )
        return result

    async def validate_channel(self, channel_address: str) -> bool:
        result: bool = await self._pool._invoke(self._key, "validate_channel", channel_address)
        return result

    async def disconnect(self) -> None:
        """Stop this key's child. A later call on this handle starts a fresh one."""
        await self._pool._retire_key(self._key)
