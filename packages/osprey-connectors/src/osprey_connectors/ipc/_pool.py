"""Several connector-host children at once, one per target, behind one object.

A process that talks to more than one control-system target — production, a
read-only view of it, and a simulator IOC serving the *same* channel names, say
— cannot hold them in one address space: a Channel Access client reads its
environment once, process-wide, and pins the process to one gateway for its
lifetime. :class:`ConnectorHostPool` gives each target a
:mod:`connector-host child <osprey_connectors.ipc._host>` of its own and hands
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
A child is keyed by a :class:`HostKey`, its target and execution mode: ``live``
and ``live`` in ``readonly`` mode are two children, each with its own gateway
selection. A caller may also name the **owner** whose switch the child's
writes follow and a program's **write authorization**
(:mod:`~osprey_connectors.write_authorization`) they are judged by; the key
then names them too, and the child's environment carries
``OSPREY_CONTROL_OWNER`` and ``OSPREY_WRITE_AUTHORIZATION`` for it. A key that
names neither is spawned exactly as before but without any authorization this
process's own environment names: a child is authorized only by its key. A new authorization is therefore a new key and a
new child, which selects its gateway on that authorization.
:meth:`ConnectorHostPool.connector` brings the key's child up if there is none
and returns a :class:`PooledConnector` bound to the *key*, not to the process —
so a handle a caller keeps goes on working across a replacement child. Calls to
different keys never wait on each other; the only lock is per key, around
bringing a child up and around :meth:`PooledConnector.disconnect`, which waits
for a start in progress and stops the child it produced.

Bringing a child up
-------------------
Each child is a :class:`~osprey_connectors.ipc.ConnectorHost`, started on a
:class:`~osprey_connectors.ipc.HostInit` for its key; what the pool adds is
what it checks before the spawn and of the child's report after it.

The ``control_system`` section is resolved against the environment
(``${VAR}`` placeholders, via :func:`~osprey_connectors.config.resolve_env_vars`)
**at every spawn**, in this process, so a port chosen after the pool was built
is the one a fresh child dials. A placeholder that stays unresolved in the
target's connector block refuses the spawn: a child sent the literal text
``${EPICS_TESTING_PORT}`` as a port would configure an endpoint nobody meant.

Two more refusals come from the resolved section alone, before anything is
spawned — the pool's share of the eligibility gates the controls MCP server's
target switch runs:

* a Channel Access type (``epics``, ``virtual_accelerator``, ``live_standin``)
  whose block has no gateway for the role this run selects. Its child would set
  no ``EPICS_CA_*`` variable at all and search by broadcast, which the real
  machine can answer — and the post-connect check below would pass it, since a
  gatewayless derivation expects a child that configured nothing;
* a simulated or stand-in target (``standin``, ``va``) whose selected endpoint
  is one the ``live`` target derives — its read-only or write-access gateway:
  the same address string, stripped and case-folded, and the same port as
  verification compares ports — such as a stand-in port variable set to the
  live gateway's port, or a live block copied into ``live_standin``. That is
  all it requires: unlike the switch, the pool asks for no loopback host (a
  stand-in may be a compose service such as ``tuning-epics-ioc:5064``) and
  resolves no name.

The child's post-connect report is then verified against what this process
derives the child should have done
(:func:`~osprey_connectors.gateways.derive_endpoints` and
:func:`~osprey_connectors.ipc.verify_host_report`, the same
check the controls MCP server's target switch runs), and the report must name
the target and connector type this process resolved. A child that fails any of
it is stopped before the caller sees anything.

Write posture is never granted by the pool. The child decides every write by
the write verdict (:func:`osprey_connectors.write_switch.write_verdict`) over
the ``control_system`` section of the project config at ``config_file`` — not
the section it was sent — and the login's switch for its target, read from the
record its inherited environment names; with no config file reachable its
writes stay off. ``execution_mode="readonly"`` can only take writes away.
Verification compares the write posture the child reports with the verdict this
process takes over the section it was given, for every connector type, gateways
or none: a child armed where that verdict refuses — or unarmed where it allows —
is refused, so the section a caller hands the pool and ``config_file`` must
agree. A switch turned between this process's verdict and the child's report
is not such a disagreement: the verdict is taken again, and when it moved the
child is replaced by a fresh one, once, before the start fails.

The gateway a child selects follows the verdict at spawn time and does not
follow a switch turned later; a caller that wants the next write to use the
write gateway after a turn-on asks for a fresh child (the notebook kernel's
runtime does, by disconnecting the handle). Until then the child refuses an
allowed write ``switching`` when a write gateway is configured and was not
selected. A block with a ``write_access`` gateway and no ``read_only`` one is
read through the write gateway while writes are off.

A key that names an authorization selects its gateway on the authorization's
verdict, channels aside (:func:`~osprey_connectors.write_authorization.authorized_write_verdict`),
here and in the child alike. A child of such a key that came up before its
authorization could write — its handle taken before the grant was approved or
the lease started — refuses a write ``switching`` without spending a step, and
that refusal comes back to the caller like any other: take the handle once the
lease is open.

Failure
-------
Every error the pool raises of its own is a :class:`ConnectorHostError`, which
is a :class:`~osprey_connectors.errors.ConnectorUnavailableError`: the
connector is gone. Its message is the operator-facing sentence, and its
``recovery`` names what the pool did about it.

* :class:`ConnectorHostStartError` — the child could not be brought up, and
  nothing was left running: unresolved config, spawn failure, no answer to its
  init frame within ``start_timeout_s``, exit before answering, an answer that is
  not a post-connect report, or a report that failed verification. ``stage``
  says which; an ``init``-stage failure names the last start phase the child
  reached, with times (``phases``), and ``cause`` says whether the child was
  silent or exited.
* A child that answers its init frame with an **error of its own** — the
  connector's ``connect()`` failing, say — has that error raised as-is, in the
  class the child raised it as, with a note naming the child and target.
* :class:`ConnectorHostLostError` — the child went away with the call in
  flight: it exited (``cause="exited"``), or the pool stopped it because it was
  closed or :meth:`PooledConnector.disconnect` was called
  (``cause="stopped"``). The call fails; the child is dropped; the **next**
  call to that key brings up a fresh one. The same error, with
  ``cause="unavailable"``, answers a call whose connector reported itself
  unavailable (a :class:`~osprey_connectors.errors.ConnectorUnavailableError`,
  or a bare :class:`ConnectionError` from a connector that does not say which)
  in a child that answered: the child is discarded once and the next call
  starts a fresh one, and ``recovery`` says so (``child_replaced``).
* :class:`ConnectorHostUnresponsiveError` — a ``ConnectorHostLostError`` that
  is also a :class:`TimeoutError` (``cause="unresponsive"``): the child let a
  call's own ``timeout_s`` plus ``timeout_grace_s`` pass — or ``call_timeout_s``
  for a call with no timeout — without replying, **and then failed to answer a
  ping** within ``ping_timeout_s``. It is killed, and every other call in flight
  on it fails the same way. A ``TimeoutError`` because that is what the call
  would have raised in-process: the write may or may not have landed.
* :class:`ConnectorHostWriteTimeoutError` — a ``ConnectorHostLostError`` that is
  also a :class:`TimeoutError` (``cause="write_timeout"``), from a pool built
  with ``kill_on_write_timeout=True`` only: a *write* missed its deadline, and
  its child was killed **without** being pinged. Any other call in flight on
  that child fails as a ``ConnectorHostLostError`` with the same cause.

Which failure restarts what is decided by the one recovery policy,
:mod:`osprey_connectors.recovery`, at lazy timing: the rules this section
states are that policy's table, and every call that fails on a child the
policy replaced carries the policy's record in ``recovery``, whichever error
reached the pool first.
A child the pool stopped itself (``cause="stopped"``) is the pool's own answer
and no recovery.

A failed call is **never retried**: a write may already have reached the IOC,
and only the caller can decide whether to send it again. A refusal is an
answer, not a failure: the recovery policy never hears of one.

A child that misses a call's deadline but **does** answer the ping is alive and
merely slow — a batched or confirmed write can legitimately take several of its
call's timeouts — so it is left running, and the call raises a plain
:class:`TimeoutError`; it may still complete in the child. Its ``recovery``
says the child was kept (``child_kept``); a child that missed the ping too is
replaced (``child_replaced``).

That has an ordering hazard for writes. The child serves each request as a task
of its own, so a write that timed out here may still reach the machine **after
a newer write to the same channel** that the caller sent and saw complete:
``SP=1`` times out, ``SP=2`` completes, then ``SP=1`` lands. A caller for whom
that matters builds the pool with ``kill_on_write_timeout=True``, which kills
the child of any write that misses its deadline, answering or not, so nothing
left in it can send the stale write later. The write itself still may or may
not have landed; reads keep the ping rule either way.

A failure about one channel the *connector* raised in a healthy child — a
:class:`~osprey_connectors.errors.ChannelUnreachableError`, say — is passed
through unchanged and leaves the child in place, as is a connector bug
(:class:`~osprey_connectors.errors.ConnectorInternalError`). A connection
failure the proxy made itself (the pipe is gone, a
:class:`~osprey_connectors.ipc.ChildLostError`) and one the child sent are
told apart by class, never by timing.

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
pipe open — and its watchdog exits it once its parent PID differs from the one
recorded at start.

Teardown is shielded from the caller's cancellation: a call cancelled while its
child is being killed does not leave the kill half-done.

Nothing in this module imports a control-system client library, and nothing
in it sets an ``EPICS_*`` variable.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import sys
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypeVar, get_args

from osprey_connectors import write_authorization, write_switch
from osprey_connectors.config import connector_block, resolve_env_vars, unresolved_placeholders
from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.connector_types import CONTROL_TARGETS, TRANSPORT_CA, ControlTarget
from osprey_connectors.control_system._base import (
    ChannelReadBatch,
    ChannelValue,
    ChannelWriteResult,
    raise_for_write_result,
)
from osprey_connectors.errors import (
    ChildLinkError,
    RecoveryAction,
    RecoveryRecord,
)
from osprey_connectors.execution_mode import ExecutionMode, is_readonly_run
from osprey_connectors.gateways import (
    TargetDerivation,
    derive_endpoints,
    live_collision,
    readonly_gateway_refusal,
)
from osprey_connectors.ipc._handle import (
    ConnectorHost,
    ConnectorHostError,
    ConnectorHostStartError,
    HostLossCause,
    HostStartStage,
    _host_env,
)
from osprey_connectors.ipc._host import WRITE_METHODS, HostInit, HostReport, ProxyMethod
from osprey_connectors.ipc._proxy import (
    ChildLostError,
    ChildUnresponsiveError,
    ConnectorHostProxy,
)
from osprey_connectors.ipc._verification import verify_host_report
from osprey_connectors.model import FrozenModel
from osprey_connectors.process import DEFAULT_TERMINATE_GRACE_S
from osprey_connectors.recovery import RecoveryPolicy, Timing
from osprey_connectors.targets import machine_kind, resolve_target

__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_PING_TIMEOUT_S",
    "ConnectorHostLostError",
    "ConnectorHostPool",
    "ConnectorHostSlowCallError",
    "ConnectorHostUnresponsiveError",
    "ConnectorHostWriteTimeoutError",
    "HostKey",
    "PooledConnector",
]

# Named for the public namespace, not this private module, so a logging config
# keyed on the name keeps matching whatever the module is called.
logger = logging.getLogger("osprey_connectors.ipc.pool")

_T = TypeVar("_T")


#: Bound on "process started and answered its init frame". Generous, because it
#: covers a cold import of a control-system client library.
DEFAULT_START_TIMEOUT_S = 30.0

#: Local deadline for a call that names no timeout of its own.
DEFAULT_CALL_TIMEOUT_S = 60.0

#: How long a child that missed a call's deadline gets to answer a ping before
#: it is judged wedged and killed.
DEFAULT_PING_TIMEOUT_S = 2.0


class HostKey(FrozenModel):
    """What one child of the pool is for (see the module docstring).

    Attributes:
        target: The control target the child serves.
        execution_mode: ``"readonly"`` for a child that refuses every write
            whatever the config arms, or ``None``.
        owner: The login whose switch the child's writes follow, or ``None``
            for the ladder the child inherits.
        authorization: The id of the write authorization the child's writes
            are judged by, or ``None``.
    """

    target: ControlTarget
    execution_mode: ExecutionMode | None = None
    owner: str | None = None
    authorization: str | None = None


def _optional_name(name: str, value: object) -> str | None:
    """*value* stripped, or ``None`` for ``None``; a non-string or blank one is refused.

    Raises:
        ValueError: *value* is not ``None`` and not a non-blank string.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be None or a non-blank name, got {value!r}")
    return value.strip()


def _child_env(key: HostKey) -> dict[str, str]:
    """The environment a child for *key* is spawned with.

    :func:`~osprey_connectors.ipc._handle._host_env`, with the key's owner
    stamped as ``OSPREY_CONTROL_OWNER`` when it names one, and
    ``OSPREY_WRITE_AUTHORIZATION`` set to the key's authorization or removed:
    a child is authorized by its key alone, never by what this process's own
    environment names.
    """
    env = _host_env()
    if key.owner is not None:
        env[write_switch.CONTROL_OWNER_ENV_VAR] = key.owner
    env.pop(write_authorization.AUTHORIZATION_ENV_VAR, None)
    if key.authorization is not None:
        env[write_authorization.AUTHORIZATION_ENV_VAR] = key.authorization
    return env


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConnectorHostLostError(ConnectorHostError):
    """The child went away with this call in flight; the call was not retried.

    Attributes:
        cause: ``"exited"`` (the child died), ``"stopped"`` (the pool was closed
            or the key disconnected), ``"unresponsive"`` (see
            :class:`ConnectorHostUnresponsiveError`), ``"write_timeout"``
            (a write on the child missed its deadline and the pool, built with
            ``kill_on_write_timeout``, killed it; see
            :class:`ConnectorHostWriteTimeoutError`) or ``"unavailable"`` (the
            child's connector reported itself unavailable and the pool
            discarded the child).
        returncode: The child's exit status once reaped, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        target: ControlTarget,
        execution_mode: ExecutionMode | None,
        pid: int | None,
        cause: HostLossCause,
        returncode: int | None,
        recovery: tuple[RecoveryRecord, ...] = (),
    ) -> None:
        super().__init__(
            message,
            target=target,
            execution_mode=execution_mode,
            pid=pid,
            recovery=recovery,
        )
        self.cause = cause
        self.returncode = returncode


class ConnectorHostUnresponsiveError(ConnectorHostLostError, TimeoutError):
    """The child missed a call's deadline and then a ping, and was killed."""


class ConnectorHostSlowCallError(ChildLinkError, TimeoutError):
    """The child missed a call's deadline but still answered a ping, and was kept.

    Not a :class:`ConnectorHostLostError`, nor any
    :class:`ConnectionError`: the connector is not gone, and the call may yet
    complete in the child. ``recovery`` holds the supervisor's ``child_kept``
    record.
    """


class ConnectorHostWriteTimeoutError(ConnectorHostLostError, TimeoutError):
    """A write missed its deadline and its child was killed for it.

    Raised only by a pool built with ``kill_on_write_timeout=True``, only to
    the write whose deadline passed, and whether or not the child would have
    answered a ping (``cause="write_timeout"``). A :class:`TimeoutError` like
    :class:`ConnectorHostUnresponsiveError`, because that is what the write
    would have raised in-process: it may or may not have landed. What the kill
    buys is that it cannot be *sent* later — nothing is left in a child that
    could put it on the wire after a newer write to the same target.
    """


# ---------------------------------------------------------------------------
# One child
# ---------------------------------------------------------------------------


@dataclass
class _PoolChild:
    key: HostKey
    host: ConnectorHost
    #: Set once the pool has decided to drop this child, to the cause it named.
    retired: HostLossCause | None = None
    #: The one teardown of this child, shared by everyone who asks for it.
    teardown: asyncio.Task[None] | None = None

    @property
    def pid(self) -> int:
        return self.host.pid

    @property
    def proxy(self) -> ConnectorHostProxy:
        return self.host.proxy

    def usable(self) -> bool:
        return self.retired is None and self.host.usable()


class _PoolTarget:
    """One failed call's child, as the recovery policy acts on it.

    The incarnation is the child's proxy. Retiring discards the child under the
    cause the failure names; the pool is lazy, so the next call starts a fresh
    child and nothing here respawns one.
    """

    in_child = True

    def __init__(
        self, pool: ConnectorHostPool, child: _PoolChild, method: ProxyMethod, exc: BaseException
    ) -> None:
        self._pool = pool
        self._child = child
        self._method = method
        self._exc = exc
        self._pinged = False

    @property
    def incarnation(self) -> ConnectorHostProxy:
        return self._child.proxy

    async def ping(self) -> bool:
        self._pinged = True
        return await self._pool._answers_ping(self._child)

    async def retire(self) -> None:
        cause: HostLossCause
        if isinstance(self._exc, ChildUnresponsiveError):
            cause = "unresponsive" if self._pinged else "write_timeout"
        elif isinstance(self._exc, ChildLinkError):
            cause = "exited"
        else:
            # The child answered: its connector reported itself unavailable.
            cause = "unavailable"
        reason = self._pool._replaced_reason(self._child, self._method, self._exc, cause)
        await self._pool._discard(self._child, reason, cause)


class _WritesMoved(Exception):
    """A starting child's writes disagreed with a verdict that has since moved to agree."""


def _label(key: HostKey) -> str:
    label = f"target {key.target!r}" + (f" ({key.execution_mode})" if key.execution_mode else "")
    if key.owner is not None:
        label += f" for {key.owner!r}"
    if key.authorization is not None:
        label += f" under authorization {key.authorization!r}"
    return label


def _landing(method: ProxyMethod) -> str:
    """What a lost call's message says about a write that may have been sent."""
    return " The write may or may not have landed." if method in WRITE_METHODS else ""


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class ConnectorHostPool:
    """One connector-host child per :class:`HostKey`, on demand (see the module docstring).

    Args:
        control_system: The ``control_system`` config section, as loaded.
            Copied; ``${VAR}`` placeholders in it are resolved at each spawn.
        config_file: The project config the children read their write posture
            and limits from. Without one, a child reads the file
            :func:`~osprey_connectors.config.locate_config` finds in its
            environment (``OSPREY_CONFIG``, else the working directory's), and
            with none its writes stay off.
        python: The interpreter children run under.
        start_timeout_s: Bound on a child starting and answering its init frame.
        call_timeout_s: Local timeout for a call that names no ``timeout_s``;
            a child silent that long is pinged.
        timeout_grace_s: How long past a call's own ``timeout_s`` the child may
            take to answer before it is pinged. The child applies the timeout
            to the control-system call and reports its own
            :class:`TimeoutError` first, which is the better error.
        ping_timeout_s: How long a child that missed a deadline gets to answer
            a ping before it is judged wedged and killed.
        terminate_grace_s: Time between ``SIGTERM`` and ``SIGKILL``.
        kill_on_write_timeout: Kill a child whose *write* misses its deadline,
            even one that still answers a ping, and fail the write with
            :class:`ConnectorHostWriteTimeoutError`. Off, a timed-out write's
            child is kept when it answers, and the write may still land after
            a newer one (see the module docstring). Reads are never affected.
    """

    def __init__(
        self,
        control_system: Mapping[str, object],
        *,
        config_file: str | PathLike[str] | None = None,
        python: str = sys.executable,
        start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        timeout_grace_s: float = 1.0,
        ping_timeout_s: float = DEFAULT_PING_TIMEOUT_S,
        terminate_grace_s: float = DEFAULT_TERMINATE_GRACE_S,
        kill_on_write_timeout: bool = False,
    ) -> None:
        self._section: dict[str, object] = copy.deepcopy(dict(control_system))
        self._config_file = str(Path(config_file).resolve()) if config_file else None
        self._python = python
        self._start_timeout_s = start_timeout_s
        self._call_timeout_s = call_timeout_s
        self._timeout_grace_s = timeout_grace_s
        self._ping_timeout_s = ping_timeout_s
        self._terminate_grace_s = terminate_grace_s
        self._policy = RecoveryPolicy(
            timing=Timing.LAZY, replace_on_write_timeout=kill_on_write_timeout
        )
        self._teardowns: set[asyncio.Task[None]] = set()
        self._children: dict[HostKey, _PoolChild] = {}
        self._locks: dict[HostKey, asyncio.Lock] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    # -- public surface --------------------------------------------------

    async def connector(
        self,
        target: ControlTarget,
        *,
        execution_mode: ExecutionMode | None = None,
        owner: str | None = None,
        authorization: str | None = None,
    ) -> PooledConnector:
        """The connector for *target*, bringing its child up if there is none.

        Args:
            target: ``"live"``, ``"va"`` or ``"standin"``, resolved per
                :func:`~osprey_connectors.targets.resolve_target`.
            execution_mode: ``None``, or ``"readonly"`` for a child that
                refuses every write whatever the config arms.
            owner: The login whose switch the child's writes follow, stamped
                as ``OSPREY_CONTROL_OWNER`` in its environment; ``None`` leaves
                the owner to the ladder the child inherits.
            authorization: The id of the program's write authorization the
                child's writes are judged by, set as
                ``OSPREY_WRITE_AUTHORIZATION`` in its environment; ``None`` is
                a child that names none.

        Raises:
            ValueError: *target* does not resolve on this deployment,
                *execution_mode* is not one of the two accepted values, or
                *owner* or *authorization* is not a non-blank name.
            ConnectorHostStartError: The child could not be brought up.
            Exception: The child's own typed error from ``connect()``.
        """
        if execution_mode is not None and execution_mode not in get_args(ExecutionMode):
            accepted = " or ".join(repr(mode) for mode in get_args(ExecutionMode))
            raise ValueError(f"execution_mode must be None or {accepted}, got {execution_mode!r}")
        # Both are refused before either is used, owner first.
        owner_name = _optional_name("owner", owner)
        authorization_name = _optional_name("authorization", authorization)
        if target not in CONTROL_TARGETS:
            # A key names a known target; an unknown one is refused with the
            # resolver's own sentence, before anything is started.
            resolve_target(ControlSystemSection.read(self._section), target)
        key = HostKey(
            target=target,
            execution_mode=execution_mode,
            owner=owner_name,
            authorization=authorization_name,
        )
        await self._ensure(key)
        return PooledConnector(self, key)

    def pids(self) -> dict[HostKey, int]:
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

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- calls -------------------------------------------------------------

    async def _call(
        self, key: HostKey, method: ProxyMethod, call: Callable[[ConnectorHostProxy], Awaitable[_T]]
    ) -> _T:
        """Run *call*, the proxy's *method*, on *key*'s child. Never retried."""
        child = await self._ensure(key)
        ticket = self._policy.begin()
        try:
            return await call(child.proxy)
        except ChildUnresponsiveError as exc:
            if child.retired is not None:
                # Another call already retired this child; its cause stands.
                reason = (
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} was "
                    f"retired ({child.retired}) while {method!r} was in flight; the call was "
                    f"not retried.{_landing(method)}"
                )
                await self._discard(child, reason, child.retired)
                raise self._lost(
                    child,
                    reason,
                    child.retired,
                    method,
                    recovery=await self._recovered(child, method, exc, ticket),
                ) from exc
            await self._recover(child, method, exc, ticket)
            raise
        except ConnectionError as exc:
            if isinstance(exc, ChildLostError) and child.retired == "stopped":
                # The pool stopped this child itself; that is no failure to recover from.
                reason = (
                    f"The connector-host child (pid {child.pid}) serving {_label(key)} was "
                    f"stopped — the pool was closed, or the target disconnected — while "
                    f"{method!r} was in flight; the call was not retried."
                )
                await self._discard(child, reason, "exited")
                raise self._lost(child, reason, "stopped", method) from exc
            await self._recover(child, method, exc, ticket)
            raise

    async def _recover(
        self, child: _PoolChild, method: ProxyMethod, exc: BaseException, ticket: int
    ) -> None:
        """Answer a failed call as the recovery policy decides; return only when it restarts nothing."""
        target = _PoolTarget(self, child, method, exc)
        record = await self._policy.recover(
            exc, target, ticket=ticket, write=method in WRITE_METHODS
        )
        if record is None:
            return
        if record.action is RecoveryAction.CHILD_KEPT:
            raise ConnectorHostSlowCallError(
                f"The connector-host child (pid {child.pid}) serving {_label(child.key)} did not "
                f"answer {method!r} in time ({exc}), but it still answers a ping, so it was "
                "left running. The call may yet complete in the child; it was not retried.",
                recovery=(record,),
            ) from exc
        cause = child.retired or "exited"
        reason = self._replaced_reason(child, method, exc, cause)
        raise self._lost(child, reason, cause, method, recovery=(record,)) from exc

    async def _recovered(
        self, child: _PoolChild, method: ProxyMethod, exc: BaseException, ticket: int
    ) -> tuple[RecoveryRecord, ...]:
        """The record of the recovery that replaced a retired child; none when the pool stopped it."""
        if child.retired == "stopped":
            return ()
        record = await self._policy.recover(
            exc,
            _PoolTarget(self, child, method, exc),
            ticket=ticket,
            write=method in WRITE_METHODS,
        )
        if record is None or record.action is RecoveryAction.CHILD_KEPT:
            return ()
        return (record,)

    def _replaced_reason(
        self, child: _PoolChild, method: ProxyMethod, exc: BaseException, cause: HostLossCause
    ) -> str:
        """The sentence a call fails with once the pool has replaced the child it was on."""
        label = _label(child.key)
        if cause == "unavailable" and not isinstance(exc, ChildLinkError):
            return (
                f"The connector in the connector-host child (pid {child.pid}) serving "
                f"{label} is unavailable ({exc}); the child was discarded, the call "
                "was not retried, and the next call starts a fresh child."
            )
        if cause == "write_timeout":
            if method in WRITE_METHODS and isinstance(exc, ChildUnresponsiveError):
                return (
                    f"The connector-host child (pid {child.pid}) serving {label} did not "
                    f"answer {method!r} in time ({exc}) and was killed, as this pool kills a "
                    "child on any write timeout. The write may or may not have landed; it was "
                    "not retried, and the next call starts a fresh child."
                )
            return (
                f"The connector-host child (pid {child.pid}) serving {label} was "
                f"killed because another write on it missed its deadline, while "
                f"{method!r} was in flight; the call was not retried, and the next call "
                f"starts a fresh child.{_landing(method)}"
            )
        if cause == "unresponsive" and isinstance(exc, ChildUnresponsiveError):
            return (
                f"The connector-host child (pid {child.pid}) serving {label} did not "
                f"answer {method!r} in time, nor a ping afterwards, and was killed; the call "
                "was not retried, and the next call starts a fresh child."
            )
        return (
            f"The connector-host child (pid {child.pid}) serving {label} was lost "
            f"while {method!r} was in flight ({exc}); the call was not retried, and "
            "the next call starts a fresh child."
        )

    async def _answers_ping(self, child: _PoolChild) -> bool:
        """Whether a child that missed a deadline is still alive enough to answer."""
        if not child.usable():
            return False
        return await child.host.ping(timeout_s=self._ping_timeout_s)

    def _lost(
        self,
        child: _PoolChild,
        detail: str,
        cause: HostLossCause,
        method: ProxyMethod,
        *,
        recovery: tuple[RecoveryRecord, ...] = (),
    ) -> ConnectorHostLostError:
        error_class: type[ConnectorHostLostError] = ConnectorHostLostError
        if cause == "unresponsive":
            error_class = ConnectorHostUnresponsiveError
        elif cause == "write_timeout" and method in WRITE_METHODS:
            # Only a write is a write timeout: a read that was in flight on the
            # child a write timeout killed was simply lost with it.
            error_class = ConnectorHostWriteTimeoutError
        return error_class(
            detail,
            target=child.key.target,
            execution_mode=child.key.execution_mode,
            pid=child.pid,
            cause=cause,
            returncode=child.host.returncode,
            recovery=recovery,
        )

    async def _retire_key(self, key: HostKey) -> None:
        """Stop *key*'s child in an orderly way; the next call starts another.

        Taken under the key's lock, so a disconnect that lands while the key's
        child is still starting waits for that start and then stops the child
        it produced: the in-flight :meth:`connector` call returns its handle,
        and the child behind it is gone by the time this returns. Without the
        lock the disconnect would find no child yet, return at once, and the
        child would come up anyway behind a caller that believes the key is
        disconnected. Nothing that already holds a key's lock reaches this.
        """
        self._check_loop()
        async with self._locks.setdefault(key, asyncio.Lock()):
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

    def _is_closed(self) -> bool:
        """Whether :meth:`close` has run, read afresh: it can run while a call awaits."""
        return self._closed

    async def _ensure(self, key: HostKey) -> _PoolChild:
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
                    "exited",
                )
            child = await self._start(key)
            if self._is_closed():
                # close() ran while this child was starting; it waits on this
                # lock, so stopping the child here is what it is waiting for.
                await self._stop(child)
                raise RuntimeError("ConnectorHostPool was closed while this child was starting")
            self._children[key] = child
            return child

    async def _start(self, key: HostKey) -> _PoolChild:
        try:
            return await self._start_once(key, may_respawn=True)
        except _WritesMoved:
            logger.info(
                "connector host pool: the write verdict for %s moved while its child "
                "was starting; starting a fresh one",
                _label(key),
            )
            return await self._start_once(key, may_respawn=False)

    async def _start_once(self, key: HostKey, *, may_respawn: bool) -> _PoolChild:
        """Spawn, initialise and verify one child for *key*.

        With *may_respawn*, a child whose reported writes disagree with the
        verdict taken before it was spawned — where the verdict taken again now
        agrees with the child, so the switch moved in between — is discarded and
        :class:`_WritesMoved` raised for :meth:`_start` to try once more.
        """
        target, mode = key.target, key.execution_mode
        section: dict[str, object] = resolve_env_vars(copy.deepcopy(self._section))
        # Derived once, so the config refusal and the post-connect verification
        # judge the same posture snapshot.
        derivation, readonly_run, writes = self._derive(key, section)
        connector_type = derivation.connector_type

        unresolved = unresolved_placeholders(
            connector_block({"control_system": section}, connector_type)
        )
        if unresolved:
            raise ConnectorHostStartError(
                f"Refusing to start a connector-host child for {_label(key)}: "
                f"'control_system.connector.{connector_type}' still carries "
                f"{', '.join(sorted(set(unresolved)))} after environment resolution. "
                "Set the variable before the first call to this target.",
                target=target,
                execution_mode=mode,
                pid=None,
                stage="config",
            )
        refusal = self._config_refusal(key, section, derivation)
        if refusal is not None:
            raise ConnectorHostStartError(
                f"Refusing to start a connector-host child for {_label(key)}: {refusal}",
                target=target,
                execution_mode=mode,
                pid=None,
                stage="config",
            )

        host = await ConnectorHost.start(
            HostInit(
                control_system=section,
                target=target,
                config_file=self._config_file,
                execution_mode=mode,
            ),
            python=self._python,
            env=_child_env(key),
            start_timeout_s=self._start_timeout_s,
            call_timeout_s=self._call_timeout_s,
            timeout_grace_s=self._timeout_grace_s,
            terminate_grace_s=self._terminate_grace_s,
            start_marks=True,
            label=_label(key),
        )
        child = _PoolChild(key=key, host=host)
        report = host.report

        def failure(stage: HostStartStage, detail: str) -> ConnectorHostStartError:
            return ConnectorHostStartError(
                f"The connector-host child (pid {host.pid}) for {_label(key)} {detail}",
                target=target,
                execution_mode=mode,
                pid=host.pid,
                stage=stage,
            )

        try:
            if may_respawn and self._writes_moved(key, section, readonly_run, writes, report):
                raise _WritesMoved
            self._verify(derivation, readonly_run, writes, report, failure)
        except BaseException:
            await self._discard(
                child, f"The connector-host child for {_label(key)} failed to start.", "exited"
            )
            raise

        logger.info(
            "connector host pool: child pid %s serving %s as %r (role=%r %s:%s)",
            host.pid,
            _label(key),
            connector_type,
            report.selected_role,
            report.host,
            report.port,
        )
        return child

    def _derive(
        self, key: HostKey, section: Mapping[str, object]
    ) -> tuple[TargetDerivation, bool, bool]:
        """The derivation for *key*, and the ``(readonly_run, writes)`` it was taken under.

        That posture is what the child will select its gateway on, and the
        derivation's role is :func:`derive_endpoints`'s own: the one gateway rule.
        """
        readonly_run = key.execution_mode == "readonly" or is_readonly_run()
        writes = False if readonly_run else self._key_writes_allowed(section, key)
        derivation = derive_endpoints(
            {"control_system": section},
            key.target,
            writes_enabled=writes,
            readonly_run=readonly_run,
            # The child fills an unset VA gateway port from config_file, so the
            # derivation must read the same file or the ports disagree.
            config_path=self._config_file,
        )
        return derivation, readonly_run, writes

    def _key_writes_allowed(self, section: Mapping[str, object], key: HostKey) -> bool:
        """:meth:`_writes_allowed` for *key*; one that names neither owner nor authorization asks it with two arguments."""
        if key.owner is None and key.authorization is None:
            return self._writes_allowed(section, key.target)
        return self._writes_allowed(section, key.target, key.owner, key.authorization)

    @staticmethod
    def _writes_allowed(
        section: Mapping[str, object],
        target: ControlTarget,
        owner: str | None = None,
        authorization: str | None = None,
    ) -> bool:
        """Whether the write verdict allows a write to *target*, launch pin honoured.

        The verdict the child's own write guard takes, over the same record: the
        child inherits this process's environment, so it names the same owner
        and the same agent-data root. A key's *owner* is the one stamped in the
        child's environment; a key's *authorization* is judged as the child
        judges its gateway
        (:func:`~osprey_connectors.write_authorization.authorized_write_verdict`);
        a key that names none is judged by
        :func:`~osprey_connectors.write_switch.write_verdict`.
        """
        read = ControlSystemSection.read(section)
        if authorization is None:
            return write_switch.write_verdict(read, target, owner=owner).allowed
        return write_authorization.authorized_write_verdict(
            read, target, authorization=authorization, owner=owner
        ).allowed

    def _writes_moved(
        self,
        key: HostKey,
        section: Mapping[str, object],
        readonly_run: bool,
        writes: bool,
        report: HostReport,
    ) -> bool:
        """Whether a writes mismatch in *report* is a switch that moved, not a disagreement.

        True only when the child reports the other posture *and* the verdict,
        taken again now, has moved to it. A mismatch the verdict does not
        explain — ``config_file`` arming what the section does not — is left
        for :meth:`_verify` to refuse.
        """
        if readonly_run or report.writes_enabled is not (not writes):
            return False
        return self._key_writes_allowed(section, key) is not writes

    def _config_refusal(
        self, key: HostKey, section: Mapping[str, object], derivation: TargetDerivation
    ) -> str | None:
        """Why *key* must not be spawned at all, from config alone, or ``None``.

        The two eligibility gates the controls MCP server's target switch runs
        before it spawns, restated for what the pool can see — the
        ``control_system`` section, not the full project config:

        * A connector whose transport is Channel Access must select a gateway. ``connect()`` configures
          the process only ``if gateway_config:``, so a block with no gateway
          for the role this run selects sets no ``EPICS_CA_*`` variable at all
          — not even ``EPICS_CA_AUTO_ADDR_LIST=NO`` — and libca broadcasts its
          searches on the subnet, where the real machine may answer. The
          post-connect check cannot catch it: with no rows derived, a child
          that configured nothing is exactly what it expects.
        * A simulated or stand-in type must not select an endpoint the ``live``
          target derives from the same section — read-only or write-access
          alike — or its writes reach the real machine under a soft label. The
          switch proves a stand-in positively (the co-deployed port on
          loopback); that needs ``virtual_accelerator.live_standin`` from the
          full config, which the pool is never handed, and a stand-in the pool
          serves may legitimately sit on a named, non-loopback host. So here
          the refusal is the negative half — whatever the stand-in is, it is
          not the machine — with addresses compared as written, never resolved.

        A connector whose transport is in process — the simulator served inside
        the child, and the test suite's dotted doubles, which have no transport
        row — derives no rows and dials no gateway, so neither gate applies to
        it.
        """
        target = key.target
        connector_type = derivation.connector_type
        selected = derivation.selected_endpoint()
        block_key = f"control_system.connector.{connector_type}"

        if derivation.transport == TRANSPORT_CA and selected is None:
            if "write_access" in derivation.endpoints:
                # Only a readonly run selects no endpoint while a write gateway
                # is configured: it never selects write_access, and this block
                # has no read gateway for it.
                return readonly_gateway_refusal(
                    machine_kind(ControlSystemSection.read(section), target)
                )
            configured = sorted(derivation.endpoints) or "none"
            return (
                f"it resolves to {connector_type!r}, which speaks Channel Access, and "
                f"'{block_key}.gateways.{derivation.selected_role}' — the gateway this run "
                f"selects — is missing or empty (configured roles: {configured}). A child "
                "with no gateway to pin it to would broadcast its channel searches on the "
                "local network, and any IOC that answers — the real machine included — "
                f"would serve it. Configure '{block_key}.gateways.{derivation.selected_role}'."
            )

        collision = live_collision(
            {"control_system": section}, derivation, config_path=self._config_file
        )
        if collision is None:
            return None
        chosen, row = collision.selected, collision.live
        return (
            f"'{block_key}.gateways.{derivation.selected_role}' selects "
            f"{chosen.host}:{chosen.port}, which is the {collision.role!r} gateway the "
            f"live machine derives ('control_system.connector.{collision.connector_type}'"
            f".gateways.{collision.role} at {row.host}:{row.port}). Target {target!r} is "
            f"{connector_type!r}, never the real machine, so its reads and writes "
            "must not reach that endpoint. Point its gateways at its own server — "
            "check any port variable it takes, such as EPICS_TESTING_PORT."
        )

    def _verify(
        self,
        derivation: TargetDerivation,
        readonly_run: bool,
        writes: bool,
        report: HostReport,
        failure: Callable[[HostStartStage, str], ConnectorHostStartError],
    ) -> None:
        """Refuse a child whose report is not what this process derives."""
        verification = verify_host_report(
            derivation, report, readonly_run=readonly_run, writes_enabled=writes
        )
        if verification.ok:
            return
        if verification.field == "connector_type":
            raise failure(
                "verify",
                f"reports connector_type {verification.got!r} where "
                f"{verification.expected!r} was derived.",
            )
        if verification.field == "transport":
            raise failure(
                "verify",
                f"reports transport {verification.got!r} where "
                f"{verification.expected!r} was derived.",
            )
        if verification.field == "readonly_run":
            raise failure(
                "verify",
                "was asked to run readonly but reports it is not in a readonly run.",
            )
        if verification.field == "writes_enabled":
            child_writes = verification.got is True
            source = self._config_file or "none given, so OSPREY_CONFIG or the working directory's"
            raise failure(
                "verify",
                f"reports writes {'armed' if child_writes else 'off'} where the section "
                f"given to the pool has them {'armed' if writes else 'off'}. The child "
                f"takes its write tier from config_file ({source}), not from the "
                "section; the two must agree.",
            )
        raise failure(
            "verify",
            f"came up somewhere other than derived: {verification.detail}",
        )

    async def _discard(self, child: _PoolChild, reason: str, cause: HostLossCause) -> None:
        """Drop a child that failed: attribute, kill, and let the proxy settle.

        Only the key's *current* child is removed from the pool — a stale
        failure processed after a respawn must not drop the replacement. The
        first caller starts the teardown; every caller waits for it.
        """
        if self._children.get(child.key) is child:
            del self._children[child.key]
        if child.teardown is None:
            child.retired = cause
            child.teardown = self._track(self._kill(child, reason))
        await asyncio.shield(child.teardown)

    async def _stop(self, child: _PoolChild) -> None:
        """Stop a healthy child in order: acknowledge, then make sure it is gone."""
        if child.teardown is None:
            child.retired = "stopped"
            child.teardown = self._track(self._orderly_stop(child))
        await asyncio.shield(child.teardown)

    def _track(self, teardown: Coroutine[object, object, None]) -> asyncio.Task[None]:
        """Run a teardown as its own task, so a cancelled caller cannot cut it short."""
        task: asyncio.Task[None] = asyncio.get_running_loop().create_task(teardown)
        self._teardowns.add(task)
        task.add_done_callback(self._teardowns.discard)
        return task

    async def _kill(self, child: _PoolChild, reason: str) -> None:
        await child.host.retire(reason)

    async def _orderly_stop(self, child: _PoolChild) -> None:
        await child.host.stop()


# ---------------------------------------------------------------------------
# The caller's handle
# ---------------------------------------------------------------------------


class PooledConnector:
    """A target's connector, as the pool serves it.

    Bound to a pool key, not to a process: every call goes to the key's current
    child, starting a fresh one if the last one was lost. It is a
    :class:`~osprey_connectors.control_system.ControlSystem`, with the
    signatures of
    :class:`~osprey_connectors.control_system.ControlSystemConnector` for the
    methods it offers; writes are guarded and limit-checked in the child,
    exactly as they would be in-process.

    Because the handle names a key and the pool is lazy, a handle is never
    used up: after :meth:`disconnect`, or after its child was lost, the next
    call on it starts a fresh child for the key. That is the pool's property,
    not a second meaning of ``disconnect``.
    """

    def __init__(self, pool: ConnectorHostPool, key: HostKey) -> None:
        self._pool = pool
        self._key = key

    @property
    def key(self) -> HostKey:
        """The key this handle is bound to: its target, execution mode, owner and authorization."""
        return self._key

    @property
    def pid(self) -> int | None:
        """The current child's pid, or ``None`` when none is running."""
        child = self._pool._children.get(self._key)
        return child.pid if child is not None and child.usable() else None

    @property
    def report(self) -> HostReport | None:
        """The current child's post-connect report, or ``None`` when none is running."""
        child = self._pool._children.get(self._key)
        return child.host.report if child is not None and child.usable() else None

    async def read_channel(self, address: str, *, timeout_s: float | None = None) -> ChannelValue:
        return await self._pool._call(
            self._key,
            "read_channel",
            lambda proxy: proxy.read_channel(address, timeout_s=timeout_s),
        )

    async def read_multiple_channels(
        self, addresses: Sequence[str], /, *, timeout_s: float | None = None
    ) -> ChannelReadBatch:
        return await self._pool._call(
            self._key,
            "read_multiple_channels",
            lambda proxy: proxy.read_multiple_channels(addresses, timeout_s=timeout_s),
        )

    async def write_channel(
        self,
        address: str,
        value: object,
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        return await self._pool._call(
            self._key,
            "write_channel",
            lambda proxy: proxy.write_channel(address, value, timeout_s=timeout_s, confirm=confirm),
        )

    async def write_multiple_channels(
        self,
        operations: Sequence[tuple[str, object]],
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> list[ChannelWriteResult]:
        return await self._pool._call(
            self._key,
            "write_multiple_channels",
            lambda proxy: proxy.write_multiple_channels(
                operations, timeout_s=timeout_s, confirm=confirm
            ),
        )

    async def write_channel_checked(
        self,
        address: str,
        value: object,
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Write, and raise unless the write came back confirmed or unrequested.

        ``ChannelWriteBlockedError`` for a refusal — writes disabled, a
        readonly child, limits — and ``ChannelWriteFailedError`` for a write
        that was sent but not confirmed. ``timeout_s`` and ``confirm`` pass
        through to :meth:`write_channel`, the one supervised call; the result
        is held to the denial contract here, as the proxy does.
        """
        return raise_for_write_result(
            await self.write_channel(address, value, timeout_s=timeout_s, confirm=confirm)
        )

    async def validate_channel(self, address: str) -> bool:
        return await self._pool._call(
            self._key, "validate_channel", lambda proxy: proxy.validate_channel(address)
        )

    async def disconnect(self) -> None:
        """Finish with this handle: stop this key's child.

        A child still starting is waited for and then stopped, so no child for
        this key is running when this returns. The handle stays bound to its
        key, and the pool starts a fresh child on the next call (see the class
        docstring), which is how a caller asks for a child that selects its
        gateway afresh.
        """
        await self._pool._retire_key(self._key)
