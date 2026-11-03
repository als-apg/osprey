"""MCP Server Context — singleton config and connector management.

Provides centralized configuration access and connector lifecycle
management for all MCP tools. Mirrors the RegistryManager pattern
from the main OSPREY framework, adapted for the simpler MCP context.

Usage in tools:
    from osprey.mcp_server.control_system.server_context import get_server_context

    registry = get_server_context()
    config = registry.config                          # Full parsed config
    connector = await registry.control_system()       # Supervised connector
    archiver = await registry.archiver()              # Supervised connector
    channel_finder_cfg = registry.channel_finder_config()
    hosts = registry.connector_hosts                  # Connector-host supervisor

The connector-host supervisor is this module's public face on
:mod:`osprey.mcp_server.control_system.connector_host_manager`, which owns the
child process, the control target and the target switch. It lives in a sibling
module because a process supervisor and a config cache have nothing to do with
each other beyond the config, and the switch algorithm is long enough that
folding it in here would bury both.

Where a tool's connector comes from
-----------------------------------
There are two serving paths, and which one a deployment gets is decided once,
by :func:`~osprey.mcp_server.control_system.connector_host_manager.switch_capable`:

* **Switch-capable** (at least two targets configured): every call through
  ``control_system()`` reaches the connector-host child. The in-process
  connector is never built — this server holds no control-system client library
  at all, which is the only way a target switch can actually move where tool
  calls land. A child that has died is not papered over: the call that finds
  it refuses with
  :class:`~osprey.mcp_server.control_system.connector_host_manager.NoConnectorHostError`,
  which names the fresh child the supervisor started, and the next call is
  served by that child.
* **Everything else** — a deployment on the simulator in process alone, a
  single-target EPICS deployment,
  any config that never named a second target — keeps the in-process cached
  connector, unchanged and untouched by any of this.

The archiver is on neither path: it speaks HTTP or pymongo, holds no Channel
Access context, and keeps serving whatever is happening to the child.

Who recovers a lost connector
-----------------------------
This context owns every connector's lifetime, so it is the one that recovers
from a failure, and the tools' error envelope only reports what it did. Each
connector is handed out as a handle bound to its key, the pool's
:class:`~osprey_connectors.ipc.PooledConnector` shape: every call resolves
the connector serving the key now. A control-system handle is bound to the
target and generation it was handed out on, and a call made after the
deployment moved off them is refused with nothing sent, so a write approved
for one target never reaches another. A call's failure is answered by its
class:

* a failure about one channel (:class:`~osprey_connectors.errors.ChannelError`)
  and a connector bug recover nothing;
* :class:`~osprey_connectors.errors.ConnectorUnavailableError`, or a bare
  :class:`ConnectionError` from a connector that does not say which, replaces
  the connector it came from — a fresh connector-host child, or the in-process
  instance dropped for the next call to rebuild;
* a call the child did not answer in time pings that child: one that answers
  is kept, one that does not is replaced.

Each failed connector instance or child is recovered at most once, however
many calls saw it fail, and every failure it answered carries the one record
of what was done as ``recovery``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, overload

from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
    NoConnectorHostError,
    SwitchError,
    baseline_target,
    switch_capable,
)
from osprey_connectors import control_context
from osprey_connectors.archiver import ArchiverConnector
from osprey_connectors.config import mapping_or_empty
from osprey_connectors.control_system import ControlSystemConnector
from osprey_connectors.errors import ConfigurationError, ConnectorUnavailableError
from osprey_connectors.ipc import DEFAULT_PING_TIMEOUT_S, ChildUnresponsiveError, ConnectorHostProxy
from osprey_connectors.recovery import RecoveryAction
from osprey_connectors.types import configured_targets, tier_allows

__all__ = [
    "ConnectorEntry",
    "ConnectorHostManager",
    "ConnectorHostProxy",
    "ControlSystemContext",
    "MCPServerConfig",
    "NoConnectorHostError",
    "SwitchError",
    "claim_control_context",
    "get_server_context",
    "initialize_server_context",
    "launch_target_from_record",
    "reset_server_context",
]

logger = logging.getLogger("osprey.mcp_server.control_system.server_context")


# ---------------------------------------------------------------------------
# The deployment's control-context record
# ---------------------------------------------------------------------------


def launch_target_from_record() -> str | None:
    """The target this deployment's first connector host should come up on.

    The target belongs to the deployment, not to whichever server process
    happens to be serving it: a server that starts while the record says
    ``va`` brings its first child up on ``va``, publishes ``applied (va, N)``
    and mints nothing. Starting on the baseline instead would serve the wrong
    machine until somebody noticed, and then charge the operator a generation
    to get back to where the deployment already was.

    ``None`` means there is no record to follow — an empty deployment, an
    unresolvable agent-data root, or a file too degraded to name a target — and
    is the one case where the supervisor falls back to the deployment baseline,
    exactly as it did before a record existed.
    """
    record = control_context.read_record()
    return None if record is None else record.target


def claim_control_context(
    *, baseline: str, server_pid: int | None = None
) -> control_context.ControlContext | None:
    """Own the control-context record when nothing alive already owns it.

    A controls server is the *fallback* owner. It claims over an owner that is
    absent, dead, or too degraded to name a process, and it is a follower
    behind any live one: behind a web terminal, which outranks it and is what
    an operator is actually looking at, and behind another controls server,
    which got there first. A follower writes nothing here — it files requests.

    **A claim is a merge.** The target, generation and switch in the record are
    what this deployment is pointed at, and they outlive every process that
    reads them; taking the file over changes who may write it, and nothing else
    of a record somebody has switched. A record nobody has switched follows the
    deployment's baseline; every applied switch mints ``generation + 1``, so
    generation 0 is a claim and never a switch (:func:`control_context.claimed`).
    A record that is absent or too degraded to read at all starts one, at
    *baseline*, generation 0, every switch off.

    The write is read-verified: two servers starting together can both find the
    record ownerless, and the one that lost the race has to find that out
    rather than go on believing it owns a file somebody else is writing.

    Never raises. A server that cannot claim is a server that follows, and
    refusing to start over it would cost the operator every control-system
    tool instead.

    Args:
        baseline: The deployment's own configured target. It starts an absent
            record, and it is what an unswitched (generation 0) record follows.
        server_pid: The PID to claim as. Defaults to this process, which is the
            only value production uses.

    Returns:
        The record this server works from afterwards — the one it claimed, the
        one a live owner holds, or the one that beat it to the claim — and
        ``None`` when there is no record and none could be written.
    """
    try:
        record = control_context.read_record()
        owner = control_context.live_owner(record)
        if record is not None and owner is not None:
            logger.info(
                "Following the control context owned by %s pid %s: target %r, generation %s",
                owner.kind,
                owner.pid,
                record.target,
                record.generation,
            )
            return record

        pid = os.getpid() if server_pid is None else server_pid
        owner = control_context.Owner(
            kind=control_context.OWNER_CONTROLS_SERVER, pid=pid, port=None
        )
        claimed = (
            control_context.ControlContext(target=baseline, generation=0, owner=owner)
            if record is None
            else replace(control_context.claimed(record, baseline), owner=owner)
        )
        control_context.write_record(claimed)

        confirmed = control_context.read_record()
        if confirmed is None or confirmed.owner is None or confirmed.owner.pid != pid:
            logger.warning(
                "The control-context claim did not stick; another process owns the record now"
            )
            return confirmed
        logger.info(
            "Claimed the control context: target %r, generation %s",
            confirmed.target,
            confirmed.generation,
        )
        return confirmed
    except Exception:
        logger.warning(
            "Could not claim the control-context record; this server follows whatever owns it",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Registration metadata (mirrors osprey.registry.base patterns)
# ---------------------------------------------------------------------------


@dataclass
class ConnectorEntry:
    """Cached connector with its config for reconnection."""

    config: dict[str, Any]
    instance: ControlSystemConnector | ArchiverConnector | None = None
    connector_type: str = ""  # "control_system" or "archiver"


@dataclass
class MCPServerConfig:
    """Parsed and validated server configuration."""

    raw: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    def _section(self, key: str) -> dict[str, Any]:
        """Read the section at ``key`` as a mapping; empty reads as ``{}``."""
        return mapping_or_empty(self.raw.get(key), key, self.config_path)

    @property
    def control_system(self) -> dict[str, Any]:
        return self._section("control_system")

    @property
    def archiver(self) -> dict[str, Any]:
        return self._section("archiver")

    @property
    def channel_finder(self) -> dict[str, Any]:
        return self._section("channel_finder")

    @property
    def ariel(self) -> dict[str, Any]:
        return self._section("ariel")


# ---------------------------------------------------------------------------
# The supervised handle
# ---------------------------------------------------------------------------

#: One thing the supervisor did about a failure: ``{"action": <word>, "text": <sentence>}``.
Recovery = dict[str, str]


#: The contract each key's connector states, which a handle reads when none serves.
_CONTRACTS: dict[str, type] = {
    "control_system": ControlSystemConnector,
    "archiver": ArchiverConnector,
}


class _SupervisedConnector:
    """A connector as the tools see it: bound to its key, never to one instance.

    Every call resolves the connector serving the key now and runs under the
    supervision :class:`ControlSystemContext` gives it (see the module
    docstring). A control-system handle on a child-served deployment is also
    bound to the ``(target, generation)`` it was handed out on. A coroutine
    method of the connector is a supervised call; any other attribute is read
    off the connector serving now.
    """

    def __init__(
        self,
        context: ControlSystemContext,
        name: str,
        binding: tuple[str, int] | None = None,
    ) -> None:
        self._context = context
        self._name = name
        self.binding = binding

    @property
    def connector(self) -> Any:
        """The connector serving this key now, or ``None`` when none is."""
        return self._context._serving(self._name)

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        current = self._context._serving(self._name)
        if current is not None:
            value = getattr(current, attribute)
            if not inspect.iscoroutinefunction(value):
                return value
        elif not inspect.iscoroutinefunction(getattr(_CONTRACTS[self._name], attribute, None)):
            # With nothing serving, only a call the contract states can be offered:
            # the call resolves a connector, an attribute read has none to read.
            raise AttributeError(f"no {self._name} connector is serving to read {attribute!r} off")

        async def supervised(*args: Any, **kwargs: Any) -> Any:
            return await self._context._supervised_call(self, attribute, args, kwargs)

        supervised.__name__ = attribute
        return supervised


# ---------------------------------------------------------------------------
# ControlSystemContext
# ---------------------------------------------------------------------------


class ControlSystemContext:
    """Singleton registry that caches config and control-system connectors for MCP tools.

    Responsibilities:
      1. Load and cache config.yml once at startup
      2. Register connector types with ConnectorFactory
      3. Provide cached connector instances (lazy-created, auto-reconnect)
      4. Expose config sections to tools
    """

    def __init__(self) -> None:
        self._config: MCPServerConfig | None = None
        self._connectors: dict[str, ConnectorEntry] = {}
        self._initialized = False
        self._connector_hosts: ConnectorHostManager | None = None
        self._switch_capable: bool | None = None
        #: The one handle each key is served through, by key.
        self._handles: dict[str, _SupervisedConnector] = {}
        #: The latest recovery each key ran, with the instance or child it was for.
        self._recoveries: dict[str, tuple[Any, asyncio.Task[Recovery | None]]] = {}

    def initialize(self) -> None:
        """Load config and register connector types.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        # 1. Load config
        self._config = self._load_config()
        logger.info("ControlSystemContext: config loaded from %s", self._config.config_path)

        # 2. Register connector types with ConnectorFactory
        self._register_connector_types()

        # 3. Pre-populate connector entries (not connected yet — lazy)
        self._connectors["control_system"] = ConnectorEntry(
            config=self._config.control_system,
            connector_type="control_system",
        )
        self._connectors["archiver"] = ConnectorEntry(
            config=self._config.archiver,
            connector_type="archiver",
        )

        # 4. Validate config: warnings for the misconfigurations a tool can
        #    still work around, and one hard refusal for the pairing no tool can
        #    (see _validate).
        self._validate()

        self._initialized = True
        logger.info(
            "ControlSystemContext: initialized (control_system=%s, archiver=%s, "
            "targets=%s, tier_allows_writes=%s, serving=%s)",
            self._config.control_system.get("type", "not configured"),
            self._config.archiver.get("type", "not configured"),
            # The targets this config configures, not every target the
            # connectors package names. Whether a write lands is the write
            # verdict's, per call; what is known at start is the tier alone.
            list(configured_targets(self._config.control_system)),
            tier_allows(self._config.control_system),
            "connector-host child" if self.switch_capable else "in-process connector",
        )

    @property
    def config(self) -> MCPServerConfig:
        """Full parsed configuration."""
        if self._config is None:
            raise RuntimeError("ControlSystemContext not initialized — call initialize() first")
        return self._config

    @property
    def connector_hosts(self) -> ConnectorHostManager:
        """The connector-host supervisor, created on first use.

        Created lazily and never started here: a deployment that serves its
        tools from the in-process connector never spawns a child, and asking
        for the supervisor is not the same as running one. Whether a child is
        alive is :meth:`ConnectorHostManager.has_child`'s answer, not this
        property's.
        """
        if self._connector_hosts is None:
            self._connector_hosts = ConnectorHostManager(self.config)
        return self._connector_hosts

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-path access into raw config: registry.get('archiver.type')."""
        parts = key.split(".")
        value: Any = self.config.raw
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def baseline(self) -> str:
        """The target this deployment's own config selects.

        Derived from the config through
        :func:`~osprey.mcp_server.control_system.connector_host_manager.baseline_target`
        rather than asked of the connector-host supervisor: a deployment that
        serves its tools in-process must never construct one, and the baseline
        is a property of the file it was built from either way.
        """
        return baseline_target(self.config.raw)

    @property
    def switch_capable(self) -> bool:
        """Whether this deployment serves its control system from a child.

        The predicate itself lives in
        :func:`~osprey.mcp_server.control_system.connector_host_manager.switch_capable`
        and is spelled exactly once there; this caches the answer, which cannot
        change while the config is loaded.
        """
        if self._switch_capable is None:
            self._switch_capable = switch_capable(self.config.raw)
        return self._switch_capable

    async def control_system(self) -> _SupervisedConnector:
        """The supervised handle every tool reads and writes the control system through.

        On a switch-capable deployment every call through it reaches the
        **connector-host child** serving the target the handle was handed out
        on, which is what makes a target switch reach the tools at all: the
        child holds the Channel Access context for the recorded control target,
        and a switch replaces the child. On every other deployment it reaches
        the in-process connector, built once and cached.

        The connector is resolved here too, so the first tool call brings the
        first child up, and a call that finds the child dead is refused here
        with the restart named (see :meth:`ConnectorHostManager.serving_proxy`).

        Raises:
            NoConnectorHostError: No child was live to serve this call.
        """
        return await self._handle("control_system")

    async def archiver(self) -> _SupervisedConnector:
        """The supervised handle on the cached archiver connector.

        Never routed through a connector host: the archiver speaks HTTP or
        pymongo, holds no Channel Access context, and therefore keeps serving
        while no child is alive.
        """
        return await self._handle("archiver")

    async def _handle(self, name: str) -> _SupervisedConnector:
        """Resolve *name*'s connector, then hand out the one handle bound to it."""
        await self._get_connector(name)
        binding = self._child_binding(name)
        handle = self._handles.get(name)
        if handle is None or handle.binding != binding:
            handle = _SupervisedConnector(self, name, binding)
            self._handles[name] = handle
        return handle

    def _served_from_a_child(self, name: str) -> bool:
        """Whether *name* is the control system served from a connector-host child."""
        return name == "control_system" and self.switch_capable

    def _child_binding(self, name: str) -> tuple[str, int] | None:
        """The ``(target, generation)`` a child-served handle is bound to, else ``None``."""
        if not self._served_from_a_child(name):
            return None
        return self.connector_hosts.active_binding()

    def _serving(self, name: str) -> Any:
        """The connector serving *name* now, without resolving anything, or ``None``."""
        if self._served_from_a_child(name):
            return self.connector_hosts.active_proxy() if self._connector_hosts else None
        entry = self._connectors.get(name)
        return entry.instance if entry is not None else None

    async def _supervised_call(
        self,
        handle: _SupervisedConnector,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run one call on the connector serving *handle*'s key, under supervision."""
        name = handle._name
        connector = await self._get_connector(name)
        binding = self._child_binding(name)
        if binding != handle.binding and handle.binding is not None and binding is not None:
            (was, was_generation), (now, now_generation) = handle.binding, binding
            raise ConnectorUnavailableError(
                f"The control-system target moved from {was!r} (generation {was_generation}) "
                f"to {now!r} (generation {now_generation}) before {method!r}; nothing was sent."
            )
        try:
            return await getattr(connector, method)(*args, **kwargs)
        except ChildUnresponsiveError as exc:
            if self._served_from_a_child(name):
                exc.recovery = self._records(  # type: ignore[attr-defined]
                    await self._recover(name, connector, self._answer_unresponsive)
                )
            raise
        except ConnectorUnavailableError as exc:
            exc.recovery = self._records(await self._recover(name, connector, self._replace))
            raise
        except ConnectionError as exc:
            # A connector that does not say which failure it had: its own loss.
            record = await self._recover(name, connector, self._replace)
            raise ConnectorUnavailableError(str(exc), recovery=self._records(record)) from exc

    @staticmethod
    def _records(record: Recovery | None) -> tuple[Recovery, ...]:
        return (record,) if record is not None else ()

    async def _recover(
        self,
        name: str,
        incarnation: Any,
        act: Callable[[str, Any], Awaitable[Recovery]],
    ) -> Recovery | None:
        """Run *act* once for the instance or child that failed; share its record.

        Every call that saw the same incarnation fail waits on, and reports,
        the one recovery already run or running for it. A kept child is not a
        recovery that settles anything, so a later failure of it runs afresh.
        """
        held = self._recoveries.get(name)
        if held is not None and held[0] is incarnation:
            task = held[1]
            if not task.done() or (task.result() or {}).get("action") != RecoveryAction.CHILD_KEPT:
                return await asyncio.shield(task)
        task = asyncio.ensure_future(self._guarded(act, name, incarnation))
        self._recoveries[name] = (incarnation, task)
        return await asyncio.shield(task)

    @staticmethod
    async def _guarded(
        act: Callable[[str, Any], Awaitable[Recovery]], name: str, incarnation: Any
    ) -> Recovery | None:
        """*act*'s record; a recovery that itself failed is logged and reports nothing."""
        try:
            return await act(name, incarnation)
        except Exception:
            logger.exception("Recovering the %s connector failed", name)
            return None

    async def _replace(self, name: str, incarnation: Any) -> Recovery:
        """Replace the connector a call found unavailable, and say what became of it.

        The record states what happened to the instance or child that served the
        call: it is gone or replaced whether this recovery, a switch or the
        reconciler replaced it first, so the record names no actor and no target.
        """
        replaced = await self.invalidate_connector(name, incarnation=incarnation)
        if not self._served_from_a_child(name):
            return {
                "action": RecoveryAction.CONNECTOR_DROPPED,
                "text": f"The {name.replace('_', ' ')} connector was dropped; "
                "the next call builds a new one.",
            }
        if replaced:
            return {
                "action": RecoveryAction.CHILD_REPLACED,
                "text": "The connector-host child that served this call has been replaced; "
                "the call was not retried.",
            }
        return self._respawn_refused()

    async def _answer_unresponsive(self, name: str, proxy: Any) -> Recovery:
        """Ping the child a call did not hear back from; replace it if it is silent.

        A child that answers the ping is alive and merely slow, so it is kept:
        the call may yet complete in it. The ping goes to the child the call was
        sent to, not to whichever is active now. Nothing is retried.
        """
        try:
            await proxy.supervisor_request("ping", {}, DEFAULT_PING_TIMEOUT_S)
        except Exception:
            if not await self.invalidate_connector(name, incarnation=proxy):
                return self._respawn_refused()
            return {
                "action": RecoveryAction.CHILD_REPLACED,
                "text": "The connector-host child did not answer a ping and has been replaced; "
                "the call was not retried.",
            }
        return {
            "action": RecoveryAction.CHILD_KEPT,
            "text": "The connector-host child still answers a ping, so it was kept; the call "
            "may yet complete in it and was not retried.",
        }

    def _respawn_refused(self) -> Recovery:
        """The record of a replacement the manager refused, saying what still serves."""
        if self.connector_hosts.has_child():
            still = "the previous child still serves this target."
        else:
            still = "no child is serving; control-system calls are refused until one is running."
        return {
            "action": RecoveryAction.RESPAWN_REFUSED,
            "text": f"Replacing the connector-host child was refused; {still}",
        }

    @overload
    async def _get_connector(
        self, name: Literal["control_system"]
    ) -> ControlSystemConnector | ConnectorHostProxy: ...

    @overload
    async def _get_connector(self, name: Literal["archiver"]) -> ArchiverConnector: ...

    @overload
    async def _get_connector(
        self, name: str
    ) -> ControlSystemConnector | ArchiverConnector | ConnectorHostProxy: ...

    async def _get_connector(
        self, name: str
    ) -> ControlSystemConnector | ArchiverConnector | ConnectorHostProxy:
        """Resolve the connector serving *name* now: lazily built, or the live child's proxy."""
        entry = self._connectors.get(name)
        if entry is None:
            raise ValueError(f"Unknown connector: {name}")

        if self._served_from_a_child(name):
            return await self._connector_host()

        if entry.instance is not None:
            return entry.instance

        from osprey_connectors.factory import ConnectorFactory

        instance: ControlSystemConnector | ArchiverConnector
        if name == "control_system":
            # A deployment on this path serves one target and never switches, so
            # the target it is on is the deployment's own baseline. Naming it
            # rather than leaving the stamp blank is what makes the rebuild after
            # invalidate_connector() take the write verdict for the same target
            # the instance it replaces was taking it for.
            instance = await ConnectorFactory.create_control_system_connector(
                entry.config, control_target=self.baseline
            )
        elif name == "archiver":
            instance = await ConnectorFactory.create_archiver_connector(entry.config)
        else:
            raise ValueError(f"Unknown connector: {name}")
        entry.instance = instance

        logger.info("ControlSystemContext: created %s connector", name)
        return instance

    async def _connector_host(self) -> ConnectorHostProxy:
        """The live child's proxy, or the no-child refusal that names its restart.

        The in-process connector is never built on this path — building one
        would load a control-system client into the server that is supposed to
        hold none, and pin it to whatever target the config happened to
        describe at start, which is precisely the bug a switch has to avoid.

        The deployment's very first child comes up on the target the record
        names (:func:`launch_target_from_record`), not on the baseline. The
        record is consulted only while no child has ever started: after that the
        supervisor holds the target of record, ``ensure_started`` is a no-op,
        and re-reading a file on the way to every tool call would buy nothing.
        """
        manager = self.connector_hosts
        await manager.ensure_started(None if manager.is_started() else launch_target_from_record())
        return await manager.serving_proxy()

    async def invalidate_connector(self, name: str, *, incarnation: Any = None) -> bool:
        """Disconnect and remove a cached connector, so the next call rebuilds it.

        When the control system is being served from a connector-host child,
        "drop the instance so the next call rebuilds it" is a kill and a
        respawn of that child on the *same* target — the generation does not
        move, because the session is still pointed where it was. A deployment
        with no running child takes the in-process path below, unchanged.

        It has two callers. This context's supervisor passes the *incarnation*
        that failed — the in-process instance, or the child's proxy — so a
        connector another call or a switch already replaced is left alone, and
        a dead child is restarted rather than replaced by force. The
        session-control reconciler
        (:mod:`osprey.mcp_server.control_system.session_control`) passes none: it
        rebuilds the connector when the write verdict for the target the
        deployment is on moves, because the child connected on a gateway role
        chosen under the old verdict and only a rebuild moves it. Neither takes
        a lock of its own — the manager's is the lock, and a second one around
        the same operation is how two things that must agree stop agreeing.

        Returns:
            Whether the connector this call was asked about is now gone or
            replaced. ``False`` means one specific thing: the connector-host
            respawn was attempted and refused, so the OLD child is still
            serving and whatever the caller invalidated it for has not
            happened. Every caller reads it: the reconciler to report a
            realignment truthfully — one reported as done on a child that never
            restarted would tell an operator their switch had taken effect when
            it had not — and the supervisor to report ``respawn_refused``
            instead of ``child_replaced`` in the recovery record it attaches.
        """
        if (
            name == "control_system"
            and self._connector_hosts is not None
            and self._connector_hosts.is_started()
        ):
            try:
                if incarnation is None:
                    await self._connector_hosts.respawn_same_target()
                else:
                    await self._connector_hosts.replace_child(incarnation)
            except SwitchError as exc:
                # Spawn-then-swap all the way down: a respawn that fails leaves
                # the existing child in place rather than tearing down the only
                # thing still able to serve. If it is dead too, has_child()
                # already says so and the tools refuse on that. Swallowed here
                # and reported as False: this is not an error the *caller* can
                # act on, but it is a fact the caller has to know.
                logger.error("Could not respawn the connector host: %s", exc.detail)
                return False
            return True

        entry = self._connectors.get(name)
        if entry and entry.instance and (incarnation is None or entry.instance is incarnation):
            try:
                await entry.instance.disconnect()
            except Exception:
                logger.debug("Error disconnecting %s (ignored)", name, exc_info=True)
            entry.instance = None
            logger.info("ControlSystemContext: invalidated %s connector", name)
        return True

    def channel_finder_config(self) -> dict[str, Any]:
        """Config section for ChannelFinderService."""
        return self.config.channel_finder

    @staticmethod
    def _load_config() -> MCPServerConfig:
        """Load config.yml via the shared config loader."""
        from osprey_connectors.workspace import load_osprey_config, resolve_config_path

        raw = load_osprey_config()
        config_path = resolve_config_path()
        if not config_path.exists():
            logger.warning("Config file not found: %s", config_path)

        return MCPServerConfig(raw=raw, config_path=config_path)

    @staticmethod
    def _register_connector_types() -> None:
        """Register all connector types with ConnectorFactory."""
        from osprey_connectors.factory import register_builtin_connectors

        register_builtin_connectors()

    def _validate(self) -> None:
        """Warn about common misconfigurations, and refuse the one that lies.

        Warnings for the rest: an unknown connector type still produces a server
        whose other tools work, and a missing section is often a project mid-edit.

        The exception is a virtual accelerator paired with the mock archiver.
        Every tool in this server would answer, and the archiver's answers would
        be invented — so the failure is not visible in any single call, only in
        the agent's account of a machine it is also reading live. This is the
        honesty rule's runtime site: it catches a ``config.yml`` hand-edited into
        the pairing long after the build refused to write it.

        Raises:
            ConfigurationError: If ``config.yml`` pairs a virtual accelerator
                with the mock archiver (an unset ``archiver.type`` included —
                the factory resolves it to the mock).
        """
        self._refuse_invented_history()

        from osprey_connectors.factory import ConnectorFactory

        cs = self.config.control_system
        if not cs:
            logger.warning("No control_system section in config.yml")
        else:
            cs_type = cs.get("type")
            known = set(ConnectorFactory.list_control_systems())
            if cs_type and cs_type not in known and "." not in cs_type:
                logger.warning("Unknown control_system.type: %s (registered: %s)", cs_type, known)

        arch = self.config.archiver
        if not arch:
            logger.warning("No archiver section in config.yml")
        else:
            arch_type = arch.get("type")
            known_arch = set(ConnectorFactory.list_archivers())
            if arch_type and arch_type not in known_arch and "." not in arch_type:
                logger.warning("Unknown archiver.type: %s (registered: %s)", arch_type, known_arch)

    def _refuse_invented_history(self) -> None:
        """Abort startup on a machine with no recorded past and a synthesizing
        archiver.

        Judged by :func:`~osprey_connectors.honesty.pairing_in_rendered_config`,
        which resolves both keys through *nested sections only* — exactly as
        :attr:`MCPServerConfig.control_system` and :attr:`MCPServerConfig.archiver`
        do a few lines above, and exactly as the factory then reads what they
        hand it. A guard that resolved a config differently from the reader it
        guards would not be a guard; the divergence would be the way through.

        The refused machine is *named* rather than assumed: the pairing covers
        every type in :data:`~osprey_connectors.types.INVENTED_HISTORY_TYPES`,
        so a deployment baselined on the live stand-in is told about the
        stand-in and not sent looking for a virtual accelerator its
        ``control_system:`` section does not name.
        """
        from osprey_connectors.honesty import VA_MOCK_ARCHIVER_WHY, pairing_in_rendered_config
        from osprey_connectors.types import MONGODB_ARCHIVER, resolve_control_system_type

        pairing = pairing_in_rendered_config(self.config.raw)
        if not pairing.is_invented_history:
            return

        # The same section the pairing just judged, read by the same resolver,
        # so the message cannot name a machine other than the refused one.
        control_system_type = resolve_control_system_type(self.config.control_system)

        raise ConfigurationError(
            f"Refusing to start: {self.config.config_path} pairs control_system.type "
            f"{control_system_type!r} with archiver.type "
            f"{pairing.archiver_phrase} — {VA_MOCK_ARCHIVER_WHY} "
            f"Under this file's `archiver:` section, set `type:` to a connector that "
            f"reads a store this deployment actually writes (a project built from the "
            f"control-assistant preset deploys one, and its type reads "
            f"{MONGODB_ARCHIVER!r}); or, if this deployment is meant to be a "
            f"simulation nothing is real in, serve the simulator in process "
            f"(`control_system.connector.virtual_accelerator.serving: in_process`), "
            f"whose synthesized archive claims nothing it cannot back up."
        )

    async def shutdown(self) -> None:
        """Disconnect all connectors. Called on server shutdown."""
        if self._connector_hosts is not None:
            await self._connector_hosts.shutdown()
        for name in list(self._connectors):
            await self.invalidate_connector(name)
        logger.info("ControlSystemContext: shutdown complete")


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors osprey.registry.get_registry())
# ---------------------------------------------------------------------------

_registry: ControlSystemContext | None = None


def get_server_context() -> ControlSystemContext:
    """Get the MCP server registry singleton.

    Raises RuntimeError if initialize_server_context() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError("MCP registry not initialized. Call initialize_server_context() first.")
    return _registry


def initialize_server_context() -> ControlSystemContext:
    """Create and initialize the MCP registry singleton."""
    global _registry
    _registry = ControlSystemContext()
    _registry.initialize()
    return _registry


def reset_server_context() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
