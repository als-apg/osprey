"""MCP Server Context — singleton config and connector management.

Provides centralized configuration access and connector lifecycle
management for all MCP tools. Mirrors the RegistryManager pattern
from the main OSPREY framework, adapted for the simpler MCP context.

Usage in tools:
    from osprey.mcp_server.control_system.server_context import get_server_context

    registry = get_server_context()
    config = registry.config                          # Full parsed config
    connector = await registry.control_system()       # Cached connector
    archiver = await registry.archiver()              # Cached connector
    channel_finder_cfg = registry.channel_finder_config()
    hosts = registry.connector_hosts                  # Connector-host supervisor

The connector-host supervisor is this module's public face on
:mod:`osprey.mcp_server.control_system.connector_host_manager`, which owns the
child process, the session target and the target switch. It lives in a sibling
module because a process supervisor and a config cache have nothing to do with
each other beyond the config, and the switch algorithm is long enough that
folding it in here would bury both.

Where a tool's connector comes from
-----------------------------------
There are two serving paths, and which one a deployment gets is decided once,
by :func:`~osprey.mcp_server.control_system.connector_host_manager.switch_capable`:

* **Switch-capable** (at least two targets configured, and the deployment's own
  control system is one of them): ``control_system()`` returns the proxy onto the
  connector-host child. The in-process connector is never built — this server
  holds no control-system client library at all, which is the only way a target
  switch can actually move where tool calls land. A child that has died is not
  papered over: the accessor raises
  :class:`~osprey.mcp_server.control_system.connector_host_manager.NoConnectorHostError`
  and every control-system-routed operation refuses with that reason until one
  is running again.
* **Everything else** — a mock deployment, a single-target EPICS deployment,
  any config that never named a second target — keeps the in-process cached
  connector, unchanged and untouched by any of this.

The archiver is on neither path: it speaks HTTP or pymongo, holds no Channel
Access context, and keeps serving whatever is happening to the child.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.errors import ConfigurationError
from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
    NoConnectorHostError,
    SwitchError,
    baseline_target,
    switch_capable,
)
from osprey_connectors.ipc.proxy import ConnectorHostProxy
from osprey_connectors.types import configured_targets, target_writes_enabled

__all__ = [
    "ConnectorEntry",
    "ConnectorHostManager",
    "ConnectorHostProxy",
    "ControlSystemContext",
    "MCPServerConfig",
    "NoConnectorHostError",
    "SwitchError",
    "get_server_context",
    "initialize_server_context",
    "reset_server_context",
]

logger = logging.getLogger("osprey.mcp_server.control_system.server_context")


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

    @property
    def control_system(self) -> dict[str, Any]:
        return self.raw.get("control_system", {})

    @property
    def archiver(self) -> dict[str, Any]:
        return self.raw.get("archiver", {})

    @property
    def channel_finder(self) -> dict[str, Any]:
        return self.raw.get("channel_finder", {})

    @property
    def ariel(self) -> dict[str, Any]:
        return self.raw.get("ariel", {})

    @property
    def writes_enabled(self) -> bool:
        return self.control_system.get("writes_enabled", False)


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
            "writes_by_target=%s, serving=%s)",
            self._config.control_system.get("type", "not configured"),
            self._config.archiver.get("type", "not configured"),
            # The targets this config configures, not every target the
            # connectors package names: a log line over the constant would
            # report a posture for a `standin` soft IOC this deployment never
            # stood up, answered out of the deployment-wide key by
            # `target_writes_enabled`'s unresolvable-target fallback.
            {
                target: target_writes_enabled(self._config.control_system, target)
                for target in configured_targets(self._config.control_system)
            },
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

    async def control_system(self) -> ControlSystemConnector | ConnectorHostProxy:
        """The control-system connector every tool reads and writes through.

        On a switch-capable deployment this is the **proxy onto the
        connector-host child**, which is what makes a target switch reach the
        tools at all: the child holds the Channel Access context for the
        session's target, and a switch replaces the child. On every other
        deployment it is the in-process connector, cached exactly as before.

        The two are not related by inheritance — the proxy mirrors the
        connector's call surface deliberately rather than subclassing it (see
        :class:`~osprey_connectors.ipc.proxy.ConnectorHostProxy`) — so the
        return type names both.
        """
        return await self._get_connector("control_system")

    async def archiver(self) -> ArchiverConnector:
        """Get or create the cached archiver connector.

        Never routed through a connector host: the archiver speaks HTTP or
        pymongo, holds no Channel Access context, and therefore keeps serving
        while no child is alive.
        """
        return await self._get_connector("archiver")

    async def _get_connector(self, name: str) -> Any:
        """Lazy-create and cache a connector, reconnecting on failure."""
        entry = self._connectors.get(name)
        if entry is None:
            raise ValueError(f"Unknown connector: {name}")

        if name == "control_system" and self.switch_capable:
            return await self._connector_host()

        if entry.instance is not None:
            return entry.instance

        from osprey.connectors.factory import ConnectorFactory

        if name == "control_system":
            # A deployment on this path serves one target and never switches, so
            # the target it is on is the deployment's own baseline. Naming it
            # rather than leaving the stamp blank is what makes the rebuild after
            # invalidate_connector() carry the same session posture the instance
            # it replaces was reading. Derived from config through the same
            # function the supervisor's own `baseline` property uses, rather than
            # through the supervisor: asking this path for one would construct a
            # supervisor a non-switch-capable deployment must never have.
            entry.instance = await ConnectorFactory.create_control_system_connector(
                entry.config, control_target=baseline_target(self.config.raw)
            )
        elif name == "archiver":
            entry.instance = await ConnectorFactory.create_archiver_connector(entry.config)

        logger.info("ControlSystemContext: created %s connector", name)
        return entry.instance

    async def _connector_host(self) -> ConnectorHostProxy:
        """The live child's proxy, or the no-child refusal.

        The in-process connector is never built on this path — building one
        would load a control-system client into the server that is supposed to
        hold none, and pin it to whatever target the config happened to
        describe at start, which is precisely the bug a switch has to avoid.
        """
        manager = self.connector_hosts
        await manager.ensure_started()
        proxy = manager.active_proxy()
        if proxy is None:
            raise NoConnectorHostError(manager.active_target(), manager.active_generation())
        return proxy

    async def invalidate_connector(self, name: str) -> bool:
        """Disconnect and remove a cached connector (e.g., on error).

        The next call to control_system() or archiver() will recreate it.

        When the control system is being served from a connector-host child,
        "drop the instance so the next call rebuilds it" is a kill and a
        respawn of that child on the *same* target — the generation does not
        move, because the session is still pointed where it was. A deployment
        with no running child takes the in-process path below, unchanged.

        The second caller is the session-control reconciler
        (:mod:`osprey.mcp_server.control_system.session_control`), which rebuilds
        the connector when the operator narrows the posture of the target the
        session is on: the child connected on a gateway role chosen under the
        old posture, and only a rebuild moves it. That caller deliberately takes
        no lock of its own — the one below is the lock, and a second one around
        the same operation is how two things that must agree stop agreeing.

        Returns:
            Whether the connector this call was asked about is now gone or
            replaced. ``False`` means one specific thing: the connector-host
            respawn was attempted and refused, so the OLD child is still
            serving and whatever the caller invalidated it for has not
            happened. The reconciler needs that answer — a realignment reported
            as done on a child that never restarted would tell an operator
            their read-only toggle had taken effect when it had not — and the
            error paths that invalidate on a ``ConnectionError`` are free to
            ignore it, because their next call rebuilds regardless.
        """
        if (
            name == "control_system"
            and self._connector_hosts is not None
            and self._connector_hosts.is_started()
        ):
            try:
                await self._connector_hosts.respawn_same_target()
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
        if entry and entry.instance:
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
        from osprey.utils.workspace import load_osprey_config, resolve_config_path

        raw = load_osprey_config()
        config_path = resolve_config_path()
        if not config_path.exists():
            logger.warning("Config file not found: %s", config_path)

        return MCPServerConfig(raw=raw, config_path=config_path)

    @staticmethod
    def _register_connector_types() -> None:
        """Register all connector types with ConnectorFactory."""
        from osprey.connectors.factory import register_builtin_connectors

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

        from osprey.connectors.factory import ConnectorFactory

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

        Judged by :func:`~osprey.connectors.honesty.pairing_in_rendered_config`,
        which resolves both keys through *nested sections only* — exactly as
        :attr:`MCPServerConfig.control_system` and :attr:`MCPServerConfig.archiver`
        do a few lines above, and exactly as the factory then reads what they
        hand it. A guard that resolved a config differently from the reader it
        guards would not be a guard; the divergence would be the way through.

        The refused machine is *named* rather than assumed: the pairing covers
        every type in :data:`~osprey.connectors.types.INVENTED_HISTORY_TYPES`,
        so a deployment baselined on the live stand-in is told about the
        stand-in and not sent looking for a virtual accelerator its
        ``control_system:`` section does not name.
        """
        from osprey.connectors.honesty import VA_MOCK_ARCHIVER_WHY, pairing_in_rendered_config
        from osprey.connectors.types import MOCK, MONGODB_ARCHIVER, resolve_control_system_type

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
            f"simulation nothing is real in, set the `type:` under `control_system:` "
            f"to {MOCK!r} — a mock machine with a mock archive claims nothing it "
            f"cannot back up."
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
