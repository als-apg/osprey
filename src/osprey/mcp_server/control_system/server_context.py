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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector

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

        # 4. Validate config (warnings, not fatal)
        self._validate()

        self._initialized = True
        logger.info(
            "ControlSystemContext: initialized (control_system=%s, archiver=%s, writes=%s)",
            self._config.control_system.get("type", "not configured"),
            self._config.archiver.get("type", "not configured"),
            self._config.writes_enabled,
        )

    @property
    def config(self) -> MCPServerConfig:
        """Full parsed configuration."""
        if self._config is None:
            raise RuntimeError("ControlSystemContext not initialized — call initialize() first")
        return self._config

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

    async def control_system(self) -> ControlSystemConnector:
        """Get or create the cached control-system connector."""
        return await self._get_connector("control_system")

    async def archiver(self) -> ArchiverConnector:
        """Get or create the cached archiver connector."""
        return await self._get_connector("archiver")

    async def _get_connector(self, name: str) -> Any:
        """Lazy-create and cache a connector, reconnecting on failure."""
        entry = self._connectors.get(name)
        if entry is None:
            raise ValueError(f"Unknown connector: {name}")

        if entry.instance is not None:
            return entry.instance

        from osprey.connectors.factory import ConnectorFactory

        if name == "control_system":
            entry.instance = await ConnectorFactory.create_control_system_connector(entry.config)
        elif name == "archiver":
            entry.instance = await ConnectorFactory.create_archiver_connector(entry.config)

        logger.info("ControlSystemContext: created %s connector", name)
        return entry.instance

    async def invalidate_connector(self, name: str) -> None:
        """Disconnect and remove a cached connector (e.g., on error).

        The next call to control_system() or archiver() will recreate it.
        """
        entry = self._connectors.get(name)
        if entry and entry.instance:
            try:
                await entry.instance.disconnect()
            except Exception:
                logger.debug("Error disconnecting %s (ignored)", name, exc_info=True)
            entry.instance = None
            logger.info("ControlSystemContext: invalidated %s connector", name)

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
        """Emit warnings for common misconfigurations."""
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

    async def shutdown(self) -> None:
        """Disconnect all connectors. Called on server shutdown."""
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
