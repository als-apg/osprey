"""Channel Finder In-Context MCP Server Context -- singleton config and database management.

Provides centralized configuration access and channel database lifecycle
management for all in-context channel finder MCP tools.

The server context conditionally imports the appropriate database class based
on the configured database type (``template`` or ``flat``).

Usage in tools:
    from osprey.mcp_server.channel_finder_in_context.server_context import (
        get_cf_ic_context,
    )

    ctx = get_cf_ic_context()
    db = ctx.database
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from osprey.services.channel_finder.core.base_database import BaseDatabase

logger = logging.getLogger("osprey.mcp_server.channel_finder_in_context.server_context")


class ChannelFinderICContext:
    """Singleton registry for in-context channel finder MCP server state.

    Responsibilities:
      1. Load and cache config.yml once at startup
      2. Parse channel_finder.pipelines.in_context config section
      3. Conditionally load the correct database class (flat or template)
      4. Provide a cached database instance for all tools
    """

    def __init__(self) -> None:
        self._raw_config: dict[str, Any] = {}
        self._database: BaseDatabase | None = None
        self._facility_name: str = "control system"
        self._initialized = False

    def initialize(self) -> None:
        """Load config, select database type, and instantiate the database.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self._raw_config = self._load_config()

        cf_config = self._raw_config.get("channel_finder", {})
        ic_config = cf_config.get("pipelines", {}).get("in_context", {})
        db_config = ic_config.get("database", {})
        db_path = db_config.get("path")
        db_type = db_config.get("type", "template")

        if db_path:
            db_path = self._resolve_path(db_path)

            if db_type == "template":
                from osprey.services.channel_finder.databases.template import (
                    ChannelDatabase,
                )
            else:
                from osprey.services.channel_finder.databases.flat import (
                    ChannelDatabase,
                )

            self._database = ChannelDatabase(db_path)
            logger.info(
                "ChannelFinderICContext: loaded %s database from %s (%d channels)",
                db_type,
                db_path,
                len(self._database.get_all_channels()),
            )
        else:
            logger.warning(
                "No database path configured at "
                "channel_finder.pipelines.in_context.database.path -- "
                "channel finder tools will fail until config is provided"
            )

        facility = self._raw_config.get("facility", {})
        self._facility_name = facility.get("name", "control system")

        self._initialized = True
        logger.info("ChannelFinderICContext: initialized")

    @property
    def database(self) -> BaseDatabase:
        """The loaded channel database instance.

        Raises:
            RuntimeError: If no database has been configured/loaded.
        """
        if self._database is None:
            raise RuntimeError(
                "Channel database not available. Check that config.yml has "
                "channel_finder.pipelines.in_context.database.path configured."
            )
        return self._database

    @property
    def facility_name(self) -> str:
        """Name of the facility from config."""
        return self._facility_name

    @property
    def raw_config(self) -> dict[str, Any]:
        """Full raw config dict."""
        return self._raw_config

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load config.yml from OSPREY_CONFIG env var or cwd."""
        config_path = Path(
            os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
        )
        raw: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            logger.info("ChannelFinderICContext: config loaded from %s", config_path)
        else:
            logger.warning("Config file not found: %s", config_path)

        return raw

    @staticmethod
    def _resolve_path(path_str: str) -> str:
        """Resolve path relative to config file directory."""
        config_path = Path(
            os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
        )
        p = Path(path_str)
        if not p.is_absolute():
            p = config_path.parent / p
        return str(p.resolve())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ChannelFinderICContext | None = None


def get_cf_ic_context() -> ChannelFinderICContext:
    """Get the channel finder IC MCP registry singleton.

    Raises RuntimeError if initialize_cf_ic_context() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError(
            "Channel Finder IC MCP registry not initialized. Call initialize_cf_ic_context() first."
        )
    return _registry


def initialize_cf_ic_context() -> ChannelFinderICContext:
    """Create and initialize the channel finder IC MCP registry singleton."""
    global _registry
    _registry = ChannelFinderICContext()
    _registry.initialize()
    return _registry


def reset_cf_ic_context() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
