"""Middle Layer Channel Finder MCP Registry — singleton config and database management.

Provides centralized configuration access and MiddleLayerDatabase lifecycle
management for all Middle Layer channel finder MCP tools.

Usage in tools:
    from osprey.mcp_server.channel_finder_middle_layer.server_context import get_cf_ml_context

    registry = get_cf_ml_context()
    systems = registry.database.list_systems()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from osprey.mcp_server.channel_finder_common import load_cf_config, resolve_cf_path

if TYPE_CHECKING:
    from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.server_context")


class ChannelFinderMLContext:
    """Singleton registry for Middle Layer channel finder MCP server state.

    Responsibilities:
      1. Load and cache config.yml once at startup
      2. Parse channel_finder.pipelines.middle_layer config section
      3. Provide a cached MiddleLayerDatabase instance
      4. Expose facility name for tool descriptions
    """

    def __init__(self) -> None:
        self._raw_config: dict[str, Any] = {}
        self._database: MiddleLayerDatabase | None = None
        self._facility_name: str = "control system"
        self._duckdb_path: str | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Load config and initialize the database.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self._raw_config = load_cf_config(logger)

        cf_config = self._raw_config.get("channel_finder", {})
        ml_config = cf_config.get("pipelines", {}).get("middle_layer", {})
        db_config = ml_config.get("database", {})
        db_path = db_config.get("path")

        if db_path:
            db_path = resolve_cf_path(db_path)

            from osprey.services.channel_finder.databases.middle_layer import (
                MiddleLayerDatabase,
            )

            self._database = MiddleLayerDatabase(db_path)
            logger.info("ChannelFinderMLContext: database loaded from %s", db_path)
        else:
            logger.warning(
                "No database path configured at "
                "channel_finder.pipelines.middle_layer.database.path — "
                "channel finder tools will fail until config is provided"
            )

        duckdb_path = db_config.get("duckdb_path")
        if duckdb_path:
            self._duckdb_path = resolve_cf_path(duckdb_path)
            logger.info("ChannelFinderMLContext: DuckDB path configured at %s", self._duckdb_path)

        facility = self._raw_config.get("facility", {})
        self._facility_name = facility.get("name", "control system")

        self._initialized = True
        logger.info("ChannelFinderMLContext: initialized")

    @property
    def database(self) -> MiddleLayerDatabase:
        """Get the MiddleLayerDatabase instance.

        Raises:
            RuntimeError: If the database is not configured.
        """
        if self._database is None:
            raise RuntimeError(
                "Channel finder database not configured. Check that config.yml "
                "has channel_finder.pipelines.middle_layer.database.path set."
            )
        return self._database

    @property
    def facility_name(self) -> str:
        """Facility name from config (e.g. 'ALS')."""
        return self._facility_name

    @property
    def duckdb_path(self) -> str | None:
        """Path to the DuckDB channel database, or None if not configured."""
        return self._duckdb_path


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ChannelFinderMLContext | None = None


def get_cf_ml_context() -> ChannelFinderMLContext:
    """Get the Channel Finder ML MCP registry singleton.

    Raises RuntimeError if initialize_cf_ml_context() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError(
            "Channel Finder ML MCP registry not initialized. Call initialize_cf_ml_context() first."
        )
    return _registry


def initialize_cf_ml_context() -> ChannelFinderMLContext:
    """Create and initialize the Channel Finder ML MCP registry singleton."""
    global _registry
    _registry = ChannelFinderMLContext()
    _registry.initialize()
    return _registry


def reset_cf_ml_context() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
