"""Hierarchical Channel Finder MCP Registry — singleton config and database management.

Provides centralized configuration access and HierarchicalChannelDatabase lifecycle
management for all Hierarchical channel finder MCP tools.

Usage in tools:
    from osprey.mcp_server.channel_finder_hierarchical.registry import get_cf_hier_registry

    registry = get_cf_hier_registry()
    options = registry.database.get_options_at_level("system", {})
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )
    from osprey.services.channel_finder.feedback.store import FeedbackStore

logger = logging.getLogger("osprey.mcp_server.channel_finder_hierarchical.registry")


class ChannelFinderHierRegistry:
    """Singleton registry for Hierarchical channel finder MCP server state.

    Loads config.yml once at startup, provides a cached
    HierarchicalChannelDatabase instance, and exposes the facility name
    for tool descriptions.
    """

    def __init__(self) -> None:
        self._raw_config: dict[str, Any] = {}
        self._database: HierarchicalChannelDatabase | None = None
        self._feedback_store: FeedbackStore | None = None
        self._facility_name: str = "control system"
        self._initialized = False

    def initialize(self) -> None:
        """Load config and initialize the database.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self._raw_config = self._load_config()

        cf_config = self._raw_config.get("channel_finder", {})
        hier_config = cf_config.get("pipelines", {}).get("hierarchical", {})
        db_config = hier_config.get("database", {})
        db_path = db_config.get("path")

        if db_path:
            db_path = self._resolve_path(db_path)

            from osprey.services.channel_finder.databases.hierarchical import (
                HierarchicalChannelDatabase,
            )

            self._database = HierarchicalChannelDatabase(db_path)
            logger.info("ChannelFinderHierRegistry: database loaded from %s", db_path)
        else:
            logger.warning(
                "No database path configured at "
                "channel_finder.pipelines.hierarchical.database.path — "
                "channel finder tools will fail until config is provided"
            )

        # Initialize feedback store if configured and enabled
        feedback_config = hier_config.get("feedback", {})
        if feedback_config.get("enabled", False) and feedback_config.get("store_path"):
            try:
                from osprey.services.channel_finder.feedback.store import FeedbackStore

                store_path = self._resolve_path(feedback_config["store_path"])
                self._feedback_store = FeedbackStore(store_path)
                logger.info("ChannelFinderHierRegistry: feedback store loaded from %s", store_path)
            except Exception:
                logger.warning(
                    "ChannelFinderHierRegistry: failed to initialize feedback store",
                    exc_info=True,
                )

        facility = self._raw_config.get("facility", {})
        self._facility_name = facility.get("name", "control system")

        self._initialized = True
        logger.info("ChannelFinderHierRegistry: initialized")

    @property
    def database(self) -> HierarchicalChannelDatabase:
        """Get the HierarchicalChannelDatabase instance.

        Raises:
            RuntimeError: If the database is not configured.
        """
        if self._database is None:
            raise RuntimeError(
                "Channel finder database not configured. Check that config.yml "
                "has channel_finder.pipelines.hierarchical.database.path set."
            )
        return self._database

    @property
    def facility_name(self) -> str:
        """Facility name from config (e.g. 'ALS')."""
        return self._facility_name

    @property
    def feedback_store(self) -> FeedbackStore | None:
        """FeedbackStore instance, or None if not configured."""
        return self._feedback_store

    def _resolve_path(self, path_str: str) -> str:
        """Resolve path relative to config file directory."""
        config_path = Path(
            os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
        )
        p = Path(path_str)
        if not p.is_absolute():
            p = config_path.parent / p
        return str(p.resolve())

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
            logger.info("ChannelFinderHierRegistry: config loaded from %s", config_path)
        else:
            logger.warning("Config file not found: %s", config_path)

        return raw


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ChannelFinderHierRegistry | None = None


def get_cf_hier_registry() -> ChannelFinderHierRegistry:
    """Get the Channel Finder Hierarchical MCP registry singleton.

    Raises RuntimeError if initialize_cf_hier_registry() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError(
            "Channel Finder Hierarchical MCP registry not initialized. "
            "Call initialize_cf_hier_registry() first."
        )
    return _registry


def initialize_cf_hier_registry() -> ChannelFinderHierRegistry:
    """Create and initialize the Channel Finder Hierarchical MCP registry singleton."""
    global _registry
    _registry = ChannelFinderHierRegistry()
    _registry.initialize()
    return _registry


def reset_cf_hier_registry() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
