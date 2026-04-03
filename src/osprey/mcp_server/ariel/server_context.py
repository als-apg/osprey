"""ARIEL MCP Server Context — singleton config and service management.

Provides centralized configuration access and ARIEL service lifecycle
management for all ARIEL MCP tools.

Usage in tools:
    from osprey.mcp_server.ariel.server_context import get_ariel_context

    ctx = get_ariel_context()
    service = await ctx.service()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.services.ariel_search.service import ARIELSearchService

logger = logging.getLogger("osprey.mcp_server.ariel.server_context")


class ARIELContext:
    """Singleton registry for ARIEL MCP server state.

    Responsibilities:
      1. Load and cache config.yml once at startup
      2. Parse ARIEL config section into ARIELConfig
      3. Provide a cached ARIELSearchService instance (lazy-created)
      4. Manage service lifecycle (pool shutdown)
    """

    def __init__(self) -> None:
        self._raw_config: dict[str, Any] = {}
        self._ariel_config: Any = None  # ARIELConfig
        self._service: ARIELSearchService | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Load config and parse ARIEL section.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self._raw_config = self._load_config()

        ariel_section = self._raw_config.get("ariel", {})
        if not ariel_section:
            logger.warning(
                "No 'ariel' section in config.yml — ARIEL tools will fail until config is provided"
            )
            self._initialized = True
            return

        # Handle config key mismatch: config.yml uses connection_string,
        # but DatabaseConfig.from_dict() expects "uri"
        db_section = ariel_section.get("database", {})
        if "connection_string" in db_section and "uri" not in db_section:
            db_section["uri"] = db_section["connection_string"]

        from osprey.services.ariel_search.config import ARIELConfig

        self._ariel_config = ARIELConfig.from_dict(ariel_section)

        self._initialized = True
        logger.info("ARIELContext: initialized")

    @property
    def config(self) -> Any:
        """Parsed ARIELConfig instance."""
        if self._ariel_config is None:
            raise RuntimeError(
                "ARIEL config not available. Check that config.yml has an "
                "'ariel' section with database configuration."
            )
        return self._ariel_config

    @property
    def raw_config(self) -> dict[str, Any]:
        """Full raw config dict."""
        return self._raw_config

    async def service(self) -> ARIELSearchService:
        """Get or create the cached ARIEL search service.

        The service (and its DB pool) is created lazily on first call
        and reused for all subsequent calls.
        """
        if self._service is not None:
            return self._service

        from osprey.services.ariel_search.service import create_ariel_service

        # Call directly — NOT as context manager. We manage pool shutdown
        # ourselves in shutdown().
        self._service = await create_ariel_service(self.config)
        logger.info("ARIELContext: created ARIEL service")
        return self._service

    async def shutdown(self) -> None:
        """Close the service's DB pool. Called on server shutdown."""
        if self._service is not None:
            try:
                await self._service.pool.close()
            except Exception:
                logger.debug("Error closing ARIEL pool (ignored)", exc_info=True)
            self._service = None
            logger.info("ARIELContext: shutdown complete")

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load config.yml via the shared config loader."""
        from osprey.utils.workspace import load_osprey_config, resolve_config_path

        raw = load_osprey_config()
        config_path = resolve_config_path()
        if config_path.exists():
            logger.info("ARIELContext: config loaded from %s", config_path)
        else:
            logger.warning("Config file not found: %s", config_path)

        return raw


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ARIELContext | None = None


def get_ariel_context() -> ARIELContext:
    """Get the ARIEL MCP registry singleton.

    Raises RuntimeError if initialize_ariel_context() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError(
            "ARIEL MCP registry not initialized. Call initialize_ariel_context() first."
        )
    return _registry


def initialize_ariel_context() -> ARIELContext:
    """Create and initialize the ARIEL MCP registry singleton."""
    global _registry
    _registry = ARIELContext()
    _registry.initialize()
    return _registry


def reset_ariel_context() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
