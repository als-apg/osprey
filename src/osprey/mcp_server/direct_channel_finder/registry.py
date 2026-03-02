"""Direct Channel Finder MCP Registry -- singleton config and backend management.

Provides centralized configuration access and PV info backend lifecycle
management for all direct channel finder MCP tools.

Usage in tools:
    from osprey.mcp_server.direct_channel_finder.registry import (
        get_dcf_registry,
    )

    registry = get_dcf_registry()
    backend = registry.backend
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from osprey.services.channel_finder.backends.base import PVInfoBackend

logger = logging.getLogger("osprey.mcp_server.direct_channel_finder.registry")


class DirectChannelFinderRegistry:
    """Singleton registry for direct channel finder MCP server state.

    Responsibilities:
      1. Load and cache config.yml once at startup
      2. Parse channel_finder.direct config section
      3. Instantiate the configured PV info backend
      4. Provide a cached backend instance for all tools
    """

    def __init__(self) -> None:
        self._raw_config: dict[str, Any] = {}
        self._backend: PVInfoBackend | None = None
        self._facility_name: str = "control system"
        self._initialized = False

    def initialize(self) -> None:
        """Load config, select backend, and instantiate it.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self._raw_config = self._load_config()

        cf_config = self._raw_config.get("channel_finder", {})
        direct_config = cf_config.get("direct", {})
        backend_type = direct_config.get("backend", "mock")

        if backend_type == "mock":
            from osprey.services.channel_finder.backends.mock import MockPVInfoBackend

            self._backend = MockPVInfoBackend()
            logger.info(
                "DirectChannelFinderRegistry: using MockPVInfoBackend (%d PVs)",
                self._backend.total_pv_count,
            )
        elif backend_type == "als_channel_finder":
            from osprey.services.channel_finder.backends.als_channel_finder import (
                ALSChannelFinderBackend,
            )

            backend_url = direct_config.get("backend_url", "https://localhost:8443/ChannelFinder")
            self._backend = ALSChannelFinderBackend(backend_url)
            logger.info(
                "DirectChannelFinderRegistry: using ALSChannelFinderBackend at %s",
                backend_url,
            )
        else:
            logger.warning(
                "Unknown backend type %r — only 'mock' is currently supported. "
                "Direct channel finder tools will fail.",
                backend_type,
            )

        facility = self._raw_config.get("facility", {})
        self._facility_name = facility.get("name", "control system")

        self._initialized = True
        logger.info("DirectChannelFinderRegistry: initialized")

    @property
    def backend(self) -> PVInfoBackend:
        """The loaded PV info backend instance.

        Raises:
            RuntimeError: If no backend has been configured/loaded.
        """
        if self._backend is None:
            raise RuntimeError(
                "PV info backend not available. Check that config.yml has "
                "channel_finder.direct.backend configured."
            )
        return self._backend

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
            logger.info("DirectChannelFinderRegistry: config loaded from %s", config_path)
        else:
            logger.warning("Config file not found: %s", config_path)

        return raw


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: DirectChannelFinderRegistry | None = None


def get_dcf_registry() -> DirectChannelFinderRegistry:
    """Get the direct channel finder MCP registry singleton.

    Raises RuntimeError if initialize_dcf_registry() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError(
            "Direct Channel Finder MCP registry not initialized. "
            "Call initialize_dcf_registry() first."
        )
    return _registry


def initialize_dcf_registry() -> DirectChannelFinderRegistry:
    """Create and initialize the direct channel finder MCP registry singleton."""
    global _registry
    _registry = DirectChannelFinderRegistry()
    _registry.initialize()
    return _registry


def reset_dcf_registry() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
