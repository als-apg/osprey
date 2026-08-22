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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.services.ariel_search.exceptions import ConfigurationError
    from osprey.services.ariel_search.service import ARIELSearchService

logger = logging.getLogger("osprey.mcp_server.ariel.server_context")

#: Every ``validate()`` message whose key starts with this prefix is reported as
#: a :class:`VocabularyError`, so the surface that catches it can name the
#: vocabulary remedy rather than a generic one.
_VOCABULARY_KEY_PREFIX = "ariel.vocabulary"


def _config_key_of(error: str) -> str:
    """Extract the dotted config key a validation message leads with.

    Every message this module treats as fatal is formatted by
    :meth:`ARIELConfig.validate` as ``<dotted.key><separator> <detail>``, so the
    first whitespace-delimited token (minus a trailing colon) is the key.

    Args:
        error: A single message from :meth:`ARIELConfig.validate`.

    Returns:
        The leading dotted key, e.g. ``ariel.vocabulary.path``.
    """
    return error.split(" ", 1)[0].rstrip(":")


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
        self._config_path: Path = Path.cwd() / "config.yml"
        self._ariel_config: Any = None  # ARIELConfig
        self._service: ARIELSearchService | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Load config, parse the ARIEL section, and validate it eagerly.

        Called once during create_server(), before any tool module is imported,
        so a raise here refuses server startup. Subsequent calls are no-ops.

        Raises:
            ConfigurationError: The ARIEL section is invalid — the message names
                the failing key (e.g. ``ariel.vocabulary.path``) and the remedy.
                No database is contacted before this point, so a bad vocabulary
                refuses startup even with no database running.
        """
        if self._initialized:
            return

        self._raw_config = self._load_config()

        ariel_section = self._raw_config.get("ariel", {})
        if not ariel_section:
            logger.warning(
                "No 'ariel' section in config.yml — ARIEL tools will fail until an "
                "'ariel' section is added to the build profile (profile.yml on the "
                "host) and the project is rebuilt and redeployed"
            )
            self._initialized = True
            return

        from osprey.services.ariel_search.config import ARIELConfig

        services = self._raw_config.get("services") or {}
        # A relative ariel.vocabulary.path resolves against the directory of the
        # config file this process actually loaded — never the CWD the MCP
        # server happens to be started in.
        config = ARIELConfig.from_dict(
            ariel_section,
            services.get("postgresql") or {},
            config_dir=self._config_path.parent,
        )
        self._validate_or_refuse(config)

        self._ariel_config = config
        self._initialized = True
        logger.info("ARIELContext: initialized")

    @staticmethod
    def _validate_or_refuse(config: Any) -> None:
        """Validate the parsed config and refuse startup on any error.

        The MCP server has no settings editor to keep reachable — unlike the web
        panel, which degrades so an operator can repair the config through it —
        so every error :meth:`ARIELConfig.validate` reports is fatal here.
        Vocabulary errors are reported first and as a :class:`VocabularyError`,
        so the message always names ``ariel.vocabulary.*`` and carries the
        vocabulary remedy even when other keys are wrong too.

        Args:
            config: The freshly parsed ``ARIELConfig``.

        Raises:
            VocabularyError: The vocabulary could not be loaded, or another
                ``ariel.vocabulary.*`` key is invalid.
            ConfigurationError: Any other configuration key is invalid.
        """
        from osprey.services.ariel_search.exceptions import (
            ConfigurationError,
            VocabularyError,
            summarize_errors,
        )

        # Loader errors first: they are the only class that needs no validate()
        # pass to be decisive, and they are what an operator sees most often.
        if config.vocabulary.enabled and config.vocabulary_errors:
            raise ARIELContext._refuse(VocabularyError(config.vocabulary_errors))

        errors = config.validate()
        if not errors:
            return

        vocabulary_errors = [
            error for error in errors if _config_key_of(error).startswith(_VOCABULARY_KEY_PREFIX)
        ]
        if vocabulary_errors:
            raise ARIELContext._refuse(
                VocabularyError(vocabulary_errors, config_key=_config_key_of(vocabulary_errors[0]))
            )

        raise ARIELContext._refuse(
            ConfigurationError(
                summarize_errors(errors, "the ARIEL configuration is not usable"),
                _config_key_of(errors[0]),
                {"errors": errors},
            )
        )

    @staticmethod
    def _refuse(error: ConfigurationError) -> ConfigurationError:
        """Log a startup refusal and hand the exception back to be raised.

        Args:
            error: The ``ConfigurationError`` (or subclass) about to be raised.

        Returns:
            The same exception, so the caller can ``raise`` it.
        """
        remedy = getattr(error, "remedy", "fix the named key in config.yml and restart")
        # Every validate() message already leads with its own dotted key, so the
        # key is not repeated here.
        logger.error("ARIEL MCP server refusing to start: %s. Remedy: %s", error.message, remedy)
        return error

    @property
    def config(self) -> Any:
        """Parsed ARIELConfig instance."""
        if self._ariel_config is None:
            raise RuntimeError(
                "ARIEL config not available. Add an 'ariel' section with database "
                "configuration in the build profile (profile.yml on the host), "
                "then rebuild and redeploy."
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

    def _load_config(self) -> dict[str, Any]:
        """Load config.yml via the shared config loader.

        Also records the resolved config path on the context, so relative paths
        inside the ARIEL section resolve against the file they came from.

        Returns:
            The raw config dict (empty when no config file is reachable).
        """
        from osprey.utils.workspace import load_osprey_config, resolve_config_path

        raw = load_osprey_config()
        config_path = resolve_config_path()
        self._config_path = config_path
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
