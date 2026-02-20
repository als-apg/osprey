"""ARIEL capability support - service factory and helpers.

This module provides utilities for the LogbookSearchCapability to interface
with the ARIELSearchService.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.utils.config import get_config_value

if TYPE_CHECKING:
    from osprey.services.ariel_search.service import ARIELSearchService

_ariel_service_instance: ARIELSearchService | None = None


async def get_ariel_search_service() -> ARIELSearchService:
    """Get or create the ARIEL search service singleton.

    Lazily initializes the ARIELSearchService from configuration.
    The service is created once and reused for subsequent calls.

    Returns:
        ARIELSearchService instance, lazily initialized from config.

    Raises:
        ConfigurationError: If ARIEL is not configured in config.yml.
    """
    global _ariel_service_instance

    if _ariel_service_instance is None:
        from osprey.services.ariel_search import ARIELConfig, ConfigurationError
        from osprey.services.ariel_search.service import create_ariel_service

        config = get_config_value("ariel", {})

        if not config:
            raise ConfigurationError(
                "ARIEL not configured. Add 'ariel:' section to config.yml",
                config_key="ariel",
            )

        ariel_config = ARIELConfig.from_dict(config)
        _ariel_service_instance = await create_ariel_service(ariel_config)

    return _ariel_service_instance


def reset_ariel_service() -> None:
    """Reset the service singleton (for testing).

    Note: This does NOT close the service's connection pool.
    Use close_ariel_service() for proper cleanup.
    """
    global _ariel_service_instance
    _ariel_service_instance = None


async def close_ariel_service() -> None:
    """Close and reset the service singleton.

    Properly closes the service's connection pool before resetting.
    Use this in tests to avoid connection leaks.
    """
    global _ariel_service_instance
    if _ariel_service_instance is not None:
        await _ariel_service_instance.pool.close()
    _ariel_service_instance = None


__all__ = [
    "get_ariel_search_service",
    "reset_ariel_service",
    "close_ariel_service",
]
