"""ARIEL ingestion adapter discovery.

This module provides adapter discovery and instantiation from the Osprey registry.

See 01_DATA_LAYER.md Section 5.12 for specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from osprey.services.ariel_search.exceptions import AdapterNotFoundError
from osprey.services.ariel_search.ingestion.base import BaseAdapter

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig


def get_adapter(config: ARIELConfig) -> BaseAdapter:
    """Load adapter based on configuration using the Osprey registry.

    Looks up the adapter class from the central Osprey registry, which supports
    framework defaults and user-registered custom adapters.

    Args:
        config: ARIEL configuration with ingestion.adapter set

    Returns:
        Instantiated adapter

    Raises:
        AdapterNotFoundError: If adapter name not recognized
    """
    from osprey.registry import get_registry

    registry = get_registry()

    if not config.ingestion:
        raise AdapterNotFoundError(
            "No ingestion configuration found. Set ariel.ingestion.adapter in config.yml",
            adapter_name="(none)",
            available_adapters=registry.list_ariel_ingestion_adapters(),
        )

    adapter_name = config.ingestion.adapter
    result = registry.get_ariel_ingestion_adapter(adapter_name)

    if result is None:
        raise AdapterNotFoundError(
            f"Unknown adapter '{adapter_name}'",
            adapter_name=adapter_name,
            available_adapters=registry.list_ariel_ingestion_adapters(),
        )

    adapter_class, _registration = result
    return cast(BaseAdapter, adapter_class(config))
