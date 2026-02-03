"""ARIEL ingestion adapter discovery.

This module provides adapter discovery and instantiation from configuration.

See 01_DATA_LAYER.md Section 5.12 for specification.
"""

import importlib
from typing import TYPE_CHECKING, cast

from osprey.services.ariel_search.exceptions import AdapterNotFoundError
from osprey.services.ariel_search.ingestion.base import BaseAdapter

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig


# Known adapters: (module_path, class_name)
# MVP: Hardcoded list. V2 may add plugin discovery.
KNOWN_ADAPTERS: dict[str, tuple[str, str]] = {
    "als_logbook": (
        "osprey.services.ariel_search.ingestion.adapters.als",
        "ALSLogbookAdapter",
    ),
    "jlab_logbook": (
        "osprey.services.ariel_search.ingestion.adapters.jlab",
        "JLabLogbookAdapter",
    ),
    "ornl_logbook": (
        "osprey.services.ariel_search.ingestion.adapters.ornl",
        "ORNLLogbookAdapter",
    ),
    "generic_json": (
        "osprey.services.ariel_search.ingestion.adapters.generic",
        "GenericJSONAdapter",
    ),
}


def get_adapter(config: "ARIELConfig") -> BaseAdapter:
    """Load adapter based on configuration.

    Args:
        config: ARIEL configuration with ingestion.adapter set

    Returns:
        Instantiated adapter

    Raises:
        AdapterNotFoundError: If adapter name not recognized
    """
    if not config.ingestion:
        raise AdapterNotFoundError(
            "No ingestion configuration found. Set ariel.ingestion.adapter in config.yml",
            adapter_name="(none)",
            available_adapters=list(KNOWN_ADAPTERS.keys()),
        )

    adapter_name = config.ingestion.adapter
    if adapter_name not in KNOWN_ADAPTERS:
        raise AdapterNotFoundError(
            f"Unknown adapter '{adapter_name}'",
            adapter_name=adapter_name,
            available_adapters=list(KNOWN_ADAPTERS.keys()),
        )

    module_path, class_name = KNOWN_ADAPTERS[adapter_name]

    try:
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)
        return cast(BaseAdapter, adapter_class(config))
    except (ImportError, AttributeError) as e:
        raise AdapterNotFoundError(
            f"Failed to load adapter '{adapter_name}': {e}",
            adapter_name=adapter_name,
            available_adapters=list(KNOWN_ADAPTERS.keys()),
        ) from e
