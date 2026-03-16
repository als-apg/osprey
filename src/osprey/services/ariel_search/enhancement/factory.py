"""ARIEL enhancement module factory.

This module provides factory functions for creating enhancement modules.
"""

import importlib
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule

logger = logging.getLogger(__name__)


def _get_ordered_registrations() -> list:
    """Get enhancement module registrations sorted by execution_order.

    Reads directly from ``registry.config.ariel_enhancement_modules``,
    which is populated at ``RegistryManager`` construction time and does
    NOT require ``initialize()`` to have been called.
    """
    from osprey.registry import get_registry

    registry = get_registry()
    registrations = list(registry.config.ariel_enhancement_modules)
    registrations.sort(key=lambda r: r.execution_order)
    return registrations


def create_enhancers_from_config(
    config: "ARIELConfig",
) -> list["BaseEnhancementModule"]:
    """Create enhancement module instances for enabled modules in execution order.

    Uses ``registry.config.ariel_enhancement_modules`` for module discovery,
    which is available without calling ``registry.initialize()``.

    Args:
        config: ARIEL configuration with enhancement_modules settings

    Returns:
        List of configured enhancement module instances, in execution order
    """
    enhancers: list[BaseEnhancementModule] = []
    for reg in _get_ordered_registrations():
        if not config.is_enhancement_module_enabled(reg.name):
            continue
        try:
            module = importlib.import_module(reg.module_path)
            cls = getattr(module, reg.class_name)
        except (ImportError, AttributeError) as e:
            logger.warning(
                f"Skipping enhancement module '{reg.name}' (import failed): {e}"
            )
            continue
        enhancer = cls()
        if hasattr(enhancer, "configure"):
            module_config = config.get_enhancement_module_config(reg.name)
            if module_config:
                enhancer.configure(module_config)
        enhancers.append(enhancer)
    return enhancers


def get_enhancer_names() -> list[str]:
    """Return list of available enhancer names.

    Returns:
        List of enhancer names in execution order
    """
    return [reg.name for reg in _get_ordered_registrations()]
