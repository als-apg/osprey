"""ARIEL enhancement module factory.

This module provides factory functions for creating enhancement modules.

See 01_DATA_LAYER.md Section 6.2.1 for specification.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule


def create_enhancers_from_config(
    config: "ARIELConfig",
) -> list["BaseEnhancementModule"]:
    """Create enhancement module instances for enabled modules in execution order.

    Uses the central Osprey registry for module discovery with explicit
    execution ordering.

    Follows Osprey's factory pattern:
    - Zero-argument instantiation
    - Optional configure() for module-specific settings
    - Lazy loading of expensive resources

    Args:
        config: ARIEL configuration with enhancement_modules settings

    Returns:
        List of configured enhancement module instances, in execution order
    """
    from osprey.registry import get_registry

    registry = get_registry()
    ordered_names = registry.list_ariel_enhancement_modules()
    enhancers: list[BaseEnhancementModule] = []
    for name in ordered_names:
        if not config.is_enhancement_module_enabled(name):
            continue
        result = registry.get_ariel_enhancement_module(name)
        if result is None:
            continue
        cls, _reg = result
        enhancer = cls()
        if hasattr(enhancer, "configure"):
            module_config = config.get_enhancement_module_config(name)
            if module_config:
                enhancer.configure(module_config)
        enhancers.append(enhancer)
    return enhancers


def get_enhancer_names() -> list[str]:
    """Return list of available enhancer names.

    Returns:
        List of enhancer names in execution order
    """
    from osprey.registry import get_registry

    registry = get_registry()
    return registry.list_ariel_enhancement_modules()
