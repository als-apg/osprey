"""ARIEL enhancement module factory.

This module provides factory functions for creating enhancement modules.

See 01_DATA_LAYER.md Section 6.2.1 for specification.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from osprey.services.ariel_search.exceptions import ConfigurationError

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule


# Known enhancement modules: (module_path, class_name)
# MVP: Hardcoded list. V2 may add plugin discovery.
KNOWN_ENHANCERS: dict[str, tuple[str, str]] = {
    "semantic_processor": (
        "osprey.services.ariel_search.enhancement.semantic_processor.processor",
        "SemanticProcessorModule",
    ),
    "text_embedding": (
        "osprey.services.ariel_search.enhancement.text_embedding.embedder",
        "TextEmbeddingModule",
    ),
}

# Fixed execution order (not alphabetical)
# Rationale: semantic_processor extracts keywords before text_embedding generates vectors
EXECUTION_ORDER = ["semantic_processor", "text_embedding"]


def create_enhancers_from_config(
    config: "ARIELConfig",
) -> list["BaseEnhancementModule"]:
    """Create enhancement module instances for enabled modules in execution order.

    Follows Osprey's factory pattern:
    - Zero-argument instantiation
    - Optional configure() for module-specific settings
    - Lazy loading of expensive resources

    Args:
        config: ARIEL configuration with enhancement_modules settings

    Returns:
        List of configured enhancement module instances, in execution order

    Raises:
        ConfigurationError: If an enabled module is not in KNOWN_ENHANCERS
    """
    enhancers: list[BaseEnhancementModule] = []

    for name in EXECUTION_ORDER:
        # Skip if module not enabled in config
        if not config.is_enhancement_module_enabled(name):
            continue

        # Check if module is known
        if name not in KNOWN_ENHANCERS:
            raise ConfigurationError(
                config_key=f"ariel.enhancement_modules.{name}",
                message=f"Unknown enhancement module: {name}",
            )

        # Import and instantiate (zero-argument constructor)
        module_path, class_name = KNOWN_ENHANCERS[name]
        module = import_module(module_path)
        enhancer_class = getattr(module, class_name)
        enhancer = enhancer_class()

        # Configure if module accepts configuration
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
    return list(EXECUTION_ORDER)
