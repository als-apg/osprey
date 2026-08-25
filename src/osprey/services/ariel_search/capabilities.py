"""ARIEL capabilities assembly.

Builds the capabilities response that the frontend uses to discover
available search modes and their tunable parameters.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.search.base import ParameterDescriptor

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

# Shared parameters available across all modes
SHARED_PARAMETERS = [
    ParameterDescriptor(
        name="max_results",
        label="Max Results",
        description="Maximum number of entries to return",
        param_type="int",
        default=10,
        min_value=1,
        max_value=100,
        step=1,
        section="General",
    ),
    ParameterDescriptor(
        name="start_date",
        label="Start Date",
        description="Filter entries after this date",
        param_type="date",
        default=None,
        section="Filters",
    ),
    ParameterDescriptor(
        name="end_date",
        label="End Date",
        description="Filter entries before this date",
        param_type="date",
        default=None,
        section="Filters",
    ),
    ParameterDescriptor(
        name="author",
        label="Author",
        description="Filter entries by author name",
        param_type="text",
        default=None,
        placeholder="Filter by author...",
        section="Filters",
    ),
    ParameterDescriptor(
        name="source_system",
        label="Source System",
        description="Filter entries by source system",
        param_type="dynamic_select",
        default=None,
        options_endpoint="/api/filter-options/source_systems",
        section="Filters",
    ),
]


def shared_parameters(config: ARIELConfig) -> list[ParameterDescriptor]:
    """Return the shared parameter descriptors for this configuration.

    The five filter/limit descriptors are always present. ``expand_query`` is
    appended only when a vocabulary is configured — a facility that ships no
    vocabulary gets no toggle to click and no capability entry to explain.

    Args:
        config: ARIEL configuration.

    Returns:
        The shared descriptors, in advertised order.
    """
    parameters = [*SHARED_PARAMETERS]
    if config.vocabulary.enabled:
        parameters.append(
            ParameterDescriptor(
                name="expand_query",
                label="Expand vocabulary",
                description=(
                    "Expand facility shorthand and acronyms using the configured vocabulary"
                ),
                param_type="bool",
                default=config.vocabulary.expand_by_default,
                section="General",
            )
        )
    return parameters


def get_capabilities(config: ARIELConfig) -> dict[str, Any]:
    """Build the capabilities response for the frontend.

    Iterates enabled search modules (keyword, semantic) as "direct" modes.
    Collects parameter descriptors from each.

    Args:
        config: ARIEL configuration

    Returns:
        Dict matching the capabilities response schema:
        {
            "categories": {
                "direct": {"label": "Direct", "modes": [...]},
            },
            "default_mode": "hybrid",
            "shared_parameters": [...],
            "vocabulary": {"enabled": ..., "concepts": ..., "expand_by_default": ...},
        }
    """
    categories: dict[str, dict[str, Any]] = {
        "direct": {"label": "Direct", "modes": []},
    }

    _add_search_modules(config, categories)

    enabled = bool(config.vocabulary.enabled)
    return {
        "categories": categories,
        "default_mode": config.resolve_default_search_mode(),
        "shared_parameters": [p.to_dict() for p in shared_parameters(config)],
        "vocabulary": {
            "enabled": enabled,
            "concepts": config.loaded_vocabulary.concept_count if config.loaded_vocabulary else 0,
            "expand_by_default": config.vocabulary.expand_by_default if enabled else False,
        },
    }


def _add_search_modules(
    config: ARIELConfig,
    categories: dict[str, dict[str, Any]],
) -> None:
    """Add enabled search modules to the capabilities via the registry."""
    from osprey.registry import get_registry

    registry = get_registry()
    for name in registry.list_ariel_search_modules():
        if not config.is_search_module_enabled(name):
            continue
        module = registry.get_ariel_search_module(name)
        if module is None:
            continue
        descriptor = module.get_tool_descriptor()
        parameters: list[dict[str, Any]] = []
        get_params = getattr(module, "get_parameter_descriptors", None)
        if get_params:
            # Some modules derive their defaults from the deployment's config
            # (the built-ins report the operator's own ``rerank`` and
            # ``similarity_threshold``, not the shipped ones); others are fixed.
            # Passing ``config`` only when the signature names it keeps the "add
            # a module, get a UI knob for free" contract intact -- a facility's
            # third-party module stays a zero-argument function and is never
            # forced to grow a parameter it has no use for. A callable whose
            # signature cannot be read at all is treated as zero-argument, the
            # older and safer of the two shapes.
            try:
                accepts_config = "config" in inspect.signature(get_params).parameters
            except (TypeError, ValueError):
                accepts_config = False
            descriptors = get_params(config) if accepts_config else get_params()
            parameters = [p.to_dict() for p in descriptors]
        categories["direct"]["modes"].append(
            {
                "name": name,
                "label": name.replace("_", " ").title(),
                "description": descriptor.description,
                "parameters": parameters,
            }
        )


__all__ = ["SHARED_PARAMETERS", "get_capabilities", "shared_parameters"]
