"""ARIEL capabilities assembly.

Builds the capabilities response that the frontend uses to discover
available search modes and their tunable parameters.
"""

from __future__ import annotations

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


def get_capabilities(config: ARIELConfig) -> dict[str, Any]:
    """Build the capabilities response for the frontend.

    Iterates enabled search modules (keyword, semantic) as "direct" modes
    and pipeline descriptors (RAG, Agent) as "llm" modes. Collects parameter
    descriptors from each.

    Args:
        config: ARIEL configuration

    Returns:
        Dict matching the capabilities response schema:
        {
            "categories": {
                "llm": {"label": "LLM", "modes": [...]},
                "direct": {"label": "Direct", "modes": [...]},
            },
            "shared_parameters": [...],
        }
    """
    categories: dict[str, dict[str, Any]] = {
        "llm": {"label": "LLM", "modes": []},
        "direct": {"label": "Direct", "modes": []},
    }

    _add_search_modules(config, categories)
    _add_pipelines(config, categories)

    return {
        "categories": categories,
        "shared_parameters": [p.to_dict() for p in SHARED_PARAMETERS],
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
            parameters = [p.to_dict() for p in get_params()]
        categories["direct"]["modes"].append(
            {
                "name": name,
                "label": name.replace("_", " ").title(),
                "description": descriptor.description,
                "parameters": parameters,
            }
        )


def _add_pipelines(
    config: ARIELConfig,
    categories: dict[str, dict[str, Any]],
) -> None:
    """Add enabled pipeline descriptors to the capabilities via the registry."""
    from osprey.registry import get_registry

    registry = get_registry()
    for name in registry.list_ariel_pipelines():
        if not config.is_pipeline_enabled(name):
            continue
        module = registry.get_ariel_pipeline(name)
        if module is None:
            continue
        descriptor = module.get_pipeline_descriptor(name)
        categories[descriptor.category]["modes"].append(
            {
                "name": descriptor.name,
                "label": descriptor.label,
                "description": descriptor.description,
                "parameters": [p.to_dict() for p in descriptor.parameters],
            }
        )


__all__ = ["SHARED_PARAMETERS", "get_capabilities"]
