"""ARIEL capabilities assembly.

Builds the capabilities response that the frontend uses to discover
available search modes and their tunable parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.pipelines import get_pipeline_descriptors
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

    # Add search modules as "direct" modes
    _add_search_modules(config, categories)

    # Add pipelines as "llm" modes
    _add_pipelines(categories)

    return {
        "categories": categories,
        "shared_parameters": [p.to_dict() for p in SHARED_PARAMETERS],
    }


def _add_search_modules(
    config: ARIELConfig,
    categories: dict[str, dict[str, Any]],
) -> None:
    """Add enabled search modules to the capabilities."""
    search_modules = {
        "keyword": {
            "label": "Keyword",
            "description": "Fast text-based lookup using full-text search",
            "import": "osprey.services.ariel_search.search.keyword",
        },
        "semantic": {
            "label": "Semantic",
            "description": "Find conceptually related entries using AI embeddings",
            "import": "osprey.services.ariel_search.search.semantic",
        },
    }

    for name, info in search_modules.items():
        if not config.is_search_module_enabled(name):
            continue

        # Dynamically import to get parameter descriptors
        parameters: list[dict[str, Any]] = []
        try:
            import importlib

            module = importlib.import_module(info["import"])
            get_params = getattr(module, "get_parameter_descriptors", None)
            if get_params:
                parameters = [p.to_dict() for p in get_params()]
        except (ImportError, AttributeError):
            pass

        categories["direct"]["modes"].append(
            {
                "name": name,
                "label": info["label"],
                "description": info["description"],
                "parameters": parameters,
            }
        )


def _add_pipelines(categories: dict[str, dict[str, Any]]) -> None:
    """Add pipeline descriptors to the capabilities."""
    for pipeline in get_pipeline_descriptors():
        categories[pipeline.category]["modes"].append(
            {
                "name": pipeline.name,
                "label": pipeline.label,
                "description": pipeline.description,
                "parameters": [p.to_dict() for p in pipeline.parameters],
            }
        )


__all__ = ["SHARED_PARAMETERS", "get_capabilities"]
