"""MCP tool: ariel_capabilities — report ARIEL service capabilities."""

import json
import logging

from osprey.interfaces.ariel.mcp.registry import get_ariel_registry
from osprey.interfaces.ariel.mcp.server import make_error, mcp
from osprey.services.ariel_search.models import SearchMode

logger = logging.getLogger("osprey.interfaces.ariel.mcp.tools.capabilities")


@mcp.tool()
async def ariel_capabilities() -> str:
    """Report available ARIEL search capabilities.

    Returns enabled search modules, pipelines, search modes,
    default settings, and reasoning configuration.
    Does NOT require database connectivity.

    Returns:
        JSON with capabilities information.
    """
    try:
        registry = get_ariel_registry()
        config = registry.config

        return json.dumps(
            {
                "search_modes": [m.value for m in SearchMode],
                "enabled_search_modules": config.get_enabled_search_modules(),
                "enabled_pipelines": config.get_enabled_pipelines(),
                "default_max_results": config.default_max_results,
                "reasoning": {
                    "provider": config.reasoning.provider,
                    "model_id": config.reasoning.model_id,
                    "max_iterations": config.reasoning.max_iterations,
                    "temperature": config.reasoning.temperature,
                },
                "embedding": {
                    "provider": config.embedding.provider,
                },
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("ariel_capabilities failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get capabilities: {exc}",
                ["Check ARIEL configuration in config.yml."],
            )
        )
