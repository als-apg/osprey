"""MCP tool: capabilities — report ARIEL service capabilities."""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server import make_error, mcp
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.services.ariel_search.capabilities import get_capabilities

logger = logging.getLogger("osprey.mcp_server.ariel.tools.capabilities")


@mcp.tool()
async def capabilities() -> str:
    """Report available ARIEL search capabilities.

    Returns enabled search modules, search modes, and the embedding provider.

    ``search_modes`` lists the search modules that are both registered and
    enabled — the modes a ``search`` call may actually route to. It is derived
    from the same registry-backed source the capabilities API uses, so the two
    can never drift apart. ``sql_query`` is a separate MCP tool rather than a
    search mode: it bypasses mode dispatch by design and is therefore
    intentionally absent from this list.

    Does NOT require database connectivity, so this is *not* a health check: a
    successful response says nothing about whether the database is reachable.
    For live database/health status (connectivity, entry counts) call the
    ``status`` tool, or run ``osprey ariel status``.

    Returns:
        JSON with capabilities information.
    """
    try:
        context = get_ariel_context()
        config = context.config
        modes = get_capabilities(config)["categories"]["direct"]["modes"]

        return json.dumps(
            {
                "search_modes": [mode["name"] for mode in modes],
                "enabled_search_modules": config.get_enabled_search_modules(),
                "embedding": {
                    "provider": config.embedding.provider,
                },
            },
            default=str,
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("capabilities failed")
        return make_error(
            "internal_error",
            f"Failed to get capabilities: {exc}",
            ["Check ARIEL configuration in config.yml."],
        )
