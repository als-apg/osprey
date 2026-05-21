"""Middle Layer Channel Finder MCP Server.

FastMCP server that exposes the Middle Layer channel finder database
as MCP tools for Claude Code. All tools are synchronous since the
database is loaded from JSON in memory.

Usage:
    python -m osprey.mcp_server.channel_finder_middle_layer
"""

import logging

from fastmcp import FastMCP

from osprey.mcp_server.errors import make_error  # noqa: F401  (re-exported for tools)

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP("channel-finder-mml")


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
    )

    prime_config_builder()

    from osprey.mcp_server.channel_finder_middle_layer.server_context import (
        initialize_cf_ml_context,
    )

    initialize_cf_ml_context()
    initialize_workspace_singletons()

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.mcp_server.channel_finder_middle_layer.tools import (  # noqa: F401
        get_common_names,
        inspect_fields,
        list_channels,
        list_families,
        list_systems,
        statistics,
        validate,
    )

    # query_channels requires duckdb (optional dependency)
    try:
        from osprey.mcp_server.channel_finder_middle_layer.tools import query_channels  # noqa: F401
    except ImportError:
        logger.info("query_channels tool unavailable (duckdb not installed)")

    logger.info("Channel Finder MML MCP server initialised with all tools registered")
    return mcp
