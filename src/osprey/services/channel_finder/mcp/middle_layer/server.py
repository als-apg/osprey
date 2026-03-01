"""Middle Layer Channel Finder MCP Server.

FastMCP server that exposes the Middle Layer channel finder database
as MCP tools for Claude Code. All tools are synchronous since the
database is loaded from JSON in memory.

Usage:
    python -m osprey.services.channel_finder.mcp.middle_layer
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.services.channel_finder.mcp.middle_layer")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP("channel-finder-mml")


# ---------------------------------------------------------------------------
# Structured error helper (same contract as osprey.mcp_server.server)
# ---------------------------------------------------------------------------
def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
) -> dict:
    """Build the cross-team standard error envelope."""
    return {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    from osprey.services.channel_finder.mcp.middle_layer.registry import (
        initialize_cf_ml_registry,
    )

    initialize_cf_ml_registry()

    workspace_root = resolve_workspace_root()
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.services.channel_finder.mcp.middle_layer.tools import (  # noqa: F401
        get_common_names,
        inspect_fields,
        list_channels,
        list_families,
        list_systems,
        statistics,
        validate,
    )

    logger.info("Channel Finder MML MCP server initialised with all tools registered")
    return mcp
