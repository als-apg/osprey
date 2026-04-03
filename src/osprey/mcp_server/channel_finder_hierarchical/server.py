"""Hierarchical Channel Finder MCP Server.

FastMCP server that exposes the Hierarchical channel finder database
as MCP tools for Claude Code. All tools are synchronous since the
database is loaded from JSON in memory.

Usage:
    python -m osprey.mcp_server.channel_finder_hierarchical
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.channel_finder_hierarchical")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "channel-finder-hier",
    instructions="Find control-system channel addresses using hierarchical search",
)


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

    from osprey.mcp_server.channel_finder_hierarchical.server_context import (
        initialize_cf_hier_context,
    )

    initialize_cf_hier_context()

    workspace_root = resolve_workspace_root()
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.mcp_server.channel_finder_hierarchical.tools import (  # noqa: F401
        build_channels,
        get_options,
        view_examples,
    )

    logger.info("Channel Finder Hierarchical MCP server initialised with all tools registered")
    return mcp
