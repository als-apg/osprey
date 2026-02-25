"""In-Context Channel Finder MCP Server.

FastMCP server that exposes the in-context channel finder database
as MCP tools for Claude Code. Provides synchronous access to
channel databases (flat or template format).

Usage:
    python -m osprey.services.channel_finder.mcp.in_context
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.services.channel_finder.mcp.in_context")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP("channel-finder-ic")


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
    from osprey.mcp_server.common import (
        initialize_workspace_singletons,
        prime_config_builder,
        resolve_workspace_root,
    )

    prime_config_builder()

    from osprey.services.channel_finder.mcp.in_context.registry import (
        initialize_cf_ic_registry,
    )

    initialize_cf_ic_registry()

    workspace_root = resolve_workspace_root()
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.services.channel_finder.mcp.in_context.tools import (  # noqa: F401
        get_channels,
        resolve_addresses,
        statistics,
    )

    logger.info("Channel Finder IC MCP server initialised with all tools registered")
    return mcp
