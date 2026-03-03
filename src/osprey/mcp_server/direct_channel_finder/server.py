"""Direct Channel Finder MCP Server.

FastMCP server that queries live PV metadata backends and uses curated
channel databases as reference context for naming patterns.

Usage:
    python -m osprey.mcp_server.direct_channel_finder
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.direct_channel_finder")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP("direct-channel-finder")


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

    from osprey.mcp_server.direct_channel_finder.server_context import (
        initialize_dcf_context,
    )

    initialize_dcf_context()

    workspace_root = resolve_workspace_root()
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.mcp_server.direct_channel_finder.tools import (  # noqa: F401
        get_naming_patterns,
        get_pv_metadata,
        search_pvs,
    )

    logger.info("Direct Channel Finder MCP server initialised with all tools registered")
    return mcp
