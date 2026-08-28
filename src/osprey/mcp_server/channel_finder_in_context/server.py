"""In-Context Channel Finder MCP Server.

FastMCP server that exposes the in-context channel finder database
as MCP tools for Claude Code. Provides synchronous access to
channel databases (flat or template format).

Usage:
    python -m osprey.mcp_server.channel_finder_in_context
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.channel_finder_in_context")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP("channel-finder-ic")


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.mcp_server.channel_finder_common import build_cf_server

    def _initialize_context() -> object:
        from osprey.mcp_server.channel_finder_in_context.server_context import (
            initialize_cf_ic_context,
        )

        return initialize_cf_ic_context()

    def _import_tools() -> None:
        # Import tool modules (each registers itself via @mcp.tool())
        from osprey.mcp_server.channel_finder_in_context.tools import (  # noqa: F401
            ask_channels,
        )

    return build_cf_server(
        mcp=mcp,
        logger=logger,
        initialize_context=_initialize_context,
        import_tools=_import_tools,
        ready_message="Channel Finder IC MCP server initialised with all tools registered",
    )
