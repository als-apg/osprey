"""OSPREY Control System MCP Server.

FastMCP server exposing channel_read, channel_write, and archiver_read.

Usage:
    python -m osprey.mcp_server.control_system
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.control_system")

mcp = FastMCP(
    "controls",
    instructions="Read and write control-system channels and query archiver history",
)


def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.control_system.tools import (  # noqa: F401
            archiver_read,
            channel_limits,
            channel_read,
            channel_write,
        )

    logger.info("Control System MCP server initialised with all tools registered")
    return mcp
