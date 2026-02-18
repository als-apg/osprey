"""OSPREY Workspace MCP Server.

FastMCP server exposing memory, artifact, data context, and screen capture tools.

Usage:
    python -m osprey.mcp_server.workspace
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.workspace")

mcp = FastMCP("osprey-workspace")


def create_server() -> FastMCP:
    """Initialize workspace singletons and import tool modules, then return the server."""
    from osprey.mcp_server.common import (
        initialize_workspace_singletons,
        resolve_workspace_root,
    )

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.mcp_server.workspace.tools import (  # noqa: F401
        artifact_export,
        artifact_save,
        data_context_tools,
        facility_description,
        focus_tools,
        memory,
        screen_capture,
        submit_response,
    )

    logger.info("Workspace MCP server initialised with all tools registered")
    return mcp
