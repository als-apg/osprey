"""OSPREY Python Executor MCP Server.

FastMCP server exposing the execute tool.

Usage:
    python -m osprey.mcp_server.python_executor
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.python_executor")

mcp = FastMCP("python")


def create_server() -> FastMCP:
    """Initialize config and import tool modules, then return the server."""
    from osprey.mcp_server.common import (
        initialize_workspace_singletons,
        prime_config_builder,
        resolve_workspace_root,
        startup_timer,
    )

    prime_config_builder()

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.python_executor.tools import python_execute  # noqa: F401

    logger.info("Python Executor MCP server initialised with all tools registered")
    return mcp
