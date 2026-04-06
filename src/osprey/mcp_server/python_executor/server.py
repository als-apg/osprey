"""OSPREY Python Executor MCP Server.

FastMCP server exposing the execute tool.

Usage:
    python -m osprey.mcp_server.python_executor
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.python_executor")

mcp = FastMCP(
    "python",
    instructions="Execute Python code in a sandboxed environment",
)


def create_server() -> FastMCP:
    """Initialize config and import tool modules, then return the server."""
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.python_executor.tools import (  # noqa: F401
            python_execute,
            python_execute_file,
        )

    logger.info("Python Executor MCP server initialised with all tools registered")
    return mcp
