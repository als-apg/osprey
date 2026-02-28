"""MATLAB MML MCP Server.

FastMCP server exposing search and retrieval tools for ~3K MATLAB Middle
Layer functions stored in a local SQLite FTS5 database.

Usage:
    python -m osprey.mcp_server.matlab serve
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.matlab")

mcp = FastMCP(
    "matlab",
    instructions="Search and browse the MATLAB Middle Layer codebase",
)


def create_server() -> FastMCP:
    """Import tool modules and return the configured server."""
    from osprey.mcp_server.common import startup_timer

    with startup_timer("tool_imports"):
        from osprey.mcp_server.matlab.tools import (  # noqa: F401
            browse,
            dependencies,
            get_function,
            list_groups,
            path,
            search,
            stats,
        )

    logger.info("MATLAB MML MCP server initialised with all tools registered")
    return mcp
