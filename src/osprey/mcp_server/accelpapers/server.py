"""AccelPapers MCP Server.

FastMCP server exposing search and retrieval tools for ~63K INSPIRE
accelerator physics papers stored in a local SQLite FTS5 database.

Usage:
    python -m osprey.mcp_server.accelpapers serve
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.accelpapers")

mcp = FastMCP("accelpapers")


def create_server() -> FastMCP:
    """Import tool modules and return the configured server."""
    from osprey.mcp_server.accelpapers.tools import (  # noqa: F401
        browse,
        get_paper,
        list_conferences,
        search,
        search_author,
        stats,
    )

    logger.info("AccelPapers MCP server initialised with all tools registered")
    return mcp
