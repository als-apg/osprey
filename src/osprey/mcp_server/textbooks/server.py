"""Textbooks MCP Server.

FastMCP server providing structured lookup and retrieval of accelerator physics
textbook content.  Uses pre-built indexes (concept maps, term indexes, section
headings, equation tags) for fast routing, with surgical section reads and
full-text search fallback.

Usage:
    python -m osprey.mcp_server.textbooks serve
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.textbooks")

mcp = FastMCP(
    "textbooks",
    instructions=(
        "Look up concepts, equations, and derivations in accelerator physics "
        "textbooks.  Start with textbook_lookup to find where a concept is "
        "discussed, then use textbook_read_section to read the content."
    ),
)


def create_server() -> FastMCP:
    """Import tool modules, initialise indexes, and return the configured server."""
    from osprey.mcp_server.startup import startup_timer

    with startup_timer("textbook_indexes"):
        from osprey.mcp_server.textbooks.indexer import load_all_books  # noqa: F401

        load_all_books()

    with startup_timer("tool_imports"):
        from osprey.mcp_server.textbooks.tools import (  # noqa: F401
            lookup,
            overview,
            read_section,
            search,
        )

    logger.info("Textbooks MCP server initialised with all tools registered")
    return mcp
