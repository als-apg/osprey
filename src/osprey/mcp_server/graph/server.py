"""OSPREY Graph MCP Server.

FastMCP server exposing read-only Cypher search over the facility knowledge
graph held in the ``graphdb`` service.

Usage:
    python -m osprey.mcp_server.graph
"""

import logging

from fastmcp import FastMCP

from osprey.mcp_server.errors import make_error  # noqa: F401  (re-exported for graph tools)

logger = logging.getLogger("osprey.mcp_server.graph")

mcp = FastMCP(
    "graph",
    instructions=(
        "Read-only knowledge-graph search over the facility's NARAD-convention graph. "
        "Start with example_queries or get_schema to learn the shapes that exist, "
        "then run read_cypher. Results are bounded by a row cap and a query timeout, "
        "so a truncated answer means narrow the query rather than retry it unchanged."
    ),
)


def create_server() -> FastMCP:
    """Initialize the server context and import tool modules, then return the server."""
    from osprey.mcp_server.graph.server_context import initialize_server_context
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    # Import tool modules (each registers itself via @mcp.tool()).
    with startup_timer("tool_imports"):
        from osprey.mcp_server.graph.tools import (  # noqa: F401
            capabilities,
            example_queries,
            get_schema,
            read_cypher,
        )

    logger.info("Graph MCP server initialised with all tools registered")
    return mcp
