"""OSPREY Health MCP Server.

FastMCP server exposing tiered, read-only facility health checks.

Usage:
    python -m osprey.mcp_server.health
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.health")

mcp = FastMCP(
    "health",
    instructions=(
        "Read-only health suite for the facility. "
        "health_check runs the cached poll tier; "
        "health_check_full runs the approval-gated full tier."
    ),
)


def create_server() -> FastMCP:
    """Initialize the server context and import tool modules, then return the server."""
    from osprey.mcp_server.health.server_context import initialize_server_context
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.health.tools import (  # noqa: F401
            health_check,
            health_check_full,
        )

    logger.info("Health MCP server initialised with all tools registered")
    return mcp
