"""OSPREY optimizer MCP Server.

FastMCP server that reads the optimizer service over HTTP: its environments
(definitions, status, readbacks, presets, bounds), its jobs and their configs,
its stored results and its saved models. Read-only: it calls no route that
writes, and holds no launch token.

Usage:
    python -m osprey.mcp_server.optimizer
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.optimizer")

mcp = FastMCP(
    "optimizer",
    instructions=(
        "Read the facility optimizer service: its environments (variables, "
        "observables, presets, bounds, live readbacks), its jobs and their "
        "configs, stored results and saved models. These tools only read; a "
        "run is started by a human, not by this server."
    ),
)


def create_server() -> FastMCP:
    """Resolve the optimizer URL and register tools."""
    from osprey.mcp_server.optimizer.server_context import base_url
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    with startup_timer("server_context"):
        logger.info("Optimizer URL: %s", base_url())

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.optimizer.tools import (  # noqa: F401
            environments,
            jobs,
            results,
        )

    logger.info("Optimizer MCP server initialised with all tools registered")
    return mcp
