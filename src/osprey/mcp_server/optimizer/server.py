"""OSPREY optimizer MCP server: an HTTP client of the optimizer service (``python -m osprey.mcp_server.optimizer``)."""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.optimizer")

mcp = FastMCP(
    "optimizer",
    instructions=(
        "Read the facility optimizer service: its environments (variables, "
        "observables, presets, bounds, live readbacks), its jobs and their "
        "configs, stored results and saved models. This server holds no launch "
        "token and cannot start a run: a person starts every run."
    ),
)


def create_server() -> FastMCP:
    """Register the tools."""
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.optimizer.tools import (  # noqa: F401
            environments,
            jobs,
            results,
        )

    logger.info("Optimizer MCP server initialised with all tools registered")
    return mcp
