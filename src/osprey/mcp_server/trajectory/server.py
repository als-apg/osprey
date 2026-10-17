"""OSPREY trajectory MCP server.

FastMCP server over the trajectory service's agent door: take a shot, read the
pending correction step, apply that step or put recorded values back. Every
request goes through the web terminal's proxy at ``/panel/trajectory/api/*``
with the panel token; the proxy injects the service's launch token, which this
process never holds.

Usage:
    python -m osprey.mcp_server.trajectory
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.trajectory")


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Release the agent's trajectory session when the server exits."""
    try:
        yield {}
    finally:
        from osprey.mcp_server.trajectory.client import release_session

        await release_session()


mcp = FastMCP(
    "trajectory",
    instructions=(
        "Drive the trajectory correction loop: trajectory_shot takes a shot and "
        "computes the correction step, trajectory_step reads the pending step, "
        "trajectory_apply writes that step by its digest, or puts recorded values "
        "back. Shots and applies are put to a human before they run."
    ),
    lifespan=_lifespan,
)


def create_server() -> FastMCP:
    """Register the trajectory tools and return the server."""
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    with startup_timer("tool_imports"):
        from osprey.mcp_server.trajectory.tools import loop  # noqa: F401

    logger.info("Trajectory MCP server initialised with all tools registered")
    return mcp
