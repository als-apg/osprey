"""OSPREY Scan MCP Server.

FastMCP server exposing Bluesky scan control as a thin HTTP client of the
facility-side Bluesky bridge: list scan plans, create a scan intent, check run
status, launch (promote) an intent to a running scan, and stop a run. This
module and its tools make no bluesky/ophyd/tiled imports — every operation is
an HTTP request to the bridge (see ``osprey.services.bluesky_bridge``).

Usage:
    python -m osprey.mcp_server.scan
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.scan")

mcp = FastMCP(
    "scan",
    instructions=(
        "Drive Bluesky scans through the facility Bluesky bridge: list available "
        "scan plans, create a scan intent (validated but not yet running), "
        "check run status, launch (promote) an intent into a running scan, "
        "and stop a running scan."
    ),
)


def create_server() -> FastMCP:
    """Initialize the Bluesky bridge context and register tools."""
    from osprey.mcp_server.scan.server_context import initialize_server_context
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.scan.tools import (  # noqa: F401
            launch,
            read_tools,
            stop,
        )

    logger.info("Scan MCP server initialised with all tools registered")
    return mcp
