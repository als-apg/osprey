"""OSPREY Bluesky MCP Server.

FastMCP server exposing Bluesky plan/run control as a thin HTTP client of the
facility-side Bluesky bridge: list plans, check run status, stage a shared
plan draft, launch a run from that draft, stop a run, and author
(write_plan) plus validate (validate_plan) a session-tier plan
file. A Bluesky plan is an arbitrary generator (count, mv, scan, grid_scan,
custom acquisition routines) — nothing here is scan-specific. This module and
its tools make no bluesky/ophyd/tiled imports — every operation is an HTTP
request to the bridge (see ``osprey.services.bluesky_bridge``).

Usage:
    python -m osprey.mcp_server.bluesky
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.bluesky")

mcp = FastMCP(
    "bluesky",
    instructions=(
        "Drive Bluesky plans through the facility Bluesky bridge: list available "
        "plans, check run status, stage a shared plan draft, "
        "launch a run from that draft, "
        "and stop a running run. A plan is any Bluesky generator (count, mv, "
        "scan, grid_scan, custom) — not only scans."
    ),
)


def create_server() -> FastMCP:
    """Initialize the Bluesky bridge context and register tools."""
    from osprey.mcp_server.bluesky.server_context import initialize_server_context
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    # Session working root used by other tools at call time; the artifact
    # store itself is rooted at the shared data root inside
    # initialize_workspace_singletons().
    logger.info("Workspace root: %s", resolve_workspace_root())
    initialize_workspace_singletons()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.bluesky.tools import (  # noqa: F401
            authoring,
            draft,
            launch,
            read_tools,
            stop,
        )

    logger.info("Bluesky MCP server initialised with all tools registered")
    return mcp
