"""OSPREY Phoebus MCP Server.

FastMCP server exposing perceive + drive tools that talk to a running
Phoebus product's agent bridge over JSON/HTTP.

Usage:
    python -m osprey.mcp_server.phoebus
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.phoebus")

mcp = FastMCP(
    "phoebus",
    instructions=(
        "Perceive and drive live Phoebus control panels: list open displays, "
        "read the widget tree with PV values, snapshot a widget as PNG, and "
        "click/type into widgets via synthetic GUI events or the semantic PV path."
    ),
)


def create_server() -> FastMCP:
    """Initialize workspace singletons (for snapshot artifacts) and register tools."""
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    # Snapshot results are persisted through the ArtifactStore, which lives in
    # the workspace singletons — prime them so phoebus_snapshot can save PNGs.
    prime_config_builder()

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.phoebus.tools import bridge_tools  # noqa: F401

    logger.info("Phoebus MCP server initialised with all tools registered")
    return mcp
