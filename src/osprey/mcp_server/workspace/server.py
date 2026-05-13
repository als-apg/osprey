"""OSPREY Workspace MCP Server.

FastMCP server exposing artifact, data context, and screen capture tools.

Usage:
    python -m osprey.mcp_server.workspace
"""

import logging

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.workspace")

mcp = FastMCP(
    "workspace",
    instructions="Manage artifacts, capture screens, and run data visualizations",
)


def create_server() -> FastMCP:
    """Initialize workspace singletons and import tool modules, then return the server."""
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    # Prime the main ConfigBuilder and registry so downstream services
    # (LimitsValidator, pattern_detection) find their config when tools
    # like create_static_plot call execute_code().
    prime_config_builder()

    workspace_root = resolve_workspace_root()
    logger.info("Workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.workspace.tools import (  # noqa: F401
            archiver_downsample,
            artifact_export,
            artifact_save,
            create_dashboard,
            create_document,
            create_interactive_plot,
            create_static_plot,
            data_context_tools,
            facility_description,
            focus_tools,
            lattice_tools,
            panel_tools,
            screen_capture,
            session_log,
            session_summary,
            setup,
            submit_response,
        )

    logger.info("Workspace MCP server initialised with all tools registered")
    return mcp
