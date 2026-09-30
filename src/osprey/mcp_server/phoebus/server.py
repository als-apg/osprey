"""OSPREY Phoebus MCP Server.

FastMCP server exposing perceive tools, and the drive tool when
``phoebus.agent_access`` is ``read_write``, that talk to a running Phoebus
product's agent bridge over JSON/HTTP.

Usage:
    python -m osprey.mcp_server.phoebus
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from osprey.phoebus_agent_access import DRIVE_TOOL, drive_offered
from osprey.utils.workspace import load_osprey_config

if TYPE_CHECKING:
    import mcp.types as mt
    from fastmcp.tools import Tool

logger = logging.getLogger("osprey.mcp_server.phoebus")


class DriveOfferMiddleware(Middleware):
    """Leave ``phoebus_drive`` out of ``tools/list`` unless ``phoebus.agent_access`` is read_write.

    It hides the tool and does not refuse it. The refusal lives in the tool,
    so a call that reaches the server anyway gets an answer naming the key,
    not a bare "Unknown tool".
    """

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        """Return the listed tools, without the drive when it is not offered."""
        tools = await call_next(context)
        if drive_offered(load_osprey_config()):
            return tools
        return [tool for tool in tools if tool.name != DRIVE_TOOL]


mcp = FastMCP(
    "phoebus",
    instructions=(
        "Perceive live Phoebus control panels: list open displays, read the "
        "widget tree with PV values, snapshot a widget as PNG, open a registered "
        "panel. Also brings up live Data Browser archiver plots for a PV list + "
        "time range via phoebus_open_databrowser. When phoebus.agent_access is "
        "read_write, phoebus_drive also clicks and types into widgets via "
        "synthetic GUI events or the semantic PV path."
    ),
)
mcp.add_middleware(DriveOfferMiddleware())


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

    # Session working root used by other tools at call time; the artifact
    # store itself is rooted at the shared data root inside
    # initialize_workspace_singletons().
    logger.info("Workspace root: %s", resolve_workspace_root())
    initialize_workspace_singletons()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.phoebus.tools import (
            bridge_tools,  # noqa: F401
            databrowser_tools,  # noqa: F401
        )

    logger.info("Phoebus MCP server initialised with all tools registered")
    return mcp
