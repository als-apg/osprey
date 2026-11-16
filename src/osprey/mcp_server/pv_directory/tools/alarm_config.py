"""MCP tool: alarm_config — one channel's place in the alarm configuration."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from osprey.mcp_server.pv_directory.tools._channel import answer


async def alarm_config(pv: str) -> str:
    """Look up one channel address in the alarm configuration.

    Answers whether the alarm configuration holds the channel (``in_tree``:
    "yes", "no" or "unknown"), its path, guidance, displays, and whether its
    alarm is enabled and latching. ``label`` says where that came from: "live
    configuration" from the alarm feed, else "last logged configuration" from
    the alarm logger. ``links.panel`` is the panel page showing that channel.

    Args:
        pv: One channel address; a protocol prefix (``ca://``, ``pva://``) and
            a field suffix are dropped for the lookup.

    Returns:
        JSON object: ``name`` as asked, ``lookup`` as looked up, ``outcome``,
        ``source``, ``caveats`` (the deployment's notes on that source),
        ``record``, ``label`` and ``links``.
    """
    return await answer("alarm_config", pv, lambda facts, name: facts.alarm_config(name))


def register(server: FastMCP) -> Any:
    """Register :func:`alarm_config` on *server*."""
    return server.tool(
        alarm_config,
        title="Look up a channel's alarm configuration",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
