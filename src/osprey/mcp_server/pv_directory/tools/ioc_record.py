"""MCP tool: ioc_record — one IOC's facts from the control system's own directory."""

from __future__ import annotations

import json
import logging

from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.pv_directory.server import (
    get_tool_context,
    mcp,
    refuse_unavailable,
)

logger = logging.getLogger("osprey.mcp_server.pv_directory.tools.ioc_record")


@mcp.tool(
    title="Look up an IOC",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def ioc_record(ioc: str) -> str:
    """Look up one IOC in the control system's own directory.

    Answers the IOC's host, its ports, how many channels it serves (and how
    many are inactive, and how many the curated channel list declares, with
    ``curated_reason`` when that cannot be counted), its engineer and where
    that came from (``engineer_source``), and ``health_pvs``: the addresses of
    the IOC's health channels. Names only, never values: to tell whether the
    IOC is alive, read its health channels with channel_read; a HEARTBEAT
    that advances between two reads means alive. On the simulator the engineer
    is the address of a channel; read it with channel_read for the name.
    With an alarm feed configured, ``alarms`` lists the IOC's channels in alarm
    now (``outcome``, ``total``, ``rows`` of pv, severity, status, latched,
    acknowledged and time), or ``unavailable`` with the feed's ``reason``.
    ``links.panel`` is the panel page showing this IOC. ``not_listed`` means
    the directory lists no channel served by that IOC name.

    Args:
        ioc: The IOC name, as the directory records it.

    Returns:
        JSON object: ``ioc``, ``outcome``, ``source``, ``host``, ``ports``,
        ``pv_count``, ``inactive_count``, ``curated_count``, ``engineer``,
        ``engineer_source``, ``health_pvs``, ``links``, ``alarms`` with an alarm
        feed configured, and the reasons ``curated_reason`` and
        ``health_pvs_reason`` where they apply.
    """
    name = ioc.strip()
    if not name:
        make_error("validation_error", "ioc_record needs an IOC name.", ["Name one IOC."])
    try:
        facts = await get_tool_context().pv_facts.ioc_facts(name)
        if facts.get("outcome") == "unavailable":
            refuse_unavailable("ioc_record", facts)
        return json.dumps(facts)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("ioc_record failed")
        make_error(
            "internal_error",
            f"ioc_record failed: {exc}",
            ["Check the MCP server logs for details."],
        )
