"""MCP tool: active_alarms — every channel in alarm now, with the counts per severity."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.pv_directory.server import (
    get_tool_context,
    panel_link,
    refuse_unavailable,
)
from osprey.services.pv_directory.facts import ACTIVE_ALARMS_MAX_ROWS

logger = logging.getLogger("osprey.mcp_server.pv_directory.tools.active_alarms")


async def active_alarms(severities: list[str] | None = None, limit: int = 100) -> str:
    """List every channel in alarm now, from the alarm feed, with the counts per severity.

    ``counts`` (per severity, ``OK`` included) and ``total`` are over every
    channel the feed holds a state for; ``rows`` are the active ones only (not
    ``OK``), most severe first as the alarm feed ranks severities (an unknown
    severity first), then newest first. A row holds ``pv`` (the channel address),
    ``path`` (its place in the alarm tree), ``severity``, ``status``,
    ``latched``, ``acknowledged``, ``message``, ``time``, ``stale`` and
    ``links.panel``, the panel page showing that channel. ``stale`` means the
    alarm feed is not updating: the severities are the last received, not a
    clear state. ``last_contact`` is when the feed was last heard from.

    Args:
        severities: Only rows of these severities (for example ``["MAJOR"]``);
            omitted lists every active row. The counts are never filtered.
        limit: The most rows answered, 0 to 1000; a cut sets ``truncated``.

    Returns:
        JSON object: ``outcome``, ``source``, ``caveats`` (the deployment's
        notes on that source), ``counts``, ``total``, ``rows``, ``truncated``,
        ``stale`` and ``last_contact``.
    """
    if not 0 <= limit <= ACTIVE_ALARMS_MAX_ROWS:
        make_error(
            "validation_error",
            f"active_alarms takes a limit of 0 to {ACTIVE_ALARMS_MAX_ROWS} (got {limit}).",
            [f"Ask for at most {ACTIVE_ALARMS_MAX_ROWS} rows per call."],
        )
    try:
        facts = get_tool_context().pv_facts
        answer = await facts.active_alarms(severities=tuple(severities or ()), limit=limit)
        if answer.get("outcome") == "unavailable":
            refuse_unavailable("active_alarms", answer)
        rows = [{**row, "links": panel_link("pv", row["pv"])} for row in answer["rows"]]
        return json.dumps({**answer, "rows": rows})
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("active_alarms failed")
        make_error(
            "internal_error",
            f"active_alarms failed: {exc}",
            ["Check the MCP server logs for details."],
        )


def register(server: FastMCP) -> Any:
    """Register :func:`active_alarms` on *server*."""
    return server.tool(
        active_alarms,
        title="List the channels in alarm",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
