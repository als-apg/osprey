"""MCP tool: alarm_state — the current alarm of each named channel address."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.channel_directory.server import (
    get_tool_context,
    panel_link,
    refuse_unavailable,
)
from osprey.mcp_server.errors import answer_map, make_error, refuse_nothing_answered
from osprey.services.channel_directory.origin import normalise_address

logger = logging.getLogger("osprey.mcp_server.channel_directory.tools.alarm_state")


#: The most names one call looks up.
MAX_NAMES = 50


async def alarm_state(channels: list[str]) -> str:
    """Read the current alarm of each channel address from the alarm feed.

    For each address, one entry: ``record`` holds ``severity``, ``status``,
    ``latched``, ``acknowledged``, ``message``, ``time`` and ``stale``. A
    ``stale`` record means the alarm feed is not updating: the severity is the
    last one received, not a current "no alarm". ``not_listed`` means the feed
    holds no alarm for that address; ``unavailable`` carries its ``reason``
    (for example "alarm feed still loading"). Each entry carries
    ``links.panel``, the panel page showing that channel.

    Args:
        channels: Up to 50 channel addresses; a protocol prefix (``ca://``,
            ``pva://``) and a field suffix are dropped for the lookup.

    Returns:
        JSON object: ``channels`` (one entry per address asked: ``name`` as
        asked, ``lookup`` as looked up, ``outcome``, ``source``, ``caveats``
        (the deployment's notes on that source), ``record``, ``reason`` when
        there is one, ``links``) and ``missing`` (each address with no answer,
        with the reason).
    """
    names = list(dict.fromkeys(name.strip() for name in channels if name.strip()))
    if not names or len(names) > MAX_NAMES:
        make_error(
            "validation_error",
            f"alarm_state takes 1 to {MAX_NAMES} channel addresses (got {len(names)}).",
            [f"Ask for at most {MAX_NAMES} addresses per call."],
        )
    try:
        facts = get_tool_context().channel_facts
        lookups = [normalise_address(name).name for name in names]
        answers = await asyncio.gather(*(facts.alarm_state(lookup) for lookup in lookups))
        entries = [
            {"name": name, "lookup": lookup, **answer, "links": panel_link("channel", lookup)}
            for name, lookup, answer in zip(names, lookups, answers, strict=True)
        ]
        if all(_no_backend(answer) for answer in answers):
            refuse_unavailable("alarm_state", answers[0])
        missing = {
            channel["name"]: channel.get("reason")
            for channel in entries
            if channel.get("outcome") == "unavailable"
        }
        if len(missing) == len(entries):
            refuse_nothing_answered("alarm_state", "channels", missing)
        return json.dumps({"channels": entries, **answer_map("alarm_state", missing=missing)})
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("alarm_state failed")
        make_error(
            "internal_error",
            f"alarm_state failed: {exc}",
            ["Check the MCP server logs for details."],
        )


def _no_backend(section: Mapping[str, Any]) -> bool:
    """Whether *section* is unavailable because no backend serves the current target."""
    source = section.get("source")
    return section.get("outcome") == "unavailable" and (
        not isinstance(source, Mapping) or source.get("service") == "none"
    )


def register(server: FastMCP) -> Any:
    """Register :func:`alarm_state` on *server*."""
    return server.tool(
        alarm_state,
        title="Read channels' current alarm",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
