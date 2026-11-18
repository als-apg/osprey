"""MCP tool: channel_record — every configured section about each named channel address."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.channel_directory.server import (
    get_tool_context,
    mcp,
    panel_link,
    refuse_unavailable,
)
from osprey.mcp_server.errors import answer_map, make_error, refuse_nothing_answered
from osprey.services.channel_directory.origin import normalise_address

logger = logging.getLogger("osprey.mcp_server.channel_directory.tools.channel_record")

#: The most names one call looks up.
MAX_NAMES = 50


@mcp.tool(
    title="Look up channel records",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def channel_record(channels: list[str]) -> str:
    """Look up channel addresses in the control system's own directory.

    For each address, one entry with every configured section: the directory
    record (its ``origin``: "curated", "directory" or "unknown", and its
    properties), and, where the deployment configures them, the archive's facts
    (``archive``: whether the archive records the channel, ``null`` when the
    archiver cannot say, its first and last sample), alarm state, alarm
    configuration, alarm history and put history. Each section has its own
    ``outcome``; a section with no answer is ``unavailable`` with its
    ``reason``. ``not_listed`` means the directory does not list that address,
    which is not evidence the channel does not exist. The entry's own
    ``outcome`` sums that up: ``found`` (the directory lists it),
    ``not_listed`` (it does not, but another section found something),
    ``unknown`` (nothing found it) or ``unavailable`` with the directory's
    ``reason``. Each entry carries ``links.panel``, the panel page showing
    that channel. No value is read
    here: read current values with channel_read.

    Args:
        channels: Up to 50 channel addresses; a protocol prefix (``ca://``,
            ``pva://``) and a field suffix are dropped for the lookup.

    Returns:
        JSON object: ``channels`` (one entry per address asked: ``name`` as
        asked, ``lookup`` as looked up, ``outcome``, ``reason`` when there is
        one, ``sections``, ``links``) and
        ``missing`` (each address no section answered, with the reason).
    """
    names = list(dict.fromkeys(name.strip() for name in channels if name.strip()))
    if not names or len(names) > MAX_NAMES:
        make_error(
            "validation_error",
            f"channel_record takes 1 to {MAX_NAMES} channel addresses (got {len(names)}).",
            [f"Ask for at most {MAX_NAMES} addresses per call."],
        )
    try:
        facts = get_tool_context().channel_facts
        lookups = [normalise_address(name).name for name in names]
        answers = await asyncio.gather(*(facts.channel_facts(lookup) for lookup in lookups))
        entries = [
            {
                "name": name,
                "lookup": lookup,
                **{key: answer[key] for key in ("outcome", "reason") if key in answer},
                "sections": answer["sections"],
                "links": panel_link("channel", lookup),
            }
            for name, lookup, answer in zip(names, lookups, answers, strict=True)
        ]
        sections = [section for channel in entries for section in channel["sections"]]
        if not sections or all(_no_backend(section) for section in sections):
            refuse_unavailable("channel_record", _first_directory(entries) or await _no_source())
        missing = {
            channel["name"]: channel["sections"][0].get("reason")
            for channel in entries
            if all(s.get("outcome") == "unavailable" for s in channel["sections"])
        }
        if len(missing) == len(entries):
            refuse_nothing_answered("channel_record", "channels", missing)
        return json.dumps({"channels": entries, **answer_map("channel_record", missing=missing)})
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("channel_record failed")
        make_error(
            "internal_error",
            f"channel_record failed: {exc}",
            ["Check the MCP server logs for details."],
        )


def _no_backend(section: Mapping[str, Any]) -> bool:
    """Whether *section* is unavailable because no backend serves the current target."""
    source = section.get("source")
    return section.get("outcome") == "unavailable" and (
        not isinstance(source, Mapping) or source.get("service") == "none"
    )


def _first_directory(channels: list[dict[str, Any]]) -> dict[str, Any] | None:
    for channel in channels:
        for section in channel["sections"]:
            if section["section"] == "directory":
                return dict(section)
    return None


async def _no_source() -> dict[str, Any]:
    """The service's own answer for a call with no section configured."""
    return await get_tool_context().service.count()
