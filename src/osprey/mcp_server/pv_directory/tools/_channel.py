"""What the gated tools share: one channel address in, one section payload out."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any

from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.pv_directory.server import (
    get_tool_context,
    panel_link,
    refuse_unavailable,
)
from osprey.services.pv_directory.facts import Facts
from osprey.services.pv_directory.origin import normalise_address
from osprey_connectors.config import get_facility_timezone

logger = logging.getLogger("osprey.mcp_server.pv_directory.tools")

#: The most rows one history call asks for.
MAX_LIMIT = 200


def lookup(tool: str, name: str) -> str:
    """The address *name* is looked up under; an empty name is refused."""
    if not name.strip():
        make_error("validation_error", f"{tool} needs a channel address.", ["Name one channel."])
    return normalise_address(name.strip()).name


def since_of(tool: str, since: str | None) -> datetime | None:
    """*since* as an aware moment; an offset-free time is facility-local."""
    if since is None or not since.strip():
        return None
    try:
        moment = datetime.fromisoformat(since.strip())
    except ValueError:
        make_error(
            "validation_error",
            f"{tool}: since {since!r} is not an ISO-8601 time.",
            ["Spell since as ISO-8601, for example 2026-04-03T14:00 or 2026-04-03T14:00+00:00."],
        )
    return moment if moment.tzinfo else moment.replace(tzinfo=get_facility_timezone())


def check_limit(tool: str, limit: int) -> None:
    if not 1 <= limit <= MAX_LIMIT:
        make_error(
            "validation_error",
            f"{tool}: limit is 1 to {MAX_LIMIT} rows (got {limit}).",
            [f"Ask for at most {MAX_LIMIT} rows per call."],
        )


async def answer(
    tool: str, name: str, read: Callable[[Facts, str], Awaitable[Mapping[str, Any]]]
) -> str:
    """*read*'s section for *name* with its panel link; a section with no answer refuses."""
    address = lookup(tool, name)
    try:
        section = await read(get_tool_context().pv_facts, address)
        if section.get("outcome") == "unavailable":
            refuse_unavailable(tool, section)
        return json.dumps(
            {"name": name, "lookup": address, **section, "links": panel_link("pv", address)}
        )
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("%s failed", tool)
        make_error(
            "internal_error", f"{tool} failed: {exc}", ["Check the MCP server logs for details."]
        )
