"""MCP tool: directory_search — one page of the control system's own channel directory."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.pv_directory.server import (
    get_tool_context,
    mcp,
    panel_link,
    refuse_unavailable,
)

logger = logging.getLogger("osprey.mcp_server.pv_directory.tools.search")

#: Rows per page; the panel routes cap a page at the same size.
PAGE_SIZE = 50

#: What a search with no rows says: a miss in the directory is not an absence.
NO_MATCH_NOTE = (
    "No channel in the directory matched this search. A miss does not show the "
    "channel does not exist: names are the control system's own, so try other words, "
    "a name pattern, or a property value from directory_properties."
)


@mcp.tool(
    title="Search the channel directory",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def directory_search(
    words: str = "",
    name: str | None = None,
    properties: dict[str, str] | None = None,
    status: str | None = None,
    page: int = 0,
) -> str:
    """Search the control system's own directory of channel addresses.

    The directory lists every channel address the control system announces,
    curated or not. Names are the control system's own: a name that matches a
    word is not a channel that measures what the word means. Check the curated
    channels first. Each row carries ``origin`` ("curated": the curated channel
    list declares it; "directory": only the control system lists it; "unknown":
    no curated list to compare) and ``links.panel``, the panel page showing that
    channel. ``searched`` states the search in plain words and ``request`` the
    literal request sent. A part that missed the deadline is listed in
    ``partial`` with its reason. ``no_match`` is not evidence the channel does
    not exist.

    Args:
        words: Words matched against channel names and descriptions.
        name: A channel-name glob (``*`` any run, ``?`` one character); without
            a leading or trailing ``*`` it matches anywhere in the name.
        properties: Directory property values a channel must have, by property
            name; directory_properties lists the values a property holds.
        status: "Active" or "Inactive", as the directory records it.
        page: Page number from 0; a page holds 50 channels.

    Returns:
        JSON object: ``outcome``, ``searched``, ``request``, ``source``, ``total``,
        ``rows``, ``truncated``, ``partial``, ``facets``, and ``note`` on no match.
    """
    try:
        answer = await get_tool_context().service.search(
            words=words,
            name=name,
            properties=dict(properties or {}),
            status=status,
            page=max(page, 0),
            page_size=PAGE_SIZE,
        )
        if answer.get("outcome") == "unavailable":
            refuse_unavailable("directory_search", answer)
        rows = [{**row, "links": panel_link("pv", row["name"])} for row in answer["rows"]]
        result: dict[str, Any] = {**answer, "rows": rows}
        if answer.get("outcome") == "no_match":
            result["note"] = NO_MATCH_NOTE
        return json.dumps(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("directory_search failed")
        make_error(
            "internal_error",
            f"directory_search failed: {exc}",
            ["Check the MCP server logs for details."],
        )
