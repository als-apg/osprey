"""MCP tool: alarm_history — one channel's logged alarm events."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from osprey.mcp_server.pv_directory.tools._channel import answer, check_limit, since_of


async def alarm_history(pv: str, since: str | None = None, limit: int = 20) -> str:
    """List the alarm events the alarm logger recorded for one channel address.

    Rows are ``{time, severity, status, message, user, kind}`` as the logger
    recorded them, ``kind`` being "state" or "config". No rows (``no_match``,
    "no logged events") means nothing was logged in what the logger still
    holds, not that nothing happened; ``retention_known`` says whether how far
    back it holds is known, and ``caveats`` are the deployment's notes on this
    log. ``links.panel`` is the panel page showing that channel.

    Args:
        pv: One channel address; a protocol prefix (``ca://``, ``pva://``) and
            a field suffix are dropped for the lookup.
        since: Only events at or after this ISO-8601 time; a time without an
            offset is facility-local. Omitted: the newest events.
        limit: The most rows to return, 1 to 200.

    Returns:
        JSON object: ``name`` as asked, ``lookup`` as looked up, ``outcome``,
        ``source``, ``rows``, ``truncated``, ``retention_known``, ``caveats``,
        ``reason`` when there is one, and ``links``.
    """
    moment = since_of("alarm_history", since)
    check_limit("alarm_history", limit)
    return await answer(
        "alarm_history",
        pv,
        lambda facts, name: facts.history("alarm_log", name, since=moment, limit=limit),
    )


def register(server: FastMCP) -> Any:
    """Register :func:`alarm_history` on *server*."""
    return server.tool(
        alarm_history,
        title="List a channel's logged alarm events",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
