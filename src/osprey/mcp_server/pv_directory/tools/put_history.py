"""MCP tool: put_history — one channel's logged writes."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from osprey.mcp_server.pv_directory.tools._channel import answer, check_limit, since_of


async def put_history(pv: str, since: str | None = None, limit: int = 20) -> str:
    """List the writes the put log recorded for one channel address.

    Rows are ``{at, pv, field, old, new, user, client}`` as the log recorded
    them; ``field`` names a field write such as ``.HIHI``. No rows
    (``no_match``, "no logged puts") means nothing was logged in what the log
    still holds, not that the channel never changed; ``retention_known``
    says whether how far back it holds is known, and ``caveats`` are the
    deployment's notes on this log. ``links.panel`` is the panel page showing
    that channel.

    Args:
        pv: One channel address; a protocol prefix (``ca://``, ``pva://``) and
            a field suffix are dropped for the lookup, so field writes are listed.
        since: Only writes at or after this ISO-8601 time; a time without an
            offset is facility-local. Omitted: the newest writes.
        limit: The most rows to return, 1 to 200.

    Returns:
        JSON object: ``name`` as asked, ``lookup`` as looked up, ``outcome``,
        ``source``, ``rows``, ``truncated``, ``retention_known``, ``caveats``,
        ``reason`` when there is one, and ``links``.
    """
    moment = since_of("put_history", since)
    check_limit("put_history", limit)
    return await answer(
        "put_history",
        pv,
        lambda facts, name: facts.history("put_log", name, since=moment, limit=limit),
    )


def register(server: FastMCP) -> Any:
    """Register :func:`put_history` on *server*."""
    return server.tool(
        put_history,
        title="List a channel's logged writes",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
