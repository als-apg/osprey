"""ARIEL MCP Server.

FastMCP server that exposes the full ARIEL logbook search service
as MCP tools for Claude Code. Independent from the main OSPREY MCP server.

Usage:
    python -m osprey.mcp_server.ariel
"""

import logging
from datetime import datetime

from fastmcp import FastMCP

from osprey.mcp_server.errors import make_error  # noqa: F401  (re-exported for ARIEL tools)

logger = logging.getLogger("osprey.mcp_server.ariel")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "ariel",
    instructions="Search facility logbook entries and operational records",
)


# ---------------------------------------------------------------------------
# Shared helpers for ARIEL tool modules
# ---------------------------------------------------------------------------


def parse_date_filters(
    start_date: str | None,
    end_date: str | None,
) -> tuple[datetime | None, datetime | None]:
    """Parse optional ISO-8601 date strings into datetime objects.

    Naive datetime inputs are assumed to be in the facility timezone.

    Args:
        start_date: ISO-8601 date string or None.
        end_date: ISO-8601 date string or None.

    Returns:
        Tuple of (parsed_start, parsed_end), either may be None.
    """
    from osprey.utils.config import get_facility_timezone

    parsed_start = datetime.fromisoformat(start_date) if start_date else None
    parsed_end = datetime.fromisoformat(end_date) if end_date else None

    # Localize naive datetimes to facility timezone
    if parsed_start and parsed_start.tzinfo is None:
        tz = get_facility_timezone()
        parsed_start = parsed_start.replace(tzinfo=tz)
    if parsed_end and parsed_end.tzinfo is None:
        tz = get_facility_timezone()
        parsed_end = parsed_end.replace(tzinfo=tz)

    return parsed_start, parsed_end


def serialize_entry(entry: dict, text_limit: int = 300) -> dict:
    """Serialize an EnhancedLogbookEntry dict into a compact response dict.

    Timestamps are converted to the facility timezone for agent consumption.

    Args:
        entry: EnhancedLogbookEntry TypedDict (plain dict).
        text_limit: Maximum characters of raw_text to include.

    Returns:
        Serialized dict suitable for JSON response.
    """
    from osprey.utils.config import get_facility_timezone

    ts = entry["timestamp"]
    if hasattr(ts, "astimezone"):
        tz = get_facility_timezone()
        ts = ts.astimezone(tz).isoformat()

    result = {
        "entry_id": entry["entry_id"],
        "timestamp": ts,
        "author": entry.get("author", ""),
        "source_system": entry["source_system"],
        "raw_text": entry["raw_text"][:text_limit],
        "summary": entry.get("summary"),
    }
    if "_score" in entry:
        result["score"] = entry["_score"]
    return result


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()
    initialize_ariel_context()

    workspace_root = resolve_workspace_root()
    logger.info("ARIEL workspace root: %s", workspace_root)
    initialize_workspace_singletons()

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.mcp_server.ariel.tools import (  # noqa: F401
        browse,
        capabilities,
        entry,
        keyword_search,
        publish,
        semantic_search,
        sql_query,
        status,
    )

    logger.info("ARIEL MCP server initialised with all tools registered")
    return mcp
