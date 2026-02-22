"""ARIEL MCP Server.

FastMCP server that exposes the full ARIEL logbook search service
as MCP tools for Claude Code. Independent from the main OSPREY MCP server.

Usage:
    python -m osprey.interfaces.ariel.mcp
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from fastmcp import FastMCP

from osprey.mcp_server.common import make_error  # noqa: F401  (re-exported for ARIEL tools)

logger = logging.getLogger("osprey.interfaces.ariel.mcp")

# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP("ariel")


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
# Workspace save helper
# ---------------------------------------------------------------------------
def save_to_workspace(
    category: str,
    data: dict,
    description: str,
    tool_name: str,
) -> Path:
    """Save tool output to the osprey-workspace directory.

    .. deprecated::
        Use ``DataContext.save()`` from ``osprey.mcp_server.data_context`` instead.
        This function is retained for backward compatibility only.

    Args:
        category: Subdirectory name (e.g., "search_results")
        data: Data to serialize as JSON
        description: Human-readable description
        tool_name: Name of the calling tool

    Returns:
        Path to the saved file
    """
    from osprey.mcp_server.common import resolve_workspace_root

    workspace = resolve_workspace_root() / category
    workspace.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{tool_name}_{ts}.json"
    filepath = workspace / filename

    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.debug("Saved %s output to %s", tool_name, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.interfaces.ariel.mcp.registry import initialize_ariel_registry
    from osprey.mcp_server.common import (
        initialize_workspace_singletons,
        prime_config_builder,
        resolve_workspace_root,
    )

    prime_config_builder()
    initialize_ariel_registry()

    workspace_root = resolve_workspace_root()
    logger.info("ARIEL workspace root: %s", workspace_root)
    initialize_workspace_singletons(workspace_root)

    # Import tool modules (each registers itself via @mcp.tool())
    from osprey.interfaces.ariel.mcp.tools import (  # noqa: F401
        browse,
        capabilities,
        entry,
        keyword_search,
        semantic_search,
        sql_query,
        status,
    )

    logger.info("ARIEL MCP server initialised with all tools registered")
    return mcp
