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
    instructions=(
        "Search facility logbook entries and operational records. "
        "When an entry includes an `entry_url`, link to it verbatim; "
        "never construct, guess, or reuse another host to build a logbook "
        "entry URL yourself."
    ),
)

# The source_system value ARIEL stamps on its own natively-created entries
# (see tools/entry.py entry_create direct mode). Such entries are not (yet) in
# the facility logbook, so no canonical entry_url exists for them.
ARIEL_NATIVE_SOURCE_SYSTEM = "ARIEL MCP"

# One-time guard so a malformed template does not spam the per-entry hot path.
_entry_url_template_warned = False


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


def build_entry_url(entry_id: str | None, source_system: str | None = None) -> "str | None":
    """Render the config-driven canonical logbook entry URL, or ``None``.

    An egress transform mirroring ``to_facility_iso``: it reads the
    facility-supplied ``ariel.entry_url_template`` from the merged config and
    renders it with the URL-encoded ``entry_id`` so the agent links entries
    verbatim instead of inventing a URL.

    Returns ``None`` (emit no URL) when:

    - ``source_system`` marks an ARIEL-native entry not yet in the facility
      logbook (``ARIEL_NATIVE_SOURCE_SYSTEM``);
    - ``entry_id`` is empty/blank;
    - no ``ariel.entry_url_template`` is configured (non-ALS / unconfigured
      deployments emit no ``entry_url`` and the agent shows plain IDs);
    - the template is malformed (fail-safe — this runs per-entry on the search
      hot path, so a one-character typo in the template must degrade to "no URL",
      never crash a read).
    """
    from urllib.parse import quote

    from osprey.utils.config import get_config_value

    if source_system == ARIEL_NATIVE_SOURCE_SYSTEM:
        return None
    if not entry_id or not str(entry_id).strip():
        return None

    try:
        template = get_config_value("ariel.entry_url_template", None)
    except Exception:
        # Config not loaded/resolvable (e.g. no config.yml): treat as an
        # unconfigured deployment — emit no URL rather than crash the read.
        return None
    if not template:
        return None

    try:
        return template.format(entry_id=quote(str(entry_id), safe=""))
    except Exception:
        global _entry_url_template_warned
        if not _entry_url_template_warned:
            _entry_url_template_warned = True
            logger.warning(
                "Malformed ariel.entry_url_template %r (needs a single {entry_id} "
                "placeholder); emitting no entry_url.",
                template,
            )
        return None


def serialize_entry(entry: dict, text_limit: int = 300) -> dict:
    """Serialize an EnhancedLogbookEntry dict into a compact response dict.

    Timestamps are converted to the facility timezone for agent consumption.

    Args:
        entry: EnhancedLogbookEntry TypedDict (plain dict).
        text_limit: Maximum characters of raw_text to include.

    Returns:
        Serialized dict suitable for JSON response.
    """
    from osprey.utils.config import to_facility_iso

    ts = to_facility_iso(entry["timestamp"])

    result = {
        "entry_id": entry["entry_id"],
        "timestamp": ts,
        "author": entry.get("author", ""),
        "source_system": entry["source_system"],
        "raw_text": entry["raw_text"][:text_limit],
        "summary": entry.get("summary"),
    }
    entry_url = build_entry_url(entry["entry_id"], entry["source_system"])
    if entry_url is not None:
        result["entry_url"] = entry_url
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

    # Session working root used by other tools at call time; the artifact
    # store itself is rooted at the shared data root inside
    # initialize_workspace_singletons().
    logger.info("ARIEL workspace root: %s", resolve_workspace_root())
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
