"""ARIEL MCP Server.

FastMCP server that exposes the full ARIEL logbook search service
as MCP tools for Claude Code. Independent from the main OSPREY MCP server.

Usage:
    python -m osprey.mcp_server.ariel
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from osprey.ariel_attachment_view import DEFAULT_VIEW_ENABLED, VIEW_ENABLED_KEY
from osprey.mcp_server.errors import make_error  # re-exported for ARIEL tools

if TYPE_CHECKING:
    import mcp.types as mt
    from fastmcp.tools import Tool

logger = logging.getLogger("osprey.mcp_server.ariel")

#: The tool ``ariel.attachments.view.enabled`` withholds.
ATTACHMENT_VIEW_TOOL = "attachment_view"


def attachment_view_offered() -> bool:
    """Return whether ``attachment_view`` is offered on this deployment.

    Reads ``ariel.attachments.view.enabled`` from the ARIEL context. With no
    context to read it from (not initialised, or initialised from a config
    without an ``ariel`` section) the key's default applies, so a process that
    only lists tools sees the full surface.
    """
    from osprey.mcp_server.ariel.server_context import get_ariel_context

    try:
        config = get_ariel_context().config
    except RuntimeError:
        return DEFAULT_VIEW_ENABLED
    return bool(config.attachments.view_enabled)


def check_attachment_view_offered() -> None:
    """Refuse with ``not_supported`` naming the key when the view is switched off."""
    if attachment_view_offered():
        return
    make_error(
        "not_supported",
        f"Viewing logbook pictures is off on this deployment: {VIEW_ENABLED_KEY} is false.",
        [
            f"An administrator can allow it with {VIEW_ENABLED_KEY}: true "
            "in the build profile, then rebuild."
        ],
        details={"key": VIEW_ENABLED_KEY, "value": False},
    )


class AttachmentViewOfferMiddleware(Middleware):
    """Leave ``attachment_view`` out of ``tools/list`` when the view is switched off.

    It hides the tool and does not refuse it. The refusal lives in the tool,
    so a call that reaches the server anyway gets an answer naming the key,
    not a bare "Unknown tool".
    """

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        """Return the listed tools, without ``attachment_view`` when it is not offered."""
        tools = await call_next(context)
        if attachment_view_offered():
            return tools
        return [tool for tool in tools if tool.name != ATTACHMENT_VIEW_TOOL]


# ---------------------------------------------------------------------------
# FastMCP server instance -- imported by every tool module
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "ariel",
    instructions=(
        "Search facility logbook entries and operational records. "
        "When an entry includes an `entry_url`, link to it verbatim; "
        "never construct, guess, or reuse another host to build a logbook "
        "entry URL yourself. "
        "An entry with `raw_text_truncated` shows only the start of its text; "
        "`raw_text_length` is the full length, and `entry_get` returns the whole entry."
    ),
)
mcp.add_middleware(AttachmentViewOfferMiddleware())

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
    from osprey.utils.config import localize_facility

    return (
        localize_facility(datetime.fromisoformat(start_date) if start_date else None),
        localize_facility(datetime.fromisoformat(end_date) if end_date else None),
    )


def build_entry_url(entry_id: str | None, source_system: str | None = None) -> str | None:
    """Render the config-driven canonical logbook entry URL, or ``None``.

    An egress transform mirroring ``to_facility_iso``: it reads the
    facility-supplied ``ariel.entry_url_template`` from the merged config and
    renders it with the URL-encoded ``entry_id`` so the agent links entries
    verbatim instead of inventing a URL.

    Returns ``None`` (emit no URL) when:

    - ``source_system`` marks an ARIEL-native entry not yet in the facility
      logbook (``ARIEL_NATIVE_SOURCE_SYSTEM``);
    - ``entry_id`` is empty/blank;
    - no ``ariel.entry_url_template`` is configured (a deployment that
      configures no template emits no ``entry_url`` and the agent shows
      plain IDs);
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
        if not isinstance(template, str):
            raise TypeError(f"expected a string, got {type(template).__name__}")
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


def serialize_entry(
    entry: Mapping[str, Any],
    *,
    text_limit: int,
    attachment_limit: int,
    attachment_rows: Sequence[Mapping[str, Any]] | None,
    model_id: str | None,
    file_source: bool,
    full_captions: bool = False,
    view_enabled: bool = True,
) -> dict[str, Any]:
    """Serialize an EnhancedLogbookEntry dict into a compact response dict.

    Timestamps are converted to the facility timezone for agent consumption.
    ``_score``, ``_matched_via`` and ``_matched_attachment_ids`` on the entry
    come out as ``score``, ``matched_via`` and ``matched_attachment_ids``.
    With ``view_enabled`` false the entry carries no attachment keys at all
    (no ``attachments``, ``attachment_count`` or ``matched_attachment_ids``).

    Args:
        entry: EnhancedLogbookEntry TypedDict (plain dict).
        text_limit: Characters of `raw_text` to include; a longer text is cut to this
            and marked with `raw_text_truncated` and `raw_text_length`.
        attachment_limit: Attachment summaries to include under ``attachments``;
            ``0`` omits them. ``attachment_count`` (only when above zero) counts
            them all.
        attachment_rows: The entry's ``attachment_files`` rows, or None when the
            store holds no copy state.
        model_id: The configured caption model id.
        file_source: Whether the entry's source resolves relative attachment paths.
        full_captions: Emit captions and visible text uncut.
        view_enabled: ``ariel.attachments.view.enabled``.

    Returns:
        Serialized dict suitable for JSON response.
    """
    from osprey.services.ariel_search.attachments.summaries import build_attachment_summaries
    from osprey.services.ariel_search.models import entry_text_fields
    from osprey.utils.config import to_facility_iso

    ts = to_facility_iso(entry["timestamp"])

    result = {
        "entry_id": entry["entry_id"],
        "timestamp": ts,
        "author": entry.get("author", ""),
        "source_system": entry["source_system"],
        **entry_text_fields(entry["raw_text"], text_limit, field="raw_text"),
        "summary": entry.get("summary"),
    }
    entry_url = build_entry_url(entry["entry_id"], entry["source_system"])
    if entry_url is not None:
        result["entry_url"] = entry_url
    if "_score" in entry:
        result["score"] = entry["_score"]
    if "_matched_via" in entry:
        result["matched_via"] = list(entry["_matched_via"])
    if not view_enabled:
        return result
    if "_matched_attachment_ids" in entry:
        result["matched_attachment_ids"] = list(entry["_matched_attachment_ids"])

    summaries = build_attachment_summaries(
        entry,
        attachment_rows,
        None,
        entry.get("_matched_attachment_ids", ()),
        file_source=file_source,
        full_captions=full_captions,
        model_id=model_id,
    )
    if summaries:
        result["attachment_count"] = len(summaries)
        kept = summaries[: max(attachment_limit, 0)]
        if kept:
            result["attachments"] = kept
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
        attachment,
        browse,
        capabilities,
        entry,
        hybrid_search,
        keyword_search,
        publish,
        semantic_search,
        sql_query,
        status,
    )

    logger.info("ARIEL MCP server initialised with all tools registered")
    return mcp
