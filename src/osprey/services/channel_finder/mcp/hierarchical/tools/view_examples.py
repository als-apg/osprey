"""MCP tool: view_examples — query operator-verified search examples.

Returns prior navigation paths from the FeedbackStore so the agent can
shortcut hierarchy exploration for queries similar to previously verified ones.
"""

import json
import logging

from osprey.services.channel_finder.feedback.formatters import (
    format_failure,
    format_success,
)
from osprey.services.channel_finder.mcp.hierarchical.registry import get_cf_hier_registry
from osprey.services.channel_finder.mcp.hierarchical.server import mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.hierarchical.tools.view_examples")


@mcp.tool()
def view_examples(keywords: str | None = None) -> str:
    """View operator-verified search examples from prior sessions.

    Call this FIRST before navigating the hierarchy. Pass a broad
    comma-separated keyword list extracted from the user's query
    (e.g. "magnets, corrector, horizontal" or "BPM, beam position").

    Args:
        keywords: Comma-separated keywords from the user's query.
                  Use broad terms to avoid missing relevant examples.
                  If omitted, returns all available examples.

    Returns:
        JSON with keyword-matched examples (if keywords provided)
        and all example summaries.
    """
    registry = get_cf_hier_registry()
    store = registry.feedback_store

    if store is None:
        return json.dumps({"examples": [], "message": "No feedback data available yet."})

    result: dict = {}

    # Keyword-based search
    if keywords:
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        facility = registry.facility_name

        # Exact match via SHA-256 key lookup
        exact_hints = store.get_hints(keywords, facility)
        if exact_hints:
            result["exact_match"] = [
                format_success(
                    {"selections": h["selections"], "channel_count": h["channel_count"]}
                )
                for h in exact_hints
            ]

        # Keyword overlap search
        keyword_matches = store.search_by_keywords(keyword_list)
        if keyword_matches:
            formatted = []
            for match in keyword_matches:
                entry_lines = [f'**"{match["query"]}"** (score: {match["score"]})']
                for s in match.get("successes", []):
                    entry_lines.append(format_success(s))
                for f in match.get("failures", []):
                    entry_lines.append(format_failure(f))
                formatted.append("\n".join(entry_lines))
            result["keyword_matches"] = formatted

    # Always include full summary list
    all_keys = store.list_keys()
    if all_keys:
        all_examples = []
        for summary in all_keys:
            entry = store.get_entry(summary["key"])
            if entry is None:
                continue
            entry_lines = [f'**"{summary["query"]}"** ({summary["facility"]})']
            for s in entry.get("successes", []):
                entry_lines.append(format_success(s))
            for f in entry.get("failures", []):
                entry_lines.append(format_failure(f))
            all_examples.append("\n".join(entry_lines))
        result["all_examples"] = all_examples
    else:
        result["all_examples"] = []

    if not result.get("exact_match") and not result.get("keyword_matches") and not all_keys:
        result["message"] = "No feedback data available yet."

    return json.dumps(result)
