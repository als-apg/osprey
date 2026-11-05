"""MCP tool: session_summary.

Returns a compact inventory of all artifacts in the current workspace.
Designed for the session report workflow — gives the report generator a
quick overview of what data is available without needing to read
individual files.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.session_summary")


def _safe_file_size(path: str | Path) -> int:
    """Return file size in bytes, or 0 if the file doesn't exist."""
    try:
        return Path(path).stat().st_size
    except (OSError, ValueError):
        return 0


def _extract_channels(entry) -> list[str]:
    """Extract channel names from an artifact entry's summary or access_details."""
    channels: list[str] = []
    summary = entry.summary or {}
    access = entry.access_details or {}

    # Archiver data stores channels in summary or access_details
    for source in [summary, access]:
        for key in ("channels", "columns", "channel_names", "pvs"):
            val = source.get(key)
            if isinstance(val, list):
                channels.extend(str(c) for c in val)

    return list(dict.fromkeys(channels))


def entry_forms(store, entries) -> list[tuple[str, str]]:
    """The ``(form, label)`` each of *entries* is counted under.

    The gallery's one grouping level: a run head is ``runs``, an output its
    intrinsic form, an entry of a registered extension type its ``kind``. A
    run's other parts (its preview, notebook) are listed under their run in the
    gallery; an inventory still names each entry, so a part counts under its
    intrinsic form.
    """
    from osprey.mcp_server.workspace.tools.artifact_query import (
        extension_registry,
        group_of,
    )
    from osprey.stores.artifact_forms import form_label, form_of

    try:
        registry = extension_registry() or {}
    except Exception:  # a broken registry never breaks the inventory
        logger.debug("Extension registry unavailable", exc_info=True)
        registry = {}
    groups = store.run_groups()
    forms: list[tuple[str, str]] = []
    for entry in entries:
        key = group_of(entry, groups, registry) or form_of(entry, registry)
        ext = registry.get(key) if entry.kind and key == entry.kind else None
        forms.append((key, ext.label if ext is not None else form_label(key)))
    return forms


def form_totals(forms: list[tuple[str, str]]) -> list[dict]:
    """``[{key, label, count}]`` for *forms*, in first-seen order."""
    counts = Counter(forms)
    return [{"key": key, "label": label, "count": n} for (key, label), n in counts.items()]


@mcp.tool()
async def session_summary() -> str:
    """Return a compact inventory of all data and artifacts in this session.

    Scans the unified artifact store to produce a summary suitable for
    planning a session report. No parameters needed.

    Returns:
        JSON with entries[], and totals.
    """
    # --- Unified artifact store (shared-root singleton) ---
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    # The shipped example is not session output; a report must never cite it.
    all_entries = store.list_entries(exclude_examples=True)

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for entry, (form, label) in zip(all_entries, entry_forms(store, all_entries), strict=True):
        size = entry.size_bytes or _safe_file_size(store.get_file_path(entry.id) or "")
        total_bytes += size
        channels = _extract_channels(entry)
        entries.append(
            {
                "id": entry.id,
                "tool": entry.tool_source,
                "form": form,
                "form_label": label,
                "artifact_type": entry.artifact_type,
                "title": entry.title,
                "description": entry.description,
                "size_bytes": size,
                "channels": channels,
                "timestamp": entry.timestamp,
            }
        )

    # --- Totals ---
    tool_counts = dict(Counter(e["tool"] for e in entries if e["tool"]))
    artifact_type_counts = dict(Counter(e["artifact_type"] for e in entries if e["artifact_type"]))

    result = {
        "entries": entries,
        "totals": {
            "entry_count": len(entries),
            "total_bytes": total_bytes,
            "forms": form_totals([(e["form"], e["form_label"]) for e in entries]),
            "tools_used": tool_counts,
            "artifact_types": artifact_type_counts,
        },
    }

    return json.dumps(result, indent=2)
