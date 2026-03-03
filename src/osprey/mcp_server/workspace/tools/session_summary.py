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

from osprey.mcp_server.errors import make_error
from osprey.utils.workspace import resolve_workspace_root
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
    channels = []
    summary = entry.summary or {}
    access = entry.access_details or {}

    # Archiver data stores channels in summary or access_details
    for source in [summary, access]:
        for key in ("channels", "columns", "channel_names", "pvs"):
            val = source.get(key)
            if isinstance(val, list):
                channels.extend(str(c) for c in val)

    return list(dict.fromkeys(channels))


@mcp.tool()
async def session_summary() -> str:
    """Return a compact inventory of all data and artifacts in this session.

    Scans the unified artifact store to produce a summary suitable for
    planning a session report. No parameters needed.

    Returns:
        JSON with entries[], and totals.
    """
    try:
        workspace_root = resolve_workspace_root()
    except Exception as e:
        return json.dumps(
            make_error(
                "internal_error",
                f"Could not resolve workspace root: {e}",
            )
        )

    # --- Unified artifact store ---
    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=workspace_root)
    all_entries = store.list_entries()

    entries = []
    total_bytes = 0
    for entry in all_entries:
        size = entry.size_bytes or _safe_file_size(store.get_file_path(entry.id) or "")
        total_bytes += size
        channels = _extract_channels(entry)
        entries.append(
            {
                "id": entry.id,
                "tool": entry.tool_source,
                "category": entry.category,
                "artifact_type": entry.artifact_type,
                "title": entry.title,
                "description": entry.description,
                "size_bytes": size,
                "channels": channels,
                "timestamp": entry.timestamp,
            }
        )

    # --- Totals ---
    category_counts = dict(Counter(e["category"] for e in entries if e["category"]))
    tool_counts = dict(Counter(e["tool"] for e in entries if e["tool"]))
    artifact_type_counts = dict(Counter(e["artifact_type"] for e in entries if e["artifact_type"]))

    result = {
        "entries": entries,
        "totals": {
            "entry_count": len(entries),
            "total_bytes": total_bytes,
            "categories": category_counts,
            "tools_used": tool_counts,
            "artifact_types": artifact_type_counts,
        },
    }

    return json.dumps(result, indent=2)
