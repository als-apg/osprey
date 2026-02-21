"""MCP tool: session_summary.

Returns a compact inventory of all data context entries and artifacts
in the current workspace. Designed for the session report workflow —
gives the report generator a quick overview of what data is available
without needing to read individual files.
"""

import json
import logging
from pathlib import Path

from osprey.mcp_server.common import make_error, resolve_workspace_root
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.session_summary")


def _safe_file_size(path: str | Path) -> int:
    """Return file size in bytes, or 0 if the file doesn't exist."""
    try:
        return Path(path).stat().st_size
    except (OSError, ValueError):
        return 0


def _extract_channels(entry) -> list[str]:
    """Extract channel names from a data context entry's summary or access_details."""
    channels = []
    summary = entry.summary or {}
    access = entry.access_details or {}

    # Archiver data stores channels in summary or access_details
    for source in [summary, access]:
        for key in ("channels", "columns", "channel_names", "pvs"):
            val = source.get(key)
            if isinstance(val, list):
                channels.extend(str(c) for c in val)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique


@mcp.tool()
async def session_summary() -> str:
    """Return a compact inventory of all data and artifacts in this session.

    Scans the data context index and artifact index to produce a summary
    suitable for planning a session report. No parameters needed.

    Returns:
        JSON with data_entries[], artifacts[], and totals.
    """
    try:
        workspace_root = resolve_workspace_root()
    except Exception as e:
        return json.dumps(make_error(
            "internal_error",
            f"Could not resolve workspace root: {e}",
        ))

    # --- Data context entries ---
    from osprey.mcp_server.data_context import DataContext

    ctx = DataContext(workspace_root=workspace_root)
    entries = ctx.list_entries()

    data_entries = []
    total_data_bytes = 0
    for entry in entries:
        size = entry.size_bytes or _safe_file_size(entry.data_file)
        total_data_bytes += size
        channels = _extract_channels(entry)
        data_entries.append({
            "id": entry.id,
            "tool": entry.tool,
            "data_type": entry.data_type,
            "description": entry.description,
            "size_bytes": size,
            "channels": channels,
            "timestamp": entry.timestamp,
        })

    # --- Artifacts ---
    from osprey.mcp_server.artifact_store import ArtifactStore

    art_store = ArtifactStore(workspace_root=workspace_root)
    artifacts_list = art_store.list_entries()

    artifacts = []
    total_artifact_bytes = 0
    for art in artifacts_list:
        size = art.size_bytes
        total_artifact_bytes += size
        artifacts.append({
            "id": art.id,
            "type": art.artifact_type,
            "title": art.title,
            "size_bytes": size,
            "timestamp": art.timestamp,
        })

    # --- Totals ---
    data_type_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    for de in data_entries:
        dt = de["data_type"]
        data_type_counts[dt] = data_type_counts.get(dt, 0) + 1
        t = de["tool"]
        tool_counts[t] = tool_counts.get(t, 0) + 1

    artifact_type_counts: dict[str, int] = {}
    for a in artifacts:
        at = a["type"]
        artifact_type_counts[at] = artifact_type_counts.get(at, 0) + 1

    result = {
        "data_entries": data_entries,
        "artifacts": artifacts,
        "totals": {
            "data_entry_count": len(data_entries),
            "artifact_count": len(artifacts),
            "total_data_bytes": total_data_bytes,
            "total_artifact_bytes": total_artifact_bytes,
            "data_types": data_type_counts,
            "tools_used": tool_counts,
            "artifact_types": artifact_type_counts,
        },
    }

    return json.dumps(result, indent=2)
