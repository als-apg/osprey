"""MCP tools: artifact_list, artifact_read.

The read side of the artifact store: listing artifacts (optionally scoped to
one category, tool or agent) and reading a stored artifact's full content.

Everything the workspace server stores — plots, documents, screenshots,
archiver datasets, scan results — is one record type in one store, the
``ArtifactStore``. Datasets are not a separate namespace; they are artifacts
with a ``category``, so ``category=`` is how a caller narrows to them.
Mutations live in ``artifact_save`` (``artifact_save`` / ``artifact_delete`` /
``artifact_delete_all``).
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.artifact_query")


@mcp.tool()
async def artifact_list(
    tool: str | None = None,
    category: str | None = None,
    last_n: int | None = None,
    source_agent: str | None = None,
) -> str:
    """List the artifacts currently available in the OSPREY workspace.

    This is the primary way to see what OSPREY tools have produced — plots,
    documents and screenshots alongside archiver datasets and scan results.
    Each entry includes a compact summary, its ``category`` and its
    ``artifact_id``.

    Narrow to a data namespace with ``category`` (e.g. "archiver_data")
    rather than reaching for a different tool.

    Args:
        tool: Only show artifacts from this tool (e.g. "archiver_read").
        category: Only show artifacts of this category (e.g. "archiver_data").
        last_n: Show only the most recent N artifacts.
        source_agent: Only show artifacts from this agent
            (e.g. "logbook-search").

    Returns:
        JSON with the list of artifacts.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entries = store.list_entries(
            tool_filter=tool,
            category_filter=category,
            last_n=last_n,
            source_agent_filter=source_agent,
        )

        return json.dumps(
            {
                "total_entries": len(entries),
                "filters_applied": {
                    "tool": tool,
                    "category": category,
                    "last_n": last_n,
                    "source_agent": source_agent,
                },
                "entries": [e.to_dict() for e in entries],
            },
            default=str,
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_list failed")
        return make_error(
            "internal_error",
            f"Failed to list artifacts: {exc}",
            ["Check that the _agent_data directory is accessible."],
        )


# Inline read cap. Above this, the file is too large to safely return in a
# tool result (single MCP messages above ~1 MB break the Claude Agent SDK
# stdio transport's JSON buffer, and even smaller payloads pollute the
# model's context with bulk data that should be processed via `execute`).
_MAX_INLINE_SIZE = 100 * 1024

# Hard ceiling for over-cap preview generation. Files above this size skip
# JSON parsing entirely to avoid pathological memory use on the error path.
_PREVIEW_PARSE_LIMIT = 16 * 1024 * 1024


def _build_oversize_preview(file_path, size: int) -> dict:
    """Build a structured peek of an over-cap data file.

    Tries (in order): split-orient dataframe → top-level JSON shape → raw text
    head/tail. Always returns a dict with at least ``shape`` set.
    """
    preview: dict = {"shape": "unknown"}

    if size > _PREVIEW_PARSE_LIMIT:
        preview["shape"] = "skipped_too_large_for_preview"
        return preview

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        preview["shape"] = "binary"
        return preview

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        preview["shape"] = "text"
        preview["head"] = text[:512]
        preview["tail"] = text[-256:] if len(text) > 512 else ""
        return preview

    # Unwrap the legacy OSPREY metadata envelope (still present on old files).
    if isinstance(data, dict) and "_osprey_metadata" in data and "data" in data:
        data = data["data"]

    # Archiver-style wrapper: {"query": ..., "dataframe": {split-orient}}.
    df_block = data.get("dataframe") if isinstance(data, dict) else None
    if (
        isinstance(df_block, dict)
        and "index" in df_block
        and "columns" in df_block
        and "data" in df_block
    ):
        index = df_block.get("index") or []
        columns = df_block.get("columns") or []
        rows = df_block.get("data") or []
        preview["shape"] = "dataframe_split"
        preview["columns"] = columns
        preview["row_count"] = len(rows)
        preview["first_rows"] = [
            {"index": index[i] if i < len(index) else None, "values": rows[i]}
            for i in range(min(5, len(rows)))
        ]
        preview["last_rows"] = [
            {"index": index[i] if i < len(index) else None, "values": rows[i]}
            for i in range(max(min(5, len(rows)), len(rows) - 5), len(rows))
        ]
        return preview

    # Archiver payload from archiver_read: {"query": ..., "series": {channel:
    # {timestamps, values}}}. Guarded on every entry carrying `timestamps` so
    # an unrelated top-level `series` key falls through to generic handling.
    series = data.get("series") if isinstance(data, dict) else None
    if (
        isinstance(series, dict)
        and series
        and all(isinstance(v, dict) and "timestamps" in v for v in series.values())
    ):
        per_channel = {}
        total = 0
        for channel, entry in series.items():
            timestamps = entry.get("timestamps") or []
            values = entry.get("values") or []
            total += len(timestamps)
            per_channel[channel] = {
                "points": len(timestamps),
                "first": [timestamps[0], values[0]] if timestamps and values else None,
                "last": [timestamps[-1], values[-1]] if timestamps and values else None,
            }
        preview["shape"] = "timeseries_series"
        preview["channels"] = list(series.keys())
        preview["row_count"] = total
        preview["per_channel"] = per_channel
        if isinstance(data.get("query"), dict):
            preview["query"] = data["query"]
        return preview

    if isinstance(data, dict):
        preview["shape"] = "json_object"
        preview["top_level_keys"] = list(data.keys())
        return preview
    if isinstance(data, list):
        preview["shape"] = "json_array"
        preview["length"] = len(data)
        preview["first_items"] = data[:5]
        return preview

    preview["shape"] = "json_scalar"
    preview["value_preview"] = str(data)[:200]
    return preview


@mcp.tool()
async def artifact_read(artifact_id: str) -> str:
    """Read a stored artifact's full content (small artifacts only).

    Returns the complete JSON payload stored on disk when the file is at most
    100 KB. Larger files raise ``file_too_large`` with a structured preview
    (schema + first/last rows when applicable) and suggestions for processing
    the data via the Python sandbox instead.

    For the artifact's metadata and on-disk path without its content, use
    ``artifact_get``. Bulk data should be loaded through ``execute``
    (``with open(path) as f: json.load(f)``) or via the ``data_source=``
    parameter on the plot tools, both of which keep the payload out of the
    agent context.

    Args:
        artifact_id: ID of the artifact to read.

    Returns:
        The full JSON content of the artifact.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.get_entry(artifact_id)

        if entry is None:
            return make_error(
                "not_found",
                f"Artifact '{artifact_id}' not found.",
                ["Use artifact_list to see available artifacts."],
            )

        file_path = store.get_file_path(artifact_id)
        if file_path is None or not file_path.exists():
            return make_error(
                "file_not_found",
                f"File for artifact '{artifact_id}' is missing from disk.",
                [
                    "The file may have been deleted externally.",
                    "Use artifact_list to find other artifacts.",
                ],
            )

        size = file_path.stat().st_size
        if size > _MAX_INLINE_SIZE:
            preview = _build_oversize_preview(file_path, size)
            return make_error(
                "file_too_large",
                (
                    f"File for artifact '{artifact_id}' is {size:,} bytes "
                    f"(>{_MAX_INLINE_SIZE:,} bytes / "
                    f"{_MAX_INLINE_SIZE // 1024} KB)."
                ),
                [
                    (
                        "Load the file in the Python sandbox: pass the code "
                        f"`with open('{file_path}') as f: data = json.load(f)` "
                        "to the `execute` tool."
                    ),
                    (
                        "For visualization, pass artifact_id as `data_source=` to "
                        "`create_static_plot` or `create_interactive_plot` — the "
                        "plot tool auto-loads the data inside its sandbox."
                    ),
                    (
                        "Use `archiver_downsample` to reduce row count before "
                        "reading if this is timeseries data."
                    ),
                ],
                details={
                    "artifact_id": artifact_id,
                    "size_bytes": size,
                    "limit_bytes": _MAX_INLINE_SIZE,
                    "file_path": str(file_path),
                    "preview": preview,
                },
            )

        content = file_path.read_text(encoding="utf-8")
        return content

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_read failed")
        return make_error(
            "internal_error",
            f"Failed to read artifact: {exc}",
            ["Check that the _agent_data directory is accessible."],
        )
