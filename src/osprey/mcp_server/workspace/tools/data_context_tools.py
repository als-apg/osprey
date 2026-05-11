"""MCP tools: data_list, data_read, data_delete.

Provides Claude with structured access to the OSPREY data store —
listing entries, reading full data content, and deleting entries.

All data is stored in the unified ArtifactStore.
"""

import json
import logging

from fastmcp.exceptions import ToolError
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.data_context_tools")


@mcp.tool()
async def data_list(
    tool_filter: str | None = None,
    category_filter: str | None = None,
    last_n: int | None = None,
    source_agent_filter: str | None = None,
) -> str:
    """List all data currently available in the OSPREY workspace.

    This is the primary way to see what data has been collected by OSPREY
    tools. Each entry includes a compact summary and metadata.

    Args:
        tool_filter: Only show entries from this tool (e.g. "archiver_read").
        category_filter: Only show entries of this category (e.g. "archiver_data").
        last_n: Show only the most recent N entries.
        source_agent_filter: Only show entries from this agent
            (e.g. "logbook-search").

    Returns:
        JSON with the list of data entries.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entries = store.list_entries(
            tool_filter=tool_filter,
            category_filter=category_filter,
            last_n=last_n,
            source_agent_filter=source_agent_filter,
        )

        return json.dumps(
            {
                "total_entries": len(entries),
                "filters_applied": {
                    "tool": tool_filter,
                    "category": category_filter,
                    "last_n": last_n,
                    "source_agent": source_agent_filter,
                },
                "entries": [e.to_dict() for e in entries],
            },
            default=str,
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("data_list failed")
        return make_error(
                "internal_error",
                f"Failed to list data: {exc}",
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
async def data_read(entry_id: str) -> str:
    """Read the full data content of a data entry (small entries only).

    Returns the complete JSON payload stored in the data file when the file
    is at most 100 KB. Larger files raise ``file_too_large`` with a
    structured preview (schema + first/last rows when applicable) and
    suggestions for processing the data via the Python sandbox instead.

    Bulk data should be loaded through ``execute`` (``with open(path) as
    f: json.load(f)``) or via the ``data_source=`` parameter on the plot
    tools, both of which keep the payload out of the agent context.

    Args:
        entry_id: ID of the artifact/data entry to read.

    Returns:
        The full JSON data content of the entry.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.get_entry(entry_id)

        if entry is None:
            return make_error(
                    "not_found",
                    f"Data entry '{entry_id}' not found.",
                    ["Use data_list to see available entries."],
                )

        # Get the data file path
        file_path = store.get_file_path(entry_id)
        if file_path is None or not file_path.exists():
            return make_error(
                    "file_not_found",
                    f"Data file for entry '{entry_id}' is missing from disk.",
                    [
                        "The data file may have been deleted externally.",
                        "Use data_list to find other entries.",
                    ],
                )

        size = file_path.stat().st_size
        if size > _MAX_INLINE_SIZE:
            preview = _build_oversize_preview(file_path, size)
            return make_error(
                    "file_too_large",
                    (
                        f"Data file for entry '{entry_id}' is {size:,} bytes "
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
                            "For visualization, pass entry_id as `data_source=` to "
                            "`create_static_plot` or `create_interactive_plot` — the "
                            "plot tool auto-loads the data inside its sandbox."
                        ),
                        (
                            "Use `archiver_downsample` to reduce row count before "
                            "reading if this is timeseries data."
                        ),
                    ],
                    details={
                        "entry_id": entry_id,
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
        logger.exception("data_read failed")
        return make_error(
                "internal_error",
                f"Failed to read data entry: {exc}",
                ["Check that the _agent_data directory is accessible."],
            )


@mcp.tool()
async def data_delete(entry_id: str) -> str:
    """Delete a data entry from the OSPREY workspace.

    Removes both the data file and its index entry.

    Args:
        entry_id: ID of the data entry to delete.

    Returns:
        JSON confirmation of deletion.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        deleted = store.delete_entry(entry_id)

        if not deleted:
            return make_error(
                    "not_found",
                    f"Data entry '{entry_id}' not found.",
                    ["Check the entry_id from a previous tool response."],
                )

        return json.dumps(
            {
                "status": "success",
                "entry_id": entry_id,
                "message": f"Data entry '{entry_id}' deleted.",
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("data_delete failed")
        return make_error(
                "internal_error",
                f"Failed to delete data entry: {exc}",
                ["Check that the _agent_data directory is accessible."],
            )
