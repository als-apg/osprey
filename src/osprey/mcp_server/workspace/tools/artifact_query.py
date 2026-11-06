"""MCP tools: artifact_list, artifact_read.

The read side of the artifact store: listing the artifacts this session owns
(or, with ``scope="all"``, every session's, each naming its owner), optionally
narrowed to one form or tool, and reading one artifact by id (metadata, path
and small content).

Everything the workspace server stores — plots, documents, screenshots,
archiver datasets, run results — is one record type in one store, the
``ArtifactStore``. Each artifact sits in exactly one group: a base form (Runs,
Plots, Trends, Tables, Documents, Dashboards, Images, Data) or a registered
extension type's key. The store computes it from what the writer recorded;
``form=`` is how a caller narrows to one. Mutations live in ``artifact_register`` (``artifact_register`` / ``artifact_delete`` /
``artifact_delete_all``).
"""

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.artifact_query")


def known_forms(registry: Mapping[str, Any]) -> list[str]:
    """Every form a caller may name: the base forms, then extension type keys."""
    from osprey.stores.artifact_forms import BASE_FORMS

    return [*BASE_FORMS, *(k for k in registry if k not in BASE_FORMS)]


def require_form(form: str, registry: Mapping[str, Any], extra: Iterable[str] = ()) -> None:
    """Refuse a form that is neither a base form, an extension key nor in *extra*."""
    valid = [*extra, *known_forms(registry)]
    if form not in valid:
        make_error(
            "validation_error",
            f"Unknown form '{form}'.",
            [f"Use one of: {', '.join(valid)}."],
        )


def extension_registry() -> dict[str, Any]:
    """The process's extension type registry (empty when it cannot load)."""
    from osprey.stores.type_registry import get_extension_types

    return get_extension_types()


#: ``artifact_list``'s default and largest ``last_n``.
_LIST_DEFAULT = 20
_LIST_MAX = 200
#: A row's description is the first line of the artifact's, cut to this length.
_ROW_DESCRIPTION_MAX = 160
#: The owner a ``scope="all"`` row names when no session owns it.
_EARLIER = "Earlier"
_DISPATCH_HINT = "; read the artifacts you were given by their ids (artifact_read)"


def _short(text: str) -> str:
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line if len(line) <= _ROW_DESCRIPTION_MAX else line[: _ROW_DESCRIPTION_MAX - 1] + "…"


def _row(entry: Any, form: str, label: str) -> dict[str, Any]:
    from osprey.stores.artifact_inventory import entry_channels

    row: dict[str, Any] = {
        "artifact_id": entry.id,
        "title": entry.title,
        "form": form,
        "form_label": label,
        "tool": entry.tool_source,
        "timestamp": entry.timestamp,
        "description": _short(entry.description),
    }
    channels = entry_channels(entry)
    if channels:
        row["channels"] = channels
    return row


@mcp.tool()
async def artifact_list(
    scope: Literal["session", "all"] = "session",
    form: str | None = None,
    tool: str | None = None,
    last_n: int = _LIST_DEFAULT,
) -> str:
    """List the artifacts this session made — plots, documents, runs, datasets.

    ``scope="session"`` (default) lists what the current conversation owns:
    the artifacts the gallery sidebar files under this session's row. After a
    ``/clear`` it starts empty. ``scope="all"`` lists every session's artifacts
    and names each row's ``owner`` (a transcript id, or ``"Earlier"``); reach
    for it only when asked about other or earlier sessions.

    Rows are compact, one per run (a run's code and outputs are parts of it),
    newest first; read one with ``artifact_read(artifact_id)``. Narrow with
    ``form``: "runs", "plots", "trends", "tables", "documents", "dashboards",
    "images", "data", or a registered extension type's key.

    Args:
        scope: "session" (this conversation) or "all" (every session).
        form: Only list rows in this form or extension type.
        tool: Only list rows from this tool (e.g. "archiver_read").
        last_n: How many rows to return, newest first (1-200, default 20).

    Returns:
        JSON with ``form_totals`` (the whole scope), ``total`` (after ``form``
        and ``tool``), ``omitted`` (cut by ``last_n``), ``entries``, ``session``,
        ``ledger`` ("ok" or "unavailable") and, when the session is unknown, the
        ``reason`` the list is empty.
    """
    try:
        from osprey.stores.artifact_inventory import form_totals, listed_rows
        from osprey.stores.artifact_scope import current_scope, owned_by, read_ledger
        from osprey.stores.artifact_store import get_artifact_store, operator_visible, owner_of

        if scope not in ("session", "all"):
            make_error(
                "validation_error",
                f"Unknown scope '{scope}'.",
                ['Use "session" (this conversation) or "all" (every session).'],
            )
        if (
            isinstance(last_n, bool)
            or not isinstance(last_n, int)
            or not (1 <= last_n <= _LIST_MAX)
        ):
            make_error(
                "validation_error",
                f"last_n must be between 1 and {_LIST_MAX}, got {last_n!r}.",
                [f"Use a last_n from 1 to {_LIST_MAX}."],
            )
        registry = extension_registry()
        if form:
            require_form(form, registry)

        store = get_artifact_store()
        current = current_scope(root=store._workspace)
        # The shipped example is the person's, never the agent's to list.
        entries = store.list_entries(exclude_examples=True)
        answer: dict[str, Any] = {"scope": scope, "session": current.transcript}
        if scope == "session":
            ledger, ledger_ok = current.ledger, current.ledger_ok
            in_scope = [e for e in entries if owned_by(e, current)]
            if current.transcript is None:
                reason = current.reason
                if current.kind == "dispatch":
                    reason += _DISPATCH_HINT
                answer["reason"] = reason
        else:
            ledger, ledger_ok = read_ledger()
            in_scope = [e for e in entries if operator_visible(e)]

        rows = list(reversed(listed_rows(store, in_scope, registry)))
        totals = form_totals((key, label) for _e, key, label in rows)
        if form:
            rows = [r for r in rows if r[1] == form]
        if tool:
            rows = [r for r in rows if r[0].tool_source == tool]
        shown = []
        for entry, key, label in rows[:last_n]:
            row = _row(entry, key, label)
            if scope == "all":
                row["owner"] = owner_of(entry, ledger if ledger_ok else None) or _EARLIER
            shown.append(row)

        answer.update(
            {
                "ledger": "ok" if ledger_ok else "unavailable",
                "total": len(rows),
                "omitted": len(rows) - len(shown),
                "form_totals": totals,
                "entries": shown,
            }
        )
        return json.dumps(answer, default=str)

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


def _read_metadata(store: Any, entry: Any, file_path: Any) -> dict[str, Any]:
    """The by-id metadata answer: fixed fields, never arbitrary metadata.

    ``logbook_picture`` is the one metadata shape projected (it reconnects a
    gallery image to ARIEL); everything else an artifact carries stays out.
    """
    from osprey.mcp_server.http import gallery_url
    from osprey.stores.artifact_store import listed_form, registry_or_default

    run_group = store.run_group(entry.exec_id) if entry.exec_id else None
    result: dict[str, Any] = {
        "artifact_id": entry.id,
        "title": entry.title,
        "description": entry.description,
        "artifact_type": entry.artifact_type,
        "mime_type": entry.mime_type,
        "size_bytes": entry.size_bytes,
        "timestamp": entry.timestamp,
        # A run's part is read as part of its run.
        "form": listed_form(entry, run_group, registry_or_default()) or "runs",
        "file_path": str(file_path.resolve()) if file_path else None,
        "gallery_url": gallery_url(),
    }
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    logbook_picture = metadata.get("logbook_picture")
    if isinstance(logbook_picture, dict):
        result["logbook_picture"] = logbook_picture
    return result


def _oversize_suggestions(file_path: Any) -> list[str]:
    """How to work with an artifact too large to return inline."""
    return [
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
            "Use `archiver_downsample` to reduce row count before reading if this is timeseries data."
        ),
    ]


@mcp.tool()
async def artifact_read(artifact_id: str) -> str:
    """Read a stored artifact by ID: its metadata, file path and, when small, its content.

    Always returns the artifact's metadata and absolute ``file_path`` (pass it
    to tools like ``graph_extract(image_path=...)``). ``content`` is included
    when the file is at most 100 KB and is text; a JSON file comes back as its
    parsed value. Otherwise ``content_omitted`` says why: ``"binary"`` (an
    image or other non-text file) or ``"too_large"``, which also carries a
    structured ``preview`` and ``suggestions`` for processing the data in the
    Python sandbox (``execute``) or via ``data_source=`` on the plot tools.

    Args:
        artifact_id: ID of the artifact to read.

    Returns:
        JSON with the artifact's metadata, ``file_path`` and ``content`` or
        ``content_omitted``.
    """
    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.get_entry(artifact_id)

        if entry is None:
            return make_error(
                "not_found",
                f"Artifact '{artifact_id}' not found.",
                ['Use artifact_list(scope="all") to see available artifacts.'],
            )

        file_path = store.get_file_path(artifact_id)
        result = _read_metadata(store, entry, file_path)
        if file_path is None or not file_path.exists():
            return make_error(
                "file_not_found",
                f"File for artifact '{artifact_id}' is missing from disk.",
                [
                    "The file may have been deleted externally.",
                    'Use artifact_list(scope="all") to find other artifacts.',
                ],
                details=result,
            )

        size = file_path.stat().st_size
        if size > _MAX_INLINE_SIZE:
            result.update(
                {
                    "content_omitted": "too_large",
                    "size_bytes": size,
                    "limit_bytes": _MAX_INLINE_SIZE,
                    "preview": _build_oversize_preview(file_path, size),
                    "suggestions": _oversize_suggestions(result["file_path"]),
                }
            )
            return json.dumps(result, default=str)

        try:
            text = file_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            result["content_omitted"] = "binary"
            return json.dumps(result, default=str)

        try:
            result["content"] = json.loads(text)
        except json.JSONDecodeError:
            result["content"] = text
        return json.dumps(result, default=str)

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("artifact_read failed")
        return make_error(
            "internal_error",
            f"Failed to read artifact: {exc}",
            ["Check that the _agent_data directory is accessible."],
        )
