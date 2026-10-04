"""MCP tools for one stored logbook picture: attachment_view and attachment_to_artifact.

``attachment_view`` answers with two content blocks: a JSON text block
describing the picture and an image block carrying its stored display
rendition. ``attachment_to_artifact`` copies that same rendition into the
artifact gallery. Both serve stored bytes only; neither fetches, renders or
prepares a picture.

Every error message is built from server values (copy status and skip codes)
alone. The requested id is checked against the attachment id pattern before
any lookup and is never echoed; a stored filename or declared type reaches an
error only through ``details``, made inert first.
"""

import base64
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, NoReturn

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent

from osprey.mcp_server.ariel.server import (
    ATTACHMENT_TO_ARTIFACT_TOOL,
    check_attachment_view_offered,
    make_error,
    mcp,
)
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.services.ariel_search.attachments import ATTACHMENT_ID_RE
from osprey.services.ariel_search.attachments.compose import (
    FILENAME_MAX_CHARS,
    _inert,
    caption_model_id,
)
from osprey.services.ariel_search.attachments.formats import is_viewable
from osprey.services.ariel_search.attachments.summaries import (
    build_attachment_summaries,
    file_source_for,
)

logger = logging.getLogger("osprey.mcp_server.ariel.tools.attachment")

#: The one validation message; it never carries the rejected input.
INVALID_ID_MESSAGE = "attachment_id is not a valid attachment id"

#: Appended when the next sync prepares the picture.
NEXT_SYNC_NOTE = "it will be prepared on the next sync"

_FIND_IDS = "Read attachment_id values from the attachments of entry_get or a search result."


def _not_found() -> NoReturn:
    make_error(
        "not_found",
        "No stored picture has this attachment_id.",
        [_FIND_IDS],
    )


def _prepared_on_next_sync(row: Mapping[str, Any]) -> bool:
    """Whether the next sync prepares this row: pending, or copied without a rendition."""
    status = row.get("copy_status")
    if status == "pending":
        return True
    return (
        status == "copied"
        and row.get("skip_reason") is None
        and row.get("rendition_sha256") is None
    )


def _not_available(row: Mapping[str, Any]) -> NoReturn:
    """Raise ``no_results`` for a stored row that cannot be shown."""
    message = f"picture not available: copy_status={_inert(row.get('copy_status'), 40)}"
    skip_reason = row.get("skip_reason")
    if skip_reason is not None:
        message += f", skip_reason={_inert(skip_reason, 40)}"
    if _prepared_on_next_sync(row):
        message += f"; {NEXT_SYNC_NOTE}"
    make_error(
        "no_results",
        message,
        ["Use the entry's text and the picture's caption, if any, instead."],
        details={
            "filename": _inert(row.get("filename"), FILENAME_MAX_CHARS),
            "mime_type": _inert(row.get("mime_type"), FILENAME_MAX_CHARS),
        },
    )


async def _read_stored_row(repository: Any, attachment_id: str) -> dict[str, Any] | None:
    """Read an attachment's row columns by id, without the original or the rendition.

    Answers the rows ``get_rendition`` leaves out (pending, copied without a
    rendition, skipped, failed), so the tool can say why a picture cannot be
    shown. None when there is no such row or the store lacks the copy state.
    """
    from psycopg.rows import dict_row

    from osprey.services.ariel_search.database.repository import ATTACHMENT_ROW_COLUMNS

    if not (await repository.schema_facts()).has_copy_state:
        return None
    async with repository.pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                SELECT {", ".join(ATTACHMENT_ROW_COLUMNS)}
                FROM attachment_files
                WHERE attachment_id = %(attachment_id)s
                """,
                {"attachment_id": attachment_id},
            )
            row = await cur.fetchone()
    return dict(row) if row else None


def _servable(row: Mapping[str, Any]) -> bool:
    return (
        is_viewable(row)
        and bool(row.get("rendition_bytes"))
        and isinstance(row.get("rendition_mime"), str)
    )


def _check_id(attachment_id: Any) -> None:
    """Refuse a malformed id with ``validation_error``, never echoing it."""
    if not isinstance(attachment_id, str) or re.fullmatch(ATTACHMENT_ID_RE, attachment_id) is None:
        make_error("validation_error", INVALID_ID_MESSAGE, [_FIND_IDS])


async def _viewable_picture(
    attachment_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return ``(row, entry, summary)`` for a viewable stored picture.

    ``row`` is the ``get_rendition`` row with its ``rendition_bytes``, ``entry``
    the entry it belongs to, and ``summary`` its attachment summary with the
    caption in full. Raises ``not_found`` for an unknown id and ``no_results``
    for a picture that is not viewable.
    """
    from osprey.services.ariel_search.database.repository import (
        warn_attachment_schema_gap_once,
    )
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    registry = get_ariel_context()
    service = await registry.service()
    repository = service.repository

    # A failing reader reads as a store without copy state: nothing to show.
    try:
        row = await repository.get_rendition(attachment_id)
    except DatabaseQueryError:
        warn_attachment_schema_gap_once()
        row = None
    if not row:
        # No rendition: read the row alone to tell an unknown id from an unviewable one.
        try:
            row = await _read_stored_row(repository, attachment_id)
        except Exception:
            logger.warning("the attachment row could not be read", exc_info=True)
            row = None
    if not row:
        _not_found()
    if not _servable(row):
        _not_available(row)

    entry = await repository.get_entry(row["entry_id"])
    if not entry:
        _not_found()
    config = registry.config
    metadata = {key: value for key, value in row.items() if key != "rendition_bytes"}
    summaries = build_attachment_summaries(
        entry,
        [metadata],
        None,
        (),
        file_source=file_source_for(config),
        full_captions=True,
        model_id=caption_model_id(config),
    )
    summary = next((item for item in summaries if item.get("attachment_id") == attachment_id), None)
    if summary is None:
        _not_found()
    return dict(row), dict(entry), summary


@mcp.tool()
async def attachment_view(attachment_id: str) -> ToolResult:
    """Look at a picture attached to a logbook entry.

    Returns the picture itself, so you can see it, together with its summary.
    Only a viewable picture can be shown: one whose copy finished
    (`copy_status` "copied"), that was not skipped, that is an image, and whose
    display rendition is stored. Attachment summaries mark this as `viewable`;
    call this tool only for attachments with `viewable: true`.

    Args:
        attachment_id: The `attachment_id` of an attachment summary, as listed
            by entry_get or a search result (for example "att-0123456789abcdef01234567").

    Returns:
        A JSON block with the attachment summary (filename, type, caption in
        full and its `caption_source`) plus `entry_id`, `source_url`,
        `size_bytes` (the original's size, null when not stored),
        `rendition_size` and `rendition_sha256`, followed by the picture.
        Errors: validation_error for a malformed id, not_found for an unknown
        attachment, no_results for a picture that is not viewable,
        not_supported when the deployment has picture viewing switched off.
    """
    check_attachment_view_offered()
    _check_id(attachment_id)

    try:
        row, entry, summary = await _viewable_picture(attachment_id)
        rendition = bytes(row["rendition_bytes"])
        payload = {
            **summary,
            "entry_id": entry["entry_id"],
            "source_url": summary.get("url"),
            "size_bytes": row.get("size_bytes"),
            "rendition_size": len(rendition),
            "rendition_sha256": row.get("rendition_sha256"),
        }
        return ToolResult(
            content=[
                TextContent(type="text", text=json.dumps(payload, default=str)),
                ImageContent(
                    type="image",
                    data=base64.b64encode(rendition).decode("ascii"),
                    mimeType=row["rendition_mime"],
                ),
            ]
        )

    except ToolError:
        raise
    except Exception:
        logger.exception("attachment_view failed")
        make_error(
            "internal_error",
            "Failed to read the picture.",
            ["Check ARIEL database connectivity."],
        )


#: The artifact category a logbook picture is filed under in the gallery.
PICTURE_ARTIFACT_CATEGORY = "visualization"

_RENDITION_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _artifact_filename(filename: Any, rendition_mime: str) -> str:
    """A safe stored filename: the original's stem with the rendition's extension."""
    from pathlib import PurePosixPath

    from osprey.stores.artifact_store import _slugify

    stem = PurePosixPath(str(filename or "picture").replace("\\", "/")).stem
    return f"{_slugify(stem) or 'picture'}{_RENDITION_EXTENSIONS.get(rendition_mime, '.img')}"


def _provenance(entry_id: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    """The picture's provenance, kept in the artifact's metadata."""
    provenance: dict[str, Any] = {
        "entry_id": entry_id,
        "attachment_id": summary.get("attachment_id"),
        "filename": summary.get("filename"),
    }
    if summary.get("caption"):
        provenance["caption"] = summary["caption"]
        provenance["caption_source"] = summary.get("caption_source")
    return provenance


def _description(provenance: Mapping[str, Any]) -> str:
    lines = [
        f"Logbook picture {provenance['attachment_id']} from entry {provenance['entry_id']} "
        f"({_inert(provenance.get('filename'), FILENAME_MAX_CHARS)})."
    ]
    if provenance.get("caption"):
        source = provenance.get("caption_source") or "unknown"
        lines.append(f"Caption ({_inert(source, 40)}): {provenance['caption']}")
    return "\n".join(lines)


def _focus_in_gallery(artifact_id: str) -> bool:
    """Select the artifact in the gallery; whether the gallery accepted it."""
    import urllib.error

    from osprey.mcp_server.http import _post_json_with_response, gallery_url

    try:
        status, _ = _post_json_with_response(
            f"{gallery_url()}/api/focus", {"artifact_id": artifact_id}
        )
    except (urllib.error.URLError, OSError) as exc:
        logger.info("the artifact gallery could not be focused: %s", exc)
        return False
    return 200 <= status < 300


@mcp.tool()
async def attachment_to_artifact(attachment_id: str) -> str:
    """Keep a logbook picture in the artifact gallery and show it there.

    Copies the picture's prepared display rendition (the copy attachment_view
    returns, never the original upload) into the gallery as an image
    artifact, with the entry id, attachment id, filename and caption recorded
    with it, and selects it in the gallery. The same picture saved again
    returns the artifact it already has. Only a `viewable` picture can be
    kept; use this rather than redrawing a logbook plot from its caption.

    Args:
        attachment_id: The `attachment_id` of an attachment summary, as listed
            by entry_get or a search result (for example "att-0123456789abcdef01234567").

    Returns:
        JSON with `artifact_id`, `title`, `artifact_type`, `category`,
        `gallery_url`, `entry_id`, `attachment_id`, `created` (false when the
        picture was already in the gallery) and `focused` (whether the gallery
        selected it). Errors: validation_error for a malformed id, not_found
        for an unknown attachment, no_results for a picture that is not
        viewable, not_supported when the deployment has picture viewing
        switched off.
    """
    check_attachment_view_offered()
    _check_id(attachment_id)

    try:
        row, entry, summary = await _viewable_picture(attachment_id)
        from osprey.mcp_server.http import gallery_url
        from osprey.stores.artifact_store import get_artifact_store

        entry_id = str(entry["entry_id"])
        rendition = bytes(row["rendition_bytes"])
        rendition_mime = str(row["rendition_mime"])
        sha256 = hashlib.sha256(rendition).hexdigest()
        provenance = _provenance(entry_id, summary)
        filename = _inert(summary.get("filename"), FILENAME_MAX_CHARS) or "picture"
        title = f"{filename} (entry {entry_id})"

        store = get_artifact_store()
        known = {e.id for e in store.list_entries(tool_filter=ATTACHMENT_TO_ARTIFACT_TOOL)}
        artifact = store.save_or_touch_by_sha256(
            sha256,
            origin="",
            save_kwargs={
                "file_content": rendition,
                "filename": _artifact_filename(summary.get("filename"), rendition_mime),
                "artifact_type": "image",
                "title": title,
                "description": _description(provenance),
                "mime_type": rendition_mime,
                "tool_source": ATTACHMENT_TO_ARTIFACT_TOOL,
                "metadata": {"sha256": sha256, "logbook_picture": provenance},
                "category": PICTURE_ARTIFACT_CATEGORY,
            },
        )
        focused = _focus_in_gallery(artifact.id)
        response = artifact.to_tool_response(gallery_url=gallery_url())
        response.update(
            {
                "entry_id": entry_id,
                "attachment_id": attachment_id,
                "created": artifact.id not in known,
                "focused": focused,
            }
        )
        return json.dumps(response, default=str)

    except ToolError:
        raise
    except Exception:
        logger.exception("attachment_to_artifact failed")
        make_error(
            "internal_error",
            "Failed to keep the picture in the gallery.",
            ["Check ARIEL database connectivity and the artifact store."],
        )
