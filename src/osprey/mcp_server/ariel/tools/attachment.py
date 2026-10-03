"""MCP tool: attachment_view -- look at a stored logbook picture.

The tool answers with two content blocks: a JSON text block describing the
picture and an image block carrying its stored display rendition. It serves
stored bytes only; it never fetches, renders or prepares a picture.

Every error message is built from server values (copy status and skip codes)
alone. The requested id is checked against the attachment id pattern before
any lookup and is never echoed; a stored filename or declared type reaches an
error only through ``details``, made inert first.
"""

import base64
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, NoReturn

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent

from osprey.mcp_server.ariel.server import check_attachment_view_offered, make_error, mcp
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
    if not isinstance(attachment_id, str) or re.fullmatch(ATTACHMENT_ID_RE, attachment_id) is None:
        make_error("validation_error", INVALID_ID_MESSAGE, [_FIND_IDS])

    try:
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
                logger.warning("attachment_view could not read the attachment row", exc_info=True)
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
        summary = next(
            (item for item in summaries if item.get("attachment_id") == attachment_id), None
        )
        if summary is None:
            _not_found()

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
