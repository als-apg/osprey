"""MCP tools: entry_get + entry_create — logbook entry CRUD.

PROMPT-PROVIDER: Tool docstrings are static prompts visible to the agent.
  Future: source from FrameworkPromptProvider.get_logbook_search_prompt_builder()
  Facility-customizable: shift identifiers, attachment limits, logbook name
  conventions
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server import (
    ARIEL_NATIVE_SOURCE_SYSTEM,
    build_entry_url,
    make_error,
    mcp,
)
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.mcp_server.ariel.tools.search_envelope import serialize_page
from osprey.mcp_server.http import notify_agent_activity_async
from osprey.services.ariel_search.attachments.compose import caption_model_id
from osprey.services.ariel_search.attachments.summaries import (
    build_attachment_summaries,
    file_source_for,
)
from osprey.services.ariel_search.entry_fields import (
    MAX_LISTED_CHOICES,
    EntryFieldDeclarationError,
    EntryFieldError,
    entry_field_descriptors,
    resolve_entry_write,
    validate_entry_fields,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.ariel_search.search.base import ParameterDescriptor

logger = logging.getLogger("osprey.mcp_server.ariel.tools.entry")


def ariel_panel_url(route: str) -> str:
    """Return the ARIEL web page URL that opens ``route`` (the part after ``#``).

    The base defaults to the web terminal's origin-relative proxy path for the
    ARIEL panel. The web terminal embeds the panel at /panel/ariel and resolves
    this URL with `new URL(url, origin)`, so a relative path loads through the
    proxy in both the clickable link and the auto-focus iframe. An absolute
    container-internal address (e.g. 127.0.0.1:10300, the ariel slot at the
    default port base) is unreachable from the user's browser. Set
    ARIEL_WEB_URL to an absolute base only for standalone (non-proxied) ARIEL
    deployments.
    """
    base_url = os.environ.get("ARIEL_WEB_URL", "/panel/ariel")
    return f"{base_url}/#{route}"


def _focus_ariel_panel(url: str) -> None:
    """Ask the web terminal to show the ARIEL panel at ``url``; non-fatal without one."""
    try:
        from osprey.mcp_server.http import notify_panel_focus

        notify_panel_focus("ariel", url=url)
    except ToolError:
        raise
    except Exception:
        pass  # Non-fatal — web terminal may not be running


def _get_drafts_dir() -> Path:
    """Resolve the drafts directory at call time (not import time)."""
    from osprey.utils.workspace import resolve_shared_data_root

    return resolve_shared_data_root() / "drafts"


async def _resolve_artifacts(artifact_ids: list[str]) -> list[str]:
    """Resolve artifact IDs to file paths, auto-converting artifacts to logbook-friendly formats.

    Uses the converter registry to transform artifacts: rendered content (HTML,
    markdown, notebooks, JSON, text) is converted to PNG; images and PDFs pass
    through unchanged.

    Returns:
        List of absolute file paths ready for attachment.

    Raises:
        ValueError: If an artifact ID is not found.
    """
    import tempfile

    from osprey.agent_runner.artifact_resolve import resolve_artifact_path
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    resolved: list[str] = []
    output_dir = Path(tempfile.mkdtemp(prefix="ariel_convert_"))

    for aid in artifact_ids:
        resolved.append(await resolve_artifact_path(store, aid, output_dir))

    return resolved


@mcp.tool()
async def entry_get(
    entry_id: str,
) -> str:
    """Get a single logbook entry by its ID.

    Args:
        entry_id: The unique entry identifier.

    Returns:
        JSON with the full entry data, or a not_found error.
    """
    if not entry_id or not entry_id.strip():
        return make_error(
            "validation_error",
            "entry_id is required.",
            ["Provide a valid entry ID."],
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        entry = await service.repository.get_entry(entry_id)
        if not entry:
            return make_error(
                "not_found",
                f"Entry {entry_id} not found.",
                [
                    "Check the entry_id is correct.",
                    "Use keyword_search/semantic_search or browse to find valid entry IDs.",
                ],
            )

        from osprey.services.ariel_search.database.repository import read_attachment_rows

        config = registry.config
        view_enabled = config.attachments.view_enabled
        summaries: list[dict[str, Any]] = []
        if view_enabled:
            # A failing reader reads as a store without copy state: the fallback
            # summaries, never an error envelope.
            rows_map = await read_attachment_rows(service.repository, [entry["entry_id"]])
            summaries = build_attachment_summaries(
                entry,
                None if rows_map is None else rows_map.get(entry["entry_id"], []),
                None,
                (),
                file_source=file_source_for(config),
                full_captions=True,
                model_id=caption_model_id(config),
            )

        # TypedDict -- dict access, not attribute access. Localize the three
        # timestamp fields through the shared egress helper so single-entry get
        # matches search/browse (serialize_entry) instead of emitting raw UTC.
        from osprey.utils.config import to_facility_iso

        result: dict[str, Any] = {
            "entry_id": entry["entry_id"],
            "source_system": entry["source_system"],
            "timestamp": to_facility_iso(entry["timestamp"]),
            "author": entry.get("author", ""),
            "raw_text": entry["raw_text"],
            # With the view off the stored items go out as stored.
            "attachments": summaries if view_enabled else entry.get("attachments", []),
            "metadata": entry.get("metadata", {}),
            "summary": entry.get("summary"),
            "keywords": entry.get("keywords", []),
            "created_at": to_facility_iso(entry["created_at"]),
            "updated_at": to_facility_iso(entry["updated_at"]),
        }
        if summaries:
            result["attachment_count"] = len(summaries)
        entry_url = build_entry_url(entry["entry_id"], entry["source_system"])
        if entry_url is not None:
            result["entry_url"] = entry_url
        return json.dumps(result, default=str)

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("entry_get failed")
        return make_error(
            "internal_error",
            f"Failed to get entry: {exc}",
            ["Check ARIEL database connectivity."],
        )


@mcp.tool()
async def entries_by_ids(
    entry_ids: list[str],
) -> str:
    """Get multiple logbook entries by their IDs in a single call.

    Efficient batch retrieval for reading entries found via search. Each entry
    carries more of its text than a search result does; an entry cut short is
    marked `raw_text_truncated`, and `entry_get` returns it whole. Each entry
    lists only its first few attachment summaries, with `attachment_count`
    giving the total; call `entry_get` for the full attachment list.

    Args:
        entry_ids: List of entry IDs to retrieve (max 50 per call).

    Returns:
        JSON with list of found entries. May return fewer than requested
        if some IDs don't exist.
    """
    if not entry_ids:
        return make_error(
            "validation_error",
            "entry_ids list is empty.",
            ["Provide at least one entry ID."],
        )

    if len(entry_ids) > 50:
        return make_error(
            "validation_error",
            f"Too many entry IDs ({len(entry_ids)}). Maximum is 50 per call.",
            ["Split into multiple calls of 50 or fewer IDs."],
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        entries = await service.repository.get_entries_by_ids(entry_ids)

        # A batch read carries more of each entry than a search result; a cut entry says so.
        config = registry.config
        entries_out = await serialize_page(
            entries, config, service.repository, text_limit=config.entry_text.read_chars
        )

        return json.dumps(
            {
                "requested": len(entry_ids),
                "found": len(entries_out),
                "entries": entries_out,
            },
            default=str,
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("entries_by_ids failed")
        return make_error(
            "internal_error",
            f"Failed to get entries: {exc}",
            ["Check ARIEL database connectivity."],
        )


@mcp.tool()
async def entry_open(
    entry_id: str,
    attachment_id: str | None,
) -> str:
    """Show a logbook entry to the operator in the ARIEL panel.

    Opens the entry's detail card in the ARIEL web panel and brings the panel
    to the front. With `attachment_id` it also opens that picture enlarged,
    when the picture is one of this entry's and `viewable`. Use it whenever
    the operator asks to see an entry or one of its pictures; it reads no
    picture into this conversation and writes nothing.

    Args:
        entry_id: The entry to show, as listed by a search, browse or entry_get.
        attachment_id: Required, may be null. The `attachment_id` of this
            entry's picture whenever this conversation holds one (a picture
            viewed, or one a subagent's reply listed next to the entry id),
            even when the operator asks only for the entry; null only when no
            picture of this entry came up. For example
            "att-0123456789abcdef01234567".

    Returns:
        JSON with `entry_id`, `attachment_id`, `opened` ("entry" or
        "entry_and_picture"), `pictures` (the `attachment_id` and `filename`
        of each viewable picture of the entry; absent while the attachment
        view is off), the panel `url` (a link the
        operator can follow when no web terminal is running) and a `message`.
        Opened without `attachment_id`, the message names the entry's viewable
        pictures, so a call that left out the picture under discussion can be
        repeated with it. Errors:
        validation_error for a missing entry id or a malformed attachment id,
        not_found for an unknown entry or an attachment that is not the
        entry's.
    """
    import re
    from urllib.parse import quote

    from osprey.services.ariel_search.attachments import ATTACHMENT_ID_RE

    if not isinstance(entry_id, str) or not entry_id.strip():
        make_error("validation_error", "entry_id is required.", ["Provide a valid entry ID."])
    if attachment_id is not None and (
        not isinstance(attachment_id, str) or re.fullmatch(ATTACHMENT_ID_RE, attachment_id) is None
    ):
        make_error(
            "validation_error",
            "attachment_id is not a valid attachment id",
            ["Read attachment_id values from the attachments of entry_get or a search result."],
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        entry = await service.repository.get_entry(entry_id)
        if not entry:
            make_error(
                "not_found",
                f"Entry {entry_id} not found.",
                [
                    "Check the entry_id is correct.",
                    "Use keyword_search/semantic_search or browse to find valid entry IDs.",
                ],
            )

        config = registry.config
        view_enabled = config.attachments.view_enabled
        summaries: list[dict] = []
        if view_enabled or attachment_id is not None:
            from osprey.services.ariel_search.database.repository import read_attachment_rows

            rows_map = await read_attachment_rows(service.repository, [entry["entry_id"]])
            summaries = build_attachment_summaries(
                entry,
                None if rows_map is None else rows_map.get(entry["entry_id"], []),
                None,
                (),
                file_source=file_source_for(config),
                full_captions=False,
                model_id=caption_model_id(config),
            )
        pictures = [
            {"attachment_id": item["attachment_id"], "filename": item.get("filename")}
            for item in summaries
            if view_enabled and item.get("viewable") and item.get("attachment_id")
        ]

        route = f"entry?id={quote(entry['entry_id'], safe='')}"
        opened = "entry"
        message = f"The ARIEL panel shows entry {entry['entry_id']}."
        if attachment_id is None and pictures:
            listed = ", ".join(f"{p['filename']} ({p['attachment_id']})" for p in pictures)
            message += (
                f" Its pictures are listed on the entry card, none enlarged: {listed}. If this "
                "conversation holds the attachment_id of one of them, call entry_open again "
                "with it, so the operator sees that picture enlarged."
            )
        if attachment_id is not None:
            summary = next(
                (item for item in summaries if item.get("attachment_id") == attachment_id), None
            )
            if summary is None:
                make_error(
                    "not_found",
                    f"Entry {entry['entry_id']} has no attachment with this attachment_id.",
                    ["Read attachment_id values from this entry's attachments in entry_get."],
                )
            if summary.get("viewable"):
                route += f"&attachment={attachment_id}"
                opened = "entry_and_picture"
                message = (
                    f"The ARIEL panel shows entry {entry['entry_id']} with picture "
                    f"{attachment_id} enlarged."
                )
            else:
                message += (
                    f" Picture {attachment_id} is not viewable, so it is listed on the "
                    "entry card but not enlarged."
                )

        url = ariel_panel_url(route)
        _focus_ariel_panel(url)
        result: dict = {
            "entry_id": entry["entry_id"],
            "attachment_id": attachment_id,
            "opened": opened,
            "url": url,
            "message": f"{message} If no ARIEL panel is in view, open {url}",
        }
        if view_enabled:
            result["pictures"] = pictures
        return json.dumps(result)

    except ToolError:
        raise
    except Exception:
        logger.exception("entry_open failed")
        make_error(
            "internal_error",
            "Failed to open the entry.",
            ["Check ARIEL database connectivity."],
        )


def _entry_field_refusal(
    exc: EntryFieldError, descriptors: list["ParameterDescriptor"]
) -> NoReturn:
    """Raise the ``validation_error`` for one invalid entry field.

    The envelope names the field; for a ``select`` it also lists up to
    :data:`MAX_LISTED_CHOICES` of the allowed values.
    """
    details: dict[str, Any] = {"field": exc.field}
    for descriptor in descriptors:
        if descriptor.name == exc.field and descriptor.param_type == "select":
            allowed = [str(option.get("value")) for option in descriptor.options or []]
            details["allowed"] = allowed[:MAX_LISTED_CHOICES]
    make_error(
        "validation_error",
        exc.message,
        ["Correct the named field; capabilities lists each entry field and its values."],
        details=details,
    )


async def _check_entry_fields(
    fields: dict[str, Any] | None, *, partial: bool
) -> tuple[list["ParameterDescriptor"], dict[str, Any]]:
    """Look up the declared entry fields and check ``fields`` against them.

    Nothing is checked live: a ``dynamic_select`` value is checked for its
    type only. An undeclared key is refused.

    Args:
        fields: The submitted values keyed by field name.
        partial: Accept a missing required value (draft mode).

    Returns:
        The declarations and the coerced declared values.
    """
    try:
        config = get_ariel_context().config
    except RuntimeError:
        descriptors: list[ParameterDescriptor] = []
    else:
        try:
            descriptors = entry_field_descriptors(config)
        except EntryFieldDeclarationError as exc:
            return make_error(
                "internal_error",
                exc.message,
                ["The facility adapter declares its entry fields wrongly; check its definition."],
                details={"field": exc.field},
            )
    try:
        declared = await validate_entry_fields(
            None, descriptors, fields or {}, partial=partial, check_live=False, strict=True
        )
    except EntryFieldError as exc:
        return _entry_field_refusal(exc, descriptors)
    return descriptors, declared


@mcp.tool()
async def entry_create(
    subject: str,
    details: str,
    author: str | None = None,
    logbook: str | None = None,
    shift: str | None = None,
    tags: list[str] | None = None,
    file_paths: list[str] | None = None,
    artifact_ids: list[str] | None = None,
    draft: bool = True,
    fields: dict[str, Any] | None = None,
) -> str:
    """Create a new logbook entry, optionally with file attachments.

    By default (draft=True), this creates a draft that pre-fills the ARIEL
    web form so a human can review, edit, and submit. Set draft=False to
    write directly to the database without human review.

    Args:
        subject: Entry subject/title (required).
        details: Entry body/details (required).
        author: Author name (default: "Anonymous").
        logbook: Logbook name to file under.
        shift: Shift or run-block identifier as your facility names it (free-form).
        tags: List of tags for the entry.
        file_paths: Local file paths to attach (max 10 MB each).
        artifact_ids: Artifact IDs from the gallery to attach. HTML artifacts
            (e.g. Plotly plots) are auto-converted to PNG.
        draft: If True (default), create a draft for human review in the web UI.
            If False, write directly to the database.
        fields: Values for the facility's entry fields, keyed by the names
            ``capabilities`` lists under ``entry_fields``. An undeclared name or
            a wrong value is refused naming the field. A draft may leave a
            required field empty; a direct write may not.

    Returns:
        JSON with draft_id and URL (draft mode), or entry_id and confirmation (direct mode).
    """
    if not subject or not subject.strip():
        return make_error(
            "validation_error",
            "subject is required.",
            ["Provide a subject/title for the entry."],
        )
    if not details or not details.strip():
        return make_error(
            "validation_error",
            "details is required.",
            ["Provide details/body for the entry."],
        )

    # --- Check declared entry fields before anything is written ---
    descriptors: list[ParameterDescriptor] = []
    declared: dict[str, Any] = {}
    if fields or not draft:
        descriptors, declared = await _check_entry_fields(fields, partial=draft)

    # --- Resolve artifact_ids to file paths (both modes) ---
    artifact_paths: list[str] = []
    if artifact_ids:
        try:
            artifact_paths = await _resolve_artifacts(artifact_ids)
        except ValueError as exc:
            return make_error(
                "validation_error",
                str(exc),
                ["Check artifact IDs via the gallery or artifact_register output."],
            )
        except ToolError:
            raise
        except Exception as exc:
            logger.warning("Artifact resolution failed (non-fatal for draft): %s", exc)
            # For draft mode, store IDs for deferred resolution
            if not draft:
                return make_error(
                    "internal_error",
                    f"Failed to resolve artifacts: {exc}",
                    ["Check MCP server logs for details."],
                )

    # Merge artifact-resolved paths with explicit file_paths (used by both modes)
    all_file_paths = list(file_paths or []) + artifact_paths

    # Resolve relative paths to absolute so downstream consumers always get
    # a usable path regardless of their working directory.
    all_file_paths = [str(Path(p).resolve()) for p in all_file_paths]

    if all_file_paths:
        from osprey.services.ariel_search.attachments import (
            AttachmentValidationError,
            read_local_file,
        )

        try:
            for path in all_file_paths:
                read_local_file(path)  # validates existence + size
        except AttachmentValidationError as exc:
            return make_error(
                "validation_error",
                str(exc),
                ["Check file path and ensure file is under 10 MB."],
            )

    # --- Draft mode: write a JSON file for the web UI to pick up ---
    if draft:
        try:
            draft_id = f"draft-{uuid.uuid4().hex[:12]}"

            drafts_dir = _get_drafts_dir()
            drafts_dir.mkdir(parents=True, exist_ok=True)
            from osprey.mcp_server.session import gather_session_metadata

            draft_data = {
                "draft_id": draft_id,
                "subject": subject.strip(),
                "details": details.strip(),
                "author": author,
                "logbook": logbook,
                "shift": shift,
                "tags": tags or [],
                "metadata": {
                    "session_metadata": gather_session_metadata("ariel-mcp"),
                },
            }
            if fields:
                draft_data["fields"] = declared
            if all_file_paths:
                draft_data["attachment_paths"] = all_file_paths
            filepath = drafts_dir / f"{draft_id}.json"
            filepath.write_text(json.dumps(draft_data, indent=2))

            url = ariel_panel_url(f"create?draft={draft_id}")

            logger.info("Draft %s created at %s", draft_id, filepath)

            _focus_ariel_panel(url)

            return json.dumps(
                {
                    "draft_id": draft_id,
                    "url": url,
                    "message": (
                        f"Draft {draft_id} created. Open the URL to review and submit: {url}"
                    ),
                }
            )
        except ToolError:
            raise
        except Exception as exc:
            logger.exception("entry_create (draft) failed")
            return make_error(
                "internal_error",
                f"Failed to create draft: {exc}",
                ["Check that _agent_data/drafts/ is writable."],
            )

    # --- Direct mode: write straight to the database ---

    try:
        from osprey.mcp_server.session import gather_session_metadata

        registry = get_ariel_context()
        service = await registry.service()

        resolved = resolve_entry_write(
            descriptors,
            declared,
            logbook=logbook,
            shift=shift,
            tags=tags or [],
            created_via="ariel-mcp",
            session_metadata=gather_session_metadata("ariel-mcp"),
        )

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        entry: EnhancedLogbookEntry = {
            "entry_id": entry_id,
            "source_system": ARIEL_NATIVE_SOURCE_SYSTEM,
            "timestamp": now,
            "author": author or "Anonymous",
            "raw_text": f"{subject}\n\n{details}",
            "attachments": [],
            "metadata": resolved.local_metadata,
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

        # Agent-activity highlight for the ARIEL panel, emitted the moment the
        # entry is persisted and before attachments are processed — an
        # attachment failure must not lose the signal for an entry that already
        # exists. Passive on purpose: unlike the draft branch above, a direct
        # write does not steal focus. notify_agent_activity_async never raises; the
        # blocking call runs off the event loop.
        await notify_agent_activity_async("entry_create", "panel", panel="ariel", detail=entry_id)

        # Process attachments if provided
        attachment_count = 0
        if all_file_paths:
            from osprey.services.ariel_search.attachments import (
                process_attachments_for_entry,
            )

            attachment_infos = await process_attachments_for_entry(
                entry_id=entry_id,
                file_paths=all_file_paths,
                repository=service.repository,
            )
            # Update entry's attachments JSONB
            entry["attachments"] = attachment_infos
            await service.repository.upsert_entry(entry)
            attachment_count = len(attachment_infos)

        return json.dumps(
            {
                "entry_id": entry_id,
                "message": f"Entry {entry_id} created successfully",
                "source_system": ARIEL_NATIVE_SOURCE_SYSTEM,
                "attachment_count": attachment_count,
            },
            default=str,
        )

    except EntryFieldError as exc:
        return _entry_field_refusal(exc, descriptors)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("entry_create failed")
        return make_error(
            "internal_error",
            f"Failed to create entry: {exc}",
            ["Check ARIEL database connectivity."],
        )
