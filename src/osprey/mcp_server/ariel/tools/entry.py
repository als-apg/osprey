"""MCP tools: entry_get + entry_create — logbook entry CRUD.

PROMPT-PROVIDER: Tool docstrings are static prompts visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_logbook_search_prompt_builder()
  Facility-customizable: shift identifiers (e.g., "Day"/"Swing"/"Owl"),
  attachment limits, logbook name conventions
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from osprey.mcp_server.ariel.server import make_error, mcp, serialize_entry
from osprey.mcp_server.ariel.server_context import get_ariel_context

logger = logging.getLogger("osprey.mcp_server.ariel.tools.entry")


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

    from osprey.mcp_server.ariel.converters import get_converter
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    resolved: list[str] = []
    output_dir = Path(tempfile.mkdtemp(prefix="ariel_convert_"))

    for aid in artifact_ids:
        entry = store.get_entry(aid)
        if entry is None:
            raise ValueError(f"Artifact '{aid}' not found.")

        file_path = store.get_file_path(aid)
        if file_path is None or not file_path.exists():
            raise ValueError(f"Artifact file not found on disk for '{aid}'.")

        converter = get_converter(entry.mime_type)
        result_path = await converter(file_path, output_dir)
        resolved.append(str(result_path))

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
        return json.dumps(
            make_error(
                "validation_error",
                "entry_id is required.",
                ["Provide a valid entry ID."],
            )
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        entry = await service.repository.get_entry(entry_id)
        if not entry:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Entry {entry_id} not found.",
                    [
                        "Check the entry_id is correct.",
                        "Use keyword_search/semantic_search or browse to find valid entry IDs.",
                    ],
                )
            )

        # TypedDict -- dict access, not attribute access
        return json.dumps(
            {
                "entry_id": entry["entry_id"],
                "source_system": entry["source_system"],
                "timestamp": entry["timestamp"],
                "author": entry.get("author", ""),
                "raw_text": entry["raw_text"],
                "attachments": entry.get("attachments", []),
                "metadata": entry.get("metadata", {}),
                "summary": entry.get("summary"),
                "keywords": entry.get("keywords", []),
                "created_at": entry["created_at"],
                "updated_at": entry["updated_at"],
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("entry_get failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get entry: {exc}",
                ["Check ARIEL database connectivity."],
            )
        )


@mcp.tool()
async def entries_by_ids(
    entry_ids: list[str],
) -> str:
    """Get multiple logbook entries by their IDs in a single call.

    Efficient batch retrieval for reading full details of entries found
    via search. Returns full entry content with a higher text limit than
    search results.

    Args:
        entry_ids: List of entry IDs to retrieve (max 50 per call).

    Returns:
        JSON with list of found entries. May return fewer than requested
        if some IDs don't exist.
    """
    if not entry_ids:
        return json.dumps(
            make_error(
                "validation_error",
                "entry_ids list is empty.",
                ["Provide at least one entry ID."],
            )
        )

    if len(entry_ids) > 50:
        return json.dumps(
            make_error(
                "validation_error",
                f"Too many entry IDs ({len(entry_ids)}). Maximum is 50 per call.",
                ["Split into multiple calls of 50 or fewer IDs."],
            )
        )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        entries = await service.repository.get_entries_by_ids(entry_ids)

        # Serialize with longer text limit for full details
        entries_out = [serialize_entry(e, text_limit=1000) for e in entries]

        return json.dumps(
            {
                "requested": len(entry_ids),
                "found": len(entries_out),
                "entries": entries_out,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("entries_by_ids failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get entries: {exc}",
                ["Check ARIEL database connectivity."],
            )
        )


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
        shift: Shift identifier (e.g. "Day", "Swing", "Owl").
        tags: List of tags for the entry.
        file_paths: Local file paths to attach (max 10 MB each).
        artifact_ids: Artifact IDs from the gallery to attach. HTML artifacts
            (e.g. Plotly plots) are auto-converted to PNG.
        draft: If True (default), create a draft for human review in the web UI.
            If False, write directly to the database.

    Returns:
        JSON with draft_id and URL (draft mode), or entry_id and confirmation (direct mode).
    """
    if not subject or not subject.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "subject is required.",
                ["Provide a subject/title for the entry."],
            )
        )
    if not details or not details.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "details is required.",
                ["Provide details/body for the entry."],
            )
        )

    # --- Resolve artifact_ids to file paths (both modes) ---
    artifact_paths: list[str] = []
    if artifact_ids:
        try:
            artifact_paths = await _resolve_artifacts(artifact_ids)
        except ValueError as exc:
            return json.dumps(
                make_error(
                    "validation_error",
                    str(exc),
                    ["Check artifact IDs via the gallery or artifact_save output."],
                )
            )
        except Exception as exc:
            logger.warning("Artifact resolution failed (non-fatal for draft): %s", exc)
            # For draft mode, store IDs for deferred resolution
            if not draft:
                return json.dumps(
                    make_error(
                        "internal_error",
                        f"Failed to resolve artifacts: {exc}",
                        ["Check MCP server logs for details."],
                    )
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
            return json.dumps(
                make_error(
                    "validation_error",
                    str(exc),
                    ["Check file path and ensure file is under 10 MB."],
                )
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
            if all_file_paths:
                draft_data["attachment_paths"] = all_file_paths
            filepath = drafts_dir / f"{draft_id}.json"
            filepath.write_text(json.dumps(draft_data, indent=2))

            base_url = os.environ.get("ARIEL_WEB_URL", "http://127.0.0.1:8085")
            url = f"{base_url}/#create?draft={draft_id}"

            logger.info("Draft %s created at %s", draft_id, filepath)

            try:
                from osprey.mcp_server.http import notify_panel_focus

                notify_panel_focus("ariel", url=url)
            except Exception:
                pass  # Non-fatal — web terminal may not be running

            return json.dumps(
                {
                    "draft_id": draft_id,
                    "url": url,
                    "message": (
                        f"Draft {draft_id} created. Open the URL to review and submit: {url}"
                    ),
                }
            )
        except Exception as exc:
            logger.exception("entry_create (draft) failed")
            return json.dumps(
                make_error(
                    "internal_error",
                    f"Failed to create draft: {exc}",
                    ["Check that _agent_data/drafts/ is writable."],
                )
            )

    # --- Direct mode: write straight to the database ---

    try:
        from osprey.mcp_server.session import gather_session_metadata

        registry = get_ariel_context()
        service = await registry.service()

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        entry = {
            "entry_id": entry_id,
            "source_system": "ARIEL MCP",
            "timestamp": now,
            "author": author or "Anonymous",
            "raw_text": f"{subject}\n\n{details}",
            "attachments": [],
            "metadata": {
                "logbook": logbook,
                "shift": shift,
                "tags": tags or [],
                "created_via": "ariel-mcp",
                "session_metadata": gather_session_metadata("ariel-mcp"),
            },
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

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
                "source_system": "ARIEL MCP",
                "attachment_count": attachment_count,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("entry_create failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to create entry: {exc}",
                ["Check ARIEL database connectivity."],
            )
        )
