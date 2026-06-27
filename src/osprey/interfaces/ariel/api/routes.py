"""ARIEL Web API routes.

REST endpoints for search, entry management, status, and settings.
"""

from __future__ import annotations

import json as _json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from osprey.interfaces.ariel.api.schemas import (
    DiagnosticResponse,
    EntriesListResponse,
    EntryCreateRequest,
    EntryCreateResponse,
    EntryResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from osprey.utils.config import to_facility_iso
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search import ARIELSearchService

router = APIRouter(prefix="/api")
logger = get_logger("ariel")


def _parse_metadata_form(raw: str | None) -> dict[str, Any]:
    """Parse a JSON metadata string from a form field, returning {} on failure."""
    if not raw:
        return {}
    try:
        parsed = _json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _localize_facility(dt: datetime | None) -> datetime | None:
    """Attach the facility timezone to a naive operator-provided datetime.

    Mirrors the MCP ``parse_date_filters`` contract so the web UI interprets
    operator-supplied dates as facility-local (not box-local / UTC) before they
    drive a ``TIMESTAMPTZ`` query. Aware datetimes pass through unchanged.
    """
    if dt is not None and dt.tzinfo is None:
        from osprey.utils.config import get_facility_timezone

        return dt.replace(tzinfo=get_facility_timezone())
    return dt


def _require_service(request: Request) -> ARIELSearchService:
    """Get the ARIEL service or raise 503 if the database is unavailable."""
    service = getattr(request.app.state, "ariel_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — search, browse, and entry creation "
            "require a database connection. Drafts and settings still work.",
        )
    return service


def _entry_to_response(
    entry: dict,
    score: float | None = None,
    highlights: list[str] | None = None,
) -> EntryResponse:
    """Convert database entry to response model."""
    from osprey.services.ariel_search.attachments import guess_mime_type

    # Enrich attachments that have no MIME type but have a filename
    attachments = entry.get("attachments", [])
    for att in attachments:
        if not att.get("type") and att.get("filename"):
            att["type"] = guess_mime_type(att["filename"])

    # Render the three timestamp fields facility-local (ISO with offset) via the
    # shared egress helper, so the web wire format matches the MCP path
    # (serialize_entry) instead of leaking the DB's raw UTC. The schema fields are
    # ``str`` so Pydantic passes these pre-formatted strings through unchanged.
    return EntryResponse(
        entry_id=entry["entry_id"],
        source_system=entry["source_system"],
        timestamp=to_facility_iso(entry["timestamp"]),
        author=entry.get("author", ""),
        raw_text=entry["raw_text"],
        attachments=attachments,
        metadata=entry.get("metadata", {}),
        created_at=to_facility_iso(entry["created_at"]),
        updated_at=to_facility_iso(entry["updated_at"]),
        summary=entry.get("summary"),
        keywords=entry.get("keywords", []),
        score=score,
        highlights=highlights or [],
    )


@router.get("/capabilities")
async def get_capabilities(request: Request) -> dict:
    """Return available search modes and their tunable parameters.

    The frontend calls this at startup to dynamically render
    mode tabs and advanced options.
    """
    from osprey.services.ariel_search.capabilities import get_capabilities as _get_caps

    service = _require_service(request)
    return _get_caps(service.config)


@router.get("/publish-info")
async def get_publish_info(request: Request) -> dict:
    """Describe the configured logbook's write capability for the create form.

    Lets the UI adapt its credential prompt to the actual adapter instead of
    showing fixed text: a logbook that requires authentication asks for
    credentials, a no-auth logbook publishes without them, and a read-only
    adapter saves to ARIEL only. ``requires_auth`` is reported as
    ``supports_write and requires_write_auth`` — a read-only adapter cannot
    publish, so credentials are irrelevant there.
    """
    service = _require_service(request)

    from osprey.services.ariel_search.exceptions import AdapterNotFoundError
    from osprey.services.ariel_search.ingestion import get_adapter

    try:
        adapter = get_adapter(service.config)
    except AdapterNotFoundError:
        # No ingestion adapter configured — entries can only be saved locally.
        return {"supports_write": False, "requires_auth": False, "source_system": None}

    return {
        "supports_write": adapter.supports_write,
        "requires_auth": adapter.supports_write and adapter.requires_write_auth,
        "source_system": adapter.source_system_name,
    }


@router.get("/filter-options/{field_name}")
async def get_filter_options(request: Request, field_name: str) -> dict:
    """Return distinct values for a filterable field.

    Used by dynamic_select parameters to populate dropdown options.
    """
    service = _require_service(request)

    field_methods = {
        "authors": "get_distinct_authors",
        "source_systems": "get_distinct_source_systems",
    }

    method_name = field_methods.get(field_name)
    if not method_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown filter field: {field_name}. Available: {', '.join(field_methods)}",
        )

    try:
        method = getattr(service.repository, method_name)
        values = await method()
        return {
            "field": field_name,
            "options": [{"value": v, "label": v} for v in values],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, search_req: SearchRequest) -> SearchResponse:
    """Execute search query.

    Supports keyword and semantic modes.
    """
    service = _require_service(request)
    start_time = time.time()

    try:
        # Map API mode to service mode
        from osprey.services.ariel_search.models import SearchMode as ServiceSearchMode

        mode_map = {
            SearchMode.KEYWORD: ServiceSearchMode.KEYWORD,
            SearchMode.SEMANTIC: ServiceSearchMode.SEMANTIC,
        }
        service_mode = mode_map.get(search_req.mode)

        # advanced_params takes precedence over top-level filter fields
        adv = search_req.advanced_params
        start_date = adv.pop("start_date", None) or search_req.start_date
        end_date = adv.pop("end_date", None) or search_req.end_date
        author = adv.pop("author", None) or search_req.author
        source_system = adv.pop("source_system", None) or search_req.source_system

        if isinstance(start_date, str) and start_date:
            start_date = _localize_facility(datetime.fromisoformat(start_date))
        if isinstance(end_date, str) and end_date:
            end_date = _localize_facility(datetime.fromisoformat(end_date))

        time_range = None
        if start_date or end_date:
            time_range = (start_date, end_date)

        # Re-inject non-date filters into advanced_params for downstream use
        if author:
            adv["author"] = author
        if source_system:
            adv["source_system"] = source_system

        result = await service.search(
            query=search_req.query,
            max_results=search_req.max_results,
            time_range=time_range,
            mode=service_mode,
            advanced_params=adv,
        )

        execution_time = int((time.time() - start_time) * 1000)

        entries = [
            _entry_to_response(e, score=e.get("_score"), highlights=e.get("_highlights"))
            for e in result.entries
        ]

        return SearchResponse(
            entries=entries,
            answer=result.answer,
            sources=list(result.sources),
            search_modes_used=[m.value for m in result.search_modes_used],
            reasoning=result.reasoning,
            total_results=len(entries),
            execution_time_ms=execution_time,
            diagnostics=[
                DiagnosticResponse(
                    level=d.level.value,
                    source=d.source,
                    message=d.message,
                    category=d.category,
                )
                for d in result.diagnostics
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entries", response_model=EntriesListResponse)
async def list_entries(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    sort_order: str = "desc",
) -> EntriesListResponse:
    """List entries with pagination and filtering."""
    service = _require_service(request)

    try:
        # Operator-supplied query params are parsed naive by FastAPI; interpret
        # them as facility-local before they hit the TIMESTAMPTZ column.
        local_start = _localize_facility(start_date)
        local_end = _localize_facility(end_date)

        # Count with the same filters so total_pages reflects the filtered set,
        # not the whole table.
        total = await service.repository.count_entries(
            start=local_start,
            end=local_end,
            author=author,
            source_system=source_system,
        )

        offset = max(page - 1, 0) * page_size
        entries = await service.repository.search_by_time_range(
            start=local_start,
            end=local_end,
            limit=page_size,
            offset=offset,
            author=author,
            source_system=source_system,
        )

        entry_responses = [_entry_to_response(e) for e in entries]

        total_pages = (total + page_size - 1) // page_size

        return EntriesListResponse(
            entries=entry_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entries/{entry_id}", response_model=EntryResponse)
async def get_entry(request: Request, entry_id: str) -> EntryResponse:
    """Get a single entry by ID."""
    service = _require_service(request)

    try:
        entry = await service.repository.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")

        return _entry_to_response(entry)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _auth_required_response(exc: Exception) -> JSONResponse:
    """Build a 401 that asks the operator for logbook credentials.

    The ``code`` discriminator lets the frontend show a credential prompt
    instead of a generic error, and keep the form populated for resubmission.
    """
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc), "code": "auth_required"},
    )


async def _publish_or_local(
    service: ARIELSearchService,
    facility_request: Any,
    *,
    fallback_metadata: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Publish an entry via the facility adapter, or save local-only if read-only.

    Returns ``(entry_id, source_system, sync_status, message)`` for the text
    entry. Only ``NotImplementedError`` (the adapter genuinely cannot write)
    falls back to a direct local DB insert — this is the legitimate use of the
    fallback. ``AuthenticationRequiredError`` (credentials needed) and
    ``IngestionError`` (the publish attempt failed) propagate to the caller so
    nothing is silently saved.

    Args:
        service: The ARIEL search service.
        facility_request: A ``FacilityEntryCreateRequest`` for the text entry.
        fallback_metadata: Fields for the local-only insert (``author``,
            ``raw_text``, ``metadata``) used when the adapter is read-only.
    """
    try:
        result = await service.create_entry(facility_request)
        return (
            result.entry_id,
            result.source_system,
            result.sync_status.value,
            result.message,
        )
    except NotImplementedError:
        logger.warning("Facility adapter does not support writes, falling back to direct DB insert")

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        entry = {
            "entry_id": entry_id,
            "source_system": "ARIEL Web",
            "timestamp": now,
            "author": fallback_metadata.get("author") or "Anonymous",
            "raw_text": fallback_metadata["raw_text"],
            "attachments": [],
            "metadata": fallback_metadata["metadata"],
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

        return (
            entry_id,
            "ARIEL Web",
            "local_only",
            f"Entry {entry_id} created (saved locally, not published to external logbook)",
        )


@router.post("/entries", response_model=EntryCreateResponse)
async def create_entry(
    request: Request,
    entry_req: EntryCreateRequest,
) -> EntryCreateResponse | JSONResponse:
    """Create a new logbook entry.

    Delegates to the facility adapter when write support is available. A
    read-only adapter falls back to a local-only save; a logbook that requires
    credentials returns 401 (so the UI can prompt) and a genuine publish failure
    returns 502 — neither silently saves local-only.
    """
    service = _require_service(request)

    from osprey.services.ariel_search.exceptions import (
        AuthenticationRequiredError,
        IngestionError,
    )
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

    facility_request = FacilityEntryCreateRequest(
        subject=entry_req.subject,
        details=entry_req.details,
        author=entry_req.author,
        logbook=entry_req.logbook,
        shift=entry_req.shift,
        tags=entry_req.tags,
        auth_user=entry_req.auth_user,
        auth_password=entry_req.auth_password,
    )

    try:
        entry_id, source_system, sync_status, message = await _publish_or_local(
            service,
            facility_request,
            fallback_metadata={
                "author": entry_req.author,
                "raw_text": f"{entry_req.subject}\n\n{entry_req.details}",
                "metadata": {
                    "logbook": entry_req.logbook,
                    "shift": entry_req.shift,
                    "tags": entry_req.tags,
                    "created_via": "ariel-web",
                    **(entry_req.metadata or {}),
                },
            },
        )
    except AuthenticationRequiredError as e:
        return _auth_required_response(e)
    except IngestionError as e:
        raise HTTPException(status_code=502, detail=f"Publish failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return EntryCreateResponse(
        entry_id=entry_id,
        message=message,
        sync_status=sync_status,
        source_system=source_system,
    )


@router.get("/attachments/{attachment_id}")
async def get_attachment(request: Request, attachment_id: str) -> Response:
    """Serve an attachment file by its ID.

    Returns the raw binary data with the correct Content-Type header.
    """
    service = _require_service(request)

    try:
        attachment = await service.repository.get_attachment(attachment_id)
        if not attachment:
            raise HTTPException(status_code=404, detail=f"Attachment {attachment_id} not found")

        return Response(
            content=attachment["data"],
            media_type=attachment.get("mime_type") or "application/octet-stream",
            headers={
                "Content-Disposition": f'inline; filename="{attachment.get("filename", "file")}"',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _store_and_link_attachments(
    service: ARIELSearchService,
    entry_id: str,
    staged: list[tuple[str, str | None, bytes]],
) -> int:
    """Persist staged ``(filename, mime_type, data)`` files and link them on the entry.

    Files are stored in ARIEL's own attachment store and referenced on the entry
    record. Note: the OLOG write API cannot accept file uploads, so attachments
    are never pushed to an external logbook — they live in ARIEL only.

    Returns:
        The number of attachments stored.
    """
    from osprey.services.ariel_search.attachments import generate_attachment_id

    attachment_infos: list[dict[str, Any]] = []
    for filename, mime_type, data in staged:
        attachment_id = generate_attachment_id()
        await service.repository.store_attachment(
            entry_id=entry_id,
            attachment_id=attachment_id,
            filename=filename,
            mime_type=mime_type,
            data=data,
            size_bytes=len(data),
        )
        attachment_infos.append(
            {
                "url": f"/api/attachments/{attachment_id}",
                "type": mime_type,
                "filename": filename,
            }
        )

    if attachment_infos:
        entry = await service.repository.get_entry(entry_id)
        if entry is not None:
            entry["attachments"] = attachment_infos
            await service.repository.upsert_entry(entry)

    return len(attachment_infos)


@router.post("/entries/upload", response_model=EntryCreateResponse)
async def create_entry_with_attachments(
    request: Request,
    subject: str = Form(...),
    details: str = Form(...),
    author: str | None = Form(None),
    logbook: str | None = Form(None),
    shift: str | None = Form(None),
    tags: str = Form(""),
    metadata: str | None = Form(None),
    auth_user: str | None = Form(None),
    auth_password: str | None = Form(None),
    files: list[UploadFile] = File(default=[]),
) -> EntryCreateResponse | JSONResponse:
    """Create a new logbook entry with file attachments via multipart form.

    The text body is published through the facility adapter with the same
    semantics as ``POST /entries`` — a logbook that requires credentials returns
    401, a genuine publish failure returns 502, and a read-only adapter falls
    back to a local-only save. Attachments are then stored in ARIEL. Because the
    OLOG write API cannot accept file uploads, attachments are never published to
    an external logbook; when the text body does publish externally, the response
    says so explicitly rather than silently dropping the files.
    """
    service = _require_service(request)

    from osprey.services.ariel_search.attachments import (
        AttachmentValidationError,
        guess_mime_type,
        validate_file_size,
    )
    from osprey.services.ariel_search.exceptions import (
        AuthenticationRequiredError,
        IngestionError,
    )
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    parsed_metadata = _parse_metadata_form(metadata)

    # Read and validate files up front — before any publish — so a rejected file
    # can never leave a half-published entry behind.
    staged: list[tuple[str, str | None, bytes]] = []
    for upload_file in files:
        if not upload_file.filename:
            continue
        data = await upload_file.read()
        try:
            validate_file_size(len(data), upload_file.filename)
        except AttachmentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        mime_type = upload_file.content_type or guess_mime_type(upload_file.filename)
        staged.append((upload_file.filename, mime_type, data))

    facility_request = FacilityEntryCreateRequest(
        subject=subject,
        details=details,
        author=author,
        logbook=logbook,
        shift=shift,
        tags=tag_list,
        auth_user=auth_user,
        auth_password=auth_password,
        metadata=parsed_metadata,
    )

    # Publish the text body (or save local-only for a read-only adapter).
    try:
        entry_id, source_system, sync_status, message = await _publish_or_local(
            service,
            facility_request,
            fallback_metadata={
                "author": author,
                "raw_text": f"{subject}\n\n{details}",
                "metadata": {
                    "logbook": logbook,
                    "shift": shift,
                    "tags": tag_list,
                    "created_via": "ariel-web",
                    **parsed_metadata,
                },
            },
        )
    except AuthenticationRequiredError as e:
        return _auth_required_response(e)
    except IngestionError as e:
        raise HTTPException(status_code=502, detail=f"Publish failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Store attachments in ARIEL and link them on the entry.
    attachment_count = await _store_and_link_attachments(service, entry_id, staged)

    # If the text published externally, be explicit that the files did not —
    # the OLOG write API can't accept file uploads.
    if attachment_count and sync_status != "local_only":
        message = (
            f"{message}. {attachment_count} attachment(s) saved to ARIEL only "
            "(the logbook API cannot accept file uploads yet)."
        )

    return EntryCreateResponse(
        entry_id=entry_id,
        message=message,
        sync_status=sync_status,
        source_system=source_system,
        attachment_count=attachment_count,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request) -> StatusResponse:
    """Get service status and health information."""
    service = _require_service(request)

    try:
        status = await service.get_status()

        return StatusResponse(
            healthy=status.healthy,
            database_connected=status.database_connected,
            database_uri=status.database_uri,
            entry_count=status.entry_count,
            embedding_tables=[
                {
                    "table_name": t.table_name,
                    "entry_count": t.entry_count,
                    "dimension": t.dimension,
                    "is_active": t.is_active,
                }
                for t in status.embedding_tables
            ],
            active_embedding_model=status.active_embedding_model,
            enabled_search_modules=status.enabled_search_modules,
            enabled_enhancement_modules=status.enabled_enhancement_modules,
            last_ingestion=status.last_ingestion,
            errors=status.errors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Settings endpoints — config.yml and Claude Code setup files
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Find the OSPREY project root (directory containing config.yml)."""
    candidates = [
        Path(os.environ.get("CONFIG_FILE", "")).parent if os.environ.get("CONFIG_FILE") else None,
        Path("/app"),
        Path.cwd(),
    ]
    for p in candidates:
        if p and (p / "config.yml").exists():
            return p
    return Path.cwd()


def _config_path() -> Path:
    return _find_project_root() / "config.yml"


@router.get("/config")
async def get_config() -> dict:
    """Return the current config.yml as a dict and raw YAML."""
    path = _config_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="config.yml not found")
    raw = path.read_text()
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        parsed = {}
    return {"raw": raw, "parsed": parsed}


class ConfigUpdateRequest(BaseModel):
    content: str


@router.put("/config")
async def update_config(req: ConfigUpdateRequest) -> dict:
    """Write new content to config.yml with backup + fsync."""
    path = _config_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="config.yml not found")

    try:
        yaml.safe_load(req.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}") from e

    bak = path.with_suffix(".yml.bak")
    bak.write_text(path.read_text())

    # fsync for crash safety
    fd = os.open(str(path), os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
    try:
        os.write(fd, req.content.encode())
        os.fsync(fd)
    finally:
        os.close(fd)

    return {"status": "ok", "requires_restart": True}
