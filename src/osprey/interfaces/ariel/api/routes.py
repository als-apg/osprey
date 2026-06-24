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
from fastapi.responses import Response
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

if TYPE_CHECKING:
    from osprey.services.ariel_search import ARIELSearchService

router = APIRouter(prefix="/api")


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
        total = await service.repository.count_entries()

        # Operator-supplied query params are parsed naive by FastAPI; interpret
        # them as facility-local before they hit the TIMESTAMPTZ column.
        # TODO: add offset pagination when repository supports it
        entries = await service.repository.search_by_time_range(
            start=_localize_facility(start_date),
            end=_localize_facility(end_date),
            limit=page_size,
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


@router.post("/entries", response_model=EntryCreateResponse)
async def create_entry(
    request: Request,
    entry_req: EntryCreateRequest,
) -> EntryCreateResponse:
    """Create a new logbook entry.

    Delegates to the facility adapter when write support is available.
    Falls back to direct database insert if the adapter doesn't support writes.
    """
    service = _require_service(request)

    try:
        from osprey.services.ariel_search.exceptions import IngestionError
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

        result = await service.create_entry(facility_request)

        return EntryCreateResponse(
            entry_id=result.entry_id,
            message=result.message,
            sync_status=result.sync_status.value,
            source_system=result.source_system,
        )

    except (NotImplementedError, IngestionError):
        # Adapter doesn't support writes — fall back to direct DB insert
        import logging

        logging.getLogger("ariel").warning(
            "Facility adapter does not support writes, falling back to direct DB insert"
        )

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        entry = {
            "entry_id": entry_id,
            "source_system": "ARIEL Web",
            "timestamp": now,
            "author": entry_req.author or "Anonymous",
            "raw_text": f"{entry_req.subject}\n\n{entry_req.details}",
            "attachments": [],
            "metadata": {
                "logbook": entry_req.logbook,
                "shift": entry_req.shift,
                "tags": entry_req.tags,
                "created_via": "ariel-web",
                **(entry_req.metadata or {}),
            },
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

        return EntryCreateResponse(
            entry_id=entry_id,
            message=f"Entry {entry_id} created (saved locally, not published to external logbook)",
            sync_status="local_only",
            source_system="ARIEL Web",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
    files: list[UploadFile] = File(default=[]),
) -> EntryCreateResponse:
    """Create a new logbook entry with file attachments via multipart form."""
    service = _require_service(request)

    from osprey.services.ariel_search.attachments import (
        AttachmentValidationError,
        generate_attachment_id,
        guess_mime_type,
        validate_file_size,
    )

    try:
        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        entry: dict[str, Any] = {
            "entry_id": entry_id,
            "source_system": "ARIEL Web",
            "timestamp": now,
            "author": author or "Anonymous",
            "raw_text": f"{subject}\n\n{details}",
            "attachments": [],
            "metadata": {
                "logbook": logbook,
                "shift": shift,
                "tags": tag_list,
                "created_via": "ariel-web",
                **_parse_metadata_form(metadata),
            },
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

        attachment_count = 0
        if files:
            attachment_infos = []
            for upload_file in files:
                if not upload_file.filename:
                    continue

                data = await upload_file.read()
                try:
                    validate_file_size(len(data), upload_file.filename)
                except AttachmentValidationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                attachment_id = generate_attachment_id()
                mime_type = upload_file.content_type or guess_mime_type(upload_file.filename)

                await service.repository.store_attachment(
                    entry_id=entry_id,
                    attachment_id=attachment_id,
                    filename=upload_file.filename,
                    mime_type=mime_type,
                    data=data,
                    size_bytes=len(data),
                )

                attachment_infos.append(
                    {
                        "url": f"/api/attachments/{attachment_id}",
                        "type": mime_type,
                        "filename": upload_file.filename,
                    }
                )

            if attachment_infos:
                entry["attachments"] = attachment_infos
                await service.repository.upsert_entry(entry)
                attachment_count = len(attachment_infos)

        return EntryCreateResponse(
            entry_id=entry_id,
            message=f"Entry {entry_id} created successfully",
            attachment_count=attachment_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
