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
    AgentStepResponse,
    AgentToolInvocationResponse,
    DiagnosticResponse,
    EntriesListResponse,
    EntryCreateRequest,
    EntryCreateResponse,
    EntryResponse,
    PipelineDetailsResponse,
    RAGStageStatsResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)

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

    return EntryResponse(
        entry_id=entry["entry_id"],
        source_system=entry["source_system"],
        timestamp=entry["timestamp"],
        author=entry.get("author", ""),
        raw_text=entry["raw_text"],
        attachments=attachments,
        metadata=entry.get("metadata", {}),
        created_at=entry["created_at"],
        updated_at=entry["updated_at"],
        summary=entry.get("summary"),
        keywords=entry.get("keywords", []),
        score=score,
        highlights=highlights or [],
    )


def _pipeline_details_to_response(pd: Any) -> PipelineDetailsResponse | None:
    """Convert a PipelineDetails dataclass to its API response model."""
    if pd is None:
        return None

    rag_stats = None
    if pd.rag_stats is not None:
        rag_stats = RAGStageStatsResponse(
            keyword_retrieved=pd.rag_stats.keyword_retrieved,
            semantic_retrieved=pd.rag_stats.semantic_retrieved,
            fused_count=pd.rag_stats.fused_count,
            context_included=pd.rag_stats.context_included,
            context_truncated=pd.rag_stats.context_truncated,
        )

    return PipelineDetailsResponse(
        pipeline_type=pd.pipeline_type,
        rag_stats=rag_stats,
        agent_tool_invocations=[
            AgentToolInvocationResponse(
                tool_name=inv.tool_name,
                tool_args=inv.tool_args,
                result_summary=inv.result_summary,
                order=inv.order,
            )
            for inv in pd.agent_tool_invocations
        ],
        agent_steps=[
            AgentStepResponse(
                step_type=s.step_type,
                content=s.content,
                tool_name=s.tool_name,
                order=s.order,
            )
            for s in pd.agent_steps
        ],
        step_summary=pd.step_summary,
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

    Supports keyword, semantic, RAG, and agent modes.
    """
    service = _require_service(request)
    start_time = time.time()

    try:
        # Map API mode to service mode
        from osprey.services.ariel_search.models import SearchMode as ServiceSearchMode

        mode_map = {
            SearchMode.KEYWORD: ServiceSearchMode.KEYWORD,
            SearchMode.SEMANTIC: ServiceSearchMode.SEMANTIC,
            SearchMode.RAG: ServiceSearchMode.RAG,
            SearchMode.AGENT: ServiceSearchMode.AGENT,
        }
        service_mode = mode_map.get(search_req.mode)

        # Merge filter values: advanced_params takes precedence over top-level fields
        adv = search_req.advanced_params
        start_date = adv.pop("start_date", None) or search_req.start_date
        end_date = adv.pop("end_date", None) or search_req.end_date
        author = adv.pop("author", None) or search_req.author
        source_system = adv.pop("source_system", None) or search_req.source_system

        # Parse date strings from advanced_params if needed
        if isinstance(start_date, str) and start_date:
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str) and end_date:
            end_date = datetime.fromisoformat(end_date)

        # Build time range if provided
        time_range = None
        if start_date or end_date:
            time_range = (start_date, end_date)

        # Re-inject non-date filters into advanced_params for downstream use
        if author:
            adv["author"] = author
        if source_system:
            adv["source_system"] = source_system

        # Execute search
        result = await service.search(
            query=search_req.query,
            max_results=search_req.max_results,
            time_range=time_range,
            mode=service_mode,
            advanced_params=adv,
        )

        execution_time = int((time.time() - start_time) * 1000)

        # Convert entries to response format
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
            pipeline_details=_pipeline_details_to_response(
                getattr(result, "pipeline_details", None)
            ),
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
        # Get total count for pagination
        total = await service.repository.count_entries()

        # Fetch entries (offset calculation would be used when repository supports it)
        entries = await service.repository.search_by_time_range(
            start=start_date,
            end=end_date,
            limit=page_size,
        )

        # Convert to response format
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
        from osprey.services.ariel_search.models import FacilityEntryCreateRequest

        facility_request = FacilityEntryCreateRequest(
            subject=entry_req.subject,
            details=entry_req.details,
            author=entry_req.author,
            logbook=entry_req.logbook,
            shift=entry_req.shift,
            tags=entry_req.tags,
        )

        result = await service.create_entry(facility_request)

        return EntryCreateResponse(
            entry_id=result.entry_id,
            message=result.message,
            sync_status=result.sync_status.value,
            source_system=result.source_system,
        )

    except NotImplementedError:
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
            message=f"Entry {entry_id} created (local only — adapter does not support writes)",
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
        # Generate entry ID
        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        # Parse tags from comma-separated string
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        # Build entry
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

        # Process uploaded files
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
            enabled_pipelines=status.enabled_pipelines,
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

    # Validate YAML
    try:
        yaml.safe_load(req.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}") from e

    # Backup
    bak = path.with_suffix(".yml.bak")
    bak.write_text(path.read_text())

    # Write with fsync
    fd = os.open(str(path), os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
    try:
        os.write(fd, req.content.encode())
        os.fsync(fd)
    finally:
        os.close(fd)

    return {"status": "ok", "requires_restart": True}


# Claude Setup file management

_CLAUDE_SETUP_FILES = [
    ("CLAUDE.md", "Claude Code Config"),
    (".claude/settings.json", "Claude Code Config"),
    (".claude/rules/safety.md", "Claude Code Config"),
    (".mcp.json", "Claude Code Config"),
    ("config.yml", "OSPREY Config"),
]


@router.get("/claude-setup")
async def list_claude_setup_files() -> dict:
    """List Claude Code setup files with categories."""
    root = _find_project_root()
    files = []
    for rel, category in _CLAUDE_SETUP_FILES:
        fpath = root / rel
        files.append(
            {
                "path": rel,
                "category": category,
                "exists": fpath.exists(),
                "size": fpath.stat().st_size if fpath.exists() else 0,
            }
        )
    return {"project_root": str(root), "files": files}


@router.get("/claude-setup/{file_path:path}")
async def get_claude_setup_file(file_path: str) -> dict:
    """Read a Claude Code setup file."""
    root = _find_project_root()
    target = (root / file_path).resolve()

    # Path traversal protection
    if not target.is_relative_to(root.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return {"path": file_path, "content": target.read_text()}


class ClaudeSetupUpdateRequest(BaseModel):
    content: str


@router.put("/claude-setup/{file_path:path}")
async def update_claude_setup_file(file_path: str, req: ClaudeSetupUpdateRequest) -> dict:
    """Update a Claude Code setup file with backup + fsync."""
    root = _find_project_root()
    target = (root / file_path).resolve()

    # Path traversal protection
    if not target.is_relative_to(root.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    # Backup
    bak = target.with_suffix(target.suffix + ".bak")
    bak.write_text(target.read_text())

    # Write with fsync
    fd = os.open(str(target), os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
    try:
        os.write(fd, req.content.encode())
        os.fsync(fd)
    finally:
        os.close(fd)

    return {"status": "ok", "requires_restart": True}
