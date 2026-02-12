"""ARIEL Web API routes.

REST endpoints for search, entry management, and status.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from osprey.interfaces.ariel.api.schemas import (
    EntriesListResponse,
    EntryCreateRequest,
    EntryCreateResponse,
    EntryResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search import ARIELSearchService

router = APIRouter(prefix="/api")


def _entry_to_response(
    entry: dict,
    score: float | None = None,
    highlights: list[str] | None = None,
) -> EntryResponse:
    """Convert database entry to response model."""
    return EntryResponse(
        entry_id=entry["entry_id"],
        source_system=entry["source_system"],
        timestamp=entry["timestamp"],
        author=entry.get("author", ""),
        raw_text=entry["raw_text"],
        attachments=entry.get("attachments", []),
        metadata=entry.get("metadata", {}),
        created_at=entry["created_at"],
        updated_at=entry["updated_at"],
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

    service: ARIELSearchService = request.app.state.ariel_service
    return _get_caps(service.config)


@router.get("/filter-options/{field_name}")
async def get_filter_options(request: Request, field_name: str) -> dict:
    """Return distinct values for a filterable field.

    Used by dynamic_select parameters to populate dropdown options.
    """
    service: ARIELSearchService = request.app.state.ariel_service

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
    service: ARIELSearchService = request.app.state.ariel_service
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
        entries = [_entry_to_response(e, highlights=e.get("_highlights")) for e in result.entries]

        return SearchResponse(
            entries=entries,
            answer=result.answer,
            sources=list(result.sources),
            search_modes_used=[m.value for m in result.search_modes_used],
            reasoning=result.reasoning,
            total_results=len(entries),
            execution_time_ms=execution_time,
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
    service: ARIELSearchService = request.app.state.ariel_service

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
    service: ARIELSearchService = request.app.state.ariel_service

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
    """Create a new logbook entry."""
    service: ARIELSearchService = request.app.state.ariel_service

    try:
        # Generate entry ID
        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        # Build entry
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
            },
            "created_at": now,
            "updated_at": now,
        }

        await service.repository.upsert_entry(entry)

        return EntryCreateResponse(
            entry_id=entry_id,
            message=f"Entry {entry_id} created successfully",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request) -> StatusResponse:
    """Get service status and health information."""
    service: ARIELSearchService = request.app.state.ariel_service

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
