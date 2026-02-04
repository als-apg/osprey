"""ARIEL Web API routes.

REST endpoints for search, entry management, and status.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
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


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, search_req: SearchRequest) -> SearchResponse:
    """Execute search query.

    Supports keyword, semantic, RAG, and auto modes.
    """
    service: ARIELSearchService = request.app.state.ariel_service
    start_time = time.time()

    try:
        # Map API mode to service mode
        from osprey.services.ariel_search.models import SearchMode as ServiceSearchMode

        mode_map = {
            SearchMode.AUTO: None,
            SearchMode.KEYWORD: ServiceSearchMode.KEYWORD,
            SearchMode.SEMANTIC: ServiceSearchMode.SEMANTIC,
            SearchMode.RAG: ServiceSearchMode.RAG,
        }
        service_mode = mode_map.get(search_req.mode)

        # Build time range if provided
        time_range = None
        if search_req.start_date or search_req.end_date:
            time_range = (search_req.start_date, search_req.end_date)

        # Execute search
        result = await service.search(
            query=search_req.query,
            max_results=search_req.max_results,
            time_range=time_range,
            mode=service_mode,
        )

        execution_time = int((time.time() - start_time) * 1000)

        # Convert entries to response format
        entries = [_entry_to_response(e) for e in result.entries]

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
            enabled_enhancement_modules=status.enabled_enhancement_modules,
            last_ingestion=status.last_ingestion,
            errors=status.errors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
