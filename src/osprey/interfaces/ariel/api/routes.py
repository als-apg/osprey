"""ARIEL Web API routes.

REST endpoints for search, entry management, and status.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request

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
    """Create a new logbook entry.

    Delegates to the facility adapter when write support is available.
    Falls back to direct database insert if the adapter doesn't support writes.
    """
    service: ARIELSearchService = request.app.state.ariel_service

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
        now = datetime.now()

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
            message=f"Entry {entry_id} created (local only — adapter does not support writes)",
            sync_status="local_only",
            source_system="ARIEL Web",
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
