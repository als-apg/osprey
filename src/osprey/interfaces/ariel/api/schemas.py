"""Pydantic schemas for ARIEL Web API.

Request and response models for the ARIEL search interface.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    """Search mode options."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    RAG = "rag"
    AGENT = "agent"


class AttachmentResponse(BaseModel):
    """Attachment metadata in response."""

    url: str
    type: str | None = None
    filename: str | None = None
    thumbnail_url: str | None = None
    caption: str | None = None


class EntryResponse(BaseModel):
    """Single logbook entry in response."""

    entry_id: str
    source_system: str
    timestamp: datetime
    author: str
    raw_text: str
    attachments: list[AttachmentResponse] = []
    metadata: dict = {}
    created_at: datetime
    updated_at: datetime
    summary: str | None = None
    keywords: list[str] = []
    score: float | None = None
    highlights: list[str] = []


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str = Field(..., min_length=1, description="Search query text")
    mode: SearchMode = Field(SearchMode.RAG, description="Search mode")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    start_date: datetime | None = Field(None, description="Filter start date")
    end_date: datetime | None = Field(None, description="Filter end date")
    author: str | None = Field(None, description="Filter by author")
    source_system: str | None = Field(None, description="Filter by source system")
    advanced_params: dict[str, Any] = Field(
        default_factory=dict, description="Mode-specific advanced parameters"
    )


class SearchResponse(BaseModel):
    """Search response payload."""

    entries: list[EntryResponse]
    answer: str | None = None
    sources: list[str] = []
    search_modes_used: list[str] = []
    reasoning: str = ""
    total_results: int = 0
    execution_time_ms: int = 0


class EntriesListResponse(BaseModel):
    """Response for entry listing."""

    entries: list[EntryResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class EntryCreateRequest(BaseModel):
    """Request to create a new logbook entry."""

    subject: str = Field(..., min_length=1, description="Entry subject/title")
    details: str = Field(..., min_length=1, description="Entry details/body")
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] = []
    attachment_ids: list[str] = []


class EntryCreateResponse(BaseModel):
    """Response after creating an entry."""

    entry_id: str
    message: str = "Entry created successfully"


class EmbeddingTableStatus(BaseModel):
    """Status of an embedding table."""

    table_name: str
    entry_count: int
    dimension: int | None = None
    is_active: bool = False


class StatusResponse(BaseModel):
    """Service status response."""

    healthy: bool
    database_connected: bool
    database_uri: str
    entry_count: int | None = None
    embedding_tables: list[EmbeddingTableStatus] = []
    active_embedding_model: str | None = None
    enabled_search_modules: list[str] = []
    enabled_pipelines: list[str] = []
    enabled_enhancement_modules: list[str] = []
    last_ingestion: datetime | None = None
    errors: list[str] = []
