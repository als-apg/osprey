"""Pydantic schemas for ARIEL Web API.

Request and response models for the ARIEL search interface.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search mode options."""

    AUTO = "auto"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    RAG = "rag"
    MULTI = "multi"
    AGENT = "agent"


class FusionStrategy(str, Enum):
    """Fusion strategy for hybrid search."""

    RRF = "rrf"
    WEIGHTED = "weighted"


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
    mode: SearchMode = Field(SearchMode.AUTO, description="Search mode")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    start_date: datetime | None = Field(None, description="Filter start date")
    end_date: datetime | None = Field(None, description="Filter end date")
    author: str | None = Field(None, description="Filter by author")
    source_system: str | None = Field(None, description="Filter by source system")

    # Advanced retrieval parameters
    similarity_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity for semantic search"
    )
    include_highlights: bool | None = Field(
        None, description="Include highlighted snippets for keyword search"
    )
    fuzzy_fallback: bool | None = Field(
        None, description="Fall back to fuzzy search if no exact matches"
    )

    # Advanced assembly parameters
    assembly_max_items: int | None = Field(
        None, ge=1, le=50, description="Maximum items in context"
    )
    assembly_max_chars: int | None = Field(
        None, ge=1000, le=50000, description="Maximum total characters in context"
    )
    assembly_max_chars_per_item: int | None = Field(
        None, ge=100, le=10000, description="Maximum characters per item"
    )

    # Advanced processing parameters (RAG mode)
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="LLM temperature for generation"
    )
    max_tokens: int | None = Field(
        None, ge=64, le=4096, description="Maximum tokens in generated response"
    )

    # Advanced fusion parameters (MULTI mode)
    fusion_strategy: FusionStrategy | None = Field(
        None, description="Fusion strategy for hybrid search"
    )
    keyword_weight: float | None = Field(
        None, ge=0.0, le=1.0, description="Weight for keyword search in fusion"
    )
    semantic_weight: float | None = Field(
        None, ge=0.0, le=1.0, description="Weight for semantic search in fusion"
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


class EntriesListRequest(BaseModel):
    """Request for listing entries with pagination."""

    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    start_date: datetime | None = None
    end_date: datetime | None = None
    author: str | None = None
    source_system: str | None = None
    sort_by: str = Field("timestamp", description="Sort field")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")


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
    enabled_enhancement_modules: list[str] = []
    last_ingestion: datetime | None = None
    errors: list[str] = []
