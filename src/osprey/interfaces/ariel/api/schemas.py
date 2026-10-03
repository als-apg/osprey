"""Pydantic schemas for ARIEL Web API.

Request and response models for the ARIEL search interface.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AttachmentResponse(BaseModel):
    """One attachment of an entry: its summary plus the web-only ``display_url``.

    The fields other than ``display_url`` are the attachment summary keys
    (``SUMMARY_KEYS``). ``url`` is the attachment's absolute http(s) source url,
    when it has one; ``display_url`` is where the web page shows or downloads
    the attachment from, or null when there is nowhere to.
    """

    attachment_id: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    viewable: bool = False
    copy_status: str | None = None
    skip_reason: str | None = None
    caption: str | None = None
    caption_source: str | None = None
    visible_text: str | None = None
    url: str | None = None
    display_url: str | None = None


class EntryResponse(BaseModel):
    """Single logbook entry in response."""

    entry_id: str
    source_system: str
    # Pre-rendered facility-local ISO-8601 strings (with explicit offset) via
    # to_facility_iso in _entry_to_response — kept as ``str`` (not ``datetime``) so
    # Pydantic does not re-parse and re-emit them in the DB's stored UTC offset,
    # which would silently undo the localization and reintroduce the web/MCP drift
    # this fixes. Nullable because the shared helper is None-safe (a missing value
    # renders as null rather than 500-ing); in practice the DB columns are present.
    timestamp: str | None
    author: str
    raw_text: str
    attachments: list[AttachmentResponse] = []
    metadata: dict = {}
    created_at: str | None
    updated_at: str | None
    summary: str | None = None
    keywords: list[str] = []
    score: float | None = None
    highlights: list[str] = []
    # How a search found the entry, and which of its attachments it matched.
    matched_via: list[str] = []
    matched_attachment_ids: list[str] = []


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str = Field(..., min_length=1, description="Search query text")
    # Name of a registered search module (as advertised by /api/capabilities).
    # Validated by the route against the enabled modules; omitting it selects
    # the service's default mode.
    mode: str | None = Field(None, description="Search module name")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    start_date: datetime | None = Field(None, description="Filter start date")
    end_date: datetime | None = Field(None, description="Filter end date")
    author: str | None = Field(None, description="Filter by author")
    source_system: str | None = Field(None, description="Filter by source system")
    advanced_params: dict[str, Any] = Field(
        default_factory=dict, description="Mode-specific advanced parameters"
    )


class DiagnosticResponse(BaseModel):
    """Structured diagnostic from search execution."""

    level: str
    source: str
    message: str
    category: str | None = None


class ExpandedTermResponse(BaseModel):
    """One vocabulary expansion group that the executed search contained.

    Mirrors ``ARIELSearchResult.expanded_terms`` on the wire: ``original`` is
    the span as the operator typed it, ``alternatives`` the canonical terms it
    was expanded to (several when a form is bound to several concepts).
    """

    original: str
    alternatives: list[str] = []


class SearchResponse(BaseModel):
    """Search response payload."""

    entries: list[EntryResponse]
    answer: str | None = None
    sources: list[str] = []
    search_modes_used: list[str] = []
    reasoning: str = ""
    total_results: int = 0
    execution_time_ms: int = 0
    diagnostics: list[DiagnosticResponse] = []
    expanded_terms: list[ExpandedTermResponse] = []


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
    metadata: dict | None = None
    auth_user: str | None = None
    auth_password: str | None = None


class EntryCreateResponse(BaseModel):
    """Response after creating an entry."""

    entry_id: str
    message: str = "Entry created successfully"
    sync_status: str | None = None
    source_system: str | None = None
    attachment_count: int = 0


class EmbeddingTableStatus(BaseModel):
    """Status of an embedding table."""

    table_name: str
    entry_count: int
    dimension: int | None = None
    is_active: bool = False


class HealthFacts(BaseModel):
    """The status facts the open ``/health`` page carries.

    An allow-list: a status fact reaches the page only by being declared here.
    Nothing that names the store's address, its login or its schema belongs on
    it, because the page answers without a credential.
    """

    entry_count: int
    last_ingestion: datetime | None = None
    enabled_search_modules: list[str]
    enabled_enhancement_modules: list[str]


class HealthResponse(BaseModel):
    """The open ``/health`` page.

    ``service`` is ``None`` when the panel has no search service or the store
    did not answer; the other fields are always present.
    """

    status: Literal["healthy", "degraded"]
    message: str
    config_status: str | None = None
    service: HealthFacts | None = None


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
