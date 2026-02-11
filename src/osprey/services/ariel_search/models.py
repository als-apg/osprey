"""ARIEL core models.

This module defines the core data models for ARIEL search service:
- EnhancedLogbookEntry: The core logbook entry data model
- ARIELSearchRequest: Request model for search operations
- ARIELSearchResult: Result model from search operations
- SearchMode: Search mode enumeration
- Supporting models for health checks, embedding tables, etc.

See 04_OSPREY_INTEGRATION.md Sections 4.1-4.5 for full specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, NotRequired, TypedDict


class AttachmentInfo(TypedDict):
    """Normalized attachment metadata.

    Captures the union of ALS/JLab/ORNL attachment fields.
    Only `url` is required; other fields depend on the source facility.
    """

    url: str  # Full URL to attachment
    type: NotRequired[str | None]  # MIME type (e.g., "image/png")
    filename: NotRequired[str | None]  # Display filename
    thumbnail_url: NotRequired[str | None]  # Thumbnail URL (JLab)
    caption: NotRequired[str | None]  # Caption (JLab)


class EnhancedLogbookEntry(TypedDict):
    """ARIEL's enriched logbook entry - the core data model.

    Core fields are always present (created by core_migration.py).
    Enhancement fields are added by enabled enhancement modules during ingestion.
    """

    # === CORE FIELDS (always present) ===
    entry_id: str  # Unique identifier from source system (PRIMARY KEY)
    source_system: str  # e.g., "ALS eLog", "JLab Logbook"
    timestamp: datetime  # Entry creation time in source system
    author: str  # Entry author (may be empty string)
    raw_text: str  # Merged content (subject+details for ALS, title+body for JLab)
    attachments: list[AttachmentInfo]  # May be empty list
    metadata: dict[str, Any]  # Facility-specific fields
    created_at: datetime  # When ingested into ARIEL
    updated_at: datetime  # Last modification in ARIEL

    # === ENHANCEMENT FIELDS (optional, added by modules) ===
    # Added by semantic_processor if enabled
    summary: NotRequired[str | None]  # AI-generated summary (V2)
    keywords: NotRequired[list[str]]  # Extracted keywords

    # Added by core migration for tracking
    enhancement_status: NotRequired[dict[str, Any]]  # Per-module status tracking


def enhanced_entry_from_row(row: Any) -> EnhancedLogbookEntry:
    """Convert database row to EnhancedLogbookEntry TypedDict.

    Args:
        row: psycopg3 Row object or dict from database query.
            psycopg3 Row objects support dict-like access via keys().

    Returns:
        EnhancedLogbookEntry with all fields populated.
        Enhancement fields are included only if present in the row.
    """
    # psycopg3 Row objects support dict() conversion
    row_dict = dict(row) if hasattr(row, "keys") else row

    # Build entry with core fields (always present)
    entry: EnhancedLogbookEntry = {
        "entry_id": row_dict["entry_id"],
        "source_system": row_dict["source_system"],
        "timestamp": row_dict["timestamp"],  # psycopg3 handles datetime conversion
        "author": row_dict.get("author", ""),
        "raw_text": row_dict["raw_text"],
        "attachments": row_dict.get("attachments", []),
        "metadata": row_dict.get("metadata", {}),
        "created_at": row_dict["created_at"],
        "updated_at": row_dict["updated_at"],
    }

    # Add enhancement fields if present (NotRequired fields)
    if row_dict.get("summary") is not None:
        entry["summary"] = row_dict["summary"]
    if row_dict.get("keywords") is not None:
        entry["keywords"] = row_dict["keywords"]
    if row_dict.get("enhancement_status") is not None:
        entry["enhancement_status"] = row_dict["enhancement_status"]

    return entry


class SearchMode(Enum):
    """Search mode enumeration.

    Attributes:
        KEYWORD: PostgreSQL full-text search (direct function call)
        SEMANTIC: Embedding similarity search (direct function call)
        RAG: Deterministic RAG pipeline with hybrid retrieval, RRF fusion, and LLM generation
        AGENT: Agentic orchestration with ReAct agent (AgentExecutor)
    """

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    RAG = "rag"
    AGENT = "agent"


@dataclass
class ARIELSearchRequest:
    """Request model for ARIEL search service.

    Captures all information needed to execute a search workflow.

    Attributes:
        query: The search query text
        modes: Search modes to use (default: [RAG])
        time_range: Default time range filter (see Time Range Semantics)
        facility: Facility filter
        max_results: Maximum results to return (default: 10, range: 1-100)
        include_images: Include image attachments (default: False)
        capability_context_data: Context from main graph state
    """

    query: str
    modes: list[SearchMode] = field(default_factory=lambda: [SearchMode.RAG])
    time_range: tuple[datetime, datetime] | None = None
    facility: str | None = None
    max_results: int = 10
    include_images: bool = False
    capability_context_data: dict[str, Any] = field(default_factory=dict)
    advanced_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request fields."""
        if not self.query or not self.query.strip():
            raise ValueError("query is required and cannot be empty")
        if self.max_results < 1:
            self.max_results = 1
        elif self.max_results > 100:
            self.max_results = 100


@dataclass(frozen=True)
class ARIELSearchResult:
    """Structured, type-safe result from ARIEL search service.

    Frozen dataclass for immutability.

    Attributes:
        entries: Matching entries, ranked by relevance
        answer: RAG-generated answer (optional)
        sources: Entry IDs used as sources
        search_modes_used: Modes that were executed
        reasoning: Explanation of results
    """

    entries: tuple[EnhancedLogbookEntry, ...]
    answer: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)
    search_modes_used: tuple[SearchMode, ...] = field(default_factory=tuple)
    reasoning: str = ""


@dataclass
class EmbeddingTableInfo:
    """Information about an embedding model table.

    Attributes:
        table_name: e.g., "text_embeddings_nomic_embed_text"
        entry_count: Number of entries with embeddings
        dimension: Vector dimension (from table schema)
        is_active: True if this is the configured search model
    """

    table_name: str
    entry_count: int
    dimension: int | None = None
    is_active: bool = False


@dataclass
class ARIELStatusResult:
    """Detailed ARIEL service status for CLI display.

    Attributes:
        healthy: Overall health status
        database_connected: Whether database is reachable
        database_uri: Masked URI for security
        entry_count: Number of entries in database
        embedding_tables: Information about embedding tables
        active_embedding_model: Currently configured search model
        enabled_search_modules: List of enabled search modules
        enabled_enhancement_modules: List of enabled enhancement modules
        last_ingestion: Timestamp of last completed ingestion
        errors: List of error messages
    """

    healthy: bool
    database_connected: bool
    database_uri: str
    entry_count: int | None
    embedding_tables: list[EmbeddingTableInfo]
    active_embedding_model: str | None
    enabled_search_modules: list[str]
    enabled_pipelines: list[str]
    enabled_enhancement_modules: list[str]
    last_ingestion: datetime | None
    errors: list[str]


@dataclass
class IngestionEntryError:
    """Error information for a failed entry during ingestion.

    Attributes:
        entry_id: ID of the entry that failed (if available)
        error: Error message
        raw_data: Original entry data (truncated)
    """

    entry_id: str | None
    error: str
    raw_data: str | None = None


@dataclass
class IngestionProgress:
    """Progress information during ingestion.

    Attributes:
        total: Total entries to process
        processed: Entries processed so far
        succeeded: Entries successfully ingested
        failed: Entries that failed
    """

    total: int
    processed: int
    succeeded: int
    failed: int


@dataclass
class IngestionResult:
    """Result from an ingestion operation.

    Attributes:
        source_system: Source system name
        total_entries: Total entries processed
        succeeded: Entries successfully ingested
        failed: Entries that failed
        errors: Detailed error information
        duration_seconds: Time taken for ingestion
    """

    source_system: str
    total_entries: int
    succeeded: int
    failed: int
    errors: list[IngestionEntryError]
    duration_seconds: float


class MetadataSchema(TypedDict, total=False):
    """Unified metadata schema for cross-facility standardization.

    All fields are optional to accommodate different facilities.
    See 01_DATA_LAYER.md Section 5.9 for full specification.
    """

    # ALS-specific
    logbook: str | None
    tag: str | None
    shift: str | None
    activity_type: str | None

    # JLab-specific
    logbook_name: str | None
    entry_type: str | None
    references: list[str] | None

    # ORNL-specific
    event_time: str | None  # When the event occurred (vs entry_time)
    facility_section: str | None


def resolve_time_range(
    tool_start: datetime | None,
    tool_end: datetime | None,
    request: ARIELSearchRequest,
) -> tuple[datetime | None, datetime | None]:
    """Resolve time range with 3-tier priority.

    Priority:
    1. Explicit tool params override request context
    2. Fall back to request context
    3. No filtering

    Args:
        tool_start: Start date from tool call (highest priority)
        tool_end: End date from tool call (highest priority)
        request: The original search request with potential time_range

    Returns:
        Tuple of (start_date, end_date), either or both may be None
    """
    # Explicit tool params override request context
    if tool_start is not None or tool_end is not None:
        return (tool_start, tool_end)
    # Fall back to request context
    if request.time_range:
        return request.time_range
    # No filtering
    return (None, None)
