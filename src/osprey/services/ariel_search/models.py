"""ARIEL core models.

This module defines the core data models for ARIEL search service:
- EnhancedLogbookEntry: The core logbook entry data model
- ARIELSearchRequest: Request model for search operations
- ARIELSearchResult: Result model from search operations
- SearchMode: Search mode enumeration
- Supporting models for health checks, embedding tables, etc.

"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, NotRequired, TypedDict


class SyncStatus(Enum):
    """Synchronization status for facility-written entries."""

    SYNCED = "synced"
    PENDING_SYNC = "pending_sync"
    LOCAL_ONLY = "local_only"


@dataclass
class FacilityEntryCreateRequest:
    """Request to create a logbook entry through a facility adapter."""

    subject: str
    details: str
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] = field(default_factory=list)
    attachment_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FacilityEntryCreateResult:
    """Result from creating a logbook entry through a facility adapter."""

    entry_id: str
    source_system: str
    sync_status: SyncStatus
    message: str
    attachment_count: int = 0


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

    Core fields are always present.
    Enhancement fields are added by enabled enhancement modules during ingestion.
    """

    entry_id: str  # Unique identifier from source system (PRIMARY KEY)
    source_system: str  # e.g., "ALS eLog", "JLab Logbook"
    timestamp: datetime  # Entry creation time in source system
    author: str  # Entry author (may be empty string)
    raw_text: str  # Merged content (subject+details for ALS, title+body for JLab)
    attachments: list[AttachmentInfo]  # May be empty list
    metadata: dict[str, Any]  # Facility-specific fields
    created_at: datetime  # When ingested into ARIEL
    updated_at: datetime  # Last modification in ARIEL

    summary: NotRequired[str | None]
    keywords: NotRequired[list[str]]
    enhancement_status: NotRequired[dict[str, Any]]


def enhanced_entry_from_row(row: Any) -> EnhancedLogbookEntry:
    """Convert database row to EnhancedLogbookEntry TypedDict.

    Args:
        row: psycopg3 Row object or dict from database query.
            psycopg3 Row objects support dict-like access via keys().

    Returns:
        EnhancedLogbookEntry with all fields populated.
        Enhancement fields are included only if present in the row.
    """
    row_dict = dict(row) if hasattr(row, "keys") else row

    entry: EnhancedLogbookEntry = {
        "entry_id": row_dict["entry_id"],
        "source_system": row_dict["source_system"],
        "timestamp": row_dict["timestamp"],
        "author": row_dict.get("author", ""),
        "raw_text": row_dict["raw_text"],
        "attachments": row_dict.get("attachments", []),
        "metadata": row_dict.get("metadata", {}),
        "created_at": row_dict["created_at"],
        "updated_at": row_dict["updated_at"],
    }

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


class DiagnosticLevel(Enum):
    """Severity level for search diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class SearchDiagnostic:
    """Structured diagnostic from search execution.

    Attributes:
        level: Severity level (info, warning, error)
        source: Dot-separated origin identifier (e.g. "rag.retrieve.keyword")
        message: Human-readable description
        category: Optional category mapping to ErrorCategory values
    """

    level: DiagnosticLevel
    source: str
    message: str
    category: str | None = None


@dataclass(frozen=True)
class RAGStageStats:
    """Stage-by-stage counts from the RAG pipeline.

    Attributes:
        keyword_retrieved: Number of entries from keyword search
        semantic_retrieved: Number of entries from semantic search
        fused_count: Number of unique entries after RRF fusion
        context_included: Number of entries included in LLM context
        context_truncated: Whether context was truncated to fit limits
    """

    keyword_retrieved: int = 0
    semantic_retrieved: int = 0
    fused_count: int = 0
    context_included: int = 0
    context_truncated: bool = False


@dataclass(frozen=True)
class AgentToolInvocation:
    """Record of a single tool call by the agent.

    Attributes:
        tool_name: Name of the tool invoked
        tool_args: Arguments passed to the tool
        result_summary: Truncated summary of the tool result
        order: Sequence number of this invocation
    """

    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    order: int = 0


@dataclass(frozen=True)
class AgentStep:
    """A single step in the agent's ReAct trace.

    Attributes:
        step_type: One of "user_query", "reasoning", "tool_call", "tool_result", "final_answer"
        content: Text content of the step
        tool_name: Tool name (for tool_call/tool_result steps)
        order: Sequence number of this step
    """

    step_type: str
    content: str = ""
    tool_name: str | None = None
    order: int = 0


@dataclass(frozen=True)
class PipelineDetails:
    """Intermediate pipeline data for developer visibility.

    Attributes:
        pipeline_type: "rag" or "agent"
        rag_stats: Stage counts (RAG mode only)
        agent_tool_invocations: Tool call records (agent mode only)
        agent_steps: ReAct trace steps (agent mode only)
        step_summary: Human-readable summary (e.g. "2 tool call(s): keyword, semantic")
    """

    pipeline_type: str
    rag_stats: RAGStageStats | None = None
    agent_tool_invocations: tuple[AgentToolInvocation, ...] = field(default_factory=tuple)
    agent_steps: tuple[AgentStep, ...] = field(default_factory=tuple)
    step_summary: str = ""


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

    entries: tuple[dict[str, Any], ...]
    answer: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)
    search_modes_used: tuple[SearchMode, ...] = field(default_factory=tuple)
    reasoning: str = ""
    diagnostics: tuple[SearchDiagnostic, ...] = field(default_factory=tuple)
    pipeline_details: PipelineDetails | None = None


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
    """

    # ALS
    logbook: str | None
    tag: str | None
    shift: str | None
    activity_type: str | None

    # JLab
    logbook_name: str | None
    entry_type: str | None
    references: list[str] | None

    # ORNL
    event_time: str | None  # When the event occurred (vs entry_time)
    facility_section: str | None


def _format_entry_base(entry: EnhancedLogbookEntry) -> dict[str, Any]:
    """Format the common fields of a logbook entry for agent consumption.

    Args:
        entry: EnhancedLogbookEntry

    Returns:
        Dict with entry_id, timestamp, author, text, and title
    """
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],
        "title": entry.get("metadata", {}).get("title"),
    }


def resolve_time_range(
    tool_start: datetime | None,
    tool_end: datetime | None,
    fallback_range: tuple[datetime, datetime] | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Resolve time range with 3-tier priority.

    Priority:
    1. Explicit tool params override fallback
    2. Fall back to fallback_range
    3. No filtering

    Args:
        tool_start: Start date from tool call (highest priority)
        tool_end: End date from tool call (highest priority)
        fallback_range: Optional fallback time range tuple

    Returns:
        Tuple of (start_date, end_date), either or both may be None
    """
    if tool_start is not None or tool_end is not None:
        return (tool_start, tool_end)
    if fallback_range:
        return fallback_range
    return (None, None)
