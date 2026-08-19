"""ARIEL core models.

This module defines the core data models for ARIEL search service:
- EnhancedLogbookEntry: The core logbook entry data model
- ARIELSearchRequest: Request model for search operations
- ARIELSearchResult: Result model from search operations
- normalize_search_mode: Normalizer for search mode names
- Supporting models for health checks, embedding tables, etc.

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, NotRequired

from typing_extensions import TypedDict


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
    auth_user: str | None = None
    auth_password: str | None = None


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


class AttachmentFileRecord(TypedDict):
    """Row from the attachment_files table."""

    attachment_id: str
    entry_id: str
    filename: str
    mime_type: str | None
    data: NotRequired[bytes]  # Omitted in metadata-only queries
    size_bytes: int
    created_at: datetime


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


DEFAULT_SEARCH_MODE = "keyword"

PREFERRED_SEARCH_MODE = "hybrid"


def resolve_implicit_search_mode(is_enabled: Callable[[str], bool]) -> str:
    """Pick the mode a search runs in when the config names none.

    This is the fallback for a deployment whose config predates
    ``ariel.default_search_mode`` or deliberately leaves it unset.
    ``hybrid`` is the preferred answer, but it only exists in deployments
    that run the qmd sidecar, so it is preferred rather than assumed.
    ``keyword`` needs nothing beyond the logbook database and is always the
    safe fallback.

    A deployment that sets ``ariel.default_search_mode`` never reaches this —
    see :meth:`osprey.services.ariel_search.config.ARIELConfig
    .resolve_default_search_mode`.

    Args:
        is_enabled: Predicate answering whether a search module name is
            enabled in the current deployment's configuration.

    Returns:
        ``PREFERRED_SEARCH_MODE`` when that module is enabled, else
        ``DEFAULT_SEARCH_MODE``.
    """
    if is_enabled(PREFERRED_SEARCH_MODE):
        return PREFERRED_SEARCH_MODE
    return DEFAULT_SEARCH_MODE


def normalize_search_mode(mode: object) -> str:
    """Normalize a search mode name to its canonical lowercase form.

    A search mode is the registry name of a search module (e.g. ``"keyword"``,
    ``"semantic"``). Whether a given name is actually registered and enabled is
    decided by the search service and the API layer, not here.

    Args:
        mode: Candidate mode name. Must be a non-empty string.

    Returns:
        The mode name, stripped of surrounding whitespace and lowercased.

    Raises:
        ValueError: If ``mode`` is not a string, or is empty/whitespace-only.
    """
    if not isinstance(mode, str):
        raise ValueError(f"search mode must be a string, got {type(mode).__name__}")
    normalized = mode.strip().lower()
    if not normalized:
        raise ValueError("search mode cannot be empty")
    return normalized


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


@dataclass
class ARIELSearchRequest:
    """Request model for ARIEL search service.

    Captures all information needed to execute a search workflow.

    Attributes:
        query: The search query text
        modes: Search module names to use, normalized to lowercase
            (default: ``["keyword"]``)
        time_range: Default time range filter (see Time Range Semantics)
        facility: Facility filter
        max_results: Maximum results to return (default: 10, range: 1-100)
        include_images: Include image attachments (default: False)
    """

    query: str
    modes: list[str] = field(default_factory=lambda: [DEFAULT_SEARCH_MODE])
    time_range: tuple[datetime, datetime] | None = None
    facility: str | None = None
    max_results: int = 10
    include_images: bool = False
    advanced_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request fields and normalize mode names.

        Raises:
            ValueError: If the query is empty, or a mode is not a non-empty
                string.
        """
        if not self.query or not self.query.strip():
            raise ValueError("query is required and cannot be empty")
        self.modes = [normalize_search_mode(mode) for mode in self.modes]
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
        answer: Optional answer text (set by callers that synthesize output)
        sources: Entry IDs used as sources
        search_modes_used: Names of the search modules that were executed
        reasoning: Explanation of results
    """

    entries: tuple[dict[str, Any], ...]
    answer: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)
    search_modes_used: tuple[str, ...] = field(default_factory=tuple)
    reasoning: str = ""
    diagnostics: tuple[SearchDiagnostic, ...] = field(default_factory=tuple)


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
    enabled_enhancement_modules: list[str]
    last_ingestion: datetime | None
    errors: list[str]


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
