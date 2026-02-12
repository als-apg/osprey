"""ARIEL (Agentic Retrieval Interface for Electronic Logbooks) search service.

This module provides the public API for the ARIEL search service.

Four execution modes:
- **KEYWORD / SEMANTIC**: Direct calls to search functions
- **RAG** (deterministic): Hybrid retrieval + RRF fusion + LLM generation
- **AGENT** (agentic): ReAct agent with auto-discovered search tools

See 04_OSPREY_INTEGRATION.md Section 9.1 for the public API specification.
"""

from osprey.services.ariel_search.agent import (
    AgentExecutor,
    AgentResult,
)
from osprey.services.ariel_search.capability import (
    close_ariel_service,
    get_ariel_search_service,
    reset_ariel_service,
)
from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EnhancementModuleConfig,
    IngestionConfig,
    ModelConfig,
    PipelineModuleConfig,
    ReasoningConfig,
    SearchModuleConfig,
    WatchConfig,
)
from osprey.services.ariel_search.exceptions import (
    AdapterNotFoundError,
    ARIELException,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseQueryError,
    EmbeddingGenerationError,
    ErrorCategory,
    IngestionError,
    ModuleNotEnabledError,
    SearchExecutionError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.ingestion.scheduler import (
    IngestionPollResult,
    IngestionScheduler,
)
from osprey.services.ariel_search.models import (
    ARIELSearchRequest,
    ARIELSearchResult,
    ARIELStatusResult,
    AttachmentInfo,
    EmbeddingTableInfo,
    EnhancedLogbookEntry,
    IngestionEntryError,
    IngestionProgress,
    IngestionResult,
    MetadataSchema,
    SearchMode,
    enhanced_entry_from_row,
    resolve_time_range,
)
from osprey.services.ariel_search.rag import (
    RAGPipeline,
    RAGResult,
)
from osprey.services.ariel_search.service import (
    ARIELSearchService,
    create_ariel_service,
)

__all__ = [
    # Service
    "ARIELSearchService",
    "close_ariel_service",
    "create_ariel_service",
    "get_ariel_search_service",
    "reset_ariel_service",
    # Agent
    "AgentExecutor",
    "AgentResult",
    # Config classes
    "ARIELConfig",
    "DatabaseConfig",
    "EmbeddingConfig",
    "EnhancementModuleConfig",
    "IngestionConfig",
    "ModelConfig",
    "PipelineModuleConfig",
    "ReasoningConfig",
    "SearchModuleConfig",
    "WatchConfig",
    # Ingestion scheduler
    "IngestionPollResult",
    "IngestionScheduler",
    # Exceptions
    "AdapterNotFoundError",
    "ARIELException",
    "ConfigurationError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "EmbeddingGenerationError",
    "ErrorCategory",
    "IngestionError",
    "ModuleNotEnabledError",
    "SearchExecutionError",
    "SearchTimeoutError",
    # RAG
    "RAGPipeline",
    "RAGResult",
    # Models
    "ARIELSearchRequest",
    "ARIELSearchResult",
    "ARIELStatusResult",
    "AttachmentInfo",
    "EmbeddingTableInfo",
    "EnhancedLogbookEntry",
    "IngestionEntryError",
    "IngestionProgress",
    "IngestionResult",
    "MetadataSchema",
    "SearchMode",
    "enhanced_entry_from_row",
    "resolve_time_range",
]
