"""ARIEL (Agentic Retrieval Interface for Electronic Logbooks) search service.

This module provides the public API for the ARIEL search service.

See 04_OSPREY_INTEGRATION.md Section 9.1 for the public API specification.
"""

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
    ReasoningConfig,
    SearchModuleConfig,
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
    # Config classes
    "ARIELConfig",
    "DatabaseConfig",
    "EmbeddingConfig",
    "EnhancementModuleConfig",
    "IngestionConfig",
    "ModelConfig",
    "ReasoningConfig",
    "SearchModuleConfig",
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
