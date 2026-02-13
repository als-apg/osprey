"""ARIEL exception hierarchy.

This module defines the exception hierarchy for ARIEL search service errors.
All exceptions inherit from ARIELException and are categorized by error type
to enable appropriate recovery strategies.

See 04_OSPREY_INTEGRATION.md Sections 2.1-2.2 for full specification.
"""

from enum import Enum


class ErrorCategory(Enum):
    """Error category for recovery strategy determination.

    Attributes:
        DATABASE: Connection/query errors - may retry after delay
        EMBEDDING: Embedding model failures - retry with fallback
        SEARCH: Search execution errors - no automatic retry
        INGESTION: Data ingestion issues - no automatic retry
        CONFIGURATION: Invalid configuration - no automatic retry
        TIMEOUT: Execution timeout exceeded - no automatic retry
    """

    DATABASE = "database"
    EMBEDDING = "embedding"
    SEARCH = "search"
    INGESTION = "ingestion"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"


class ARIELException(Exception):
    """Base exception for all ARIEL search service errors.

    Attributes:
        message: Human-readable error description
        category: Error category for recovery strategy
        technical_details: Additional debugging information
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize ARIELException.

        Args:
            message: Human-readable error description
            category: Error category for recovery strategy
            technical_details: Additional debugging information
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.technical_details = technical_details or {}

    @property
    def is_retriable(self) -> bool:
        """Return True for DATABASE and EMBEDDING categories."""
        return self.category in (ErrorCategory.DATABASE, ErrorCategory.EMBEDDING)


# === DATABASE category (is_retriable=True) ===


class DatabaseConnectionError(ARIELException):
    """Database connection failure.

    Raised when unable to connect to the ARIEL PostgreSQL database.
    """

    def __init__(
        self,
        message: str,
        connection_details: dict | None = None,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize DatabaseConnectionError.

        Args:
            message: Human-readable error description
            connection_details: Connection parameters (sanitized, no passwords)
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        if connection_details:
            details["connection_details"] = connection_details
        super().__init__(message, ErrorCategory.DATABASE, details)
        self.connection_details = connection_details or {}


class DatabaseQueryError(ARIELException):
    """Database query execution failure.

    Raised when a database query fails during execution.
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize DatabaseQueryError.

        Args:
            message: Human-readable error description
            query: The failed query (may be truncated for large queries)
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        if query:
            details["query"] = query[:500] if len(query) > 500 else query
        super().__init__(message, ErrorCategory.DATABASE, details)


# === EMBEDDING category (is_retriable=True) ===


class EmbeddingGenerationError(ARIELException):
    """Embedding generation failure.

    Raised when the embedding model fails to generate embeddings.
    """

    def __init__(
        self,
        message: str,
        model_name: str,
        input_text: str | None = None,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize EmbeddingGenerationError.

        Args:
            message: Human-readable error description
            model_name: Name of the embedding model that failed
            input_text: Input text (truncated to 100 chars)
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["model_name"] = model_name
        if input_text:
            details["input_text"] = input_text[:100] if len(input_text) > 100 else input_text
        super().__init__(message, ErrorCategory.EMBEDDING, details)
        self.model_name = model_name
        self.input_text = input_text[:100] if input_text and len(input_text) > 100 else input_text


# === SEARCH category (is_retriable=False) ===


class SearchExecutionError(ARIELException):
    """Search execution failure.

    Raised when a search operation fails during execution.
    """

    def __init__(
        self,
        message: str,
        search_mode: str,
        query: str,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize SearchExecutionError.

        Args:
            message: Human-readable error description
            search_mode: The search mode that failed (keyword, semantic, rag)
            query: The search query
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["search_mode"] = search_mode
        details["query"] = query[:200] if len(query) > 200 else query
        super().__init__(message, ErrorCategory.SEARCH, details)
        self.search_mode = search_mode
        self.query = query


# === INGESTION category (is_retriable=False) ===


class IngestionError(ARIELException):
    """Data ingestion failure.

    Raised when data ingestion fails during processing.
    """

    def __init__(
        self,
        message: str,
        source_system: str,
        entries_affected: int = 0,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize IngestionError.

        Args:
            message: Human-readable error description
            source_system: The source system being ingested
            entries_affected: Number of entries affected by the error
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["source_system"] = source_system
        details["entries_affected"] = entries_affected
        super().__init__(message, ErrorCategory.INGESTION, details)
        self.source_system = source_system
        self.entries_affected = entries_affected


class AdapterNotFoundError(ARIELException):
    """Ingestion adapter not found.

    Raised when a requested ingestion adapter is not registered.
    """

    def __init__(
        self,
        message: str,
        adapter_name: str,
        available_adapters: list[str] | None = None,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize AdapterNotFoundError.

        Args:
            message: Human-readable error description
            adapter_name: Name of the adapter that was not found
            available_adapters: List of available adapter names
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["adapter_name"] = adapter_name
        if available_adapters:
            details["available_adapters"] = available_adapters
        super().__init__(message, ErrorCategory.INGESTION, details)
        self.adapter_name = adapter_name
        self.available_adapters = available_adapters or []


# === CONFIGURATION category (is_retriable=False) ===


class ConfigurationError(ARIELException):
    """Invalid configuration.

    Raised when ARIEL configuration is invalid.
    """

    def __init__(
        self,
        message: str,
        config_key: str,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Human-readable error description
            config_key: The configuration key that is invalid
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["config_key"] = config_key
        super().__init__(message, ErrorCategory.CONFIGURATION, details)
        self.config_key = config_key


class ModuleNotEnabledError(ARIELException):
    """Module not enabled in configuration.

    Raised when attempting to use a module that is not enabled.
    """

    def __init__(
        self,
        message: str,
        module_name: str,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize ModuleNotEnabledError.

        Args:
            message: Human-readable error description
            module_name: Name of the module that is not enabled
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["module_name"] = module_name
        super().__init__(message, ErrorCategory.CONFIGURATION, details)
        self.module_name = module_name


# === TIMEOUT category (is_retriable=False) ===


class SearchTimeoutError(ARIELException):
    """Search timeout exceeded.

    Raised when search execution exceeds the configured timeout.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: int,
        operation: str,
        technical_details: dict | None = None,
    ) -> None:
        """Initialize SearchTimeoutError.

        Args:
            message: Human-readable error description
            timeout_seconds: The timeout value that was exceeded
            operation: The operation that timed out
            technical_details: Additional debugging information
        """
        details = technical_details or {}
        details["timeout_seconds"] = timeout_seconds
        details["operation"] = operation
        super().__init__(message, ErrorCategory.TIMEOUT, details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
