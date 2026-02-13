"""Tests for ARIEL exception hierarchy."""

import pytest

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


class TestErrorCategory:
    """Tests for ErrorCategory enumeration."""

    def test_category_values(self) -> None:
        """Test that all expected categories exist."""
        assert ErrorCategory.DATABASE.value == "database"
        assert ErrorCategory.EMBEDDING.value == "embedding"
        assert ErrorCategory.SEARCH.value == "search"
        assert ErrorCategory.INGESTION.value == "ingestion"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.TIMEOUT.value == "timeout"


class TestARIELException:
    """Tests for ARIELException base class."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = ARIELException("Test error", ErrorCategory.SEARCH)
        assert exc.message == "Test error"
        assert exc.category == ErrorCategory.SEARCH
        assert exc.technical_details == {}
        assert str(exc) == "Test error"

    def test_with_technical_details(self) -> None:
        """Test exception with technical details."""
        details = {"key": "value", "count": 42}
        exc = ARIELException("Test error", ErrorCategory.DATABASE, details)
        assert exc.technical_details == details

    def test_is_retriable_database(self) -> None:
        """Test is_retriable for DATABASE category."""
        exc = ARIELException("DB error", ErrorCategory.DATABASE)
        assert exc.is_retriable is True

    def test_is_retriable_embedding(self) -> None:
        """Test is_retriable for EMBEDDING category."""
        exc = ARIELException("Embedding error", ErrorCategory.EMBEDDING)
        assert exc.is_retriable is True

    def test_is_retriable_search(self) -> None:
        """Test is_retriable for SEARCH category."""
        exc = ARIELException("Search error", ErrorCategory.SEARCH)
        assert exc.is_retriable is False

    def test_is_retriable_ingestion(self) -> None:
        """Test is_retriable for INGESTION category."""
        exc = ARIELException("Ingestion error", ErrorCategory.INGESTION)
        assert exc.is_retriable is False

    def test_is_retriable_configuration(self) -> None:
        """Test is_retriable for CONFIGURATION category."""
        exc = ARIELException("Config error", ErrorCategory.CONFIGURATION)
        assert exc.is_retriable is False

    def test_is_retriable_timeout(self) -> None:
        """Test is_retriable for TIMEOUT category."""
        exc = ARIELException("Timeout error", ErrorCategory.TIMEOUT)
        assert exc.is_retriable is False


class TestDatabaseConnectionError:
    """Tests for DatabaseConnectionError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = DatabaseConnectionError("Connection failed")
        assert exc.message == "Connection failed"
        assert exc.category == ErrorCategory.DATABASE
        assert exc.is_retriable is True
        assert exc.connection_details == {}

    def test_with_connection_details(self) -> None:
        """Test exception with connection details."""
        details = {"host": "localhost", "port": 5432}
        exc = DatabaseConnectionError("Connection failed", connection_details=details)
        assert exc.connection_details == details
        assert exc.technical_details["connection_details"] == details


class TestDatabaseQueryError:
    """Tests for DatabaseQueryError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = DatabaseQueryError("Query failed")
        assert exc.message == "Query failed"
        assert exc.category == ErrorCategory.DATABASE
        assert exc.is_retriable is True

    def test_with_query(self) -> None:
        """Test exception with query."""
        exc = DatabaseQueryError("Query failed", query="SELECT * FROM entries")
        assert exc.technical_details["query"] == "SELECT * FROM entries"

    def test_query_truncation(self) -> None:
        """Test that long queries are truncated."""
        long_query = "SELECT " + "x" * 600
        exc = DatabaseQueryError("Query failed", query=long_query)
        assert len(exc.technical_details["query"]) == 500


class TestEmbeddingGenerationError:
    """Tests for EmbeddingGenerationError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = EmbeddingGenerationError("Embedding failed", model_name="nomic-embed-text")
        assert exc.message == "Embedding failed"
        assert exc.category == ErrorCategory.EMBEDDING
        assert exc.is_retriable is True
        assert exc.model_name == "nomic-embed-text"

    def test_with_input_text(self) -> None:
        """Test exception with input text."""
        exc = EmbeddingGenerationError(
            "Embedding failed",
            model_name="nomic-embed-text",
            input_text="Test input",
        )
        assert exc.input_text == "Test input"
        assert exc.technical_details["input_text"] == "Test input"

    def test_input_text_truncation(self) -> None:
        """Test that long input text is truncated."""
        long_text = "x" * 200
        exc = EmbeddingGenerationError(
            "Embedding failed",
            model_name="nomic-embed-text",
            input_text=long_text,
        )
        assert len(exc.input_text) == 100
        assert len(exc.technical_details["input_text"]) == 100


class TestSearchExecutionError:
    """Tests for SearchExecutionError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = SearchExecutionError(
            "Search failed",
            search_mode="semantic",
            query="test query",
        )
        assert exc.message == "Search failed"
        assert exc.category == ErrorCategory.SEARCH
        assert exc.is_retriable is False
        assert exc.search_mode == "semantic"
        assert exc.query == "test query"

    def test_query_truncation(self) -> None:
        """Test that long queries are truncated."""
        long_query = "x" * 300
        exc = SearchExecutionError(
            "Search failed",
            search_mode="keyword",
            query=long_query,
        )
        assert len(exc.technical_details["query"]) == 200


class TestIngestionError:
    """Tests for IngestionError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = IngestionError("Ingestion failed", source_system="ALS eLog")
        assert exc.message == "Ingestion failed"
        assert exc.category == ErrorCategory.INGESTION
        assert exc.is_retriable is False
        assert exc.source_system == "ALS eLog"
        assert exc.entries_affected == 0

    def test_with_entries_affected(self) -> None:
        """Test exception with entries affected count."""
        exc = IngestionError(
            "Ingestion failed",
            source_system="ALS eLog",
            entries_affected=42,
        )
        assert exc.entries_affected == 42
        assert exc.technical_details["entries_affected"] == 42


class TestAdapterNotFoundError:
    """Tests for AdapterNotFoundError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = AdapterNotFoundError("Adapter not found", adapter_name="unknown")
        assert exc.message == "Adapter not found"
        assert exc.category == ErrorCategory.INGESTION
        assert exc.is_retriable is False
        assert exc.adapter_name == "unknown"
        assert exc.available_adapters == []

    def test_with_available_adapters(self) -> None:
        """Test exception with available adapters list."""
        adapters = ["als", "jlab", "generic"]
        exc = AdapterNotFoundError(
            "Adapter not found",
            adapter_name="unknown",
            available_adapters=adapters,
        )
        assert exc.available_adapters == adapters
        assert exc.technical_details["available_adapters"] == adapters


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = ConfigurationError("Invalid config", config_key="database.uri")
        assert exc.message == "Invalid config"
        assert exc.category == ErrorCategory.CONFIGURATION
        assert exc.is_retriable is False
        assert exc.config_key == "database.uri"


class TestModuleNotEnabledError:
    """Tests for ModuleNotEnabledError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = ModuleNotEnabledError("Module not enabled", module_name="semantic")
        assert exc.message == "Module not enabled"
        assert exc.category == ErrorCategory.CONFIGURATION
        assert exc.is_retriable is False
        assert exc.module_name == "semantic"


class TestSearchTimeoutError:
    """Tests for SearchTimeoutError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = SearchTimeoutError(
            "Search timed out",
            timeout_seconds=120,
            operation="semantic_search",
        )
        assert exc.message == "Search timed out"
        assert exc.category == ErrorCategory.TIMEOUT
        assert exc.is_retriable is False
        assert exc.timeout_seconds == 120
        assert exc.operation == "semantic_search"


class TestExceptionHierarchy:
    """Tests for exception hierarchy relationships."""

    def test_all_exceptions_inherit_from_ariel_exception(self) -> None:
        """Test that all exceptions inherit from ARIELException."""
        exceptions = [
            DatabaseConnectionError("test"),
            DatabaseQueryError("test"),
            EmbeddingGenerationError("test", model_name="test"),
            SearchExecutionError("test", search_mode="keyword", query="test"),
            IngestionError("test", source_system="test"),
            AdapterNotFoundError("test", adapter_name="test"),
            ConfigurationError("test", config_key="test"),
            ModuleNotEnabledError("test", module_name="test"),
            SearchTimeoutError("test", timeout_seconds=60, operation="test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, ARIELException)
            assert isinstance(exc, Exception)

    def test_exception_can_be_raised_and_caught(self) -> None:
        """Test that exceptions can be raised and caught."""
        with pytest.raises(ARIELException) as exc_info:
            raise DatabaseConnectionError("Test error")
        assert exc_info.value.message == "Test error"

    def test_specific_exception_catch(self) -> None:
        """Test catching specific exception types."""
        with pytest.raises(DatabaseConnectionError):
            raise DatabaseConnectionError("Connection failed")
