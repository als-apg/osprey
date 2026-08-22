"""Tests for ARIEL search service.

Tests for service routing and formatting functionality.
"""

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.models import ARIELSearchResult
from osprey.services.ariel_search.search.keyword import (
    KeywordSearchInput,
    format_keyword_result,
)
from osprey.services.ariel_search.search.semantic import (
    SemanticSearchInput,
    format_semantic_result,
)
from osprey.services.ariel_search.service import ARIELSearchService


class TestToolInputSchemas:
    """Tests for Pydantic input schemas."""

    def test_keyword_search_input_defaults(self):
        """KeywordSearchInput has correct defaults."""
        input_schema = KeywordSearchInput(query="test query")
        assert input_schema.query == "test query"
        assert input_schema.max_results == 10
        assert input_schema.start_date is None
        assert input_schema.end_date is None

    def test_keyword_search_input_validation(self):
        """KeywordSearchInput validates max_results."""
        # Valid range
        input_schema = KeywordSearchInput(query="test", max_results=25)
        assert input_schema.max_results == 25

        # Below minimum
        with pytest.raises(ValueError):
            KeywordSearchInput(query="test", max_results=0)

        # Above maximum
        with pytest.raises(ValueError):
            KeywordSearchInput(query="test", max_results=100)

    def test_semantic_search_input_defaults(self):
        """SemanticSearchInput has correct defaults."""
        input_schema = SemanticSearchInput(query="conceptual query")
        assert input_schema.query == "conceptual query"
        assert input_schema.max_results == 10
        assert input_schema.similarity_threshold == 0.5

    def test_semantic_search_input_validation(self):
        """SemanticSearchInput validates similarity_threshold."""
        # Valid range
        input_schema = SemanticSearchInput(query="test", similarity_threshold=0.5)
        assert input_schema.similarity_threshold == 0.5

        # Below minimum
        with pytest.raises(ValueError):
            SemanticSearchInput(query="test", similarity_threshold=-0.1)

        # Above maximum
        with pytest.raises(ValueError):
            SemanticSearchInput(query="test", similarity_threshold=1.5)


class TestFormatKeywordResult:
    """Tests for keyword result formatting."""

    def test_format_basic_result(self):
        """Formats basic keyword search result."""
        entry = {
            "entry_id": "entry-001",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            "author": "jsmith",
            "raw_text": "Beam current stabilized at 500mA.",
            "attachments": [],
            "metadata": {"title": "Beam Update"},
        }

        result = format_keyword_result(entry, 0.85, ["<mark>Beam</mark> current"])

        assert result["entry_id"] == "entry-001"
        assert result["author"] == "jsmith"
        assert result["title"] == "Beam Update"
        assert result["score"] == 0.85
        assert result["highlights"] == ["<mark>Beam</mark> current"]

    def test_truncates_long_text(self):
        """Truncates text longer than 500 chars."""
        long_text = "x" * 1000
        entry = {
            "entry_id": "entry-002",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            "author": "jsmith",
            "raw_text": long_text,
            "attachments": [],
            "metadata": {},
        }

        result = format_keyword_result(entry, 0.5, [])

        assert len(result["text"]) == 500


class TestFormatSemanticResult:
    """Tests for semantic result formatting."""

    def test_format_basic_result(self):
        """Formats basic semantic search result."""
        entry = {
            "entry_id": "entry-003",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            "author": "jdoe",
            "raw_text": "RF cavity tuning completed.",
            "attachments": [],
            "metadata": {"title": "RF Update"},
        }

        result = format_semantic_result(entry, 0.92)

        assert result["entry_id"] == "entry-003"
        assert result["author"] == "jdoe"
        assert result["title"] == "RF Update"
        assert result["similarity"] == 0.92


class TestServiceExports:
    """Tests for service module exports."""

    def test_ariel_search_service_exported(self):
        """ARIELSearchService is exported from package."""
        from osprey.services.ariel_search import ARIELSearchService

        assert ARIELSearchService is not None

    def test_create_ariel_service_exported(self):
        """create_ariel_service is exported from package."""
        from osprey.services.ariel_search import create_ariel_service

        assert callable(create_ariel_service)


class TestARIELSearchService:
    """Tests for ARIELSearchService class."""

    def _create_mock_service(self) -> ARIELSearchService:
        """Create a mock service for testing."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
            }
        )
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.health_check = AsyncMock(return_value=(True, "OK"))
        mock_repository.validate_search_model_table = AsyncMock()

        return ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

    def test_initialization(self):
        """Service initializes with correct attributes."""
        service = self._create_mock_service()
        assert service.config is not None
        assert service.pool is not None
        assert service.repository is not None
        assert service._embedder is None
        assert service._validated_search_model is False

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """Context manager returns self on enter."""
        service = self._create_mock_service()
        async with service as s:
            assert s is service

    @pytest.mark.asyncio
    async def test_context_manager_exit_closes_pool(self):
        """Context manager closes pool on exit."""
        service = self._create_mock_service()
        async with service:
            pass
        service.pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Health check returns healthy when database is healthy."""
        service = self._create_mock_service()
        service.repository.health_check = AsyncMock(return_value=(True, "Connected"))

        healthy, message = await service.health_check()

        assert healthy is True
        assert "ARIEL service healthy" in message

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Health check returns unhealthy when database fails."""
        service = self._create_mock_service()
        service.repository.health_check = AsyncMock(return_value=(False, "Connection failed"))

        healthy, message = await service.health_check()

        assert healthy is False
        assert "Database" in message


class TestServiceRouting:
    """Tests for service mode routing."""

    def _create_mock_service(
        self,
        search_modules: dict | None = None,
        default_search_mode: str | None = None,
    ) -> ARIELSearchService:
        """Create a mock service for testing."""
        config_dict: dict = {
            "database": {"uri": "postgresql://localhost:5432/test"},
        }
        if search_modules:
            config_dict["search_modules"] = search_modules
        if default_search_mode:
            config_dict["default_search_mode"] = default_search_mode

        config = ARIELConfig.from_dict(config_dict)
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.health_check = AsyncMock(return_value=(True, "OK"))
        mock_repository.validate_search_model_table = AsyncMock()

        return ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

    @pytest.mark.asyncio
    async def test_search_routes_to_keyword(self):
        """Search dispatches to the registered keyword module for "keyword"."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new=AsyncMock(return_value=[]),
        ) as keyword_search:
            result = await service.search("test query", mode="keyword")

        keyword_search.assert_called_once()
        assert result.search_modes_used == ("keyword",)

    @pytest.mark.asyncio
    async def test_search_routes_to_semantic(self, fake_embedding_provider):
        """Search dispatches to the registered semantic module for "semantic"."""
        service = self._create_mock_service(
            search_modules={"semantic": {"enabled": True, "model": "test"}}
        )
        # Pre-set so _get_embedder() never reaches the provider registry.
        service._embedder = fake_embedding_provider

        with patch(
            "osprey.services.ariel_search.search.semantic.semantic_search",
            new=AsyncMock(return_value=[]),
        ) as semantic_search:
            result = await service.search("test query", mode="semantic")

        semantic_search.assert_called_once()
        assert result.search_modes_used == ("semantic",)

    @pytest.mark.asyncio
    async def test_search_defaults_to_keyword_mode(self):
        """Search defaults to the keyword module when no mode is specified."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new=AsyncMock(return_value=[]),
        ) as keyword_search:
            result = await service.search("test query")

        keyword_search.assert_called_once()
        assert result.search_modes_used == ("keyword",)

    @pytest.mark.asyncio
    async def test_search_defaults_to_hybrid_when_enabled(self):
        """With no configured default, an enabled hybrid module answers."""
        service = self._create_mock_service(
            search_modules={"keyword": {"enabled": True}, "hybrid": {"enabled": True}}
        )

        with patch(
            "osprey.services.ariel_search.search.qmd.hybrid_search",
            new=AsyncMock(return_value=[]),
        ) as hybrid_search:
            result = await service.search("test query")

        hybrid_search.assert_called_once()
        assert result.search_modes_used == ("hybrid",)

    @pytest.mark.asyncio
    async def test_configured_default_search_mode_wins(self):
        """An explicit default_search_mode outranks the implicit preference."""
        service = self._create_mock_service(
            search_modules={"keyword": {"enabled": True}, "hybrid": {"enabled": True}},
            default_search_mode="keyword",
        )

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new=AsyncMock(return_value=[]),
        ) as keyword_search:
            result = await service.search("test query")

        keyword_search.assert_called_once()
        assert result.search_modes_used == ("keyword",)

    @pytest.mark.asyncio
    async def test_explicit_mode_outranks_the_default(self):
        """Naming a mode still wins over the deployment's default."""
        service = self._create_mock_service(
            search_modules={"keyword": {"enabled": True}, "hybrid": {"enabled": True}},
            default_search_mode="hybrid",
        )

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new=AsyncMock(return_value=[]),
        ) as keyword_search:
            result = await service.search("test query", mode="keyword")

        keyword_search.assert_called_once()
        assert result.search_modes_used == ("keyword",)

    @pytest.mark.asyncio
    async def test_keyword_preserves_highlights(self):
        """Keyword search preserves highlights in returned entries."""

        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_entry = {
            "entry_id": "entry-hl-001",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            "author": "jsmith",
            "raw_text": "Beam alignment completed successfully.",
            "attachments": [],
            "metadata": {},
            "created_at": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            "updated_at": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        }
        mock_highlights = ["<b>beam</b> alignment"]

        service.repository.keyword_search = AsyncMock(
            return_value=[(mock_entry, 0.8, mock_highlights)]
        )

        # Patch keyword_search to call repository directly
        async def fake_keyword_search(query, repo, config, **kwargs):
            return await repo.keyword_search(query)

        # Dispatch runs the real keyword module with keyword_search stubbed.
        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            side_effect=fake_keyword_search,
        ):
            result = await service.search("beam", mode="keyword")

        assert len(result.entries) == 1
        assert result.entries[0]["_highlights"] == ["<b>beam</b> alignment"]

    @pytest.mark.asyncio
    async def test_keyword_mode_raises_when_disabled(self):
        """A disabled keyword module raises ConfigurationError."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        service = self._create_mock_service(search_modules={"keyword": {"enabled": False}})

        with pytest.raises(ConfigurationError):
            await service.search("test query", mode="keyword")

    @pytest.mark.asyncio
    async def test_semantic_mode_degrades_gracefully_when_disabled(self):
        """SEMANTIC mode degrades gracefully (no error) when module disabled.

        Per the ARIEL contract, semantic search "degrades gracefully to
        keyword-only" when pgvector/Ollama embeddings are unavailable. The
        service must return a non-error result that points the caller to
        keyword search, not raise (which surfaces as a hard MCP tool error).
        Regression test for #276.
        """
        from osprey.services.ariel_search.models import DiagnosticLevel

        service = self._create_mock_service(search_modules={"semantic": {"enabled": False}})

        result = await service.search("test query", mode="semantic")

        # No exception, no entries, and the caller is steered to keyword search.
        assert result.entries == ()
        assert result.search_modes_used == ()
        assert "keyword" in result.reasoning.lower()
        # Surfaced as an informational (non-error) diagnostic.
        assert result.diagnostics
        assert all(d.level is DiagnosticLevel.INFO for d in result.diagnostics)

    @pytest.mark.asyncio
    async def test_unroutable_mode_raises_configuration_error(self):
        """A mode naming no registered module raises, naming the alternatives.

        Dispatch is registry-driven, so an unregistered name has nothing to
        route to and must not silently fall back to keyword search. The error
        names the requested mode and lists the modes that are actually enabled.
        ConfigurationError is an ARIELException, so it propagates unwrapped.
        """
        from osprey.services.ariel_search.exceptions import ConfigurationError

        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        with pytest.raises(ConfigurationError) as exc_info:
            await service.search("test query", mode="not_a_module")

        message = str(exc_info.value)
        assert "Unknown search mode 'not_a_module'" in message
        assert "keyword" in message.split("Available modes:")[1]
        assert exc_info.value.config_key == "modes"

    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped_in_search_execution_error(self):
        """A non-ARIEL exception is wrapped with the mode and query that failed."""
        from osprey.services.ariel_search.exceptions import SearchExecutionError

        service = self._create_mock_service(
            search_modules={
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            }
        )
        # Fails inside the try block, before routing picks an arm.
        service.repository.validate_search_model_table = AsyncMock(
            side_effect=RuntimeError("pool exhausted")
        )

        with pytest.raises(SearchExecutionError) as exc_info:
            await service.search("beam current", mode="keyword")

        error = exc_info.value
        assert error.search_mode == "keyword"
        assert error.query == "beam current"
        assert "pool exhausted" in str(error)
        assert isinstance(error.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_semantic_results_projected_to_entries_and_sources(self, fake_embedding_provider):
        """Semantic hits become entries carrying _score, plus an entry_id source tuple."""
        service = self._create_mock_service(
            search_modules={"semantic": {"enabled": True, "model": "test-model"}}
        )
        # Pre-set so _get_embedder() never reaches the provider registry.
        service._embedder = fake_embedding_provider

        seen_embedders = []

        async def fake_semantic_search(query, repository, config, embedder, **kwargs):
            seen_embedders.append(embedder)
            return [
                ({"entry_id": "entry-sem-001", "raw_text": "RF cavity trip"}, 0.91),
                ({"entry_id": "entry-sem-002", "raw_text": "Beam dump at 08:12"}, 0.77),
            ]

        with patch(
            "osprey.services.ariel_search.search.semantic.semantic_search",
            side_effect=fake_semantic_search,
        ):
            result = await service.search("cavity", mode="semantic")

        assert seen_embedders == [fake_embedding_provider]
        assert result.search_modes_used == ("semantic",)
        assert result.reasoning == "Semantic search: 2 results"
        assert result.sources == ("entry-sem-001", "entry-sem-002")
        assert [entry["_score"] for entry in result.entries] == [0.91, 0.77]
        assert result.entries[0]["raw_text"] == "RF cavity trip"


class TestCreateArielService:
    """Tests for create_ariel_service factory function."""

    @pytest.mark.asyncio
    async def test_factory_function_is_async(self):
        """Factory function is an async function."""
        import asyncio

        from osprey.services.ariel_search.service import create_ariel_service

        assert asyncio.iscoroutinefunction(create_ariel_service)


class TestFormatResultsNullHandling:
    """Tests for result formatting with null values."""

    def test_format_keyword_null_timestamp(self):
        """format_keyword_result handles null timestamp."""
        entry = {
            "entry_id": "entry-null-ts",
            "source_system": "test",
            "timestamp": None,
            "author": "jsmith",
            "raw_text": "No timestamp entry.",
            "attachments": [],
            "metadata": {},
        }

        result = format_keyword_result(entry, 0.5, [])

        assert result["entry_id"] == "entry-null-ts"
        assert result["timestamp"] is None

    def test_format_semantic_null_timestamp(self):
        """format_semantic_result handles null timestamp."""
        entry = {
            "entry_id": "entry-null-ts",
            "source_system": "test",
            "timestamp": None,
            "author": "jsmith",
            "raw_text": "No timestamp entry.",
            "attachments": [],
            "metadata": {},
        }

        result = format_semantic_result(entry, 0.5)

        assert result["entry_id"] == "entry-null-ts"
        assert result["timestamp"] is None

    def test_format_keyword_missing_metadata(self):
        """format_keyword_result handles missing metadata."""
        entry = {
            "entry_id": "entry-no-meta",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=UTC),
            "author": "jsmith",
            "raw_text": "Entry without metadata.",
            "attachments": [],
        }

        result = format_keyword_result(entry, 0.5, [])

        assert result["title"] is None

    def test_format_semantic_missing_metadata(self):
        """format_semantic_result handles missing metadata."""
        entry = {
            "entry_id": "entry-no-meta",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=UTC),
            "author": "jsmith",
            "raw_text": "Entry without metadata.",
            "attachments": [],
        }

        result = format_semantic_result(entry, 0.5)

        assert result["title"] is None


class TestServiceValidateSearchModel:
    """Tests for _validate_search_model method."""

    @pytest.mark.asyncio
    async def test_validate_search_model_called_once(self):
        """_validate_search_model only validates once."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {"semantic": {"enabled": True, "model": "test-model"}},
            }
        )
        mock_pool = MagicMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # First call - should validate
        await service._validate_search_model()
        mock_repository.validate_search_model_table.assert_called_once_with("test-model")

        # Second call - should not validate again
        await service._validate_search_model()
        mock_repository.validate_search_model_table.assert_called_once()  # Still just once

    @pytest.mark.asyncio
    async def test_validate_search_model_no_model_configured(self):
        """_validate_search_model handles no model configured."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
            }
        )
        mock_pool = MagicMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        await service._validate_search_model()
        mock_repository.validate_search_model_table.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_search_model_disables_semantic_when_table_missing(self, caplog):
        """A missing embedding table disables semantic search instead of raising.

        The config is built inside the test because this branch mutates
        ``search_modules["semantic"].enabled`` -- a shared config would leak the
        disabled flag into every test that ran afterwards.
        """
        from osprey.services.ariel_search.exceptions import ConfigurationError

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {"semantic": {"enabled": True, "model": "test-model"}},
            }
        )
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock(
            side_effect=ConfigurationError(
                "embedding table embeddings_test_model not found",
                config_key="search_modules.semantic.model",
            )
        )

        service = ARIELSearchService(
            config=config,
            pool=MagicMock(),
            repository=mock_repository,
        )

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await service._validate_search_model()

        assert config.search_modules["semantic"].enabled is False
        assert config.is_search_module_enabled("semantic") is False
        assert "Semantic search disabled" in caplog.text
        assert "embeddings_test_model" in caplog.text
        assert "quickstart" in caplog.text
        # Still marked validated, so the failure is not retried on every search.
        assert service._validated_search_model is True


class TestServiceGetStatus:
    """Tests for ARIELSearchService.get_status() method."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal ARIEL config."""
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://user:pass@localhost:5432/test"},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "semantic": {"enabled": True},
                },
                "enhancement_modules": {
                    "text_embedding": {"enabled": True},
                },
            }
        )

    def test_get_status_masks_uri(self, minimal_config):
        """get_status masks database credentials in URI."""
        service = ARIELSearchService(
            config=minimal_config,
            pool=MagicMock(),
            repository=MagicMock(),
        )
        masked = service._mask_database_uri("postgresql://user:password@host:5432/db")
        assert "***" in masked
        assert "password" not in masked
        assert "@host:5432/db" in masked

    def test_get_status_masks_uri_no_password(self, minimal_config):
        """get_status handles URI without credentials."""
        service = ARIELSearchService(
            config=minimal_config,
            pool=MagicMock(),
            repository=MagicMock(),
        )
        masked = service._mask_database_uri("postgresql://localhost:5432/db")
        # No @ in original, so no masking
        assert masked == "postgresql://localhost:5432/db"

    @pytest.mark.asyncio
    async def test_get_status_returns_status_result(self, minimal_config):
        """get_status returns ARIELStatusResult dataclass with correct fields."""
        from osprey.services.ariel_search.models import ARIELStatusResult

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()

        # Mock fetchone to return appropriate values for each query
        mock_cursor.fetchone = AsyncMock(return_value=(42,))
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_pool.connection = MagicMock(return_value=mock_conn)

        mock_repository = MagicMock()
        mock_repository.get_embedding_tables = AsyncMock(return_value=[])

        service = ARIELSearchService(
            config=minimal_config,
            pool=mock_pool,
            repository=mock_repository,
        )

        result = await service.get_status()

        # Verify result is ARIELStatusResult with expected structure
        assert isinstance(result, ARIELStatusResult)
        assert result.database_connected is True  # Connection succeeded
        assert "***" in result.database_uri  # Credentials masked
        assert result.entry_count is not None  # Entry count retrieved
        assert result.enabled_search_modules == ["keyword", "semantic"]
        assert result.enabled_enhancement_modules == ["text_embedding"]
        assert isinstance(result.errors, list)

    @pytest.mark.asyncio
    async def test_get_status_reports_unreachable_database(self, minimal_config, fake_pool_factory):
        """A pool that cannot hand out a connection becomes an error entry, not a raise."""
        failing_pool = fake_pool_factory(error=RuntimeError("connection refused"))

        mock_repository = MagicMock()
        mock_repository.get_embedding_tables = AsyncMock(return_value=[])

        service = ARIELSearchService(
            config=minimal_config,
            pool=failing_pool,
            repository=mock_repository,
        )

        result = await service.get_status()

        assert result.errors == ["Database error: connection refused"]
        assert result.healthy is False
        assert result.database_connected is False
        assert result.entry_count is None
        assert result.embedding_tables == []
        assert result.last_ingestion is None
        # Config-derived fields still populate despite the database being down.
        assert "***" in result.database_uri
        assert result.enabled_search_modules == ["keyword", "semantic"]


class TestARIELSearchResultModel:
    """Tests for ARIELSearchResult model."""

    def test_result_entries_immutable(self):
        """ARIELSearchResult entries are immutable."""
        result = ARIELSearchResult(
            entries=({"entry_id": "1"},),  # type: ignore[arg-type]
        )

        # entries is a tuple
        assert isinstance(result.entries, tuple)

    def test_result_search_modes_used_immutable(self):
        """ARIELSearchResult search_modes_used is immutable."""
        result = ARIELSearchResult(
            entries=(),
            search_modes_used=("keyword", "semantic"),
        )

        assert isinstance(result.search_modes_used, tuple)

    def test_result_default_values(self):
        """ARIELSearchResult has correct defaults."""
        result = ARIELSearchResult(
            entries=(),
        )

        assert result.answer is None
        assert result.sources == ()
        assert result.search_modes_used == ()
        assert result.reasoning == ""


class TestAdvancedParamsWiring:
    """Tests for advanced_params flowing through service.search()."""

    def _create_mock_service(self, search_modules: dict | None = None) -> ARIELSearchService:
        """Create a mock service for testing."""
        config_dict = {
            "database": {"uri": "postgresql://localhost:5432/test"},
        }
        if search_modules:
            config_dict["search_modules"] = search_modules

        config = ARIELConfig.from_dict(config_dict)
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.health_check = AsyncMock(return_value=(True, "OK"))
        mock_repository.validate_search_model_table = AsyncMock()

        return ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

    @pytest.mark.asyncio
    async def test_advanced_params_reach_keyword(self):
        """Advanced params are forwarded to the keyword module as keywords."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new=AsyncMock(return_value=[]),
        ) as keyword_search:
            await service.search(
                "test",
                mode="keyword",
                advanced_params={"include_highlights": False, "fuzzy_fallback": False},
            )

        kwargs = keyword_search.call_args.kwargs
        assert kwargs["include_highlights"] is False
        assert kwargs["fuzzy_fallback"] is False

    @pytest.mark.asyncio
    async def test_advanced_params_default_empty(self):
        """With no advanced params, only the request's own fields are passed."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new=AsyncMock(return_value=[]),
        ) as keyword_search:
            await service.search("test")

        assert set(keyword_search.call_args.kwargs) == {
            "max_results",
            "start_date",
            "end_date",
            "parsed",
        }


class TestServiceState:
    """Tests for service internal state management."""

    def test_service_embedder_initially_none(self):
        """Service embedder is None on initialization."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
            }
        )
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        assert service._embedder is None


class TestToolInputSchemaDefaults:
    """Tests for tool input schema default values."""

    def test_keyword_input_max_results_default(self):
        """KeywordSearchInput has max_results default of 10."""
        input_schema = KeywordSearchInput(query="test")
        assert input_schema.max_results == 10

    def test_semantic_input_similarity_default(self):
        """SemanticSearchInput has similarity_threshold default of 0.5."""
        input_schema = SemanticSearchInput(query="test")
        assert input_schema.similarity_threshold == 0.5
