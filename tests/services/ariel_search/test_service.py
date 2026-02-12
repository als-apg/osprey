"""Tests for ARIEL search service and agent.

Tests for service, agent executor, and routing functionality.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.agent.executor import (
    AGENT_SYSTEM_PROMPT,
    AgentExecutor,
    AgentResult,
)
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.models import ARIELSearchResult, SearchMode
from osprey.services.ariel_search.search.keyword import (
    KeywordSearchInput,
    format_keyword_result,
)
from osprey.services.ariel_search.search.semantic import (
    SemanticSearchInput,
    format_semantic_result,
)
from osprey.services.ariel_search.service import ARIELSearchService


class TestAgentSystemPrompt:
    """Tests for agent system prompt."""

    def test_agent_prompt_not_empty(self):
        """Agent system prompt is not empty."""
        assert AGENT_SYSTEM_PROMPT
        assert len(AGENT_SYSTEM_PROMPT) > 100

    def test_agent_prompt_describes_role(self):
        """Agent prompt describes role and guidelines without hardcoding tool names."""
        assert "ARIEL" in AGENT_SYSTEM_PROMPT
        assert "logbook" in AGENT_SYSTEM_PROMPT.lower()
        assert "search tools" in AGENT_SYSTEM_PROMPT.lower()
        # Tool names should NOT be hardcoded in the system prompt
        assert "keyword_search" not in AGENT_SYSTEM_PROMPT
        assert "semantic_search" not in AGENT_SYSTEM_PROMPT


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
        assert input_schema.similarity_threshold == 0.7

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

    def test_agent_executor_exported(self):
        """AgentExecutor is exported from package."""
        from osprey.services.ariel_search import AgentExecutor

        assert AgentExecutor is not None

    def test_agent_result_exported(self):
        """AgentResult is exported from package."""
        from osprey.services.ariel_search import AgentResult

        assert AgentResult is not None


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
    async def test_search_routes_to_keyword(self):
        """Search routes to _run_keyword for KEYWORD mode."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_result = ARIELSearchResult(
            entries=(),
            answer=None,
            sources=(),
            search_modes_used=(SearchMode.KEYWORD,),
            reasoning="Keyword search: 0 results",
        )
        service._run_keyword = AsyncMock(return_value=mock_result)

        result = await service.search("test query", mode=SearchMode.KEYWORD)

        service._run_keyword.assert_called_once()
        assert result.search_modes_used == (SearchMode.KEYWORD,)

    @pytest.mark.asyncio
    async def test_search_routes_to_semantic(self):
        """Search routes to _run_semantic for SEMANTIC mode."""
        service = self._create_mock_service(
            search_modules={"semantic": {"enabled": True, "model": "test"}}
        )

        mock_result = ARIELSearchResult(
            entries=(),
            answer=None,
            sources=(),
            search_modes_used=(SearchMode.SEMANTIC,),
            reasoning="Semantic search: 0 results",
        )
        service._run_semantic = AsyncMock(return_value=mock_result)

        result = await service.search("test query", mode=SearchMode.SEMANTIC)

        service._run_semantic.assert_called_once()
        assert result.search_modes_used == (SearchMode.SEMANTIC,)

    @pytest.mark.asyncio
    async def test_search_routes_to_rag(self):
        """Search routes to _run_rag for RAG mode."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_result = ARIELSearchResult(
            entries=(),
            answer="RAG answer",
            sources=(),
            search_modes_used=(SearchMode.RAG,),
            reasoning="RAG pipeline",
        )
        service._run_rag = AsyncMock(return_value=mock_result)

        result = await service.search("test query", mode=SearchMode.RAG)

        service._run_rag.assert_called_once()
        assert result.answer == "RAG answer"

    @pytest.mark.asyncio
    async def test_search_routes_to_agent(self):
        """Search routes to _run_agent for AGENT mode."""
        service = self._create_mock_service(
            search_modules={"keyword": {"enabled": True}, "semantic": {"enabled": True}}
        )

        mock_result = ARIELSearchResult(
            entries=(),
            answer="Agent answer",
            sources=(),
            search_modes_used=(SearchMode.KEYWORD, SearchMode.SEMANTIC),
            reasoning="Agent execution",
        )
        service._run_agent = AsyncMock(return_value=mock_result)

        result = await service.search("test query", mode=SearchMode.AGENT)

        service._run_agent.assert_called_once()
        assert result.answer == "Agent answer"

    @pytest.mark.asyncio
    async def test_search_defaults_to_rag_mode(self):
        """Search defaults to RAG mode when no mode specified."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_result = ARIELSearchResult(
            entries=(),
            answer="RAG answer",
            sources=(),
            search_modes_used=(SearchMode.RAG,),
            reasoning="RAG pipeline",
        )
        service._run_rag = AsyncMock(return_value=mock_result)

        await service.search("test query")

        service._run_rag.assert_called_once()

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

        # Use the real _run_keyword but with mocked keyword_search
        from unittest.mock import patch

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            side_effect=fake_keyword_search,
        ):
            result = await service.search("beam", mode=SearchMode.KEYWORD)

        assert len(result.entries) == 1
        assert result.entries[0]["_highlights"] == ["<b>beam</b> alignment"]

    @pytest.mark.asyncio
    async def test_keyword_mode_raises_when_disabled(self):
        """KEYWORD mode raises ConfigurationError when module disabled."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        service = self._create_mock_service(search_modules={"keyword": {"enabled": False}})

        with pytest.raises(ConfigurationError):
            await service.search("test query", mode=SearchMode.KEYWORD)

    @pytest.mark.asyncio
    async def test_semantic_mode_raises_when_disabled(self):
        """SEMANTIC mode raises ConfigurationError when module disabled."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        service = self._create_mock_service(search_modules={"semantic": {"enabled": False}})

        with pytest.raises(ConfigurationError):
            await service.search("test query", mode=SearchMode.SEMANTIC)


class TestAgentExecutor:
    """Tests for AgentExecutor class."""

    def _create_mock_executor(self) -> AgentExecutor:
        """Create a mock executor for testing."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "semantic": {"enabled": True, "model": "test-model"},
                },
            }
        )
        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()

        return AgentExecutor(
            repository=mock_repository,
            config=config,
            embedder_loader=mock_embedder_loader,
        )

    def test_executor_creates_keyword_tool(self):
        """Executor creates keyword_search tool when enabled."""
        executor = self._create_mock_executor()
        tools, _descriptors = executor._create_tools()

        tool_names = [t.name for t in tools]
        assert "keyword_search" in tool_names

    def test_executor_creates_semantic_tool(self):
        """Executor creates semantic_search tool when enabled."""
        executor = self._create_mock_executor()
        tools, _descriptors = executor._create_tools()

        tool_names = [t.name for t in tools]
        assert "semantic_search" in tool_names

    def test_executor_parse_agent_result_extracts_answer(self):
        """_parse_agent_result extracts answer from messages."""
        executor = self._create_mock_executor()
        _tools, descriptors = executor._create_tools()

        mock_ai_message = MagicMock()
        mock_ai_message.content = "This is the answer from the agent."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_ai_message]}

        result = executor._parse_agent_result(result_dict, descriptors)

        assert result.answer == "This is the answer from the agent."

    def test_executor_parse_agent_result_extracts_citations(self):
        """_parse_agent_result extracts citations from answer."""
        executor = self._create_mock_executor()
        _tools, descriptors = executor._create_tools()

        mock_ai_message = MagicMock()
        mock_ai_message.content = "Found in [entry-001] and [entry-002] and [#003]."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_ai_message]}

        result = executor._parse_agent_result(result_dict, descriptors)

        assert "001" in result.sources
        assert "002" in result.sources
        assert "003" in result.sources

    def test_executor_parse_agent_result_identifies_search_modes(self):
        """_parse_agent_result identifies which search modes were used."""
        executor = self._create_mock_executor()
        _tools, descriptors = executor._create_tools()

        mock_tool_message = MagicMock()
        mock_tool_message.tool_calls = [
            {"name": "keyword_search"},
            {"name": "semantic_search"},
        ]
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Answer"
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_tool_message, mock_ai_message]}

        result = executor._parse_agent_result(result_dict, descriptors)

        assert SearchMode.KEYWORD in result.search_modes_used
        assert SearchMode.SEMANTIC in result.search_modes_used


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


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_agent_result_immutable(self):
        """AgentResult is frozen (immutable)."""
        result = AgentResult(
            answer="Test answer",
            entries=(),
            sources=("001", "002"),
            search_modes_used=(SearchMode.KEYWORD,),
            reasoning="Test reasoning",
        )

        assert result.answer == "Test answer"
        assert result.sources == ("001", "002")
        assert result.search_modes_used == (SearchMode.KEYWORD,)

    def test_agent_result_defaults(self):
        """AgentResult has correct defaults."""
        result = AgentResult()

        assert result.answer is None
        assert result.entries == ()
        assert result.sources == ()
        assert result.search_modes_used == ()
        assert result.reasoning == ""


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
            search_modes_used=(SearchMode.KEYWORD, SearchMode.SEMANTIC),
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


class TestLLMConfiguration:
    """Tests for LLM configuration."""

    def test_model_id_default(self):
        """Default model_id is gpt-4o-mini."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
            }
        )
        assert config.reasoning.model_id == "gpt-4o-mini"

    def test_provider_default(self):
        """Default provider is openai."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
            }
        )
        assert config.reasoning.provider == "openai"

    def test_model_id_configurable(self):
        """model_id can be configured."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "reasoning": {"model_id": "gpt-4-turbo"},
            }
        )
        assert config.reasoning.model_id == "gpt-4-turbo"

    def test_provider_configurable(self):
        """provider can be configured."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "reasoning": {"provider": "anthropic"},
            }
        )
        assert config.reasoning.provider == "anthropic"

    def test_reasoning_config_fields(self):
        """Reasoning config fields are properly parsed."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "reasoning": {"provider": "anthropic", "model_id": "claude-haiku"},
            }
        )
        assert config.reasoning.provider == "anthropic"
        assert config.reasoning.model_id == "claude-haiku"


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
        """Advanced params are forwarded to _run_keyword."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_result = ARIELSearchResult(
            entries=(),
            search_modes_used=(SearchMode.KEYWORD,),
            reasoning="Keyword search: 0 results",
        )
        service._run_keyword = AsyncMock(return_value=mock_result)

        await service.search(
            "test",
            mode=SearchMode.KEYWORD,
            advanced_params={"include_highlights": False, "fuzzy_fallback": False},
        )

        # Verify the request passed to _run_keyword has the advanced_params
        call_args = service._run_keyword.call_args[0]
        request = call_args[0]
        assert request.advanced_params == {"include_highlights": False, "fuzzy_fallback": False}

    @pytest.mark.asyncio
    async def test_advanced_params_reach_rag(self):
        """Advanced params are forwarded to _run_rag."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_result = ARIELSearchResult(
            entries=(),
            answer="test",
            search_modes_used=(SearchMode.RAG,),
            reasoning="RAG pipeline",
        )
        service._run_rag = AsyncMock(return_value=mock_result)

        await service.search(
            "test",
            mode=SearchMode.RAG,
            advanced_params={"temperature": 0.5, "max_context_chars": 8000},
        )

        call_args = service._run_rag.call_args[0]
        request = call_args[0]
        assert request.advanced_params["temperature"] == 0.5
        assert request.advanced_params["max_context_chars"] == 8000

    @pytest.mark.asyncio
    async def test_advanced_params_default_empty(self):
        """advanced_params defaults to empty dict when not provided."""
        service = self._create_mock_service(search_modules={"keyword": {"enabled": True}})

        mock_result = ARIELSearchResult(
            entries=(),
            search_modes_used=(SearchMode.RAG,),
            reasoning="RAG pipeline",
        )
        service._run_rag = AsyncMock(return_value=mock_result)

        await service.search("test")

        call_args = service._run_rag.call_args[0]
        request = call_args[0]
        assert request.advanced_params == {}


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


class TestToolInstanceTypes:
    """Quality assertions for tool instance types."""

    def test_executor_creates_structured_tools(self):
        """Verify executor creates StructuredTool instances."""
        from langchain_core.tools import StructuredTool

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "semantic": {"enabled": True, "model": "test-model"},
                },
            }
        )

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()

        executor = AgentExecutor(
            repository=mock_repository,
            config=config,
            embedder_loader=mock_embedder_loader,
        )

        tools, _descriptors = executor._create_tools()

        # All tools should be StructuredTool instances
        assert len(tools) == 2
        for tool in tools:
            assert isinstance(tool, StructuredTool), f"Tool {tool.name} is not a StructuredTool"

    def test_tools_have_required_attributes(self):
        """All tools have required StructuredTool attributes."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()

        executor = AgentExecutor(
            repository=mock_repository,
            config=config,
            embedder_loader=mock_embedder_loader,
        )

        tools, _descriptors = executor._create_tools()

        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "args_schema")
            assert hasattr(tool, "coroutine") or hasattr(tool, "func")


class TestToolInputSchemaDefaults:
    """Tests for tool input schema default values."""

    def test_keyword_input_max_results_default(self):
        """KeywordSearchInput has max_results default of 10."""
        input_schema = KeywordSearchInput(query="test")
        assert input_schema.max_results == 10

    def test_semantic_input_similarity_default(self):
        """SemanticSearchInput has similarity_threshold default of 0.7."""
        input_schema = SemanticSearchInput(query="test")
        assert input_schema.similarity_threshold == 0.7


class TestCitationInstruction:
    """Quality assertions for citation instructions."""

    def test_agent_prompt_contains_citation_instruction(self):
        """Verify agent prompt contains citation instruction."""
        # System prompt should instruct about citations
        assert any(
            word in AGENT_SYSTEM_PROMPT.lower()
            for word in ["cite", "citation", "reference", "source"]
        )
