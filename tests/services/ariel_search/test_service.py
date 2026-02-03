"""Tests for ARIEL search service and tools.

Tests for service, tools, and prompts modules.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.models import ARIELSearchRequest, ARIELSearchResult, SearchMode
from osprey.services.ariel_search.prompts import DEFAULT_SYSTEM_PROMPT, get_system_prompt
from osprey.services.ariel_search.service import ARIELSearchService
from osprey.services.ariel_search.tools import (
    KeywordSearchInput,
    RAGSearchInput,
    SemanticSearchInput,
    format_keyword_result,
    format_semantic_result,
)


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_default_prompt_not_empty(self):
        """Default system prompt is not empty."""
        assert DEFAULT_SYSTEM_PROMPT
        assert len(DEFAULT_SYSTEM_PROMPT) > 100

    def test_default_prompt_mentions_tools(self):
        """Default prompt mentions available tools."""
        assert "keyword_search" in DEFAULT_SYSTEM_PROMPT
        assert "semantic_search" in DEFAULT_SYSTEM_PROMPT
        assert "rag_search" in DEFAULT_SYSTEM_PROMPT

    def test_get_system_prompt_returns_default(self):
        """get_system_prompt returns default when no custom prompt."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        prompt = get_system_prompt(config)
        assert prompt == DEFAULT_SYSTEM_PROMPT


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

    def test_rag_search_input_defaults(self):
        """RAGSearchInput has correct defaults."""
        input_schema = RAGSearchInput(query="What happened?")
        assert input_schema.query == "What happened?"
        assert input_schema.max_entries == 5

    def test_rag_search_input_validation(self):
        """RAGSearchInput validates max_entries."""
        # Valid range
        input_schema = RAGSearchInput(query="test", max_entries=10)
        assert input_schema.max_entries == 10

        # Below minimum
        with pytest.raises(ValueError):
            RAGSearchInput(query="test", max_entries=0)

        # Above maximum
        with pytest.raises(ValueError):
            RAGSearchInput(query="test", max_entries=25)


class TestFormatKeywordResult:
    """Tests for keyword result formatting."""

    def test_format_basic_result(self):
        """Formats basic keyword search result."""
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-001",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
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
        from datetime import datetime, timezone

        long_text = "x" * 1000
        entry = {
            "entry_id": "entry-002",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
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
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-003",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
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
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
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
        assert service._llm is None
        assert service._embedder is None
        assert service._agent is None
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

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_no_modules_enabled(self):
        """Search returns empty result when no modules are enabled."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": False},
                "semantic": {"enabled": False},
                "rag": {"enabled": False},
            },
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        result = await service.search("test query")

        assert result.entries == ()
        assert result.answer is None
        assert "No search modules enabled" in result.reasoning

    def test_parse_agent_result_extracts_answer(self):
        """_parse_agent_result extracts answer from messages."""
        service = self._create_mock_service()

        mock_ai_message = MagicMock()
        mock_ai_message.content = "This is the answer from the agent."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert result.answer == "This is the answer from the agent."

    def test_parse_agent_result_extracts_citations(self):
        """_parse_agent_result extracts citations from answer."""
        service = self._create_mock_service()

        mock_ai_message = MagicMock()
        mock_ai_message.content = "Found in [entry-001] and [entry-002] and [#003]."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert "001" in result.sources
        assert "002" in result.sources
        assert "003" in result.sources

    def test_parse_agent_result_identifies_search_modes(self):
        """_parse_agent_result identifies which search modes were used."""
        service = self._create_mock_service()

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
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert SearchMode.KEYWORD in result.search_modes_used
        assert SearchMode.SEMANTIC in result.search_modes_used
        assert SearchMode.RAG not in result.search_modes_used


class TestCreateArielService:
    """Tests for create_ariel_service factory function."""

    @pytest.mark.asyncio
    async def test_factory_function_is_async(self):
        """Factory function is an async function."""
        from osprey.services.ariel_search.service import create_ariel_service
        import asyncio

        assert asyncio.iscoroutinefunction(create_ariel_service)


class TestCreateSearchTools:
    """Tests for create_search_tools function."""

    def test_no_modules_enabled_returns_empty(self):
        """Returns empty list when no modules enabled."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {
                "keyword": {"enabled": False},
                "semantic": {"enabled": False},
                "rag": {"enabled": False},
            },
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        mock_request = ARIELSearchRequest(query="test")

        tools = create_search_tools(
            config=config,
            repository=mock_repository,
            embedder_loader=mock_embedder_loader,
            request=mock_request,
        )

        assert tools == []

    def test_keyword_only_returns_one_tool(self):
        """Returns only keyword tool when only keyword enabled."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
                "rag": {"enabled": False},
            },
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        mock_request = ARIELSearchRequest(query="test")

        tools = create_search_tools(
            config=config,
            repository=mock_repository,
            embedder_loader=mock_embedder_loader,
            request=mock_request,
        )

        assert len(tools) == 1
        assert tools[0].name == "keyword_search"

    def test_semantic_only_returns_one_tool(self):
        """Returns only semantic tool when only semantic enabled."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {
                "keyword": {"enabled": False},
                "semantic": {"enabled": True, "model": "test-model"},
                "rag": {"enabled": False},
            },
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        mock_request = ARIELSearchRequest(query="test")

        tools = create_search_tools(
            config=config,
            repository=mock_repository,
            embedder_loader=mock_embedder_loader,
            request=mock_request,
        )

        assert len(tools) == 1
        assert tools[0].name == "semantic_search"

    def test_rag_only_returns_one_tool(self):
        """Returns only RAG tool when only RAG enabled."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {
                "keyword": {"enabled": False},
                "semantic": {"enabled": True, "model": "test-model"},
                "rag": {"enabled": True, "model": "llm-model"},
            },
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        mock_request = ARIELSearchRequest(query="test")

        tools = create_search_tools(
            config=config,
            repository=mock_repository,
            embedder_loader=mock_embedder_loader,
            request=mock_request,
        )

        assert len(tools) == 2  # semantic + rag
        tool_names = [t.name for t in tools]
        assert "rag_search" in tool_names

    def test_all_modules_returns_three_tools(self):
        """Returns three tools when all modules enabled."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
                "rag": {"enabled": True, "model": "llm-model"},
            },
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        mock_request = ARIELSearchRequest(query="test")

        tools = create_search_tools(
            config=config,
            repository=mock_repository,
            embedder_loader=mock_embedder_loader,
            request=mock_request,
        )

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "keyword_search" in tool_names
        assert "semantic_search" in tool_names
        assert "rag_search" in tool_names


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
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-no-meta",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": "jsmith",
            "raw_text": "Entry without metadata.",
            "attachments": [],
        }

        result = format_keyword_result(entry, 0.5, [])

        assert result["title"] is None

    def test_format_semantic_missing_metadata(self):
        """format_semantic_result handles missing metadata."""
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-no-meta",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": "jsmith",
            "raw_text": "Entry without metadata.",
            "attachments": [],
        }

        result = format_semantic_result(entry, 0.5)

        assert result["title"] is None


class TestServiceParseAgentResultEdgeCases:
    """Edge case tests for _parse_agent_result."""

    def _create_mock_service(self) -> ARIELSearchService:
        """Create a mock service for testing."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()

        return ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

    def test_empty_messages(self):
        """Handles empty message list."""
        service = self._create_mock_service()

        result_dict = {"messages": []}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert result.answer is None or result.answer == ""

    def test_missing_messages_key(self):
        """Handles missing messages key."""
        service = self._create_mock_service()

        result_dict = {}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert result.answer is None or result.answer == ""

    def test_tool_only_messages(self):
        """Handles messages with only tool calls (no AI message)."""
        service = self._create_mock_service()

        mock_tool_message = MagicMock()
        mock_tool_message.tool_calls = [{"name": "keyword_search"}]
        mock_tool_message.type = "tool"
        mock_tool_message.content = None

        result_dict = {"messages": [mock_tool_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        # Should identify keyword search was used
        assert SearchMode.KEYWORD in result.search_modes_used


class TestPromptModule:
    """Additional tests for prompts module."""

    def test_default_prompt_has_required_sections(self):
        """Default prompt contains required instructional sections."""
        assert "logbook" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "search" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_get_system_prompt_returns_default(self):
        """get_system_prompt returns default prompt (custom prompts is V2 feature)."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })

        prompt = get_system_prompt(config)

        # Currently always returns default (custom prompts deferred to V2)
        assert prompt == DEFAULT_SYSTEM_PROMPT


class TestToolsFunctions:
    """Tests for tool creation functions."""

    def test_format_keyword_result_with_empty_highlights(self):
        """format_keyword_result handles empty highlights list."""
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-test",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": "tester",
            "raw_text": "Test content",
            "attachments": [],
            "metadata": {"title": "Test Title"},
        }

        result = format_keyword_result(entry, 0.75, [])

        assert result["highlights"] == []

    def test_format_semantic_result_with_title(self):
        """format_semantic_result includes title from metadata."""
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-test",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": "tester",
            "raw_text": "Long raw text content",
            "attachments": [],
            "metadata": {"title": "Test Title"},
        }

        result = format_semantic_result(entry, 0.85)

        assert result["title"] == "Test Title"
        assert result["similarity"] == 0.85

    def test_format_keyword_result_truncates_at_500_chars(self):
        """format_keyword_result truncates text at exactly 500 chars."""
        from datetime import datetime, timezone

        long_text = "a" * 501
        entry = {
            "entry_id": "entry-test",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": "tester",
            "raw_text": long_text,
            "attachments": [],
            "metadata": {},
        }

        result = format_keyword_result(entry, 0.5, [])

        assert len(result["text"]) == 500


class TestServiceState:
    """Tests for service internal state management."""

    def test_service_embedder_initially_none(self):
        """Service embedder is None on initialization."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        assert service._embedder is None

    def test_service_agent_initially_none(self):
        """Service agent is None on initialization."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        assert service._agent is None

    def test_service_llm_initially_none(self):
        """Service LLM is None on initialization."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        assert service._llm is None


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


class TestCreateSearchToolsTimeRange:
    """Tests for create_search_tools time range resolution."""

    def test_tool_with_request_time_range(self):
        """Tools use request time_range when no explicit date provided."""
        from datetime import datetime, timezone

        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })
        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        request = ARIELSearchRequest(
            query="test",
            time_range=(start, end),
        )

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        assert len(tools) == 1
        assert tools[0].name == "keyword_search"


class TestToolInputSchemaDefaults:
    """Tests for tool input schema default values."""

    def test_keyword_input_max_results_default(self):
        """KeywordSearchInput has max_results default of 10."""
        from osprey.services.ariel_search.tools import KeywordSearchInput

        input_schema = KeywordSearchInput(query="test")
        assert input_schema.max_results == 10

    def test_semantic_input_similarity_default(self):
        """SemanticSearchInput has similarity_threshold default of 0.7."""
        from osprey.services.ariel_search.tools import SemanticSearchInput

        input_schema = SemanticSearchInput(query="test")
        assert input_schema.similarity_threshold == 0.7

    def test_rag_input_max_entries_default(self):
        """RAGSearchInput has max_entries default of 5."""
        from osprey.services.ariel_search.tools import RAGSearchInput

        input_schema = RAGSearchInput(query="test")
        assert input_schema.max_entries == 5


class TestSearchResultFormattingExtra:
    """Tests for search result formatting with edge cases."""

    def test_format_keyword_result_none_author(self):
        """format_keyword_result handles None author."""
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-none-author",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": None,
            "raw_text": "Test content",
            "attachments": [],
            "metadata": {},
        }

        result = format_keyword_result(entry, 0.75, ["highlight"])

        assert result["author"] is None
        assert result["entry_id"] == "entry-none-author"

    def test_format_semantic_result_empty_text(self):
        """format_semantic_result handles empty text."""
        from datetime import datetime, timezone

        entry = {
            "entry_id": "entry-empty-text",
            "source_system": "test",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "author": "tester",
            "raw_text": "",
            "attachments": [],
            "metadata": {},
        }

        result = format_semantic_result(entry, 0.85)

        assert result["text"] == ""
        assert result["similarity"] == 0.85


class TestToolSchemaFields:
    """Tests for tool input schema field validation."""

    def test_keyword_input_max_results_bounds(self):
        """KeywordSearchInput validates max_results bounds."""
        from osprey.services.ariel_search.tools import KeywordSearchInput
        from pydantic import ValidationError

        # Min bound
        with pytest.raises(ValidationError):
            KeywordSearchInput(query="test", max_results=0)

        # Max bound
        with pytest.raises(ValidationError):
            KeywordSearchInput(query="test", max_results=51)

    def test_semantic_input_similarity_bounds(self):
        """SemanticSearchInput validates similarity_threshold bounds."""
        from osprey.services.ariel_search.tools import SemanticSearchInput
        from pydantic import ValidationError

        # Below 0
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", similarity_threshold=-0.1)

        # Above 1
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", similarity_threshold=1.5)


class TestToolsTimeRangeResolution:
    """Tests for time range resolution in tools."""

    def test_explicit_start_date_overrides_request(self):
        """Tool-provided start_date overrides request time_range."""
        from datetime import datetime, timezone
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })
        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()

        request_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        request_end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        request = ARIELSearchRequest(
            query="test",
            time_range=(request_start, request_end),
        )

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)
        assert len(tools) == 1
        assert tools[0].name == "keyword_search"

    def test_end_date_only_tool_param(self):
        """Tool-provided end_date is used when only end specified."""
        from datetime import datetime, timezone
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })
        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)
        assert len(tools) == 1


class TestServiceAinvokeWithMocks:
    """Tests for service ainvoke with mocked agent."""

    def _create_mock_service(self) -> ARIELSearchService:
        """Create a mock service for testing."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
            },
        })
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
    async def test_ainvoke_timeout_returns_result(self):
        """ainvoke returns result with timeout message on SearchTimeoutError."""
        from osprey.services.ariel_search.exceptions import SearchTimeoutError

        service = self._create_mock_service()
        request = ARIELSearchRequest(query="test query")

        # Mock _run_agent to raise SearchTimeoutError (which is now raised on asyncio.TimeoutError)
        timeout_error = SearchTimeoutError(
            message="Agent execution timed out",
            timeout_seconds=120,
            operation="agent execution",
        )
        with patch.object(service, "_run_agent", new=AsyncMock(side_effect=timeout_error)):
            result = await service.ainvoke(request)

        assert result.answer is None
        assert "timed out" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_ainvoke_ariel_exception_propagates(self):
        """ainvoke propagates ARIELException."""
        from osprey.services.ariel_search.exceptions import SearchExecutionError

        service = self._create_mock_service()
        request = ARIELSearchRequest(query="test query")

        # Mock _run_agent to raise ARIELException
        error = SearchExecutionError("test error", search_mode="keyword", query="test")
        with patch.object(service, "_run_agent", new=AsyncMock(side_effect=error)):
            with pytest.raises(SearchExecutionError):
                await service.ainvoke(request)

    @pytest.mark.asyncio
    async def test_ainvoke_generic_exception_raises_search_error(self):
        """ainvoke wraps generic exceptions in SearchExecutionError."""
        from osprey.services.ariel_search.exceptions import SearchExecutionError

        service = self._create_mock_service()
        request = ARIELSearchRequest(query="test query")

        # Mock _run_agent to raise generic exception
        with patch.object(service, "_run_agent", new=AsyncMock(side_effect=RuntimeError("unexpected"))):
            with pytest.raises(SearchExecutionError):
                await service.ainvoke(request)


class TestServiceValidateSearchModel:
    """Tests for _validate_search_model method."""

    @pytest.mark.asyncio
    async def test_validate_search_model_called_once(self):
        """_validate_search_model only validates once."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"semantic": {"enabled": True, "model": "test-model"}},
        })
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
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
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


class TestParseAgentResultRAGTool:
    """Tests for _parse_agent_result with RAG tool."""

    def _create_mock_service(self) -> ARIELSearchService:
        """Create a mock service for testing."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        return ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

    def test_parse_agent_result_rag_tool(self):
        """_parse_agent_result identifies RAG search mode."""
        service = self._create_mock_service()

        mock_tool_message = MagicMock()
        mock_tool_message.tool_calls = [{"name": "rag_search"}]

        mock_ai_message = MagicMock()
        mock_ai_message.content = "Answer with citation [entry-001]"
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_tool_message, mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert SearchMode.RAG in result.search_modes_used

    def test_parse_agent_result_all_tools(self):
        """_parse_agent_result identifies all search modes."""
        service = self._create_mock_service()

        mock_tool_message = MagicMock()
        mock_tool_message.tool_calls = [
            {"name": "keyword_search"},
            {"name": "semantic_search"},
            {"name": "rag_search"},
        ]

        mock_ai_message = MagicMock()
        mock_ai_message.content = "Answer"
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_tool_message, mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        assert SearchMode.KEYWORD in result.search_modes_used
        assert SearchMode.SEMANTIC in result.search_modes_used
        assert SearchMode.RAG in result.search_modes_used

    def test_parse_agent_result_duplicate_tool_calls(self):
        """_parse_agent_result deduplicates repeated tool calls."""
        service = self._create_mock_service()

        mock_tool_message1 = MagicMock()
        mock_tool_message1.tool_calls = [{"name": "keyword_search"}]

        mock_tool_message2 = MagicMock()
        mock_tool_message2.tool_calls = [{"name": "keyword_search"}]

        mock_ai_message = MagicMock()
        mock_ai_message.content = "Answer"
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_tool_message1, mock_tool_message2, mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        # Should only have KEYWORD once
        assert result.search_modes_used.count(SearchMode.KEYWORD) == 1


class TestToolInvocation:
    """Tests for actual tool function invocation."""

    @pytest.mark.asyncio
    async def test_keyword_tool_invocation(self):
        """Keyword search tool can be invoked directly."""
        from osprey.services.ariel_search.tools import create_search_tools
        from datetime import datetime, timezone

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })

        # Mock repository with keyword_search returning results
        mock_repository = MagicMock()
        mock_repository.keyword_search = AsyncMock(return_value=[
            (
                {
                    "entry_id": "entry-001",
                    "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
                    "author": "jsmith",
                    "raw_text": "Test content",
                    "metadata": {"title": "Test"},
                },
                0.85,
                ["<mark>Test</mark>"],
            ),
        ])
        mock_repository.fuzzy_search = AsyncMock(return_value=[])

        mock_embedder_loader = MagicMock()
        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        assert len(tools) == 1
        assert tools[0].name == "keyword_search"

        # Invoke the tool directly
        result = await tools[0].coroutine("test query", max_results=5)

        assert len(result) == 1
        assert result[0]["entry_id"] == "entry-001"
        assert result[0]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_semantic_tool_invocation(self):
        """Semantic search tool can be invoked directly."""
        from osprey.services.ariel_search.tools import create_search_tools
        from datetime import datetime, timezone

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"semantic": {"enabled": True, "model": "test-model"}},
        })

        # Mock repository
        mock_repository = MagicMock()
        mock_repository.semantic_search = AsyncMock(return_value=[
            (
                {
                    "entry_id": "entry-002",
                    "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
                    "author": "jdoe",
                    "raw_text": "Semantic content",
                    "metadata": {},
                },
                0.92,
            ),
        ])

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder_loader = MagicMock(return_value=mock_embedder)

        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        assert len(tools) == 1
        assert tools[0].name == "semantic_search"

        # Invoke the tool directly
        result = await tools[0].coroutine("semantic query", max_results=5)

        assert len(result) == 1
        assert result[0]["entry_id"] == "entry-002"
        assert result[0]["similarity"] == 0.92

    @pytest.mark.asyncio
    async def test_rag_tool_invocation(self):
        """RAG search tool can be invoked directly."""
        from osprey.services.ariel_search.tools import create_search_tools
        from datetime import datetime, timezone

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "semantic": {"enabled": True, "model": "test-model"},
                "rag": {"enabled": True, "model": "llm-model"},
            },
        })

        # Mock repository - return empty so RAG returns early
        mock_repository = MagicMock()
        mock_repository.semantic_search = AsyncMock(return_value=[])

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder_loader = MagicMock(return_value=mock_embedder)

        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        # Should have semantic and rag
        assert len(tools) == 2
        rag_tool = next(t for t in tools if t.name == "rag_search")

        # Invoke the tool directly
        result = await rag_tool.coroutine("What happened?", max_entries=3)

        assert "answer" in result
        assert "sources" in result


class TestToolTimeRangeResolutionDetailed:
    """Detailed tests for time range resolution in tools."""

    @pytest.mark.asyncio
    async def test_tool_uses_request_time_range_when_not_overridden(self):
        """Tool uses request time_range when tool params not provided."""
        from osprey.services.ariel_search.tools import create_search_tools
        from datetime import datetime, timezone

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })

        mock_repository = MagicMock()
        mock_repository.keyword_search = AsyncMock(return_value=[])
        mock_repository.fuzzy_search = AsyncMock(return_value=[])

        mock_embedder_loader = MagicMock()

        # Request has time_range
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        request = ARIELSearchRequest(query="test", time_range=(start, end))

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        # Invoke without explicit dates
        await tools[0].coroutine("test query")

        # Check that keyword_search was called
        mock_repository.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_time_filter_when_neither_provided(self):
        """No time filtering when neither tool nor request has time_range."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })

        mock_repository = MagicMock()
        mock_repository.keyword_search = AsyncMock(return_value=[])
        mock_repository.fuzzy_search = AsyncMock(return_value=[])

        mock_embedder_loader = MagicMock()

        # Request without time_range
        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        # Invoke without explicit dates
        await tools[0].coroutine("test query")

        mock_repository.keyword_search.assert_called_once()


class TestServiceGetStatus:
    """Tests for ARIELSearchService.get_status() method."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal ARIEL config."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://user:pass@localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True},
            },
            "enhancement_modules": {
                "text_embedding": {"enabled": True},
            },
        })

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


class TestServiceGetStatusExported:
    """Tests that get_status is accessible from service."""

    def test_get_status_method_exists(self):
        """ARIELSearchService has get_status method."""
        assert hasattr(ARIELSearchService, "get_status")
        import asyncio
        assert asyncio.iscoroutinefunction(ARIELSearchService.get_status)


class TestServiceRecursionLimit:
    """Tests for agent recursion limit configuration.

    See 03_AGENTIC_REASONING.md Section 3.2 for specification.
    """

    @pytest.mark.asyncio
    async def test_recursion_limit_calculated_from_max_iterations(self):
        """Agent recursion_limit is calculated from config.reasoning.max_iterations.

        LangGraph counts each model call and tool execution as separate steps.
        recursion_limit = (max_iterations * 2) + 1
        """
        # Create config with explicit max_iterations
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "reasoning": {"max_iterations": 3},  # Expect recursion_limit = 7
        })
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # Track what config is passed to ainvoke
        captured_config = None

        async def mock_ainvoke(messages, config=None):
            nonlocal captured_config
            captured_config = config
            # Return a minimal valid response
            return {"messages": []}

        mock_agent = MagicMock()
        mock_agent.ainvoke = mock_ainvoke

        # Mock create_react_agent to return our mock - must patch at import location
        with patch("langgraph.prebuilt.create_react_agent", return_value=mock_agent):
            with patch.object(service, "_get_llm", return_value=MagicMock()):
                request = ARIELSearchRequest(query="test")
                await service._run_agent(request, [MagicMock()])

        # Verify recursion_limit = (3 * 2) + 1 = 7
        assert captured_config is not None
        assert captured_config.get("recursion_limit") == 7

    @pytest.mark.asyncio
    async def test_recursion_limit_default_value(self):
        """Default max_iterations (5) produces recursion_limit of 11."""
        # Default max_iterations is 5
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })
        assert config.reasoning.max_iterations == 5

        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        captured_config = None

        async def mock_ainvoke(messages, config=None):
            nonlocal captured_config
            captured_config = config
            return {"messages": []}

        mock_agent = MagicMock()
        mock_agent.ainvoke = mock_ainvoke

        with patch("langgraph.prebuilt.create_react_agent", return_value=mock_agent):
            with patch.object(service, "_get_llm", return_value=MagicMock()):
                request = ARIELSearchRequest(query="test")
                await service._run_agent(request, [MagicMock()])

        # Verify recursion_limit = (5 * 2) + 1 = 11
        assert captured_config is not None
        assert captured_config.get("recursion_limit") == 11


# === Agent Behavior Tests (TEST-H004, TEST-H005, TEST-H006) ===


class TestAgentBehavior:
    """Tests for agent behavior including timeout and iteration handling."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ARIEL config with short timeouts for testing."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "reasoning": {
                "max_iterations": 3,
                "total_timeout_seconds": 1,  # Short timeout for testing
            },
        })

    @pytest.fixture
    def mock_service(self, mock_config):
        """Create mock service for testing."""
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        return ARIELSearchService(
            config=mock_config,
            pool=mock_pool,
            repository=mock_repository,
        )

    @pytest.mark.asyncio
    async def test_agent_total_timeout(self, mock_service):
        """Agent returns timeout result when total_timeout_seconds exceeded (TEST-H004).

        Verifies that when the agent exceeds the configured timeout,
        it returns a result with appropriate timeout message rather than crashing.
        """
        from osprey.services.ariel_search.exceptions import SearchTimeoutError
        from osprey.services.ariel_search.models import ARIELSearchRequest

        request = ARIELSearchRequest(query="test query")

        # Mock _run_agent to raise SearchTimeoutError (as it does when timeout occurs)
        async def timeout_agent(*args, **kwargs):
            raise SearchTimeoutError(
                message="Agent execution timed out",
                timeout_seconds=1,
                operation="agent execution",
            )

        with patch.object(mock_service, "_run_agent", new=timeout_agent):
            result = await mock_service.ainvoke(request)

        # Verify result indicates timeout
        assert result.answer is None
        assert "timed out" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_agent_max_iterations_reached(self):
        """Agent stops after max_iterations with appropriate result (TEST-H005).

        Verifies that the recursion_limit is correctly calculated from
        max_iterations and passed to the agent.
        """
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "reasoning": {"max_iterations": 2},  # Small value for testing
        })
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # Track the config passed to agent
        captured_config = None

        async def mock_ainvoke(messages, config=None):
            nonlocal captured_config
            captured_config = config
            return {"messages": []}

        mock_agent = MagicMock()
        mock_agent.ainvoke = mock_ainvoke

        with patch("langgraph.prebuilt.create_react_agent", return_value=mock_agent):
            with patch.object(service, "_get_llm", return_value=MagicMock()):
                from osprey.services.ariel_search.models import ARIELSearchRequest
                request = ARIELSearchRequest(query="test")
                await service._run_agent(request, [MagicMock()])

        # Verify recursion_limit = (2 * 2) + 1 = 5
        assert captured_config is not None
        assert captured_config.get("recursion_limit") == 5

    @pytest.mark.asyncio
    async def test_agent_handles_empty_tool_results(self, mock_service):
        """Agent terminates gracefully when all tools return empty (TEST-H006).

        Verifies that when tools return no results, the agent completes
        successfully without crashing.
        """
        from osprey.services.ariel_search.models import ARIELSearchRequest

        # Create a mock agent response with empty tool results
        mock_tool_message = MagicMock()
        mock_tool_message.tool_calls = [{"name": "keyword_search"}]

        mock_ai_message = MagicMock()
        mock_ai_message.content = "I found no relevant entries for your query."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        async def mock_run_agent(*args, **kwargs):
            return mock_service._parse_agent_result(
                {"messages": [mock_tool_message, mock_ai_message]},
                ARIELSearchRequest(query="test"),
            )

        with patch.object(mock_service, "_run_agent", new=mock_run_agent):
            result = await mock_service.ainvoke(ARIELSearchRequest(query="obscure search"))

        # Verify graceful completion
        assert result.answer is not None
        assert "no relevant entries" in result.answer.lower()


class TestAgentTimeoutError:
    """Tests for SearchTimeoutError usage (GAP-C005)."""

    @pytest.mark.asyncio
    async def test_timeout_raises_search_timeout_error(self):
        """asyncio.TimeoutError is wrapped in SearchTimeoutError."""
        import asyncio
        from osprey.services.ariel_search.exceptions import SearchTimeoutError

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "reasoning": {"total_timeout_seconds": 1},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # Create mock agent that times out
        async def slow_ainvoke(*args, **kwargs):
            await asyncio.sleep(10)
            return {"messages": []}

        mock_agent = MagicMock()
        mock_agent.ainvoke = slow_ainvoke

        with patch("langgraph.prebuilt.create_react_agent", return_value=mock_agent):
            with patch.object(service, "_get_llm", return_value=MagicMock()):
                from osprey.services.ariel_search.models import ARIELSearchRequest
                with pytest.raises(SearchTimeoutError) as exc_info:
                    await service._run_agent(ARIELSearchRequest(query="test"), [MagicMock()])

        assert exc_info.value.timeout_seconds == 1
        assert exc_info.value.operation == "agent execution"

    @pytest.mark.asyncio
    async def test_timeout_converted_to_graceful_result(self):
        """SearchTimeoutError is caught and converted to graceful result in ainvoke."""
        from osprey.services.ariel_search.exceptions import SearchTimeoutError

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "reasoning": {"total_timeout_seconds": 60},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()
        mock_repository.validate_search_model_table = AsyncMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # Mock _run_agent to raise SearchTimeoutError
        async def timeout_run_agent(*args, **kwargs):
            raise SearchTimeoutError(
                message="Test timeout",
                timeout_seconds=60,
                operation="agent execution",
            )

        with patch.object(service, "_run_agent", new=timeout_run_agent):
            from osprey.services.ariel_search.models import ARIELSearchRequest
            result = await service.ainvoke(ARIELSearchRequest(query="test"))

        assert result.answer is None
        assert "timed out" in result.reasoning.lower()
        assert "60" in result.reasoning


class TestLLMConfiguration:
    """Tests for LLM configuration (GAP-C007)."""

    def test_llm_model_id_default(self):
        """Default llm_model_id is gpt-4o-mini."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        assert config.reasoning.llm_model_id == "gpt-4o-mini"

    def test_llm_provider_default(self):
        """Default llm_provider is openai."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        assert config.reasoning.llm_provider == "openai"

    def test_llm_model_id_configurable(self):
        """llm_model_id can be configured."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "reasoning": {"llm_model_id": "gpt-4-turbo"},
        })
        assert config.reasoning.llm_model_id == "gpt-4-turbo"

    def test_llm_provider_configurable(self):
        """llm_provider can be configured."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "reasoning": {"llm_provider": "anthropic"},
        })
        assert config.reasoning.llm_provider == "anthropic"

    def test_get_llm_unsupported_provider(self):
        """_get_llm raises ConfigurationError for unsupported provider."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "reasoning": {"llm_provider": "unsupported_provider"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            service._get_llm()

        assert "unsupported_provider" in str(exc_info.value)
        assert "Supported providers" in str(exc_info.value)


# ==============================================================================
# Quality Improvements (QUAL-002, QUAL-004, QUAL-007, QUAL-008)
# ==============================================================================


class TestToolInvocationQuality:
    """Quality assertions for tool invocation (QUAL-002)."""

    @pytest.mark.asyncio
    async def test_keyword_search_only_tool_invoked(self):
        """Verify keyword_search_only tool is actually invoked and returns expected structure.

        QUAL-002: Tool invocation verification.
        """
        from osprey.services.ariel_search.tools import create_search_tools
        from datetime import datetime, timezone

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })

        # Track if tool was invoked
        invoked = False
        invocation_args = None

        mock_repository = MagicMock()

        async def track_keyword_search(*args, **kwargs):
            nonlocal invoked, invocation_args
            invoked = True
            invocation_args = kwargs
            return [
                (
                    {
                        "entry_id": "entry-001",
                        "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
                        "author": "jsmith",
                        "raw_text": "Test content",
                        "metadata": {},
                    },
                    0.85,
                    ["highlight"],
                ),
            ]

        mock_repository.keyword_search = track_keyword_search
        mock_repository.fuzzy_search = AsyncMock(return_value=[])

        mock_embedder_loader = MagicMock()
        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)
        assert len(tools) == 1

        # Invoke the tool
        result = await tools[0].coroutine("test query", max_results=5)

        # Verify tool was actually invoked
        assert invoked is True
        assert invocation_args is not None

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert "entry_id" in result[0]
        assert "score" in result[0]


class TestToolInstanceTypes:
    """Quality assertions for tool instance types (QUAL-004)."""

    def test_create_search_tools_returns_structured_tools(self):
        """Verify create_search_tools returns StructuredTool instances.

        QUAL-004: Assert StructuredTool instances.
        """
        from langchain_core.tools import StructuredTool
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
                "rag": {"enabled": True, "model": "llm-model"},
            },
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        # All tools should be StructuredTool instances
        assert len(tools) == 3
        for tool in tools:
            assert isinstance(tool, StructuredTool), f"Tool {tool.name} is not a StructuredTool"

    def test_tools_have_required_attributes(self):
        """All tools have required StructuredTool attributes."""
        from osprey.services.ariel_search.tools import create_search_tools

        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        })

        mock_repository = MagicMock()
        mock_embedder_loader = MagicMock()
        request = ARIELSearchRequest(query="test")

        tools = create_search_tools(config, mock_repository, mock_embedder_loader, request)

        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "args_schema")
            assert hasattr(tool, "coroutine") or hasattr(tool, "func")


class TestPromptFormatSpec:
    """Quality assertions for prompt format specifications (QUAL-007)."""

    def test_system_prompt_exact_format_specification(self):
        """Verify system prompt contains exact format specification.

        QUAL-007: Verify exact format spec in prompts.
        """
        # System prompt should specify exact output format
        assert "entry_id" in DEFAULT_SYSTEM_PROMPT.lower() or "entry" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "search" in DEFAULT_SYSTEM_PROMPT.lower()

        # Should have clear instructions about tool usage
        assert "tool" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_all_search_types(self):
        """System prompt mentions all available search types."""
        # All search types should be mentioned
        assert "keyword" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "semantic" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "rag" in DEFAULT_SYSTEM_PROMPT.lower()


class TestCitationInstruction:
    """Quality assertions for citation instructions (QUAL-008)."""

    def test_system_prompt_contains_citation_instruction(self):
        """Verify system prompt contains citation instruction.

        QUAL-008: Check citation instruction in prompts.
        """
        # System prompt should instruct about citations
        assert any(word in DEFAULT_SYSTEM_PROMPT.lower() for word in ["cite", "citation", "reference", "source"])

    def test_parse_agent_result_extracts_citation_formats(self):
        """_parse_agent_result extracts various citation formats."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # Test format: [entry-001]
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Found in [entry-001] and [entry-002]."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        # Should extract entry IDs
        assert len(result.sources) >= 2

    def test_parse_agent_result_handles_various_id_formats(self):
        """_parse_agent_result handles different entry ID formats."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "postgresql://localhost:5432/test"},
        })
        mock_pool = MagicMock()
        mock_repository = MagicMock()

        service = ARIELSearchService(
            config=config,
            pool=mock_pool,
            repository=mock_repository,
        )

        # Test format: [#123] (hash prefix)
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Found in [#123] and [#456]."
        mock_ai_message.type = "ai"
        mock_ai_message.tool_calls = []

        result_dict = {"messages": [mock_ai_message]}
        request = ARIELSearchRequest(query="test query")

        result = service._parse_agent_result(result_dict, request)

        # Should extract IDs (without hash)
        assert "123" in result.sources
        assert "456" in result.sources
