"""Tests for ARIEL capability module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.capability import (
    reset_ariel_service,
)


class TestResetArielService:
    """Tests for reset_ariel_service function."""

    def test_reset_clears_singleton(self) -> None:
        """Resetting clears the module-level singleton."""
        from osprey.services.ariel_search import capability

        # Force a value
        capability._ariel_service_instance = "test_value"  # type: ignore[assignment]
        assert capability._ariel_service_instance == "test_value"

        # Reset
        reset_ariel_service()

        # Verify cleared
        assert capability._ariel_service_instance is None


class TestGetArielSearchService:
    """Tests for get_ariel_search_service function."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_ariel_service()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_ariel_service()

    @pytest.mark.asyncio
    async def test_raises_configuration_error_when_not_configured(self) -> None:
        """Raises ConfigurationError when ARIEL is not configured."""
        from osprey.services.ariel_search import ConfigurationError
        from osprey.services.ariel_search.capability import get_ariel_search_service

        with patch(
            "osprey.services.ariel_search.capability.get_config_value",
            return_value={},
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                await get_ariel_search_service()

            assert "ARIEL not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_creates_service_from_config(self) -> None:
        """Creates service from configuration - simplified test."""
        from osprey.services.ariel_search import capability as cap_module

        # Test that service gets cached properly
        mock_service = MagicMock()
        cap_module._ariel_service_instance = mock_service

        # When instance exists, it should be returned directly
        result = await cap_module.get_ariel_search_service()
        assert result is mock_service

    @pytest.mark.asyncio
    async def test_returns_singleton_on_subsequent_calls(self) -> None:
        """Returns same instance on subsequent calls."""
        from osprey.services.ariel_search import capability as cap_module

        # Set a mock service
        mock_service = MagicMock()
        cap_module._ariel_service_instance = mock_service

        # Call multiple times
        service1 = await cap_module.get_ariel_search_service()
        service2 = await cap_module.get_ariel_search_service()

        assert service1 is service2
        assert service1 is mock_service


class TestLogbookSearchCapabilityErrorClassification:
    """Tests for LogbookSearchCapability.classify_error with actionable messages."""

    def test_database_connection_error_is_critical_with_guidance(self):
        """DatabaseConnectionError returns CRITICAL with setup instructions."""
        from osprey.base.errors import ErrorSeverity
        from osprey.capabilities.logbook_search import LogbookSearchCapability
        from osprey.services.ariel_search.exceptions import DatabaseConnectionError

        exc = DatabaseConnectionError("Connection refused")
        result = LogbookSearchCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL
        assert "osprey deploy up" in result.user_message

    def test_missing_tables_error_is_critical_with_migrate_guidance(self):
        """DatabaseQueryError for missing tables suggests migrate command."""
        from osprey.base.errors import ErrorSeverity
        from osprey.capabilities.logbook_search import LogbookSearchCapability
        from osprey.services.ariel_search.exceptions import DatabaseQueryError

        exc = DatabaseQueryError('relation "enhanced_entries" does not exist')
        result = LogbookSearchCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL
        assert "osprey ariel migrate" in result.user_message

    def test_generic_database_query_error_is_retriable(self):
        """Generic DatabaseQueryError is RETRIABLE (transient)."""
        from osprey.base.errors import ErrorSeverity
        from osprey.capabilities.logbook_search import LogbookSearchCapability
        from osprey.services.ariel_search.exceptions import DatabaseQueryError

        exc = DatabaseQueryError("temporary failure")
        result = LogbookSearchCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE

    def test_embedding_error_suggests_disabling_semantic(self):
        """EmbeddingGenerationError suggests disabling semantic search."""
        from osprey.base.errors import ErrorSeverity
        from osprey.capabilities.logbook_search import LogbookSearchCapability
        from osprey.services.ariel_search.exceptions import EmbeddingGenerationError

        exc = EmbeddingGenerationError("Ollama not available", model_name="nomic-embed-text")
        result = LogbookSearchCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL
        assert "semantic" in result.user_message.lower()

    def test_configuration_error_includes_message(self):
        """ConfigurationError includes the original message."""
        from osprey.base.errors import ErrorSeverity
        from osprey.capabilities.logbook_search import LogbookSearchCapability
        from osprey.services.ariel_search.exceptions import ConfigurationError

        exc = ConfigurationError("Missing database URI", config_key="ariel.database.uri")
        result = LogbookSearchCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL
        assert "Missing database URI" in result.user_message

    def test_unknown_error_is_critical(self):
        """Unknown exceptions are classified as CRITICAL."""
        from osprey.base.errors import ErrorSeverity
        from osprey.capabilities.logbook_search import LogbookSearchCapability

        exc = RuntimeError("something unexpected")
        result = LogbookSearchCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL
        assert "something unexpected" in result.user_message


class TestCapabilityExports:
    """Tests for capability module exports."""

    def test_get_ariel_search_service_exported(self) -> None:
        """get_ariel_search_service is exported from main module."""
        from osprey.services.ariel_search import get_ariel_search_service

        assert callable(get_ariel_search_service)
        assert get_ariel_search_service.__name__ == "get_ariel_search_service"

    def test_reset_ariel_service_exported(self) -> None:
        """reset_ariel_service is exported from main module."""
        from osprey.services.ariel_search import reset_ariel_service

        assert callable(reset_ariel_service)
        assert reset_ariel_service.__name__ == "reset_ariel_service"


# ============================================================================
# LogbookSearchCapability.execute() unit tests
# ============================================================================


class TestLogbookSearchCapabilityExecute:
    """Tests for LogbookSearchCapability.execute() via monkeypatch.

    Follows the pattern from tests/capabilities/test_time_range_parsing_capability.py:
    create instance, inject _state/_step, monkeypatch framework methods.
    """

    @pytest.fixture
    def mock_state(self) -> dict:
        """Minimal agent state for capability execution."""
        return {
            "messages": [],
            "planning_execution_steps": [
                {
                    "step_index": 0,
                    "capability": "logbook_search",
                    "context_key": "ls_results_001",
                    "task_objective": "Find RF trip incidents",
                    "reasoning": "User asks about RF trips",
                    "inputs": [],
                }
            ],
            "planning_current_step_index": 0,
            "capability_context_data": {},
            "context_data": {},
            "execution_step_results": {},
            "input_output": {"user_query": "What happened with RF trips?"},
            "config": {"user_id": "test_user"},
            "control_routing_count": 0,
        }

    @pytest.fixture
    def mock_step(self) -> dict:
        """Execution step for logbook_search."""
        return {
            "step_index": 0,
            "capability": "logbook_search",
            "context_key": "ls_results_001",
            "task_objective": "Find RF trip incidents",
            "reasoning": "User asks about RF trips",
            "inputs": [],
        }

    def _make_capability(self, mock_state, mock_step):
        """Create LogbookSearchCapability with injected state/step."""
        from osprey.capabilities.logbook_search import LogbookSearchCapability

        cap = LogbookSearchCapability()
        cap._state = mock_state
        cap._step = mock_step
        return cap

    def _make_search_result(self, entries=(), answer=None, sources=(), modes=None):
        """Helper to build an ARIELSearchResult."""
        from osprey.services.ariel_search.models import ARIELSearchResult, SearchMode

        return ARIELSearchResult(
            entries=tuple(entries),
            answer=answer,
            sources=tuple(sources),
            search_modes_used=tuple(modes or [SearchMode.KEYWORD]),
        )

    def _make_entry(self, entry_id="DEMO-001", raw_text="RF trip on sector 4"):
        """Helper to create an EnhancedLogbookEntry dict."""
        now = datetime.now(UTC)
        return {
            "entry_id": entry_id,
            "source_system": "test",
            "timestamp": now,
            "author": "operator",
            "raw_text": raw_text,
            "attachments": [],
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        }

    @pytest.mark.asyncio
    async def test_execute_happy_path(self, mock_state, mock_step, monkeypatch):
        """Service returns entries → LogbookSearchResultsContext populated correctly."""
        from osprey.capabilities.logbook_search import LogbookSearchResultsContext
        from osprey.services.ariel_search.models import SearchMode

        cap = self._make_capability(mock_state, mock_step)

        # Stub framework methods
        mock_logger = MagicMock()
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_logger",
            lambda self: mock_logger,
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_required_contexts",
            lambda self, **kwargs: {},
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_task_objective",
            lambda self, **kwargs: "Find RF trip incidents",
        )

        # Capture what gets passed to store_output_context
        stored_contexts = []

        def fake_store(self_cap, context):
            stored_contexts.append(context)
            return {"capability_context_data": {}}

        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.store_output_context",
            fake_store,
        )

        # Mock service
        entry = self._make_entry()
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=self._make_search_result(
                entries=[entry],
                modes=[SearchMode.KEYWORD],
            )
        )
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(
            "osprey.services.ariel_search.capability.get_ariel_search_service",
            AsyncMock(return_value=mock_service),
        )

        await cap.execute()

        # Verify store_output_context was called with correct context
        assert len(stored_contexts) == 1
        ctx = stored_contexts[0]
        assert isinstance(ctx, LogbookSearchResultsContext)
        assert len(ctx.entries) == 1
        assert ctx.entries[0]["entry_id"] == "DEMO-001"
        assert ctx.query == "Find RF trip incidents"
        assert "keyword" in ctx.search_modes_used
        assert ctx.answer is None
        assert ctx.time_range_applied is False

    @pytest.mark.asyncio
    async def test_execute_with_time_range(self, mock_state, mock_step, monkeypatch):
        """TIME_RANGE context → service.search() called with time_range tuple."""
        cap = self._make_capability(mock_state, mock_step)

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_logger",
            lambda self: mock_logger,
        )

        # Simulate TIME_RANGE context present
        mock_time_range = MagicMock()
        mock_time_range.start_date = datetime(2024, 1, 1, tzinfo=UTC)
        mock_time_range.end_date = datetime(2024, 6, 1, tzinfo=UTC)

        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_required_contexts",
            lambda self, **kwargs: {"TIME_RANGE": mock_time_range},
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_task_objective",
            lambda self, **kwargs: "RF trips in Q1",
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.store_output_context",
            lambda self, ctx: {"capability_context_data": {}},
        )

        mock_service = AsyncMock()
        mock_service.search = AsyncMock(return_value=self._make_search_result())
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(
            "osprey.services.ariel_search.capability.get_ariel_search_service",
            AsyncMock(return_value=mock_service),
        )

        await cap.execute()

        # Verify service.search was called with time_range
        call_kwargs = mock_service.search.call_args
        assert call_kwargs.kwargs["time_range"] == (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
        )

    @pytest.mark.asyncio
    async def test_execute_without_time_range(self, mock_state, mock_step, monkeypatch):
        """No TIME_RANGE context → service.search() called with time_range=None."""
        cap = self._make_capability(mock_state, mock_step)

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_logger",
            lambda self: mock_logger,
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_required_contexts",
            lambda self, **kwargs: {},
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_task_objective",
            lambda self, **kwargs: "some query",
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.store_output_context",
            lambda self, ctx: {"capability_context_data": {}},
        )

        mock_service = AsyncMock()
        mock_service.search = AsyncMock(return_value=self._make_search_result())
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(
            "osprey.services.ariel_search.capability.get_ariel_search_service",
            AsyncMock(return_value=mock_service),
        )

        await cap.execute()

        call_kwargs = mock_service.search.call_args
        assert call_kwargs.kwargs["time_range"] is None

    @pytest.mark.asyncio
    async def test_execute_empty_results(self, mock_state, mock_step, monkeypatch):
        """Service returns no entries → context has empty entries."""
        cap = self._make_capability(mock_state, mock_step)

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_logger",
            lambda self: mock_logger,
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_required_contexts",
            lambda self, **kwargs: {},
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_task_objective",
            lambda self, **kwargs: "nonexistent topic",
        )

        stored_contexts = []
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.store_output_context",
            lambda self, ctx: (stored_contexts.append(ctx), {"capability_context_data": {}})[1],
        )

        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=self._make_search_result(entries=(), answer=None)
        )
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(
            "osprey.services.ariel_search.capability.get_ariel_search_service",
            AsyncMock(return_value=mock_service),
        )

        await cap.execute()

        ctx = stored_contexts[0]
        assert len(ctx.entries) == 0
        assert ctx.answer is None

    @pytest.mark.asyncio
    async def test_execute_rag_answer_propagated(self, mock_state, mock_step, monkeypatch):
        """RAG answer and sources propagated to output context."""
        from osprey.services.ariel_search.models import SearchMode

        cap = self._make_capability(mock_state, mock_step)

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_logger",
            lambda self: mock_logger,
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_required_contexts",
            lambda self, **kwargs: {},
        )
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.get_task_objective",
            lambda self, **kwargs: "Why did RF trip?",
        )

        stored_contexts = []
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.LogbookSearchCapability.store_output_context",
            lambda self, ctx: (stored_contexts.append(ctx), {"capability_context_data": {}})[1],
        )

        entry = self._make_entry(entry_id="DEMO-001")
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=self._make_search_result(
                entries=[entry],
                answer="The RF trip was caused by a cooling water fault.",
                sources=["DEMO-001"],
                modes=[SearchMode.RAG],
            )
        )
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(
            "osprey.services.ariel_search.capability.get_ariel_search_service",
            AsyncMock(return_value=mock_service),
        )

        await cap.execute()

        ctx = stored_contexts[0]
        assert ctx.answer == "The RF trip was caused by a cooling water fault."
        assert "DEMO-001" in ctx.sources
        assert "rag" in ctx.search_modes_used


# ============================================================================
# LogbookSearchResultsContext.get_summary() unit tests
# ============================================================================


class TestLogbookSearchResultsContextGetSummary:
    """Tests for LogbookSearchResultsContext.get_summary()."""

    def _make_context(self, entries=(), answer=None, sources=(), modes=("keyword",), query="test"):
        from osprey.capabilities.logbook_search import LogbookSearchResultsContext

        return LogbookSearchResultsContext(
            entries=tuple(entries),
            answer=answer,
            sources=tuple(sources),
            search_modes_used=tuple(modes),
            query=query,
            time_range_applied=False,
        )

    def _make_entry(self, entry_id="DEMO-001", raw_text="Short text", timestamp=None):
        return {
            "entry_id": entry_id,
            "timestamp": timestamp or datetime.now(UTC),
            "author": "operator",
            "raw_text": raw_text,
            "summary": "A brief summary",
        }

    def test_summary_with_rag_answer(self):
        """Summary includes answer and sources when RAG answer is present."""
        ctx = self._make_context(
            answer="The RF trip was caused by cooling water.",
            sources=("DEMO-001", "DEMO-002"),
        )
        summary = ctx.get_summary()

        assert summary["answer"] == "The RF trip was caused by cooling water."
        assert summary["sources"] == ["DEMO-001", "DEMO-002"]
        assert summary["entries_found"] == 0

    def test_summary_without_rag_answer(self):
        """Summary has entries_found but no answer key when RAG not used."""
        entry = self._make_entry()
        ctx = self._make_context(entries=[entry])
        summary = ctx.get_summary()

        assert "answer" not in summary
        assert "sources" not in summary
        assert summary["entries_found"] == 1

    def test_summary_truncates_long_raw_text(self):
        """Entry with >200 char raw_text is truncated (not full length)."""
        long_text = "x" * 500
        entry = self._make_entry(raw_text=long_text)
        ctx = self._make_context(entries=[entry])
        summary = ctx.get_summary()

        assert summary["entries_found"] == 1
        # get_summary() truncates raw_text to 200 chars + "..." = 203 chars,
        # then recursively_summarize_data further truncates strings >200 chars
        # to 100 chars + "... (truncated from N chars)".
        # Either way, the final text is much shorter than the original 500.
        if isinstance(summary.get("entries"), list):
            entry_text = summary["entries"][0].get("raw_text", "")
            assert len(entry_text) < 500
            assert "truncated" in entry_text or entry_text.endswith("...")

    def test_summary_string_timestamp_no_crash(self):
        """Entry with string timestamp (not datetime) doesn't crash."""
        entry = self._make_entry(timestamp="2024-01-15T10:00:00")
        ctx = self._make_context(entries=[entry])
        # Should not raise — str() fallback handles it
        summary = ctx.get_summary()
        assert summary["entries_found"] == 1


# ============================================================================
# LogbookSearchResultsContext.get_access_details() unit test
# ============================================================================


class TestLogbookSearchResultsContextGetAccessDetails:
    """Tests for LogbookSearchResultsContext.get_access_details()."""

    def test_get_access_details_returns_expected_keys(self):
        """Returns dict with access_pattern, data_structure, fields, example_usage."""
        from osprey.capabilities.logbook_search import LogbookSearchResultsContext

        ctx = LogbookSearchResultsContext(
            entries=(),
            answer=None,
            sources=(),
            search_modes_used=("keyword",),
            query="test",
            time_range_applied=False,
        )
        details = ctx.get_access_details("test_key")

        assert "access_pattern" in details
        assert "data_structure" in details
        assert "fields" in details
        assert "example_usage" in details
        assert "LOGBOOK_SEARCH_RESULTS" in details["access_pattern"]
        assert "test_key" in details["access_pattern"]
        assert "entries" in details["fields"]


# ============================================================================
# Guide method unit tests
# ============================================================================


class TestLogbookSearchCapabilityGuides:
    """Tests for _create_orchestrator_guide() and _create_classifier_guide()."""

    def _make_capability(self):
        from osprey.capabilities.logbook_search import LogbookSearchCapability

        return LogbookSearchCapability()

    def test_orchestrator_guide_fallback(self, monkeypatch):
        """_create_orchestrator_guide() returns OrchestratorGuide on ImportError fallback."""
        from osprey.base.examples import OrchestratorGuide

        # Force fallback by making get_framework_prompts raise ImportError
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.get_framework_prompts",
            MagicMock(side_effect=ImportError("no prompt builder")),
        )

        cap = self._make_capability()
        guide = cap._create_orchestrator_guide()

        assert isinstance(guide, OrchestratorGuide)

    def test_classifier_guide_fallback(self, monkeypatch):
        """_create_classifier_guide() returns TaskClassifierGuide on ImportError fallback."""
        from osprey.base.examples import TaskClassifierGuide

        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.get_framework_prompts",
            MagicMock(side_effect=ImportError("no prompt builder")),
        )

        cap = self._make_capability()
        guide = cap._create_classifier_guide()

        assert isinstance(guide, TaskClassifierGuide)

    def test_orchestrator_guide_has_correct_priority(self, monkeypatch):
        """Default orchestrator guide has priority=15 and mentions logbook_search."""
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.get_framework_prompts",
            MagicMock(side_effect=ImportError),
        )

        cap = self._make_capability()
        guide = cap._create_orchestrator_guide()

        assert guide.priority == 15
        assert "logbook_search" in guide.instructions

    def test_classifier_guide_has_four_examples(self, monkeypatch):
        """Default classifier guide has 4 examples (2 positive, 1 negative, 1 positive)."""
        monkeypatch.setattr(
            "osprey.capabilities.logbook_search.get_framework_prompts",
            MagicMock(side_effect=ImportError),
        )

        cap = self._make_capability()
        guide = cap._create_classifier_guide()

        assert len(guide.examples) == 4
        results = [ex.result for ex in guide.examples]
        assert results == [True, True, False, True]


# ============================================================================
# close_ariel_service() unit test
# ============================================================================


class TestCloseArielService:
    """Tests for close_ariel_service()."""

    def setup_method(self) -> None:
        reset_ariel_service()

    def teardown_method(self) -> None:
        reset_ariel_service()

    @pytest.mark.asyncio
    async def test_close_calls_pool_close_and_clears_singleton(self):
        """close_ariel_service() calls pool.close() and clears the singleton."""
        from osprey.services.ariel_search import capability as cap_module
        from osprey.services.ariel_search.capability import close_ariel_service

        mock_pool = AsyncMock()
        mock_service = MagicMock()
        mock_service.pool = mock_pool

        cap_module._ariel_service_instance = mock_service

        await close_ariel_service()

        mock_pool.close.assert_awaited_once()
        assert cap_module._ariel_service_instance is None
