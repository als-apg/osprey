"""Tests for ARIEL search diagnostics.

Tests for SearchDiagnostic, DiagnosticLevel, RAG pipeline diagnostics,
service-level diagnostics, and API schema diagnostics.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.models import (
    ARIELSearchResult,
    DiagnosticLevel,
    SearchDiagnostic,
)
from osprey.services.ariel_search.rag import RAGPipeline, RAGResult


def _make_entry(entry_id: str, text: str = "Test content") -> dict:
    """Create a mock EnhancedLogbookEntry dict."""
    return {
        "entry_id": entry_id,
        "source_system": "ALS eLog",
        "timestamp": datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
        "author": "jsmith",
        "raw_text": text,
        "attachments": [],
        "metadata": {"title": f"Entry {entry_id}"},
        "created_at": datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
        "updated_at": datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
    }


def _make_config(keyword_enabled=True, semantic_enabled=False) -> ARIELConfig:
    """Create a minimal ARIELConfig for testing."""
    modules = {}
    if keyword_enabled:
        modules["keyword"] = {"enabled": True}
    if semantic_enabled:
        modules["semantic"] = {"enabled": True, "model": "test-model"}
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": modules,
        }
    )


class TestDiagnosticLevel:
    """Tests for DiagnosticLevel enum."""

    def test_values(self) -> None:
        assert DiagnosticLevel.INFO.value == "info"
        assert DiagnosticLevel.WARNING.value == "warning"
        assert DiagnosticLevel.ERROR.value == "error"

    def test_all_levels(self) -> None:
        assert len(DiagnosticLevel) == 3


class TestSearchDiagnostic:
    """Tests for SearchDiagnostic dataclass."""

    def test_basic_creation(self) -> None:
        diag = SearchDiagnostic(
            level=DiagnosticLevel.ERROR,
            source="rag.retrieve.keyword",
            message="Keyword retrieval failed: connection refused",
        )
        assert diag.level == DiagnosticLevel.ERROR
        assert diag.source == "rag.retrieve.keyword"
        assert "connection refused" in diag.message
        assert diag.category is None

    def test_with_category(self) -> None:
        diag = SearchDiagnostic(
            level=DiagnosticLevel.WARNING,
            source="rag.retrieve.semantic",
            message="Embedder load failed",
            category="embedding",
        )
        assert diag.category == "embedding"

    def test_frozen(self) -> None:
        diag = SearchDiagnostic(
            level=DiagnosticLevel.INFO,
            source="rag.assemble",
            message="Context truncated",
        )
        with pytest.raises(AttributeError):
            diag.message = "changed"  # type: ignore[misc]


class TestARIELSearchResultDiagnostics:
    """Tests for diagnostics field on ARIELSearchResult."""

    def test_default_empty(self) -> None:
        result = ARIELSearchResult(entries=())
        assert result.diagnostics == ()

    def test_with_diagnostics(self) -> None:
        diag = SearchDiagnostic(
            level=DiagnosticLevel.ERROR,
            source="service.keyword",
            message="Search failed",
        )
        result = ARIELSearchResult(entries=(), diagnostics=(diag,))
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].level == DiagnosticLevel.ERROR


class TestRAGResultDiagnostics:
    """Tests for diagnostics field on RAGResult."""

    def test_default_empty(self) -> None:
        result = RAGResult(answer="test")
        assert result.diagnostics == ()

    def test_with_diagnostics(self) -> None:
        diag = SearchDiagnostic(
            level=DiagnosticLevel.INFO,
            source="rag.assemble",
            message="Context was truncated",
        )
        result = RAGResult(answer="test", diagnostics=(diag,))
        assert len(result.diagnostics) == 1


class TestRAGPipelineDiagnostics:
    """Tests for diagnostics accumulation in RAGPipeline.execute()."""

    @pytest.mark.asyncio
    async def test_retrieval_failure_emits_error_diagnostic(self) -> None:
        """When keyword search raises, an ERROR diagnostic is emitted."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(),
        )

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new_callable=AsyncMock,
            side_effect=RuntimeError("similarity() function missing"),
        ):
            result = await pipeline.execute("test query")

        # Should have retrieval error diagnostic
        error_diags = [d for d in result.diagnostics if d.level == DiagnosticLevel.ERROR]
        assert len(error_diags) >= 1
        assert any("rag.retrieve.keyword" in d.source for d in error_diags)
        assert any("similarity()" in d.message for d in error_diags)

    @pytest.mark.asyncio
    async def test_both_retrievals_fail_emits_all_failed_diagnostic(self) -> None:
        """When both retrievals fail, an 'all failed' diagnostic is emitted."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(),
        )

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("keyword error"),
            ),
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("semantic error"),
            ),
        ):
            result = await pipeline.execute("test query")

        sources = [d.source for d in result.diagnostics]
        assert "rag.retrieve" in sources  # the "all failed" diagnostic

    @pytest.mark.asyncio
    async def test_embedder_load_failure_emits_warning(self) -> None:
        """When embedder fails to load, a WARNING diagnostic is emitted."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(side_effect=RuntimeError("No Ollama")),
        )

        kw_results = [(_make_entry("001", "Found"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer [#001].",
            ),
        ):
            result = await pipeline.execute("test query")

        warning_diags = [d for d in result.diagnostics if d.level == DiagnosticLevel.WARNING]
        assert len(warning_diags) >= 1
        assert any("rag.retrieve.semantic" in d.source for d in warning_diags)
        assert any(d.category == "embedding" for d in warning_diags)

    @pytest.mark.asyncio
    async def test_context_truncation_emits_info_diagnostic(self) -> None:
        """When context is truncated, an INFO diagnostic is emitted."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(),
            max_context_chars=100,
        )

        kw_results = [
            (_make_entry("001", "A" * 200), 0.9, []),
            (_make_entry("002", "B" * 200), 0.8, []),
        ]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer.",
            ),
        ):
            result = await pipeline.execute("test query")

        info_diags = [d for d in result.diagnostics if d.level == DiagnosticLevel.INFO]
        assert any("rag.assemble" in d.source for d in info_diags)
        assert result.context_truncated is True

    @pytest.mark.asyncio
    async def test_llm_failure_emits_error_diagnostic(self) -> None:
        """When LLM call fails, an ERROR diagnostic is emitted."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(),
        )

        kw_results = [(_make_entry("001", "Some entry"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                side_effect=RuntimeError("API connection failed"),
            ),
        ):
            result = await pipeline.execute("test query")

        error_diags = [d for d in result.diagnostics if d.level == DiagnosticLevel.ERROR]
        assert any("rag.generate" in d.source for d in error_diags)

    @pytest.mark.asyncio
    async def test_llm_import_error_emits_warning_diagnostic(self) -> None:
        """When LLM module is not available, a WARNING diagnostic is emitted."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(),
        )

        kw_results = [(_make_entry("001", "Some entry"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.services.ariel_search.rag.get_chat_completion",
                side_effect=ImportError("No module"),
                create=True,
            ),
            patch.dict("sys.modules", {"osprey.models.completion": None}),
        ):
            result = await pipeline.execute("test query")

        warning_diags = [d for d in result.diagnostics if d.level == DiagnosticLevel.WARNING]
        assert any("rag.generate" in d.source for d in warning_diags)

    @pytest.mark.asyncio
    async def test_successful_pipeline_no_diagnostics(self) -> None:
        """Successful pipeline run produces no diagnostics."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline = RAGPipeline(
            repository=MagicMock(),
            config=config,
            embedder_loader=MagicMock(),
        )

        kw_results = [(_make_entry("001", "Good result"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Clean answer [#001].",
            ),
        ):
            result = await pipeline.execute("test query")

        assert result.diagnostics == ()


class TestServiceDiagnostics:
    """Tests for service-level diagnostic emission."""

    @pytest.mark.asyncio
    async def test_keyword_search_failure_returns_diagnostic(self) -> None:
        """_run_keyword wraps exception in graceful result with diagnostic."""
        from osprey.services.ariel_search.service import ARIELSearchService

        config = _make_config(keyword_enabled=True)
        service = ARIELSearchService(
            config=config,
            pool=MagicMock(),
            repository=MagicMock(),
        )

        request = MagicMock()
        request.query = "test"
        request.max_results = 10
        request.time_range = None
        request.advanced_params = {}

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new_callable=AsyncMock,
            side_effect=RuntimeError("pg_trgm missing"),
        ):
            result = await service._run_keyword(request)

        assert result.entries == ()
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].level == DiagnosticLevel.ERROR
        assert result.diagnostics[0].source == "service.keyword"
        assert "pg_trgm" in result.diagnostics[0].message

    @pytest.mark.asyncio
    async def test_semantic_search_failure_returns_diagnostic(self) -> None:
        """_run_semantic wraps exception in graceful result with diagnostic."""
        from osprey.services.ariel_search.service import ARIELSearchService

        config = _make_config(keyword_enabled=False, semantic_enabled=True)
        service = ARIELSearchService(
            config=config,
            pool=MagicMock(),
            repository=MagicMock(),
        )

        request = MagicMock()
        request.query = "test"
        request.max_results = 10
        request.time_range = None
        request.advanced_params = {}

        with patch(
            "osprey.services.ariel_search.search.semantic.semantic_search",
            new_callable=AsyncMock,
            side_effect=RuntimeError("embedding table missing"),
        ):
            result = await service._run_semantic(request)

        assert result.entries == ()
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].level == DiagnosticLevel.ERROR
        assert result.diagnostics[0].source == "service.semantic"

    @pytest.mark.asyncio
    async def test_timeout_returns_diagnostic(self) -> None:
        """SearchTimeoutError produces a timeout diagnostic."""
        from osprey.services.ariel_search.exceptions import SearchTimeoutError
        from osprey.services.ariel_search.models import ARIELSearchRequest
        from osprey.services.ariel_search.service import ARIELSearchService

        config = _make_config(keyword_enabled=True)
        service = ARIELSearchService(
            config=config,
            pool=MagicMock(),
            repository=MagicMock(),
        )

        request = ARIELSearchRequest(query="test")

        with patch.object(
            service,
            "_run_rag",
            new_callable=AsyncMock,
            side_effect=SearchTimeoutError(
                message="timed out",
                timeout_seconds=30,
                operation="RAG pipeline",
            ),
        ):
            result = await service.ainvoke(request)

        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].level == DiagnosticLevel.ERROR
        assert result.diagnostics[0].source == "service.timeout"
        assert result.diagnostics[0].category == "timeout"


class TestDiagnosticResponseSchema:
    """Tests for DiagnosticResponse in API schemas."""

    def test_diagnostic_response_creation(self) -> None:
        from osprey.interfaces.ariel.api.schemas import DiagnosticResponse

        diag = DiagnosticResponse(
            level="error",
            source="rag.retrieve.keyword",
            message="Keyword retrieval failed",
            category="search",
        )
        assert diag.level == "error"
        assert diag.source == "rag.retrieve.keyword"
        assert diag.message == "Keyword retrieval failed"
        assert diag.category == "search"

    def test_diagnostic_response_optional_category(self) -> None:
        from osprey.interfaces.ariel.api.schemas import DiagnosticResponse

        diag = DiagnosticResponse(
            level="info",
            source="rag.assemble",
            message="Context truncated",
        )
        assert diag.category is None

    def test_search_response_includes_diagnostics(self) -> None:
        from osprey.interfaces.ariel.api.schemas import (
            DiagnosticResponse,
            SearchResponse,
        )

        response = SearchResponse(
            entries=[],
            diagnostics=[
                DiagnosticResponse(
                    level="error",
                    source="service.keyword",
                    message="Search failed",
                ),
            ],
        )
        assert len(response.diagnostics) == 1
        assert response.diagnostics[0].level == "error"

    def test_search_response_default_empty(self) -> None:
        from osprey.interfaces.ariel.api.schemas import SearchResponse

        response = SearchResponse(entries=[])
        assert response.diagnostics == []
