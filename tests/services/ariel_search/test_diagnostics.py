"""Tests for ARIEL search diagnostics.

Tests for SearchDiagnostic, DiagnosticLevel, and service-level diagnostics.
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
            "_run_keyword",
            new_callable=AsyncMock,
            side_effect=SearchTimeoutError(
                message="timed out",
                timeout_seconds=30,
                operation="keyword search",
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
