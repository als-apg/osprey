"""ARIEL Web API route tests.

Tests for all REST API endpoints in the ARIEL web frontend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from httpx import AsyncClient

    from osprey.services.ariel_search.models import ARIELStatusResult


# =============================================================================
# POST /api/search Tests
# =============================================================================


class TestSearchEndpoint:
    """Tests for POST /api/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_basic(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search with query returns results."""
        response = await client.post(
            "/api/search",
            json={"query": "beam alignment"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total_results" in data
        assert data["total_results"] >= 0
        mock_ariel_service.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_mode_keyword(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search with explicit keyword mode."""
        response = await client.post(
            "/api/search",
            json={"query": "test query", "mode": "keyword"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_search_with_mode_semantic(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search with explicit semantic mode."""
        response = await client.post(
            "/api/search",
            json={"query": "conceptual query", "mode": "semantic"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_search_with_mode_rag(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search with explicit RAG mode."""
        response = await client.post(
            "/api/search",
            json={"query": "What happened yesterday?", "mode": "rag"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "answer" in data

    @pytest.mark.asyncio
    async def test_search_with_date_filters(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search with start/end date filtering."""
        response = await client.post(
            "/api/search",
            json={
                "query": "beam",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_search_with_max_results(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search with custom max_results."""
        response = await client.post(
            "/api/search",
            json={"query": "test", "max_results": 25},
        )

        assert response.status_code == 200
        # Verify max_results was passed to service
        call_kwargs = mock_ariel_service.search.call_args.kwargs
        assert call_kwargs.get("max_results") == 25

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(
        self,
        client: AsyncClient,
    ) -> None:
        """Empty query returns validation error."""
        response = await client.post(
            "/api/search",
            json={"query": ""},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_search_service_error(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Service exception returns 500 error."""
        mock_ariel_service.search = AsyncMock(
            side_effect=RuntimeError("Database connection failed")
        )

        response = await client.post(
            "/api/search",
            json={"query": "test"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_search_response_structure(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search response has expected structure."""
        response = await client.post(
            "/api/search",
            json={"query": "test"},
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields per SearchResponse schema
        assert "entries" in data
        assert "total_results" in data
        assert "execution_time_ms" in data
        assert isinstance(data["entries"], list)
        assert isinstance(data["total_results"], int)
        assert isinstance(data["execution_time_ms"], int)


# =============================================================================
# GET /api/entries Tests
# =============================================================================


class TestListEntriesEndpoint:
    """Tests for GET /api/entries endpoint."""

    @pytest.mark.asyncio
    async def test_list_entries_default(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """List entries with default pagination."""
        response = await client.get("/api/entries")

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

    @pytest.mark.asyncio
    async def test_list_entries_pagination(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """List entries with custom page and page_size."""
        response = await client.get("/api/entries?page=2&page_size=50")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 50

    @pytest.mark.asyncio
    async def test_list_entries_date_filters(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """List entries with date filters."""
        response = await client.get(
            "/api/entries?start_date=2024-01-01T00:00:00Z&end_date=2024-12-31T23:59:59Z"
        )

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_list_entries_author_filter(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """List entries filtered by author."""
        response = await client.get("/api/entries?author=jsmith")

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_list_entries_source_filter(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """List entries filtered by source system."""
        response = await client.get("/api/entries?source_system=ALS%20Logbook")

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_list_entries_sort_order(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """List entries with sort order."""
        response = await client.get("/api/entries?sort_order=asc")

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_list_entries_service_error(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Service exception returns 500 error."""
        mock_ariel_service.repository.count_entries = AsyncMock(
            side_effect=RuntimeError("Database error")
        )

        response = await client.get("/api/entries")

        assert response.status_code == 500


# =============================================================================
# GET /api/entries/{entry_id} Tests
# =============================================================================


class TestGetEntryEndpoint:
    """Tests for GET /api/entries/{entry_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_entry_found(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
        sample_entry: dict,
    ) -> None:
        """Returns entry by ID when found."""
        response = await client.get("/api/entries/test-001")

        assert response.status_code == 200
        data = response.json()
        assert data["entry_id"] == "test-001"
        assert "raw_text" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_get_entry_not_found(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Returns 404 for missing entry."""
        mock_ariel_service.repository.get_entry = AsyncMock(return_value=None)

        response = await client.get("/api/entries/nonexistent-id")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_entry_service_error(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Service exception returns 500 error."""
        mock_ariel_service.repository.get_entry = AsyncMock(
            side_effect=RuntimeError("Database error")
        )

        response = await client.get("/api/entries/test-001")

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_get_entry_response_structure(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Entry response has expected structure."""
        response = await client.get("/api/entries/test-001")

        assert response.status_code == 200
        data = response.json()

        # Required fields per EntryResponse schema
        assert "entry_id" in data
        assert "source_system" in data
        assert "timestamp" in data
        assert "author" in data
        assert "raw_text" in data
        assert "created_at" in data
        assert "updated_at" in data


# =============================================================================
# POST /api/entries Tests
# =============================================================================


class TestCreateEntryEndpoint:
    """Tests for POST /api/entries endpoint."""

    @pytest.mark.asyncio
    async def test_create_entry_success(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Create entry with valid data."""
        response = await client.post(
            "/api/entries",
            json={
                "subject": "Test Entry Subject",
                "details": "Detailed description of the logbook entry.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "entry_id" in data
        assert "message" in data
        assert data["entry_id"].startswith("ariel-")
        mock_ariel_service.repository.upsert_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entry_with_optional_fields(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Create entry with all optional fields."""
        response = await client.post(
            "/api/entries",
            json={
                "subject": "Test Subject",
                "details": "Test details",
                "author": "jsmith",
                "logbook": "operations",
                "shift": "day",
                "tags": ["beam", "alignment"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "entry_id" in data

    @pytest.mark.asyncio
    async def test_create_entry_missing_subject(
        self,
        client: AsyncClient,
    ) -> None:
        """Missing subject returns validation error."""
        response = await client.post(
            "/api/entries",
            json={"details": "Only details, no subject"},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_entry_missing_details(
        self,
        client: AsyncClient,
    ) -> None:
        """Missing details returns validation error."""
        response = await client.post(
            "/api/entries",
            json={"subject": "Only subject, no details"},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_entry_empty_subject(
        self,
        client: AsyncClient,
    ) -> None:
        """Empty subject returns validation error."""
        response = await client.post(
            "/api/entries",
            json={"subject": "", "details": "Some details"},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_entry_empty_details(
        self,
        client: AsyncClient,
    ) -> None:
        """Empty details returns validation error."""
        response = await client.post(
            "/api/entries",
            json={"subject": "Some subject", "details": ""},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_entry_service_error(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Service exception returns 500 error."""
        mock_ariel_service.repository.upsert_entry = AsyncMock(
            side_effect=RuntimeError("Database error")
        )

        response = await client.post(
            "/api/entries",
            json={"subject": "Test", "details": "Test details"},
        )

        assert response.status_code == 500


# =============================================================================
# GET /api/status Tests
# =============================================================================


class TestStatusEndpoint:
    """Tests for GET /api/status endpoint."""

    @pytest.mark.asyncio
    async def test_status_healthy(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Returns healthy status."""
        response = await client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert data["database_connected"] is True
        assert "entry_count" in data
        assert "embedding_tables" in data
        assert "enabled_search_modules" in data

    @pytest.mark.asyncio
    async def test_status_with_errors(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
        sample_status_result: ARIELStatusResult,
    ) -> None:
        """Returns degraded status with errors."""
        from osprey.services.ariel_search.models import ARIELStatusResult

        # Create degraded status
        degraded_status = ARIELStatusResult(
            healthy=False,
            database_connected=False,
            database_uri="postgresql://***@localhost:5432/ariel",
            entry_count=None,
            embedding_tables=[],
            active_embedding_model=None,
            enabled_search_modules=["keyword"],
            enabled_enhancement_modules=[],
            last_ingestion=None,
            errors=["Database connection failed", "Embedding service unavailable"],
        )
        mock_ariel_service.get_status = AsyncMock(return_value=degraded_status)

        response = await client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is False
        assert data["database_connected"] is False
        assert len(data["errors"]) == 2

    @pytest.mark.asyncio
    async def test_status_response_structure(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Status response has expected structure."""
        response = await client.get("/api/status")

        assert response.status_code == 200
        data = response.json()

        # Required fields per StatusResponse schema
        assert "healthy" in data
        assert "database_connected" in data
        assert "database_uri" in data
        assert "embedding_tables" in data
        assert "enabled_search_modules" in data
        assert "enabled_enhancement_modules" in data
        assert "errors" in data
        assert isinstance(data["embedding_tables"], list)
        assert isinstance(data["errors"], list)

    @pytest.mark.asyncio
    async def test_status_service_error(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Service exception returns 500 error."""
        mock_ariel_service.get_status = AsyncMock(side_effect=RuntimeError("Status check failed"))

        response = await client.get("/api/status")

        assert response.status_code == 500


# =============================================================================
# GET /health Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Health endpoint returns OK when service is healthy."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data

    @pytest.mark.asyncio
    async def test_health_endpoint_degraded(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Health endpoint returns degraded when service has issues."""
        mock_ariel_service.health_check = AsyncMock(
            return_value=(False, "Database connection lost")
        )

        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "message" in data


# =============================================================================
# Entry Response Format Tests
# =============================================================================


class TestEntryResponseFormat:
    """Tests for entry response formatting."""

    @pytest.mark.asyncio
    async def test_entry_includes_optional_fields(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
        sample_entry: dict,
    ) -> None:
        """Entry response includes optional fields when present."""
        response = await client.get("/api/entries/test-001")

        assert response.status_code == 200
        data = response.json()

        # Optional fields from sample_entry
        assert "summary" in data
        assert "keywords" in data
        assert isinstance(data["keywords"], list)

    @pytest.mark.asyncio
    async def test_search_entry_includes_score(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search results include relevance score."""
        response = await client.post(
            "/api/search",
            json={"query": "beam"},
        )

        assert response.status_code == 200
        data = response.json()

        # Entries in search results may include score
        if data["entries"]:
            # Score field exists but may be None for non-ranked results
            entry = data["entries"][0]
            assert "score" in entry

    @pytest.mark.asyncio
    async def test_search_entry_includes_highlights(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search results include highlights."""
        response = await client.post(
            "/api/search",
            json={"query": "beam"},
        )

        assert response.status_code == 200
        data = response.json()

        if data["entries"]:
            entry = data["entries"][0]
            assert "highlights" in entry
            assert isinstance(entry["highlights"], list)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_body(
        self,
        client: AsyncClient,
    ) -> None:
        """Invalid JSON returns 422 error."""
        response = await client.post(
            "/api/search",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_required_field(
        self,
        client: AsyncClient,
    ) -> None:
        """Missing required field returns 422 error."""
        response = await client.post(
            "/api/search",
            json={},  # Missing required 'query' field
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_search_mode(
        self,
        client: AsyncClient,
    ) -> None:
        """Invalid search mode returns 422 error."""
        response = await client.post(
            "/api/search",
            json={"query": "test", "mode": "invalid_mode"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_date_format(
        self,
        client: AsyncClient,
    ) -> None:
        """Invalid date format returns 422 error."""
        response = await client.post(
            "/api/search",
            json={"query": "test", "start_date": "not-a-date"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_max_results_below_minimum(
        self,
        client: AsyncClient,
    ) -> None:
        """max_results below minimum returns 422 error."""
        response = await client.post(
            "/api/search",
            json={"query": "test", "max_results": 0},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_max_results_above_maximum(
        self,
        client: AsyncClient,
    ) -> None:
        """max_results above maximum returns 422 error."""
        response = await client.post(
            "/api/search",
            json={"query": "test", "max_results": 101},
        )

        assert response.status_code == 422


# =============================================================================
# Integration with ARIEL Service Tests
# =============================================================================


class TestServiceIntegration:
    """Tests for integration with ARIELSearchService."""

    @pytest.mark.asyncio
    async def test_search_passes_correct_params(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search endpoint passes correct parameters to service."""
        await client.post(
            "/api/search",
            json={
                "query": "beam alignment",
                "max_results": 15,
            },
        )

        mock_ariel_service.search.assert_called_once()
        call_kwargs = mock_ariel_service.search.call_args.kwargs
        assert call_kwargs["query"] == "beam alignment"
        assert call_kwargs["max_results"] == 15

    @pytest.mark.asyncio
    async def test_search_passes_time_range(
        self,
        client: AsyncClient,
        mock_ariel_service: MagicMock,
    ) -> None:
        """Search endpoint passes time range to service."""
        await client.post(
            "/api/search",
            json={
                "query": "test",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
            },
        )

        mock_ariel_service.search.assert_called_once()
        call_kwargs = mock_ariel_service.search.call_args.kwargs
        assert call_kwargs["time_range"] is not None
        start, end = call_kwargs["time_range"]
        assert start is not None
        assert end is not None
