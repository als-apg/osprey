"""ARIEL Web API test fixtures.

Provides mocked service and FastAPI test client for API testing.

Requires optional dependencies: fastapi, httpx
Install with: pip install fastapi httpx
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from osprey.services.ariel_search.models import ARIELSearchResult, ARIELStatusResult

# Check for optional dependencies at collection time
import importlib.util

_fastapi_available = importlib.util.find_spec("fastapi") is not None
_httpx_available = importlib.util.find_spec("httpx") is not None
_skip_reason = None

if not _fastapi_available:
    _skip_reason = "fastapi not installed (pip install fastapi)"
elif not _httpx_available:
    _skip_reason = "httpx not installed (pip install httpx)"


def pytest_collection_modifyitems(items):
    """Skip all tests if dependencies are not available."""
    if not _fastapi_available or not _httpx_available:
        skip_marker = pytest.mark.skip(reason=_skip_reason)
        for item in items:
            if "ariel_web" in str(item.fspath):
                item.add_marker(skip_marker)


# Only define fixtures if dependencies are available
if _fastapi_available and _httpx_available:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    @pytest.fixture
    def sample_entry() -> dict:
        """Sample logbook entry matching database format.

        Returns:
            Dict with all required entry fields.
        """
        now = datetime.now(UTC)
        return {
            "entry_id": "test-001",
            "source_system": "ALS Logbook",
            "timestamp": now,
            "author": "test_user",
            "raw_text": "Test entry content about beam alignment",
            "attachments": [],
            "metadata": {"title": "Beam Alignment Update", "logbook": "operations"},
            "created_at": now,
            "updated_at": now,
            "summary": "Test summary",
            "keywords": ["beam", "alignment"],
        }

    @pytest.fixture
    def sample_search_result(sample_entry: dict) -> ARIELSearchResult:
        """Sample ARIELSearchResult with test data.

        Args:
            sample_entry: Sample entry fixture.

        Returns:
            ARIELSearchResult with one entry.
        """
        from osprey.services.ariel_search.models import ARIELSearchResult, SearchMode

        return ARIELSearchResult(
            entries=(sample_entry,),
            answer="Found entry about beam alignment.",
            sources=("test-001",),
            search_modes_used=(SearchMode.KEYWORD,),
            reasoning="Keyword search matched query terms.",
        )

    @pytest.fixture
    def sample_status_result() -> ARIELStatusResult:
        """Sample ARIELStatusResult for status endpoint testing.

        Returns:
            ARIELStatusResult with healthy status.
        """
        from osprey.services.ariel_search.models import ARIELStatusResult, EmbeddingTableInfo

        return ARIELStatusResult(
            healthy=True,
            database_connected=True,
            database_uri="postgresql://***@localhost:5432/ariel",
            entry_count=100,
            embedding_tables=[
                EmbeddingTableInfo(
                    table_name="text_embeddings_nomic_embed_text",
                    entry_count=100,
                    dimension=768,
                    is_active=True,
                ),
            ],
            active_embedding_model="nomic-embed-text",
            enabled_search_modules=["keyword", "semantic"],
            enabled_pipelines=["rag", "agent"],
            enabled_enhancement_modules=["text_embedding"],
            last_ingestion=datetime.now(UTC),
            errors=[],
        )

    @pytest.fixture
    def mock_ariel_service(
        sample_entry: dict,
        sample_search_result: ARIELSearchResult,
        sample_status_result: ARIELStatusResult,
    ) -> MagicMock:
        """Mocked ARIELSearchService for unit testing.

        Args:
            sample_entry: Sample entry fixture.
            sample_search_result: Sample search result fixture.
            sample_status_result: Sample status result fixture.

        Returns:
            MagicMock with ARIELSearchService methods stubbed.
        """
        service = MagicMock()
        service.health_check = AsyncMock(return_value=(True, "ARIEL service healthy"))
        service.search = AsyncMock(return_value=sample_search_result)
        service.get_status = AsyncMock(return_value=sample_status_result)

        # Mock repository for direct access from routes
        service.repository = MagicMock()
        service.repository.count_entries = AsyncMock(return_value=100)
        service.repository.search_by_time_range = AsyncMock(return_value=[sample_entry])
        service.repository.get_entry = AsyncMock(return_value=sample_entry)
        service.repository.upsert_entry = AsyncMock()

        return service

    @pytest.fixture
    def app_with_mock_service(mock_ariel_service: MagicMock) -> FastAPI:
        """FastAPI app with mocked ARIEL service injected.

        Creates a fresh app instance to avoid state pollution between tests.

        Args:
            mock_ariel_service: Mocked service fixture.

        Returns:
            FastAPI app with ariel_service in state.
        """
        from osprey.interfaces.ariel.api.routes import router as api_router

        # Create fresh app without lifespan (we inject the service manually)
        app = FastAPI(title="ARIEL Test App")
        app.include_router(api_router)

        # Inject mocked service
        app.state.ariel_service = mock_ariel_service

        # Add health endpoint (from main.py)
        @app.get("/health")
        async def health():
            if hasattr(app.state, "ariel_service"):
                healthy, message = await app.state.ariel_service.health_check()
                return {"status": "healthy" if healthy else "degraded", "message": message}
            return {"status": "starting", "message": "Service initializing"}

        return app

    @pytest.fixture
    async def client(app_with_mock_service: FastAPI) -> AsyncGenerator[AsyncClient, None]:
        """Async HTTP client for testing FastAPI app.

        Args:
            app_with_mock_service: FastAPI app fixture.

        Yields:
            AsyncClient configured for the test app.
        """
        transport = ASGITransport(app=app_with_mock_service)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
