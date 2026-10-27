"""An ``ARIELSearchService`` over mocks, for the create and publish orchestration suites."""

from unittest.mock import AsyncMock, MagicMock

from osprey.services.ariel_search.ingestion.base import FacilityAdapter


def make_mock_service(adapter_supports_write: bool = True, source_system: str = "Generic JSON"):
    """Build a mock ARIELSearchService with mocked adapter and repository."""
    from osprey.services.ariel_search.config import ARIELConfig

    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": "/tmp/test.json"},
        }
    )

    mock_pool = MagicMock()
    mock_repository = AsyncMock()
    mock_repository.upsert_entry = AsyncMock()

    from osprey.services.ariel_search.service import ARIELSearchService

    service = ARIELSearchService(config=config, pool=mock_pool, repository=mock_repository)

    mock_adapter = AsyncMock(spec=FacilityAdapter)
    mock_adapter.supports_write = adapter_supports_write
    mock_adapter.source_system_name = source_system
    mock_adapter.create_entry = AsyncMock(return_value="test-entry-001")

    async def empty_fetch(**kwargs):
        return
        yield  # Make it an async generator

    mock_adapter.fetch_entries = empty_fetch

    return service, mock_adapter, mock_repository
