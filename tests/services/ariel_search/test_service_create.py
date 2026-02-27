"""Tests for ARIELSearchService.create_entry() orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.models import (
    FacilityEntryCreateRequest,
    FacilityEntryCreateResult,
    SyncStatus,
)


def _make_mock_service(adapter_supports_write: bool = True, source_system: str = "Generic JSON"):
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

    # Build mock adapter
    mock_adapter = AsyncMock()
    mock_adapter.supports_write = adapter_supports_write
    mock_adapter.source_system_name = source_system
    mock_adapter.create_entry = AsyncMock(return_value="test-entry-001")

    # For non-local adapters, mock fetch_entries to return empty
    async def empty_fetch(**kwargs):
        return
        yield  # Make it an async generator

    mock_adapter.fetch_entries = empty_fetch

    return service, mock_adapter, mock_repository


@pytest.mark.asyncio
async def test_create_entry_via_generic_adapter():
    """Full orchestration: adapter write + local upsert for Generic JSON."""
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="Generic JSON",
    )

    request = FacilityEntryCreateRequest(
        subject="Test entry",
        details="Test details",
        author="tester",
        tags=["test"],
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(request)

    assert isinstance(result, FacilityEntryCreateResult)
    assert result.entry_id == "test-entry-001"
    assert result.source_system == "Generic JSON"
    assert result.sync_status == SyncStatus.LOCAL_ONLY

    # Adapter was called
    mock_adapter.create_entry.assert_called_once_with(request)

    # Repository upsert was called for optimistic local insert
    mock_repository.upsert_entry.assert_called_once()
    upserted = mock_repository.upsert_entry.call_args[0][0]
    assert upserted["entry_id"] == "test-entry-001"
    assert upserted["source_system"] == "Generic JSON"


@pytest.mark.asyncio
async def test_create_entry_unsupported_adapter():
    """NotImplementedError when adapter doesn't support writes."""
    service, mock_adapter, _mock_repository = _make_mock_service(
        adapter_supports_write=False,
        source_system="JLab Logbook",
    )

    request = FacilityEntryCreateRequest(
        subject="Test",
        details="Details",
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        with pytest.raises(NotImplementedError, match="does not support"):
            await service.create_entry(request)


@pytest.mark.asyncio
async def test_create_entry_sync_status_local_only():
    """Generic JSON adapter gets LOCAL_ONLY sync status."""
    service, mock_adapter, _mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="Generic JSON",
    )

    request = FacilityEntryCreateRequest(subject="Test", details="Details")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(request)

    assert result.sync_status == SyncStatus.LOCAL_ONLY


@pytest.mark.asyncio
async def test_create_entry_sync_status_pending():
    """Non-local adapter (ALS) gets PENDING_SYNC when re-ingestion doesn't find entry."""
    service, mock_adapter, _mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="ALS eLog",
    )

    request = FacilityEntryCreateRequest(subject="Test", details="Details")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(request)

    assert result.sync_status == SyncStatus.PENDING_SYNC
    assert result.source_system == "ALS eLog"


@pytest.mark.asyncio
async def test_create_entry_sync_status_synced():
    """Non-local adapter gets SYNCED when re-ingestion finds the entry."""
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="ALS eLog",
    )

    # Mock fetch_entries to return the newly created entry
    fetched_entry = {
        "entry_id": "test-entry-001",
        "source_system": "ALS eLog",
        "timestamp": None,
        "author": "tester",
        "raw_text": "Test entry\n\nTest details",
        "attachments": [],
        "metadata": {},
        "created_at": None,
        "updated_at": None,
    }

    async def fetch_with_entry(**kwargs):
        yield fetched_entry

    mock_adapter.fetch_entries = fetch_with_entry

    request = FacilityEntryCreateRequest(subject="Test entry", details="Test details")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(request)

    assert result.sync_status == SyncStatus.SYNCED

    # Repository was called twice: once for optimistic, once for synced
    assert mock_repository.upsert_entry.call_count == 2
