"""Tests for ARIELSearchService.publish_entry() orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.models import (
    FacilityEntryCreateRequest,
    FacilityEntryCreateResult,
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

    mock_adapter = AsyncMock()
    mock_adapter.supports_write = adapter_supports_write
    mock_adapter.source_system_name = source_system
    mock_adapter.create_entry = AsyncMock(return_value="test-entry-001")

    async def empty_fetch(**kwargs):
        return
        yield  # Make it an async generator

    mock_adapter.fetch_entries = empty_fetch

    return service, mock_adapter, mock_repository


@pytest.mark.asyncio
async def test_publish_entry_happy_path():
    """publish_entry extracts fields from stored entry and delegates to create_entry."""
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="Generic JSON",
    )

    mock_repository.get_entry = AsyncMock(
        return_value={
            "raw_text": "First line subject\nRemaining details",
            "author": "tester",
            "metadata": {"tags": ["beam", "ops"]},
        }
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.publish_entry("entry-123")

    assert isinstance(result, FacilityEntryCreateResult)
    assert result.entry_id == "test-entry-001"

    # Verify the request passed to adapter.create_entry
    request = mock_adapter.create_entry.call_args[0][0]
    assert isinstance(request, FacilityEntryCreateRequest)
    assert request.subject == "First line subject"
    assert request.details == "First line subject\nRemaining details"
    assert request.author == "tester"
    assert request.tags == ["beam", "ops"]


@pytest.mark.asyncio
async def test_publish_entry_not_found():
    """KeyError when entry_id doesn't exist in the repository."""
    service, _mock_adapter, mock_repository = _make_mock_service()

    mock_repository.get_entry = AsyncMock(return_value=None)

    with pytest.raises(KeyError, match="entry-999"):
        await service.publish_entry("entry-999")


@pytest.mark.asyncio
async def test_publish_entry_adapter_not_supported():
    """NotImplementedError when adapter doesn't support writes."""
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=False,
        source_system="JLab Logbook",
    )

    mock_repository.get_entry = AsyncMock(
        return_value={
            "raw_text": "Some text",
            "author": "tester",
            "metadata": {"tags": []},
        }
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        with pytest.raises(NotImplementedError, match="does not support"):
            await service.publish_entry("entry-123")


@pytest.mark.asyncio
async def test_publish_entry_subject_extraction_single_line():
    """Single-line raw_text: subject and details are the same string."""
    service, mock_adapter, mock_repository = _make_mock_service()

    mock_repository.get_entry = AsyncMock(
        return_value={
            "raw_text": "Just a subject",
            "author": "tester",
            "metadata": {"tags": []},
        }
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        await service.publish_entry("entry-123")

    request = mock_adapter.create_entry.call_args[0][0]
    assert request.subject == "Just a subject"
    assert request.details == "Just a subject"


@pytest.mark.asyncio
async def test_publish_entry_logbook_passthrough():
    """logbook kwarg is passed through to FacilityEntryCreateRequest."""
    service, mock_adapter, mock_repository = _make_mock_service()

    mock_repository.get_entry = AsyncMock(
        return_value={
            "raw_text": "Test entry",
            "author": "tester",
            "metadata": {"tags": []},
        }
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        await service.publish_entry("entry-123", logbook="Operations")

    request = mock_adapter.create_entry.call_args[0][0]
    assert request.logbook == "Operations"
