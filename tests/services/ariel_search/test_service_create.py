"""Tests for ARIELSearchService.create_entry() orchestration."""

import logging
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


@pytest.mark.asyncio
async def test_create_entry_reingestion_failure_is_warned_not_raised(caplog):
    """A failing re-ingestion leaves the entry PENDING_SYNC and logs a warning.

    The facility write already succeeded at this point, so a read-back failure
    must not surface as an error -- the entry syncs on the next poll.
    """
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="ALS eLog",
    )

    async def failing_fetch(**kwargs):
        raise RuntimeError("logbook unreachable")
        yield  # Unreachable; makes this an async generator

    mock_adapter.fetch_entries = failing_fetch

    request = FacilityEntryCreateRequest(subject="Test", details="Details")

    with caplog.at_level(logging.WARNING, logger="ariel"):
        with patch(
            "osprey.services.ariel_search.ingestion.get_adapter",
            return_value=mock_adapter,
        ):
            result = await service.create_entry(request)

    assert result.sync_status == SyncStatus.PENDING_SYNC
    assert result.entry_id == "test-entry-001"
    assert "Re-ingestion after write failed for test-entry-001" in caplog.text
    assert "logbook unreachable" in caplog.text
    assert "sync on next poll" in caplog.text

    # Only the optimistic upsert landed; the sync upsert never ran.
    mock_repository.upsert_entry.assert_called_once()


@pytest.mark.asyncio
async def test_create_entry_mirrors_inline_when_qmd_export_enabled(tmp_path):
    """An enabled qmd_export mirrors the new entry at creation time.

    The optimistic upsert alone would leave the entry invisible to hybrid
    search until the next batch enhancement run; the inline mirror write is
    what makes an agent-created entry searchable within one sidecar poll.
    """
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.enhancement.qmd_export import TOUCH_MARKER_NAME
    from osprey.services.ariel_search.service import ARIELSearchService

    mirror = tmp_path / "mirror"
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": "/tmp/test.json"},
            "enhancement_modules": {
                "qmd_export": {"enabled": True, "settings": {"mirror_path": str(mirror)}},
            },
        }
    )
    service = ARIELSearchService(config=config, pool=MagicMock(), repository=AsyncMock())

    mock_adapter = AsyncMock()
    mock_adapter.supports_write = True
    mock_adapter.source_system_name = "Generic JSON"
    mock_adapter.create_entry = AsyncMock(return_value="inline-mirror-001")

    request = FacilityEntryCreateRequest(subject="Inline mirror", details="Body", author="tester")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(request)

    assert result.entry_id == "inline-mirror-001"
    written = list(mirror.rglob("*.md"))
    assert len(written) == 1
    assert "inline-mirror-001" in written[0].read_text()
    assert (mirror / TOUCH_MARKER_NAME).is_file()


@pytest.mark.asyncio
async def test_create_entry_mirror_failure_is_warned_not_raised(tmp_path, caplog):
    """A broken mirror config logs a warning and never fails the create.

    The entry is already durable in Postgres when the mirror write runs, so a
    misconfigured exporter (enabled, but no mirror_path) must degrade to a
    warning — the batch resync remains the backstop.
    """
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.service import ARIELSearchService

    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": "/tmp/test.json"},
            "enhancement_modules": {"qmd_export": {"enabled": True}},
        }
    )
    service = ARIELSearchService(config=config, pool=MagicMock(), repository=AsyncMock())

    mock_adapter = AsyncMock()
    mock_adapter.supports_write = True
    mock_adapter.source_system_name = "Generic JSON"
    mock_adapter.create_entry = AsyncMock(return_value="inline-mirror-002")

    request = FacilityEntryCreateRequest(subject="Broken mirror", details="Body")

    with caplog.at_level(logging.WARNING, logger="ariel"):
        with patch(
            "osprey.services.ariel_search.ingestion.get_adapter",
            return_value=mock_adapter,
        ):
            result = await service.create_entry(request)

    assert result.entry_id == "inline-mirror-002"
    assert "could not mirror entry 'inline-mirror-002' inline" in caplog.text


@pytest.mark.asyncio
async def test_create_entry_skips_mirror_when_qmd_export_disabled(tmp_path):
    """With qmd_export disabled the create writes no mirror files at all."""
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.service import ARIELSearchService

    mirror = tmp_path / "mirror"
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": "/tmp/test.json"},
            "enhancement_modules": {
                "qmd_export": {"enabled": False, "settings": {"mirror_path": str(mirror)}},
            },
        }
    )
    service = ARIELSearchService(config=config, pool=MagicMock(), repository=AsyncMock())

    mock_adapter = AsyncMock()
    mock_adapter.supports_write = True
    mock_adapter.source_system_name = "Generic JSON"
    mock_adapter.create_entry = AsyncMock(return_value="inline-mirror-003")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        await service.create_entry(FacilityEntryCreateRequest(subject="No mirror", details="x"))

    assert not mirror.exists()
