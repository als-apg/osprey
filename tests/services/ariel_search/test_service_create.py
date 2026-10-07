"""Tests for ARIELSearchService.create_entry() orchestration."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.entry_fields import resolve_entry_write
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.models import (
    FacilityEntryCreateRequest,
    FacilityEntryCreateResult,
    SyncStatus,
)
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    EXAMPLE_SOURCE_SYSTEM,
    dict_repository_fixture,
    example_config,
    example_descriptors,
    example_entry_fields_fixture,
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
    mock_adapter = AsyncMock(spec=FacilityAdapter)
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
    """A non-local adapter gets PENDING_SYNC when re-ingestion doesn't find the entry."""
    service, mock_adapter, _mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="Example eLog",
    )

    request = FacilityEntryCreateRequest(subject="Test", details="Details")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(request)

    assert result.sync_status == SyncStatus.PENDING_SYNC
    assert result.source_system == "Example eLog"


@pytest.mark.asyncio
async def test_create_entry_sync_status_synced():
    """Non-local adapter gets SYNCED when re-ingestion finds the entry."""
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="Example eLog",
    )

    # Mock fetch_entries to return the newly created entry
    fetched_entry = {
        "entry_id": "test-entry-001",
        "source_system": "Example eLog",
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
        source_system="Example eLog",
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

    mock_adapter = AsyncMock(spec=FacilityAdapter)
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
async def test_create_entry_mirror_failure_is_warned_not_raised(caplog):
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

    mock_adapter = AsyncMock(spec=FacilityAdapter)
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

    mock_adapter = AsyncMock(spec=FacilityAdapter)
    mock_adapter.supports_write = True
    mock_adapter.source_system_name = "Generic JSON"
    mock_adapter.create_entry = AsyncMock(return_value="inline-mirror-003")

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        await service.create_entry(FacilityEntryCreateRequest(subject="No mirror", details="x"))

    assert not mirror.exists()


def _example_service(repository):
    from osprey.services.ariel_search.service import ARIELSearchService

    return ARIELSearchService(config=example_config(), pool=MagicMock(), repository=repository)


@pytest.mark.asyncio
async def test_create_entry_keeps_declared_date_in_local_copy(
    example_entry_fields, dict_repository
):
    """A declared date value lands in the optimistic copy as its ISO string."""
    resolved = resolve_entry_write(
        example_descriptors(),
        {"book": "ops", "day": "2026-10-04"},
        logbook=None,
        shift=None,
        tags=["rf"],
        created_via="ariel-web",
    )
    request = FacilityEntryCreateRequest(
        subject="Beam lost",
        details="RF trip",
        author="op",
        tags=["rf"],
        metadata=resolved.adapter_metadata,
    )

    result = await _example_service(dict_repository).create_entry(
        request, local_metadata=resolved.local_metadata
    )

    assert result.entry_id == "example-1"
    assert result.sync_status == SyncStatus.PENDING_SYNC
    stored = dict_repository.entries["example-1"]["metadata"]
    assert stored["day"] == "2026-10-04"
    assert stored["book"] == "ops"
    assert stored["tags"] == ["rf"]
    assert stored["created_via"] == "ariel-web"
    assert stored["sync_status"] == SyncStatus.PENDING_SYNC.value
    # The adapter receives the request as built, with adapter metadata only.
    assert example_entry_fields.state.created == [request]
    assert "created_via" not in example_entry_fields.state.created[0].metadata


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_entry_fields")
async def test_create_entry_keeps_provenance_on_writable_adapter(dict_repository):
    """session_metadata and created_via survive a write through a writable adapter."""
    session = {"session_id": "s-42", "agent": "logbook"}
    resolved = resolve_entry_write(
        example_descriptors(),
        {"book": "physics"},
        logbook="control-room",
        shift="day",
        tags=[],
        created_via="ariel-mcp",
        session_metadata=session,
    )
    request = FacilityEntryCreateRequest(
        subject="Orbit drift",
        details="",
        logbook=resolved.logbook,
        shift=resolved.shift,
        metadata=resolved.adapter_metadata,
    )

    await _example_service(dict_repository).create_entry(
        request, local_metadata=resolved.local_metadata
    )

    stored = dict_repository.entries["example-1"]["metadata"]
    assert stored["session_metadata"] == session
    assert stored["created_via"] == "ariel-mcp"
    assert stored["logbook"] == "control-room"
    assert stored["shift"] == "day"
    assert stored["book"] == "physics"
    assert dict_repository.entries["example-1"]["source_system"] == EXAMPLE_SOURCE_SYSTEM


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_entry_fields")
async def test_create_entry_service_sets_sync_status_over_local_metadata(dict_repository):
    """A sync_status in the local metadata never overrides the service's own."""
    local = {"created_via": "ariel-web", "sync_status": "synced"}

    await _example_service(dict_repository).create_entry(
        FacilityEntryCreateRequest(subject="x", details="y"), local_metadata=local
    )

    stored = dict_repository.entries["example-1"]["metadata"]
    assert stored["sync_status"] == SyncStatus.PENDING_SYNC.value
    # The caller's dict is copied, not mutated.
    assert local == {"created_via": "ariel-web", "sync_status": "synced"}
    assert stored is not local


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_entry_fields")
async def test_create_entry_without_local_metadata_keeps_default_copy(dict_repository):
    """With no local metadata the optimistic copy holds logbook, shift, tags, sync_status."""
    request = FacilityEntryCreateRequest(
        subject="x", details="y", logbook="ops", shift="night", tags=["a"]
    )

    await _example_service(dict_repository).create_entry(request)

    assert dict_repository.entries["example-1"]["metadata"] == {
        "logbook": "ops",
        "shift": "night",
        "tags": ["a"],
        "sync_status": SyncStatus.PENDING_SYNC.value,
    }


@pytest.mark.asyncio
async def test_create_entry_synced_copy_is_facility_record_not_merged():
    """After re-ingest finds the entry, the facility record replaces the local copy."""
    service, mock_adapter, mock_repository = _make_mock_service(
        adapter_supports_write=True,
        source_system="Example eLog",
    )
    fetched_entry = {
        "entry_id": "test-entry-001",
        "source_system": "Example eLog",
        "timestamp": None,
        "author": "tester",
        "raw_text": "Test entry",
        "attachments": [],
        "metadata": {"facility": "yes"},
        "created_at": None,
        "updated_at": None,
    }

    async def fetch_with_entry(**kwargs):
        yield fetched_entry

    mock_adapter.fetch_entries = fetch_with_entry

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        result = await service.create_entry(
            FacilityEntryCreateRequest(subject="Test entry", details=""),
            local_metadata={"created_via": "ariel-web", "session_metadata": {"s": 1}},
        )

    assert result.sync_status == SyncStatus.SYNCED
    optimistic = mock_repository.upsert_entry.call_args_list[0][0][0]
    assert optimistic["metadata"]["created_via"] == "ariel-web"
    synced = mock_repository.upsert_entry.call_args_list[1][0][0]
    assert synced["metadata"] == {"facility": "yes"}
