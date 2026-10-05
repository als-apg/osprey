"""Tests for ARIELSearchService.publish_entry() orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.entry_fields import EntryFieldError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.models import (
    FacilityEntryCreateRequest,
    FacilityEntryCreateResult,
)
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    dict_repository_fixture,
    example_config,
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

    mock_adapter = AsyncMock(spec=FacilityAdapter)
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


@pytest.mark.asyncio
async def test_publish_entry_no_declarations_request_pinned():
    """With no declared fields the request is exactly the one built without entry fields."""
    service, mock_adapter, mock_repository = _make_mock_service()

    mock_repository.get_entry = AsyncMock(
        return_value={
            "raw_text": "Subject\nBody",
            "author": "tester",
            "metadata": {"logbook": "Stored", "shift": "Day", "tags": ["rf"]},
        }
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        await service.publish_entry("entry-123", logbook="Operations")

    request = mock_adapter.create_entry.call_args[0][0]
    assert request == FacilityEntryCreateRequest(
        subject="Subject",
        details="Subject\nBody",
        author="tester",
        logbook="Operations",
        tags=["rf"],
    )
    assert request.shift is None
    assert request.metadata == {}


@pytest.mark.asyncio
async def test_publish_entry_no_declarations_refuses_fields():
    """Fields given while none are declared are refused, naming the field."""
    service, mock_adapter, mock_repository = _make_mock_service()
    mock_repository.get_entry = AsyncMock(
        return_value={"raw_text": "x", "author": "tester", "metadata": {"tags": []}}
    )

    with patch(
        "osprey.services.ariel_search.ingestion.get_adapter",
        return_value=mock_adapter,
    ):
        with pytest.raises(EntryFieldError) as excinfo:
            await service.publish_entry("entry-123", fields={"book": "ops"})

    assert excinfo.value.field == "book"
    mock_adapter.create_entry.assert_not_called()


# --- Declared entry fields, through the example adapter ----------------------


def _example_service(repository):
    from osprey.services.ariel_search.service import ARIELSearchService

    return ARIELSearchService(config=example_config(), pool=MagicMock(), repository=repository)


def _seed(repository, metadata, entry_id="e1"):
    repository.entries[entry_id] = {
        "entry_id": entry_id,
        "raw_text": "Beam lost\nRF trip",
        "author": "op",
        "metadata": metadata,
    }


@pytest.mark.asyncio
async def test_publish_missing_required_field_names_it(example_entry_fields, dict_repository):
    """A stored entry without the required ``book`` is refused, naming ``book``."""
    _seed(dict_repository, {"tags": ["rf"]})

    with pytest.raises(EntryFieldError) as excinfo:
        await _example_service(dict_repository).publish_entry("e1")

    assert excinfo.value.field == "book"
    assert example_entry_fields.state.created == []


@pytest.mark.asyncio
async def test_publish_fields_fill_missing_required(example_entry_fields, dict_repository):
    """The same entry published with ``fields={"book": "ops"}`` reaches the adapter."""
    _seed(dict_repository, {"tags": ["rf"]})

    result = await _example_service(dict_repository).publish_entry("e1", fields={"book": "ops"})

    assert result.entry_id == "example-1"
    sent = example_entry_fields.state.created[-1]
    assert sent.metadata == {"book": "ops"}
    assert sent.subject == "Beam lost"
    assert sent.tags == ["rf"]


@pytest.mark.asyncio
async def test_publish_fields_override_stored(example_entry_fields, dict_repository):
    """A ``fields`` value wins over the stored declared value."""
    _seed(dict_repository, {"book": "ops", "tags": []})

    await _example_service(dict_repository).publish_entry("e1", fields={"book": "physics"})

    assert example_entry_fields.state.created[-1].metadata == {"book": "physics"}


@pytest.mark.asyncio
async def test_publish_stored_declared_values_carried(example_entry_fields, dict_repository):
    """Stored declared values are sent; ARIEL's own keys are not."""
    _seed(
        dict_repository,
        {
            "book": "physics",
            "day": "2026-10-01",
            "tags": ["rf"],
            "created_via": "ariel-web",
            "sync_status": "local_only",
        },
    )

    await _example_service(dict_repository).publish_entry("e1")

    sent = example_entry_fields.state.created[-1]
    assert sent.metadata == {"book": "physics", "day": "2026-10-01"}


@pytest.mark.asyncio
async def test_publish_unknown_field_refused(example_entry_fields, dict_repository):
    """A ``fields`` key that is not declared is refused, naming it."""
    _seed(dict_repository, {"book": "ops", "tags": []})

    with pytest.raises(EntryFieldError) as excinfo:
        await _example_service(dict_repository).publish_entry("e1", fields={"bogus": "x"})

    assert excinfo.value.field == "bogus"
    assert example_entry_fields.state.created == []


@pytest.mark.asyncio
async def test_publish_invalid_dynamic_value_checked_live(example_entry_fields, dict_repository):
    """A stored dynamic value no longer offered is refused after a live check."""
    example_entry_fields.state.options_table = {"2026-10-01": [{"value": "s1", "label": "S1"}]}
    _seed(dict_repository, {"book": "ops", "day": "2026-10-01", "scan": "s9", "tags": []})

    with pytest.raises(EntryFieldError) as excinfo:
        await _example_service(dict_repository).publish_entry("e1")

    assert excinfo.value.field == "scan"
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-01"})]


@pytest.mark.asyncio
async def test_publish_logbook_argument_wins(example_entry_fields, dict_repository):
    """Stored declared logbook A with argument B: the adapter sees B in both places."""
    example_entry_fields.state.declare_logbook = True
    _seed(dict_repository, {"book": "ops", "logbook": "control-room", "tags": []})

    await _example_service(dict_repository).publish_entry("e1", logbook="maintenance")

    sent = example_entry_fields.state.created[-1]
    assert sent.logbook == "maintenance"
    assert sent.metadata["logbook"] == "maintenance"


@pytest.mark.asyncio
async def test_publish_logbook_argument_validated(example_entry_fields, dict_repository):
    """A logbook argument that is not a declared choice is refused, naming ``logbook``."""
    example_entry_fields.state.declare_logbook = True
    _seed(dict_repository, {"book": "ops", "logbook": "control-room", "tags": []})

    with pytest.raises(EntryFieldError) as excinfo:
        await _example_service(dict_repository).publish_entry("e1", logbook="nowhere")

    assert excinfo.value.field == "logbook"
    assert example_entry_fields.state.created == []


@pytest.mark.asyncio
async def test_publish_logbook_argument_undeclared(example_entry_fields, dict_repository):
    """With ``logbook`` undeclared, the argument goes to the request and metadata."""
    _seed(dict_repository, {"book": "ops", "logbook": "stored", "tags": []})

    await _example_service(dict_repository).publish_entry("e1", logbook="ops-book")

    sent = example_entry_fields.state.created[-1]
    assert sent.logbook == "ops-book"
    assert sent.metadata == {"book": "ops", "logbook": "ops-book"}


@pytest.mark.asyncio
async def test_publish_read_only_refused_without_options_calls(
    example_entry_fields, dict_repository
):
    """A read-only adapter refuses before any field check or options call."""
    example_entry_fields.state.supports_write = False
    example_entry_fields.state.options_table = {"2026-10-01": [{"value": "s1", "label": "S1"}]}
    _seed(dict_repository, {"book": "ops", "day": "2026-10-01", "scan": "s1", "tags": []})

    with pytest.raises(NotImplementedError, match="does not support"):
        await _example_service(dict_repository).publish_entry("e1", fields={"bogus": "x"})

    assert example_entry_fields.state.options_calls == []
    assert example_entry_fields.state.created == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_entry_fields")
async def test_publish_local_copy_keeps_declared_and_provenance(dict_repository):
    """The optimistic copy keeps declared values and the stored provenance."""
    session = {"session_id": "s-1", "user": "op"}
    _seed(
        dict_repository,
        {
            "book": "ops",
            "tags": ["rf"],
            "created_via": "ariel-mcp",
            "session_metadata": session,
        },
    )

    result = await _example_service(dict_repository).publish_entry(
        "e1", fields={"book": "physics", "day": "2026-10-02"}
    )

    local = dict_repository.entries[result.entry_id]["metadata"]
    assert local["book"] == "physics"
    assert local["day"] == "2026-10-02"
    assert local["created_via"] == "ariel-mcp"
    assert local["session_metadata"] == session
    assert local["tags"] == ["rf"]
    assert local["sync_status"] == "pending_sync"


@pytest.mark.asyncio
@pytest.mark.usefixtures("example_entry_fields")
async def test_publish_local_copy_without_stored_provenance(dict_repository):
    """An entry with no stored ``created_via`` gains none in its published copy."""
    _seed(dict_repository, {"book": "ops", "tags": []})

    result = await _example_service(dict_repository).publish_entry("e1")

    local = dict_repository.entries[result.entry_id]["metadata"]
    assert "created_via" not in local
    assert "session_metadata" not in local
    assert local["book"] == "ops"
