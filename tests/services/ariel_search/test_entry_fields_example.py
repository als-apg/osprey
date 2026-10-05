"""Tests for the shared example entry-fields adapter and its dict repository."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import MagicMock

import pytest

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.models import FacilityEntryCreateRequest, SyncStatus
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    EXAMPLE_SOURCE_SYSTEM,
    OPTIONS_FAIL,
    OPTIONS_HANG,
    DictRepository,
    ExampleEntryFieldsAdapter,
    dict_repository_fixture,
    example_config,
    example_entry_fields_fixture,
)

RESERVED_NAMES = {"tags", "sync_status", "created_via", "session_metadata", "title"}


def _service(repository):
    from osprey.services.ariel_search.service import ARIELSearchService

    return ARIELSearchService(config=example_config(), pool=MagicMock(), repository=repository)


def test_patch_returns_shared_instance(example_entry_fields):
    """get_adapter looked up at call time returns the one adapter of the test."""
    from osprey.services.ariel_search.ingestion import get_adapter

    assert get_adapter(example_config()) is example_entry_fields
    assert get_adapter(example_config()) is example_entry_fields


def test_descriptors_shape(example_entry_fields):
    """book is a required select, day a date, scan a dynamic_select on day."""
    by_name = {d.name: d.to_dict() for d in example_entry_fields.get_entry_field_descriptors()}
    assert list(by_name) == ["book", "day", "scan"]
    assert by_name["book"]["type"] == "select"
    assert by_name["book"]["required"] is True
    assert [o["value"] for o in by_name["book"]["options"]] == ["ops", "physics"]
    assert by_name["day"]["type"] == "date"
    assert "required" not in by_name["day"]
    assert by_name["scan"]["type"] == "dynamic_select"
    assert by_name["scan"]["depends_on"] == ["day"]
    assert "options_endpoint" not in by_name["scan"]
    assert not RESERVED_NAMES & set(by_name)


def test_descriptors_fresh_list(example_entry_fields):
    """A caller mutating the declared list cannot change the next answer."""
    first = example_entry_fields.get_entry_field_descriptors()
    first.clear()
    assert len(example_entry_fields.get_entry_field_descriptors()) == 3


def test_logbook_variant(example_entry_fields):
    """The logbook toggle appends a logbook select after the example fields."""
    example_entry_fields.state.declare_logbook = True
    descriptors = example_entry_fields.get_entry_field_descriptors()
    assert [d.name for d in descriptors] == ["book", "day", "scan", "logbook"]
    assert descriptors[-1].param_type == "select"
    assert descriptors[-1].options


@pytest.mark.asyncio
async def test_options_from_table_recorded(example_entry_fields):
    """Options come from the per-test table by day, and every call is recorded."""
    day = datetime.date(2026, 10, 4)
    example_entry_fields.state.options_table[day] = [{"value": "s1", "label": "Scan 1"}]
    assert await example_entry_fields.get_entry_field_options("scan", {"day": day}) == [
        {"value": "s1", "label": "Scan 1"}
    ]
    other = datetime.date(2026, 10, 5)
    assert await example_entry_fields.get_entry_field_options("scan", {"day": other}) == []
    assert example_entry_fields.state.options_calls == [
        ("scan", {"day": day}),
        ("scan", {"day": other}),
    ]


@pytest.mark.asyncio
async def test_options_fail_toggle(example_entry_fields):
    """The fail toggle makes the options call raise IngestionError."""
    example_entry_fields.state.options_mode = OPTIONS_FAIL
    with pytest.raises(IngestionError):
        await example_entry_fields.get_entry_field_options("scan", {"day": None})
    assert len(example_entry_fields.state.options_calls) == 1


@pytest.mark.asyncio
async def test_options_hang_toggle(example_entry_fields):
    """The hang toggle makes the options call never return."""
    example_entry_fields.state.options_mode = OPTIONS_HANG
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            example_entry_fields.get_entry_field_options("scan", {"day": None}), timeout=0.05
        )


@pytest.mark.asyncio
async def test_create_entry_records_request(example_entry_fields):
    """create_entry records the request and returns a fresh id each time."""
    first = FacilityEntryCreateRequest(subject="a", details="b", metadata={"book": "ops"})
    second = FacilityEntryCreateRequest(subject="c", details="d")
    assert await example_entry_fields.create_entry(first) == "example-1"
    assert await example_entry_fields.create_entry(second) == "example-2"
    assert example_entry_fields.state.created == [first, second]


@pytest.mark.asyncio
async def test_supports_write_toggle(example_entry_fields, dict_repository):
    """With writes off the service refuses and nothing is recorded."""
    example_entry_fields.state.supports_write = False
    assert example_entry_fields.supports_write is False
    with pytest.raises(NotImplementedError):
        await _service(dict_repository).create_entry(
            FacilityEntryCreateRequest(subject="s", details="d")
        )
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}


def test_state_is_per_test(example_entry_fields):
    """Each test starts from a fresh state with the defaults."""
    state = example_entry_fields.state
    assert state.supports_write is True
    assert state.declare_logbook is False
    assert state.options_table == {}
    assert state.created == []
    assert state.options_calls == []


@pytest.mark.asyncio
async def test_dict_repository_round_trip():
    """An upserted entry reads back by id; an unknown id reads back as None."""
    repo = DictRepository()
    await repo.upsert_entry({"entry_id": "e1", "raw_text": "x"})
    assert await repo.get_entry("e1") == {"entry_id": "e1", "raw_text": "x"}
    assert await repo.get_entry("missing") is None


@pytest.mark.asyncio
async def test_write_then_publish_through_services(example_entry_fields, dict_repository):
    """A write through one service is read back and published by another."""
    request = FacilityEntryCreateRequest(
        subject="Beam lost", details="RF trip", author="op", tags=["rf"]
    )
    result = await _service(dict_repository).create_entry(request)
    assert result.entry_id == "example-1"
    assert result.source_system == EXAMPLE_SOURCE_SYSTEM
    assert result.sync_status == SyncStatus.PENDING_SYNC
    assert dict_repository.entries["example-1"]["author"] == "op"

    from osprey.services.ariel_search.entry_fields import EntryFieldError

    with pytest.raises(EntryFieldError, match="Book is required"):
        await _service(dict_repository).publish_entry("example-1", logbook="ops")

    published = await _service(dict_repository).publish_entry(
        "example-1", logbook="ops", fields={"book": "ops"}
    )
    assert published.entry_id == "example-2"
    sent = example_entry_fields.state.created[-1]
    assert sent.subject == "Beam lost"
    assert sent.author == "op"
    assert sent.tags == ["rf"]
    assert sent.logbook == "ops"
    assert sent.metadata["book"] == "ops"


def test_adapter_constructible_without_fixture():
    """The adapter builds standalone with a fresh state of its own."""
    adapter = ExampleEntryFieldsAdapter(example_config())
    assert adapter.source_system_name == EXAMPLE_SOURCE_SYSTEM
    assert adapter.state.created == []
