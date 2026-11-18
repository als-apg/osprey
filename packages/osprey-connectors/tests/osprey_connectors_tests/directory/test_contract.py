"""The channel directory's records and its query."""

from __future__ import annotations

import pytest

from osprey_connectors.directory import (
    DirectoryDetail,
    DirectoryEntry,
    DirectoryPage,
    DirectoryQuery,
)
from osprey_connectors.service_answers import Source

SOURCE = Source(kind="directory", target="va", service="simulator_view", rehearsal=False)


@pytest.mark.parametrize(
    ("record", "fields"),
    [
        (
            DirectoryEntry,
            [
                "address",
                "status",
                "description",
                "channel_type",
                "server",
                "host",
                "announced_at",
                "recorded_stamp",
            ],
        ),
        (DirectoryDetail, ["entry", "properties", "tags"]),
        (
            DirectoryQuery,
            [
                "words",
                "address_pattern",
                "properties",
                "status",
                "server",
                "channel_type",
                "page",
                "page_size",
            ],
        ),
        (
            DirectoryPage,
            ["outcome", "source", "rows", "truncated", "note", "searched", "request", "total"],
        ),
    ],
)
def test_a_record_has_the_contract_fields_in_order(record, fields):
    assert [*record.model_fields] == fields


def test_neither_a_row_nor_a_query_carries_an_origin():
    assert "origin" not in DirectoryEntry.model_fields
    assert "origin" not in DirectoryQuery.model_fields


def test_the_directory_names_no_backends_property_map():
    import osprey_connectors.directory as directory

    assert not {"ROW_FIELD_PROPERTIES", "DEFAULT_DESCRIPTION_PROPERTY"} & set(dir(directory))
    assert "SERVED_SIGNAL_PROPERTY" not in dir(directory)


def test_a_page_carries_rows_and_its_count_and_no_framework_fields():
    page = DirectoryPage(outcome="no_match", searched="q", request="r", source=SOURCE, total=0)

    assert (page.rows, page.total, page.note) == ((), 0, None)
    assert not {"partial", "facets", "facets_reason", "reason"} & set(DirectoryPage.model_fields)


def test_a_record_converts_to_the_json_shape():
    record = DirectoryDetail(entry=DirectoryEntry(address="A"), properties={"caPort": "5064"})

    assert record.model_dump()["entry"]["address"] == "A"
    assert record.model_dump()["properties"] == {"caPort": "5064"}


# -- DirectoryQuery keys a cache ----------------------------------------------


def test_equal_queries_hash_alike_whatever_the_filter_spelling():
    first = DirectoryQuery(words="bpm", server=["a", "b"], properties={"x": "1"})
    second = DirectoryQuery(words="bpm", server=("a", "b"), properties={"x": "1"})

    assert first == second
    assert hash(first) == hash(second)


def test_queries_that_differ_only_in_properties_are_different_keys():
    first = DirectoryQuery(properties={"x": "1"})
    second = DirectoryQuery(properties={"x": "2"})

    assert first != second
    assert len({first, second}) == 2


@pytest.mark.parametrize("name", ["server", "channel_type"])
def test_one_filter_value_is_one_value_not_its_letters(name):
    assert getattr(DirectoryQuery(**{name: "abc"}), name) == ("abc",)
    assert getattr(DirectoryQuery(**{name: None}), name) == ()
