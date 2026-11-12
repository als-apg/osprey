"""The channel directory's records and its query."""

from __future__ import annotations

import dataclasses

import pytest

from osprey_connectors.directory import (
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
)
from osprey_connectors.service_answers import Outcome, Source

SOURCE = Source(kind="directory", target="va", service="simulator_view", rehearsal=False)


@pytest.mark.parametrize(
    ("record", "fields"),
    [
        (
            DirectoryRow,
            [
                "name",
                "status",
                "description",
                "record_type",
                "ioc",
                "host",
                "last_announced",
            ],
        ),
        (DirectoryRecord, ["row", "properties", "tags"]),
        (
            DirectoryQuery,
            [
                "words",
                "name",
                "properties",
                "status",
                "ioc",
                "record_type",
                "page",
                "page_size",
            ],
        ),
        (
            DirectoryPage,
            [
                "outcome",
                "searched",
                "request",
                "source",
                "total",
                "rows",
                "truncated",
                "partial",
                "facets",
                "facets_reason",
                "reason",
            ],
        ),
    ],
)
def test_a_record_has_the_contract_fields_in_order(record, fields):
    assert [f.name for f in dataclasses.fields(record)] == fields


def test_neither_a_row_nor_a_query_carries_an_origin():
    assert "origin" not in {field.name for field in dataclasses.fields(DirectoryRow)}
    assert "origin" not in {field.name for field in dataclasses.fields(DirectoryQuery)}


def test_the_directory_names_no_backends_property_map():
    import osprey_connectors.directory as directory

    assert not {"ROW_FIELD_PROPERTIES", "DEFAULT_DESCRIPTION_PROPERTY"} & set(dir(directory))
    assert "SERVED_SIGNAL_PROPERTY" not in dir(directory)


def test_a_page_of_any_other_outcome_needs_no_reason():
    page = DirectoryPage(outcome=Outcome.NO_MATCH, searched="q", request="r", source=SOURCE)

    assert (page.rows, page.partial, page.total, page.facets) == ([], [], None, None)


def test_a_record_converts_to_the_json_shape():
    record = DirectoryRecord(row=DirectoryRow(name="A"), properties={"caPort": "5064"})

    assert dataclasses.asdict(record)["row"]["name"] == "A"
    assert dataclasses.asdict(record)["properties"] == {"caPort": "5064"}


# -- DirectoryQuery keys a cache ----------------------------------------------


def test_equal_queries_hash_alike_whatever_the_filter_spelling():
    first = DirectoryQuery(words="bpm", ioc=["a", "b"], properties={"x": "1"})
    second = DirectoryQuery(words="bpm", ioc=("a", "b"), properties={"x": "1"})

    assert first == second
    assert hash(first) == hash(second)


def test_queries_that_differ_only_in_properties_are_different_keys():
    first = DirectoryQuery(properties={"x": "1"})
    second = DirectoryQuery(properties={"x": "2"})

    assert first != second
    assert len({first, second}) == 2


@pytest.mark.parametrize("name", ["ioc", "record_type"])
def test_one_filter_value_is_one_value_not_its_letters(name):
    assert getattr(DirectoryQuery(**{name: "abc"}), name) == ("abc",)
    assert getattr(DirectoryQuery(**{name: None}), name) == ()
