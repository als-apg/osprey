"""A scenario's ``logbook`` block, read as the entries it narrates."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest

from osprey.facility.scenarios import ScenarioLogEntry, scenario_logbook
from osprey_connectors.relative_time import RelativeTimestamp

FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"


def _entry(**fields) -> dict:
    entry = {
        "entry_id": "E-1",
        "when": {"days_ago": 2, "time": "03:20:00"},
        "author": "A. Author",
        "title": "Title",
        "text": "Body",
    }
    entry.update(fields)
    return entry


def test_a_logbook_block_reads_as_its_entries_in_order():
    scenario = {
        "name": "burst",
        "logbook": [
            _entry(tags=["rf"], categories=["Operations"], loto_tag="L-7", extra={"shift": "owl"}),
            _entry(entry_id="E-2"),
        ],
    }

    first, second = scenario_logbook(scenario)

    assert first == ScenarioLogEntry(
        entry_id="E-1",
        when=RelativeTimestamp(days_ago=2, time=time(3, 20)),
        author="A. Author",
        title="Title",
        text="Body",
        tags=("rf",),
        categories=("Operations",),
        loto_tag="L-7",
        extra={"shift": "owl"},
    )
    assert second.entry_id == "E-2"
    assert second.tags == ()
    assert second.loto_tag is None


def test_a_scenario_without_a_logbook_narrates_nothing():
    assert scenario_logbook({"name": "quiet"}) == ()


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"entry_id": ""}, "'entry_id'"),
        ({"when": {"days_ago": -1, "time": "03:20:00"}}, "'days_ago'"),
        ({"when": {"days_ago": 1, "time": 12000}}, "'when.time'"),
        ({"when": {"days_ago": 1, "time": "03:20:00+01:00"}}, "timezone"),
        ({"title": 3}, "'title'"),
        ({"tags": "rf"}, "'tags'"),
        ({"loto_tag": 7}, "'loto_tag'"),
        ({"extra": []}, "'extra'"),
    ],
)
def test_a_malformed_entry_is_refused_by_name(fields, match):
    with pytest.raises(ValueError, match=match) as raised:
        scenario_logbook({"name": "burst", "logbook": [_entry(**fields)]})

    assert "'burst'" in str(raised.value)


def test_an_entrys_pictures_resolve_against_the_scenario_directory(tmp_path):
    picture = tmp_path / "plots" / "trend.png"
    picture.parent.mkdir()
    picture.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
    scenario = {"name": "burst", "logbook": [_entry(attachments=[{"path": "plots/trend.png"}])]}

    (entry,) = scenario_logbook(scenario, tmp_path)

    assert entry.attachments == (picture.resolve(),)


def test_an_entry_attaching_a_picture_needs_the_scenario_directory():
    scenario = {"name": "burst", "logbook": [_entry(attachments=[{"path": "plots/trend.png"}])]}

    with pytest.raises(ValueError, match="'attachments'") as raised:
        scenario_logbook(scenario)

    assert "'burst'" in str(raised.value)


def test_an_attachment_outside_the_scenario_directory_is_refused(tmp_path):
    scenario = {"name": "burst", "logbook": [_entry(attachments=[{"path": "../trend.png"}])]}

    with pytest.raises(ValueError, match="relative to the scenario directory"):
        scenario_logbook(scenario, tmp_path)
