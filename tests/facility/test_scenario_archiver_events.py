"""A scenario's archiver events, checked at build time.

Each event of an ``archiver`` entry has a known shape, that shape's keys,
exactly one position key, a number wherever one is read, and a shape other
than ``step`` only on a ``float`` channel. A malformed event stops the build
with one line naming the scenario file and the channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.scenarios import check_scenario_events
from tests.facility._synthetic_trees import plain_tree, write_tree

CHANNELS = [
    {"id": "SR:X", "value_type": "float"},
    {"id": "SR:ON", "value_type": "bool", "options": ["OFF", "ON"]},
]


def _document(*events: Any, channel: str = "SR:X") -> dict[str, Any]:
    archiver = [{"channel": channel, "events": list(events)}]
    return {"channels": CHANNELS, "scenarios": [{"name": "warm", "archiver": archiver}]}


def _line(document: dict[str, Any]) -> str:
    (error,) = check_scenario_events(document)
    return error.format_message()


@pytest.mark.parametrize(
    ("event", "channel", "line"),
    [
        (
            {"shape": "wave", "at": 0.5, "to": 1.0},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X has shape 'wave', not one of `ramp`, `spike`, `step`; fix: write "
            "`shape` as one of `ramp`, `spike`, `step`",
        ),
        (
            {"shape": "spike", "at": 0.5, "amplitude": 1.0},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X lacks `width`; fix: give the `spike` event `width`",
        ),
        (
            {"shape": "ramp", "at": 0.2, "to": 1.0},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X lacks `until`; fix: give the `ramp` event `until`",
        ),
        (
            {"shape": "step", "to": 1.0},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X states 0 position keys; fix: state exactly one of `at`, "
            "`at_offset`, `at_time`, `at_when`",
        ),
        (
            {"shape": "step", "at": 0.5, "at_offset": -60, "to": 1.0},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X states 2 position keys (`at`, `at_offset`); fix: state exactly "
            "one of `at`, `at_offset`, `at_time`, `at_when`",
        ),
        (
            {"shape": "ramp", "at_time": "08:00:00", "until": 0.5, "to": 1.0},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X places a `ramp` by `at_time`; fix: place the `ramp` by `at` with "
            "`until`, or by `at_offset` with `until_offset`",
        ),
        (
            {"shape": "spike", "at_offset": -60, "amplitude": "big", "width": 5},
            "SR:X",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:X has `amplitude` 'big', not a number; fix: write `amplitude` as "
            "a number",
        ),
        (
            {"shape": "ramp", "at": 0.2, "until": 0.4, "to": 1.0},
            "SR:ON",
            "facility: value-invalid: scenario warm — scenarios/warm.yaml `archiver` event 1 "
            "of channel SR:ON is a `ramp` on a bool channel; fix: use a `step` event, or "
            "move the event to a float channel",
        ),
    ],
    ids=[
        "unknown-shape",
        "missing-key",
        "missing-until",
        "no-position",
        "two-positions",
        "ramp-at-time",
        "not-a-number",
        "ramp-on-bool",
    ],
)
def test_a_malformed_event_stops_naming_the_file_and_the_channel(
    event: dict[str, Any], channel: str, line: str
) -> None:
    assert _line(_document(event, channel=channel)) == line


def test_a_step_to_a_label_on_a_bool_channel_passes() -> None:
    event = {"shape": "step", "at_offset": -60, "to": "ON"}
    assert check_scenario_events(_document(event, channel="SR:ON")) == []


def test_well_formed_events_of_every_shape_pass() -> None:
    events = [
        {"shape": "step", "at": 0.1, "to": 2.0},
        {"shape": "ramp", "at_offset": -600, "until_offset": -60, "to": 3},
        {"shape": "spike", "at_time": "14:32:08", "amplitude": -5.0, "width": 30},
        {
            "shape": "spike",
            "at_when": {"days_ago": 1, "time": "03:20:00"},
            "amplitude": 1,
            "width": 2,
        },
    ]
    assert check_scenario_events(_document(*events)) == []


def test_an_unknown_channel_is_left_to_the_reference_check() -> None:
    event = {"shape": "wave"}
    assert check_scenario_events(_document(event, channel="SR:NOPE")) == []


def test_the_stops_follow_scenario_entry_and_event_order() -> None:
    document = _document({"shape": "wave"}, {"shape": "step", "to": 1.0})
    assert [e.detail.split(" of channel")[0] for e in check_scenario_events(document)] == [
        "scenarios/warm.yaml `archiver` event 1",
        "scenarios/warm.yaml `archiver` event 2",
    ]


def test_the_build_stops_on_a_malformed_event(tmp_path: Path) -> None:
    tree = plain_tree()
    tree["scenarios/warm.yaml"] = {
        "archiver": [{"channel": "BPM1:X", "events": [{"shape": "wave", "at": 0.5}]}]
    }

    with pytest.raises(FacilityBuildError) as caught:
        build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    assert caught.value.kind == "value-invalid"
    assert "scenarios/warm.yaml `archiver` event 1 of channel BPM1:X" in caught.value.detail


def test_the_shipped_scenarios_build_with_no_stop(built_control_assistant: Any) -> None:
    facility = built_control_assistant.facility
    assert any(s.get("archiver") for s in facility["scenarios"])
    assert check_scenario_events(facility) == []
