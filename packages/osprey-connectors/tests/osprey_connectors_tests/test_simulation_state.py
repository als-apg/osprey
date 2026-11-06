"""The active-scenarios state: the set a reader serves and the file that records it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from osprey_connectors.simulation.state import (
    Overlap,
    composed_set,
    parse_active_state,
    scenario_targets,
    validate_composition,
    write_active_state,
)
from osprey_connectors_tests._paths import REPO_ROOT

VIEW = {
    "nominal": set(),
    "burst": {"SR:VAC:PRESSURE"},
    "leak": {"SR:VAC:PRESSURE"},
    "thermal": {"SR:RF:TEMP"},
}


def test_a_composing_set_is_served_as_it_is() -> None:
    assert composed_set(VIEW, ["nominal", "burst", "thermal"]) == (
        ["nominal", "burst", "thermal"],
        [],
    )


def test_a_set_writing_one_target_twice_is_served_as_nominal_alone() -> None:
    assert composed_set(VIEW, ["nominal", "burst", "leak"]) == (
        ["nominal"],
        [Overlap(target="SR:VAC:PRESSURE", first="burst", second="leak")],
    )


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown scenarios \\['nope'\\]"):
        composed_set(VIEW, ["nominal", "nope"])


# -- still -------------------------------------------------------------------------


def _targets(**scenarios: dict) -> dict[str, set[str]]:
    return {name: scenario_targets(scenario) for name, scenario in scenarios.items()}


def test_still_composes_with_overrides_and_faults_on_the_same_address() -> None:
    view = _targets(
        quiet={"still": ["SR:BPM1:X"]},
        pinned={"overrides": {"SR:BPM1:X": 1.0}},
        faulted={"faults": {"optics": {"writes": {"SR:BPM1:X": {"offset": 1e-4}}}}},
        logged={"archiver": [{"channel": "SR:BPM1:X", "events": []}]},
    )

    assert validate_composition(view, ["quiet", "pinned"]) == []
    assert validate_composition(view, ["quiet", "faulted"]) == []
    assert validate_composition(view, ["quiet", "logged"]) == []


def test_a_still_overlaps_a_noise_scenario_on_each_reading_they_share() -> None:
    view = _targets(
        quiet={"still": ["SR:BPM1:X", "SR:BPM2:X", "SR:BPM3:X"]},
        loud={"noise": {"SR:BPM1:X": {"absolute": 0.1}, "SR:BPM2:X": {"absolute": 0.1}}},
    )

    overlaps = validate_composition(view, ["nominal", "quiet", "loud"])

    assert overlaps == [
        Overlap(target="SR:BPM1:X", first="quiet", second="loud", motion=True),
        Overlap(target="SR:BPM2:X", first="quiet", second="loud", motion=True),
    ]
    assert str(overlaps[0]) == (
        "Scenarios 'quiet' and 'loud' both set the motion of 'SR:BPM1:X'; active scenarios must not"
    )


def test_a_listed_still_overlaps_a_couple_on_that_address_only() -> None:
    view = _targets(
        quiet={"still": ["SR:BPM1:X"]},
        thermal={
            "couple": {
                "SR:BPM1:X": [{"driver": "hall", "gain": 1.0}],
                "SR:BPM2:X": [{"driver": "hall", "gain": 1.0}],
            }
        },
    )

    assert validate_composition(view, ["thermal", "quiet"]) == [
        Overlap(target="SR:BPM1:X", first="thermal", second="quiet", motion=True)
    ]


def test_a_channel_fault_and_an_override_on_one_address_overlap() -> None:
    view = _targets(
        frozen={"channel_faults": {"SR:BPM1:X": "frozen"}},
        pinned={"overrides": {"SR:BPM1:X": 1.0}},
        disconnected={"channel_faults": {"SR:BPM1:X": "disconnected"}},
        quiet={"still": ["SR:BPM1:X"]},
    )

    assert validate_composition(view, ["frozen", "pinned"]) == [
        Overlap(target="SR:BPM1:X", first="frozen", second="pinned")
    ]
    assert validate_composition(view, ["frozen", "disconnected"]) == [
        Overlap(target="SR:BPM1:X", first="frozen", second="disconnected")
    ]
    assert validate_composition(view, ["frozen", "quiet"]) == []


def test_two_stills_on_one_reading_compose() -> None:
    view = _targets(
        a={"still": ["SR:BPM1:X"]},
        b={"still": ["SR:BPM1:X", "SR:BPM2:X"]},
        c={"still": ["SR:BPM1:X"]},
    )

    assert validate_composition(view, ["nominal", "a", "b", "c"]) == []


# -- archiver events -----------------------------------------------------------

#: The example facility's scenario files, as the facility ships them.
EXAMPLE_SCENARIOS = REPO_ROOT / "src/osprey/templates/facilities/example/scenarios"

#: The archiver block a demo-history scenario states when the example facility
#: does not ship one yet: events on channels rf-thermal and vacuum-burst script too.
DEMO_HISTORY_FIXTURE: dict = {
    "archiver": [
        {
            "channel": "SR:DIAG:DCCT:01:CURRENT:RB",
            "events": [{"shape": "step", "at": -0.5, "amplitude": -1.0}],
        },
        {
            "channel": "SR:RF:CAVITY:01:TEMPERATURE:RB",
            "events": [{"shape": "spike", "at_time": "09:00:00", "amplitude": 0.5, "width": 60}],
        },
        {
            "channel": "SR:VAC:GAUGE:SR07:PRESSURE:RB",
            "events": [{"shape": "spike", "at_time": "10:00:00", "amplitude": 1e-8, "width": 30}],
        },
    ]
}


def _example(name: str) -> dict:
    return yaml.safe_load((EXAMPLE_SCENARIOS / f"{name}.yaml").read_text(encoding="utf-8"))


def _demo_history() -> dict:
    path = EXAMPLE_SCENARIOS / "demo-history.yaml"
    if path.is_file():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return DEMO_HISTORY_FIXTURE


def test_an_archiver_entry_writes_no_target() -> None:
    assert scenario_targets({"archiver": [{"channel": "SR:VAC:PRESSURE", "events": []}]}) == set()


def test_archiver_events_on_one_channel_compose() -> None:
    view = _targets(
        burst={"archiver": [{"channel": "SR:VAC:PRESSURE", "events": [{"shape": "spike"}]}]},
        leak={"archiver": [{"channel": "SR:VAC:PRESSURE", "events": [{"shape": "step"}]}]},
    )

    assert validate_composition(view, ["nominal", "burst", "leak"]) == []


def test_the_shipped_vacuum_burst_and_rf_thermal_compose() -> None:
    view = _targets(**{name: _example(name) for name in ("vacuum-burst", "rf-thermal")})

    assert validate_composition(view, ["vacuum-burst", "rf-thermal"]) == []
    assert validate_composition(view, ["rf-thermal", "vacuum-burst"]) == []


@pytest.mark.parametrize("other", ["rf-thermal", "vacuum-burst"])
def test_demo_history_composes_with_a_scenario_scripting_its_channels(other: str) -> None:
    demo_history = _demo_history()
    view = _targets(**{other: _example(other), "demo-history": demo_history})
    shared = {str(entry["channel"]) for entry in demo_history.get("archiver") or []} & {
        str(entry["channel"]) for entry in _example(other).get("archiver") or []
    }

    assert shared, "the pair must script events on a shared channel"
    assert validate_composition(view, [other, "demo-history"]) == []
    assert validate_composition(view, ["demo-history", other]) == []


def test_an_override_on_a_channel_another_scenario_archives_composes() -> None:
    view = _targets(
        logged={"archiver": [{"channel": "SR:RF:TEMP", "events": [{"shape": "spike"}]}]},
        pinned={"overrides": {"SR:RF:TEMP": 26.5}},
        repinned={"overrides": {"SR:RF:TEMP": 30.0}},
    )

    assert validate_composition(view, ["logged", "pinned"]) == []
    assert validate_composition(view, ["pinned", "logged"]) == []
    assert validate_composition(view, ["logged", "pinned", "repinned"]) == [
        Overlap(target="SR:RF:TEMP", first="pinned", second="repinned")
    ]


# -- the writer ----------------------------------------------------------------

ANCHOR = datetime(2026, 3, 14, 9, 26, 53, 120000, tzinfo=UTC)


def test_a_written_set_and_anchor_read_back_through_the_parser(tmp_path: Path) -> None:
    path = tmp_path / "state" / "active_scenarios"

    resolved = write_active_state(path, VIEW, ["thermal", "burst"], anchor=ANCHOR)

    assert resolved == ["nominal", "thermal", "burst"]
    assert parse_active_state(path.read_text(encoding="utf-8")) == (
        ["thermal", "burst"],
        ANCHOR.timestamp(),
    )
    assert [entry.name for entry in path.parent.iterdir()] == ["active_scenarios"]


def test_nominal_alone_is_written_as_nominal_without_an_anchor(tmp_path: Path) -> None:
    path = tmp_path / "active_scenarios"

    assert write_active_state(path, VIEW, []) == ["nominal"]
    assert path.read_text(encoding="utf-8") == "nominal\n"
    assert parse_active_state(path.read_text(encoding="utf-8")) == (["nominal"], None)


def test_a_rewrite_replaces_the_file_rather_than_truncating_it(tmp_path: Path) -> None:
    path = tmp_path / "active_scenarios"
    write_active_state(path, VIEW, ["burst"])
    before = path.stat().st_ino

    write_active_state(path, VIEW, ["thermal"])

    assert path.stat().st_ino != before
    assert path.read_text(encoding="utf-8") == "thermal\n"


@pytest.mark.parametrize(
    ("names", "message"),
    [(["nope"], "Unknown scenarios \\['nope'\\]"), (["burst", "leak"], "'SR:VAC:PRESSURE'")],
)
def test_a_set_that_cannot_be_activated_is_refused_before_the_file_is_written(
    tmp_path: Path, names: list[str], message: str
) -> None:
    path = tmp_path / "active_scenarios"
    path.write_text("thermal\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        write_active_state(path, VIEW, names, anchor=ANCHOR)

    assert path.read_text(encoding="utf-8") == "thermal\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["active_scenarios"]
