"""The active-scenarios state: the set a reader serves and the file that records it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey_connectors.simulation.state import (
    Overlap,
    composed_set,
    parse_active_state,
    scenario_targets,
    validate_composition,
    write_active_state,
)

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


def test_still_all_overlaps_a_noise_scenario() -> None:
    view = _targets(
        quiet={"still": "all"},
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


def test_two_stills_on_one_reading_compose() -> None:
    view = _targets(a={"still": ["SR:BPM1:X"]}, b={"still": "all"}, c={"still": ["SR:BPM1:X"]})

    assert validate_composition(view, ["nominal", "a", "b", "c"]) == []


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
