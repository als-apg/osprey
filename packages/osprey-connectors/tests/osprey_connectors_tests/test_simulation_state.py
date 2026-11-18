"""The active-scenarios state: the set a reader serves and the file that records it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from osprey_connectors.errors import NaiveDatetimeError
from osprey_connectors.simulation.records import ScenarioRecord
from osprey_connectors.simulation.state import (
    ActiveState,
    Overlap,
    _scenario_targets,
    _Targets,
    compose,
    parse_active_state,
    validate_composition,
    write_active_state,
)
from osprey_connectors_tests._paths import REPO_ROOT


def _record(name: str, scenario: dict) -> ScenarioRecord:
    """A scenario's blocks as the view's record, a facility file's ``faults`` as ``{writes}``."""
    faults = {
        model: writes if "writes" in writes else {"writes": writes}
        for model, writes in (scenario.get("faults") or {}).items()
    }
    return ScenarioRecord.model_validate(
        {"name": name, **scenario, **({"faults": faults} if faults else {})}
    )


def _scenarios(**scenarios: dict) -> dict[str, ScenarioRecord]:
    return {name: _record(name, scenario) for name, scenario in scenarios.items()}


VIEW = _scenarios(
    nominal={},
    burst={"overrides": {"SR:VAC:PRESSURE": 1.0}},
    leak={"overrides": {"SR:VAC:PRESSURE": 2.0}},
    thermal={"overrides": {"SR:RF:TEMP": 30.0}},
)


# -- the active set and its composition -------------------------------------------


def test_an_active_state_resolves_its_names() -> None:
    assert ActiveState().names == ("nominal",)
    assert ActiveState(names=["thermal", "nominal", "burst", "thermal"]).names == (
        "nominal",
        "thermal",
        "burst",
    )


def test_a_composing_set_is_served_as_it_is() -> None:
    composed = compose(VIEW, ActiveState(names=("burst", "thermal")))

    assert composed.names == ("nominal", "burst", "thermal")
    assert [scenario.name for scenario in composed.served] == ["nominal", "burst", "thermal"]
    assert composed.overlaps == ()
    assert composed.overrides == {"SR:VAC:PRESSURE": 1.0, "SR:RF:TEMP": 30.0}


def test_a_set_writing_one_target_twice_is_served_as_nominal_alone() -> None:
    composed = compose(VIEW, ActiveState(names=("burst", "leak")))

    assert composed.names == ("nominal",)
    assert composed.overlaps == (Overlap(target="SR:VAC:PRESSURE", first="burst", second="leak"),)
    assert composed.overrides == {}


def test_an_unknown_name_is_refused_by_the_check_and_dropped_by_the_composition(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="Unknown scenarios \\['nope'\\]"):
        validate_composition(VIEW, ["nominal", "nope"])

    composed = compose(VIEW, ActiveState(names=("nope", "thermal")))

    assert composed.names == ("nominal", "thermal")
    assert "Unknown scenario 'nope' in the active set; ignoring" in caplog.text


def test_nominal_is_served_when_the_view_does_not_list_it() -> None:
    view = {name: record for name, record in VIEW.items() if name != "nominal"}

    composed = compose(view, ActiveState(names=("thermal",)))

    assert composed.names == ("nominal", "thermal")
    assert [scenario.name for scenario in composed.served] == ["thermal"]


def test_the_blocks_merge_in_set_order(caplog: pytest.LogCaptureFixture) -> None:
    drive = {"kind": "wander", "amplitude": 2.0, "period_s": 300.0}
    view = _scenarios(
        warm={
            "drivers": {"hall": drive},
            "couple": {"T:A": [{"driver": "hall", "gain": 0.5}, {"driver": "hall", "gain": 2.0}]},
            "noise": {"T:A": {"absolute": 0.1}},
            "faults": {"SR": {"writes": {"k": 1}}, "LTB": {"writes": {"q": 2}, "inactive": "off"}},
            "archiver": [{"channel": "T:A", "events": [{"shape": "step", "at": 0.5, "to": 1}]}],
        },
        later={
            "couple": {
                "T:B": [{"driver": "nowhere", "gain": 9.0}],
                "T:F": [{"driver": "hall", "gain": 3.0}],
            },
            "noise": {"T:E": {"relative": 0.01}},
            "faults": {"SR": {"writes": {"j": 4}}},
            "still": ["T:C"],
            "channel_faults": {"T:D": "frozen"},
            "archiver": [{"channel": "T:A", "events": [{"shape": "step", "at": 0.7, "to": 2}]}],
        },
    )

    composed = compose(view, ActiveState(names=("warm", "later")))

    assert composed.overlaps == ()
    assert [term.gain for term in composed.couplings["T:A"]] == [0.5, 2.0]
    assert [term.drive.amplitude for term in composed.couplings["T:F"]] == [2.0]
    assert composed.couplings["T:B"] == ()
    assert "T:B couples to undeclared driver 'nowhere'" in caplog.text
    assert sorted(composed.noise) == ["T:A", "T:E"]
    assert composed.faults == {"SR": {"k": 1, "j": 4}}
    assert composed.still == frozenset({"T:C"})
    assert composed.channel_faults == {"T:D": "frozen"}
    assert [event.to for event in composed.archiver_events["T:A"]] == [1, 2]


# -- still -------------------------------------------------------------------------


def test_still_composes_with_overrides_and_faults_on_the_same_address() -> None:
    view = _scenarios(
        quiet={"still": ["SR:BPM1:X"]},
        pinned={"overrides": {"SR:BPM1:X": 1.0}},
        faulted={"faults": {"optics": {"writes": {"SR:BPM1:X": {"offset": 1e-4}}}}},
        logged={"archiver": [{"channel": "SR:BPM1:X", "events": []}]},
    )

    assert validate_composition(view, ["quiet", "pinned"]) == []
    assert validate_composition(view, ["quiet", "faulted"]) == []
    assert validate_composition(view, ["quiet", "logged"]) == []


def test_a_still_overlaps_a_noise_scenario_on_each_reading_they_share() -> None:
    view = _scenarios(
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
    view = _scenarios(
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
    view = _scenarios(
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
    view = _scenarios(
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
    scenario = _record("logged", {"archiver": [{"channel": "SR:VAC:PRESSURE", "events": []}]})

    assert _scenario_targets(scenario) == _Targets()


def test_archiver_events_on_one_channel_compose() -> None:
    view = _scenarios(
        burst={
            "archiver": [
                {
                    "channel": "SR:VAC:PRESSURE",
                    "events": [
                        {"shape": "spike", "at_offset": -60, "amplitude": 1.0, "width": 5.0}
                    ],
                }
            ]
        },
        leak={
            "archiver": [
                {
                    "channel": "SR:VAC:PRESSURE",
                    "events": [{"shape": "step", "at_offset": -60, "to": 1.0}],
                }
            ]
        },
    )

    assert validate_composition(view, ["nominal", "burst", "leak"]) == []


def test_the_shipped_vacuum_burst_and_rf_thermal_compose() -> None:
    view = _scenarios(**{name: _example(name) for name in ("vacuum-burst", "rf-thermal")})

    assert validate_composition(view, ["vacuum-burst", "rf-thermal"]) == []
    assert validate_composition(view, ["rf-thermal", "vacuum-burst"]) == []


@pytest.mark.parametrize("other", ["rf-thermal", "vacuum-burst"])
def test_demo_history_composes_with_a_scenario_scripting_its_channels(other: str) -> None:
    demo_history = _demo_history()
    view = _scenarios(**{other: _example(other), "demo-history": demo_history})
    shared = {str(entry["channel"]) for entry in demo_history.get("archiver") or []} & {
        str(entry["channel"]) for entry in _example(other).get("archiver") or []
    }

    assert shared, "the pair must script events on a shared channel"
    assert validate_composition(view, [other, "demo-history"]) == []
    assert validate_composition(view, ["demo-history", other]) == []


def test_an_override_on_a_channel_another_scenario_archives_composes() -> None:
    view = _scenarios(
        logged={
            "archiver": [
                {
                    "channel": "SR:RF:TEMP",
                    "events": [
                        {"shape": "spike", "at_offset": -60, "amplitude": 1.0, "width": 5.0}
                    ],
                }
            ]
        },
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

    written = write_active_state(path.parent, VIEW, ["thermal", "burst"], anchor_at=ANCHOR)

    assert written == ActiveState(names=("nominal", "thermal", "burst"), anchor_at=ANCHOR)
    assert parse_active_state(path.read_text(encoding="utf-8")) == written
    assert [entry.name for entry in path.parent.iterdir()] == ["active_scenarios"]


def test_the_anchor_is_written_with_its_own_offset(tmp_path: Path) -> None:
    """The bytes the previous writer wrote: the anchor's ISO form, offset included."""
    pacific = datetime(2026, 3, 14, 2, 26, 53, 120000, tzinfo=timezone(timedelta(hours=-7)))

    write_active_state(tmp_path, VIEW, ["thermal"], anchor_at=pacific)

    assert (tmp_path / "active_scenarios").read_bytes() == (
        b"anchor=2026-03-14T02:26:53.120000-07:00\nthermal\n"
    )
    assert parse_active_state("anchor=2026-03-14T02:26:53.120000-07:00\n").anchor_at == ANCHOR


@pytest.mark.parametrize(
    ("names", "anchor_at", "expected"),
    [
        ([], None, b"nominal\n"),
        (["nominal"], ANCHOR, b"anchor=2026-03-14T09:26:53.120000+00:00\nnominal\n"),
        (
            ["thermal", "burst", "thermal", "nominal"],
            ANCHOR,
            b"anchor=2026-03-14T09:26:53.120000+00:00\nthermal\nburst\n",
        ),
        (["burst"], datetime(2026, 1, 1, tzinfo=UTC), b"anchor=2026-01-01T00:00:00+00:00\nburst\n"),
    ],
)
def test_the_file_holds_the_bytes_the_previous_writer_wrote(
    tmp_path: Path, names: list[str], anchor_at: datetime | None, expected: bytes
) -> None:
    """Captured from the writer that took a target map and returned a list."""
    write_active_state(tmp_path, VIEW, names, anchor_at=anchor_at)

    assert (tmp_path / "active_scenarios").read_bytes() == expected


def test_a_naive_anchor_is_refused_before_the_file_is_written(tmp_path: Path) -> None:
    path = tmp_path / "active_scenarios"
    path.write_text("thermal\n", encoding="utf-8")

    with pytest.raises(NaiveDatetimeError, match="anchor_at") as raised:
        write_active_state(tmp_path, VIEW, ["burst"], anchor_at=ANCHOR.replace(tzinfo=None))

    assert raised.value.parameter == "anchor_at"
    assert path.read_text(encoding="utf-8") == "thermal\n"


def test_nominal_alone_is_written_as_nominal_without_an_anchor(tmp_path: Path) -> None:
    path = tmp_path / "active_scenarios"

    assert write_active_state(tmp_path, VIEW, []) == ActiveState()
    assert path.read_text(encoding="utf-8") == "nominal\n"
    assert parse_active_state(path.read_text(encoding="utf-8")) == ActiveState(names=("nominal",))


def test_a_rewrite_replaces_the_file_rather_than_truncating_it(tmp_path: Path) -> None:
    path = tmp_path / "active_scenarios"
    write_active_state(tmp_path, VIEW, ["burst"])
    before = path.stat().st_ino

    write_active_state(tmp_path, VIEW, ["thermal"])

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
        write_active_state(tmp_path, VIEW, names, anchor_at=ANCHOR)

    assert path.read_text(encoding="utf-8") == "thermal\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["active_scenarios"]
