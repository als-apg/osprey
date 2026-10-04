"""The simulation package root and its state helpers."""

import pytest

from osprey_connectors.simulation import state


def test_engine_takes_the_state_names_from_the_state_module():
    from osprey_connectors.simulation import engine

    assert engine.ACTIVE_SCENARIOS_FILENAME is state.ACTIVE_SCENARIOS_FILENAME
    assert engine.resolve_active_scenarios is state.resolve_active_scenarios


def test_two_scenarios_writing_one_address_give_one_overlap_naming_it():
    view = {"a": {"SR:BPM1:X", "SR:Q1:I"}, "b": {"SR:BPM1:X", "SR:Q2:I"}}

    overlaps = state.validate_composition(view, ["nominal", "a", "b"])

    assert overlaps == [state.Overlap(target="SR:BPM1:X", first="a", second="b")]
    assert "SR:BPM1:X" in str(overlaps[0])


def test_disjoint_scenarios_compose():
    view = {"a": {"X"}, "b": {"Y"}}

    assert state.validate_composition(view, ["a", "b"]) == []


def test_an_unknown_scenario_is_refused_by_name():
    with pytest.raises(ValueError, match="'c'"):
        state.validate_composition({"a": {"X"}}, ["a", "c"])


def test_overlap_record_prints_its_log_origin():
    overlap = state.Overlap(target="SR:BPM1:X", first="a", second="b")

    record = state.overlap_record(overlap, instance="live_standin", pid=4242)
    line = state.format_overlap_record("SR", record)

    assert record == {
        "instance": "live_standin",
        "pid": 4242,
        "event": state.OVERLAP_EVENT,
        "target": "SR:BPM1:X",
    }
    assert line.startswith("SR (log, instance live_standin, pid 4242): ")
    assert "SR:BPM1:X" in line
