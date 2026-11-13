"""The pyat engine's turn-by-turn records: the turn state and the variables over it.

A turn-by-turn ring holds one turn state (contract C8): per monitor element
an armed flag and three turn arrays, and for the ring the turns of the last
tracked shot. The records the wiring declares expose it -- ``armed``, the
arrays and ``turns_tracked`` read it live and refuse a write; ``arm``,
``pilot_tone`` and ``stored`` are plain settings that hold what is written,
so restore and replay reach them through ``set`` and ``set`` never arms.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from lume.exceptions import ReadOnlyError

from osprey.facility.records import FacilityRecords
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines.pyat_turn_state import TurnState, turn_state_of
from osprey.simulation.engines.pyat_variables import (
    TurnArrayVariable,
    TurnSetpointVariable,
    TurnStateVariable,
    pilot_tone_on,
)

MODEL = "SR"
INJECTION = "INJ"
TURNS = 40
TRIGGER = 5
SAMPLES = TRIGGER + TURNS

ARM_1 = "SR:DIAG:BPM:01:TBT:ARM"
ARMED_1 = "SR:DIAG:BPM:01:TBT:ARMED"
X_1 = "SR:DIAG:BPM:01:TBT:X"
Y_1 = "SR:DIAG:BPM:01:TBT:Y"
SUM_1 = "SR:DIAG:BPM:01:TBT:SUM"
ARM_2 = "SR:DIAG:BPM:02:TBT:ARM"
ARMED_2 = "SR:DIAG:BPM:02:TBT:ARMED"
X_2 = "SR:DIAG:BPM:02:TBT:X"
TURNS_TRACKED = "SR:DIAG:TBT:TURNS"
TONE = "SR:DIAG:TBT:PILOT_TONE"
STORED = "SR:DIAG:TBT:STORED"
STORED_ON_MONITOR = "SR:DIAG:BPM:01:TBT:GAIN"
MONITOR_X = "SR:DIAG:BPM:01:POSITION:X"

ARRAYS = (X_1, Y_1, SUM_1, X_2)
STATE_SCALARS = (ARMED_1, ARMED_2, TURNS_TRACKED)


@pytest.fixture(scope="module")
def demo(built_demo: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(FacilityRecords(facility), MODEL),
        "deck": built_demo.facility_dir / model["deck"],
        "settings": model.get("settings"),
    }


def _turn_settings(demo: dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(demo["settings"]) or {}
    settings.setdefault("pyat", {})["turn_by_turn"] = {
        "injection_element": INJECTION,
        "turns": TURNS,
        "trigger_sample": TRIGGER,
    }
    return settings


def _record(
    signal: str,
    address: str,
    *,
    element: str | None = None,
    write: bool = False,
    default: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": f"{MODEL}/{address}",
        "address": address,
        "engine": {"signal": signal},
        "direction": "write" if write else "read",
    }
    if element is not None:
        record["element"] = element
    if default is not None:
        record["default"] = default
    return record


def _turn_records() -> list[dict[str, Any]]:
    return [
        _record("arm", ARM_1, element="BPM01", write=True, default=0.0),
        _record("armed", ARMED_1, element="BPM01"),
        _record("turn_x", X_1, element="BPM01"),
        _record("turn_y", Y_1, element="BPM01"),
        _record("turn_sum", SUM_1, element="BPM01"),
        _record("arm", ARM_2, element="BPM02", write=True, default=0.0),
        _record("armed", ARMED_2, element="BPM02"),
        _record("turn_x", X_2, element="BPM02"),
        _record("turns_tracked", TURNS_TRACKED),
        _record("pilot_tone", TONE, write=True, default=0.0),
        _record("stored", STORED, write=True, default=3.5),
        _record("stored", STORED_ON_MONITOR, element="BPM01", write=True, default=0.0),
    ]


def _build(
    demo: dict[str, Any],
    *,
    records: list[dict[str, Any]] | None = None,
    active: dict[str, Any] | None = None,
    settings: Any = None,
) -> Any:
    wiring = [*demo["wiring"], *(_turn_records() if records is None else records)]
    return engine.build(
        MODEL,
        wiring,
        demo["deck"],
        settings if settings is not None else _turn_settings(demo),
        active or {},
    )


@pytest.fixture(scope="module")
def model(demo: dict[str, Any]) -> Any:
    """One turn-by-turn ring, for the read-only checks."""
    return _build(demo)


class TestStartValues:
    def test_every_turn_record_is_exposed_at_its_start_value(self, model: Any) -> None:
        values = model.get([*ARRAYS, *STATE_SCALARS, ARM_1, ARM_2, TONE, STORED_ON_MONITOR])
        for name in ARRAYS:
            assert isinstance(values[name], np.ndarray)
            assert values[name].dtype == np.float64
            assert values[name].shape == (SAMPLES,)
            assert not values[name].any()
        for name in (*STATE_SCALARS, ARM_1, ARM_2, TONE, STORED_ON_MONITOR):
            assert values[name] == 0.0
        assert model.get([STORED])[STORED] == 3.5

    def test_each_signal_is_built_as_its_kind(self, model: Any) -> None:
        variables = model.supported_variables
        for name in ARRAYS:
            assert isinstance(variables[name], TurnArrayVariable)
            assert variables[name].read_only
        for name in STATE_SCALARS:
            assert isinstance(variables[name], TurnStateVariable)
            assert variables[name].read_only
        for name in (ARM_1, ARM_2, TONE, STORED, STORED_ON_MONITOR):
            assert isinstance(variables[name], TurnSetpointVariable)
            assert not variables[name].read_only

    def test_the_active_scenario_sets_a_setting_start(self, demo: dict[str, Any]) -> None:
        built = _build(demo, active={STORED: 7.0, TONE: 1.0})
        assert built.get([STORED])[STORED] == 7.0
        assert pilot_tone_on(built) is True


class TestTheTurnState:
    def test_a_turn_by_turn_ring_holds_one_state_per_wired_monitor(self, model: Any) -> None:
        state = turn_state_of(model)
        assert isinstance(state, TurnState)
        assert sorted(state.monitors) == ["BPM01", "BPM02"]
        assert state.settings.samples == SAMPLES
        assert state.turns_tracked == 0
        assert state.inputs is None
        assert state.previous_inputs is None
        assert state.emitted_pending is False
        assert all(not monitor.armed for monitor in state.monitors.values())

    def test_variables_read_the_state_live(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        state = turn_state_of(built)
        state.monitors["BPM01"].armed = True
        state.monitors["BPM01"].x[TRIGGER] = 1.5e-3
        state.turns_tracked = 12
        values = built.get([ARMED_1, ARMED_2, X_1, X_2, TURNS_TRACKED])
        assert values[ARMED_1] == 1.0
        assert values[ARMED_2] == 0.0
        assert values[X_1][TRIGGER] == 1.5e-3
        assert not values[X_2].any()
        assert values[TURNS_TRACKED] == 12.0

    def test_a_read_array_is_a_copy(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        built.get([X_1])[X_1][0] = 9.0
        assert not built.get([X_1])[X_1].any()

    def test_state_reset_returns_start_values(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        state = turn_state_of(built)
        state.monitors["BPM02"].armed = True
        state.monitors["BPM02"].sum[:] = 2.0
        state.turns_tracked = 7
        state.emitted_pending = True
        state.reset()
        values = built.get([ARMED_2, X_2, TURNS_TRACKED])
        assert values[ARMED_2] == 0.0
        assert not built.get([SUM_1])[SUM_1].any()
        assert not state.monitors["BPM02"].sum.any()
        assert values[TURNS_TRACKED] == 0.0
        assert state.emitted_pending is False

    def test_a_ring_that_is_not_turn_by_turn_has_no_state(self, demo: dict[str, Any]) -> None:
        built = _build(demo, records=[], settings=demo["settings"])
        assert turn_state_of(built) is None

    def test_a_turn_by_turn_ring_with_no_turn_records_holds_an_empty_state(
        self, demo: dict[str, Any]
    ) -> None:
        state = turn_state_of(_build(demo, records=[]))
        assert isinstance(state, TurnState)
        assert state.monitors == {}


class TestSettings:
    def test_a_stored_setting_reads_back(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        before = built.get([MONITOR_X])[MONITOR_X]
        built.set({STORED: 42.0, STORED_ON_MONITOR: -1.25})
        values = built.get([STORED, STORED_ON_MONITOR, MONITOR_X])
        assert values[STORED] == 42.0
        assert values[STORED_ON_MONITOR] == -1.25
        assert values[MONITOR_X] == before

    def test_the_pilot_tone_reads_back(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        assert pilot_tone_on(built) is False
        built.set({TONE: 1.0})
        assert built.get([TONE])[TONE] == 1.0
        assert pilot_tone_on(built) is True
        built.set({TONE: 0.0})
        assert pilot_tone_on(built) is False

    def test_a_ring_with_no_tone_record_has_the_tone_off(self, demo: dict[str, Any]) -> None:
        assert pilot_tone_on(_build(demo, records=[])) is False

    def test_set_holds_an_arm_value_but_never_arms(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        built.set({ARM_1: 1.0})
        assert built.get([ARM_1])[ARM_1] == 1.0
        assert built.get([ARMED_1])[ARMED_1] == 0.0
        assert turn_state_of(built).monitors["BPM01"].armed is False

    def test_reset_returns_settings_to_their_defaults(self, demo: dict[str, Any]) -> None:
        built = _build(demo)
        built.set({STORED: 1.0, TONE: 1.0, ARM_2: 1.0})
        built.reset()
        values = built.get([STORED, TONE, ARM_2])
        assert values == {STORED: 3.5, TONE: 0.0, ARM_2: 0.0}


class TestRefusals:
    @pytest.mark.parametrize("name", [X_1, Y_1, SUM_1])
    def test_writing_an_array_is_refused(self, model: Any, name: str) -> None:
        with pytest.raises(ReadOnlyError):
            model.set({name: np.ones(SAMPLES)})

    @pytest.mark.parametrize("name", [ARMED_1, TURNS_TRACKED])
    def test_writing_armed_or_turns_tracked_is_refused(self, model: Any, name: str) -> None:
        with pytest.raises(ReadOnlyError):
            model.set({name: 1.0})
        assert model.get([name])[name] == 0.0

    def test_a_ring_that_is_not_turn_by_turn_refuses_turn_records(
        self, demo: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match=rf"model {MODEL}: {ARMED_1}.*turn_by_turn"):
            _build(
                demo,
                records=[_record("armed", ARMED_1, element="BPM01")],
                settings=demo["settings"],
            )

    def test_a_monitor_the_deck_does_not_hold_is_refused(self, demo: dict[str, Any]) -> None:
        with pytest.raises(Exception, match="NOPE"):
            _build(demo, records=[_record("armed", ARMED_1, element="NOPE")])
