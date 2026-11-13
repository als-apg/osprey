"""The pyat engine's RF switch: one setpoint that turns every cavity off and on.

Written zero, the switch zeroes every cavity's voltage on the model's own
lattice; written anything else, it puts back exactly the voltages the deck
holds. It is a plain setting -- restore and replay reach it through ``set``
-- and starts on. With it off a turn-by-turn ring stays served (no closed
orbit, monitors NaN). :func:`rf_on` reports the RF state the shot handling
reads: the switch's value, or on for a ring wired with no switch.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

at = pytest.importorskip("at")

from osprey.facility.records import FacilityRecords  # noqa: E402
from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_variables import (  # noqa: E402
    RFSwitchVariable,
    rf_on,
)

MODEL = "SR"
INJECTION = "INJ"
SWITCH = "SR:RF:SWITCH"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
MONITOR_X = "SR:DIAG:BPM:01:POSITION:X"


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
        "launch": {"x": 1.0e-3, "y": -5.0e-4},
    }
    return settings


def _switch_record(**extra: Any) -> dict[str, Any]:
    return {
        "id": "rf-switch",
        "address": SWITCH,
        "engine": {"signal": "rf_switch"},
        "direction": "write",
        "default": 1.0,
        **extra,
    }


def _build(
    demo: dict[str, Any],
    *,
    switch: dict[str, Any] | None = None,
    active: dict[str, Any] | None = None,
    settings: Any = None,
) -> Any:
    wiring = list(demo["wiring"])
    wiring.append(switch if switch is not None else _switch_record())
    return engine.build(
        MODEL,
        wiring,
        demo["deck"],
        settings if settings is not None else _turn_settings(demo),
        active or {},
    )


def _voltages(lattice: Any) -> list[float]:
    voltages = [float(element.Voltage) for element in lattice if isinstance(element, at.RFCavity)]
    assert voltages, "the demo SR deck must hold a cavity"
    return voltages


def _deck_voltages(demo: dict[str, Any]) -> list[float]:
    return _voltages(at.load_lattice(str(demo["deck"])))


class TestTheSwitch:
    def test_it_is_built_as_a_switch_variable_that_starts_on(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        assert isinstance(model.supported_variables[SWITCH], RFSwitchVariable)
        assert model.get([SWITCH])[SWITCH] == 1.0
        assert _voltages(model.lattice) == _deck_voltages(demo)
        assert any(voltage != 0.0 for voltage in _voltages(model.lattice))
        assert rf_on(model) is True

    def test_off_zeroes_every_cavity(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        model.set({SWITCH: 0.0})
        assert all(voltage == 0.0 for voltage in _voltages(model.lattice))
        assert model.get([SWITCH])[SWITCH] == 0.0

    def test_off_then_on_restores_the_voltages_exactly(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        original = _voltages(model.lattice)
        model.set({SWITCH: 0.0})
        model.set({SWITCH: 1.0})
        assert _voltages(model.lattice) == original
        assert model.get([SWITCH])[SWITCH] == 1.0

    def test_any_non_zero_value_is_on(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        original = _voltages(model.lattice)
        model.set({SWITCH: 0.0})
        model.set({SWITCH: 2.0})
        assert _voltages(model.lattice) == original
        assert rf_on(model) is True

    def test_writing_on_twice_does_not_scale_the_voltages(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        original = _voltages(model.lattice)
        model.set({SWITCH: 1.0})
        model.set({SWITCH: 1.0})
        assert _voltages(model.lattice) == original

    def test_reset_turns_the_rf_back_on(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        original = _voltages(model.lattice)
        model.set({SWITCH: 0.0})
        model.reset()
        assert _voltages(model.lattice) == original
        assert rf_on(model) is True
        assert not model.no_closed_orbit

    def test_the_active_scenario_can_start_it_off(self, demo: dict[str, Any]) -> None:
        model = _build(demo, active={SWITCH: 0.0})
        assert all(voltage == 0.0 for voltage in _voltages(model.lattice))
        assert rf_on(model) is False
        assert model.no_closed_orbit

    def test_a_record_without_a_default_starts_on(self, demo: dict[str, Any]) -> None:
        record = _switch_record()
        del record["default"]
        model = _build(demo, switch=record)
        assert model.supported_variables[SWITCH].default_value == 1.0
        assert rf_on(model) is True

    def test_its_value_range_is_honoured(self, demo: dict[str, Any]) -> None:
        model = _build(demo, switch=_switch_record(value_range=[0.0, 1.0]))
        assert model.supported_variables[SWITCH].value_range == (0.0, 1.0)


class TestRingStaysServed:
    def test_with_it_off_the_ring_is_served_without_a_closed_orbit(
        self, demo: dict[str, Any]
    ) -> None:
        model = _build(demo)
        model.set({SWITCH: 0.0})
        assert model.no_closed_orbit
        model.set({CORRECTOR: 0.5})  # still writable
        values = model.get([CORRECTOR, MONITOR_X])
        assert values[CORRECTOR] == 0.5
        assert math.isnan(values[MONITOR_X])

    def test_back_on_the_orbit_returns(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        reference = model.get([MONITOR_X])[MONITOR_X]
        model.set({SWITCH: 0.0})
        model.set({SWITCH: 1.0})
        assert not model.no_closed_orbit
        assert model.get([MONITOR_X])[MONITOR_X] == pytest.approx(reference, abs=1e-12)

    def test_a_ring_that_is_not_turn_by_turn_refuses_the_switch(self, demo: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="turn"):
            _build(demo, settings=demo["settings"] or {"pyat": {}})


class TestRFState:
    def test_it_follows_the_switch(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        assert rf_on(model) is True
        model.set({SWITCH: 0.0})
        assert rf_on(model) is False
        model.set({SWITCH: 1.0})
        assert rf_on(model) is True

    def test_a_ring_without_a_switch_is_on(self, demo: dict[str, Any]) -> None:
        model = engine.build(MODEL, demo["wiring"], demo["deck"], _turn_settings(demo), {})
        assert rf_on(model) is True


class TestTheVariable:
    def _ring(self, demo: dict[str, Any]) -> Any:
        return at.load_lattice(str(demo["deck"]))

    def test_it_captures_the_deck_voltages(self, demo: dict[str, Any]) -> None:
        ring = self._ring(demo)
        variable = RFSwitchVariable.from_wiring(_switch_record(), lattice=ring)
        assert list(variable.deck_voltages.values()) == _voltages(ring)

    def test_a_deck_without_cavities_is_refused(self) -> None:
        ring = at.Lattice([at.Drift("D", 1.0)], energy=1.0e9)
        with pytest.raises(ValueError, match="cavit"):
            RFSwitchVariable.from_wiring(_switch_record(), lattice=ring)

    def test_its_snapshot_targets_are_the_cavity_voltages(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        variable = model.supported_variables[SWITCH]
        targets = variable.snapshot_targets(model.simulator)
        cavities = [
            index for index, element in enumerate(model.lattice) if isinstance(element, at.RFCavity)
        ]
        assert targets == [(index, "Voltage") for index in cavities]
