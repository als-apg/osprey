"""The pyat engine's cavity setpoints: frequency, voltage and time lag.

A record on a cavity's ``Voltage`` or ``TimeLag`` is an RF setpoint like one
on ``Frequency``: written through its calibration onto every cavity it names,
read back by a readback on the same element and attribute. The voltage and
the RF switch share one cavity attribute, so the switch holds the commanded
voltage while it is off. Each of the three carries the one supply fault that
names a real error of it: a voltage scale error, a phase offset, a frequency
offset.
"""

from __future__ import annotations

import copy
from typing import Any

import at
import pytest

from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines.pyat_faults import fault_variables
from osprey.simulation.engines.pyat_variables import RFVariable, RFVoltageVariable
from osprey_facility.records import FacilityRecords

MODEL = "SR"
CAVITY = "CAV"
SWITCH = "SR:RF:SWITCH"
VOLTAGE = "SR:RF:VOLTAGE:SP"
VOLTAGE_RB = "SR:RF:VOLTAGE:RB"
PHASE = "SR:RF:PHASE:SP"
PHASE_RB = "SR:RF:PHASE:RB"
FREQUENCY = "SR:RF:CAVITY:01:FREQUENCY:SP"

#: Volts per megavolt: the voltage setpoint is published in MV.
MV = 1.0e6
#: A made-up RF wavelength share: metres of time lag per degree of phase.
M_PER_DEG = 0.6 / 360.0


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


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def _rf_records() -> list[dict[str, Any]]:
    voltage = {"element": CAVITY, "engine": {"attribute": "Voltage"}, "calibration": _linear(MV)}
    phase = {
        "element": CAVITY,
        "engine": {"attribute": "TimeLag"},
        "calibration": _linear(M_PER_DEG),
    }
    return [
        {**voltage, "address": VOLTAGE, "direction": "write", "default": 1.0, "unit": "MV"},
        {**voltage, "address": VOLTAGE_RB, "direction": "read", "default": 1.0, "unit": "MV"},
        {**phase, "address": PHASE, "direction": "write", "default": 0.0, "unit": "deg"},
        {**phase, "address": PHASE_RB, "direction": "read", "default": 0.0, "unit": "deg"},
    ]


def _switch_record() -> dict[str, Any]:
    return {
        "address": SWITCH,
        "engine": {"signal": "rf_switch"},
        "direction": "write",
        "default": 1.0,
    }


def _wiring(demo: dict[str, Any], *, switch: bool = False) -> list[dict[str, Any]]:
    wiring = [*demo["wiring"], *_rf_records()]
    if switch:
        wiring.append(_switch_record())
    return wiring


def _settings(demo: dict[str, Any], *, turns: bool) -> Any:
    settings = copy.deepcopy(demo["settings"]) or {}
    if turns:
        settings.setdefault("pyat", {})["turn_by_turn"] = {
            "injection_element": "INJ",
            "launch": {"x": 1.0e-3, "y": -5.0e-4},
        }
    return settings


def _build(
    demo: dict[str, Any], *, switch: bool = False, active: dict[str, Any] | None = None
) -> Any:
    return engine.build(
        MODEL,
        _wiring(demo, switch=switch),
        demo["deck"],
        _settings(demo, turns=switch),
        active or {},
    )


def _cavity(model: Any) -> Any:
    (cavity,) = [element for element in model.lattice if isinstance(element, at.RFCavity)]
    return cavity


class TestTheSetpoints:
    def test_voltage_and_time_lag_are_rf_variables(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        assert isinstance(model.supported_variables[VOLTAGE], RFVoltageVariable)
        assert isinstance(model.supported_variables[PHASE], RFVariable)
        assert isinstance(model.supported_variables[FREQUENCY], RFVariable)

    @pytest.mark.parametrize("attribute", ["Frequency", "Voltage", "TimeLag"])
    def test_each_is_described_as_an_rf_setting(self, attribute: str) -> None:
        record = {
            "address": "RF:SP",
            "element": CAVITY,
            "engine": {"attribute": attribute},
            "direction": "write",
        }
        assert engine.describe(record)["kind"] == "rf"

    def test_a_voltage_write_lands_on_the_cavity_through_the_calibration(
        self, demo: dict[str, Any]
    ) -> None:
        model = _build(demo)
        model.set({VOLTAGE: 1.25})
        assert _cavity(model).Voltage == pytest.approx(1.25 * MV)
        assert model.get([VOLTAGE, VOLTAGE_RB]) == {
            VOLTAGE: pytest.approx(1.25),
            VOLTAGE_RB: pytest.approx(1.25),
        }

    def test_a_phase_write_lands_as_time_lag(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        model.set({PHASE: 20.0})
        assert _cavity(model).TimeLag == pytest.approx(20.0 * M_PER_DEG)
        assert model.get([PHASE_RB])[PHASE_RB] == pytest.approx(20.0)

    def test_the_voltage_moves_the_synchrotron_tune(self, demo: dict[str, Any]) -> None:
        model = _build(demo)
        before = _synchrotron_tune(model.lattice)
        model.set({VOLTAGE: 1.5})
        assert _synchrotron_tune(model.lattice) > before * 1.1

    def test_a_voltage_over_two_cavities_is_shared_by_the_slice_weights(self) -> None:
        cavities = [
            at.RFCavity(name, 0.0, 0.0, 1.0e8, 10, 1.0e9, PassMethod="RFCavityPass")
            for name in ("C1", "C2")
        ]
        record = {
            "address": VOLTAGE,
            "slices": [{"element": "C1", "weight": 0.5}, {"element": "C2", "weight": 0.5}],
            "engine": {"attribute": "Voltage"},
            "calibration": _linear(MV),
            "default": 0.0,
        }
        variable = RFVoltageVariable.from_wiring(record, deck_energy_gev=1.0)
        simulator = _Elements(cavities)
        variable._set(simulator, 2.0)
        assert [cavity.Voltage for cavity in cavities] == [1.0 * MV, 1.0 * MV]
        assert variable._get(simulator) == pytest.approx(2.0)


class _Elements:
    """The part of a simulator a setpoint write reads: elements by name."""

    def __init__(self, elements: list[Any]) -> None:
        self.lattice = at.Lattice(elements, energy=1.0e9, periodicity=1)
        self._elements = {element.FamName: element for element in elements}

    def element(self, name: str) -> Any:
        return self._elements[name]


def _synchrotron_tune(lattice: Any) -> float:
    return float(lattice.get_tune()[2])


class TestTheVoltageAndTheSwitch:
    def test_off_keeps_the_commanded_voltage_and_on_restores_it(self, demo: dict[str, Any]) -> None:
        model = _build(demo, switch=True)
        model.set({VOLTAGE: 1.25})
        model.set({SWITCH: 0.0})
        assert _cavity(model).Voltage == 0.0
        assert model.get([VOLTAGE])[VOLTAGE] == pytest.approx(1.25)
        model.set({SWITCH: 1.0})
        assert _cavity(model).Voltage == pytest.approx(1.25 * MV)

    def test_a_voltage_written_while_off_waits_for_the_switch(self, demo: dict[str, Any]) -> None:
        model = _build(demo, switch=True)
        model.set({SWITCH: 0.0})
        model.set({VOLTAGE: 0.8})
        assert _cavity(model).Voltage == 0.0
        assert model.get([SWITCH, VOLTAGE]) == {SWITCH: 0.0, VOLTAGE: pytest.approx(0.8)}
        model.set({SWITCH: 1.0})
        assert _cavity(model).Voltage == pytest.approx(0.8 * MV)

    def test_one_batch_writes_the_two_in_either_order(self, demo: dict[str, Any]) -> None:
        for batch in ({SWITCH: 0.0, VOLTAGE: 0.8}, {VOLTAGE: 0.8, SWITCH: 0.0}):
            model = _build(demo, switch=True)
            model.set(batch)
            assert _cavity(model).Voltage == 0.0
            model.set({SWITCH: 1.0})
            assert _cavity(model).Voltage == pytest.approx(0.8 * MV)

    def test_a_zero_voltage_does_not_read_as_switched_off(self, demo: dict[str, Any]) -> None:
        model = _build(demo, switch=True)
        model.set({VOLTAGE: 0.0})
        assert model.get([SWITCH])[SWITCH] == 1.0
        model.set({VOLTAGE: 0.9})
        assert _cavity(model).Voltage == pytest.approx(0.9 * MV)

    def test_reset_returns_both_to_their_defaults(self, demo: dict[str, Any]) -> None:
        model = _build(demo, switch=True)
        model.set({VOLTAGE: 0.8, SWITCH: 0.0})
        model.reset()
        assert _cavity(model).Voltage == pytest.approx(1.0 * MV)
        assert model.get([SWITCH])[SWITCH] == 1.0


class TestTheFaults:
    def test_each_rf_setpoint_declares_its_one_fault(self, demo: dict[str, Any]) -> None:
        slots = fault_variables(_wiring(demo))
        for address in (VOLTAGE, PHASE, FREQUENCY):
            declared = sorted(slot.field for slot in slots.values() if slot.address == address)
            assert declared == (["cal_factor"] if address == VOLTAGE else ["cal_offset"])

    def test_a_voltage_scale_error_scales_what_the_cavity_gets(self, demo: dict[str, Any]) -> None:
        model = _build(demo, active={f"{VOLTAGE}/cal_factor": 1.05})
        assert _cavity(model).Voltage == pytest.approx(1.05 * MV)
        assert model.get([VOLTAGE_RB])[VOLTAGE_RB] == pytest.approx(1.0)

    def test_a_phase_offset_is_in_the_setpoints_unit(self, demo: dict[str, Any]) -> None:
        model = _build(demo, active={f"{PHASE}/cal_offset": 30.0})
        assert _cavity(model).TimeLag == pytest.approx(30.0 * M_PER_DEG)
        model.set({PHASE: -30.0})
        assert _cavity(model).TimeLag == pytest.approx(0.0, abs=1e-15)

    def test_a_frequency_offset_shifts_the_cavity_frequency(self, demo: dict[str, Any]) -> None:
        clean = _cavity(_build(demo)).Frequency
        model = _build(demo, active={f"{FREQUENCY}/cal_offset": 2.0e-4})
        assert _cavity(model).Frequency - clean == pytest.approx(200.0, abs=1e-3)

    @pytest.mark.parametrize(
        "name", [f"{VOLTAGE}/cal_offset", f"{PHASE}/cal_factor", f"{FREQUENCY}/cal_factor"]
    )
    def test_the_other_field_is_no_fault_of_it(self, demo: dict[str, Any], name: str) -> None:
        with pytest.raises(ValueError, match="does not declare"):
            _build(demo, active={name: 1.0})
