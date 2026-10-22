"""The RF-on setting: a discrete switch bound to the cavity voltage.

A write record whose engine block names the ``Voltage`` of a cavity is built
as a :class:`~osprey.simulation.engines.pyat_variables.PyATWritableEnumVariable`.
Its options are the indices of the record's labels (``FALSE``/``TRUE`` for a
bool channel that states none), its default is the on index, and each binding
is weighted by the voltage the bound cavity carries in the deck, so a write
of the on index puts the deck voltage back and the off index puts zero there.
A value outside the labels is refused by the enum's own domain check.

With RF off a 6D periodic lattice has no closed orbit: a model stating an
``injection`` keeps the write, and its orbit, tune and chromaticity read NaN.
Switching RF back on restores finite readings on the next solve. These run
against a small purpose-built 6D ring with one cavity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_model import CHROMATICITY, TUNES  # noqa: E402
from osprey.simulation.engines.pyat_variables import (  # noqa: E402
    VOLTAGE_ATTRIBUTE,
    PyATWritableEnumVariable,
    is_rf_switch_record,
)

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
CAVITY = "CAV"
ENERGY_EV = 3.0e9
HARMONIC = 10
#: The voltage the cavity carries in the deck.
DECK_VOLTAGE = 1.0e6
#: The ring's circumference: eight cells of 3.4 m.
CIRCUMFERENCE = 8 * 3.4
#: The frequency the cavity carries in the deck.
DECK_FREQUENCY = at.constants.clight / CIRCUMFERENCE * HARMONIC

RF_ON = "RF:ON:SP"
RF_ON_READBACK = "RF:ON:RB"
RF_FREQUENCY = "RF:FREQ:SP"
TUNE_X = "TUNE:X"
TUNE_Y = "TUNE:Y"
CHROMATICITY_X = "CHROM:X"

#: The index of each label of a bool channel.
OFF, ON = 0, 1

SETTINGS = {"pyat": {"injection": {"marker": MARKER}}}

MONITORS = [f"BPM:{cell:02d}:{axis}" for cell in range(1, N_CELLS + 1) for axis in ("X", "Y")]
#: Every reading a lost closed orbit voids.
VOIDED = [*MONITORS, TUNE_X, TUNE_Y, CHROMATICITY_X]


def lattice() -> Any:
    """A 6D ring without radiation: a marker, a cavity, then a FODO cell per bend."""
    elements: list[Any] = [at.Marker(MARKER)]
    for cell in range(1, N_CELLS + 1):
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(f"BPM_{cell:02d}"),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0),
        ]
    ring = at.Lattice(elements, name="TEST", energy=ENERGY_EV, periodicity=1)
    ring.insert(1, at.RFCavity(CAVITY, 0.0, DECK_VOLTAGE, DECK_FREQUENCY, HARMONIC, ENERGY_EV))
    ring = at.Lattice(ring, energy=ENERGY_EV, periodicity=1)
    ring.enable_6d(dipole_pass=None, quadrupole_pass=None)
    return ring


def rf_on(**extra: Any) -> dict[str, Any]:
    """The RF-on setting: a bool channel on the cavity voltage."""
    return {
        "address": RF_ON,
        "element": CAVITY,
        "direction": "write",
        "engine": {"attribute": "Voltage"},
        "value_type": "bool",
        **extra,
    }


def rf_on_readback() -> dict[str, Any]:
    return {
        "address": RF_ON_READBACK,
        "element": CAVITY,
        "direction": "read",
        "engine": {"attribute": "Voltage"},
        "value_type": "bool",
    }


def rf_frequency() -> dict[str, Any]:
    """The cavity frequency setpoint, wired as the example facility wires it."""
    return {
        "address": RF_FREQUENCY,
        "element": CAVITY,
        "direction": "write",
        "engine": {"attribute": "Frequency"},
        "unit": "Hz",
        "default": DECK_FREQUENCY,
    }


def monitors() -> list[dict[str, Any]]:
    return [
        {
            "address": f"BPM:{cell:02d}:{axis.upper()}",
            "element": f"BPM_{cell:02d}",
            "direction": "read",
            "engine": {"axis": axis},
            "unit": "m",
        }
        for cell in range(1, N_CELLS + 1)
        for axis in ("x", "y")
    ]


def optics() -> list[dict[str, Any]]:
    return [
        {"address": TUNE_X, "direction": "read", "engine": {"attribute": "tune", "axis": "x"}},
        {"address": TUNE_Y, "direction": "read", "engine": {"attribute": "tune", "axis": "y"}},
        {
            "address": CHROMATICITY_X,
            "direction": "read",
            "engine": {"attribute": "chromaticity", "axis": "x"},
        },
    ]


def wiring(**extra: Any) -> list[dict[str, Any]]:
    return [*monitors(), *optics(), rf_on(**extra), rf_on_readback(), rf_frequency()]


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "ring.json"
    at.save_lattice(lattice(), str(path))
    return path


@pytest.fixture
def built(deck: Path) -> Any:
    return engine.build(MODEL, wiring(), deck, SETTINGS)


def _cavity(model: Any) -> Any:
    return model.simulator.element(CAVITY)


def _finite(model: Any, names: list[str]) -> bool:
    values = model.get(names)
    return all(np.isfinite(np.asarray(values[name], dtype=np.float64)).all() for name in names)


def _all_nan(model: Any, names: list[str]) -> bool:
    values = model.get(names)
    return all(np.isnan(np.asarray(values[name], dtype=np.float64)).all() for name in names)


class TestRecognition:
    def test_an_element_record_on_the_voltage_is_the_rf_switch(self):
        assert VOLTAGE_ATTRIBUTE == "Voltage"
        assert is_rf_switch_record(rf_on())

    def test_the_frequency_record_is_not_the_switch(self):
        assert not is_rf_switch_record(rf_frequency())

    def test_a_voltage_record_naming_no_element_is_not_the_switch(self):
        record = rf_on()
        del record["element"]

        assert not is_rf_switch_record(record)

    def test_the_setting_is_described_as_rf(self):
        assert engine.describe(rf_on())["kind"] == "rf"


class TestBuild:
    def test_the_setting_is_an_enum_over_the_label_indices(self, built: Any):
        variable = built.supported_variables[RF_ON]

        assert isinstance(variable, PyATWritableEnumVariable)
        assert list(variable.options) == [OFF, ON]

    def test_its_default_is_on(self, built: Any):
        assert built.supported_variables[RF_ON].default_value == ON
        assert built.get([RF_ON])[RF_ON] == ON

    def test_each_binding_is_weighted_by_the_deck_voltage(self, built: Any):
        (binding,) = built.supported_variables[RF_ON].bindings

        assert binding.element_name == CAVITY
        assert binding.attribute == "Voltage"
        assert binding.weight == pytest.approx(DECK_VOLTAGE)

    def test_the_built_model_has_a_closed_orbit(self, built: Any):
        assert not built.no_closed_orbit
        assert _finite(built, VOIDED)

    def test_the_declared_labels_size_the_domain(self, deck: Path):
        model = engine.build(
            MODEL, wiring(value_type="enum", options=["Off", "On"]), deck, SETTINGS
        )

        assert list(model.supported_variables[RF_ON].options) == [OFF, ON]

    def test_labels_that_are_not_two_states_are_refused(self, deck: Path):
        with pytest.raises(FacilityBuildError, match="two labels"):
            engine.build(
                MODEL, wiring(value_type="enum", options=["Off", "Low", "On"]), deck, SETTINGS
            )

    def test_a_cavity_at_zero_voltage_is_refused(self, tmp_path: Path):
        ring = lattice()
        for element in ring:
            if element.FamName == CAVITY:
                element.Voltage = 0.0
        path = tmp_path / "dead.json"
        at.save_lattice(ring, str(path))

        with pytest.raises(FacilityBuildError, match="Voltage"):
            engine.build(MODEL, wiring(), path, SETTINGS)

    def test_an_active_value_does_not_switch_the_rf(self, deck: Path):
        model = engine.build(MODEL, wiring(), deck, SETTINGS, active={RF_ON: OFF})

        assert model.get([RF_ON])[RF_ON] == ON
        assert not model.no_closed_orbit


class TestSwitching:
    def test_off_clears_the_closed_orbit_and_tunes_to_nan(self, built: Any):
        built.set({RF_ON: OFF})

        assert built.no_closed_orbit
        assert _cavity(built).Voltage == 0.0
        assert _all_nan(built, VOIDED)
        assert np.isnan(built.get([TUNES])[TUNES]).all()
        assert np.isnan(built.get([CHROMATICITY])[CHROMATICITY]).all()

    def test_off_names_the_voided_readings_in_the_condition(self, built: Any):
        built.set({RF_ON: OFF})
        text, names = engine.condition(built)

        assert text == engine.NO_CLOSED_ORBIT
        assert set(VOIDED) <= names
        assert names.isdisjoint({RF_ON, RF_ON_READBACK, RF_FREQUENCY})

    def test_on_again_restores_finite_values(self, built: Any):
        before = built.get(VOIDED)
        built.set({RF_ON: OFF})
        built.set({RF_ON: ON})

        assert not built.no_closed_orbit
        assert _cavity(built).Voltage == pytest.approx(DECK_VOLTAGE)
        after = built.get(VOIDED)
        for name in VOIDED:
            assert after[name] == pytest.approx(before[name], abs=1e-9), name

    def test_the_readback_follows_the_setting(self, built: Any):
        built.set({RF_ON: OFF})
        assert built.get([RF_ON_READBACK])[RF_ON_READBACK] == OFF

        built.set({RF_ON: ON})
        assert built.get([RF_ON_READBACK])[RF_ON_READBACK] == ON

    def test_reset_switches_the_rf_back_on(self, built: Any):
        built.set({RF_ON: OFF})
        built.reset()

        assert built.get([RF_ON])[RF_ON] == ON
        assert not built.no_closed_orbit

    def test_the_cavity_frequency_wiring_is_untouched(self, built: Any):
        frequency = _cavity(built).Frequency
        built.set({RF_ON: OFF})
        built.set({RF_ON: ON})

        assert _cavity(built).Frequency == frequency == pytest.approx(DECK_FREQUENCY)


class TestDomain:
    @pytest.mark.parametrize("value", [2, -1, 0.5])
    def test_a_value_outside_the_labels_is_refused(self, built: Any, value: float):
        with pytest.raises(ValueError, match="not one of the allowed options"):
            built.set({RF_ON: value})

        assert built.get([RF_ON])[RF_ON] == ON
        assert _cavity(built).Voltage == pytest.approx(DECK_VOLTAGE)

    def test_the_channel_coercion_refuses_an_unknown_label(self):
        from osprey_connectors.simulation.values import coerce

        with pytest.raises(ValueError, match="label not in"):
            coerce("MAYBE", "bool")
        with pytest.raises(ValueError, match="index outside"):
            coerce(2, "bool")


class TestStartValues:
    def test_the_setting_and_its_readback_start_on(self, deck: Path):
        records = [rf_on(), rf_on_readback()]
        values = engine.start_values(
            deck,
            records,
            SETTINGS,
            readbacks={RF_ON_READBACK: RF_ON},
            value_types={RF_ON: "bool", RF_ON_READBACK: "bool"},
        )

        assert values == {RF_ON: ON, RF_ON_READBACK: ON}

    def test_a_bool_channel_on_another_attribute_is_still_refused(self, deck: Path):
        record = rf_frequency()
        with pytest.raises(FacilityBuildError, match="float channels only"):
            engine.start_values(deck, [record], SETTINGS, value_types={RF_FREQUENCY: "bool"})
