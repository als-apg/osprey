"""The pyat engine's tolerant periodic solve: a turn-by-turn ring keeps serving without a closed orbit.

The RF is switched off by zeroing every cavity's voltage on the model's own
lattice, the way a later setpoint will: a six-dimensional ring then has no
closed orbit. A turn-by-turn ring stays built, its orbit and optics read NaN
and its settings stay writable; any other ring refuses the write that
reaches that state and rolls it back; with the voltages restored, the orbit
returns.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from lume_pyat.exceptions import OrbitSolveError  # noqa: E402
from lume_pyat.simulator import PyATSimulator  # noqa: E402

from osprey.facility.records import FacilityRecords  # noqa: E402
from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    BETA_AT_MONITORS,
    CHROMATICITY,
    ORBIT_AT_MONITORS,
    TUNES,
)
from osprey.simulation.engines.pyat_tolerant import (  # noqa: E402
    NoClosedOrbit,
    TolerantPeriodicSimulator,
)

MODEL = "SR"
INJECTION = "INJ"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
MONITOR_X = "SR:DIAG:BPM:01:POSITION:X"
TUNE_X = "SR:DIAG:TUNE:X"
CHROM_X = "SR:DIAG:CHROM:X"


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
    """The demo SR settings with a turn-by-turn block launching at the injection marker."""
    settings = copy.deepcopy(demo["settings"]) or {}
    settings.setdefault("pyat", {})["turn_by_turn"] = {
        "injection_element": INJECTION,
        "launch": {"x": 1.0e-3, "y": -5.0e-4},
    }
    return settings


def _build(demo: dict[str, Any], settings: Any) -> Any:
    return engine.build(MODEL, demo["wiring"], demo["deck"], settings, {})


def _cavities(lattice: Any) -> list[Any]:
    cavities = [element for element in lattice if isinstance(element, at.RFCavity)]
    assert cavities, "the demo SR deck must hold a cavity"
    return cavities


def _set_voltage(lattice: Any, voltages: list[float] | float) -> list[float]:
    """Set every cavity's voltage; return what they held before."""
    cavities = _cavities(lattice)
    before = [float(cavity.Voltage) for cavity in cavities]
    if not isinstance(voltages, list):
        voltages = [voltages] * len(cavities)
    for cavity, voltage in zip(cavities, voltages, strict=True):
        cavity.Voltage = voltage
    return before


# -- the simulator alone ----------------------------------------------------------


class TestSimulator:
    @pytest.fixture
    def ring(self, demo: dict[str, Any]) -> Any:
        return at.load_lattice(str(demo["deck"]))

    def test_a_lost_closed_orbit_is_a_nan_solution(self, ring: Any) -> None:
        simulator = TolerantPeriodicSimulator(ring)
        assert not simulator.no_closed_orbit
        _set_voltage(ring, 0.0)
        solution = simulator.solve()
        assert isinstance(solution, NoClosedOrbit)
        assert simulator.no_closed_orbit
        assert simulator.last_solution is solution
        assert solution.reason
        monitors = [element.FamName for element in ring if isinstance(element, at.Monitor)]
        assert set(solution) == set(monitors)
        assert all(math.isnan(x) and math.isnan(y) for x, y in solution.values())

    def test_the_closed_orbit_returns_with_the_voltage(self, ring: Any) -> None:
        simulator = TolerantPeriodicSimulator(ring)
        before = _set_voltage(ring, 0.0)
        simulator.solve()
        _set_voltage(ring, before)
        solution = simulator.solve()
        assert not isinstance(solution, NoClosedOrbit)
        assert not simulator.no_closed_orbit
        assert all(math.isfinite(x) and math.isfinite(y) for x, y in solution.values())

    def test_the_strict_simulator_still_raises(self, ring: Any) -> None:
        simulator = PyATSimulator(ring)
        _set_voltage(ring, 0.0)
        with pytest.raises(OrbitSolveError):
            simulator.solve()

    def test_the_flag_is_restored_with_the_solution(self, ring: Any) -> None:
        simulator = TolerantPeriodicSimulator(ring)
        simulator.solve()
        held = simulator.snapshot_solution()
        _set_voltage(ring, 0.0)
        simulator.solve()
        assert simulator.no_closed_orbit
        simulator.restore_solution(held)
        assert not simulator.no_closed_orbit


# -- through the engine's build -----------------------------------------------------


class TestTurnByTurnRing:
    def test_it_is_built_tolerant(self, demo: dict[str, Any]) -> None:
        model = _build(demo, _turn_settings(demo))
        assert isinstance(model.simulator, TolerantPeriodicSimulator)
        assert not model.no_closed_orbit

    def test_rf_off_keeps_it_built_and_its_orbit_reads_failed(self, demo: dict[str, Any]) -> None:
        model = _build(demo, _turn_settings(demo))
        _set_voltage(model.lattice, 0.0)
        model.set({CORRECTOR: 0.5})  # a write solves; it must not be refused
        assert model.no_closed_orbit
        values = model.get([CORRECTOR, MONITOR_X, TUNE_X, CHROM_X])
        assert values[CORRECTOR] == 0.5
        assert math.isnan(values[MONITOR_X])
        assert math.isnan(values[TUNE_X])
        assert math.isnan(values[CHROM_X])
        arrays = model.get([TUNES, BETA_AT_MONITORS, ORBIT_AT_MONITORS, CHROMATICITY])
        for name, array in arrays.items():
            assert np.isnan(array).all(), name

    def test_settings_stay_writable_without_a_closed_orbit(self, demo: dict[str, Any]) -> None:
        model = _build(demo, _turn_settings(demo))
        _set_voltage(model.lattice, 0.0)
        model.set({CORRECTOR: 0.5})
        model.set({CORRECTOR: -0.25})
        assert model.get([CORRECTOR])[CORRECTOR] == -0.25
        assert model.no_closed_orbit

    def test_restoring_the_voltage_brings_the_orbit_back(self, demo: dict[str, Any]) -> None:
        model = _build(demo, _turn_settings(demo))
        reference = model.get([MONITOR_X, TUNE_X])
        before = _set_voltage(model.lattice, 0.0)
        model.set({CORRECTOR: 0.0})
        assert model.no_closed_orbit
        _set_voltage(model.lattice, before)
        model.set({CORRECTOR: 0.0})
        assert not model.no_closed_orbit
        values = model.get([MONITOR_X, TUNE_X, CHROM_X])
        assert all(math.isfinite(value) for value in values.values())
        assert values[TUNE_X] == pytest.approx(reference[TUNE_X], abs=1e-9)
        assert values[MONITOR_X] == pytest.approx(reference[MONITOR_X], abs=1e-12)

    def test_reset_without_a_closed_orbit_is_served(self, demo: dict[str, Any]) -> None:
        model = _build(demo, _turn_settings(demo))
        _set_voltage(model.lattice, 0.0)
        model.reset()
        assert model.no_closed_orbit


class TestOtherRing:
    def test_it_is_built_strict(self, demo: dict[str, Any]) -> None:
        model = _build(demo, demo["settings"])
        assert not isinstance(model.simulator, TolerantPeriodicSimulator)
        assert not model.no_closed_orbit

    def test_rf_off_refuses_the_write_and_rolls_it_back(self, demo: dict[str, Any]) -> None:
        model = _build(demo, demo["settings"])
        held = model.get([CORRECTOR, MONITOR_X])
        before = _set_voltage(model.lattice, 0.0)
        with pytest.raises(OrbitSolveError):
            model.set({CORRECTOR: 0.5})
        assert not model.no_closed_orbit
        assert model.get([CORRECTOR, MONITOR_X]) == held
        _set_voltage(model.lattice, before)
        model.set({CORRECTOR: 0.5})
        assert math.isfinite(model.get([MONITOR_X])[MONITOR_X])
