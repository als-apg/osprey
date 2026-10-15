"""The pyat engine's tolerant periodic solve: a lost closed orbit is a state, not a refusal.

These run against a small purpose-built periodic lattice with one marker to
launch at and one quadrupole that, driven hard enough, leaves the lattice with
no closed orbit. A periodic model that states an ``injection`` keeps that
write and reads NaN; one that states none refuses it, and so does a line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from lume_pyat.exceptions import OrbitSolveError  # noqa: E402
from lume_pyat.simulator import PyATSimulator  # noqa: E402

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    BETA_AT_MONITORS,
    CHROMATICITY,
    ORBIT_AT_MONITORS,
    TUNES,
    PyATLatticeModel,
)
from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator  # noqa: E402
from osprey.simulation.engines.pyat_tolerant import (  # noqa: E402
    NoClosedOrbit,
    TolerantPeriodicSimulator,
)
from osprey.simulation.engines.pyat_variables import variable_from_wiring  # noqa: E402

MODEL = "SR"
N_CELLS = 8
DECK_ENERGY_GEV = 3.0
MARKER = "INJ"

QUAD = "QF:01:SP"
CORRECTOR = "HCOR:01:SP"
MONITOR_X = "BPM:01:X"
#: The quadrupole current the lattice is built at, and one it cannot close at.
QUAD_NOMINAL = 100.0
QUAD_UNSTABLE = 1.0e4

SETTINGS = {"pyat": {"injection": {"marker": MARKER}}}
TWISS_IN = {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]}

INJECTION = engine.Injection(marker=MARKER, index=0, launch=np.zeros(6), pre_trigger_samples=10)


def lattice() -> Any:
    """An eight-cell 4D FODO lattice: a marker, then a monitor and a corrector per cell."""
    elements: list[Any] = [at.Marker(MARKER)]
    for cell in range(1, N_CELLS + 1):
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(f"HCOR_{cell:02d}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(f"BPM_{cell:02d}"),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0),
        ]
    built = at.Lattice(elements, name="TEST", energy=DECK_ENERGY_GEV * 1e9, periodicity=1)
    built.disable_6d()
    return built


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def wiring() -> list[dict[str, Any]]:
    """Both planes of every monitor, the first cell's corrector and quadrupole."""
    records: list[dict[str, Any]] = []
    for cell in range(1, N_CELLS + 1):
        for axis in ("x", "y"):
            records.append(
                {
                    "address": f"BPM:{cell:02d}:{axis.upper()}",
                    "element": f"BPM_{cell:02d}",
                    "direction": "read",
                    "engine": {"axis": axis},
                    "unit": "mm",
                    "calibration": _linear(1.0e-3),
                }
            )
    records.append(
        {
            "address": CORRECTOR,
            "element": "HCOR_01",
            "direction": "write",
            "engine": {"attribute": "KickAngle", "index": 0},
            "unit": "A",
            "default": 0.0,
            "calibration": _linear(1.0e-5),
        }
    )
    records.append(
        {
            "address": QUAD,
            "element": "QF_01",
            "direction": "write",
            "engine": {"attribute": "PolynomB", "index": 1},
            "unit": "A",
            "default": QUAD_NOMINAL,
            "calibration": _linear(1.0e-2),
        }
    )
    return records


def model(**kwargs: Any) -> PyATLatticeModel:
    channels = [
        variable_from_wiring(record, deck_energy_gev=DECK_ENERGY_GEV) for record in wiring()
    ]
    return PyATLatticeModel(kwargs.pop("lattice", None) or lattice(), channels, **kwargs)


def strength(built: PyATLatticeModel) -> float:
    return float(built.simulator.element("QF_01").PolynomB[1])


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(lattice(), str(path))
    return path


@pytest.fixture
def tolerant() -> PyATLatticeModel:
    return model(injection=INJECTION)


class TestSimulator:
    def test_a_closed_orbit_solves_as_the_strict_simulator_does(self):
        strict = PyATSimulator(lattice()).solve()
        simulator = TolerantPeriodicSimulator(lattice())

        assert simulator.solve() == strict
        assert not simulator.no_closed_orbit

    def test_the_flag_is_down_before_any_solve(self):
        assert not TolerantPeriodicSimulator(lattice()).no_closed_orbit

    def test_a_lost_closed_orbit_installs_nan_at_every_monitor(self):
        simulator = TolerantPeriodicSimulator(lattice())
        simulator.element("QF_01").PolynomB[1] = 100.0

        solution = simulator.solve()

        assert simulator.no_closed_orbit
        assert simulator.last_solution is solution
        assert isinstance(solution, NoClosedOrbit)
        assert sorted(solution) == [f"BPM_{cell:02d}" for cell in range(1, N_CELLS + 1)]
        assert all(np.isnan(value) for reading in solution.values() for value in reading)
        assert "unstable" in solution.reason

    def test_every_failed_solve_makes_a_new_solution_object(self):
        simulator = TolerantPeriodicSimulator(lattice())
        simulator.element("QF_01").PolynomB[1] = 100.0

        assert simulator.solve() is not simulator.solve()

    def test_the_flag_is_restored_with_the_solution(self):
        simulator = TolerantPeriodicSimulator(lattice())
        good = simulator.solve()
        simulator.element("QF_01").PolynomB[1] = 100.0
        lost = simulator.solve()

        simulator.restore_solution(good)
        assert not simulator.no_closed_orbit
        simulator.restore_solution(lost)
        assert simulator.no_closed_orbit


class TestSeam:
    def test_a_periodic_model_with_an_injection_gets_the_tolerant_simulator(self, tolerant):
        assert type(tolerant.simulator) is TolerantPeriodicSimulator
        assert tolerant.injection is INJECTION
        assert not tolerant.no_closed_orbit

    def test_a_periodic_model_without_an_injection_gets_the_strict_simulator(self):
        strict = model()

        assert type(strict.simulator) is PyATSimulator
        assert strict.injection is None
        assert not strict.no_closed_orbit

    def test_a_single_pass_model_keeps_its_own_simulator_whatever_its_injection(self):
        twiss_in = {
            "beta": np.array([5.0, 3.0]),
            "alpha": np.zeros(2),
            "dispersion": np.zeros(4),
            "closed_orbit": np.zeros(6),
        }
        line = model(solve="single_pass", twiss_in=twiss_in, injection=INJECTION)

        assert type(line.simulator) is SinglePassSimulator
        assert not line.no_closed_orbit


class TestDegradedModel:
    def test_a_destabilising_write_commits_and_raises_the_flag(self, tolerant):
        tolerant.set({QUAD: QUAD_UNSTABLE})

        assert tolerant.no_closed_orbit
        assert strength(tolerant) == pytest.approx(QUAD_UNSTABLE * 1.0e-2)
        assert tolerant.get([QUAD])[QUAD] == QUAD_UNSTABLE

    def test_every_monitor_reads_nan_under_the_flag(self, tolerant):
        tolerant.set({QUAD: QUAD_UNSTABLE})

        monitors = [record["address"] for record in wiring() if "axis" in record["engine"]]
        readings = tolerant.get(monitors)
        assert len(readings) == 2 * N_CELLS
        assert all(np.isnan(value) for value in readings.values())

    def test_the_optics_are_nan_of_their_declared_shapes_under_the_flag(self, tolerant):
        tolerant.set({QUAD: QUAD_UNSTABLE})

        optics = tolerant.get([TUNES, BETA_AT_MONITORS, ORBIT_AT_MONITORS, CHROMATICITY])
        shapes = {
            TUNES: (2,),
            BETA_AT_MONITORS: (N_CELLS, 2),
            ORBIT_AT_MONITORS: (N_CELLS, 2),
            CHROMATICITY: (2,),
        }
        for name, shape in shapes.items():
            assert optics[name].shape == shape, name
            assert optics[name].dtype == np.float64, name
            assert np.isnan(optics[name]).all(), name

    def test_a_write_that_restores_the_closed_orbit_clears_the_flag(self, tolerant):
        before = tolerant.get([MONITOR_X, TUNES, CHROMATICITY])
        tolerant.set({QUAD: QUAD_UNSTABLE})
        assert np.isnan(tolerant.get([TUNES])[TUNES]).all()

        tolerant.set({QUAD: QUAD_NOMINAL})

        assert not tolerant.no_closed_orbit
        after = tolerant.get([MONITOR_X, TUNES, CHROMATICITY])
        assert after[MONITOR_X] == pytest.approx(before[MONITOR_X], abs=1e-12)
        np.testing.assert_allclose(after[TUNES], before[TUNES])
        np.testing.assert_allclose(after[CHROMATICITY], before[CHROMATICITY])

    def test_a_write_under_the_flag_that_leaves_it_up_still_commits(self, tolerant):
        tolerant.set({QUAD: QUAD_UNSTABLE})

        tolerant.set({CORRECTOR: 1.0})

        assert tolerant.no_closed_orbit
        assert tolerant.get([CORRECTOR])[CORRECTOR] == 1.0

    def test_reset_returns_to_the_defaults_and_clears_the_flag(self, tolerant):
        tolerant.set({QUAD: QUAD_UNSTABLE})

        tolerant.reset()

        assert not tolerant.no_closed_orbit
        assert strength(tolerant) == pytest.approx(QUAD_NOMINAL * 1.0e-2)

    def test_a_lattice_handed_over_without_a_closed_orbit_boots_under_the_flag(self):
        unstable = lattice()
        unstable[2].PolynomB[1] = 100.0

        booted = model(lattice=unstable, injection=INJECTION)

        assert booted.no_closed_orbit
        assert np.isnan(booted.get([MONITOR_X])[MONITOR_X])


class TestStrictModel:
    def test_a_periodic_model_without_an_injection_refuses_a_destabilising_write(self):
        strict = model()
        before = strict.get([MONITOR_X, QUAD])

        with pytest.raises(OrbitSolveError):
            strict.set({QUAD: QUAD_UNSTABLE})

        assert strength(strict) == pytest.approx(1.0)
        assert strict.get([MONITOR_X, QUAD]) == before
        assert not strict.no_closed_orbit

    def test_a_lattice_handed_over_without_a_closed_orbit_is_refused(self):
        unstable = lattice()
        unstable[2].PolynomB[1] = 100.0

        with pytest.raises(OrbitSolveError):
            model(lattice=unstable)


class TestThroughBuild:
    def test_an_injection_in_the_settings_builds_the_tolerant_simulator(self, deck: Path):
        built = engine.build(MODEL, wiring(), deck, SETTINGS)

        assert type(built.simulator) is TolerantPeriodicSimulator
        assert built.injection.marker == MARKER
        assert not built.no_closed_orbit

    def test_no_injection_in_the_settings_builds_the_strict_simulator(self, deck: Path):
        built = engine.build(MODEL, wiring(), deck, None)

        assert type(built.simulator) is PyATSimulator

    def test_an_unstable_active_scenario_builds_under_the_flag(self, deck: Path):
        built = engine.build(MODEL, wiring(), deck, SETTINGS, {QUAD: QUAD_UNSTABLE})

        assert built.no_closed_orbit
        assert np.isnan(built.get([MONITOR_X])[MONITOR_X])
        assert np.isnan(built.get([TUNES])[TUNES]).all()

    def test_a_built_model_keeps_a_destabilising_write_and_recovers(self, deck: Path):
        built = engine.build(MODEL, wiring(), deck, SETTINGS)

        built.set({QUAD: QUAD_UNSTABLE})
        assert built.no_closed_orbit
        built.set({QUAD: QUAD_NOMINAL})
        assert not built.no_closed_orbit
        assert np.isfinite(built.get([MONITOR_X])[MONITOR_X])

    def test_the_same_deck_without_the_block_refuses_the_unstable_scenario(self, deck: Path):
        with pytest.raises(OrbitSolveError):
            engine.build(MODEL, wiring(), deck, None, {QUAD: QUAD_UNSTABLE})

    def test_a_built_model_without_the_block_refuses_a_destabilising_write(self, deck: Path):
        built = engine.build(MODEL, wiring(), deck, None)

        with pytest.raises(OrbitSolveError):
            built.set({QUAD: QUAD_UNSTABLE})

        assert not built.no_closed_orbit

    def test_a_lossy_single_pass_line_still_refuses(self, tmp_path: Path):
        narrow = lattice()
        for element in narrow:
            if isinstance(element, at.Drift):
                element.RApertures = np.array([-0.01, 0.01, -0.01, 0.01])
        path = tmp_path / "line.json"
        at.save_lattice(narrow, str(path))
        settings = {"pyat": {"solve": "single_pass", "twiss_in": TWISS_IN}}
        records = [record for record in wiring() if record["address"] != QUAD]

        built = engine.build(MODEL, records, path, settings)
        assert type(built.simulator) is SinglePassSimulator
        assert built.injection is not None

        with pytest.raises(OrbitSolveError):
            built.set({CORRECTOR: 1.0e4})

        assert built.get([CORRECTOR])[CORRECTOR] == 0.0
        assert not built.no_closed_orbit
