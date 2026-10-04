"""The pyat engine's LUME model, over variables built from wiring records.

These run against a small purpose-built periodic lattice: every element name,
axis and engine attribute comes from the wiring records below, so what is
under test is the model's own work -- the fault roster it derives from those
variables, the fault seeds it writes onto elements, and the optics it computes
once per solve.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from osprey.simulation.engines import pyat_model  # noqa: E402
from osprey.simulation.engines.pyat_faults import supply_attribute  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    BETA_AT_MONITORS,
    CHROMATICITY,
    ORBIT_AT_MONITORS,
    TUNES,
    PyATLatticeModel,
)
from osprey.simulation.engines.pyat_variables import EV_PER_GEV, variable_from_wiring  # noqa: E402

ENGINES = Path(__file__).resolve().parents[2] / "src/osprey/simulation/engines"

N_CELLS = 8
DECK_ENERGY_GEV = 3.0


def _named(kind: str, cell: int) -> str:
    return f"{kind}_{cell:02d}"


def build_test_lattice() -> Any:
    """An eight-cell 4D FODO lattice with a monitor and two correctors per cell."""
    elements: list[Any] = []
    for cell in range(1, N_CELLS + 1):
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(_named("QF", cell), 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(_named("HCOR", cell), 0.0, [0.0, 0.0]),
            at.Corrector(_named("VCOR", cell), 0.0, [0.0, 0.0]),
            at.Dipole(_named("BEND", cell), 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(_named("BPM", cell)),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(_named("QD", cell), 0.3, -1.0),
        ]
    lattice = at.Lattice(elements, name="TEST", energy=DECK_ENERGY_GEV * EV_PER_GEV, periodicity=1)
    lattice.disable_6d()
    return lattice


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def _monitor(cell: int, axis: str) -> dict[str, Any]:
    return {
        "address": f"BPM:{cell:02d}:{axis.upper()}",
        "element": _named("BPM", cell),
        "engine": {"axis": axis},
        "unit": "mm",
        "calibration": _linear(1.0e-3),
    }


def _corrector(cell: int, axis: str) -> dict[str, Any]:
    kind = "HCOR" if axis == "x" else "VCOR"
    return {
        "address": f"{kind}:{cell:02d}:SP",
        "element": _named(kind, cell),
        "engine": {"attribute": "KickAngle", "index": 0 if axis == "x" else 1},
        "unit": "A",
        "default": 0.0,
        "calibration": _linear(1.0e-5),
    }


def _quad(cell: int) -> dict[str, Any]:
    return {
        "address": f"QF:{cell:02d}:SP",
        "element": _named("QF", cell),
        "engine": {"attribute": "PolynomB", "index": 1},
        "unit": "A",
        "default": 100.0,
        "calibration": _linear(1.0e-2),
    }


def wiring() -> list[dict[str, Any]]:
    """Both planes of every monitor, both correctors of every cell, one quadrupole."""
    records: list[dict[str, Any]] = []
    for cell in range(1, N_CELLS + 1):
        records += [_monitor(cell, "x"), _monitor(cell, "y")]
        records += [_corrector(cell, "x"), _corrector(cell, "y")]
    records.append(_quad(1))
    return records


def variables(records: list[dict[str, Any]] | None = None) -> list[Any]:
    built = [
        variable_from_wiring(record, deck_energy_gev=DECK_ENERGY_GEV)
        for record in (records if records is not None else wiring())
    ]
    return [variable for variable in built if variable is not None]


def build(**kwargs: Any) -> PyATLatticeModel:
    records = kwargs.pop("records", None)
    return PyATLatticeModel(build_test_lattice(), variables(records), **kwargs)


@pytest.fixture
def model() -> PyATLatticeModel:
    return build()


class TestImports:
    def test_engine_modules_import_nothing_under_the_old_accelerator_package(self):
        modules = sorted(info.name for info in pkgutil.iter_modules([str(ENGINES)]))
        assert "pyat_model" in modules
        code = (
            "import importlib, sys\n"
            f"for name in {modules!r}:\n"
            "    importlib.import_module('osprey.simulation.engines.' + name)\n"
            "loaded = sorted(m for m in sys.modules "
            "if m.startswith('osprey.services.virtual_accelerator'))\n"
            "assert not loaded, loaded\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr


class TestFaultRoster:
    def test_a_monitor_carries_its_readout_faults_per_address_and_one_roll(self, model):
        names = set(model.supported_variables)
        for field in ("offset", "gain", "noise", "polarity"):
            assert f"BPM:01:X/{field}" in names
            assert f"BPM:01:Y/{field}" in names
        assert "BPM:01:X/roll" in names
        assert "BPM:01:Y/roll" not in names

    def test_a_monitor_with_one_readback_carries_its_roll_there(self):
        records = [_monitor(cell, "y") for cell in range(1, N_CELLS + 1)]
        model = build(records=records)
        assert "BPM:01:Y/roll" in model.supported_variables

    def test_a_kick_or_strength_setpoint_carries_its_calibration(self, model):
        for address in ("HCOR:01:SP", "VCOR:01:SP", "QF:01:SP"):
            assert f"{address}/cal_factor" in model.supported_variables
            assert f"{address}/cal_offset" in model.supported_variables

    def test_derived_names_are_the_faults_and_the_optics(self, model):
        channels = {record["address"] for record in wiring()}
        assert model.derived_names == frozenset(model.supported_variables) - channels
        assert {TUNES, CHROMATICITY, BETA_AT_MONITORS, ORBIT_AT_MONITORS} <= model.derived_names

    def test_faults_sit_on_the_element_attributes_at_identity(self, model):
        monitor = model.simulator.element("BPM_01")
        assert monitor.readout_offset_x == 0.0
        assert monitor.readout_gain_y == 1.0
        assert monitor.readout_noise_x == 0.0
        assert monitor.readout_polarity_y == 1.0
        assert monitor.readout_roll == 0.0
        corrector = model.simulator.element("HCOR_01")
        assert getattr(corrector, supply_attribute("HCOR:01:SP", "cal_factor")) == 1.0
        assert getattr(corrector, supply_attribute("HCOR:01:SP", "cal_offset")) == 0.0

    def test_fault_units_follow_the_variable_they_perturb(self, model):
        variables = model.supported_variables
        assert variables["BPM:01:X/offset"].unit == "mm"
        assert variables["BPM:01:Y/noise"].unit == "mm"
        assert variables["BPM:01:X/roll"].unit == "rad"
        assert variables["BPM:01:X/gain"].unit is None
        assert variables["HCOR:01:SP/cal_offset"].unit == "A"
        assert variables["HCOR:01:SP/cal_factor"].unit is None

    def test_a_seed_is_the_default_and_lands_on_the_element(self):
        model = build(faults={"BPM:02:Y/offset": 0.25, "QF:01:SP/cal_factor": 1.1})
        assert model.simulator.element("BPM_02").readout_offset_y == 0.25
        quad = model.simulator.element("QF_01")
        assert getattr(quad, supply_attribute("QF:01:SP", "cal_factor")) == 1.1
        model.set({"BPM:02:Y/offset": 0.5})
        model.reset()
        assert model.get(["BPM:02:Y/offset"])["BPM:02:Y/offset"] == 0.25
        assert model.simulator.element("BPM_02").readout_offset_y == 0.25

    def test_a_seed_naming_no_fault_is_refused_before_the_lattice_is_touched(self):
        lattice = build_test_lattice()
        with pytest.raises(ValueError, match=r"BPM:01:X/drift"):
            PyATLatticeModel(lattice, variables(), faults={"BPM:01:X/drift": 1.0})
        assert not hasattr(lattice[lattice.get_uint32_index("BPM_01")[0]], "readout_offset_x")

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("BPM:01:X/gain", 20.0),
            ("BPM:01:X/roll", 0.5),
            ("BPM:01:X/noise", -1.0),
            ("BPM:01:X/polarity", 0.5),
            ("HCOR:01:SP/cal_factor", 6.0),
        ],
    )
    def test_a_seed_outside_its_bounds_is_refused(self, name, value):
        with pytest.raises(ValidationError):
            build(faults={name: value})

    def test_a_fault_write_leaves_the_orbit_alone(self, model):
        before = model.get(["BPM:03:X"])["BPM:03:X"]
        model.set({"BPM:03:X/offset": 1.0, "HCOR:01:SP/cal_factor": 2.0})
        assert model.get(["BPM:03:X"])["BPM:03:X"] == before

    def test_each_setpoint_on_a_shared_element_carries_its_own_calibration(self):
        twin = dict(_quad(1), address="QF:01:TRIM", unit="T/m")
        model = build(records=[*wiring(), twin])
        for address, unit in (("QF:01:SP", "A"), ("QF:01:TRIM", "T/m")):
            assert f"{address}/cal_factor" in model.supported_variables
            assert model.supported_variables[f"{address}/cal_offset"].unit == unit

    def test_a_combined_corrector_calibrates_each_plane_alone(self):
        records = [
            *wiring(),
            {**_corrector(1, "x"), "address": "HV:01:X"},
            {**_corrector(1, "y"), "address": "HV:01:Y", "element": _named("HCOR", 1)},
        ]
        model = build(records=records)
        model.set({"HV:01:X/cal_factor": 2.0})
        model.set({"HV:01:X": 5.0, "HV:01:Y": 5.0})
        kick = model.simulator.element(_named("HCOR", 1)).KickAngle
        assert kick[0] == pytest.approx(2.0 * 5.0e-5, rel=1e-12)
        assert kick[1] == pytest.approx(5.0e-5, rel=1e-12)

    def test_a_trim_calibration_leaves_its_family_supply_alone(self):
        family = {
            "address": "QF:ALL:SP",
            "slices": [{"element": _named("QF", cell)} for cell in range(1, N_CELLS + 1)],
            "engine": {"attribute": "PolynomB", "index": 1},
            "unit": "A",
            "default": 100.0,
            "calibration": _linear(1.0e-2),
        }
        model = build(records=[*wiring(), family])
        model.set({"QF:01:SP/cal_factor": 1.05})
        model.set({"QF:ALL:SP": 102.0})
        for cell in range(1, N_CELLS + 1):
            quad = model.simulator.element(_named("QF", cell))
            assert quad.PolynomB[1] == pytest.approx(1.02, rel=1e-12)
        model.set({"QF:ALL:SP/cal_factor": 1.5})
        model.set({"QF:01:SP": 101.0})
        assert model.simulator.element(_named("QF", 1)).PolynomB[1] == pytest.approx(
            1.05 * 1.01, rel=1e-12
        )
        assert model.simulator.element(_named("QF", 2)).PolynomB[1] == pytest.approx(
            1.02, rel=1e-12
        )

    def test_a_derived_name_already_a_channel_is_refused(self):
        clash = dict(_monitor(1, "x"), address="BPM:02:X/offset")
        with pytest.raises(ValueError, match="BPM:02:X/offset"):
            build(records=[*wiring(), clash])


def _bound_fields(model: PyATLatticeModel) -> dict[tuple[str, str, int | None], float]:
    """Every element field a writable variable binds, as the lattice holds it."""
    fields: dict[tuple[str, str, int | None], float] = {}
    for variable in model.supported_variables.values():
        for binding in getattr(variable, "bindings", ()):
            held = getattr(model.simulator.element(binding.element_name), binding.attribute)
            key = (binding.element_name, binding.attribute, binding.index)
            fields[key] = float(held if binding.index is None else held[binding.index])
    return fields


class TestBatchOrder:
    @pytest.mark.parametrize("calibration_first", [True, False])
    def test_a_batch_writes_its_calibration_before_its_setpoint(self, calibration_first):
        batch = {"QF:01:SP/cal_factor": 1.05, "QF:01:SP": 104.0}
        if not calibration_first:
            batch = dict(reversed(batch.items()))
        together = build()
        together.set(batch)
        apart = build()
        apart.set({"QF:01:SP/cal_factor": 1.05})
        apart.set({"QF:01:SP": 104.0})
        assert together.simulator.element("QF_01").PolynomB[1] == pytest.approx(
            apart.simulator.element("QF_01").PolynomB[1], rel=1e-15
        )
        assert together.simulator.element("QF_01").PolynomB[1] == pytest.approx(
            1.05 * 1.04, rel=1e-12
        )

    def test_a_reset_after_a_calibration_fault_is_a_fresh_model(self):
        fresh = build()
        model = build()
        model.set({"QF:01:SP/cal_factor": 1.05, "HCOR:02:SP": 3.0})
        model.set({"QF:01:SP": 104.0})
        model.reset()
        assert _bound_fields(model) == _bound_fields(fresh)
        readings = [f"BPM:{cell:02d}:{axis}" for cell in range(1, N_CELLS + 1) for axis in "XY"]
        assert model.get(readings) == fresh.get(readings)


class TestOptics:
    def test_optics_match_a_direct_linear_optics_pass(self, model):
        monitors = model.lattice.get_uint32_index(at.Monitor)
        _, lattice_data, element_data = at.get_optics(model.lattice, refpts=monitors)
        values = model.get([TUNES, BETA_AT_MONITORS, ORBIT_AT_MONITORS])
        np.testing.assert_allclose(values[TUNES], lattice_data.tune[:2], atol=1e-12)
        np.testing.assert_allclose(values[BETA_AT_MONITORS], element_data.beta, atol=1e-12)
        assert values[ORBIT_AT_MONITORS].shape == (N_CELLS, 2)

    def test_the_orbit_rows_are_the_monitor_readings_in_lattice_order(self, model):
        model.set({"HCOR:01:SP": 5.0, "VCOR:02:SP": 5.0})
        orbit = model.get([ORBIT_AT_MONITORS])[ORBIT_AT_MONITORS]
        for row, cell in enumerate(range(1, N_CELLS + 1)):
            readings = model.get([f"BPM:{cell:02d}:X", f"BPM:{cell:02d}:Y"])
            # The monitor readings are published in millimetres.
            np.testing.assert_allclose(
                orbit[row] * 1e3, list(readings.values()), rtol=1e-12, atol=1e-15
            )

    def test_optics_are_computed_once_per_solve_and_never_on_a_write(self, model, monkeypatch):
        calls: list[Any] = []
        real = at.get_optics

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(pyat_model.at, "get_optics", counting)
        model.set({"HCOR:01:SP": 1.0})
        assert calls == []
        model.get([TUNES])
        model.get([BETA_AT_MONITORS, "BPM:01:X"])
        assert len(calls) == 1
        model.set({"HCOR:01:SP": 2.0})
        model.get([TUNES])
        assert len(calls) == 2

    def test_a_read_returns_a_copy(self, model):
        tunes = model.get([TUNES])[TUNES]
        tunes[:] = 0.0
        assert np.all(model.get([TUNES])[TUNES] != 0.0)

    def test_the_chromaticity_matches_a_chromatic_optics_pass(self, model):
        _, lattice_data, _ = at.get_optics(model.lattice, get_chrom=True)
        values = model.get([CHROMATICITY])
        assert values[CHROMATICITY].shape == (2,)
        np.testing.assert_allclose(values[CHROMATICITY], lattice_data.chromaticity, atol=1e-9)

    def test_the_chromatic_solve_runs_once_per_solve_and_only_when_named(self, model, monkeypatch):
        chromatic: list[bool] = []
        real = at.get_optics

        def counting(*args: Any, **kwargs: Any) -> Any:
            chromatic.append(bool(kwargs.get("get_chrom", False)))
            return real(*args, **kwargs)

        monkeypatch.setattr(pyat_model.at, "get_optics", counting)
        model.set({"HCOR:01:SP": 1.0})
        model.get([TUNES, BETA_AT_MONITORS, ORBIT_AT_MONITORS, "BPM:01:X"])
        assert chromatic == [False]
        model.get([CHROMATICITY])
        model.get([CHROMATICITY, TUNES])
        assert chromatic == [False, True]
        model.set({"HCOR:01:SP": 2.0})
        model.get([CHROMATICITY])
        assert chromatic == [False, True, True]
