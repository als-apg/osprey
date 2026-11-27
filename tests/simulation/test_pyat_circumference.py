"""The ring-length error: one fault of the lattice itself.

A ring built a little longer or shorter than its design is out of step with
its RF: the beam still makes ``h`` RF periods per turn, so it moves to the
energy whose orbit has the design length. The engine declares the error as
``lattice/circumference_error``, in metres, and ``build`` seeds it once by
scaling every drift, which is the same machine as the design ring driven at a
shifted frequency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import at
import numpy as np
import pytest

from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines.pyat_faults import (
    CIRCUMFERENCE_ERROR,
    LATTICE_SCOPE,
    MAX_CIRCUMFERENCE_ERROR_M,
    fault_variables,
)
from osprey_facility.build import build_facility
from osprey_facility.errors import FacilityBuildError
from osprey_facility.records import FacilityRecords
from tests._builds import FRAMEWORK_CHECKS
from tests.facility._synthetic_trees import plain_tree, write_tree

MODEL = "SR"
FREQUENCY = "SR:RF:CAVITY:01:FREQUENCY:SP"
ERROR_M = 1.0e-4

WIRING = [
    {"address": "Q1:SP", "element": "Q1", "engine": {"attribute": "PolynomB", "index": 1}},
    {"address": "BPM1:X", "element": "BPM1", "engine": {"axis": "x"}},
]

# -- the roster ---------------------------------------------------------------------


def test_the_fault_is_named_for_the_lattice() -> None:
    assert CIRCUMFERENCE_ERROR == "lattice/circumference_error"


@pytest.mark.parametrize("settings", [None, {}, {"pyat": {"solve": "periodic"}}])
def test_a_periodic_model_declares_it(settings: Any) -> None:
    slot = fault_variables(WIRING, settings)[CIRCUMFERENCE_ERROR]

    assert (slot.address, slot.field, slot.scope) == (
        "lattice",
        "circumference_error",
        LATTICE_SCOPE,
    )
    assert slot.value_range == (-MAX_CIRCUMFERENCE_ERROR_M, MAX_CIRCUMFERENCE_ERROR_M)
    assert slot.element is None


def test_a_single_pass_model_declares_none() -> None:
    settings = {"pyat": {"solve": "single_pass", "twiss_in": {"beta": [1, 1], "alpha": [0, 0]}}}

    assert CIRCUMFERENCE_ERROR not in fault_variables(WIRING, settings)


# -- the scenario check, through the build -----------------------------------------


def _tree(faults: dict[str, Any]) -> dict[str, Any]:
    tree = plain_tree()
    tree["simulation/models.yaml"] = [{"name": "optics", "engine": "pyat", "wiring": WIRING}]
    tree["simulation/scenarios/seeded.yaml"] = {"faults": {"optics": faults}}
    return tree


@pytest.mark.parametrize(
    "faults",
    [{"lattice/circumference_error": 2.0e-4}, {"lattice": {"circumference_error": 2.0e-4}}],
    ids=["engine-name", "map"],
)
def test_a_scenario_states_it_in_either_spelling(tmp_path: Path, faults: dict[str, Any]) -> None:
    document = build_facility(
        write_tree(tmp_path / "facility", _tree(faults)),
        project_name="demo",
        extra_checks=FRAMEWORK_CHECKS,
    )

    assert [s["faults"] for s in document["scenarios"]] == [{"optics": faults}]


def test_a_scenario_value_outside_its_bounds_stops(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as stopped:
        build_facility(
            write_tree(tmp_path / "facility", _tree({CIRCUMFERENCE_ERROR: 1.0})),
            project_name="demo",
            extra_checks=FRAMEWORK_CHECKS,
        )

    assert stopped.value.kind == "seed-invalid"


# -- the build ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo(built_demo: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    wiring = simulator_wiring(FacilityRecords(facility), MODEL)
    return {
        "wiring": wiring,
        "deck": built_demo.facility_dir / model["deck"],
        "settings": model.get("settings"),
        "monitors": [e["address"] for e in wiring if engine._is_monitor_reading(e)],
        "frequency": next(e for e in wiring if e["address"] == FREQUENCY)["default"],
    }


def _build(demo: dict[str, Any], active: dict[str, Any] | None = None) -> Any:
    return engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], active or {})


def _lengths(lattice: Any, kind: type) -> list[float]:
    return [float(element.Length) for element in lattice if isinstance(element, kind)]


def test_the_ring_grows_by_the_error_in_its_drifts_alone(demo: dict[str, Any]) -> None:
    design = _build(demo).lattice
    longer = _build(demo, {CIRCUMFERENCE_ERROR: ERROR_M}).lattice

    assert longer.circumference - design.circumference == pytest.approx(ERROR_M, rel=1e-6)
    assert _lengths(longer, at.Dipole) == _lengths(design, at.Dipole)
    assert _lengths(longer, at.Quadrupole) == _lengths(design, at.Quadrupole)
    ratios = np.array(_lengths(longer, at.Drift)) / np.array(_lengths(design, at.Drift))
    assert np.ptp(ratios) < 1e-12


def test_a_longer_ring_is_the_design_ring_at_a_higher_frequency(demo: dict[str, Any]) -> None:
    design = _build(demo)
    nominal = np.array(list(design.get(demo["monitors"]).values()))
    longer = _build(demo, {CIRCUMFERENCE_ERROR: ERROR_M})
    by_length = np.array(list(longer.get(demo["monitors"]).values())) - nominal

    shifted = demo["frequency"] * (1.0 + ERROR_M / design.lattice.circumference)
    design.set({FREQUENCY: shifted})
    by_frequency = np.array(list(design.get(demo["monitors"]).values())) - nominal

    assert np.abs(by_frequency).max() > 1e-3  # the units are the monitors', mm
    assert by_length == pytest.approx(by_frequency, rel=2e-3, abs=1e-6)


def test_a_setpoint_write_and_a_reset_keep_it(demo: dict[str, Any]) -> None:
    model = _build(demo, {CIRCUMFERENCE_ERROR: ERROR_M})
    length = model.lattice.circumference
    model.set({FREQUENCY: demo["frequency"] + 1.0e-4})
    model.reset()

    assert model.lattice.circumference == length


def test_the_deck_on_disk_is_never_lengthened(demo: dict[str, Any]) -> None:
    before = at.load_lattice(str(demo["deck"])).circumference
    _build(demo, {CIRCUMFERENCE_ERROR: ERROR_M})

    assert _build(demo).lattice.circumference == pytest.approx(before, abs=1e-12)


@pytest.mark.parametrize("value", [1.0, -1.0, math.nan, math.inf])
def test_a_value_outside_its_bounds_is_refused(demo: dict[str, Any], value: float) -> None:
    with pytest.raises(ValueError, match="circumference_error"):
        _build(demo, {CIRCUMFERENCE_ERROR: value})
