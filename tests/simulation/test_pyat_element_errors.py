"""Element errors: misalignments and multipole deltas seeded through a scenario.

A series supply feeds many magnets from one address, so a magnet's own error
is keyed by its served deck element name, ``<element>/<field>``, never by the
address. The engine declares the slots (``fault_variables``), the scenario
check holds a scenario to them, and ``build`` seeds them once: the
misalignment as the element's transforms, the multipole as a delta that every
later setpoint write keeps. An element no wiring record binds carries the
same slots when the model's ``pyat`` settings declare it under
``error_elements``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import build_facility
from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.simulation.engines import pyat_faults
from osprey.simulation.engines.pyat_faults import (
    ELEMENT_SCOPE,
    MULTIPOLE_INDICES,
    ErrorElementsError,
    error_elements,
    fault_variables,
    multipole_field,
)
from tests.facility._synthetic_trees import plain_tree, write_tree

# -- the roster, from the wiring alone ---------------------------------------------

SETTING = {"attribute": "PolynomB", "index": 1}

SERIES = [
    {
        "address": "SF:SP",
        "direction": "write",
        "slices": [{"element": "SF_1"}, {"element": "SF_2"}, {"element": "SF_3"}],
        "engine": {"attribute": "PolynomB", "index": 2},
    },
    {
        "address": "HCM:SP",
        "direction": "write",
        "element": "HCM1",
        "engine": {"attribute": "KickAngle", "index": 0},
    },
    {"address": "BPM1:X", "direction": "read", "element": "BPM1", "engine": {"axis": "x"}},
    {"address": "BPM1:Y", "direction": "read", "element": "BPM1", "engine": {"axis": "y"}},
    {"address": "TUNE:X", "direction": "read", "engine": {"axis": "x", "attribute": "tune"}},
]


def _element_slots(wiring: list[dict[str, Any]], settings: Any = None) -> dict[str, Any]:
    roster = fault_variables(wiring, settings)
    return {n: s for n, s in roster.items() if s.scope == ELEMENT_SCOPE}


def _declaring(*items: dict[str, Any]) -> dict[str, Any]:
    """A model's settings declaring ``items`` as its error elements."""
    return {"pyat": {"error_elements": list(items)}}


def test_every_magnet_on_a_series_supply_carries_its_own_slots() -> None:
    slots = _element_slots(SERIES)

    for element in ("SF_1", "SF_2", "SF_3"):
        for field in ("dx", "dy", "roll"):
            slot = slots[f"{element}/{field}"]
            assert (slot.address, slot.element, slot.field) == (element, element, field)
        for index in MULTIPOLE_INDICES:
            assert f"{element}/dB{index}" in slots
            assert f"{element}/dA{index}" in slots
    assert "SF:SP/dx" not in fault_variables(SERIES)


def test_misalignments_are_bounded_and_multipoles_finite_only() -> None:
    slots = _element_slots(SERIES)

    assert slots["SF_1/dx"].value_range == (-0.01, 0.01)
    assert slots["SF_1/dy"].value_range == (-0.01, 0.01)
    assert slots["SF_1/roll"].value_range == (-0.1, 0.1)
    assert slots["SF_1/dB2"].value_range == (-math.inf, math.inf)


def test_a_kick_element_takes_a_misalignment_but_no_multipole() -> None:
    slots = _element_slots(SERIES)

    assert "HCM1/dx" in slots
    assert not any(name.startswith("HCM1/dB") or name.startswith("HCM1/dA") for name in slots)


def test_a_monitor_and_an_optics_reading_carry_no_element_error() -> None:
    slots = _element_slots(SERIES)

    assert not any(slot.element == "BPM1" for slot in slots.values())
    assert fault_variables(SERIES)["BPM1:X/roll"].scope == pyat_faults.ADDRESS_SCOPE
    assert {slot.element for slot in slots.values()} == {"SF_1", "SF_2", "SF_3", "HCM1"}


def test_an_element_named_like_a_wired_address_carries_none() -> None:
    wiring = [
        {"address": "Q1", "direction": "write", "element": "Q1", "engine": dict(SETTING)},
        {"address": "Q2:SP", "direction": "write", "element": "Q2", "engine": dict(SETTING)},
    ]

    slots = _element_slots(wiring)

    assert "Q1/dx" not in slots
    assert "Q2/dx" in slots


# -- the roster, with elements the settings declare ---------------------------------

ENERGY = {"address": "BEND:SP", "direction": "write", "engine": {"attribute": "energy"}}


def test_a_declared_element_no_record_binds_carries_the_slots_of_a_bound_one() -> None:
    wiring = [*SERIES, ENERGY]
    assert not any(slot.element == "BEND_1" for slot in _element_slots(wiring).values())

    slots = _element_slots(wiring, _declaring({"element": "BEND_1", "multipoles": True}))

    declared = {slot.field: slot for slot in slots.values() if slot.element == "BEND_1"}
    bound = {slot.field: slot for slot in slots.values() if slot.element == "SF_1"}
    assert set(declared) == set(bound)
    for field, slot in declared.items():
        assert (slot.address, slot.scope) == ("BEND_1", ELEMENT_SCOPE)
        assert (slot.value_range, slot.options) == (bound[field].value_range, None)


def test_a_declared_element_without_multipoles_takes_a_misalignment_only() -> None:
    slots = _element_slots(SERIES, _declaring({"element": "BEND_1"}))

    assert {s.field for s in slots.values() if s.element == "BEND_1"} == {"dx", "dy", "roll"}


@pytest.mark.parametrize(
    "settings",
    [None, {}, {"pyat": {}}, {"pyat": {"solve": "periodic"}}, {"pyat": {"error_elements": []}}],
    ids=["none", "empty", "empty-block", "other-keys", "empty-list"],
)
def test_a_model_declaring_no_error_elements_has_the_roster_of_its_wiring(settings: Any) -> None:
    assert fault_variables(SERIES, settings) == fault_variables(SERIES)
    assert list(fault_variables(SERIES, settings)) == list(fault_variables(SERIES))


def test_an_element_both_bound_and_declared_carries_the_union_of_slots() -> None:
    wired = _element_slots(SERIES)
    restated = _declaring(
        {"element": "SF_1", "multipoles": True},
        {"element": "SF_2"},
        {"element": "HCM1"},
    )
    assert _element_slots(SERIES, restated) == wired

    widened = _element_slots(SERIES, _declaring({"element": "HCM1", "multipoles": True}))

    assert set(widened) - set(wired) == {
        f"HCM1/{prefix}{index}" for prefix in ("dB", "dA") for index in MULTIPOLE_INDICES
    }
    assert {name: widened[name] for name in wired} == wired


@pytest.mark.parametrize("element", ["BPM1", "SF:SP"], ids=["monitor", "address"])
def test_a_declared_monitor_or_an_element_named_like_an_address_is_refused(element: str) -> None:
    settings = _declaring({"element": "SF_1"}, {"element": element, "multipoles": True})

    with pytest.raises(ErrorElementsError) as refused:
        fault_variables(SERIES, settings)

    assert f"pyat.error_elements names {element}, which can carry no element error" in str(
        refused.value
    )


def test_the_declared_elements_read_in_list_order_with_multipoles_off_by_default() -> None:
    settings = _declaring({"element": "B2", "multipoles": True}, {"element": "B1"})

    assert error_elements(settings) == ({"B2": True, "B1": False}, [])
    assert error_elements(None) == ({}, [])


@pytest.mark.parametrize(
    ("block", "problem"),
    [
        ({"B1": True}, "pyat.error_elements is not a list"),
        (["B1"], "pyat.error_elements[0] is 'B1', not a map"),
        ([{"element": "B1", "polynomial": True}], "[0] carries unknown keys polynomial"),
        ([{"multipoles": True}], "[0] names element None, not a name"),
        ([{"element": "B1", "multipoles": "yes"}], "[0] has multipoles 'yes', not true or false"),
        ([{"element": "B1"}, {"element": "B1"}], "[1] names element B1 a second time"),
    ],
    ids=["map", "bare-name", "unknown-key", "no-element", "multipoles-word", "twice"],
)
def test_a_malformed_error_elements_list_is_named_item_by_item(block: Any, problem: str) -> None:
    _, problems = error_elements({"pyat": {"error_elements": block}})

    assert len(problems) == 1
    assert problem in problems[0]


def test_a_malformed_error_elements_list_gives_no_roster() -> None:
    twice = _declaring({"element": "B1"}, {"element": "B1", "multipoles": True})

    with pytest.raises(ErrorElementsError, match=r"\[1\] names element B1 a second time"):
        fault_variables([], twice)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("dB0", ("PolynomB", 0)),
        ("dB9", ("PolynomB", 9)),
        ("dA3", ("PolynomA", 3)),
        ("dB10", None),
        ("dB01", None),
        ("dB", None),
        ("dx", None),
        ("dC1", None),
    ],
)
def test_a_multipole_field_names_its_polynomial_and_index(field: str, expected: Any) -> None:
    assert multipole_field(field) == expected


# -- the scenario check, through the build -----------------------------------------


def _tree(faults: dict[str, Any], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    tree = plain_tree()
    tree["models.yaml"] = [
        {
            "name": "optics",
            "engine": "pyat",
            **({} if settings is None else {"settings": settings}),
            "wiring": [
                {"address": "Q1:SP", "element": "Q1", "engine": dict(SETTING)},
                {"address": "BPM1:X", "element": "BPM1", "engine": {"axis": "x"}},
            ],
        }
    ]
    tree["scenarios/seeded.yaml"] = {"faults": {"optics": faults}}
    return tree


@pytest.mark.parametrize(
    "faults",
    [
        {"Q1/dx": 1.0e-4, "Q1/roll": -2.0e-4, "Q1/dB2": 0.5, "Q1/dA1": 1.0e-3},
        {"Q1": {"dx": 1.0e-4, "dB2": 0.5}},
    ],
    ids=["engine-names", "element-map"],
)
def test_a_scenario_seeds_element_errors(tmp_path: Path, faults: dict[str, Any]) -> None:
    document = build_facility(write_tree(tmp_path / "facility", _tree(faults)), project_name="demo")

    assert [s["faults"] for s in document["scenarios"]] == [{"optics": faults}]


@pytest.mark.parametrize(
    ("faults", "kind", "names"),
    [
        ({"Q1/dx": 0.5}, "seed-invalid", "faults.optics.Q1/dx` 0.5 lies above"),
        ({"Q1": {"roll": -1.0}}, "seed-invalid", "faults.optics.Q1.roll` -1 lies below"),
        ({"Q1/dz": 1.0e-4}, "engine-invalid", "faults.optics.Q1/dz names no channel"),
        ({"BPM1/dx": 1.0e-4}, "engine-invalid", "faults.optics.BPM1/dx names no channel"),
        ({"Q1/dB12": 1.0}, "engine-invalid", "faults.optics.Q1/dB12 names no channel"),
    ],
)
def test_a_scenario_element_error_outside_the_roster_stops(
    tmp_path: Path, faults: dict[str, Any], kind: str, names: str
) -> None:
    with pytest.raises(FacilityBuildError) as stopped:
        build_facility(write_tree(tmp_path / "facility", _tree(faults)), project_name="demo")

    assert stopped.value.kind == kind
    assert names in str(stopped.value)


@pytest.mark.parametrize(
    "faults",
    [
        {"BEND_1/dB2": 0.5, "BEND_1/dx": 1.0e-4, "BEND_2/roll": 2.0e-4},
        {"BEND_1": {"dB2": 0.5, "dA1": 1.0e-3}, "BEND_2": {"dy": -1.0e-4}},
    ],
    ids=["engine-names", "element-map"],
)
def test_a_scenario_seeds_errors_on_elements_the_settings_declare(
    tmp_path: Path, faults: dict[str, Any]
) -> None:
    settings = _declaring({"element": "BEND_1", "multipoles": True}, {"element": "BEND_2"})
    tree = _tree(faults, settings)

    document = build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")

    assert [s["faults"] for s in document["scenarios"]] == [{"optics": faults}]


@pytest.mark.parametrize(
    ("faults", "kind", "names"),
    [
        ({"BEND_1/dx": 0.5}, "seed-invalid", "faults.optics.BEND_1/dx` 0.5 lies above"),
        ({"BEND_2/dB2": 0.5}, "engine-invalid", "faults.optics.BEND_2/dB2 names no channel"),
        ({"BEND_2": {"dB2": 0.5}}, "value-invalid", "faults.optics.BEND_2.dB2 is not a fault"),
        ({"BEND_3/dx": 1.0e-4}, "engine-invalid", "faults.optics.BEND_3/dx names no channel"),
    ],
    ids=["bound", "no-multipoles", "no-multipoles-map", "undeclared"],
)
def test_a_declared_element_holds_a_scenario_to_the_slots_it_declares(
    tmp_path: Path, faults: dict[str, Any], kind: str, names: str
) -> None:
    settings = _declaring({"element": "BEND_1", "multipoles": True}, {"element": "BEND_2"})

    with pytest.raises(FacilityBuildError) as stopped:
        build_facility(
            write_tree(tmp_path / "facility", _tree(faults, settings)), project_name="demo"
        )

    assert stopped.value.kind == kind
    assert names in str(stopped.value)


@pytest.mark.parametrize(
    ("items", "detail"),
    [
        ([{"element": "BPM1"}], "pyat.error_elements names BPM1, which can carry no element"),
        ([{"element": "Q1:SP"}], "pyat.error_elements names Q1:SP, which can carry no element"),
        ([{"element": "B"}, {"element": "B"}], "[1] names element B a second time"),
    ],
    ids=["monitor", "address", "twice"],
)
def test_the_scenario_check_refuses_a_declaration_as_the_build_does(
    tmp_path: Path, items: list[dict[str, Any]], detail: str
) -> None:
    settings = _declaring(*items)
    with pytest.raises(ErrorElementsError) as built:
        fault_variables(_tree({})["models.yaml"][0]["wiring"], settings)

    with pytest.raises(FacilityBuildError) as stopped:
        build_facility(
            write_tree(tmp_path / "facility", _tree({"Q1/dx": 1.0e-4}, settings)),
            project_name="demo",
        )

    assert (stopped.value.kind, stopped.value.record_id) == ("engine-invalid", "optics")
    assert stopped.value.detail == str(built.value)
    assert detail in stopped.value.detail
    assert "Q1/dx" not in str(stopped.value)


def test_an_error_on_an_unbound_element_stops_a_model_that_does_not_declare_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(FacilityBuildError) as stopped:
        build_facility(
            write_tree(tmp_path / "facility", _tree({"BEND_1/dB2": 0.5})), project_name="demo"
        )

    assert stopped.value.kind == "engine-invalid"
    assert "faults.optics.BEND_1/dB2 names no channel" in str(stopped.value)


# -- the engine build, over the demo deck -------------------------------------------

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from lume_pyat.exceptions import OrbitSolveError, UnknownElementError  # noqa: E402

from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

MODEL = "SR"
QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"


@pytest.fixture(scope="module")
def demo(built_demo: Any) -> dict[str, Any]:
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    wiring = simulator_wiring(FacilityRecords(facility), MODEL)
    roster = fault_variables(wiring)
    return {
        "wiring": wiring,
        "deck": built_demo.facility_dir / model["deck"],
        "settings": model.get("settings"),
        "quad": roster[f"{QUADRUPOLE}/cal_factor"].element,
        "monitors": [e["address"] for e in wiring if engine._is_monitor_reading(e)],
        "default": next(e for e in wiring if e["address"] == QUADRUPOLE)["default"],
    }


def _build(demo: dict[str, Any], active: dict[str, Any] | None = None) -> Any:
    return engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], active or {})


def _element(model: Any, name: str) -> Any:
    return model.simulator.element(name)


def _orbit(model: Any, demo: dict[str, Any]) -> dict[str, float]:
    return model.get(demo["monitors"])


def test_the_roster_declares_the_demo_quadrupole_errors(demo: dict[str, Any]) -> None:
    roster = fault_variables(demo["wiring"])

    assert roster[f"{demo['quad']}/dx"].scope == ELEMENT_SCOPE
    assert f"{demo['quad']}/dB1" in roster


def test_a_misalignment_lands_as_the_element_transforms_and_moves_the_orbit(
    demo: dict[str, Any],
) -> None:
    nominal = _orbit(_build(demo), demo)

    model = _build(demo, {f"{demo['quad']}/dx": 1.0e-4, f"{demo['quad']}/dy": -5.0e-5})
    orbit = _orbit(model, demo)

    quad = _element(model, demo["quad"])
    assert quad.T1[0] == pytest.approx(-1.0e-4)
    assert quad.T1[2] == pytest.approx(5.0e-5)
    assert max(abs(orbit[name] - nominal[name]) for name in nominal) > 1e-9
    model.set({CORRECTOR: 0.0})
    model.reset()
    assert _orbit(model, demo) == pytest.approx(orbit)


def test_a_multipole_delta_composes_with_every_write_of_the_supply(demo: dict[str, Any]) -> None:
    nominal = _build(demo)
    base = float(_element(nominal, demo["quad"]).PolynomB[1])
    delta = 0.02

    model = _build(demo, {f"{demo['quad']}/dB1": delta})
    quad = _element(model, demo["quad"])
    assert quad.PolynomB[1] == pytest.approx(base + delta)
    assert model.get([QUADRUPOLE])[QUADRUPOLE] == pytest.approx(demo["default"])

    model.set({QUADRUPOLE: demo["default"] * 1.01})
    assert quad.PolynomB[1] == pytest.approx(base * 1.01 + delta)
    assert model.get([QUADRUPOLE])[QUADRUPOLE] == pytest.approx(demo["default"] * 1.01)

    model.reset()
    assert quad.PolynomB[1] == pytest.approx(base + delta)


def test_a_multipole_past_the_element_order_pads_its_polynomials(demo: dict[str, Any]) -> None:
    model = _build(demo, {f"{demo['quad']}/dB5": 3.0, f"{demo['quad']}/dA4": -1.0})
    quad = _element(model, demo["quad"])

    assert quad.MaxOrder >= 5
    assert len(quad.PolynomB) == len(quad.PolynomA) >= 6
    assert quad.PolynomB[5] == pytest.approx(3.0)
    assert quad.PolynomA[4] == pytest.approx(-1.0)


def test_the_deck_on_disk_is_never_seeded(demo: dict[str, Any]) -> None:
    _build(demo, {f"{demo['quad']}/dB1": 0.02, f"{demo['quad']}/dx": 1.0e-4})
    fresh = _build(demo)

    quad = _element(fresh, demo["quad"])
    assert not np.any(getattr(quad, "T1", np.zeros(6)))
    assert not hasattr(quad, pyat_faults.multipole_attribute("PolynomB", 1))


@pytest.mark.parametrize(
    ("field", "value"),
    [("dx", 0.5), ("roll", 1.0), ("dB1", math.nan), ("dB1", math.inf)],
)
def test_an_element_error_outside_its_bounds_is_refused(
    demo: dict[str, Any], field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=f"{demo['quad']}/{field}"):
        _build(demo, {f"{demo['quad']}/{field}": value})


def test_an_element_error_naming_no_wired_element_is_refused(demo: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="NOT_AN_ELEMENT/dx"):
        _build(demo, {"NOT_AN_ELEMENT/dx": 1.0e-4})


def test_a_multipole_on_an_element_with_no_polynomial_is_refused(demo: dict[str, Any]) -> None:
    lattice = engine._load(demo["deck"]).lattice
    bare = next(e.FamName for e in lattice if not hasattr(e, "PolynomB"))
    with pytest.raises(UnknownElementError, match=bare):
        engine._seed_multipoles(lattice.deepcopy(), {bare: {("PolynomB", 1): 1.0}})


def test_a_seed_with_no_closed_orbit_raises_the_solve_error_naming_it(
    demo: dict[str, Any],
) -> None:
    with pytest.raises(
        OrbitSolveError, match=rf"element errors seeded on 1 element\(s\) \({demo['quad']}\)"
    ):
        _build(demo, {f"{demo['quad']}/dB1": 50.0})


# -- elements the settings declare, over the demo deck --------------------------------


def _binds(entry: dict[str, Any], element: str) -> bool:
    bound = [entry.get("element"), *(item.get("element") for item in entry.get("slices") or [])]
    return element in bound


@pytest.fixture(scope="module")
def unbound(demo: dict[str, Any]) -> dict[str, Any]:
    """The demo model with every record binding its first quadrupole taken out."""
    wiring = [entry for entry in demo["wiring"] if not _binds(entry, demo["quad"])]
    assert len(wiring) < len(demo["wiring"])
    return {**demo, "wiring": wiring}


def _declared(unbound: dict[str, Any], *items: dict[str, Any]) -> dict[str, Any]:
    block = {**(unbound["settings"] or {}).get("pyat", {}), "error_elements": list(items)}
    return {**unbound, "settings": {"pyat": block}}


def test_an_unbound_element_takes_no_error_until_the_settings_declare_it(
    unbound: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=f"{unbound['quad']}/dB1"):
        _build(unbound, {f"{unbound['quad']}/dB1": 0.02})


def test_a_declared_element_no_record_binds_is_seeded_at_build(unbound: dict[str, Any]) -> None:
    quad = unbound["quad"]
    model = _declared(unbound, {"element": quad, "multipoles": True})
    nominal = _build(model)
    base = float(_element(nominal, quad).PolynomB[1])

    seeded = _build(model, {f"{quad}/dB1": 0.02, f"{quad}/dA4": -1.0, f"{quad}/dx": 1.0e-4})

    element = _element(seeded, quad)
    assert element.PolynomB[1] == pytest.approx(base + 0.02)
    assert element.PolynomA[4] == pytest.approx(-1.0)
    assert element.T1[0] == pytest.approx(-1.0e-4)
    orbit, reference = _orbit(seeded, unbound), _orbit(nominal, unbound)
    assert max(abs(orbit[name] - reference[name]) for name in reference) > 1e-9
    seeded.reset()
    assert element.PolynomB[1] == pytest.approx(base + 0.02)
    assert _orbit(seeded, unbound) == pytest.approx(orbit)


def test_a_declared_element_seeds_exactly_as_a_bound_one(
    demo: dict[str, Any], unbound: dict[str, Any]
) -> None:
    quad = demo["quad"]
    errors = {f"{quad}/dB1": 0.02, f"{quad}/dy": -5.0e-5, f"{quad}/roll": 1.0e-3}

    bound = _build(demo, errors)
    declared = _build(_declared(unbound, {"element": quad, "multipoles": True}), errors)

    for attribute in ("PolynomB", "PolynomA", "T1", "T2", "R1", "R2"):
        assert np.array_equal(
            getattr(_element(declared, quad), attribute), getattr(_element(bound, quad), attribute)
        )
    assert _orbit(declared, unbound) == pytest.approx(_orbit(bound, unbound), abs=1e-12)


def test_a_declared_element_holds_an_error_to_the_bounds_of_a_bound_one(
    unbound: dict[str, Any],
) -> None:
    quad = unbound["quad"]
    model = _declared(unbound, {"element": quad})

    with pytest.raises(ValueError, match=f"{quad}/dx is 0.5, outside its range"):
        _build(model, {f"{quad}/dx": 0.5})
    with pytest.raises(ValueError, match=f"{quad}/dB1"):
        _build(model, {f"{quad}/dB1": 0.02})


def test_declaring_a_bound_element_changes_nothing_of_the_build(demo: dict[str, Any]) -> None:
    quad = demo["quad"]
    errors = {f"{quad}/dB1": 0.02, f"{quad}/dx": 1.0e-4}

    plain = _build(demo, errors)
    restated = _build(_declared(demo, {"element": quad}), errors)

    assert _element(restated, quad).PolynomB[1] == _element(plain, quad).PolynomB[1]
    assert _orbit(restated, demo) == pytest.approx(_orbit(plain, demo), abs=1e-12)
    assert sorted(restated.supported_variables) == sorted(plain.supported_variables)


def test_a_declared_element_the_deck_does_not_hold_fails_the_build_by_name(
    unbound: dict[str, Any],
) -> None:
    model = _declared(unbound, {"element": "NOT_IN_THE_DECK", "multipoles": True})

    for refused in (
        lambda: _build(model),
        lambda: engine.prepare(model["deck"], model["settings"], model=MODEL),
    ):
        with pytest.raises(engine.ElementStop) as stopped:
            refused()
        assert (stopped.value.kind, stopped.value.record_id) == ("engine-invalid", MODEL)
        assert (stopped.value.element, stopped.value.count) == ("NOT_IN_THE_DECK", 0)
        assert stopped.value.detail == (
            "settings key pyat.error_elements: element NOT_IN_THE_DECK is not in the deck"
        )


def test_a_declared_element_the_deck_repeats_fails_the_build_by_name(
    unbound: dict[str, Any],
) -> None:
    names = [element.FamName for element in engine._load(unbound["deck"]).lattice]
    repeated = next(name for name in names if names.count(name) > 1)

    with pytest.raises(engine.ElementStop) as stopped:
        _build(_declared(unbound, {"element": repeated}))

    assert (stopped.value.element, stopped.value.count) == (repeated, names.count(repeated))
    assert "settings key pyat.error_elements" in stopped.value.detail


def test_declared_multipoles_on_an_element_with_no_polynomial_fail_the_build(
    unbound: dict[str, Any],
) -> None:
    lattice = engine._load(unbound["deck"]).lattice
    names = [element.FamName for element in lattice]
    bare = next(
        e.FamName for e in lattice if not hasattr(e, "PolynomB") and names.count(e.FamName) == 1
    )

    with pytest.raises(FacilityBuildError) as stopped:
        _build(_declared(unbound, {"element": bare, "multipoles": True}))

    assert stopped.value.kind == "engine-invalid"
    assert f"states multipoles on element {bare}" in stopped.value.detail
    _build(_declared(unbound, {"element": bare}))


@pytest.mark.parametrize(
    ("items", "detail"),
    [
        (lambda quad: [{"element": quad}, {"element": quad}], "a second time"),
        (lambda quad: [quad], "not a map"),
        (lambda quad: {"element": quad}, "is not a list"),
    ],
    ids=["twice", "bare-name", "map"],
)
def test_a_malformed_error_elements_list_fails_the_build(
    unbound: dict[str, Any], items: Any, detail: str
) -> None:
    block = {"solve": "periodic", "error_elements": items(unbound["quad"])}

    with pytest.raises(FacilityBuildError) as stopped:
        _build({**unbound, "settings": {"pyat": block}})

    assert (stopped.value.kind, stopped.value.record_id) == ("engine-invalid", MODEL)
    assert detail in stopped.value.detail


def test_a_declared_monitor_element_fails_the_build_by_name(demo: dict[str, Any]) -> None:
    monitor = fault_variables(demo["wiring"])[f"{demo['monitors'][0]}/offset"].element

    with pytest.raises(ErrorElementsError, match=rf"pyat\.error_elements names {monitor}, "):
        _build(_declared(demo, {"element": monitor}))


# -- default_scenarios, at the VA's boot ---------------------------------------------


def _demo_view(built_demo: Any, project: Path, extra: dict[str, Any]) -> Path:
    prefix = "data/simulator/"
    for name, data in built_demo.outputs[0].files.items():
        if name.startswith(prefix):
            target = project / "data" / "simulator" / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    view = project / "data" / "simulator"
    scenarios_file = view / "scenarios.json"
    document = json.loads(scenarios_file.read_text(encoding="utf-8"))
    document["scenarios"] = sorted(
        [*document["scenarios"], *({"name": n, **s} for n, s in extra.items())],
        key=lambda s: s["name"],
    )
    scenarios_file.write_text(json.dumps(document), encoding="utf-8")
    return view


def _config(project: Path, defaults: list[str]) -> dict[str, Any]:
    """The project's ``config.yml`` naming ``defaults``, written and returned."""
    import yaml

    config = {"simulation": {"default_scenarios": defaults}}
    (project / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


@pytest.mark.slow
def test_a_default_scenario_seeds_element_errors_at_boot(
    built_demo: Any, demo: dict[str, Any], tmp_path: Path
) -> None:
    from osprey.simulation.apply import activate_default_scenarios, resolve_simulation_state_dir
    from osprey_connectors.simulation.composite import Composite

    seeded = {"faults": {MODEL: {"writes": {f"{demo['quad']}/dx": 1.0e-4}}}}
    broken = {"faults": {MODEL: {"writes": {f"{demo['quad']}/dB1": 50.0}}}}
    view = _demo_view(built_demo, tmp_path, {"seeded": seeded, "broken": broken})
    reading = demo["monitors"][0]
    plain = Composite(view, clock=lambda: 1_760_000_000.0).get(reading)

    config = _config(tmp_path, ["seeded"])
    assert "seeded" in activate_default_scenarios(config, tmp_path)
    state = resolve_simulation_state_dir(config, tmp_path)
    booted = Composite(view, state_dir=state, clock=lambda: 1_760_000_000.0)

    assert booted.status(MODEL) == "ok"
    assert booted.get(reading) != pytest.approx(plain, abs=1e-12)

    other = tmp_path / "other"
    other_view = _demo_view(built_demo, other, {"broken": broken})
    config = _config(other, ["broken"])
    assert "broken" in activate_default_scenarios(config, other)
    failed = Composite(
        other_view, state_dir=resolve_simulation_state_dir(config, other), clock=lambda: 1.0
    )
    status = failed.status(MODEL)
    assert status.startswith("scenario broken: ")
    assert "element errors seeded" in status
    assert math.isnan(failed.get(reading))
