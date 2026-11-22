"""The engine plug-in contract the connectors package owns.

``osprey_connectors.simulation.engines`` names the entry-point group every
engine registers under, states what a plug-in module provides, carries the
calibration math wiring records share, and defines the exception that refuses
a request as a facility-file defect.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey_connectors.simulation import engines
from osprey_connectors.simulation.engines import (
    ENGINE_GROUP,
    OPTIONAL_NAMES,
    TIME_CONTRACT,
    Calibration,
    EnginePlugin,
    FacilityDefectError,
    Linear,
    NoInverse,
    Table,
    brho,
    check_inverse,
    curve_from_record,
    energy_factor,
    evaluate,
    field,
    shifted,
    to_hardware,
    to_physics,
)

#: The names every plug-in module states; the time contract adds none.
REQUIRED_NAMES = ("locate", "prepare", "start_values", "describe", "build")

#: The names by which one model's values reach another model's engine.
INPUT_NAMES = ("describe_input", "take_inputs")


def _function(*args, **kwargs):
    return None


def _module(name: str, names: tuple[str, ...]) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute in names:
        setattr(module, attribute, _function)
    return module


def _documented_names() -> list[str]:
    """The first-column names of the optional-name table in the protocol's docstring."""
    lines = (EnginePlugin.__doc__ or "").splitlines()
    rules = [index for index, line in enumerate(lines) if line.strip().startswith("=====")]
    assert len(rules) == 3, "the docstring states one simple table"
    rule = lines[rules[0]]
    indent = len(rule) - len(rule.lstrip())
    widths = [len(column) for column in rule.split()]
    names = []
    for line in lines[rules[1] + 1 : rules[2]]:
        assert len(line) <= len(rule), f"a table row runs past its last column: {line!r}"
        cell = line[indent : indent + widths[0]]
        match = re.match(r"``([A-Za-z_]+)", cell)
        if match:
            names.append(match.group(1))
    return names


# -- The group --------------------------------------------------------------


def test_the_group_is_named_after_the_module_that_owns_it():
    assert ENGINE_GROUP == "osprey_connectors.simulation.engines"
    assert ENGINE_GROUP == engines.__name__


def test_the_module_states_the_group_once():
    source = Path(engines.__file__).read_text(encoding="utf-8")
    assert source.count('"osprey_connectors.simulation.engines"') == 1
    assert '"osprey.simulation.engines"' not in source


# -- The protocol -----------------------------------------------------------


def test_a_module_with_the_required_names_satisfies_the_contract():
    assert isinstance(_module("five_functions", REQUIRED_NAMES), EnginePlugin)


@pytest.mark.parametrize("missing", REQUIRED_NAMES)
def test_a_module_missing_a_required_name_does_not(missing: str):
    names = tuple(name for name in REQUIRED_NAMES if name != missing)
    assert not isinstance(_module("four_functions", names), EnginePlugin)


def test_the_time_contract_does_not_grow_the_required_names():
    for name in TIME_CONTRACT + ("deckless_start_values",) + INPUT_NAMES:
        assert not hasattr(EnginePlugin, name)


def test_the_time_contract_names_are_optional_names():
    assert TIME_CONTRACT == ("advance", "next_deadline", "emitted", "client_write")
    assert set(TIME_CONTRACT) <= set(OPTIONAL_NAMES)
    assert "deckless_start_values" in OPTIONAL_NAMES
    assert not set(REQUIRED_NAMES) & set(OPTIONAL_NAMES)


def test_the_input_names_are_optional_names_outside_the_time_contract():
    assert set(INPUT_NAMES) <= set(OPTIONAL_NAMES)
    assert not set(INPUT_NAMES) & set(TIME_CONTRACT)


def test_a_time_driven_engine_with_inputs_satisfies_the_contract():
    module = _module("time_driven", REQUIRED_NAMES + TIME_CONTRACT + INPUT_NAMES)
    module.deckless_start_values = True
    assert isinstance(module, EnginePlugin)


def test_the_documented_optional_names_match_the_module():
    documented = _documented_names()
    assert len(documented) == len(set(documented))
    assert sorted(documented) == sorted(OPTIONAL_NAMES)


def test_the_time_contract_paragraph_states_how_inputs_arrive():
    text = " ".join((EnginePlugin.__doc__ or "").split())
    paragraph = text[text.index("The four names of") :]
    for phrase in (
        "``take_inputs`` only stores",
        "``baseline=True`` means",
        "immediately before its ``advance``",
        "every built child",
        "every pass",
    ):
        assert phrase in paragraph, phrase


# -- The facility-defect refusal --------------------------------------------


def test_a_facility_defect_is_a_plain_exception():
    assert issubclass(FacilityDefectError, Exception)
    assert not issubclass(FacilityDefectError, ValueError)
    assert str(FacilityDefectError("wiring names no element QF1")) == "wiring names no element QF1"


def test_the_response_matrix_row_names_the_facility_defect_refusal():
    text = " ".join((EnginePlugin.__doc__ or "").split())
    assert "``FacilityDefectError`` refuses it as a facility-file defect" in text


def test_the_contract_names_no_framework_description_type():
    source = Path(engines.__file__).read_text(encoding="utf-8")
    for name in ("FacilityBuildError", "FacilityDescription", "FacilityFileError"):
        assert name not in source, name


# -- What importing it loads ------------------------------------------------


def test_the_contract_imports_only_the_standard_library_and_lume_for_typing():
    # The layers contract cannot hold a module inside the simulation layer as a
    # leaf of its own, so this is where the leaf is held: nothing from the
    # package, nothing from the framework, lume only for an annotation.
    tree = ast.parse(Path(engines.__file__).read_text(encoding="utf-8"))
    runtime: set[str] = set()
    typing_only: set[str] = set()
    for node in tree.body:
        is_typing = isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"
        statements = node.body if is_typing else [node]
        for statement in statements:
            if isinstance(statement, ast.Import):
                names = {alias.name for alias in statement.names}
            elif isinstance(statement, ast.ImportFrom):
                names = {statement.module or ""}
            else:
                continue
            (typing_only if is_typing else runtime).update(names)
    roots = {name.split(".")[0] for name in runtime}
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}, roots
    assert {name.split(".")[0] for name in typing_only} <= {"lume"}, typing_only


def test_importing_the_contract_loads_neither_an_engine_library_nor_the_framework():
    code = (
        "import sys\n"
        "import osprey_connectors.simulation.engines\n"
        "loaded = [m for m in ('at', 'lume', 'lume_pyat', 'numpy', 'osprey')"
        " if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


# -- Calibration ------------------------------------------------------------


def test_a_linear_curve_is_gain_times_value_plus_offset():
    assert evaluate(Linear(gain=2.0, offset=1.0), 3.0) == 7.0


def test_a_table_interpolates_and_extrapolates_along_its_end_segments():
    curve = Table(grid=(0.0, 1.0, 3.0), values=(0.0, 10.0, 30.0))
    assert evaluate(curve, 0.5) == pytest.approx(5.0)
    assert evaluate(curve, 2.0) == pytest.approx(20.0)
    assert evaluate(curve, -1.0) == pytest.approx(-10.0)
    assert evaluate(curve, 4.0) == pytest.approx(40.0)


def test_a_descending_table_reads_the_same_curve():
    rising = Table(grid=(0.0, 1.0, 3.0), values=(0.0, 10.0, 40.0))
    falling = Table(grid=(3.0, 1.0, 0.0), values=(40.0, 10.0, 0.0))
    for x in (-1.0, 0.5, 2.0, 5.0):
        assert evaluate(falling, x) == pytest.approx(evaluate(rising, x))


@pytest.mark.parametrize(
    "record",
    [
        {"linear": {"gain": 2, "offset": "0.5"}},
        SimpleNamespace(linear=SimpleNamespace(gain=2, offset=0.5), table=None),
    ],
)
def test_a_linear_record_reads_by_key_or_by_attribute(record: object):
    assert curve_from_record(record) == Linear(gain=2.0, offset=0.5)


def test_a_table_record_becomes_float_tuples():
    curve = curve_from_record({"table": {"grid": [0, 1], "values": [2, 4]}})
    assert curve == Table(grid=(0.0, 1.0), values=(2.0, 4.0))


@pytest.mark.parametrize("record", [None, {}, {"poly": [1, 2]}])
def test_a_record_stating_no_shape_is_no_curve(record: object):
    assert curve_from_record(record) is None


def test_a_field_of_nothing_is_none():
    assert field(None, "gain") is None
    assert field({"gain": 1}, "offset") is None
    assert field(SimpleNamespace(gain=1), "gain") == 1


@pytest.mark.parametrize(
    "curve",
    [Linear(gain=2.0, offset=1.0), Table(grid=(0.0, 1.0, 3.0), values=(0.0, 10.0, 30.0))],
)
def test_a_forward_curve_shifted_answers_later_what_it_answered_before(curve: Calibration):
    moved = shifted(curve, 0.25)
    for x in (-0.5, 0.5, 2.0):
        assert evaluate(moved, x + 0.25) == pytest.approx(evaluate(curve, x))


@pytest.mark.parametrize(
    "curve",
    [Linear(gain=2.0, offset=1.0), Table(grid=(0.0, 1.0, 3.0), values=(0.0, 10.0, 30.0))],
)
def test_a_way_back_shifted_on_its_values_answers_more(curve: Calibration):
    moved = shifted(curve, 0.25, on_values=True)
    for x in (-0.5, 0.5, 2.0):
        assert evaluate(moved, x) == pytest.approx(evaluate(curve, x) + 0.25)


def test_a_zero_shift_is_the_curve_itself():
    curve = Linear(gain=1.0, offset=0.0)
    assert shifted(curve, 0.0) is curve


def test_no_calibration_passes_values_through():
    assert to_physics(None, 3) == 3.0
    assert to_hardware(None, None, 3) == 3.0


def test_a_linear_curve_maps_back_algebraically():
    curve = Linear(gain=2.0, offset=1.0)
    assert to_hardware(curve, None, to_physics(curve, 3.0)) == pytest.approx(3.0)


def test_a_stated_inverse_wins_over_the_algebraic_one():
    curve = Linear(gain=2.0, offset=1.0)
    assert to_hardware(curve, Linear(gain=10.0, offset=0.0), 1.0) == 10.0


@pytest.mark.parametrize(
    ("curve", "reason"),
    [
        (Table(grid=(0.0, 1.0), values=(0.0, 1.0)), "table"),
        (Linear(gain=0.0, offset=1.0), "zero-gain"),
    ],
)
def test_a_curve_with_no_way_back_is_refused(curve: Calibration, reason: str):
    with pytest.raises(NoInverse) as raised:
        to_hardware(curve, None, 1.0)
    assert raised.value.reason == reason
    assert isinstance(raised.value, ValueError)
    with pytest.raises(NoInverse):
        check_inverse(curve, None)
    check_inverse(curve, Linear(gain=1.0, offset=0.0))


def test_the_rigidity_counts_the_rest_mass():
    # 1.9 GeV kinetic: about 6.3394 T m, against 6.3377 for the massless E / c.
    assert brho(1.9) == pytest.approx(6.33942, abs=1e-5)
    assert brho(1.9) > 1.9 * engines.RIGIDITY_PER_MOMENTUM


def test_the_energy_factor_is_one_at_the_deck_energy_and_falls_above_it():
    assert energy_factor(1.9, 1.9) == pytest.approx(1.0)
    assert energy_factor(2.0, 1.9) < 1.0
    assert energy_factor(2.0, 1.9) == pytest.approx(brho(1.9) / brho(2.0))
