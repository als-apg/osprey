"""What the ``pyaml_measure`` script printed, read back and labelled.

The parser reads the script's one result line from stdout: every result lists
the setpoints the call moved with their values before the call. The labels name
every row, column and moved setpoint by the address and device id it stands
for, through the pyAML view's one name mapping, on a render of a small
two-model facility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.render import render_facility_outputs
from osprey.mcp_server.python_executor.tools._pyaml_measure_result import (
    RESULT_TAG,
    MeasureResultError,
    label_result,
    measure_labels,
    parse_measure_result,
)
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf

SHA = "a" * 64


def _line(**fields: Any) -> str:
    return f"{RESULT_TAG} {json.dumps(fields)}\n"


def _orm(**changes: Any) -> dict[str, Any]:
    return {
        "kind": "orm",
        "model": "SR",
        "sha256": SHA,
        "completed": True,
        "deltas": {"SCOR:H:SP": 1e-5, "SCOR:V:SP": 1e-5},
        "shape": [2, 2],
        "observables": [["SR_BPM1", "x"], ["SR_BPM1", "y"]],
        "variables": ["SCOR:H:SP", "SCOR:V:SP"],
        "data_file": "/exec/pyaml_measure_orm.json",
        "moved": [["SCOR:H:SP", 0.25]],
    } | changes


# --- parsing ----------------------------------------------------------------------


def test_a_completed_orm_reads_its_shape_names_and_steps():
    result = parse_measure_result("noise\n" + _line(**_orm()), "orm")

    assert result == _orm()


def test_the_stepping_and_what_sized_each_step_are_carried():
    group = _orm(
        kind="trm",
        modulation="unipolar",
        by="group",
        deltas={"SR_QF": 2.5e-4},
        delta_basis="tune excursion",
        delta_limited_by=None,
        variables=["SR_QF"],
        shape=[2, 1],
        observables=[["BETATRON_TUNE", "x"], ["BETATRON_TUNE", "y"]],
    )
    crm = _orm(kind="crm", e_delta_used=1e-3, rf_span_hz=40.0, rf_limited_by="SR:RF:SP")

    assert parse_measure_result(_line(**group), "trm") == group
    parsed = parse_measure_result(_line(**crm), "crm")
    assert (parsed["e_delta_used"], parsed["rf_span_hz"], parsed["rf_limited_by"]) == (
        1e-3,
        40.0,
        "SR:RF:SP",
    )


def test_a_limit_that_is_not_an_address_is_refused():
    with pytest.raises(MeasureResultError, match="delta_limited_by"):
        parse_measure_result(_line(**_orm(delta_limited_by=3)), "orm")


def test_a_completed_chromaticity_monitor_reads_the_chromaticity():
    line = _line(
        kind="chromaticity_monitor",
        model="SR",
        sha256=SHA,
        completed=True,
        chromaticity=[1.5, -2.5],
        data_file="/exec/c.json",
        moved=[],
    )

    result = parse_measure_result(line, "chromaticity_monitor")

    assert result["chromaticity"] == [1.5, -2.5]
    assert "deltas" not in result


def _stopped_orm() -> dict[str, Any]:
    stopped = _orm(completed=False, remainder=["SCOR:V:SP"])
    for key in ("shape", "observables", "variables", "data_file"):
        del stopped[key]
    return stopped


def test_every_result_carries_what_it_moved():
    completed = parse_measure_result(_line(**_orm()), "orm")
    stopped = parse_measure_result(_line(**_stopped_orm()), "orm")

    assert completed["moved"] == [["SCOR:H:SP", 0.25]]
    assert stopped["moved"] == [["SCOR:H:SP", 0.25]]
    assert stopped["remainder"] == ["SCOR:V:SP"]
    assert "error" not in stopped


@pytest.mark.parametrize("completed", [True, False], ids=["completed", "stopped"])
def test_a_line_without_moved_is_an_error(completed: bool):
    line = _orm() if completed else _stopped_orm()
    del line["moved"]

    with pytest.raises(MeasureResultError, match="has no 'moved' field"):
        parse_measure_result(_line(**line), "orm")


@pytest.mark.parametrize(
    "moved", [[["SCOR:H:SP"]], [[1, 0.25]], "SCOR:H:SP", [["SCOR:H:SP", True]]]
)
def test_a_moved_that_is_not_address_value_pairs_is_an_error(moved: Any):
    with pytest.raises(MeasureResultError, match="moved"):
        parse_measure_result(_line(**_orm(moved=moved)), "orm")


def test_a_stopped_line_carries_its_error():
    line = _stopped_orm() | {"error": "KeyboardInterrupt"}

    assert parse_measure_result(_line(**line), "orm")["error"] == "KeyboardInterrupt"
    with pytest.raises(MeasureResultError, match="'error' is not a string"):
        parse_measure_result(_line(**(line | {"error": 3})), "orm")


def test_a_non_finite_before_value_is_spelled():
    moved = [["SCOR:H:SP", "nan"], ["SCOR:V:SP", "-inf"], ["RF:SP", 5.0e8]]

    result = parse_measure_result(_line(**_orm(moved=moved)), "orm")

    assert result["moved"] == moved


def test_a_stopped_run_with_no_partial_column_names_the_kind():
    stopped = {
        "kind": "chromaticity_monitor",
        "model": "SR",
        "sha256": SHA,
        "completed": False,
    }

    result = parse_measure_result(_line(**stopped, moved=[]), "chromaticity_monitor")

    assert result["remainder"] == "chromaticity_monitor"


def test_a_refusal_is_returned_as_the_refusal():
    result = parse_measure_result(_line(kind="orm", refused="view changed"), "orm")

    assert result == {"kind": "orm", "refused": "view changed"}


def test_the_last_result_line_wins():
    first = _line(**_orm(sha256="b" * 64))
    result = parse_measure_result(first + _line(**_orm()), "orm")

    assert result["sha256"] == SHA


@pytest.mark.parametrize(
    ("stdout", "match"),
    [
        ("", "printed no OSPREY_PYAML_MEASURE line"),
        (f"{RESULT_TAG} not json\n", "not valid JSON"),
        (f"{RESULT_TAG} [1]\n", "not a JSON object"),
        (_line(**_orm(kind="trm")), "is for kind 'trm'"),
        (_line(**_orm(shape=[3, 2])), "does not match"),
        (_line(**_orm(observables=[["SR_BPM1", "s"]], shape=[1, 2])), "observables"),
        (_line(**_orm(deltas={"SCOR:H:SP": float("nan")})), "not a finite number"),
        (_line(**_orm(completed="yes")), "'completed' is not a boolean"),
    ],
)
def test_a_malformed_result_is_an_error(stdout: str, match: str):
    with pytest.raises(MeasureResultError, match=match):
        parse_measure_result(stdout, "orm")


@pytest.mark.parametrize(
    ("plane", "read"), [("H", "x"), ("h", "x"), ("x", "x"), ("V", "y"), ("Y", "y")]
)
def test_every_plane_spelling_reads_back_as_x_or_y(plane: str, read: str):
    line = _line(**_orm(observables=[["SR_BPM1", plane], ["SR_BPM1", "y"]]))

    result = parse_measure_result(line, "orm")

    assert result["observables"] == [["SR_BPM1", read], ["SR_BPM1", "y"]]


def test_an_observable_without_a_plane_is_refused_by_name():
    line = _line(**_orm(observables=[["SR_BPM1", ""], ["SR_BPM1", "y"]]))

    with pytest.raises(MeasureResultError, match=r"observable 'SR_BPM1' has no plane"):
        parse_measure_result(line, "orm")


def test_an_observable_with_an_unknown_plane_is_refused_by_name():
    line = _line(**_orm(observables=[["SR_BPM1", "s"], ["SR_BPM1", "y"]]))

    with pytest.raises(MeasureResultError, match=r"observable 'SR_BPM1' has plane 's'"):
        parse_measure_result(line, "orm")


def test_an_observable_without_a_name_is_refused():
    line = _line(**_orm(observables=[["", "x"], ["SR_BPM1", "y"]]))

    with pytest.raises(MeasureResultError, match="has no name"):
        parse_measure_result(line, "orm")


def test_bytes_are_read_as_text():
    result = parse_measure_result(_line(**_orm()).encode(), "orm")

    assert result["model"] == "SR"


# --- labels -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def render(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("pyaml")
    root = tmp_path_factory.mktemp("pyaml-measure-result")
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"]["kinds"] = ["orm", "dispersion", "trm"]
    tree["measurement/SR.yaml"] |= {"corrector_delta": 1.0e-5, "frequency_delta": 100.0}
    document, facility = built_document(root / "tree", tree)
    out = root / "render"
    out.mkdir()
    render_facility_outputs(out, document, {}, facility)
    return out


def _labels(render: Path) -> Any:
    return measure_labels(render, "SR", render / "data/pyaml/SR/configuration.yaml")


def test_orm_rows_and_columns_are_labelled_by_address_and_device(render: Path):
    labelled = label_result(_orm(), _labels(render))

    assert labelled["rows"] == [
        {"name": "SR_BPM1", "plane": "x", "address": "BPM1:X", "device": "SR/BPM1"},
        {"name": "SR_BPM1", "plane": "y", "address": "BPM1:Y", "device": "SR/BPM1"},
    ]
    assert labelled["columns"] == [
        {"name": "SCOR:H:SP", "address": "SCOR:H:SP", "device": "SR/COR1"},
        {"name": "SCOR:V:SP", "address": "SCOR:V:SP", "device": "SR/COR1"},
    ]
    assert labelled["deltas"] == {"SCOR:H:SP": 1e-5, "SCOR:V:SP": 1e-5}
    assert "observables" not in labelled and "variables" not in labelled


def test_a_dispersion_column_is_the_rf_setpoint(render: Path):
    result = _orm(
        kind="dispersion",
        variables=["DEFAULT_RF_PLANT"],
        shape=[2, 1],
        deltas={"DEFAULT_RF_PLANT": 100.0},
    )

    labelled = label_result(result, _labels(render))

    assert labelled["columns"] == [
        {"name": "DEFAULT_RF_PLANT", "address": "RF:SP", "device": "SR/CAV"}
    ]
    assert labelled["deltas"] == {"RF:SP": 100.0}


def test_a_tune_response_by_group_names_the_group_and_the_tune_readbacks(render: Path):
    result = _orm(
        kind="trm",
        observables=[["BETATRON_TUNE", "x"], ["BETATRON_TUNE", "y"]],
        variables=["SR_Q"],
        shape=[2, 1],
        deltas={"SR_Q": 1e-3},
    )

    labelled = label_result(result, _labels(render))

    assert labelled["rows"] == [
        {"name": "BETATRON_TUNE", "plane": "x", "address": "SR:TUNE:X", "device": None},
        {"name": "BETATRON_TUNE", "plane": "y", "address": "SR:TUNE:Y", "device": None},
    ]
    assert labelled["columns"] == [{"name": "SR_Q", "group": "SR/Q"}]
    assert labelled["deltas"] == {"SR/Q": 1e-3}


def test_a_tune_response_by_magnet_names_each_quadrupole_setpoint(render: Path):
    result = _orm(
        kind="trm",
        observables=[["BETATRON_TUNE", "x"], ["BETATRON_TUNE", "y"]],
        variables=["QD:SP", "QF:SP"],
        deltas={"QD:SP": 1e-3, "QF:SP": 1e-3},
    )

    labelled = label_result(result, _labels(render))

    assert labelled["columns"] == [
        {"name": "QD:SP", "address": "QD:SP", "device": "SR/QD"},
        {"name": "QF:SP", "address": "QF:SP", "device": "SR/QF"},
    ]


def test_a_stopped_run_s_remainder_is_named_by_address(render: Path):
    stopped = {**_orm(completed=False), "remainder": ["SCOR:V:SP"]}

    assert label_result(stopped, _labels(render))["remainder"] == ["SCOR:V:SP"]


def test_moved_is_labelled_by_address_and_device(render: Path):
    result = _orm(moved=[["SCOR:H:SP", 0.25], ["RF:SP", "nan"], ["NOT:A:CHANNEL", 1.0]])

    labelled = label_result(result, _labels(render))

    assert labelled["moved"] == [
        {"address": "SCOR:H:SP", "device": "SR/COR1", "before": 0.25},
        {"address": "RF:SP", "device": "SR/CAV", "before": "nan"},
        {"address": "NOT:A:CHANNEL", "device": None, "before": 1.0},
    ]


def test_a_refusal_is_labelled_unchanged(render: Path):
    refusal = {"kind": "orm", "refused": "nope"}

    assert label_result(refusal, _labels(render)) == refusal
