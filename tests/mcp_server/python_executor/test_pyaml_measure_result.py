"""What the ``pyaml_measure`` script printed, read back and labelled.

The parser reads the script's one result line from stdout and the guarded run's
one restore line from stderr, through the runtime's own tag. The labels name
every row and column by the address and device id it stands for, through the
pyAML view's one name mapping, on a render of a small two-model facility.
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
from osprey.runtime.guarded_run import RESTORE_REPORT_TAG, RestoreReport
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
    } | changes


# --- parsing ----------------------------------------------------------------------


def test_a_completed_orm_reads_its_shape_names_and_steps():
    result = parse_measure_result("noise\n" + _line(**_orm()), "", "orm")

    assert result == _orm()


def test_a_completed_chromaticity_monitor_reads_the_chromaticity():
    line = _line(
        kind="chromaticity_monitor",
        model="SR",
        sha256=SHA,
        completed=True,
        chromaticity=[1.5, -2.5],
        data_file="/exec/c.json",
    )

    result = parse_measure_result(line, "", "chromaticity_monitor")

    assert result["chromaticity"] == [1.5, -2.5]
    assert "deltas" not in result


def test_a_stopped_run_reads_the_restore_report_from_stderr_through_the_runtime_tag():
    report = RestoreReport(restored=["SCOR:H:SP"], aborted=True)
    stderr = f"{RESTORE_REPORT_TAG} {report.to_json()}\n"
    stopped = _orm(completed=False, remainder=["SCOR:V:SP"])
    for key in ("shape", "observables", "variables", "data_file"):
        del stopped[key]

    result = parse_measure_result(_line(**stopped), stderr, "orm")

    assert result["restore"]["restored"] == ["SCOR:H:SP"]
    assert result["restore"]["aborted"] is True
    assert result["remainder"] == ["SCOR:V:SP"]
    assert RESTORE_REPORT_TAG == "OSPREY_GUARDED_RUN_RESTORE"


def test_a_stopped_run_with_no_partial_column_names_the_kind():
    report = RestoreReport(aborted=True)
    stopped = {
        "kind": "chromaticity_monitor",
        "model": "SR",
        "sha256": SHA,
        "completed": False,
    }

    result = parse_measure_result(
        _line(**stopped), f"{RESTORE_REPORT_TAG} {report.to_json()}\n", "chromaticity_monitor"
    )

    assert result["remainder"] == "chromaticity_monitor"


def test_a_stopped_run_without_a_restore_report_is_an_error():
    stopped = _orm(completed=False)
    with pytest.raises(MeasureResultError, match="no OSPREY_GUARDED_RUN_RESTORE report"):
        parse_measure_result(_line(**stopped), "", "orm")


def test_a_restore_line_on_stdout_is_not_the_report():
    """The guarded run prints its line on stderr; a line on stdout is the script's text."""
    report = RestoreReport(aborted=True)
    stdout = f"{RESTORE_REPORT_TAG} {report.to_json()}\n" + _line(**_orm(completed=False))
    with pytest.raises(MeasureResultError, match="printed no"):
        parse_measure_result(stdout, "", "orm")


def test_a_refusal_is_returned_as_the_refusal():
    result = parse_measure_result(_line(kind="orm", refused="view changed"), "", "orm")

    assert result == {"kind": "orm", "refused": "view changed"}


def test_the_last_result_line_wins():
    first = _line(**_orm(sha256="b" * 64))
    result = parse_measure_result(first + _line(**_orm()), "", "orm")

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
        parse_measure_result(stdout, "", "orm")


def test_bytes_are_read_as_text():
    result = parse_measure_result(_line(**_orm()).encode(), b"", "orm")

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


def test_a_refusal_is_labelled_unchanged(render: Path):
    refusal = {"kind": "orm", "refused": "nope"}

    assert label_result(refusal, _labels(render)) == refusal
