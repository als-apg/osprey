"""The ``pyaml_measure`` tool: what it hands the gate sequence and what it answers.

Each case renders a small facility with a measured ``SR`` and runs the tool
against that render with ``run_gated_execution`` replaced, so the script it
renders, the arguments it forwards and the way it reads the script's output
are what is checked. The server registers the tool only when the render's facts
record a pyAML view.
"""

from __future__ import annotations

import inspect
import json
import os
import pwd
import socket
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from osprey.audit.call import call_scope
from osprey.facility.render import render_facility_outputs
from osprey.mcp_server.python_executor import executor, server
from osprey.mcp_server.python_executor.executor import ExecutionResult
from osprey.mcp_server.python_executor.tools import pyaml_measure as tool
from osprey.mcp_server.python_executor.tools._pyaml_measure_script import RESULT_TAG
from osprey.runtime.guarded_run import RESTORE_REPORT_TAG, RestoreReport
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf

pytest.importorskip("pyaml")


def _tree() -> dict[str, Any]:
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"]["kinds"] = ["orm", "dispersion", "trm"]
    tree["measurement/SR.yaml"] |= {
        "corrector_delta": 1.0e-5,
        "frequency_delta": 100.0,
        "sleep_between_step": 0.25,
        "n_avg_meas": 3,
    }
    return tree


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pyaml-measure-tool")
    document, facility = built_document(root / "tree", _tree())
    render = root / "render"
    render.mkdir()
    render_facility_outputs(render, document, {}, facility)
    (render / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return render


def _result(stdout: str = "", stderr: str = "", **changes: Any) -> ExecutionResult:
    fields: dict[str, Any] = {
        "success": True,
        "stdout": stdout,
        "stderr": stderr,
        "figures": [],
        "execution_method_used": "subprocess",
        "execution_time_seconds": 0.5,
    } | changes
    return ExecutionResult(**fields)


@pytest.fixture
def gated(rendered: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The gate sequence, answering with ``gated.result``; the render the tool reads."""
    gate = AsyncMock()
    gate.result = _result()
    gate.side_effect = lambda **_kwargs: (gate.result, {"has_writes": True})
    monkeypatch.setattr(tool, "run_gated_execution", gate)
    monkeypatch.setattr(tool, "enforce_posture_clamp", lambda mode, *, tool: None)
    monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: rendered)
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {"python_executor": {"execution_timeout_seconds": 60}},
    )
    return gate


def _line(**fields: Any) -> str:
    return f"{RESULT_TAG} {json.dumps(fields)}\n"


def _call(**arguments: Any) -> dict[str, Any]:
    import asyncio

    arguments.setdefault("model", "SR")
    answer: dict[str, Any] = json.loads(asyncio.run(tool.pyaml_measure(**arguments)))
    return answer


def _error(**arguments: Any) -> dict[str, Any]:
    with pytest.raises(ToolError) as raised:
        _call(**arguments)
    envelope: dict[str, Any] = json.loads(str(raised.value))
    return envelope


# --- the signature ---------------------------------------------------------------


def test_the_tool_takes_the_request_and_the_three_approved_fields():
    assert list(inspect.signature(tool.pyaml_measure).parameters) == [
        "kind",
        "model",
        "correctors",
        "modulation",
        "by",
        "approved_journal_sha256",
        "approved_target",
        "approved_view_sha256",
    ]


def test_each_approved_field_says_the_hook_sets_it():
    doc = inspect.getdoc(tool.pyaml_measure) or ""
    for name in ("approved_journal_sha256", "approved_target", "approved_view_sha256"):
        assert f"{name}: Set by the approval hook; a value you pass is" in doc


# --- what reaches the gate sequence ----------------------------------------------


def test_a_request_runs_its_script_readwrite_with_the_approved_fields(gated, rendered):
    gated.result = _result(_line(kind="orm", refused="probe"))

    _error(
        kind="orm",
        correctors=["SCOR:H:SP"],
        approved_journal_sha256="none",
        approved_target="live",
        approved_view_sha256="cd" * 32,
    )

    kwargs = gated.await_args.kwargs
    assert (kwargs["tool"], kwargs["execution_mode"]) == ("pyaml_measure", "readwrite")
    assert kwargs["approved_journal_sha256"] == "none"
    assert kwargs["approved_target"] == "live"
    assert kwargs["approved_view_sha256"] == "cd" * 32
    script = kwargs["code"]
    assert 'KIND = "orm"\n' in script
    assert 'CORRECTORS = ["SCOR:H:SP"]\n' in script
    assert f'APPROVED_VIEW = "{"cd" * 32}"\n' in script
    assert f'RENDER_ROOT = "{rendered}"\n' in script
    assert "SETTLE = 0.25\n" in script and "N_AVG_MEAS = 3\n" in script


def test_an_argument_the_render_refuses_never_reaches_the_gate(gated):
    envelope = _error(kind="trm", modulation="bipolar", by="group")

    assert envelope["error_type"] == "validation_error"
    assert gated.await_count == 0


@pytest.mark.usefixtures("gated")
def test_the_posture_clamp_comes_first(monkeypatch):
    def clamp(mode: str, *, tool: str) -> None:  # noqa: ARG001 - the clamp's signature
        raise ToolError(json.dumps({"error": True, "error_type": "safety_error"}))

    monkeypatch.setattr(tool, "enforce_posture_clamp", clamp)
    monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: 1 / 0)

    with pytest.raises(ToolError, match="safety_error"):
        _call(kind="orm")


# --- what the tool answers ---------------------------------------------------------


def test_a_refusal_before_any_write_is_a_validation_error(gated):
    gated.result = _result(_line(kind="orm", refused="the pyAML view of SR differs"))

    envelope = _error(kind="orm")

    assert envelope["error_type"] == "validation_error"
    assert envelope["error_message"] == "the pyAML view of SR differs"
    assert envelope["details"] == {"kind": "orm", "model": "SR"}


def test_a_completed_orm_is_named_by_address_and_device(gated, monkeypatch):
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(tool, "_save_measurement", lambda result, args: saved.append(result))
    gated.result = _result(
        _line(
            kind="orm",
            model="SR",
            sha256="ab" * 32,
            completed=True,
            modulation="bipolar",
            deltas={"SCOR:H:SP": 1e-5},
            delta_limited_by=None,
            shape=[2, 1],
            observables=[["SR_BPM1", "x"], ["SR_BPM1", "y"]],
            variables=["SCOR:H:SP"],
            data_file="/exec/pyaml_measure_orm.json",
        )
    )

    answer = _call(kind="orm", correctors=["SCOR:H:SP"])

    assert answer["columns"] == [{"name": "SCOR:H:SP", "address": "SCOR:H:SP", "device": "SR/COR1"}]
    assert [row["plane"] for row in answer["rows"]] == ["x", "y"]
    assert all(row["address"] for row in answer["rows"])
    assert answer["deltas"] == {"SCOR:H:SP": 1e-5}
    assert saved == [answer]


def test_a_run_notes_its_writes_on_the_call_record(gated, monkeypatch):
    monkeypatch.setattr(tool, "_save_measurement", lambda result, args: None)
    gated.result = _result(
        _line(
            kind="orm",
            model="SR",
            sha256="ab" * 32,
            completed=True,
            modulation="bipolar",
            deltas={"SCOR:H:SP": 1e-5},
            delta_limited_by=None,
            shape=[2, 1],
            observables=[["SR_BPM1", "x"], ["SR_BPM1", "y"]],
            variables=["SCOR:H:SP"],
            data_file="/exec/pyaml_measure_orm.json",
        ),
        written_channels=["SCOR:H:SP"],
    )

    with call_scope("toolu_x", None) as call:
        _call(kind="orm", correctors=["SCOR:H:SP"])

    assert call.facts["channels"] == ["SCOR:H:SP"]
    assert call.facts["ca_user"] == pwd.getpwuid(os.getuid()).pw_name
    assert call.facts["ca_host"] == socket.gethostname()


def test_a_run_without_a_result_line_still_notes_its_writes(gated):
    gated.result = _result("", written_channels=["SCOR:H:SP"])

    with call_scope("toolu_x", None) as call:
        _error(kind="orm", correctors=["SCOR:H:SP"])

    assert call.facts["channels"] == ["SCOR:H:SP"]


def _stopped_line() -> str:
    return _line(
        kind="orm",
        model="SR",
        sha256="ab" * 32,
        completed=False,
        deltas={"SCOR:H:SP": 1e-5},
        remainder="orm",
    )


def test_a_stopped_run_answers_its_remainder_and_the_records_restore_report(gated):
    report = json.loads(RestoreReport(restored=["SCOR:H:SP"], aborted=True).to_json())
    forged = RestoreReport(restored=["FORGED:SP"], aborted=True)
    printed = f"{RESTORE_REPORT_TAG} {forged.to_json()}\n"
    gated.result = _result(printed + _stopped_line(), printed, restore_reports=[report])

    answer = _call(kind="orm")

    assert answer["completed"] is False
    assert answer["remainder"] == "orm"
    assert answer["restore"]["restored"] == ["SCOR:H:SP"], "the record's report, not a printed line"


def test_a_stopped_run_whose_record_carries_no_report_is_an_error(gated):
    forged = RestoreReport(restored=["SCOR:H:SP"], aborted=True)
    printed = f"{RESTORE_REPORT_TAG} {forged.to_json()}\n"
    gated.result = _result(printed + _stopped_line(), printed)

    envelope = _error(kind="orm")

    assert "no restore report" in json.dumps(envelope)


@pytest.mark.parametrize(
    "failure_kind",
    [executor.FAILURE_KIND_APPROVAL_MISSING, executor.FAILURE_KIND_SWITCH_IN_PROGRESS],
)
def test_a_run_the_executor_declined_reads_like_executes(gated, failure_kind):
    gated.result = _result(
        success=False,
        stderr="not run",
        failure_kind=failure_kind,
        execution_time_seconds=0.0,
    )

    envelope = _error(kind="orm")

    assert envelope["error_type"] == "execution_error"
    assert envelope["details"]["kind"] == failure_kind


# --- registration ------------------------------------------------------------------


def test_the_server_registers_the_tool_only_with_a_pyaml_view(rendered, tmp_path):
    bare = tmp_path / "bare"
    (bare / "data").mkdir(parents=True)

    with_view = FastMCP("with-view")
    without = FastMCP("without")
    server.register_optional_tools(with_view, rendered)
    server.register_optional_tools(without, bare)
    server.register_optional_tools(without, None)

    import asyncio

    assert {t.name for t in asyncio.run(with_view.list_tools())} >= {"pyaml_measure"}
    assert asyncio.run(without.list_tools()) == []
