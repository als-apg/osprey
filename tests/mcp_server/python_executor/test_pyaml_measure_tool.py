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
        "osprey_connectors.workspace.load_osprey_config",
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


def test_the_tool_takes_the_request_and_the_two_approved_fields():
    assert list(inspect.signature(tool.pyaml_measure).parameters) == [
        "kind",
        "model",
        "correctors",
        "modulation",
        "by",
        "approved_target",
        "approved_view_sha256",
    ]


def test_each_approved_field_says_the_hook_sets_it():
    doc = inspect.getdoc(tool.pyaml_measure) or ""
    for name in ("approved_target", "approved_view_sha256"):
        assert f"{name}: Set by the approval hook; a value you pass is" in doc


# --- what reaches the gate sequence ----------------------------------------------


def test_a_request_runs_its_script_readwrite_with_the_approved_fields(gated, rendered):
    gated.result = _result(_line(kind="orm", refused="probe"))

    _error(
        kind="orm",
        correctors=["SCOR:H:SP"],
        approved_target="live",
        approved_view_sha256="cd" * 32,
    )

    kwargs = gated.await_args.kwargs
    assert (kwargs["tool"], kwargs["execution_mode"]) == ("pyaml_measure", "readwrite")
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
    monkeypatch.setattr(
        tool, "_save_measurement", lambda result, args, folder: saved.append(result)
    )
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
            moved=[["SCOR:H:SP", 0.25]],
        )
    )

    answer = _call(kind="orm", correctors=["SCOR:H:SP"])

    assert answer["columns"] == [{"name": "SCOR:H:SP", "address": "SCOR:H:SP", "device": "SR/COR1"}]
    assert [row["plane"] for row in answer["rows"]] == ["x", "y"]
    assert all(row["address"] for row in answer["rows"])
    assert answer["deltas"] == {"SCOR:H:SP": 1e-5}
    assert answer["moved"] == [{"address": "SCOR:H:SP", "device": "SR/COR1", "before": 0.25}]
    assert saved == [answer]


def _completed_orm_line(data_file: str, observables: list[list[str]] | None = None) -> str:
    return _line(
        kind="orm",
        model="SR",
        sha256="ab" * 32,
        completed=True,
        modulation="bipolar",
        deltas={"SCOR:H:SP": 1e-5},
        delta_limited_by=None,
        shape=[2, 1],
        observables=observables
        if observables is not None
        else [["SR_BPM1", "x"], ["SR_BPM1", "y"]],
        variables=["SCOR:H:SP"],
        data_file=data_file,
        moved=[["SCOR:H:SP", 0.25]],
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What reaches the artifact store: each saved file's name and bytes."""
    saved: list[dict[str, Any]] = []

    def save_file(**fields: Any) -> Any:
        saved.append(fields)
        return type("Entry", (), {"id": f"artifact-{len(saved)}"})()

    monkeypatch.setattr(
        "osprey.stores.artifact_store.get_artifact_store",
        lambda: type("Store", (), {"save_file": staticmethod(save_file)})(),
    )
    return saved


def test_the_measurement_file_saved_is_the_one_in_the_runs_own_folder(gated, store, tmp_path):
    folder = tmp_path / "execution"
    folder.mkdir()
    measured = folder / "pyaml_measure_orm.json"
    measured.write_text('{"matrix": [[1.0], [2.0]]}', encoding="utf-8")
    gated.result = _result(_completed_orm_line(str(measured.resolve())), execution_folder=folder)

    answer = _call(kind="orm", correctors=["SCOR:H:SP"])

    ((saved,),) = [store]
    assert saved["file_content"] == measured.read_bytes()
    assert saved["filename"] == "pyaml_measure_orm.json"
    assert answer["artifact_id"] == "artifact-1"
    assert "data_file_refused" not in answer


@pytest.mark.parametrize("named", ["outside", "other-name"])
def test_a_printed_measurement_file_outside_the_runs_folder_is_never_opened(
    gated, store, tmp_path, named
):
    folder = tmp_path / "execution"
    folder.mkdir()
    (folder / "pyaml_measure_orm.json").write_text("{}", encoding="utf-8")
    secret = tmp_path / "secret.json" if named == "outside" else folder / "other.json"
    secret.write_text('{"token": "x"}', encoding="utf-8")
    gated.result = _result(_completed_orm_line(str(secret.resolve())), execution_folder=folder)

    answer = _call(kind="orm", correctors=["SCOR:H:SP"])

    assert store == []
    assert answer["data_file"] is None
    assert "artifact_id" not in answer
    assert str(secret.resolve()) in answer["data_file_refused"]


def test_a_run_notes_its_writes_on_the_call_record(gated, monkeypatch):
    monkeypatch.setattr(tool, "_save_measurement", lambda result, args, folder: None)
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
            moved=[["SCOR:H:SP", 0.25]],
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


_PLANE_LESS = [["SR_BPM1", ""], ["SR_BPM1", "y"]]


def test_a_completed_run_whose_result_is_unreadable_names_its_saved_data(gated, tmp_path):
    folder = tmp_path / "execution"
    folder.mkdir()
    measured = folder / "pyaml_measure_orm.json"
    measured.write_text("{}", encoding="utf-8")
    line = _completed_orm_line(str(measured.resolve()), observables=_PLANE_LESS)
    gated.result = _result(line, execution_folder=folder)

    envelope = _error(kind="orm", correctors=["SCOR:H:SP"])

    assert envelope["error_type"] == "execution_error"
    assert "ended without a usable result" in envelope["error_message"]
    assert str(measured.resolve()) in envelope["error_message"]
    assert envelope["details"]["data_file"] == str(measured.resolve())
    assert "Nothing was stepped" not in json.dumps(envelope)


def test_a_failed_exit_after_the_run_saved_its_data_names_it(gated, tmp_path):
    folder = tmp_path / "execution"
    folder.mkdir()
    measured = folder / "pyaml_measure_orm.json"
    measured.write_text("{}", encoding="utf-8")
    line = _completed_orm_line(str(measured.resolve()), observables=_PLANE_LESS)
    gated.result = _result(
        line, "Traceback: the script failed", success=False, execution_folder=folder
    )

    envelope = _error(kind="orm", correctors=["SCOR:H:SP"])

    assert envelope["error_type"] == "execution_error"
    assert envelope["details"]["data_file"] == str(measured.resolve())
    assert "Nothing was stepped" not in json.dumps(envelope)


@pytest.mark.parametrize("success", [True, False])
def test_a_run_that_saved_nothing_names_no_data_file(gated, tmp_path, success):
    folder = tmp_path / "execution"
    folder.mkdir()
    gated.result = _result("", "the script failed", success=success, execution_folder=folder)

    envelope = _error(kind="orm", correctors=["SCOR:H:SP"])

    assert envelope["error_type"] == "execution_error"
    assert "data_file" not in envelope["details"]


def _stopped_line(**changes: Any) -> str:
    return _line(
        **{
            "kind": "orm",
            "model": "SR",
            "sha256": "ab" * 32,
            "completed": False,
            "deltas": {"SCOR:H:SP": 1e-5},
            "remainder": "orm",
            "moved": [["SCOR:H:SP", 0.25]],
        }
        | changes
    )


def test_a_stopped_run_answers_its_remainder_and_what_it_moved(gated):
    gated.result = _result(_stopped_line())

    answer = _call(kind="orm")

    assert answer["completed"] is False
    assert answer["remainder"] == "orm"
    assert answer["moved"] == [{"address": "SCOR:H:SP", "device": "SR/COR1", "before": 0.25}]
    assert "restore" not in answer


def test_a_failed_run_that_printed_its_line_answers_it_as_a_stopped_result(gated):
    line = _stopped_line(error="the tune readback went away")
    gated.result = _result(line, "Traceback: RuntimeError", success=False)

    answer = _call(kind="orm")

    assert answer["completed"] is False
    assert answer["error"] == "the tune readback went away"
    assert answer["moved"] == [{"address": "SCOR:H:SP", "device": "SR/COR1", "before": 0.25}]


@pytest.mark.parametrize("count", [2, 21])
def test_a_killed_run_names_the_channels_it_attempted(gated, count):
    channels = [f"SCOR:{index:02d}:SP" for index in range(count)]
    gated.result = _result(
        "",
        "Execution timed out after 60 seconds",
        success=False,
        failure_kind=executor.FAILURE_KIND_TIMEOUT,
        written_channels=channels,
    )

    envelope = _error(kind="orm")

    assert envelope["details"]["written_channels"] == channels
    assert envelope["details"]["kind"] == "timeout"
    message = envelope["error_message"]
    assert message.startswith("The run ended before it could list its setpoints;")
    assert "left where the run stopped" in message
    if count <= 20:
        assert ", ".join(channels) in message
    else:
        assert f"{count} channels" in message and channels[0] not in message


#: What a run on a held target says, naming its holder.
_BUSY = (
    "another guarded run is live under /repo/var/guarded_run/va "
    "(pid 31337, since 2026-10-02T09:30:00Z)"
)


@pytest.mark.parametrize(
    "failure_kind",
    [
        executor.FAILURE_KIND_APPROVAL_MISSING,
        executor.FAILURE_KIND_SWITCH_IN_PROGRESS,
        executor.FAILURE_KIND_GUARDED_RUN_BUSY,
    ],
)
def test_a_run_the_executor_declined_reads_like_executes(gated, failure_kind):
    busy = failure_kind == executor.FAILURE_KIND_GUARDED_RUN_BUSY
    gated.result = _result(
        success=False,
        stderr=_BUSY if busy else "not run",
        failure_kind=failure_kind,
        execution_time_seconds=0.0,
    )

    envelope = _error(kind="orm")

    assert envelope["error_type"] == "execution_error"
    assert envelope["details"]["kind"] == failure_kind
    if busy:
        assert "pid 31337" in envelope["error_message"]


# --- registration ------------------------------------------------------------------


def test_the_server_registers_both_tools_only_with_a_pyaml_view(rendered, tmp_path):
    bare = tmp_path / "bare"
    (bare / "data").mkdir(parents=True)

    with_view = FastMCP("with-view")
    without = FastMCP("without")
    server.register_optional_tools(with_view, rendered)
    server.register_optional_tools(without, bare)
    server.register_optional_tools(without, None)

    import asyncio

    assert {t.name for t in asyncio.run(with_view.list_tools())} == {
        "pyaml_measure",
        "model_response_matrix",
    }
    assert asyncio.run(without.list_tools()) == []
