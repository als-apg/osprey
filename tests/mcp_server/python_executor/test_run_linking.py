"""The executor tools link every artifact of one call into a run, keyed on its tool-use id.

``execute`` and ``execute_file`` stamp ``exec_id = current_tool_use_id()`` on
every artifact the call saves — figures and ``save_artifact`` outputs
(``run_part="output"``), the notebook (``code``) and the result record, which is
the run's head (``result``) and stores the run's status: ``succeeded``,
``failed``, or ``not_run`` when nothing ran. A refusal raised before the launch
leaves one ``not_run`` head with the proposed code and the message; a refusal
raised after it leaves a ``failed`` one; the error still reaches the caller.
``pyaml_measure`` shares the gate sequence but writes no head.

Each case drives the real tool inside a ``call_scope`` with ``execute_code``
stubbed below the gates, and reads the store back.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from osprey.audit.call import call_scope
from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_SETUP,
    FAILURE_KIND_SWITCH_IN_PROGRESS,
    ExecutionResult,
)
from osprey.stores.artifact_store import get_artifact_store
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

TOOL_USE_ID = "toolu_run_linking_01"

_NO_PATTERNS = {"has_writes": False, "has_reads": False, "detected_patterns": {}}


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    """Anchor the agent data root and the working directory in ``tmp_path``."""
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(tmp_path / "agent_data"))
    monkeypatch.chdir(tmp_path)


def _result(success=True, stdout="", stderr="", **changes) -> ExecutionResult:
    fields = {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "execution_method_used": "subprocess",
        "execution_time_seconds": 0.1,
    } | changes
    return ExecutionResult(**fields)


@contextmanager
def _stubbed(result: ExecutionResult | None = None, *, scope: str | None = TOOL_USE_ID):
    """No detected patterns, ``execute_code`` answering *result*, a call scope for *scope*."""
    launch = AsyncMock(return_value=result if result is not None else _result(stdout="ok\n"))
    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection."
            "detect_control_system_operations",
            return_value=dict(_NO_PATTERNS),
        ),
        patch("osprey.mcp_server.python_executor.executor.execute_code", launch),
    ):
        if scope is None:
            yield launch
        else:
            with call_scope(scope, None):
                yield launch


def _execute():
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return get_tool_fn(execute)


def _execute_file():
    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    return get_tool_fn(execute_file)


def _entries():
    store = get_artifact_store()
    store._refresh_if_stale()
    return store.list_entries()


def _heads(entries=None):
    return [e for e in (entries or _entries()) if e.metadata.get("run_part") == "result"]


def _payload(entry) -> dict:
    return json.loads(get_artifact_store().get_file_path(entry.id).read_text("utf-8"))


def _only_head() -> object:
    heads = _heads()
    assert len(heads) == 1, [h.metadata for h in heads]
    return heads[0]


# --- a run that ran ----------------------------------------------------------------


async def test_a_successful_run_links_its_figure_output_notebook_and_head(tmp_path):
    figure = tmp_path / "fig_1.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    saved = tmp_path / "table.csv"
    saved.write_text("a,b\n1,2\n", encoding="utf-8")
    result = _result(
        stdout="done\n",
        figures=[figure],
        artifacts=[
            {
                "path": saved,
                "artifact_type": "file",
                "title": "Table",
                "description": "a table",
                "mime_type": "text/csv",
                "category": "legacy",
            }
        ],
    )

    with _stubbed(result), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await _execute()(code="print('done')", description="linked run")

    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    entries = _entries()
    assert {e.exec_id for e in entries} == {TOOL_USE_ID}
    parts = sorted(e.metadata.get("run_part") for e in entries)
    assert parts == ["code", "output", "output", "result"]

    fig = next(e for e in entries if e.artifact_type == "image")
    assert fig.metadata["run_part"] == "output"
    assert fig.metadata["figure"] is True
    out = next(e for e in entries if e.title == "Table")
    assert out.metadata["run_part"] == "output"
    assert "figure" not in out.metadata
    assert out.category == ""
    notebook = next(e for e in entries if e.artifact_type == "notebook")
    assert notebook.metadata["run_part"] == "code"

    head = _only_head()
    assert head.metadata["run_status"] == "succeeded"
    assert head.category == ""
    assert get_artifact_store().run_group(TOOL_USE_ID).head_id == head.id
    assert get_artifact_store().run_group(TOOL_USE_ID).run_status == "succeeded"


async def test_a_failed_run_stores_a_failed_head_and_raises():
    with (
        _stubbed(_result(success=False, stderr="Traceback ...\nValueError: boom")),
        assert_raises_error(error_type="execution_error"),
    ):
        await _execute()(code="raise ValueError('boom')", description="failing run")

    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "failed"
    assert _payload(head)["code"] == "raise ValueError('boom')"


@pytest.mark.parametrize("kind", [FAILURE_KIND_SETUP, FAILURE_KIND_SWITCH_IN_PROGRESS])
async def test_a_run_that_never_started_stores_a_not_run_head(kind):
    never = _result(success=False, stderr="", failure_kind=kind, execution_time_seconds=None)
    with _stubbed(never), pytest.raises(Exception):
        await _execute()(code="print(1)", description="never started")

    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "not_run"


async def test_save_output_false_still_stores_the_head():
    with _stubbed():
        answer = await _execute()(code="print('ok')", description="inline", save_output=False)

    assert json.loads(answer)["stdout"] == "ok\n"
    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "succeeded"


async def test_without_a_tool_use_id_each_part_stands_alone():
    with _stubbed(scope=None):
        await _execute()(code="print('ok')", description="unlinked")

    entries = _entries()
    assert entries
    assert {e.exec_id for e in entries} == {""}
    head = _only_head()
    assert head.metadata["run_status"] == "succeeded"
    assert "call_id" not in head.metadata


# --- refusals ------------------------------------------------------------------------


async def test_a_tool_level_check_refusal_stores_one_not_run_head():
    with _stubbed() as launch, assert_raises_error(error_type="validation_error"):
        await _execute()(code="   ", description="empty")

    launch.assert_not_awaited()
    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "not_run"
    data = _payload(head)
    assert data["code"] == "   "
    assert data["error"] == "No code provided."


async def test_a_gate_refusal_stores_one_not_run_head_with_the_proposed_code():
    code = "eval('1 + 1')"
    with _stubbed() as launch, assert_raises_error(error_type="safety_error") as raised:
        await _execute()(code=code, description="unsafe")

    launch.assert_not_awaited()
    entries = _entries()
    assert len(entries) == 1
    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "not_run"
    data = _payload(head)
    assert data["code"] == code
    assert data["error"] == raised["envelope"]["error_message"]
    assert head.summary["has_errors"] is True


async def test_an_execute_file_path_refusal_stores_one_not_run_head(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osprey.mcp_server.python_executor.executor._resolve_project_root", lambda: tmp_path
    )
    with _stubbed() as launch, assert_raises_error(error_type="file_not_found"):
        await _execute_file()(file_path="missing.py", description="no such file")

    launch.assert_not_awaited()
    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.tool_source == "execute_file"
    assert head.metadata["run_status"] == "not_run"
    data = _payload(head)
    assert data["code"] == ""
    assert data["file_path"] == "missing.py"
    assert "File not found" in data["error"]


async def test_an_execute_file_run_links_the_file_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osprey.mcp_server.python_executor.executor._resolve_project_root", lambda: tmp_path
    )
    (tmp_path / "script.py").write_text("print('from file')\n", encoding="utf-8")
    with _stubbed():
        await _execute_file()(file_path="script.py", description="file run")

    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "succeeded"
    assert _payload(head)["code"] == "print('from file')\n"


async def test_a_runtime_refusal_after_launch_stores_a_failed_head():
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    refused = _result(
        success=False, stderr=f"PermissionError: write refused in {READONLY_REFUSAL_MARKER}"
    )
    with _stubbed(refused) as launch, assert_raises_error(error_type="execution_error"):
        await _execute()(code="print(1)", description="refused mid-run")

    launch.assert_awaited_once()
    head = _only_head()
    assert head.metadata["run_status"] == "failed"


async def test_a_tool_error_raised_after_launch_stores_a_failed_head(monkeypatch):
    from osprey.mcp_server.errors import make_error
    from osprey.mcp_server.python_executor.tools import _execution_gates

    async def _refuse(**_kwargs):
        make_error("write_refused", "Refused after the launch.", [])

    monkeypatch.setattr(_execution_gates, "report_runtime_refusal", _refuse)
    with _stubbed() as launch, assert_raises_error(error_type="write_refused"):
        await _execute()(code="print(1)", description="refused after launch")

    launch.assert_awaited_once()
    head = _only_head()
    assert head.exec_id == TOOL_USE_ID
    assert head.metadata["run_status"] == "failed"
    data = _payload(head)
    assert data["code"] == "print(1)"
    assert data["error"] == "Refused after the launch."


# --- pyaml_measure -------------------------------------------------------------------


async def test_pyaml_measure_writes_no_head(monkeypatch):
    from osprey.mcp_server.python_executor.tools import _execution_gates
    from osprey.mcp_server.python_executor.tools import pyaml_measure as tool

    pytest.importorskip("pyaml")
    seen: dict = {}
    real = _execution_gates.run_gated_execution

    async def _gate(**kwargs):
        seen.update(kwargs)
        return await real(**kwargs)

    args = SimpleNamespace(kind="orm", model="SR", modulation="bipolar", by=None, correctors=None)
    monkeypatch.setattr(tool, "run_gated_execution", _gate)
    monkeypatch.setattr(tool, "enforce_posture_clamp", lambda mode, *, tool: None)
    monkeypatch.setattr(_execution_gates, "enforce_posture_clamp", lambda mode, *, tool: None)
    monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: Path("."))
    monkeypatch.setattr("osprey_connectors.workspace.load_osprey_config", lambda: {})
    monkeypatch.setattr(tool, "resolve_measure_args", lambda *a, **k: args)
    monkeypatch.setattr(tool, "_script", lambda *a, **k: None)
    monkeypatch.setattr(tool, "build_measure_script", lambda _script: "eval('1')")

    with _stubbed() as launch, assert_raises_error(error_type="safety_error"):
        await tool.pyaml_measure(kind="orm", model="SR")

    launch.assert_not_awaited()
    assert "on_launch" not in seen
    assert _heads() == []
