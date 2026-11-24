"""The python executor refuses agent code on a deployment whose build lacks the protected-path list.

A deployment is a resolved project root that holds ``profile.yml``. For one,
:func:`~osprey.mcp_server.python_executor.executor.read_run_protected_entries`
reads the configuration file at the resolved location directly — once per
execution request, at the outermost entry — and either returns the
``protected_paths`` entries or refuses, naming ``osprey build``: when the key is
absent or malformed, when the file is unreadable or unparseable, or when there
is no build output at all. Outside a deployment it never refuses.

The gate chain reads once, right after the posture clamp, and hands the entries
to the pre-run check and the launch; both measurement tools read first and hand
the entries on; ``execute_code`` reads for itself only when handed none, and
then refuses with its own failure kind before an execution folder exists. The
refusal is filed as a refused decision under its own reason, and its operator
alert does not speak of a write.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from fastmcp.exceptions import ToolError

import osprey.mcp_server.python_executor.executor as executor
from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_NO_PROTECTED_PATHS,
    NOT_STARTED_FAILURE_KINDS,
    ExecutionResult,
    ProtectedPathsRefusal,
    read_run_protected_entries,
)
from osprey.mcp_server.python_executor.tools import _execution_gates as gates
from osprey.utils.workspace import load_protected_paths
from osprey_connectors.workspace import load_osprey_config, reset_config_cache
from tests.mcp_server.conftest import get_tool_fn

CONFIG = "build/config.yml"

#: The package whose reads of the config file the single-read test counts.
EXECUTOR_PACKAGE = Path(executor.__file__).parent


class _Deployment:
    """A scratch deployment: ``profile.yml`` at the root, the render under ``build/``."""

    def __init__(self, root: Path):
        self.root = root
        self.config_path = root / CONFIG

    def render(self, document: object) -> None:
        """Write the build output's config, as a build would."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(document), encoding="utf-8")


@pytest.fixture
def deployment(tmp_path, monkeypatch) -> _Deployment:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    (root / "var" / "agent_data").mkdir(parents=True)
    monkeypatch.setenv("OSPREY_CONFIG", str(root / CONFIG))
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.chdir(root)
    reset_config_cache()
    yield _Deployment(root)
    reset_config_cache()


@pytest.fixture
def no_deployment(tmp_path, monkeypatch) -> Path:
    """A project directory without ``profile.yml``: a render, as in a container."""
    root = (tmp_path / "render").resolve()
    root.mkdir()
    monkeypatch.setenv("OSPREY_CONFIG", str(root / "config.yml"))
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.chdir(root)
    reset_config_cache()
    yield root
    reset_config_cache()


@pytest.fixture
def alerts(monkeypatch) -> AsyncMock:
    """The gate chain's operator alert, captured."""
    mock = AsyncMock()
    monkeypatch.setattr(gates, "notify_agent_activity_async", mock)
    return mock


@pytest.fixture
def launch(monkeypatch) -> AsyncMock:
    """The executor's execute call, replaced; a refusal must leave it unawaited."""
    result = ExecutionResult(
        success=True, stdout="", stderr="", figures=[], execution_time_seconds=0.1
    )
    mock = AsyncMock(return_value=result)
    monkeypatch.setattr(executor, "execute_code", mock)
    return mock


@pytest.fixture
def no_spawn(monkeypatch):
    """Any child process the executor tries to start fails the test."""

    async def _spawn(*_args, **_kwargs):
        raise AssertionError("a child process was started")

    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", _spawn)


#: Each state of a deployment's build output that must refuse.
REFUSING_STATES = {
    "key_absent": lambda d: d.render({"project_name": "scratch"}),
    "key_null": lambda d: d.render({"protected_paths": None}),
    "key_a_string": lambda d: d.render({"protected_paths": "plans"}),
    "key_holds_a_non_string": lambda d: d.render({"protected_paths": ["plans", 3]}),
    "unparseable": lambda d: (
        d.config_path.parent.mkdir(parents=True, exist_ok=True),
        d.config_path.write_text("protected_paths: [unclosed\n", encoding="utf-8"),
    ),
    "unreadable": lambda d: (
        d.config_path.parent.mkdir(parents=True, exist_ok=True),
        d.config_path.write_bytes(b"\xff\xfe\x00not utf-8"),
    ),
    "no_build_output": lambda d: None,
}

_STATES = pytest.mark.parametrize("state", sorted(REFUSING_STATES))

#: No test here may start a child process, and none alerts a live Web Terminal.
pytestmark = pytest.mark.usefixtures("no_spawn", "alerts")


# -- the function -------------------------------------------------------------


def test_a_deployment_with_the_list_returns_its_entries(deployment):
    deployment.render({"protected_paths": ["plans", "~/notes.md"]})

    assert read_run_protected_entries() == ("plans", "~/notes.md")


def test_an_empty_list_is_a_list(deployment):
    deployment.render({"protected_paths": []})

    assert read_run_protected_entries() == ()


@_STATES
def test_every_deployment_state_without_the_list_refuses_naming_the_build(deployment, state):
    REFUSING_STATES[state](deployment)

    with pytest.raises(ProtectedPathsRefusal) as exc_info:
        read_run_protected_entries()

    message = str(exc_info.value)
    assert "`osprey build`" in message
    assert str(deployment.config_path) in message


def test_outside_a_deployment_the_loaded_entries_are_used_and_nothing_refuses(no_deployment):
    (no_deployment / "config.yml").write_text(
        yaml.safe_dump({"protected_paths": ["plans"]}), encoding="utf-8"
    )

    assert read_run_protected_entries() == ("plans",)


@pytest.mark.parametrize(
    "document", [None, {"project_name": "x"}, {"protected_paths": "plans"}], ids=str
)
def test_outside_a_deployment_a_config_without_the_list_gives_none(no_deployment, document):
    if document is not None:
        (no_deployment / "config.yml").write_text(yaml.safe_dump(document), encoding="utf-8")

    assert read_run_protected_entries() == ()


def test_a_rebuild_is_seen_without_a_restart(deployment):
    """A warm configuration cache from a build without the list does not decide."""
    deployment.render({"project_name": "scratch"})
    assert "protected_paths" not in load_osprey_config()  # the cache is now warm
    with pytest.raises(ProtectedPathsRefusal):
        read_run_protected_entries()

    deployment.render({"project_name": "scratch", "protected_paths": ["plans"]})

    assert "protected_paths" not in load_osprey_config()  # still the stale copy
    assert read_run_protected_entries() == ("plans",)


# -- the code tool and the file tool ------------------------------------------


async def _run_tool(tool_name: str, root: Path):
    if tool_name == "execute":
        from osprey.mcp_server.python_executor.tools.python_execute import execute

        return await get_tool_fn(execute)(
            code="print(1)\n",
            description="refusal probe",
            execution_mode="readonly",
            save_output=False,
        )

    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    script = root / "probe.py"
    script.write_text("print(1)\n", encoding="utf-8")
    return await get_tool_fn(execute_file)(
        file_path=str(script),
        description="refusal probe",
        execution_mode="readonly",
        save_output=False,
    )


def _envelope(exc_info) -> dict:
    return json.loads(str(exc_info.value))


def _executor_records(recorded_entries) -> list[dict]:
    from osprey.audit.envelope import SURFACE_EXECUTOR

    entries = sorted(recorded_entries("audit"), key=lambda e: e.ts)
    return [dict(e.body) for e in entries if e.body["surface"] == SURFACE_EXECUTOR]


_TOOLS = pytest.mark.parametrize("tool_name", ["execute", "execute_file"])


@_TOOLS
@_STATES
async def test_the_tools_refuse_a_deployment_without_the_list(deployment, tool_name, state, launch):
    REFUSING_STATES[state](deployment)

    with pytest.raises(ToolError) as exc_info:
        await _run_tool(tool_name, deployment.root)

    envelope = _envelope(exc_info)
    assert envelope["error_type"] == gates.NO_PROTECTED_PATHS_ERROR_TYPE
    assert "`osprey build`" in envelope["error_message"]
    launch.assert_not_awaited()


@_TOOLS
@pytest.mark.usefixtures("launch")
async def test_the_refusal_is_recorded_under_its_own_reason(
    deployment, tool_name, recorded_entries
):
    from osprey.audit.envelope import DECISION_REFUSED

    deployment.render({"project_name": "scratch"})

    with pytest.raises(ToolError):
        await _run_tool(tool_name, deployment.root)

    records = _executor_records(recorded_entries)
    assert [(r["decision"], r["reason"]) for r in records] == [
        (DECISION_REFUSED, gates.REASON_NO_PROTECTED_PATHS)
    ]
    assert "osprey build" in records[0]["detail"]


@pytest.mark.usefixtures("launch")
async def test_the_alert_does_not_speak_of_a_write(deployment, alerts):
    deployment.render({"project_name": "scratch"})

    with pytest.raises(ToolError):
        await _run_tool("execute", deployment.root)

    alerts.assert_awaited_once()
    detail = alerts.await_args.kwargs["detail"]
    assert detail == gates.NO_PROTECTED_PATHS_ALERT
    assert "write" not in detail.lower()
    assert "blocked" not in detail.lower()
    assert "osprey build" in detail


@_TOOLS
async def test_the_entries_reach_the_launch(deployment, tool_name, launch, alerts):
    deployment.render({"protected_paths": ["plans"]})

    await _run_tool(tool_name, deployment.root)

    launch.assert_awaited_once()
    assert launch.await_args.kwargs["protected_entries"] == ("plans",)
    alerts.assert_not_awaited()


@_TOOLS
async def test_outside_a_deployment_the_tools_run(no_deployment, tool_name, launch):
    await _run_tool(tool_name, no_deployment)

    launch.assert_awaited_once()
    assert launch.await_args.kwargs["protected_entries"] == ()


async def test_the_gated_path_reads_the_config_file_once(deployment, monkeypatch):
    """One read decides the refusal, the pre-run check and the guard."""
    deployment.render({"protected_paths": ["plans"]})
    via_local = AsyncMock(
        return_value=ExecutionResult(
            success=True, stdout="", stderr="", figures=[], execution_time_seconds=0.1
        )
    )
    monkeypatch.setattr(executor, "_execute_via_local", via_local)
    monkeypatch.setattr(executor, "_create_execution_folder", lambda: deployment.root / "run")

    load_osprey_config()  # the process-lifetime cache, warm as in a running server
    reads: list[Path] = []
    real_read_text = Path.read_text

    def _counting_read_text(self, *args, **kwargs):
        # The protected-path reads only: the code review reads its own
        # settings from the same file, which is not the protected-path set.
        caller = sys._getframe(1).f_code
        if self == deployment.config_path and (
            Path(caller.co_filename).is_relative_to(EXECUTOR_PACKAGE)
            or caller is load_protected_paths.__code__
        ):
            reads.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counting_read_text)

    await _run_tool("execute", deployment.root)

    assert len(reads) == 1
    assert via_local.await_args.kwargs["protected_entries"] == ("plans",)


async def test_a_path_the_entries_name_is_refused_by_the_pre_run_check(deployment, launch):
    """The handed entries, not a cached config, are what the pre-run check guards."""
    deployment.render({"protected_paths": ["plans"]})
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    with pytest.raises(ToolError) as exc_info:
        await get_tool_fn(execute)(
            code="open('plans/scan.py', 'w')\n",
            description="refusal probe",
            execution_mode="readonly",
            save_output=False,
        )

    assert "location the deployment protects" in _envelope(exc_info)["error_message"]
    launch.assert_not_awaited()


# -- the executor's own backstop ----------------------------------------------


@_STATES
async def test_execute_code_unhanded_refuses_with_its_own_kind(deployment, state, monkeypatch):
    REFUSING_STATES[state](deployment)

    def _no_folder():
        raise AssertionError("an execution folder was created")

    monkeypatch.setattr(executor, "_create_execution_folder", _no_folder)

    result = await executor.execute_code("print(1)", "readonly", "backstop probe")

    assert result.success is False
    assert result.failure_kind == FAILURE_KIND_NO_PROTECTED_PATHS
    assert result.failure_kind in NOT_STARTED_FAILURE_KINDS
    assert result.execution_time_seconds is None
    assert "`osprey build`" in result.error_message


async def test_execute_code_handed_entries_does_not_read(deployment, monkeypatch):
    """Handed entries — an empty set included — are not second-guessed."""
    via_local = AsyncMock(
        return_value=ExecutionResult(success=True, stdout="", stderr="", figures=[])
    )
    monkeypatch.setattr(executor, "_execute_via_local", via_local)
    monkeypatch.setattr(executor, "_create_execution_folder", lambda: deployment.root / "run")

    def _no_read(*_args, **_kwargs):
        raise AssertionError("the entries were read again")

    monkeypatch.setattr(executor, "read_run_protected_entries", _no_read)

    result = await executor.execute_code(
        "print(1)", "readonly", "handed probe", protected_entries=()
    )

    assert result.success is True
    assert via_local.await_args.kwargs["protected_entries"] == ()


# -- the measurement tools ----------------------------------------------------


@pytest.fixture
def matrix_tool(monkeypatch):
    """``model_response_matrix`` with its model request stubbed out."""
    from osprey.mcp_server.python_executor.tools import model_response_matrix as tool

    class _Request:
        model = "line"

    monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: Path.cwd())
    monkeypatch.setattr(tool, "resolve_model_request", lambda *a, **k: _Request())
    monkeypatch.setattr(tool, "_child_script", lambda request: "print(1)\n")
    return tool


class _Stop(Exception):
    """Raised by a stubbed launch once its arguments are captured."""


async def _matrix(tool):
    fn = tool.model_response_matrix
    return await getattr(fn, "fn", fn)(model="line")


@_STATES
async def test_model_response_matrix_refuses_first(deployment, state, matrix_tool, launch):
    REFUSING_STATES[state](deployment)

    with pytest.raises(ToolError) as exc_info:
        await _matrix(matrix_tool)

    assert _envelope(exc_info)["error_type"] == gates.NO_PROTECTED_PATHS_ERROR_TYPE
    launch.assert_not_awaited()


@pytest.mark.parametrize("where", ["deployment", "no_deployment"])
async def test_model_response_matrix_hands_its_entries_on(request, where, matrix_tool, launch):
    fixture = request.getfixturevalue(where)
    expected: tuple[str, ...] = ()
    if where == "deployment":
        fixture.render({"protected_paths": ["plans"]})
        expected = ("plans",)
    launch.side_effect = _Stop

    with pytest.raises(_Stop):
        await _matrix(matrix_tool)

    assert launch.await_args.kwargs["protected_entries"] == expected


@pytest.fixture
def measure_tool(monkeypatch):
    """``pyaml_measure`` with its arguments and script stubbed out."""
    from osprey.mcp_server.python_executor.tools import pyaml_measure as tool

    class _Args:
        kind = "orm"

    gate = AsyncMock(side_effect=_Stop)
    monkeypatch.setattr(tool, "run_gated_execution", gate)
    monkeypatch.setattr(tool, "enforce_posture_clamp", lambda mode, *, tool: None)
    monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: Path.cwd())
    monkeypatch.setattr(tool, "resolve_measure_args", lambda *a, **k: _Args())
    monkeypatch.setattr(tool, "build_measure_script", lambda script: "print(1)\n")
    monkeypatch.setattr(tool, "_script", lambda *a, **k: "print(1)\n")
    monkeypatch.setattr(tool, "_description", lambda args: "measure probe")
    monkeypatch.setattr(tool, "gate", gate, raising=False)
    return tool


async def _measure(tool):
    fn = tool.pyaml_measure
    return await getattr(fn, "fn", fn)(kind="orm")


@_STATES
async def test_pyaml_measure_refuses_first(deployment, state, measure_tool):
    REFUSING_STATES[state](deployment)

    with pytest.raises(ToolError) as exc_info:
        await _measure(measure_tool)

    assert _envelope(exc_info)["error_type"] == gates.NO_PROTECTED_PATHS_ERROR_TYPE
    measure_tool.gate.assert_not_awaited()


@pytest.mark.parametrize("where", ["deployment", "no_deployment"])
async def test_pyaml_measure_hands_its_entries_to_the_chain(request, where, measure_tool):
    fixture = request.getfixturevalue(where)
    expected: tuple[str, ...] = ()
    if where == "deployment":
        fixture.render({"protected_paths": ["plans"]})
        expected = ("plans",)

    with pytest.raises(_Stop):
        await _measure(measure_tool)

    assert measure_tool.gate.await_args.kwargs["protected_entries"] == expected
