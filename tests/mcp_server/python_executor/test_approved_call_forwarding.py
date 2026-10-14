"""The approved target travels from the tool call to the executor's check before spawning.

The approval hook writes ``approved_target`` into an ``execute`` or
``execute_file`` call, and ``approved_view_sha256`` too into a
``pyaml_measure`` call. The tool hands them to the shared gate sequence, and
the executor, before it spawns anything, refuses an asking call whose fields
differ from what the approval prompt recorded, or that was approved for another
target than the run's. Nothing of the record reaches the sandbox. The sandbox
cases run a real wrapped script whose ``osprey.runtime`` reads and writes a dict
machine kept in a JSON file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import osprey.runtime.guarded_run as guarded_run
from osprey.mcp_server.python_executor import executor
from osprey.mcp_server.python_executor.executor import ExecutionResult
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from tests._guarded_run import held_lock, lock_path
from tests.mcp_server.conftest import get_tool_fn

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"

#: How long a wrapped child gets to finish.
CHILD_TIMEOUT_S = 120.0

#: Installed as ``sitecustomize`` in the child: ``osprey.runtime`` reads and
#: writes the dict machine in ``$APPROVED_CALL_VALUES`` before any wrapper code.
_SITECUSTOMIZE = textwrap.dedent(
    """
    import json, os

    import osprey.runtime

    _path = os.environ["APPROVED_CALL_VALUES"]

    def _load():
        with open(_path) as handle:
            return json.load(handle)

    def _read_channels(addresses, **_kwargs):
        values = _load()
        return [values[a] for a in addresses]

    def _write_channel(address, value, **_kwargs):
        values = _load()
        values[address] = value
        with open(_path, "w") as handle:
            json.dump(values, handle)

    osprey.runtime.read_channels = _read_channels
    osprey.runtime.write_channel = _write_channel
    osprey.runtime.channel_limits = lambda address: None
    """
)

#: User code that records what it saw when it started.
_PROBE = textwrap.dedent(
    """
    import json, os
    with open(os.environ["APPROVED_CALL_VALUES"]) as handle:
        seen = json.load(handle)
    results = {"seen": seen}
    """
)


def _result() -> ExecutionResult:
    return ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        figures=[],
        execution_method_used="subprocess",
        execution_time_seconds=0.1,
    )


@pytest.mark.parametrize("tool_name", ["execute", "execute_file"])
async def test_each_tool_hands_the_approved_fields_to_the_gate_sequence(
    tool_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gated = AsyncMock(return_value=(_result(), {"has_writes": False}))
    if tool_name == "execute":
        from osprey.mcp_server.python_executor.tools import python_execute as module

        kwargs: dict[str, Any] = {"code": "x = 1\n"}
    else:
        from osprey.mcp_server.python_executor.tools import python_execute_file as module

        monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
        script = tmp_path / "probe.py"
        script.write_text("x = 1\n", encoding="utf-8")
        kwargs = {"file_path": str(script)}

    with (
        patch.object(module, "run_gated_execution", gated),
        patch(
            "osprey.mcp_server.python_executor.tools._response_builder.build_execution_response",
            AsyncMock(return_value="{}"),
        ),
    ):
        await get_tool_fn(getattr(module, tool_name))(
            **kwargs,
            description="probe",
            execution_mode="readwrite",
            save_output=False,
            approved_target="live",
        )

    assert gated.await_args.kwargs["tool"] == tool_name
    assert gated.await_args.kwargs["approved_target"] == "live"


async def test_the_gate_sequence_hands_the_approved_fields_to_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.mcp_server.python_executor.tools._execution_gates import run_gated_execution

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    launch = AsyncMock(return_value=_result())
    with patch("osprey.mcp_server.python_executor.executor.execute_code", launch):
        await run_gated_execution(
            tool="execute_file",
            code="x = 1\n",
            description="probe",
            execution_mode="readonly",
            project_root=tmp_path,
            approved_target="va",
        )

    assert launch.await_args.kwargs["tool"] == "execute_file"
    assert launch.await_args.kwargs["approved_target"] == "va"


#: A rendered config whose approval policy asks about every tool.
_ASKING_CONFIG: dict[str, Any] = {
    "control_system": {"type": "epics", "connector": {"epics": {}}},
    "approval": {"enabled": True, "default_policy": "always"},
}


def _approval(**changes: Any) -> dict[str, Any]:
    return {**_ASKING_CONFIG, "approval": {**_ASKING_CONFIG["approval"], **changes}}


#: The tool-use id every gate case runs its call under.
_CALL_ID = "toolu_gate1"

#: The approved fields the approval prompt bound the call to.
_BOUND = {"approved_target": "live"}


@pytest.fixture
def stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The identity state directory the approval hook files its ask stamps in."""
    from osprey.mcp_server.control_system import target_state

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(target_state, "state_dir", lambda: state)
    return state


def _stamp(state: Path, tool: str = "mcp__python__execute", **fields: str) -> None:
    payload = {"tool_use_id": _CALL_ID, "tool": tool, "approver": "human", **fields}
    (state / f"approval_ask_{_CALL_ID}.json").write_text(json.dumps(payload), encoding="utf-8")


def _gate(
    config: dict[str, Any],
    approved: dict[str, str | None],
    *,
    mode: str = "readwrite",
    tool: str = "execute",
    call_id: str | None = _CALL_ID,
    control_target: str = "live",
) -> None:
    from osprey.audit.call import call_scope

    with call_scope(call_id, None):
        executor._check_approved_call(mode, config, tool, approved, control_target=control_target)


def test_the_executor_spells_the_ask_stamp_like_the_hook() -> None:
    from tests.hooks.conftest import import_hook

    approval = import_hook("osprey_approval")
    assert executor.APPROVAL_ASK_PREFIX == approval.APPROVAL_ASK_PREFIX
    assert executor.APPROVAL_ASK_SUFFIX == approval.APPROVAL_ASK_SUFFIX
    assert all(name.startswith(approval.APPROVED_FIELD_PREFIX) for name in executor.APPROVED_FIELDS)


def test_a_stamp_equal_to_the_call_is_admitted(stamps: Path) -> None:
    _stamp(stamps, **_BOUND)

    _gate(_ASKING_CONFIG, _BOUND)


def test_execute_file_is_matched_by_its_own_name(stamps: Path) -> None:
    _stamp(stamps, tool="mcp__python__execute_file", **_BOUND)

    _gate(_ASKING_CONFIG, _BOUND, tool="execute_file")
    with pytest.raises(executor._ApprovalMissing, match="left no record"):
        _gate(_ASKING_CONFIG, _BOUND, tool="execute")


@pytest.mark.parametrize(
    "call_id",
    [pytest.param(_CALL_ID, id="no-stamp"), pytest.param(None, id="no-call-in-scope")],
)
@pytest.mark.usefixtures("stamps")
def test_an_asking_call_without_a_stamp_is_refused(call_id: str | None) -> None:
    with pytest.raises(executor._ApprovalMissing) as refusal:
        _gate(_ASKING_CONFIG, _BOUND, call_id=call_id)
    assert str(refusal.value) == (
        "not run: this call's approval prompt left no record (approval_missing). "
        "Call the tool again so the approval prompt can ask."
    )


def test_an_unresolvable_state_directory_is_no_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.mcp_server.control_system import target_state

    def unresolvable() -> Path:
        raise RuntimeError("no state directory")

    monkeypatch.setattr(target_state, "state_dir", unresolvable)
    with pytest.raises(executor._ApprovalMissing, match="left no record"):
        _gate(_ASKING_CONFIG, _BOUND)


@pytest.mark.parametrize("field", ["approved_target"])
def test_a_stamp_that_differs_from_the_call_is_refused(stamps: Path, field: str) -> None:
    _stamp(stamps, **{**_BOUND, field: "something-else"})

    with pytest.raises(executor._ApprovalMissing) as refusal:
        _gate(_ASKING_CONFIG, _BOUND)
    assert str(refusal.value) == (
        f"not run: `{field}` differs from what the approval prompt recorded "
        "(approval_missing). Call the tool again."
    )


@pytest.mark.parametrize("field", ["approved_target"])
def test_an_asking_call_missing_a_field_is_refused(stamps: Path, field: str) -> None:
    _stamp(stamps, **_BOUND)

    with pytest.raises(executor._ApprovalMissing, match=f"`{field}` differs"):
        _gate(_ASKING_CONFIG, {**_BOUND, field: None})


def test_a_field_the_stamp_records_and_the_call_omits_is_not_compared(stamps: Path) -> None:
    _stamp(stamps, **_BOUND, approved_view_sha256="cd" * 32)

    _gate(_ASKING_CONFIG, _BOUND)


def test_a_field_the_call_carries_and_the_stamp_lacks_is_refused(stamps: Path) -> None:
    _stamp(stamps, **_BOUND)

    with pytest.raises(executor._ApprovalMissing, match="`approved_view_sha256` differs"):
        _gate(_ASKING_CONFIG, {**_BOUND, "approved_view_sha256": "cd" * 32})


def test_a_call_approved_for_another_target_is_refused_before_spawn(stamps: Path) -> None:
    _stamp(stamps, **_BOUND)

    with pytest.raises(executor._ApprovalMissing) as refusal:
        _gate(_ASKING_CONFIG, _BOUND, control_target="va")
    assert str(refusal.value) == (
        "not run: this call was approved for target `live`; the run is on `va` "
        "(approval_missing). Call the tool again."
    )


def _launched_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    mode: str = "readwrite",
    tool: str = "execute",
    approved: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment a launch of one call hands its sandbox, from a fake spawn."""
    from osprey.audit.call import call_scope

    captured: dict[str, dict[str, str]] = {}

    class _Proc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def spawn(*_args: Any, **kwargs: Any) -> _Proc:
        captured["env"] = kwargs["env"]
        return _Proc()

    def stamp_target(sandbox_env: dict[str, str]) -> str:
        sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
        return "live"

    folder = tmp_path / "exec"
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: config)
    monkeypatch.setattr(executor, "_apply_target_stamp", stamp_target)
    monkeypatch.setattr(executor, "_load_limits_validator", lambda target: None)
    monkeypatch.setattr(executor, "resolve_agent_interpreter", lambda root=None: "/bin/true")
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setenv("OSPREY_APPROVED_TARGET", "inherited")

    import asyncio

    async def launch() -> ExecutionResult:
        with call_scope(_CALL_ID, None):
            return await executor._execute_via_local(
                "x = 1\n", mode, {"timeout": 30}, folder, tool=tool, **(approved or _BOUND)
            )

    asyncio.run(launch())
    return captured["env"]


@pytest.mark.parametrize(
    ("config", "mode", "stamped"),
    [
        pytest.param(_approval(tools={"execute": "skip"}), "readwrite", False, id="skip"),
        pytest.param(_approval(enabled=False), "readwrite", False, id="approval-disabled"),
        pytest.param(_ASKING_CONFIG, "readwrite", True, id="asked-and-stamped"),
        pytest.param(_ASKING_CONFIG, "readonly", False, id="readonly"),
    ],
)
def test_no_approved_field_reaches_any_sandbox(
    stamps: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    mode: str,
    stamped: bool,
) -> None:
    if stamped:
        _stamp(stamps, **_BOUND)

    env = _launched_env(tmp_path, monkeypatch, config, mode=mode)

    assert [name for name in env if name.startswith("OSPREY_APPROVED_")] == []


def test_a_readonly_call_is_not_checked(stamps: Path) -> None:
    _stamp(stamps, approved_target="va")

    _gate(_ASKING_CONFIG, _BOUND, mode="readonly")


#: Approval sections no reader can trust, each with the sentence that refuses it.
_UNTRUSTED_APPROVAL = [
    pytest.param(
        _approval(enabled=value),
        f"approval.enabled is {value!r}, not true or false; "
        "set approval.enabled to true or false in profile.yml and rebuild",
        id=f"enabled={value!r}",
    )
    for value in (None, "false", 0)
] + [
    pytest.param(
        {**_ASKING_CONFIG, "approval": section},
        f"approval is {section!r}, not a mapping; "
        "set approval.enabled to true or false in profile.yml and rebuild",
        id=f"section={section!r}",
    )
    for section in (None, "disabled")
]


@pytest.mark.parametrize("stamped", [False, True], ids=["no-stamp", "stamp"])
@pytest.mark.parametrize(("config", "sentence"), _UNTRUSTED_APPROVAL)
def test_a_non_bool_switch_runs_nothing(
    stamps: Path, config: dict[str, Any], sentence: str, stamped: bool
) -> None:
    """A switch that is not true or false refuses the run, even with a matching stamp."""
    if stamped:
        _stamp(stamps, **_BOUND)
    with pytest.raises(executor._ApprovalConfigInvalid) as refusal:
        _gate(config, _BOUND)
    assert str(refusal.value) == f"not run: {sentence} (approval_config_invalid)."


@pytest.mark.parametrize(
    "config", [pytest.param(case.values[0], id=case.id) for case in _UNTRUSTED_APPROVAL]
)
def test_a_non_bool_switch_leaves_a_readonly_call_alone(
    stamps: Path, config: dict[str, Any]
) -> None:
    _stamp(stamps, **_BOUND)
    _gate(config, _BOUND, mode="readonly")


@pytest.mark.usefixtures("stamps")
async def test_a_refused_call_spawns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.audit.call import call_scope

    folder = tmp_path / "exec"
    folder.mkdir()

    def stamp_target(sandbox_env: dict[str, str]) -> str:
        sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
        return "live"

    async def spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a refused call spawns nothing")

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: _ASKING_CONFIG)
    monkeypatch.setattr(executor, "_apply_target_stamp", stamp_target)
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spawn)

    with call_scope(_CALL_ID, None):
        result = await executor._execute_via_local(
            "x = 1\n", "readwrite", {"timeout": 30}, folder, tool="execute", **_BOUND
        )

    assert result.success is False
    assert result.failure_kind == executor.FAILURE_KIND_APPROVAL_MISSING == "approval_missing"
    assert "left no record" in (result.error_message or "")
    assert list(folder.iterdir()) == [], "the execution folder is never used"
    assert not (tmp_path / "var").exists(), "no lock was taken"


@pytest.mark.usefixtures("stamps")
async def test_a_non_bool_switch_spawns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.audit.call import call_scope

    folder = tmp_path / "exec"
    folder.mkdir()

    def stamp_target(sandbox_env: dict[str, str]) -> str:
        sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
        return "live"

    async def spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a refused call spawns nothing")

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config", lambda: _approval(enabled="false")
    )
    monkeypatch.setattr(executor, "_apply_target_stamp", stamp_target)
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spawn)

    with call_scope(_CALL_ID, None):
        result = await executor._execute_via_local(
            "x = 1\n", "readwrite", {"timeout": 30}, folder, tool="execute", **_BOUND
        )

    assert result.success is False
    assert result.failure_kind == executor.FAILURE_KIND_APPROVAL_CONFIG_INVALID
    assert result.failure_kind == "approval_config_invalid"
    assert "approval.enabled is 'false'" in (result.error_message or "")
    assert list(folder.iterdir()) == [], "the execution folder is never used"


#: The approved fields a measurement's prompt binds its call to.
_MEASURE_BOUND = {**_BOUND, "approved_view_sha256": "cd" * 32}


def test_a_measurement_whose_stamp_agrees_on_both_fields_is_admitted(stamps: Path) -> None:
    _stamp(stamps, tool="mcp__python__pyaml_measure", **_MEASURE_BOUND)

    _gate(_ASKING_CONFIG, _MEASURE_BOUND, tool="pyaml_measure")


def test_an_asking_measurement_without_a_view_digest_is_refused(stamps: Path) -> None:
    _stamp(stamps, tool="mcp__python__pyaml_measure", **_MEASURE_BOUND)

    with pytest.raises(executor._ApprovalMissing, match="`approved_view_sha256` differs"):
        _gate(_ASKING_CONFIG, _BOUND, tool="pyaml_measure")


@pytest.mark.parametrize(
    "stamped",
    [
        pytest.param(_BOUND, id="stamp-lacks-the-view"),
        pytest.param({**_BOUND, "approved_view_sha256": "ef" * 32}, id="stamp-differs"),
    ],
)
async def test_a_measurement_whose_view_digest_disagrees_spawns_nothing(
    stamps: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamped: dict[str, str]
) -> None:
    from osprey.audit.call import call_scope

    _stamp(stamps, tool="mcp__python__pyaml_measure", **stamped)
    folder = tmp_path / "exec"
    folder.mkdir()

    def stamp_target(sandbox_env: dict[str, str]) -> str:
        sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
        return "live"

    async def spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a refused call spawns nothing")

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: _ASKING_CONFIG)
    monkeypatch.setattr(executor, "_apply_target_stamp", stamp_target)
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spawn)

    with call_scope(_CALL_ID, None):
        result = await executor._execute_via_local(
            "x = 1\n",
            "readwrite",
            {"timeout": 30},
            folder,
            tool="pyaml_measure",
            **_MEASURE_BOUND,
        )

    assert result.failure_kind == executor.FAILURE_KIND_APPROVAL_MISSING
    assert "`approved_view_sha256` differs" in (result.error_message or "")
    assert list(folder.iterdir()) == []


async def test_the_gate_sequence_hands_the_view_digest_to_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.mcp_server.python_executor.tools._execution_gates import run_gated_execution

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    launch = AsyncMock(return_value=_result())
    with patch("osprey.mcp_server.python_executor.executor.execute_code", launch):
        await run_gated_execution(
            tool="pyaml_measure",
            code="x = 1\n",
            description="probe",
            execution_mode="readonly",
            project_root=tmp_path,
            approved_view_sha256="cd" * 32,
        )

    assert launch.await_args.kwargs["approved_view_sha256"] == "cd" * 32


def test_a_readonly_wrapper_takes_no_lock() -> None:
    script = ExecutionWrapper(execution_mode="readonly").create_wrapper("x = 1\n")
    assert "_osprey_lock" not in script
    assert "_osprey_lock" in ExecutionWrapper(execution_mode="readwrite").create_wrapper("x = 1\n")


class _Sandbox:
    """A deployment repo and one wrapped readwrite run in it."""

    def __init__(self, root: Path, values: dict[str, float]) -> None:
        self.root = root
        (root / "build").mkdir(parents=True)
        (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
        self.config_path = root / "build" / "config.yml"
        self.write_config(_ASKING_CONFIG)
        self.values_path = root / "values.json"
        self.values_path.write_text(json.dumps(values), encoding="utf-8")
        site = root / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
        self.site = site
        self.folder = root / "exec"
        self.folder.mkdir()

    def write_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def run(self, env_extra: dict[str, str], code: str = _PROBE) -> dict[str, Any]:
        script = self.folder / "wrapped_script.py"
        script.write_text(
            ExecutionWrapper(execution_mode="readwrite").create_wrapper(code, self.folder),
            encoding="utf-8",
        )
        env = os.environ.copy()
        for name in ("OSPREY_CONFIG", "CONFIG_FILE"):
            env.pop(name, None)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(self.site), str(_SRC_ROOT), env.get("PYTHONPATH")])
        )
        env["APPROVED_CALL_VALUES"] = str(self.values_path)
        env["OSPREY_EXECUTION_MODE"] = "readwrite"
        env["OSPREY_CONTROL_TARGET"] = "live"
        env["OSPREY_CONTROL_TARGET_GENERATION"] = "4"
        env["OSPREY_AGENT_DATA_ROOT"] = str(self.root / "agent_data")
        env.update(env_extra)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(self.root),
            env=env,
            capture_output=True,
            timeout=CHILD_TIMEOUT_S,
            check=False,
        )
        metadata = json.loads((self.folder / "execution_metadata.json").read_text())
        results_path = self.folder / "results.json"
        if results_path.exists():
            metadata["_results"] = json.loads(results_path.read_text())
        return metadata


def test_the_executor_spells_the_busy_failure_like_the_hooks_audit_reason() -> None:
    from osprey.services.python_executor.execution import wrapper
    from tests.hooks.conftest import import_hook

    approval = import_hook("osprey_approval")
    assert executor.FAILURE_KIND_GUARDED_RUN_BUSY == approval._GUARDED_RUN_BUSY_AUDIT_REASON
    assert wrapper.FAILURE_KIND_GUARDED_RUN_BUSY == executor.FAILURE_KIND_GUARDED_RUN_BUSY


def _filed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: _Sandbox, metadata: dict[str, Any]
) -> ExecutionResult:
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: sandbox.root)
    folder = tmp_path / "filed"
    folder.mkdir()
    return executor._result_from_run(
        folder,
        metadata,
        stdout_text="",
        stderr_text=metadata.get("stderr", ""),
        returncode=0,
        elapsed=0.1,
        control_target="live",
    )


def _envelope(result: ExecutionResult) -> dict[str, Any]:
    from osprey.mcp_server.python_executor.tools._response_builder import _raise_failure
    from tests.mcp_server.conftest import assert_error

    with pytest.raises(Exception) as caught:
        _raise_failure(result, result.stderr, {})
    return assert_error(str(caught.value), error_type="execution_error")


def test_a_run_on_a_held_target_is_reported_as_busy_with_its_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox refuses before user code; the result names the holder and its own kind."""
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 9.0})
    with held_lock("live", root=sandbox.root) as holder_process:
        holder = json.loads(lock_path("live", sandbox.root).read_text(encoding="utf-8"))
        metadata = sandbox.run({})

    result = _filed(tmp_path, monkeypatch, sandbox, metadata)

    assert "_results" not in metadata, "the user code never started"
    assert metadata["failure_kind"] == executor.FAILURE_KIND_GUARDED_RUN_BUSY
    assert result.success is False
    assert result.failure_kind == executor.FAILURE_KIND_GUARDED_RUN_BUSY
    assert f"pid {holder_process.pid}" in result.error_message
    assert holder["started"] in result.error_message
    envelope = _envelope(result)
    assert envelope["details"]["kind"] == executor.FAILURE_KIND_GUARDED_RUN_BUSY
    assert f"pid {holder_process.pid}" in envelope["error_message"]
    assert holder["started"] in envelope["error_message"]
    assert len(envelope["suggestions"]) == 1


def test_a_busy_raised_by_the_scripts_own_lock_is_the_scripts_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User code that asks for the lock its own run holds fails as its own code."""
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 9.0})
    code = "from osprey.runtime.guarded_run import lock\nwith lock(None):\n    pass\n"

    metadata = sandbox.run({}, code)
    result = _filed(tmp_path, monkeypatch, sandbox, metadata)

    assert metadata["error_type"] == guarded_run.OspreyRunBusy.__name__, metadata.get("traceback")
    assert "failure_kind" not in metadata
    assert result.failure_kind is None, "a script's own error stays the script's"
    assert _envelope(result)["details"]["kind"] == "script_error"
