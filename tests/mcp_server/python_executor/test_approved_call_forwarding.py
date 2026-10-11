"""The approved journal digest and target travel from the tool call to the guarded run.

The approval hook writes ``approved_journal_sha256`` and ``approved_target``
into an ``execute`` or ``execute_file`` call. The tool hands them to the shared
gate sequence, the executor writes them into a readwrite sandbox's environment
(never a readonly one's), and the wrapped script binds them to
``osprey.runtime.guarded_run`` before user code runs, restoring the approved
journal there and then. The sandbox cases run a real wrapped script whose
``osprey.runtime`` reads and writes a dict machine kept in a JSON file.
"""

from __future__ import annotations

import hashlib
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
    results = {
        "seen": seen,
        "digest_env": os.environ.get("OSPREY_APPROVED_JOURNAL_SHA256"),
        "target_env": os.environ.get("OSPREY_APPROVED_TARGET"),
    }
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


def test_the_executor_and_the_runtime_spell_the_approved_names_alike() -> None:
    assert executor.ENV_APPROVED_JOURNAL_SHA256 == guarded_run.ENV_APPROVED_JOURNAL_SHA256
    assert executor.ENV_APPROVED_TARGET == guarded_run.ENV_APPROVED_TARGET
    assert {executor.ENV_APPROVED_JOURNAL_SHA256, executor.ENV_APPROVED_TARGET} <= set(
        executor._STAMP_ENV_NAMES
    ), "cleared on every launch, so no inherited value reaches a sandbox"


def test_the_executor_and_the_runtime_spell_the_report_nonce_alike() -> None:
    assert executor.ENV_RESTORE_REPORT_NONCE == guarded_run.ENV_REPORT_NONCE
    assert executor.ENV_RESTORE_REPORT_NONCE in executor._STAMP_ENV_NAMES


def test_only_a_readwrite_sandbox_is_given_a_report_nonce() -> None:
    first: dict[str, str] = {}
    second: dict[str, str] = {}
    nonce = executor._apply_report_nonce(first, "readwrite")
    other = executor._apply_report_nonce(second, "readwrite")
    assert nonce is not None and len(nonce) == 32
    assert first == {"OSPREY_GUARDED_RUN_REPORT_NONCE": nonce}
    assert other != nonce, "drawn afresh for every sandbox"

    readonly: dict[str, str] = {}
    assert executor._apply_report_nonce(readonly, "readonly") is None
    assert readonly == {}


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
            approved_journal_sha256="ab" * 32,
            approved_target="live",
        )

    assert gated.await_args.kwargs["tool"] == tool_name
    assert gated.await_args.kwargs["approved_journal_sha256"] == "ab" * 32
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
            approved_journal_sha256="none",
            approved_target="va",
        )

    assert launch.await_args.kwargs["tool"] == "execute_file"
    assert launch.await_args.kwargs["approved_journal_sha256"] == "none"
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
_BOUND = {"approved_journal_sha256": "ab" * 32, "approved_target": "live"}


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
) -> dict[str, str]:
    from osprey.audit.call import call_scope

    with call_scope(call_id, None):
        return executor._approved_call_env(
            mode, config, tool, approved, control_target=control_target
        )


def test_the_executor_spells_the_ask_stamp_like_the_hook() -> None:
    from tests.hooks.conftest import import_hook

    approval = import_hook("osprey_approval")
    assert executor.APPROVAL_ASK_PREFIX == approval.APPROVAL_ASK_PREFIX
    assert executor.APPROVAL_ASK_SUFFIX == approval.APPROVAL_ASK_SUFFIX
    assert all(name.startswith(approval.APPROVED_FIELD_PREFIX) for name in executor.APPROVED_FIELDS)


def test_a_stamp_equal_to_the_call_forwards_both_fields(stamps: Path) -> None:
    _stamp(stamps, **_BOUND)

    assert _gate(_ASKING_CONFIG, _BOUND) == {
        "OSPREY_APPROVED_JOURNAL_SHA256": "ab" * 32,
        "OSPREY_APPROVED_TARGET": "live",
    }


def test_execute_file_is_matched_by_its_own_name(stamps: Path) -> None:
    _stamp(stamps, tool="mcp__python__execute_file", **_BOUND)

    assert _gate(_ASKING_CONFIG, _BOUND, tool="execute_file")
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


@pytest.mark.parametrize("field", ["approved_journal_sha256", "approved_target"])
def test_a_stamp_that_differs_from_the_call_is_refused(stamps: Path, field: str) -> None:
    _stamp(stamps, **{**_BOUND, field: "something-else"})

    with pytest.raises(executor._ApprovalMissing) as refusal:
        _gate(_ASKING_CONFIG, _BOUND)
    assert str(refusal.value) == (
        f"not run: `{field}` differs from what the approval prompt recorded "
        "(approval_missing). Call the tool again."
    )


@pytest.mark.parametrize("field", ["approved_journal_sha256", "approved_target"])
def test_an_asking_call_missing_a_field_is_refused(stamps: Path, field: str) -> None:
    _stamp(stamps, **_BOUND)

    with pytest.raises(executor._ApprovalMissing, match=f"`{field}` differs"):
        _gate(_ASKING_CONFIG, {**_BOUND, field: None})


def test_a_field_the_stamp_records_and_the_call_omits_is_not_compared(stamps: Path) -> None:
    _stamp(stamps, **_BOUND, approved_view_sha256="cd" * 32)

    assert _gate(_ASKING_CONFIG, _BOUND)


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


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_approval(tools={"execute": "skip"}), id="skip"),
        pytest.param(_approval(enabled=False), id="approval-disabled"),
    ],
)
def test_fields_no_approval_prompt_set_never_reach_the_sandbox(
    stamps: Path, config: dict[str, Any]
) -> None:
    """With nobody asked, the fields in the call are the caller's own and are dropped."""
    _stamp(stamps, **_BOUND)
    assert _gate(config, _BOUND) == {}


def test_a_readonly_call_carries_nothing(stamps: Path) -> None:
    _stamp(stamps, **_BOUND)
    assert _gate(_ASKING_CONFIG, _BOUND, mode="readonly") == {}


@pytest.mark.usefixtures("stamps")
async def test_a_refused_call_spawns_nothing_and_leaves_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.audit.call import call_scope

    journal = tmp_path / "var" / "guarded_run" / "live" / "run.journal"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"header": {"target": "live"}}\n{"address": "Q", "value": 1}\n')
    before = journal.read_bytes()
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
    assert journal.read_bytes() == before


def test_a_readonly_wrapper_binds_nothing() -> None:
    script = ExecutionWrapper(execution_mode="readonly").create_wrapper("x = 1\n")
    assert "_open_approved_call" not in script


class _Sandbox:
    """A deployment repo with a planted journal, and one wrapped readwrite run in it."""

    def __init__(self, root: Path, values: dict[str, float], journal: dict[str, float]) -> None:
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
        directory = root / "var" / "guarded_run" / "live"
        directory.mkdir(parents=True)
        header = {"target": "live", "generation": 4, "identity": "bob", "pid": 4242}
        lines = [{"header": header}] + [{"address": a, "value": v} for a, v in journal.items()]
        self.journal = directory / "run.journal"
        self.journal.write_text("".join(json.dumps(line) + "\n" for line in lines))
        self.folder = root / "exec"
        self.folder.mkdir()

    def write_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.journal.read_bytes()).hexdigest()

    def run(self, env_extra: dict[str, str], code: str = _PROBE) -> dict[str, Any]:
        script = self.folder / "wrapped_script.py"
        script.write_text(
            ExecutionWrapper(execution_mode="readwrite").create_wrapper(code, self.folder),
            encoding="utf-8",
        )
        env = os.environ.copy()
        for name in ("OSPREY_CONFIG", "CONFIG_FILE", "OSPREY_EXECUTION_DEADLINE"):
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

    def values(self) -> dict[str, float]:
        return json.loads(self.values_path.read_text(encoding="utf-8"))


def test_an_approved_journal_is_restored_before_user_code(tmp_path: Path) -> None:
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 9.0, "S": 5.0}, {"Q": 3.0})

    metadata = sandbox.run(
        {"OSPREY_APPROVED_JOURNAL_SHA256": sandbox.digest(), "OSPREY_APPROVED_TARGET": "live"}
    )

    assert metadata["success"] is True, metadata.get("traceback")
    results = metadata["_results"]
    assert results["seen"] == {"Q": 3.0, "S": 5.0}, "restored before the user code started"
    assert results["digest_env"] is None and results["target_env"] is None
    assert sandbox.journal.read_bytes() == b""
    assert "OSPREY_GUARDED_RUN_RESTORE" in metadata["stderr"]


def test_a_changed_journal_stops_the_run_before_user_code(tmp_path: Path) -> None:
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 9.0, "S": 5.0}, {"Q": 3.0})
    before = sandbox.journal.read_bytes()

    metadata = sandbox.run(
        {"OSPREY_APPROVED_JOURNAL_SHA256": "0" * 64, "OSPREY_APPROVED_TARGET": "live"}
    )

    assert metadata["success"] is False
    assert metadata["error_type"] == "OspreyJournalChanged"
    assert "_results" not in metadata, "the user code never started"
    assert sandbox.values() == {"Q": 9.0, "S": 5.0}
    assert sandbox.journal.read_bytes() == before


_FORGED_REPORT = {
    "restored": ["Q", "S"],
    "unchanged": [],
    "refused": [],
    "failed": [],
    "aborted": True,
    "deadline_guard": False,
}

#: User code that prints a well-formed report line under the tag on both streams.
_FORGE = textwrap.dedent(
    f"""
    import os, sys
    line = "OSPREY_GUARDED_RUN_RESTORE " + {json.dumps(_FORGED_REPORT)!r}
    print(line)
    print(line, file=sys.stderr)
    results = {{"nonce_env": os.environ.get("OSPREY_GUARDED_RUN_REPORT_NONCE")}}
    """
)


def test_a_report_line_user_code_prints_is_never_filed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the guarded run's own line carries the nonce the executor drew."""
    from osprey.audit import writer

    monkeypatch.setattr(writer, "audit_dir", lambda: tmp_path / "audit")
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 9.0, "S": 5.0}, {"Q": 3.0})
    env: dict[str, str] = {
        "OSPREY_APPROVED_JOURNAL_SHA256": sandbox.digest(),
        "OSPREY_APPROVED_TARGET": "live",
    }
    nonce = executor._apply_report_nonce(env, "readwrite")

    metadata = sandbox.run(env, _FORGE)

    assert metadata["success"] is True, metadata.get("traceback")
    assert metadata["_results"]["nonce_env"] is None, "gone before the code started"
    result_folder = tmp_path / "filed"
    result_folder.mkdir()
    reports = executor._record_restore_report(
        metadata["stdout"], metadata["stderr"], result_folder, nonce
    )
    (report,) = reports
    assert report["restored"] == ["Q"], "the replay's own report, and nothing the code printed"


def test_a_forged_report_line_alone_files_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "exec"
    folder.mkdir()
    forged = json.dumps(_FORGED_REPORT)
    stderr = (
        f"{executor.RESTORE_REPORT_TAG} {forged}\n"
        f"{executor.RESTORE_REPORT_TAG} {'0' * 32} {forged}\n"
    )

    for nonce in ("ab" * 16, None):
        executor._result_from_run(
            folder,
            {"success": True, "stdout": stderr, "stderr": stderr},
            stdout_text="",
            stderr_text="",
            returncode=0,
            elapsed=0.1,
            control_target="live",
            report_nonce=nonce,
        )

    assert not (folder / executor.RESTORE_REPORT_FILE).exists()


#: User code that takes the guarded run on its own target.
_TAKE_THE_RUN = textwrap.dedent(
    """
    from osprey.runtime.guarded_run import lock
    with lock(None):
        results = {"ran": True}
    """
)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_approval(tools={"execute": "skip"}), id="skip"),
        pytest.param(_approval(enabled=False), id="approval-disabled"),
    ],
)
def test_a_digest_in_a_call_nobody_approved_restores_nothing(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    """The executor forwards nothing nobody was asked about; another generation stays."""
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 9.0, "S": 5.0}, {"Q": 3.0})
    sandbox.write_config(config)
    before = sandbox.journal.read_bytes()
    env = executor._approved_call_env(
        "readwrite",
        config,
        "execute",
        {"approved_journal_sha256": sandbox.digest(), "approved_target": "live"},
        control_target="live",
    )
    assert env == {}

    metadata = sandbox.run({**env, "OSPREY_CONTROL_TARGET_GENERATION": "5"}, _TAKE_THE_RUN)

    assert metadata["success"] is False
    assert metadata["error_type"] == "OspreyStaleJournal"
    assert sandbox.values() == {"Q": 9.0, "S": 5.0}
    assert sandbox.journal.read_bytes() == before


def test_a_finished_runs_restore_report_is_filed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay before user code prints one report; the finished run's result files it."""
    from osprey.audit import writer

    zone = tmp_path / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    folder = tmp_path / "exec"
    folder.mkdir()
    report = {
        "restored": ["Q"],
        "unchanged": [],
        "refused": [],
        "failed": [],
        "aborted": True,
        "deadline_guard": False,
    }
    metadata = {
        "success": True,
        "stdout": "restored 1 addresses from a dead run (pid 4242)\n",
        "stderr": f"{executor.RESTORE_REPORT_TAG} {'ab' * 16} {json.dumps(report)}\n",
    }

    executor._result_from_run(
        folder,
        metadata,
        stdout_text="",
        stderr_text="",
        returncode=0,
        elapsed=0.1,
        control_target="live",
        report_nonce="ab" * 16,
    )

    saved = json.loads((folder / executor.RESTORE_REPORT_FILE).read_text(encoding="utf-8"))
    assert saved == [report]
    (ledger,) = zone.rglob("*.jsonl")
    (record,) = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert record["reason"] == "guarded_run_restore_complete"
    assert folder.name in record["subject"]


def test_a_finished_run_without_a_report_files_nothing(tmp_path: Path) -> None:
    folder = tmp_path / "exec"
    folder.mkdir()

    executor._result_from_run(
        folder,
        {"success": True, "stdout": "hello\n", "stderr": ""},
        stdout_text="",
        stderr_text="",
        returncode=0,
        elapsed=0.1,
        control_target="live",
        report_nonce="ab" * 16,
    )

    assert not (folder / executor.RESTORE_REPORT_FILE).exists()
