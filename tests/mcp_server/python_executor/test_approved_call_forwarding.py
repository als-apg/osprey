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


def test_only_a_readwrite_sandbox_carries_the_approved_fields() -> None:
    readwrite: dict[str, str] = {}
    executor._apply_approved_call(readwrite, "readwrite", "ab" * 32, "live")
    assert readwrite == {
        "OSPREY_APPROVED_JOURNAL_SHA256": "ab" * 32,
        "OSPREY_APPROVED_TARGET": "live",
    }

    readonly: dict[str, str] = {}
    executor._apply_approved_call(readonly, "readonly", "ab" * 32, "live")
    assert readonly == {}

    missing: dict[str, str] = {}
    executor._apply_approved_call(missing, "readwrite", None, None)
    assert missing == {}, "a field the call does not carry stays absent"


def test_a_readonly_wrapper_binds_nothing() -> None:
    script = ExecutionWrapper(execution_mode="readonly").create_wrapper("x = 1\n")
    assert "_open_approved_call" not in script


class _Sandbox:
    """A deployment repo with a planted journal, and one wrapped readwrite run in it."""

    def __init__(self, root: Path, values: dict[str, float], journal: dict[str, float]) -> None:
        self.root = root
        (root / "build").mkdir(parents=True)
        (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
        config = {
            "control_system": {"type": "epics", "connector": {"epics": {}}},
            "approval": {"enabled": True, "default_policy": "always"},
        }
        (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
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

    def digest(self) -> str:
        return hashlib.sha256(self.journal.read_bytes()).hexdigest()

    def run(self, env_extra: dict[str, str]) -> dict[str, Any]:
        script = self.folder / "wrapped_script.py"
        script.write_text(
            ExecutionWrapper(execution_mode="readwrite").create_wrapper(_PROBE, self.folder),
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
        "stderr": f"{executor.RESTORE_REPORT_TAG} {json.dumps(report)}\n",
    }

    executor._result_from_run(
        folder,
        metadata,
        stdout_text="",
        stderr_text="",
        returncode=0,
        elapsed=0.1,
        control_target="live",
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
    )

    assert not (folder / executor.RESTORE_REPORT_FILE).exists()
