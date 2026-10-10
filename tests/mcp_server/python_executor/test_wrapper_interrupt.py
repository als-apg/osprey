"""An interrupted sandbox run leaves its record and exits by itself.

The executor cancels a run by sending its child ``SIGINT``. The wrapped script
must then end the way a finished one does: the failure recorded in
``execution_metadata.json``, the captured output echoed to the pipes, and the
process gone through ``os._exit`` rather than interpreter shutdown, where a
control-system client's shutdown hook (pyepics' ``finalize_libca``) can wedge
it. A guarded run interrupted inside its journaled span restores the
setpoints it moved and prints one ``OSPREY_GUARDED_RUN_RESTORE`` report line;
the parent reads that report out of the drained pipes into the audit ledger and
the execution folder.

Every child here is a real :class:`ExecutionWrapper`-wrapped script run by the
test interpreter; the control system is a dict behind stubbed
``osprey.runtime`` functions. "Exits by itself" means the child closes its
pipes within :data:`CHILD_TIMEOUT_S` while an ``atexit`` hook that would block
far longer is registered - a child that went through interpreter shutdown
would hang on it and time out.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import (
    RESTORE_REPORT_FILE,
    RESTORE_REPORT_TAG,
    _record_restore_report,
)
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"

#: How long a child gets to reach a phase, and to exit after being interrupted.
CHILD_TIMEOUT_S = 60.0

#: The stand-in for a wedging shutdown hook blocks far longer than the child gets.
_HOOK_BLOCK_S = 3600

#: Setpoints of the dict machine before any tool ran.
HOME = {"A": 1.0, "B": 2.0}

#: The control target and generation a guarded-run child is stamped with.
_STAMP = {"OSPREY_CONTROL_TARGET": "live", "OSPREY_CONTROL_TARGET_GENERATION": "4"}

#: Shared by every child: a blocking shutdown hook, a dict machine behind stubbed
#: ``osprey.runtime`` functions, and ``hold(phase)``, which announces a phase by
#: creating ``<phase>.ready`` and then waits for the parent's ``<phase>.go``.
_PREAMBLE = textwrap.dedent(
    f"""
    import atexit, json, os, time
    atexit.register(lambda: time.sleep({_HOOK_BLOCK_S}))

    import osprey.runtime

    _workdir = os.environ["WRAPPER_INTERRUPT_WORKDIR"]
    values = dict({HOME!r})

    def _dump_values():
        with open(os.path.join(_workdir, "values.json"), "w") as handle:
            json.dump(values, handle)

    def hold(name):
        open(os.path.join(_workdir, name + ".ready"), "w").close()
        go = os.path.join(_workdir, name + ".go")
        while not os.path.exists(go):
            time.sleep(0.01)

    def _read_channels(addresses, *, timeout=None):
        return [values[a] for a in addresses]

    def _write_channel(address, value, **kwargs):
        values[address] = value
        _dump_values()

    osprey.runtime.read_channels = _read_channels
    osprey.runtime.write_channel = _write_channel
    osprey.runtime.channel_limits = lambda address: None
    _dump_values()
    """
)

#: A guarded run interrupted inside its journaled span after moving ``A``.
_RUN_TOOL_CODE = _PREAMBLE + textwrap.dedent(
    """
    from osprey.runtime.guarded_run import journaled_run
    from osprey.runtime.journal import journaled_write

    def put(address, value):
        journaled_write([address], lambda: osprey.runtime.write_channel(address, value))

    with journaled_run("live"):
        put("A", 5.0)
        hold("span")
        put("B", 6.0)
    print("AFTER RUN_TOOL")
    osprey.runtime.write_channel("B", 99.0)
    results = {"finished": True}
    """
)

#: Interrupted while "loading" (before any tool runs), the way a pyAML
#: configuration load is: long, and inside the user code.
_LOAD_CODE = _PREAMBLE + textwrap.dedent(
    """
    print("loading")
    hold("load")
    # Longer than the parent waits: only an honoured SIGINT ends this load.
    for _ in range(int(600 / 0.01)):
        time.sleep(0.01)
    print("LOADED")
    results = {"loaded": True}
    """
)


class _Child:
    """One wrapped script run as a subprocess; the parent SIGINTs it at a phase."""

    def __init__(
        self,
        workdir: Path,
        user_code: str,
        *,
        inherit_sig_ign: bool,
        deployment: bool = False,
    ) -> None:
        self.workdir = workdir
        self.execution_folder = workdir / "exec"
        self.execution_folder.mkdir()
        script = self.execution_folder / "wrapped_script.py"
        script.write_text(
            ExecutionWrapper(execution_mode="readwrite").create_wrapper(
                user_code, self.execution_folder
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC_ROOT), env.get("PYTHONPATH")]))
        env["WRAPPER_INTERRUPT_WORKDIR"] = str(workdir)
        env["OSPREY_AGENT_DATA_ROOT"] = str(workdir / "agent_data")
        env.pop("OSPREY_EXECUTION_DEADLINE", None)
        env.pop("CONFIG_FILE", None)
        if deployment:
            _provision_deployment(workdir)
            env.pop("OSPREY_CONFIG", None)
            env.update(_STAMP)

        def ignore_sigint() -> None:
            # The disposition an asyncio child of a process that ignores SIGINT
            # starts with: SIG_IGN survives exec, and Python keeps it.
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=ignore_sigint if inherit_sig_ign else None,
        )

    def interrupt_at(self, phase: str) -> None:
        """Wait until the child reaches *phase*, send SIGINT, then let it go on."""
        ready = self.workdir / f"{phase}.ready"
        deadline = time.monotonic() + CHILD_TIMEOUT_S
        while not ready.exists():
            if self.proc.poll() is not None:
                out, err = self.proc.communicate()
                pytest.fail(
                    f"child exited {self.proc.returncode} before {phase}:\n"
                    f"{out.decode()}{err.decode()}"
                )
            if time.monotonic() > deadline:
                pytest.fail(f"child did not reach {phase} in time")
            time.sleep(0.01)
        self.proc.send_signal(signal.SIGINT)
        (self.workdir / f"{phase}.go").touch()

    def drain(self) -> tuple[bytes, bytes]:
        """Both pipes, drained to EOF; fails if the child does not exit by itself."""
        try:
            return self.proc.communicate(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, err = self.proc.communicate()
            pytest.fail(
                "the interrupted child did not exit by itself (wedged in shutdown?):\n"
                f"{out.decode(errors='replace')}{err.decode(errors='replace')}"
            )

    def metadata(self) -> dict[str, Any]:
        path = self.execution_folder / "execution_metadata.json"
        assert path.exists(), "the interrupted run left no execution_metadata.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def values(self) -> dict[str, float]:
        return json.loads((self.workdir / "values.json").read_text(encoding="utf-8"))

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.communicate()


@pytest.fixture
def child(tmp_path: Path) -> Iterator[Callable[..., _Child]]:
    """Starts wrapped children; any still running at teardown is killed."""
    started: list[_Child] = []

    def start(user_code: str, *, inherit_sig_ign: bool = False, deployment: bool = False) -> _Child:
        workdir = tmp_path / f"run{len(started)}"
        workdir.mkdir()
        proc = _Child(workdir, user_code, inherit_sig_ign=inherit_sig_ign, deployment=deployment)
        started.append(proc)
        return proc

    yield start
    for proc in started:
        proc.kill()


def _provision_deployment(workdir: Path) -> None:
    """Make *workdir* a deployment repo on a live baseline, so a guarded run resolves."""
    (workdir / "build").mkdir()
    (workdir / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    (workdir / "build" / "config.yml").write_text(
        yaml.safe_dump({"control_system": {"type": "epics", "connector": {"epics": {}}}}),
        encoding="utf-8",
    )


def _tagged(text: str) -> list[dict[str, Any]]:
    prefix = RESTORE_REPORT_TAG + " "
    return [
        json.loads(line[len(prefix) :]) for line in text.splitlines() if line.startswith(prefix)
    ]


def _ledger_records(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


def test_sigint_mid_run_tool_persists_report(child: Callable[..., _Child], tmp_path: Path) -> None:
    """SIGINT inside the journaled span: restored, reported, recorded, and the child exits itself."""
    run = child(_RUN_TOOL_CODE, deployment=True)
    run.interrupt_at("span")
    out, err = run.drain()

    assert run.values() == HOME
    stdout, stderr = out.decode(), err.decode()
    assert "AFTER RUN_TOOL" not in stdout
    (report,) = _tagged(stdout) + _tagged(stderr)
    assert report["aborted"] is True
    assert report["restored"] == ["A"]

    metadata = run.metadata()
    assert metadata["success"] is False
    assert metadata["error_type"] == "KeyboardInterrupt"
    assert metadata["restore_report"] == report

    ledger = tmp_path / "audit" / "executor.jsonl"
    reports = _record_restore_report(out, err, run.execution_folder, ledger)
    assert reports == [report]
    saved = json.loads((run.execution_folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8"))
    assert saved == [report]
    (record,) = _ledger_records(ledger)
    assert record["surface"] == "executor"
    assert record["reason"] == "guarded_run_restore_complete"
    assert run.execution_folder.name in record["subject"]
    assert "A" in record["detail"]


def test_sigint_mid_load_persists_metadata(child: Callable[..., _Child]) -> None:
    """SIGINT while the script loads: the record says so and the child exits itself."""
    run = child(_LOAD_CODE)
    run.interrupt_at("load")
    out, err = run.drain()

    assert "LOADED" not in out.decode()
    metadata = run.metadata()
    assert metadata["success"] is False
    assert metadata["error_type"] == "KeyboardInterrupt"
    assert "loading" in metadata["stdout"]
    # The captured output reaches the pipes, as it does for a finished run.
    assert "loading" in out.decode()
    assert "KeyboardInterrupt" in err.decode()


def test_sigint_mid_load_with_inherited_sig_ign(child: Callable[..., _Child]) -> None:
    """A child started with SIGINT ignored still honours it: the prologue restores it."""
    run = child(_LOAD_CODE, inherit_sig_ign=True)
    run.interrupt_at("load")
    out, _err = run.drain()

    assert "LOADED" not in out.decode()
    metadata = run.metadata()
    assert metadata["success"] is False
    assert metadata["error_type"] == "KeyboardInterrupt"


def test_sigint_plain_script_skips_blocking_atexit(child: Callable[..., _Child]) -> None:
    """A plain interrupted script leaves through ``os._exit``, past a blocking hook."""
    code = _PREAMBLE + textwrap.dedent(
        """
        hold("work")
        while True:
            time.sleep(0.01)
        """
    )
    run = child(code)
    run.interrupt_at("work")
    run.drain()

    assert run.proc.returncode == 0
    metadata = run.metadata()
    assert metadata["error_type"] == "KeyboardInterrupt"
    assert metadata["success"] is False
    assert "restore_report" not in metadata


def test_restore_report_parsed_to_ledger_and_folder(tmp_path: Path) -> None:
    """Every tagged line on either pipe lands in the folder and the ledger; noise does not."""
    first = {
        "restored": ["SR:Q1"],
        "unchanged": [],
        "refused": [],
        "failed": [],
        "aborted": False,
        "deadline_guard": True,
    }
    second = {
        "restored": [],
        "unchanged": ["SR:Q2"],
        "refused": [["SR:Q3", "limits", 4.5]],
        "failed": [["SR:Q4", "read failed"]],
        "aborted": True,
        "deadline_guard": False,
    }
    stdout = (
        "hello\n"
        f"{RESTORE_REPORT_TAG} {json.dumps(first)}\n"
        f"{RESTORE_REPORT_TAG} not json\n"
        f"not a {RESTORE_REPORT_TAG} line\n"
    ).encode()
    stderr = f"warning\n{RESTORE_REPORT_TAG} {json.dumps(second)}\n"
    folder = tmp_path / "execution_x"
    folder.mkdir()
    ledger = tmp_path / "audit" / "executor.jsonl"

    reports = _record_restore_report(stdout, stderr, folder, ledger)

    assert reports == [first, second]
    assert json.loads((folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8")) == [
        first,
        second,
    ]
    records = _ledger_records(ledger)
    assert len(records) == 2
    assert all(r["surface"] == "executor" and folder.name in r["subject"] for r in records)
    assert "SR:Q1" in records[0]["detail"]
    assert "SR:Q3" in records[1]["detail"] and "SR:Q4" in records[1]["detail"]
    # Addresses are identifiers; the value a refused write left behind is not.
    assert "4.5" not in records[1]["detail"]
    assert records[0]["reason"] != records[1]["reason"]

    # No report: nothing written anywhere.
    quiet = tmp_path / "execution_y"
    quiet.mkdir()
    other_ledger = tmp_path / "audit" / "other.jsonl"
    assert _record_restore_report(b"plain\n", b"", quiet, other_ledger) == []
    assert not (quiet / RESTORE_REPORT_FILE).exists()
    assert not other_ledger.exists()

    # Never raises, even on an unusable folder.
    assert _record_restore_report(stdout, "", tmp_path / "missing" / "dir", ledger) == [first]


def test_restore_report_tag_matches_the_guarded_run() -> None:
    """The executor's copy of the tag is the one the guarded run's restore prints."""
    from osprey.runtime import guarded_run

    assert RESTORE_REPORT_TAG is guarded_run.RESTORE_REPORT_TAG
