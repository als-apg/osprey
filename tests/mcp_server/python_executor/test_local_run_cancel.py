"""A cancelled executor run interrupts its child, drains it, and files its report.

MCP delivers a tool call's cancellation as an anyio ``CancelScope`` that
re-cancels on every await, so ``_execute_via_local`` answers a cancel inside a
shielded scope: ``SIGINT`` the sandbox, drain both pipes until it exits (bounded
by the run's own deadline, or by :data:`CANCEL_DRAIN_CAP_S` when the run has
none), kill it only when that bound runs out, file the restore reports its
execution record holds, and re-raise. The in-flight marker outlives the child.

Every child here is a stub script run by the test interpreter in place of the
wrapped one; it announces its phases by creating files in the execution folder
and waits on files the test creates. Every test cancels through an anyio
``CancelScope``, never ``task.cancel()``, and a test that cancels a running
child does so only once the spawn call has handed that child to the executor. No test asserts wall time: the
``fail_after`` bounds are hang guards, and "was not killed" is read off the
child's exit status.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
import textwrap
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from osprey.audit import writer as audit_writer
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor
from osprey.mcp_server.python_executor.executor import (
    INFLIGHT_FILE_PREFIX,
    INFLIGHT_FILE_SUFFIX,
    RESTORE_REPORT_FILE,
)
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey_connectors import posture_store

#: Hang guard for every wait in this module; never an assertion about speed.
HANG_GUARD_S = 60.0

#: More than the 1 MB the drain has to cope with, and more than any pipe buffer.
LARGE_OUTPUT_BYTES = 2 * 1024 * 1024

REPORT = {
    "restored": ["SR:Q1"],
    "unchanged": [],
    "refused": [],
    "failed": [],
    "aborted": True,
    "deadline_guard": False,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _stub(folder: Path, markers: Path, on_sigint: str, *, ignore_sigint: bool = False) -> str:
    """A child that announces ``ready``, then runs *on_sigint* when interrupted.

    *on_sigint* is the body of the handler: ``FOLDER``, ``MARKERS``, ``hold``,
    ``note`` and ``record_report`` are in scope; ``record_report`` writes the
    execution record a wrapped script leaves, holding :data:`REPORT`. ``ignore_sigint`` installs
    ``SIG_IGN`` instead, for a child that never answers the interrupt.
    """
    handler = textwrap.indent(textwrap.dedent(on_sigint).strip(), "    ")
    install = (
        "signal.signal(signal.SIGINT, signal.SIG_IGN)"
        if ignore_sigint
        else "signal.signal(signal.SIGINT, on_sigint)"
    )
    report_json = json.dumps(REPORT)
    return (
        textwrap.dedent(
            f"""
            import glob, json, os, signal, sys, time
            FOLDER = {str(folder)!r}
            MARKERS = {str(markers)!r}

            def record_report():
                record = {{"success": False, "restore_reports": [json.loads({report_json!r})]}}
                with open(os.path.join(FOLDER, "execution_metadata.json"), "w") as fh:
                    json.dump(record, fh)

            def note(name, text=""):
                with open(os.path.join(FOLDER, name), "w") as fh:
                    fh.write(text)

            def hold(name):
                path = os.path.join(FOLDER, name)
                while not os.path.exists(path):
                    time.sleep(0.01)

            def marker_count():
                return len(glob.glob(os.path.join(MARKERS, "{INFLIGHT_FILE_PREFIX}*{INFLIGHT_FILE_SUFFIX}")))

            def on_sigint(signum, frame):
            """
        )
        + handler
        + textwrap.dedent(
            f"""

            {install}
            note("ready")
            while True:
                time.sleep(0.05)
            """
        )
    )


class _Run:
    """One ``_execute_via_local`` call against a stub child, cancelled from outside."""

    def __init__(self, folder: Path, markers: Path, ledger: Path, spawned: list[Any]) -> None:
        self.folder = folder
        self.ledger = ledger
        self.markers = markers
        self.spawned = spawned
        self.outcome: dict[str, Any] = {}
        #: Set when the spawn call hands the child to the executor; made by
        #: :meth:`cancel` inside its event loop.
        self.handed_over: anyio.Event | None = None

    @property
    def proc(self) -> Any:
        (proc,) = self.spawned
        return proc

    def live_markers(self) -> list[Path]:
        return sorted(self.markers.glob(f"{INFLIGHT_FILE_PREFIX}*{INFLIGHT_FILE_SUFFIX}"))

    async def wait_for(self, name: str) -> None:
        with anyio.fail_after(HANG_GUARD_S):
            while not (self.folder / name).exists():
                await anyio.sleep(0.01)

    async def wait_until_running(self) -> None:
        """Return once the child is ready and the executor holds its process.

        The child can announce ``ready`` while the spawn call that started it
        has not yet returned in the executor; a cancel landing in that gap is
        one before the spawn, which winds nothing down.
        """
        assert self.handed_over is not None
        await self.wait_for("ready")
        with anyio.fail_after(HANG_GUARD_S):
            await self.handed_over.wait()

    def go(self, name: str) -> None:
        (self.folder / name).write_text("", encoding="utf-8")

    def cancel(
        self,
        timeout: float | None,
        *,
        before_cancel: Callable[[_Run], Awaitable[None]] | None = None,
        after_cancel: Callable[[_Run], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Start the run, cancel its scope once the child is ready, return the outcome."""

        async def main() -> None:
            self.handed_over = anyio.Event()
            scope = anyio.CancelScope()

            async def runner() -> None:
                with scope:
                    self.outcome["result"] = await executor._execute_via_local(
                        "print('never runs')", "readwrite", {"timeout": timeout}, self.folder
                    )
                self.outcome["cancelled_caught"] = scope.cancelled_caught

            with anyio.fail_after(HANG_GUARD_S):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(runner)
                    if before_cancel is None:
                        await self.wait_until_running()
                    else:
                        await before_cancel(self)
                    scope.cancel()
                    if after_cancel is not None:
                        await after_cancel(self)

        anyio.run(main)
        return self.outcome


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the marker directory and the audit ledger into *tmp_path*."""
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    ledger = tmp_path / "audit" / "executor.jsonl"
    monkeypatch.setattr(audit_writer, "ledger_path", lambda surface, identity=None: ledger)
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(executor, "resolve_agent_interpreter", lambda root=None: sys.executable)
    # These cancels are not about approval: the deployment asks nobody.
    monkeypatch.setattr("osprey.cli.build_posture_check.ask_capable", lambda *_args: False)
    monkeypatch.setattr(executor, "_apply_target_stamp", _stamp_live)
    return ledger


def _stamp_live(sandbox_env: dict[str, str]) -> str:
    """Stamp the run onto ``live``, whose journal lives under the test's project root."""
    sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
    return "live"


def _journal(root: Path) -> Path:
    """The ``live`` target's guarded-run journal under the project root *root*."""
    return root / "var" / "guarded_run" / "live" / "run.journal"


def _plant_journal(root: Path, *addresses: str) -> bytes:
    """Leave a killed run's journal on ``live`` holding *addresses*; return its bytes."""
    lines = [{"header": {"target": "live", "generation": 1, "pid": 4242}}] + [
        {"address": address, "value": 1.5} for address in addresses
    ]
    raw = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
    path = _journal(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


@pytest.fixture
def make_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated: Path
) -> Iterator[Callable[..., _Run]]:
    """Build a :class:`_Run` whose child is the stub for *on_sigint*.

    A child a failing test left behind is killed at teardown: every stub loops
    forever until it is interrupted.
    """
    spawned: list[Any] = []

    def _make(on_sigint: str = "os._exit(0)", *, ignore_sigint: bool = False) -> _Run:
        folder = tmp_path / "execution_cancel"
        (folder / "figures").mkdir(parents=True)
        markers = target_state.state_dir()
        script = _stub(folder, markers, on_sigint, ignore_sigint=ignore_sigint)
        monkeypatch.setattr(
            ExecutionWrapper, "create_wrapper", lambda self, code, execution_folder: script
        )
        real_exec = executor.asyncio.create_subprocess_exec
        run = _Run(folder, markers, isolated, spawned)

        async def spy_exec(*args: Any, **kwargs: Any) -> Any:
            proc = await real_exec(*args, **kwargs)
            spawned.append(proc)
            # Nothing awaits between here and the executor's own wait on the
            # child, so a waiter woken by this event finds the executor there.
            assert run.handed_over is not None
            run.handed_over.set()
            return proc

        monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spy_exec)
        return run

    yield _make
    for proc in spawned:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()


def _ledger_records(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


def _sandbox_records(ledger: Path) -> list[dict[str, Any]]:
    return [r for r in _ledger_records(ledger) if r["detail"].startswith("source=sandbox ")]


def _observed_records(ledger: Path) -> list[dict[str, Any]]:
    return [
        r for r in _ledger_records(ledger) if r["detail"].startswith("source=observed_on_disk ")
    ]


def _filed(folder: Path) -> dict[str, Any]:
    return json.loads((folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cancel_sigints_child_and_keeps_marker(make_run: Callable[..., _Run]) -> None:
    """The child is sent SIGINT (not killed), sees the marker, and the cancel re-raises."""
    run = make_run(
        """
        note("signal", str(signum))
        note("markers", str(marker_count()))
        os._exit(0)
        """
    )

    outcome = run.cancel(timeout=60)

    assert outcome == {"cancelled_caught": True}
    assert int((run.folder / "signal").read_text()) == signal.SIGINT
    assert int((run.folder / "markers").read_text()) == 1
    assert run.proc.returncode == 0
    assert run.live_markers() == []


def test_anyio_scope_cancel_keeps_marker_until_child_exit(make_run: Callable[..., _Run]) -> None:
    """While the interrupted child is still winding down, the marker is live."""
    run = make_run(
        """
        note("interrupted")
        hold("go")
        os._exit(0)
        """
    )
    seen: dict[str, Any] = {}

    async def after_cancel(r: _Run) -> None:
        await r.wait_for("interrupted")
        # The runner has been cancelled; the child has not exited yet.
        seen["markers"] = len(r.live_markers())
        seen["returncode"] = r.proc.returncode
        seen["finished"] = dict(r.outcome)
        r.go("go")

    outcome = run.cancel(timeout=60, after_cancel=after_cancel)

    assert seen == {"markers": 1, "returncode": None, "finished": {}}
    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == 0
    assert run.live_markers() == []


def test_cancel_drains_large_output_after_sigint(make_run: Callable[..., _Run]) -> None:
    """A child echoing >1 MB after SIGINT is drained, exits by itself, and its report is filed."""
    run = make_run(
        f"""
        sys.stdout.write("x" * {LARGE_OUTPUT_BYTES} + "\\n")
        sys.stdout.flush()
        sys.stderr.write("y" * {LARGE_OUTPUT_BYTES} + "\\n")
        sys.stderr.flush()
        record_report()
        os._exit(0)
        """
    )

    outcome = run.cancel(timeout=60)

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == 0  # exited by itself: never killed
    assert _filed(run.folder)["sandbox"] == [REPORT]


def test_cancel_before_spawn_reraises(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel that lands while the child is being spawned re-raises at once."""
    run = make_run()
    entered = anyio.Event()

    async def never_spawns(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await anyio.sleep_forever()

    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", never_spawns)

    async def before_cancel(r: _Run) -> None:
        with anyio.fail_after(HANG_GUARD_S):
            await entered.wait()
        # The marker is written before the spawn, so it is live here.
        assert len(r.live_markers()) == 1

    outcome = run.cancel(timeout=60, before_cancel=before_cancel)

    assert outcome == {"cancelled_caught": True}
    assert run.spawned == []
    assert _filed(run.folder)["sandbox"] == []
    (record,) = _ledger_records(run.ledger)
    assert record["reason"] == executor.REASON_JOURNAL_ABSENT
    assert run.live_markers() == []


def test_no_sigkill_before_deadline(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finite timeout bounds the drain by the run's deadline, not by the cap.

    The cap is shrunk to nothing; the child takes its time after the interrupt
    and must still exit by itself, because the run's deadline is far off.
    """
    monkeypatch.setattr(executor, "CANCEL_DRAIN_CAP_S", 0.001)
    run = make_run(
        """
        note("interrupted")
        hold("go")
        record_report()
        os._exit(0)
        """
    )

    async def after_cancel(r: _Run) -> None:
        await r.wait_for("interrupted")
        # Let well over the cap pass before releasing the child.
        for _ in range(20):
            await anyio.sleep(0.01)
        r.go("go")

    outcome = run.cancel(timeout=60, after_cancel=after_cancel)

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == 0
    assert (run.folder / RESTORE_REPORT_FILE).exists()


@pytest.mark.parametrize("timeout", [None, float("inf")], ids=["null", "infinite"])
def test_cancel_null_timeout_bounded(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch, timeout: float | None
) -> None:
    """With no finite deadline the drain is bounded by the cap; then the child is killed."""
    monkeypatch.setattr(executor, "CANCEL_DRAIN_CAP_S", 0.2)
    run = make_run(ignore_sigint=True)

    outcome = run.cancel(timeout=timeout)

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == -signal.SIGKILL
    assert run.live_markers() == []


def test_restore_report_in_ledger_and_folder(make_run: Callable[..., _Run]) -> None:
    """The report the interrupted child recorded lands in the folder and the audit ledger."""
    run = make_run(
        """
        sys.stdout.write("partial output\\n")
        sys.stdout.flush()
        record_report()
        os._exit(0)
        """
    )

    outcome = run.cancel(timeout=60)

    assert outcome == {"cancelled_caught": True}
    assert _filed(run.folder)["sandbox"] == [REPORT]
    (record,) = _sandbox_records(run.ledger)
    assert record["surface"] == "executor"
    assert run.folder.name in record["subject"]
    assert "SR:Q1" in record["detail"]
    (seen,) = _observed_records(run.ledger)
    assert seen["reason"] == executor.REASON_JOURNAL_ABSENT
    assert seen["detail"] == "source=observed_on_disk journal absent after abnormal exit"


def test_cancel_report_kept_when_pipes_outlive_child(make_run: Callable[..., _Run]) -> None:
    """A descendant holding the pipes open costs neither the report nor the exit status.

    The child hands its pipes to a grandchild (as a detached EPICS helper
    would), records its report and exits by itself. The wind-down waits on the
    child, not on the pipes, and keeps every byte it read.
    """
    run = make_run(
        """
        import subprocess
        subprocess.Popen([
            sys.executable, "-c",
            "import os, time\\n"
            "path = os.path.join(%r, 'release')\\n"
            "deadline = time.time() + 120\\n"
            "while not os.path.exists(path) and time.time() < deadline:\\n"
            "    time.sleep(0.05)\\n" % FOLDER,
        ])
        record_report()
        os._exit(0)
        """
    )

    try:
        outcome = run.cancel(timeout=60)
    finally:
        run.go("release")

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == 0  # exited by itself: never killed
    assert _filed(run.folder)["sandbox"] == [REPORT]
    (record,) = _sandbox_records(run.ledger)
    assert "SR:Q1" in record["detail"]
    assert run.live_markers() == []


def test_cancel_report_kept_when_child_is_killed(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report recorded before the child hangs survives the kill at the bound."""
    monkeypatch.setattr(executor, "CANCEL_DRAIN_CAP_S", 3.0)
    run = make_run(
        """
        record_report()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            time.sleep(0.05)
        """
    )

    outcome = run.cancel(timeout=None)

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == -signal.SIGKILL
    assert _filed(run.folder)["sandbox"] == [REPORT]
    (record,) = _sandbox_records(run.ledger)
    assert "SR:Q1" in record["detail"]
    assert run.live_markers() == []


def test_a_cancel_before_the_handle_exists_is_observed_on_disk(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A child killed while it was being spawned leaves its journal; the executor names it."""
    run = make_run()
    raw = b""

    async def spawned_then_cancelled(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal raw
        raw = _plant_journal(tmp_path, "SR:Q7")
        raise asyncio.CancelledError

    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spawned_then_cancelled)

    async def main() -> None:
        with pytest.raises(asyncio.CancelledError):
            await executor._execute_via_local(
                "print('never runs')", "readwrite", {"timeout": 60}, run.folder
            )

    asyncio.run(main())

    (record,) = _ledger_records(run.ledger)
    assert record["reason"] == executor.REASON_JOURNAL_LEFT
    assert record["detail"] == "source=observed_on_disk addresses=SR:Q7"
    assert all("confirmed" not in r["detail"] for r in _ledger_records(run.ledger))
    assert _filed(run.folder)["observed_on_disk"]["state"] == "entries"
    assert _journal(tmp_path).read_bytes() == raw
    assert run.live_markers() == []


def test_a_run_killed_at_its_timeout_names_the_journal_it_left(
    make_run: Callable[..., _Run], tmp_path: Path
) -> None:
    run = make_run(ignore_sigint=True)
    raw = _plant_journal(tmp_path, "SR:Q1", "SR:Q2")

    async def main() -> Any:
        run.handed_over = anyio.Event()
        return await executor._execute_via_local(
            "print('never runs')", "readwrite", {"timeout": 0.5}, run.folder
        )

    result = asyncio.run(main())

    assert result.failure_kind == executor.FAILURE_KIND_TIMEOUT
    (record,) = _ledger_records(run.ledger)
    assert record["reason"] == executor.REASON_JOURNAL_LEFT
    assert record["detail"] == "source=observed_on_disk addresses=SR:Q1,SR:Q2"
    assert _journal(tmp_path).read_bytes() == raw


#: A readwrite script that saves a malformed artifact manifest and exits cleanly.
_MALFORMED_MANIFEST = textwrap.dedent(
    """
    import json, os
    FOLDER = {folder!r}
    os.makedirs(os.path.join(FOLDER, "artifacts"), exist_ok=True)
    with open(os.path.join(FOLDER, "artifacts", "manifest.json"), "w") as fh:
        json.dump([{{}}], fh)
    with open(os.path.join(FOLDER, "execution_metadata.json"), "w") as fh:
        json.dump({{"success": True, "stdout": "", "stderr": ""}}, fh)
    """
)


def _run_to_exit(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch, script: str
) -> tuple[_Run, Any]:
    """Run *script* in place of the wrapped one and let it exit; return the run and outcome."""
    run = make_run()
    monkeypatch.setattr(
        ExecutionWrapper,
        "create_wrapper",
        lambda self, code, execution_folder: script.format(folder=str(execution_folder)),
    )
    outcome: dict[str, Any] = {}

    async def main() -> None:
        run.handed_over = anyio.Event()
        try:
            outcome["result"] = await executor._execute_via_local(
                "print('never runs')", "readwrite", {"timeout": 60}, run.folder
            )
        except Exception as error:  # the run record is the subject, not the outcome
            outcome["error"] = error

    asyncio.run(main())
    return run, outcome


def test_a_malformed_artifact_manifest_still_files_the_run_record(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _plant_journal(tmp_path, "SR:Q3")

    run, outcome = _run_to_exit(make_run, monkeypatch, _MALFORMED_MANIFEST)

    assert "error" not in outcome
    assert outcome["result"].artifacts == []
    (record,) = _observed_records(run.ledger)
    assert record["reason"] == executor.REASON_JOURNAL_LEFT
    assert record["detail"] == "source=observed_on_disk addresses=SR:Q3"
    assert (run.folder / RESTORE_REPORT_FILE).exists()
    assert _journal(tmp_path).read_bytes() == raw


def test_a_run_whose_result_cannot_be_built_still_files_one_record(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _plant_journal(tmp_path, "SR:Q4")

    def broken(_folder: Path) -> list[dict]:
        raise RuntimeError("artifact store unavailable")

    monkeypatch.setattr(executor, "collect_artifacts", broken)

    run, outcome = _run_to_exit(make_run, monkeypatch, _MALFORMED_MANIFEST)

    assert isinstance(outcome.get("error"), RuntimeError)
    (record,) = _ledger_records(run.ledger)
    assert record["reason"] == executor.REASON_JOURNAL_LEFT
    assert record["detail"] == "source=observed_on_disk addresses=SR:Q4"
    assert (run.folder / RESTORE_REPORT_FILE).exists()
