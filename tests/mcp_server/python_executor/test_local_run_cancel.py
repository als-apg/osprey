"""A cancelled executor run interrupts its child, drains it, and files its report.

MCP delivers a tool call's cancellation as an anyio ``CancelScope`` that
re-cancels on every await, so ``_execute_via_local`` answers a cancel inside a
shielded scope: ``SIGINT`` the sandbox, drain both pipes until it exits (bounded
by the run's own deadline, or by :data:`CANCEL_DRAIN_CAP_S` when the run has
none), kill it only when that bound runs out, file any restore report it
printed, and re-raise. The in-flight marker outlives the child.

Every child here is a stub script run by the test interpreter in place of the
wrapped one; it announces its phases by creating files in the execution folder
and waits on files the test creates. Every test cancels through an anyio
``CancelScope``, never ``task.cancel()``. No test asserts wall time: the
``fail_after`` bounds are hang guards, and "was not killed" is read off the
child's exit status.
"""

from __future__ import annotations

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
    RESTORE_REPORT_TAG,
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
    ``note`` and ``REPORT_LINE`` are in scope. ``ignore_sigint`` installs
    ``SIG_IGN`` instead, for a child that never answers the interrupt.
    """
    handler = textwrap.indent(textwrap.dedent(on_sigint).strip(), "    ")
    install = (
        "signal.signal(signal.SIGINT, signal.SIG_IGN)"
        if ignore_sigint
        else "signal.signal(signal.SIGINT, on_sigint)"
    )
    report_line = f"{RESTORE_REPORT_TAG} {json.dumps(REPORT)}"
    return (
        textwrap.dedent(
            f"""
            import glob, os, signal, sys, time
            FOLDER = {str(folder)!r}
            MARKERS = {str(markers)!r}
            REPORT_LINE = {report_line!r}

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
            scope = anyio.CancelScope()

            async def runner() -> None:
                with scope:
                    self.outcome["result"] = await executor._execute_via_local(
                        "print('never runs')", "readonly", {"timeout": timeout}, self.folder
                    )
                self.outcome["cancelled_caught"] = scope.cancelled_caught

            with anyio.fail_after(HANG_GUARD_S):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(runner)
                    if before_cancel is None:
                        await self.wait_for("ready")
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
    return ledger


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

        async def spy_exec(*args: Any, **kwargs: Any) -> Any:
            proc = await real_exec(*args, **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spy_exec)
        return _Run(folder, markers, isolated, spawned)

    yield _make
    for proc in spawned:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()


def _ledger_records(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


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
    """A child echoing >1 MB after SIGINT is drained, exits by itself, and its report parses."""
    run = make_run(
        f"""
        sys.stdout.write("x" * {LARGE_OUTPUT_BYTES} + "\\n")
        sys.stdout.flush()
        sys.stderr.write("y" * {LARGE_OUTPUT_BYTES} + "\\n" + REPORT_LINE + "\\n")
        sys.stderr.flush()
        os._exit(0)
        """
    )

    outcome = run.cancel(timeout=60)

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == 0  # exited by itself: never killed
    saved = json.loads((run.folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8"))
    assert saved == [REPORT]


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
    assert not (run.folder / RESTORE_REPORT_FILE).exists()
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
        sys.stderr.write(REPORT_LINE + "\\n")
        sys.stderr.flush()
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
    """The report the interrupted child printed lands in the folder and the audit ledger."""
    run = make_run(
        """
        sys.stdout.write("partial output\\n")
        sys.stderr.write(REPORT_LINE + "\\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
        """
    )

    outcome = run.cancel(timeout=60)

    assert outcome == {"cancelled_caught": True}
    assert json.loads((run.folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8")) == [REPORT]
    (record,) = _ledger_records(run.ledger)
    assert record["surface"] == "executor"
    assert run.folder.name in record["subject"]
    assert "SR:Q1" in record["detail"]


def test_cancel_report_kept_when_pipes_outlive_child(make_run: Callable[..., _Run]) -> None:
    """A descendant holding the pipes open costs neither the report nor the exit status.

    The child hands its pipes to a grandchild (as a detached EPICS helper
    would), prints its report and exits by itself. The wind-down waits on the
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
        sys.stderr.write(REPORT_LINE + "\\n")
        sys.stderr.flush()
        os._exit(0)
        """
    )

    try:
        outcome = run.cancel(timeout=60)
    finally:
        run.go("release")

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == 0  # exited by itself: never killed
    assert json.loads((run.folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8")) == [REPORT]
    (record,) = _ledger_records(run.ledger)
    assert "SR:Q1" in record["detail"]
    assert run.live_markers() == []


def test_cancel_report_kept_when_child_is_killed(
    make_run: Callable[..., _Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report printed before the child hangs survives the kill at the bound."""
    monkeypatch.setattr(executor, "CANCEL_DRAIN_CAP_S", 3.0)
    run = make_run(
        """
        sys.stderr.write(REPORT_LINE + "\\n")
        sys.stderr.flush()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            time.sleep(0.05)
        """
    )

    outcome = run.cancel(timeout=None)

    assert outcome == {"cancelled_caught": True}
    assert run.proc.returncode == -signal.SIGKILL
    assert json.loads((run.folder / RESTORE_REPORT_FILE).read_text(encoding="utf-8")) == [REPORT]
    (record,) = _ledger_records(run.ledger)
    assert "SR:Q1" in record["detail"]
    assert run.live_markers() == []
