"""Tests for ComputeManager's supervision of the figure workers.

The unit cases run the manager over ``FakeSlots`` (conftest), whose jobs
spawn no process and end when a test says so. The real-child cases run real
``JobSlots`` over a worker command line patched to a short Python script.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest

from osprey.interfaces.lattice_dashboard import compute as compute_mod
from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.state import (
    FAST_FIGURES,
    VERIFICATION_FIGURES,
    LatticeState,
)
from osprey_connectors.process import ExitCause


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def broadcast(self, data: dict) -> None:
        self.events.append(data)

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == kind]


@pytest.fixture
def manager(tmp_path, fake_slots):
    """A ComputeManager over FakeSlots, with a seeded state."""
    state = LatticeState(tmp_path / "lattice")
    state.save(LatticeState._empty_state())
    broadcaster = RecordingBroadcaster()
    return ComputeManager(state, broadcaster, fake_slots), state, broadcaster


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def _output(state: LatticeState, name: str):
    return state.figures_dir / f"{name}.json"


class TestRefresh:
    async def test_refresh_fast_launches_all_fast(self, manager, fake_slots):
        mgr, _, broadcaster = manager
        launched = mgr.refresh_fast()
        await _settle()
        assert launched == list(FAST_FIGURES)
        assert fake_slots.launched == list(FAST_FIGURES)
        computing = [e for e in broadcaster.events if e.get("status") == "computing"]
        assert len(computing) == len(FAST_FIGURES)

    async def test_refresh_verification_launches_da_lma(self, manager):
        mgr, _, _ = manager
        assert mgr.refresh_verification() == list(VERIFICATION_FIGURES)

    async def test_refresh_one_runs_the_figure_worker(self, manager, fake_slots):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        await _settle()
        assert any("workers.optics" in part for part in fake_slots.jobs[0].argv)

    async def test_refresh_one_unknown_raises(self, manager):
        mgr, _, _ = manager
        with pytest.raises(ValueError, match="Unknown figure"):
            mgr.refresh_one("bogus")


class TestApply:
    async def test_success_marks_ready(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        _output(state, "optics").write_text("{}")
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        assert state.load()["figures"]["optics"]["status"] == "ready"
        assert broadcaster.of("figure_ready") == [{"type": "figure_ready", "name": "optics"}]

    async def test_summary_updates_merged_into_state(self, manager, fake_slots):
        """A worker's ``summary_updates`` block reaches state["summary"]."""
        mgr, state, broadcaster = manager
        seeded = state.load()
        seeded["summary"] = {"tunes": [0.30, 0.20], "energy_gev": 2.0}
        state.save(seeded)
        mgr.refresh_one("optics")
        await _settle()
        _output(state, "optics").write_text(
            json.dumps({"s_pos": [0.0], "summary_updates": {"tunes": [0.44, 0.31]}})
        )
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        summary = state.load()["summary"]
        assert summary["tunes"] == [0.44, 0.31]
        assert summary["energy_gev"] == 2.0
        assert broadcaster.of("state_updated")

    async def test_failed_worker_marks_error_with_its_stderr(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.FAILED, 1, "boom traceback")
        await _settle()

        fig = state.load()["figures"]["optics"]
        assert fig["status"] == "error"
        assert fig["error"] == "Worker exited with code 1: boom traceback"
        assert len(broadcaster.of("figure_error")) == 1

    async def test_missing_output_marks_error(self, manager, fake_slots):
        mgr, state, _ = manager
        mgr.refresh_one("optics")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        assert state.load()["figures"]["optics"]["error"] == (
            "Worker completed but no output file produced"
        )

    async def test_timed_out_worker_marks_failed(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("da")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.TIMED_OUT, -9)
        await _settle()

        fig = state.load()["figures"]["da"]
        assert fig["status"] == "error"
        assert fig["error"] == "Worker timed out after 300 s"
        assert len(broadcaster.of("figure_error")) == 1

    async def test_launch_failure_marks_error(self, manager, fake_slots, monkeypatch):
        mgr, state, broadcaster = manager

        async def boom(*_args, **_kwargs):
            raise OSError("no exec")

        mgr.refresh_one("optics")
        monkeypatch.setattr(fake_slots.jobs[0], "start", boom)
        await _settle()

        assert state.load()["figures"]["optics"]["error"] == "Failed to launch worker: no exec"
        assert len(broadcaster.of("figure_error")) == 1


class TestSupersede:
    async def test_relaunch_drops_the_old_completion(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        first = fake_slots.jobs[0]
        mgr.refresh_one("optics")
        # The first job completes before its successor reaps it.
        _output(state, "optics").write_text("{}")
        first.finish(ExitCause.COMPLETED)
        await _settle()

        assert broadcaster.of("figure_ready") == []
        assert state.load()["figures"]["optics"]["status"] == "computing"

    async def test_cancelled_worker_is_not_an_error(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.CANCELLED, -15)
        await _settle()

        assert broadcaster.of("figure_error") == []
        assert state.load()["figures"]["optics"]["status"] == "computing"

    async def test_double_refresh_never_flashes_error(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_fast()
        mgr.refresh_fast()
        await _settle()
        for job in fake_slots.jobs[len(FAST_FIGURES) :]:
            _output(state, job.name).write_text("{}")
            job.finish(ExitCause.COMPLETED)
        await _settle()

        assert broadcaster.of("figure_error") == []
        first_round = fake_slots.jobs[: len(FAST_FIGURES)]
        assert all(job._exit.cause is ExitCause.CANCELLED for job in first_round)
        assert {e["name"] for e in broadcaster.of("figure_ready")} == set(FAST_FIGURES)

    async def test_refresh_returns_before_the_previous_worker_is_reaped(self, manager, fake_slots):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        await _settle()
        first = fake_slots.jobs[0]

        mgr.refresh_one("optics")

        assert first._exit is None
        await _settle()
        assert first._exit.cause is ExitCause.CANCELLED

    async def test_stop_all_ends_every_job_and_task(self, manager, fake_slots):
        mgr, _, broadcaster = manager
        mgr.refresh_fast()
        await _settle()

        await mgr.stop_all()

        assert all(job._exit.cause is ExitCause.CANCELLED for job in fake_slots.jobs)
        assert mgr._tasks == set()
        assert broadcaster.of("figure_error") == []


# ── Real children ────────────────────────────────────────


def _script_argv(workdir, script: str):
    """A worker command line running *script* with ``{output}`` and ``{pids}`` filled in."""
    pids = workdir / "pids"
    pids.mkdir(parents=True)

    def argv(name, _job_path, output_path):  # noqa: ARG001
        body = script.format(output=str(output_path), pids=str(pids))
        return [sys.executable, "-c", body]

    return argv, pids


_SLEEPER = """
import os, pathlib, time
pathlib.Path({pids!r}, str(os.getpid())).write_text("")
time.sleep(60)
"""

_WRITER = """
import json, pathlib
pathlib.Path({output!r}).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path({output!r}).write_text(json.dumps({{"s_pos": [0.0]}}))
"""


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def _until(predicate, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


class TestRealChildren:
    async def test_a_relaunch_leaves_no_figure_error_and_the_new_result_lands(
        self, tmp_path, monkeypatch
    ):
        state = LatticeState(tmp_path / "lattice")
        state.save(LatticeState._empty_state())
        broadcaster = RecordingBroadcaster()
        mgr = ComputeManager(state, broadcaster)
        sleeper, pids = _script_argv(tmp_path / "first", _SLEEPER)
        monkeypatch.setattr(compute_mod, "worker_argv", sleeper)
        mgr.refresh_one("optics")
        await _until(lambda: any(pids.iterdir()))
        (first_pid,) = (int(p.name) for p in pids.iterdir())

        writer, _ = _script_argv(tmp_path / "second", _WRITER)
        monkeypatch.setattr(compute_mod, "worker_argv", writer)
        mgr.refresh_one("optics")
        await _until(lambda: bool(broadcaster.of("figure_ready")))

        assert broadcaster.of("figure_error") == []
        assert state.load()["figures"]["optics"]["status"] == "ready"
        assert _gone(first_pid)
        await mgr.stop_all()


def test_app_shutdown_reaps_every_worker(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from osprey.interfaces.lattice_dashboard.app import create_app

    sleeper, pids = _script_argv(tmp_path / "work", _SLEEPER)
    monkeypatch.setattr(compute_mod, "worker_argv", sleeper)
    app = create_app(workspace_root=tmp_path, render_root=tmp_path)

    with TestClient(app) as client:
        assert client.post("/api/refresh/optics").status_code == 200
        deadline = time.monotonic() + 20
        while not any(pids.iterdir()):
            assert time.monotonic() < deadline, "the worker never started"
            time.sleep(0.05)
        (pid,) = (int(p.name) for p in pids.iterdir())
        assert not _gone(pid)

    assert _gone(pid)
