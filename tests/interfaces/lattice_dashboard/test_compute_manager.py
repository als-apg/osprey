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
    Selection,
    capabilities_for,
)
from osprey.interfaces.lattice_dashboard.workers._base import save_data
from osprey.simulation.engines.pyat import Prepared
from osprey_connectors.process import ExitCause


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def broadcast(self, data: dict) -> None:
        self.events.append(data)

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == kind]


def _ready_state(root) -> LatticeState:
    """A state whose selection is a ready periodic model."""
    state = LatticeState(root / "lattice")
    state.selection = Selection(
        model="SR",
        status="ready",
        deck=root / "SR.json",
        deck_sha256="0" * 64,
        prepared=Prepared(solve="periodic", twiss_in=None, rest_mass_gev=0.000511, length_m=8.0),
        capabilities=capabilities_for("periodic"),
        families={"QF": {"param": "K"}},
        summary={"energy_gev": 2.0, "periodicity": 1},
    )
    return state


@pytest.fixture
def manager(tmp_path, fake_slots):
    """A ComputeManager over FakeSlots, with a ready selection."""
    state = _ready_state(tmp_path)
    broadcaster = RecordingBroadcaster()
    return ComputeManager(state, broadcaster, fake_slots), state, broadcaster


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def _output(state: LatticeState, name: str):
    return state.figure_path(name, state.figure_key(name))


def _status(mgr: ComputeManager, name: str) -> dict:
    return mgr.figure_status(name)


def _write(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


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

    async def test_no_ready_selection_launches_nothing(self, tmp_path, fake_slots):
        mgr = ComputeManager(LatticeState(tmp_path / "lattice"), RecordingBroadcaster(), fake_slots)

        assert mgr.refresh_fast() == []
        assert mgr.refresh_one("optics") is False
        assert fake_slots.launched == []

    async def test_the_job_file_names_the_key_and_the_job(self, manager, fake_slots):
        mgr, state, _ = manager
        mgr.refresh_one("da")
        await _settle()
        job = fake_slots.jobs[0]

        spec = json.loads(open(job.argv[-2]).read())

        assert spec["job_id"] == job.job
        assert spec["key"] == state.figure_key("da")
        assert spec["settings"] == state.get_settings()["da"]
        assert spec["families"] == {"QF": "K"}
        assert job.argv[-1] == str(_output(state, "da"))

    async def test_refresh_one_unknown_raises(self, manager):
        mgr, _, _ = manager
        with pytest.raises(ValueError, match="Unknown figure"):
            mgr.refresh_one("bogus")


class TestApply:
    async def test_success_marks_ready(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        _write(_output(state, "optics"))
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        assert _status(mgr, "optics")["status"] == "ready"
        assert broadcaster.of("figure_ready") == [{"type": "figure_ready", "name": "optics"}]
        assert list(state.jobs_dir.iterdir()) == []

    async def test_optics_asks_for_a_state_reread(self, manager, fake_slots):
        """The summary chips read the optics figure through /api/state."""
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        _write(_output(state, "optics"))
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        assert broadcaster.of("state_updated")

    async def test_failed_worker_marks_error_with_its_stderr(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.FAILED, 1, "boom traceback")
        await _settle()

        fig = _status(mgr, "optics")
        assert fig["status"] == "failed"
        assert fig["error"] == "Worker exited with code 1: boom traceback"
        assert len(broadcaster.of("figure_error")) == 1

    async def test_missing_output_marks_error(self, manager, fake_slots):
        mgr, state, _ = manager
        mgr.refresh_one("optics")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        assert _status(mgr, "optics")["error"] == "Worker completed but no output file produced"

    async def test_timed_out_worker_marks_failed(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("da")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.TIMED_OUT, -9)
        await _settle()

        fig = _status(mgr, "da")
        assert fig["status"] == "failed"
        assert fig["error"] == "Worker timed out after 300 s"
        assert len(broadcaster.of("figure_error")) == 1

    async def test_launch_failure_marks_error(self, manager, fake_slots, monkeypatch):
        mgr, state, broadcaster = manager

        async def boom(*_args, **_kwargs):
            raise OSError("no exec")

        mgr.refresh_one("optics")
        monkeypatch.setattr(fake_slots.jobs[0], "start", boom)
        await _settle()

        assert _status(mgr, "optics")["error"] == "Failed to launch worker: no exec"
        assert len(broadcaster.of("figure_error")) == 1


class TestStatus:
    async def test_a_changed_input_turns_ready_into_stale(self, manager, fake_slots):
        mgr, state, _ = manager
        mgr.refresh_one("optics")
        await _settle()
        save_data(
            {"key": state.figure_key("optics"), "job_id": 1, "deck_sha256": "0" * 64},
            {},
            _output(state, "optics"),
        )
        fake_slots.jobs[0].finish(ExitCause.COMPLETED)
        await _settle()

        state.set_param("QF", 1.2)

        assert _status(mgr, "optics")["status"] == "stale"
        assert _status(mgr, "resonance")["status"] == "not_computed"

    async def test_computing_is_the_current_job_on_the_current_key(self, manager):
        mgr, state, _ = manager
        mgr.refresh_one("optics")

        assert _status(mgr, "optics")["status"] == "computing"
        state.set_param("QF", 1.2)
        assert _status(mgr, "optics")["status"] == "not_computed"


class TestSupersede:
    async def test_relaunch_drops_the_old_completion(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        first = fake_slots.jobs[0]
        mgr.refresh_one("optics")
        # The first job completes before its successor reaps it.
        first.finish(ExitCause.COMPLETED)
        await _settle()

        assert broadcaster.of("figure_ready") == []
        assert _status(mgr, "optics")["status"] == "computing"

    async def test_cancelled_worker_is_not_an_error(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_one("optics")
        await _settle()
        fake_slots.jobs[0].finish(ExitCause.CANCELLED, -15)
        await _settle()

        assert broadcaster.of("figure_error") == []
        assert _status(mgr, "optics")["status"] == "not_computed"

    async def test_double_refresh_never_flashes_error(self, manager, fake_slots):
        mgr, state, broadcaster = manager
        mgr.refresh_fast()
        mgr.refresh_fast()
        await _settle()
        for job in fake_slots.jobs[len(FAST_FIGURES) :]:
            _write(_output(state, job.name))
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
        state = _ready_state(tmp_path)
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
        assert mgr.figure_status("optics")["status"] == "ready"
        assert _gone(first_pid)
        await mgr.stop_all()


def test_app_shutdown_reaps_every_worker(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.interfaces.lattice_dashboard.test_app import _write_render

    from osprey.interfaces.lattice_dashboard.app import create_app

    sleeper, pids = _script_argv(tmp_path / "work", _SLEEPER)
    monkeypatch.setattr(compute_mod, "worker_argv", sleeper)
    render = _write_render(tmp_path / "render", served=["SR"], models={"SR": {"solve": "periodic"}})
    app = create_app(workspace_root=tmp_path / "ws", render_root=render)

    with TestClient(app):
        # The startup resolve launches the fast figures.
        deadline = time.monotonic() + 20
        while len(list(pids.iterdir())) < len(FAST_FIGURES):
            assert time.monotonic() < deadline, "the workers never started"
            time.sleep(0.05)
        started = [int(p.name) for p in pids.iterdir()]
        assert not any(_gone(pid) for pid in started)

    assert all(_gone(pid) for pid in started)
