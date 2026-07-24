"""Tests for ComputeManager subprocess orchestration.

All subprocess spawning is faked; no real worker processes are launched.
The monitor thread is exercised by calling _monitor_worker directly so the
test stays deterministic and leaves no background threads running.
"""

from __future__ import annotations

import subprocess
import threading

import pytest

from osprey.interfaces.lattice_dashboard import compute as compute_mod
from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.state import (
    FAST_FIGURES,
    VERIFICATION_FIGURES,
    LatticeState,
)


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def broadcast(self, data: dict) -> None:
        self.events.append(data)


class FakePopen:
    """Stand-in for subprocess.Popen that never spawns a process."""

    instances: list[FakePopen] = []

    def __init__(self, cmd, stdout=None, stderr=None, poll_result=None):
        self.cmd = cmd
        self.pid = 12345
        self._poll_result = poll_result
        self.terminated = False
        self.killed = False
        FakePopen.instances.append(self)

    def poll(self):
        return self._poll_result

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def communicate(self, timeout=None):
        return (b"", b"")


@pytest.fixture(autouse=True)
def _reset_fake_popen():
    FakePopen.instances = []
    yield
    FakePopen.instances = []


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A ComputeManager whose subprocess spawns and monitor threads are inert."""
    state = LatticeState(tmp_path / "lattice")
    # Seed a state file so mark_* helpers have a figures dict to update.
    state.save(LatticeState._empty_state())
    broadcaster = RecordingBroadcaster()

    monkeypatch.setattr(compute_mod.subprocess, "Popen", FakePopen)
    # Neutralize the background monitor thread; monitor logic is tested directly.
    monkeypatch.setattr(compute_mod.threading, "Thread", _NoopThread)

    return ComputeManager(state, broadcaster), state, broadcaster


class _NoopThread:
    def __init__(self, *args, **kwargs):
        self._target = kwargs.get("target")

    def start(self):
        pass


class TestRefresh:
    def test_refresh_fast_launches_all_fast(self, manager):
        mgr, state, broadcaster = manager
        launched = mgr.refresh_fast()
        assert launched == list(FAST_FIGURES)
        assert len(FakePopen.instances) == len(FAST_FIGURES)
        # Each launch marks the figure computing and broadcasts status
        computing = [e for e in broadcaster.events if e.get("status") == "computing"]
        assert len(computing) == len(FAST_FIGURES)

    def test_refresh_verification_launches_da_lma(self, manager):
        mgr, _, _ = manager
        launched = mgr.refresh_verification()
        assert launched == list(VERIFICATION_FIGURES)

    def test_refresh_one_valid(self, manager):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        assert len(FakePopen.instances) == 1
        assert any("workers.optics" in part for part in FakePopen.instances[0].cmd)

    def test_refresh_one_unknown_raises(self, manager):
        mgr, _, _ = manager
        with pytest.raises(ValueError, match="Unknown figure"):
            mgr.refresh_one("bogus")


class TestLaunchWorker:
    def test_cancels_running_worker_before_relaunch(self, manager):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        first = FakePopen.instances[0]
        first._poll_result = None  # still running
        mgr.refresh_one("optics")
        assert first.terminated is True
        assert len(FakePopen.instances) == 2

    def test_completed_worker_not_terminated(self, manager):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        first = FakePopen.instances[0]
        first._poll_result = 0  # already finished
        mgr.refresh_one("optics")
        assert first.terminated is False

    def test_popen_failure_marks_error(self, manager, monkeypatch):
        mgr, state, broadcaster = manager

        def boom(*args, **kwargs):
            raise OSError("no exec")

        monkeypatch.setattr(compute_mod.subprocess, "Popen", boom)
        mgr.refresh_one("optics")

        s = state.load()
        assert s["figures"]["optics"]["status"] == "error"
        assert any(e.get("type") == "figure_error" for e in broadcaster.events)


class TestMonitorWorker:
    """Directly drive _monitor_worker for each terminal outcome."""

    def test_success_marks_ready(self, manager):
        mgr, state, broadcaster = manager
        output = state.figures_dir / "optics.json"
        output.write_text("{}")
        proc = FakePopen(["x"])
        proc.returncode = 0

        mgr._monitor_worker("optics", proc, output)

        assert state.load()["figures"]["optics"]["status"] == "ready"
        assert any(e.get("type") == "figure_ready" for e in broadcaster.events)

    def test_nonzero_exit_marks_error(self, manager):
        mgr, state, broadcaster = manager
        output = state.figures_dir / "optics.json"
        proc = FakePopen(["x"])
        proc.returncode = 1

        def communicate(timeout=None):
            return (b"", b"boom traceback")

        proc.communicate = communicate
        mgr._monitor_worker("optics", proc, output)

        fig = state.load()["figures"]["optics"]
        assert fig["status"] == "error"
        assert "code 1" in fig["error"]

    def test_missing_output_marks_error(self, manager):
        mgr, state, _ = manager
        output = state.figures_dir / "optics.json"  # never created
        proc = FakePopen(["x"])
        proc.returncode = 0

        mgr._monitor_worker("optics", proc, output)

        fig = state.load()["figures"]["optics"]
        assert fig["status"] == "error"
        assert "no output" in fig["error"].lower()

    def test_timeout_marks_error_and_kills(self, manager):
        mgr, state, _ = manager
        output = state.figures_dir / "optics.json"
        proc = FakePopen(["x"])

        calls = {"n": 0}

        def communicate(timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return (b"", b"")

        proc.communicate = communicate
        mgr._monitor_worker("optics", proc, output)

        assert proc.killed is True
        fig = state.load()["figures"]["optics"]
        assert fig["status"] == "error"
        assert "timed out" in fig["error"].lower()


class TestCancelAll:
    def test_terminates_running_processes(self, manager):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        mgr.refresh_one("da")
        for p in FakePopen.instances:
            p._poll_result = None  # all running
        mgr.cancel_all()
        assert all(p.terminated for p in FakePopen.instances)

    def test_leaves_finished_processes_alone(self, manager):
        mgr, _, _ = manager
        mgr.refresh_one("optics")
        FakePopen.instances[0]._poll_result = 0  # finished
        mgr.cancel_all()
        assert FakePopen.instances[0].terminated is False


def test_no_worker_threads_leak(manager):
    """The manager under test must not spawn real monitor threads."""
    mgr, _, _ = manager
    before = threading.active_count()
    mgr.refresh_fast()
    assert threading.active_count() == before
