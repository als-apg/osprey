"""Unit tests for the injected runner seam (no bluesky dependency)."""

from __future__ import annotations

from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner, PlanRunner


def test_fake_scanner_satisfies_scanner_protocol() -> None:
    assert isinstance(FakePlanRunner(), PlanRunner)


def test_initial_state_is_idle_and_inactive() -> None:
    runner = FakePlanRunner()
    assert runner.current_state == "idle"
    assert runner.last_run_uid is None
    assert runner.is_run_active() is False
    assert runner.estimate_current_completion() == 0.0


def test_reinitialize_succeeds_by_default() -> None:
    runner = FakePlanRunner()
    assert runner.reinitialize(exec_config={}) is True
    assert runner.current_state == "armed"
    assert runner.reinitialize_calls == 1


def test_reinitialize_can_be_made_to_fail() -> None:
    runner = FakePlanRunner(reinitialize_fails=True)
    assert runner.reinitialize(exec_config={}) is False
    assert runner.current_state == "error"


def test_start_run_thread_activates_and_mints_run_uid() -> None:
    runner = FakePlanRunner()
    runner.reinitialize(exec_config={})
    runner.start_run_thread()
    assert runner.is_run_active() is True
    assert runner.current_state == "running"
    assert runner.start_calls == 1
    assert runner.last_run_uid is not None


def test_start_run_thread_preserves_seeded_run_uid() -> None:
    runner = FakePlanRunner(run_uid="seeded-uid")
    runner.start_run_thread()
    assert runner.last_run_uid == "seeded-uid"


def test_stop_run_thread_deactivates() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.stop_run_thread()
    assert runner.is_run_active() is False
    assert runner.current_state == "stopped"
    assert runner.stop_calls == 1


def test_stop_run_thread_is_safe_when_not_active() -> None:
    runner = FakePlanRunner()
    runner.stop_run_thread()  # must not raise
    assert runner.is_run_active() is False


def test_simulate_progress_updates_completion_without_ending_scan() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_progress(0.42)
    assert runner.estimate_current_completion() == 0.42
    assert runner.is_run_active() is True


def test_simulate_progress_clamps_to_unit_interval() -> None:
    runner = FakePlanRunner()
    runner.simulate_progress(-1.0)
    assert runner.estimate_current_completion() == 0.0
    runner.simulate_progress(5.0)
    assert runner.estimate_current_completion() == 1.0


def test_simulate_completion_ends_scan_successfully() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_completion()
    assert runner.is_run_active() is False
    assert runner.current_state == "completed"
    assert runner.estimate_current_completion() == 1.0


def test_simulate_error_ends_scan_in_error_state() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_error("device timeout")
    assert runner.is_run_active() is False
    assert runner.current_state == "error"
    assert runner.error_message == "device timeout"
