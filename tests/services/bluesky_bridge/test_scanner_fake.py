"""Unit tests for the injected scanner seam (no bluesky dependency)."""

from __future__ import annotations

from osprey.services.bluesky_bridge.scanner import FakeScanner, Scanner


def test_fake_scanner_satisfies_scanner_protocol() -> None:
    assert isinstance(FakeScanner(), Scanner)


def test_initial_state_is_idle_and_inactive() -> None:
    scanner = FakeScanner()
    assert scanner.current_state == "idle"
    assert scanner.last_run_uid is None
    assert scanner.is_scanning_active() is False
    assert scanner.estimate_current_completion() == 0.0


def test_reinitialize_succeeds_by_default() -> None:
    scanner = FakeScanner()
    assert scanner.reinitialize(exec_config={}) is True
    assert scanner.current_state == "armed"
    assert scanner.reinitialize_calls == 1


def test_reinitialize_can_be_made_to_fail() -> None:
    scanner = FakeScanner(reinitialize_fails=True)
    assert scanner.reinitialize(exec_config={}) is False
    assert scanner.current_state == "error"


def test_start_scan_thread_activates_and_mints_run_uid() -> None:
    scanner = FakeScanner()
    scanner.reinitialize(exec_config={})
    scanner.start_scan_thread()
    assert scanner.is_scanning_active() is True
    assert scanner.current_state == "running"
    assert scanner.start_calls == 1
    assert scanner.last_run_uid is not None


def test_start_scan_thread_preserves_seeded_run_uid() -> None:
    scanner = FakeScanner(run_uid="seeded-uid")
    scanner.start_scan_thread()
    assert scanner.last_run_uid == "seeded-uid"


def test_stop_scanning_thread_deactivates() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.stop_scanning_thread()
    assert scanner.is_scanning_active() is False
    assert scanner.current_state == "stopped"
    assert scanner.stop_calls == 1


def test_stop_scanning_thread_is_safe_when_not_active() -> None:
    scanner = FakeScanner()
    scanner.stop_scanning_thread()  # must not raise
    assert scanner.is_scanning_active() is False


def test_simulate_progress_updates_completion_without_ending_scan() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_progress(0.42)
    assert scanner.estimate_current_completion() == 0.42
    assert scanner.is_scanning_active() is True


def test_simulate_progress_clamps_to_unit_interval() -> None:
    scanner = FakeScanner()
    scanner.simulate_progress(-1.0)
    assert scanner.estimate_current_completion() == 0.0
    scanner.simulate_progress(5.0)
    assert scanner.estimate_current_completion() == 1.0


def test_simulate_completion_ends_scan_successfully() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_completion()
    assert scanner.is_scanning_active() is False
    assert scanner.current_state == "completed"
    assert scanner.estimate_current_completion() == 1.0


def test_simulate_error_ends_scan_in_error_state() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_error("device timeout")
    assert scanner.is_scanning_active() is False
    assert scanner.current_state == "error"
    assert scanner.error_message == "device timeout"
