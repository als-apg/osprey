"""Unit tests for the bridge's in-memory run registry and lifecycle state machine."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge.runs import Run, RunRegistry
from osprey.services.bluesky_bridge.scanner import FakeScanner

# =========================================================================
# Run dataclass: fields/defaults
# =========================================================================


def test_run_defaults() -> None:
    run = Run(id="abc123", request={"devices": []})
    assert run.id == "abc123"
    assert run.request == {"devices": []}
    assert isinstance(run.created_at, float)
    assert run.promoted is False
    assert run.promoting is False
    assert run.scanner is None
    assert run.stopped is False
    assert run.error is None
    assert run.launched_by is None


def test_run_uid_is_none_before_a_scanner_is_attached() -> None:
    run = Run(id="abc123", request={})
    assert run.run_uid is None


def test_run_uid_reflects_the_attached_scanner() -> None:
    scanner = FakeScanner(run_uid="seeded-uid")
    run = Run(id="abc123", request={}, scanner=scanner)
    assert run.run_uid == "seeded-uid"


# =========================================================================
# status derivation
# =========================================================================


def test_status_is_intent_before_promotion() -> None:
    run = Run(id="abc123", request={})
    assert run.status == "intent"


def test_status_is_running_while_scanner_is_active() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    run = Run(id="abc123", request={}, promoted=True, scanner=scanner)
    assert run.status == "running"


def test_status_is_completed_once_scan_finishes_cleanly() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_completion()
    run = Run(id="abc123", request={}, promoted=True, scanner=scanner)
    assert run.status == "completed"


def test_status_is_stopped_when_the_run_was_stopped() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.stop_scanning_thread()
    run = Run(id="abc123", request={}, promoted=True, scanner=scanner, stopped=True)
    assert run.status == "stopped"


def test_status_is_error_when_run_error_is_set() -> None:
    run = Run(id="abc123", request={}, promoted=True, error="promotion failed: boom")
    assert run.status == "error"


def test_status_is_error_when_scanner_ends_in_an_error_state() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_error("device timeout")
    run = Run(id="abc123", request={}, promoted=True, scanner=scanner)
    assert run.status == "error"


def test_run_error_takes_precedence_even_before_promotion() -> None:
    run = Run(id="abc123", request={}, error="setup failed")
    assert run.status == "error"


# =========================================================================
# to_dict shape
# =========================================================================


def test_to_dict_intent_is_minimal() -> None:
    run = Run(id="abc123", request={})
    out = run.to_dict()
    assert out == {"id": "abc123", "status": "intent"}


def test_to_dict_includes_completion_and_run_uid_once_promoted() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_progress(0.5)
    run = Run(id="abc123", request={}, promoted=True, scanner=scanner, launched_by="agent")
    out = run.to_dict()
    assert out["id"] == "abc123"
    assert out["status"] == "running"
    assert out["completion"] == 0.5
    assert out["launched_by"] == "agent"
    assert out["run_uid"] == scanner.last_run_uid
    assert "error" not in out


def test_to_dict_omits_launched_by_when_unset() -> None:
    run = Run(id="abc123", request={})
    assert "launched_by" not in run.to_dict()


def test_to_dict_surfaces_explicit_error() -> None:
    run = Run(id="abc123", request={}, promoted=True, error="promotion failed: boom")
    out = run.to_dict()
    assert out["status"] == "error"
    assert out["error"] == "promotion failed: boom"


def test_to_dict_synthesizes_error_message_from_scanner_state() -> None:
    scanner = FakeScanner()
    scanner.start_scan_thread()
    scanner.simulate_error("device timeout")
    run = Run(id="abc123", request={}, promoted=True, scanner=scanner)
    out = run.to_dict()
    assert out["status"] == "error"
    assert "error" in out


# =========================================================================
# RunRegistry: add/get, unknown-run semantics
# =========================================================================


def test_registry_add_returns_a_run_with_a_generated_id() -> None:
    registry = RunRegistry()
    run = registry.add(request={"devices": ["motor1"]}, launched_by="agent")
    assert run.id
    assert run.request == {"devices": ["motor1"]}
    assert run.launched_by == "agent"


def test_registry_get_returns_the_added_run() -> None:
    registry = RunRegistry()
    added = registry.add(request={})
    fetched = registry.get(added.id)
    assert fetched is added


def test_registry_get_raises_404_for_missing_id() -> None:
    registry = RunRegistry()
    with pytest.raises(HTTPException) as excinfo:
        registry.get("does-not-exist")
    assert excinfo.value.status_code == 404


def test_registry_get_is_honest_after_a_simulated_restart() -> None:
    # A fresh registry has no memory of runs an earlier process created —
    # this is the "post-restart" semantics the registry must be honest about.
    old_registry = RunRegistry()
    run = old_registry.add(request={})

    new_registry = RunRegistry()
    with pytest.raises(HTTPException) as excinfo:
        new_registry.get(run.id)
    assert excinfo.value.status_code == 404


def test_registry_list_returns_newest_first_and_respects_limit() -> None:
    registry = RunRegistry()
    first = registry.add(request={})
    second = registry.add(request={})
    third = registry.add(request={})

    all_runs = registry.list(limit=100)
    assert [r.id for r in all_runs] == [third.id, second.id, first.id]

    limited = registry.list(limit=2)
    assert len(limited) == 2
