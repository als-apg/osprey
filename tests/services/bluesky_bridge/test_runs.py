"""Unit tests for the bridge's in-memory run registry and lifecycle state machine."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import Run, RunRegistry

# =========================================================================
# Run dataclass: fields/defaults
# =========================================================================


def test_run_defaults() -> None:
    run = Run(id="abc123", request={"devices": []})
    assert run.id == "abc123"
    assert run.request == {"devices": []}
    assert isinstance(run.created_at, float)
    assert run.launched is False
    assert run.launching is False
    assert run.runner is None
    assert run.stopped is False
    assert run.error is None
    assert run.launched_by is None


def test_run_uid_is_none_before_a_scanner_is_attached() -> None:
    run = Run(id="abc123", request={})
    assert run.run_uid is None


def test_run_uid_reflects_the_attached_scanner() -> None:
    runner = FakePlanRunner(run_uid="seeded-uid")
    run = Run(id="abc123", request={}, runner=runner)
    assert run.run_uid == "seeded-uid"


# =========================================================================
# status derivation
# =========================================================================


def test_status_is_intent_before_launch() -> None:
    run = Run(id="abc123", request={})
    assert run.status == "pending"


def test_status_is_running_while_scanner_is_active() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    run = Run(id="abc123", request={}, launched=True, runner=runner)
    assert run.status == "running"


def test_status_is_completed_once_scan_finishes_cleanly() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_completion()
    run = Run(id="abc123", request={}, launched=True, runner=runner)
    assert run.status == "completed"


def test_status_is_stopped_when_the_run_was_stopped() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.stop_run_thread()
    run = Run(id="abc123", request={}, launched=True, runner=runner, stopped=True)
    assert run.status == "stopped"


def test_status_is_error_when_run_error_is_set() -> None:
    run = Run(id="abc123", request={}, launched=True, error="launch failed: boom")
    assert run.status == "error"


def test_status_is_error_when_scanner_ends_in_an_error_state() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_error("device timeout")
    run = Run(id="abc123", request={}, launched=True, runner=runner)
    assert run.status == "error"


def test_run_error_takes_precedence_even_before_launch() -> None:
    run = Run(id="abc123", request={}, error="setup failed")
    assert run.status == "error"


def test_status_is_error_based_on_error_message_not_a_current_state_heuristic() -> None:
    """MANDATORY handoff fix (task 2.7): the terminal-error signal is the
    PlanRunner protocol's explicit `error_message`, not a `current_state` string
    match — a runner can report a totally custom terminal `current_state` and
    still be recognized as errored as long as `error_message` is set.
    """
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner._active = False
    runner.current_state = "some_custom_terminal_state"  # not "error"
    runner.error_message = "device timeout"

    run = Run(id="abc123", request={}, launched=True, runner=runner)
    assert run.status == "error"


def test_status_is_not_error_when_current_state_looks_like_error_but_error_message_is_none() -> (
    None
):
    """The old heuristic matched `current_state == "error"` directly; the new
    signal requires `error_message` to be set — proves the heuristic is gone.
    """
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner._active = False
    runner.current_state = "error"  # would have matched the old heuristic
    runner.error_message = None

    run = Run(id="abc123", request={}, launched=True, runner=runner)
    assert run.status == "completed"


# =========================================================================
# to_dict shape
# =========================================================================


def test_to_dict_intent_is_minimal() -> None:
    run = Run(id="abc123", request={})
    out = run.to_dict()
    assert out == {"id": "abc123", "status": "pending", "tiled_degraded": False}


def test_to_dict_includes_completion_and_run_uid_once_launched() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_progress(0.5)
    run = Run(id="abc123", request={}, launched=True, runner=runner, launched_by="agent")
    out = run.to_dict()
    assert out["id"] == "abc123"
    assert out["status"] == "running"
    assert out["completion"] == 0.5
    assert out["launched_by"] == "agent"
    assert out["run_uid"] == runner.last_run_uid
    assert "error" not in out


def test_to_dict_omits_launched_by_when_unset() -> None:
    run = Run(id="abc123", request={})
    assert "launched_by" not in run.to_dict()


def test_to_dict_surfaces_explicit_error() -> None:
    run = Run(id="abc123", request={}, launched=True, error="launch failed: boom")
    out = run.to_dict()
    assert out["status"] == "error"
    assert out["error"] == "launch failed: boom"


def test_to_dict_synthesizes_error_message_from_scanner_state() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_error("device timeout")
    run = Run(id="abc123", request={}, launched=True, runner=runner)
    out = run.to_dict()
    assert out["status"] == "error"
    assert "error" in out


def test_to_dict_uses_the_scanner_error_message_directly() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner._active = False
    runner.error_message = "device timeout"
    run = Run(id="abc123", request={}, launched=True, runner=runner)
    out = run.to_dict()
    assert out["error"] == "device timeout"


# =========================================================================
# to_dict: tiled_degraded (task 2.4)
# =========================================================================


def test_to_dict_tiled_degraded_is_false_for_a_launched_healthy_writer() -> None:
    """A runner that exposes `tiled_degraded = False` (a healthy wired Tiled
    writer) must surface `False` on the run, with the key present.
    """
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.tiled_degraded = False  # duck-typed, like BlueskyPlanRunner's property
    run = Run(id="abc123", request={}, launched=True, runner=runner)

    out = run.to_dict()

    assert "tiled_degraded" in out
    assert out["tiled_degraded"] is False


def test_to_dict_tiled_degraded_is_true_for_a_launched_degraded_writer() -> None:
    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.tiled_degraded = True
    run = Run(id="abc123", request={}, launched=True, runner=runner)

    out = run.to_dict()

    assert out["tiled_degraded"] is True


def test_to_dict_tiled_degraded_is_false_and_present_for_an_unlaunched_run() -> None:
    """No runner at all (never launched) must read `False`, key present —
    never absent, and never `True` merely because nothing has run yet.
    """
    run = Run(id="abc123", request={})

    out = run.to_dict()

    assert "tiled_degraded" in out
    assert out["tiled_degraded"] is False


def test_to_dict_tiled_degraded_is_false_and_present_for_a_scanner_without_the_attribute() -> None:
    """`FakePlanRunner` (and any `PlanRunner` that never wires Tiled at all) has no
    `tiled_degraded` attribute — must default to `False`, key still present,
    never raise `AttributeError`.
    """
    runner = FakePlanRunner()
    runner.start_run_thread()
    assert not hasattr(runner, "tiled_degraded")  # sanity: FakePlanRunner truly lacks it
    run = Run(id="abc123", request={}, launched=True, runner=runner)

    out = run.to_dict()

    assert "tiled_degraded" in out
    assert out["tiled_degraded"] is False


def test_to_dict_key_set_is_unchanged_apart_from_tiled_degraded() -> None:
    """Existing keys and their meanings must not shift — this is an additive
    HTTP-contract change consumed by the MCP tools. Pin the full key set
    for a minimal intent, a fully-populated healthy launched run, and a
    failed-launch run, so an accidental rename or drop (e.g. `error` ->
    `err`) is caught here rather than downstream — the two-key-set-only
    version of this test left `error` unpinned by any exhaustive assertion.
    """
    intent_run = Run(id="abc123", request={})
    assert set(intent_run.to_dict().keys()) == {"id", "status", "tiled_degraded"}

    runner = FakePlanRunner()
    runner.start_run_thread()
    runner.simulate_progress(0.5)
    runner.tiled_degraded = False
    full_run = Run(id="abc123", request={}, launched=True, runner=runner, launched_by="agent")
    assert set(full_run.to_dict().keys()) == {
        "id",
        "status",
        "tiled_degraded",
        "completion",
        "launched_by",
        "run_uid",
    }

    # A failed-launch run: `do_launch` (`runs.py`) only publishes
    # `run.runner` on success, so a launch that raised (unknown plan,
    # a Tiled construction failure that somehow escaped the fault-isolated
    # wrapper, thread creation itself failing, etc.) leaves `runner is None`
    # with `error` set — exactly what an operator sees after a failed
    # launch. `tiled_degraded` must still read `False`, present, here: a
    # failed launch is not evidence Tiled specifically was the cause.
    failed_launch_run = Run(id="abc123", request={}, error="launch failed: boom")
    assert set(failed_launch_run.to_dict().keys()) == {"id", "status", "tiled_degraded", "error"}
    assert failed_launch_run.to_dict()["tiled_degraded"] is False


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
