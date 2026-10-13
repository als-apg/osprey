"""Tests for the per-process control-system target scope."""

import multiprocessing

import pytest

from osprey.services.optimizer.models import process_scope


def _report_scope(queue: "multiprocessing.Queue[bool]") -> None:
    queue.put(process_scope.multiple_targets_allowed())


def test_a_process_serves_one_target_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_scope, "_multiple_targets_allowed", False)
    assert process_scope.multiple_targets_allowed() is False
    process_scope.allow_multiple_targets()
    assert process_scope.multiple_targets_allowed() is True


def test_a_spawned_process_does_not_inherit_the_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Job workers are spawned from the API, which opts out: they must still be bound."""
    monkeypatch.setattr(process_scope, "_multiple_targets_allowed", True)
    ctx = multiprocessing.get_context("spawn")  # as the API spawns job workers
    queue = ctx.Queue()
    child = ctx.Process(target=_report_scope, args=(queue,))
    child.start()
    try:
        assert queue.get(timeout=30) is False
    finally:
        child.join(timeout=10)
