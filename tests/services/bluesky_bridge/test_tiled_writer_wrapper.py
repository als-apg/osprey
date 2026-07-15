"""Tests for `_FaultIsolatedTiledWriter` (task 2.1) and `BlueskyPlanRunner.tiled_degraded` (task 2.2).

`TiledWriter` is a synchronous RunEngine callback, and bluesky's `RunEngine`
does not swallow callback exceptions by default — an exception escaping a
subscribed callback aborts the running plan. This wrapper exists so a Tiled
outage degrades persistence instead of killing a scan: it must never let an
inner-writer exception propagate, and once degraded it must stop calling the
inner writer at all.

`BlueskyPlanRunner.tiled_degraded` surfaces that latch (or a construction-time
failure) at the runner level, distinguishing "Tiled disabled" (`False`, no
factory supplied) from "Tiled wired but failing" (`True`).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

# `plan_runner_bluesky` imports `bluesky.RunEngine` at module scope, so this module
# cannot even be collected without the `bluesky-bridge` extra. Guard rather than
# error, matching the sibling bridge tests. CI's unit job installs the extra
# (pinned by tests/deployment/test_ci_workflow_wiring.py), so these guards run
# there rather than skipping.
pytest.importorskip("bluesky")

from osprey.services.bluesky_bridge.plan_runner_bluesky import (  # noqa: E402
    BlueskyPlanRunner,
    _FaultIsolatedTiledWriter,
)


class _RecordingWriter:
    """A `(name, doc)` callback that records calls and optionally raises."""

    def __init__(self, raise_on: str | None = None) -> None:
        self.raise_on = raise_on
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, name: str, doc: dict) -> None:
        self.calls.append((name, doc))
        if name == self.raise_on:
            raise RuntimeError("Tiled write failed")


def test_healthy_inner_writer_forwards_every_document_and_stays_not_degraded():
    inner = _RecordingWriter()
    wrapper = _FaultIsolatedTiledWriter(inner)

    assert wrapper.degraded is False

    wrapper("start", {"uid": "run-1"})
    wrapper("descriptor", {"uid": "desc-1"})
    wrapper("event", {"uid": "event-1"})
    wrapper("stop", {"uid": "stop-1"})

    assert inner.calls == [
        ("start", {"uid": "run-1"}),
        ("descriptor", {"uid": "desc-1"}),
        ("event", {"uid": "event-1"}),
        ("stop", {"uid": "stop-1"}),
    ]
    assert wrapper.degraded is False


def test_inner_exception_is_swallowed_and_latches_degraded():
    inner = _RecordingWriter(raise_on="event")
    wrapper = _FaultIsolatedTiledWriter(inner)

    wrapper("start", {"uid": "run-1"})
    assert wrapper.degraded is False

    # Must not raise: an escaping exception would abort the RunEngine's plan.
    wrapper("event", {"uid": "event-1"})

    assert wrapper.degraded is True


def test_subsequent_documents_short_circuit_after_latch():
    inner = _RecordingWriter(raise_on="event")
    wrapper = _FaultIsolatedTiledWriter(inner)

    wrapper("start", {"uid": "run-1"})
    wrapper("event", {"uid": "event-1"})  # raises inside inner, latches degraded
    assert wrapper.degraded is True
    assert len(inner.calls) == 2

    # Further documents must not reach the inner writer at all.
    wrapper("event", {"uid": "event-2"})
    wrapper("stop", {"uid": "stop-1"})

    assert len(inner.calls) == 2
    assert wrapper.degraded is True


def test_exception_logged_once_with_exc_info(caplog: pytest.LogCaptureFixture):
    inner = _RecordingWriter(raise_on="event")
    wrapper = _FaultIsolatedTiledWriter(inner)

    with caplog.at_level(
        logging.ERROR, logger="osprey.services.bluesky_bridge.plan_runner_bluesky"
    ):
        wrapper("start", {"uid": "run-1"})
        wrapper("event", {"uid": "event-1"})
        # Further calls short-circuit, so no additional log records.
        wrapper("event", {"uid": "event-2"})
        wrapper("stop", {"uid": "stop-1"})

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert error_records[0].exc_info is not None


# =========================================================================
# `BlueskyPlanRunner.tiled_degraded` (task 2.2)
# =========================================================================


def test_tiled_degraded_is_false_when_no_writer_is_wired():
    """Tiled disabled entirely (no `tiled_writer_factory`) must never read degraded."""
    runner = BlueskyPlanRunner(devices={})

    assert runner.tiled_degraded is False


def test_tiled_degraded_is_false_for_a_healthy_wired_writer():
    inner = _RecordingWriter()
    runner = BlueskyPlanRunner(devices={}, tiled_writer_factory=lambda: inner)

    assert runner.tiled_degraded is False


def test_tiled_degraded_is_true_when_the_factory_raises_at_construction(
    caplog: pytest.LogCaptureFixture,
):
    """A Tiled server down at promote time must leave the runner degraded,
    not silently continue with `tiled_degraded=False`.
    """

    def _boom() -> None:
        raise RuntimeError("Tiled unreachable")

    with caplog.at_level(
        logging.WARNING, logger="osprey.services.bluesky_bridge.plan_runner_bluesky"
    ):
        runner = BlueskyPlanRunner(devices={}, tiled_writer_factory=_boom)

    assert runner.tiled_degraded is True


def test_tiled_degraded_becomes_true_after_a_document_raises():
    inner = _RecordingWriter(raise_on="event")
    runner = BlueskyPlanRunner(devices={}, tiled_writer_factory=lambda: inner)
    assert runner.tiled_degraded is False

    # White-box: exercise the same wrapper `RE.subscribe` was given, without
    # driving a real RunEngine plan through this unit test.
    runner._tiled_writer("start", {"uid": "run-1"})
    assert runner.tiled_degraded is False
    runner._tiled_writer("event", {"uid": "event-1"})

    assert runner.tiled_degraded is True


def test_tiled_degraded_is_true_when_subscribe_raises_after_successful_construction(
    monkeypatch: pytest.MonkeyPatch,
):
    """A raising `RE.subscribe(...)` must degrade, not propagate.

    Pre-2.2-fix, `RE.subscribe(self._tiled_writer)` lived in the `else:`
    clause of the construction `try` — outside the exception guard — so a
    raising `subscribe()` would escape `BlueskyPlanRunner.__init__` entirely,
    turning a Tiled outage into a failed `do_promote` (FR4 violation). The
    inner writer factory here succeeds; only `RE.subscribe` is made to fail,
    and only on its *second* call (the first, inside `__init__`, wires
    `_on_document` and must keep working) — so this exercises exactly the
    call the earlier fix left unguarded.
    """
    from bluesky import RunEngine

    original_subscribe = RunEngine.subscribe
    calls = {"n": 0}

    def _flaky_subscribe(self: RunEngine, func: Any, name: str = "all") -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("RE.subscribe failed")
        return original_subscribe(self, func, name=name)

    monkeypatch.setattr(RunEngine, "subscribe", _flaky_subscribe)

    inner = _RecordingWriter()
    # Must not raise: a Tiled outage degrades the runner, never the promote.
    runner = BlueskyPlanRunner(devices={}, tiled_writer_factory=lambda: inner)

    assert runner.tiled_degraded is True
    # No partially-wired writer left behind: even if `subscribe` had managed
    # to register the wrapper before raising, the wrapper is already latched
    # degraded, so it would still no-op on any document it receives.
    assert runner._tiled_writer is not None
    runner._tiled_writer("start", {"uid": "run-1"})
    assert inner.calls == []  # never reached — latched before any document arrived
