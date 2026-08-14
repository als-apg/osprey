"""The plan stamp riding along with a run's live rows.

A reader that holds only a run id cannot tell what the run's columns mean, so
the recorder carries the plan that produced them next to the rows themselves.
Two properties are load-bearing and tested here:

- **`live_rows` never interprets the stamp.** It is stored as handed over and
  returned as stored — no validation, no defaulting, no shape assumptions. A
  later reader (the figure route) is what knows how to read it; teaching this
  module the shape would put a second, drifting copy of that knowledge here.
- **The extraction lives in the document plane**, beside the run-id read and
  for the same reason: the stamp is a start-document detail. That is what
  keeps `live_rows` testable with synthetic plain dicts, so the first half of
  this file drives the recorder directly with no document plane at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.document_plane import RunDocumentRouter
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder
from osprey.services.bluesky_bridge.queue_backend import PLAN_META_KEY

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_buffers():
    live_rows._clear()
    yield
    live_rows._clear()


def _start_doc(uid: str = "run-1", **extra: Any) -> dict[str, Any]:
    return {"uid": uid, "time": 1.0, "scan_id": 1, **extra}


def _event_doc(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data}


def _stop_doc(uid: str = "run-1") -> dict[str, Any]:
    return {"run_start": uid}


_STAMP = {"name": "grid_scan", "kwargs": {"num_points": 5}}


# =========================================================================
# The recorder: an opaque value in, the same value out
# =========================================================================


def test_snapshot_carries_the_plan_the_recorder_was_given() -> None:
    recorder = LiveRowRecorder(plan=_STAMP)
    recorder("start", _start_doc())

    assert live_rows.get("run-1")["plan"] == _STAMP


def test_plan_is_none_when_the_recorder_was_given_none() -> None:
    """A missing plan is a real answer, not a missing key the reader must guess at."""
    recorder = LiveRowRecorder()
    recorder("start", _start_doc())

    snapshot = live_rows.get("run-1")
    assert "plan" in snapshot
    assert snapshot["plan"] is None


def test_plan_survives_events_and_the_stop_document() -> None:
    """It is a property of the run, not of the moment the buffer was opened."""
    recorder = LiveRowRecorder(plan=_STAMP)
    recorder("start", _start_doc())
    recorder("event", _event_doc({"x": 1.0}))
    recorder("stop", _stop_doc())

    snapshot = live_rows.get("run-1")
    assert snapshot["plan"] == _STAMP
    assert snapshot["partial"] is False
    assert snapshot["rows"] == [[1.0]]


def test_the_plan_value_is_never_interpreted() -> None:
    """A stamp of an unexpected shape is stored verbatim, not rejected or coerced.

    The recorder has no opinion about what a plan looks like — validating here
    would duplicate the reader's knowledge of the shape and start drifting from
    it the moment the stamp changes.
    """
    recorder = LiveRowRecorder(plan="not-a-mapping")
    recorder("start", _start_doc())

    assert live_rows.get("run-1")["plan"] == "not-a-mapping"


def test_each_run_keeps_its_own_plan() -> None:
    LiveRowRecorder(plan=_STAMP)("start", _start_doc("run-1"))
    LiveRowRecorder(plan={"name": "orm", "kwargs": {}})("start", _start_doc("run-2"))

    assert live_rows.get("run-1")["plan"] == _STAMP
    assert live_rows.get("run-2")["plan"] == {"name": "orm", "kwargs": {}}


def test_a_restarted_buffer_takes_the_new_recorders_plan() -> None:
    """A second start under the same key opens a fresh buffer, stamp included."""
    LiveRowRecorder(plan=_STAMP)("start", _start_doc("run-1"))
    LiveRowRecorder(plan={"name": "orm", "kwargs": {}})("start", _start_doc("run-1"))

    assert live_rows.get("run-1")["plan"] == {"name": "orm", "kwargs": {}}


def test_plan_is_recorded_under_the_explicit_key() -> None:
    """The stamp follows the buffer key, not the run uid it was recorded from."""
    recorder = LiveRowRecorder(key="osprey-run-7", plan=_STAMP)
    recorder("start", _start_doc("engine-uid"))

    assert live_rows.get("engine-uid") is None
    assert live_rows.get("osprey-run-7")["plan"] == _STAMP


# =========================================================================
# The document plane: where the stamp is read out of the start document
# =========================================================================


def test_router_records_the_plan_stamped_on_the_start_document() -> None:
    router = RunDocumentRouter()
    router(
        "start",
        _start_doc("engine-uid", osprey_run_id="osprey-run-1", **{PLAN_META_KEY: _STAMP}),
    )

    assert live_rows.get("osprey-run-1")["plan"] == _STAMP


def test_router_records_no_plan_when_the_start_document_carries_no_stamp() -> None:
    """Out-of-band runs (the worker driven directly) still record — without a plan."""
    router = RunDocumentRouter()
    router("start", _start_doc("engine-uid"))

    snapshot = live_rows.get("engine-uid")
    assert snapshot is not None
    assert snapshot["plan"] is None


def test_router_keeps_the_stamps_of_interleaved_runs_apart() -> None:
    """One stream carries every run the worker executes; the stamps must not cross."""
    router = RunDocumentRouter()
    orm = {"name": "orm", "kwargs": {"correctors": ["c1"]}}
    router("start", _start_doc("uid-a", osprey_run_id="run-a", **{PLAN_META_KEY: _STAMP}))
    router("start", _start_doc("uid-b", osprey_run_id="run-b", **{PLAN_META_KEY: orm}))

    assert live_rows.get("run-a")["plan"] == _STAMP
    assert live_rows.get("run-b")["plan"] == orm
