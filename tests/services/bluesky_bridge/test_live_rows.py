"""Unit tests for the bounded live-row buffer (`live_rows.py`).

Feeds synthetic bluesky-shaped documents (plain dicts) directly into
`LiveRowRecorder` — no bluesky import anywhere in this file or the module
under test.
"""

from __future__ import annotations

import pytest

from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder


@pytest.fixture(autouse=True)
def _isolated_buffers():
    """Every test gets an empty module-level buffer store."""
    live_rows._clear()
    yield
    live_rows._clear()


def _start_doc(uid: str = "run-1") -> dict:
    return {"uid": uid}


def _event_doc(data: dict) -> dict:
    return {"data": data}


def _stop_doc(uid: str = "run-1") -> dict:
    return {"run_start": uid}


# =========================================================================
# Basic recording: start -> event(s) -> stop
# =========================================================================


def test_unknown_run_uid_returns_none() -> None:
    assert live_rows.get("does-not-exist") is None


def test_start_doc_creates_an_empty_partial_buffer() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))

    buf = live_rows.get("run-1")
    assert buf is not None
    assert buf["columns"] == []
    assert buf["rows"] == []
    assert buf["partial"] is True
    assert buf["total_seen"] == 0


def test_event_docs_append_rows_in_column_order() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("event", _event_doc({"x": 1.0, "y": 2.0}))
    recorder("event", _event_doc({"x": 3.0, "y": 4.0}))

    buf = live_rows.get("run-1")
    assert buf["columns"] == ["x", "y"]
    assert buf["rows"] == [[1.0, 2.0], [3.0, 4.0]]
    assert buf["total_seen"] == 2
    assert buf["partial"] is True


def test_stop_doc_flips_partial_to_false() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("event", _event_doc({"x": 1.0}))
    recorder("stop", _stop_doc("run-1"))

    buf = live_rows.get("run-1")
    assert buf["partial"] is False
    assert buf["rows"] == [[1.0]]


def test_completed_run_stays_readable_after_stop() -> None:
    """RETAINS completed runs — the whole point of this module vs. BELLA's demo-day eviction."""
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("event", _event_doc({"x": 1.0}))
    recorder("stop", _stop_doc("run-1"))

    # No new run has started; the completed run's buffer must still be there.
    buf = live_rows.get("run-1")
    assert buf is not None
    assert buf["partial"] is False
    assert buf["total_seen"] == 1


# =========================================================================
# Column discovery: incremental, first-seen order, backfill of earlier rows
# =========================================================================


def test_new_column_appearing_mid_run_backfills_earlier_rows_with_none() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("event", _event_doc({"x": 1.0}))
    recorder("event", _event_doc({"x": 2.0, "y": 5.0}))

    buf = live_rows.get("run-1")
    assert buf["columns"] == ["x", "y"]
    assert buf["rows"] == [[1.0, None], [2.0, 5.0]]


def test_event_missing_an_already_known_column_records_none() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("event", _event_doc({"x": 1.0, "y": 2.0}))
    recorder("event", _event_doc({"x": 3.0}))

    buf = live_rows.get("run-1")
    assert buf["rows"] == [[1.0, 2.0], [3.0, None]]


# =========================================================================
# Per-run row cap: total_seen keeps counting past the storage cap
# =========================================================================


def test_row_storage_cap_stops_appending_but_total_seen_keeps_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_rows, "_MAX_ROWS_PER_RUN", 3)

    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    for i in range(5):
        recorder("event", _event_doc({"x": float(i)}))

    buf = live_rows.get("run-1")
    assert len(buf["rows"]) == 3
    assert buf["total_seen"] == 5


# =========================================================================
# Run buffer retention: LRU eviction past _MAX_RUNS
# =========================================================================


def test_oldest_run_buffer_evicted_once_max_runs_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_rows, "_MAX_RUNS", 2)

    for uid in ("run-1", "run-2", "run-3"):
        recorder = LiveRowRecorder()
        recorder("start", _start_doc(uid))

    assert live_rows.get("run-1") is None
    assert live_rows.get("run-2") is not None
    assert live_rows.get("run-3") is not None


def test_writing_to_a_run_moves_it_to_the_front_of_the_eviction_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_rows, "_MAX_RUNS", 2)

    first = LiveRowRecorder()
    first("start", _start_doc("run-1"))
    second = LiveRowRecorder()
    second("start", _start_doc("run-2"))

    # Touch run-1 again so it is no longer the least-recently-used entry.
    first("event", _event_doc({"x": 1.0}))

    # A brand new run-3 should now evict run-2 (least recently touched), not run-1.
    third = LiveRowRecorder()
    third("start", _start_doc("run-3"))

    assert live_rows.get("run-1") is not None
    assert live_rows.get("run-2") is None
    assert live_rows.get("run-3") is not None


# =========================================================================
# Recorder never raises: a bad document must not propagate
# =========================================================================


def test_malformed_event_doc_does_not_raise() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    # `data` is not iterable — would normally break `for key in data`, but the
    # recorder is fully exception-wrapped.
    recorder("event", {"data": 42})

    buf = live_rows.get("run-1")
    assert buf["total_seen"] == 0  # exception raised before total_seen incremented


def test_event_before_any_start_is_ignored_without_raising() -> None:
    recorder = LiveRowRecorder()
    recorder("event", _event_doc({"x": 1.0}))  # no prior start()
    assert live_rows.get("run-1") is None


def test_stop_before_any_start_is_ignored_without_raising() -> None:
    recorder = LiveRowRecorder()
    recorder("stop", _stop_doc("run-1"))  # no prior start()
    assert live_rows.get("run-1") is None


def test_unknown_document_name_is_ignored() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("descriptor", {"data_keys": {"x": {}}})  # not handled, must not raise

    buf = live_rows.get("run-1")
    assert buf["rows"] == []


# =========================================================================
# get() returns a snapshot copy, not a live reference
# =========================================================================


def test_get_returns_a_copy_not_a_live_reference() -> None:
    recorder = LiveRowRecorder()
    recorder("start", _start_doc("run-1"))
    recorder("event", _event_doc({"x": 1.0}))

    snapshot = live_rows.get("run-1")
    snapshot["rows"].append([999.0])
    snapshot["columns"].append("bogus")

    buf = live_rows.get("run-1")
    assert buf["rows"] == [[1.0]]
    assert buf["columns"] == ["x"]
