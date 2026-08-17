"""Unit tests for the run-record projection (`runs.py`).

The bridge keeps no run state of its own: a run record is derived from the
queue server's own pending items, running item, and history. These tests feed
the projection literal manager documents — the shape
`bluesky_queueserver_api`'s `queue_get`/`history_get` return — so every status
and ordering decision is pinned without a manager, an app, or a network.
"""

from __future__ import annotations

import pytest

from osprey.services.bluesky_bridge import document_plane, live_rows, runs


@pytest.fixture(autouse=True)
def _isolated_progress_state():
    """Progress is read from the document plane, which is process-global."""
    live_rows._clear()
    document_plane._clear()
    yield
    live_rows._clear()
    document_plane._clear()


def _item(
    run_id: str | None = "run-1",
    *,
    name: str = "grid_scan",
    kwargs: dict | None = None,
    item_uid: str = "uid-1",
    result: dict | None = None,
) -> dict:
    """One queue/history item, shaped as the manager returns it."""
    item: dict = {
        "item_type": "plan",
        "name": name,
        "kwargs": kwargs if kwargs is not None else {"readbacks": ["BPM1"]},
        "item_uid": item_uid,
    }
    if run_id is not None:
        item["meta"] = {"osprey_run_id": run_id}
    if result is not None:
        item["result"] = result
    return item


def _records(running=None, pending=None, history=None, **kwargs) -> list[dict]:
    return runs.list_records(
        running_item=running,
        queue_items=pending,
        history_items=history,
        **kwargs,
    )


# =========================================================================
# Status mapping
# =========================================================================


def test_a_queued_item_is_pending() -> None:
    [record] = _records(pending=[_item()])
    assert record["status"] == "pending"
    assert record["id"] == "run-1"


def test_the_running_item_is_running() -> None:
    [record] = _records(running=_item())
    assert record["status"] == "running"


def test_a_cleanly_finished_history_item_is_completed() -> None:
    [record] = _records(history=[_item(result={"exit_status": "completed"})])
    assert record["status"] == "completed"
    assert "error" not in record


@pytest.mark.parametrize("exit_status", ["aborted", "halted", "stopped"])
def test_every_abort_verb_collapses_to_stopped(exit_status: str) -> None:
    """The manager has three spellings for "a human stopped this"; an operator
    reading a run list should not have to know which one got recorded."""
    [record] = _records(history=[_item(result={"exit_status": exit_status})])
    assert record["status"] == "stopped"


def test_a_failed_history_item_is_an_error_carrying_the_managers_message() -> None:
    [record] = _records(
        history=[_item(result={"exit_status": "failed", "msg": "device timed out"})]
    )
    assert record["status"] == "error"
    assert record["error"] == "device timed out"


def test_a_failed_item_with_no_message_still_says_something() -> None:
    [record] = _records(history=[_item(result={"exit_status": "failed"})])
    assert record["status"] == "error"
    assert "failed" in record["error"]


@pytest.mark.parametrize("result", [{}, {"exit_status": None}, {"exit_status": "who-knows"}])
def test_an_unrecognized_exit_status_reads_as_error_not_completion(result: dict) -> None:
    """Fail-closed on the terminal state: an item that left the queue without
    the manager recording a clean finish must never be mistaken for a
    successful scan."""
    [record] = _records(history=[_item(result=result)])
    assert record["status"] == "error"


# =========================================================================
# Record contents
# =========================================================================


def test_record_carries_the_plan_name_and_its_unwrapped_kwargs() -> None:
    args = {"readbacks": ["BPM1"], "axes": [{"setpoint": "COR1"}]}
    [record] = _records(pending=[_item(name="grid_scan", kwargs=args)])
    assert record["plan_name"] == "grid_scan"
    assert record["plan_args"] == args


def test_plan_args_is_a_copy_not_the_managers_own_dict() -> None:
    """A shallow copy, which is the guarantee that matters: the record is
    serialized to JSON on the way out, so nothing downstream mutates it — this
    only stops the projection from handing out an alias of the manager
    document it was handed."""
    args = {"readbacks": ["BPM1"]}
    item = _item(kwargs=args)
    [record] = _records(pending=[item])
    assert record["plan_args"] is not args
    record["plan_args"]["extra"] = True
    assert "extra" not in args


def test_plan_args_is_an_empty_dict_when_the_item_carries_none() -> None:
    item = _item()
    del item["kwargs"]
    [record] = _records(pending=[item])
    assert record["plan_args"] == {}


def test_item_uid_is_surfaced() -> None:
    [record] = _records(pending=[_item(item_uid="qs-uid-42")])
    assert record["item_uid"] == "qs-uid-42"


def test_run_uid_comes_from_the_history_result() -> None:
    [record] = _records(
        history=[_item(result={"exit_status": "completed", "run_uids": ["re-uid-9"]})]
    )
    assert record["run_uid"] == "re-uid-9"


def test_a_pending_item_has_no_run_uid_at_all() -> None:
    """The RunEngine uid only exists once the worker has actually started the
    plan — reporting one for a queued item would be an invention."""
    [record] = _records(pending=[_item()])
    assert "run_uid" not in record


# =========================================================================
# Progress
# =========================================================================


def test_progress_is_absent_when_the_document_plane_knows_nothing() -> None:
    """Never a fabricated 0.0: "no idea" and "just started" are different
    answers, and a panel renders them differently."""
    [record] = _records(running=_item())
    assert "progress" not in record


def test_progress_is_attached_when_the_document_plane_has_rows() -> None:
    document_plane.record_run_params(
        "run-1", {"axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 4}]}
    )
    recorder = live_rows.LiveRowRecorder(key="run-1")
    recorder("start", {"uid": "re-uid"})
    recorder("event", {"data": {"BPM1": 1.0}})

    [record] = _records(running=_item())
    assert record["progress"]["rows_seen"] == 1
    assert record["progress"]["expected_points"] == 4
    assert record["progress"]["fraction"] == pytest.approx(0.25)


# =========================================================================
# Ordering, identity, and bounds
# =========================================================================


def test_order_is_running_then_pending_then_history_newest_first() -> None:
    records = _records(
        running=_item("running", item_uid="u-r"),
        pending=[_item("next", item_uid="u-1"), _item("later", item_uid="u-2")],
        history=[
            _item("older", item_uid="u-3", result={"exit_status": "completed"}),
            _item("newer", item_uid="u-4", result={"exit_status": "completed"}),
        ],
    )
    assert [r["id"] for r in records] == ["running", "next", "later", "newer", "older"]


def test_items_without_an_osprey_run_id_are_absent() -> None:
    """Something enqueued out-of-band has no id `GET /runs/{id}` could answer
    on; `GET /queue` is where the manager's queue is shown whole."""
    records = _records(pending=[_item(None, item_uid="u-1"), _item("mine", item_uid="u-2")])
    assert [r["id"] for r in records] == ["mine"]


def test_a_run_id_is_reported_once_with_the_most_current_source_winning() -> None:
    records = _records(
        running=_item("run-1"),
        history=[_item("run-1", result={"exit_status": "completed"})],
    )
    assert len(records) == 1
    assert records[0]["status"] == "running"


def test_limit_bounds_the_list() -> None:
    records = _records(pending=[_item(f"run-{n}", item_uid=f"u-{n}") for n in range(10)], limit=3)
    assert [r["id"] for r in records] == ["run-0", "run-1", "run-2"]


@pytest.mark.parametrize(("limit", "expected"), [(0, 1), (-5, 1), (500, 40)])
def test_limit_is_clamped_into_one_to_one_hundred(limit: int, expected: int) -> None:
    records = _records(
        pending=[_item(f"run-{n}", item_uid=f"u-{n}") for n in range(40)], limit=limit
    )
    assert len(records) == expected


def test_a_malformed_entry_is_skipped_rather_than_crashing_the_list() -> None:
    records = _records(pending=[None, {}, _item("real")])
    assert [r["id"] for r in records] == ["real"]


# =========================================================================
# find_record
# =========================================================================


def test_find_record_locates_a_run_in_any_of_the_three_sources() -> None:
    kwargs = {
        "running": _item("running", item_uid="u-r"),
        "pending": [_item("queued", item_uid="u-1")],
        "history": [_item("done", item_uid="u-2", result={"exit_status": "completed"})],
    }
    for run_id, status in (("running", "running"), ("queued", "pending"), ("done", "completed")):
        record = runs.find_record(
            run_id,
            running_item=kwargs["running"],
            queue_items=kwargs["pending"],
            history_items=kwargs["history"],
        )
        assert record is not None
        assert record["status"] == status


def test_find_record_returns_none_for_a_run_the_manager_has_forgotten() -> None:
    assert (
        runs.find_record("gone", running_item=None, queue_items=[_item("here")], history_items=[])
        is None
    )


# =========================================================================
# Requeued-after-interruption items.
#
# Upstream (`plan_queue_ops._set_processed_item_as_stopped`) does not discard a
# plan the worker interrupted: it adds the run to HISTORY with its exit_status
# AND pushes a copy back to the FRONT of the queue under a NEW item_uid,
# keeping the `result`. Since `_ordered_records` walks the queue before history
# and emits each run id once, that copy would otherwise SHADOW the correct
# history record and publish a just-aborted scan as `pending`.
#
# Found against a real deployed manager by tests/e2e/test_bluesky_queue_e2e.py;
# the mocked-client tests here were written to the assumption that an aborted
# item simply moves to history.
# =========================================================================


@pytest.mark.parametrize("exit_status", ["aborted", "halted"])
def test_an_interrupted_item_requeued_by_the_manager_is_stopped_not_pending(
    exit_status: str,
) -> None:
    """The operator who hit Abort must not see the run listed as work still to come."""
    requeued = _item(
        "run-1",
        item_uid="new-uid-after-requeue",
        result={"exit_status": exit_status, "run_uids": ["re-uid"]},
    )
    (record,) = _records(pending=[requeued])
    assert record["status"] == "stopped", (
        "a plan that already ran and was interrupted is not pending work"
    )


def test_a_requeued_failed_item_is_an_error_not_pending() -> None:
    """The same mechanism requeues FAILED plans; they must read as errors."""
    requeued = _item("run-1", item_uid="new-uid", result={"exit_status": "failed"})
    (record,) = _records(pending=[requeued])
    assert record["status"] == "error"


def test_a_requeued_item_with_an_unrecognized_exit_status_fails_closed() -> None:
    """Same fail-closed rule as history: not-a-clean-finish never reads as success."""
    requeued = _item("run-1", item_uid="new-uid", result={"exit_status": "something-new"})
    (record,) = _records(pending=[requeued])
    assert record["status"] == "error"


def test_the_requeued_copy_and_the_history_entry_agree() -> None:
    """One run id, one status — whichever source the projection happens to take.

    This is the shape the defect actually had: both sources describe the same
    finished run, the queue is visited first, and the record it produced said
    `pending` while the history entry said `stopped`.
    """
    result = {"exit_status": "aborted", "run_uids": ["re-uid"]}
    records = _records(
        pending=[_item("run-1", item_uid="requeued-uid", result=result)],
        history=[_item("run-1", item_uid="original-uid", result=result)],
    )
    assert len(records) == 1, "the same run must not be reported twice"
    assert records[0]["status"] == "stopped"
    assert records[0]["run_uid"] == "re-uid"


def test_a_genuinely_pending_item_still_has_no_status_from_a_result() -> None:
    """Negative control: an item with no `result` (and one with an empty one) is pending.

    Without this, "look for a result" could quietly become "everything in the
    queue reads as finished" and the pending state would vanish.
    """
    assert _records(pending=[_item("run-1")])[0]["status"] == "pending"
    assert _records(pending=[_item("run-2", result={})])[0]["status"] == "pending"
    assert _records(pending=[_item("run-3", result={"exit_status": ""})])[0]["status"] == "pending"


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (None, False),
        ({}, False),
        ({"result": {}}, False),
        ({"result": {"exit_status": ""}}, False),
        ({"result": "not-a-dict"}, False),
        ({"result": {"exit_status": "aborted"}}, True),
        ({"result": {"exit_status": "failed"}}, True),
    ],
)
def test_is_requeued_after_interruption_is_exact(item: object, expected: bool) -> None:
    """The predicate the ARMING path uses to refuse a start that would re-run an
    aborted plan. Pinned separately from the projection because a false
    negative there is a silent re-run onto hardware."""
    assert runs.is_requeued_after_interruption(item) is expected
