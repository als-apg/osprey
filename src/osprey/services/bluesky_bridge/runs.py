"""Run records, projected from the queue server's own item state.

The bridge does not execute plans — the queueserver worker does (see
``qserver_startup.py``). A run is therefore not an object this process owns and
mutates through a lifecycle; it is a *view* over three things the manager
already holds: the pending queue, the running item, and the history. Each is
keyed back to OSPREY by the run id the enqueue path stamps into the item's
metadata (``queue_backend.RUN_ID_META_KEY``), which is also the key the
document plane buffers live rows under — so one id joins the queue, the live
data, and the durable Tiled record.

This module is the projection alone. It takes already-fetched manager
documents and returns the record shape ``GET /runs`` and ``GET /runs/{id}``
publish; it opens no connections and raises no ``HTTPException``. Keeping the
I/O in the route layer is what makes every status-mapping decision below
testable against a literal manager document, with no manager and no app.

Items the manager holds that carry no OSPREY run id are deliberately absent
from these records: something enqueued out-of-band (a ``qserver`` CLI, another
client) has no id for ``GET /runs/{id}`` to answer on, and inventing one would
make two id spaces look like one. ``GET /queue`` is the complete view of what
the manager is holding, and shows those items unfiltered.
"""

from __future__ import annotations

from typing import Any

from . import document_plane
from .queue_backend import run_id_of

# The lifecycle states a run record reports. Deliberately the same vocabulary
# the pre-queue bridge published, so consumers that only ever branched on
# `status` did not have to relearn it when execution moved to the worker.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_ERROR = "error"

# Queueserver's terminal `exit_status` values, mapped onto the record states
# above. The three stop-shaped spellings collapse to one: an operator who
# halted a scan does not care which of the manager's three abort verbs got
# recorded, and splitting them would push that trivia onto every consumer.
_EXIT_STATUS_TO_STATUS = {
    "completed": STATUS_COMPLETED,
    "aborted": STATUS_STOPPED,
    "halted": STATUS_STOPPED,
    "stopped": STATUS_STOPPED,
    "failed": STATUS_ERROR,
}

# History entries whose `exit_status` is missing or unrecognized. Read as an
# error rather than as completion: an item that left the queue without the
# manager recording a clean finish is exactly the case an operator must not
# mistake for a successful scan.
_UNKNOWN_EXIT_STATUS = STATUS_ERROR

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def _plan_args(item: dict[str, Any]) -> dict[str, Any]:
    """The item's plan parameters.

    Queueserver item ``kwargs`` ARE the plan's PARAMS fields, unwrapped (see
    ``qserver_startup.py``'s kwargs contract) — there is no ``params``
    envelope to unpack here.
    """
    kwargs = item.get("kwargs")
    return dict(kwargs) if isinstance(kwargs, dict) else {}


def _history_status(item: dict[str, Any]) -> str:
    """The record state for a history entry, from its recorded ``exit_status``."""
    result = item.get("result")
    exit_status = result.get("exit_status") if isinstance(result, dict) else None
    return _EXIT_STATUS_TO_STATUS.get(exit_status, _UNKNOWN_EXIT_STATUS)


def _queue_status(item: dict[str, Any]) -> str:
    """The record state for an item sitting in the QUEUE — usually, but not
    always, ``pending``.

    A plan the worker interrupted does not simply leave the queue. Upstream
    (``plan_queue_ops._set_processed_item_as_stopped``) adds it to history with
    its ``exit_status`` AND — for every exit status except ``"stopped"`` —
    pushes a COPY back to the FRONT of the queue under a NEW ``item_uid``, so
    the operator can decide whether to run it again. That copy keeps the
    ``result`` block, and it keeps the same OSPREY run id.

    That matters here because `_ordered_records` walks the queue BEFORE history
    and emits each run id once: without this, the requeued copy SHADOWS the
    correct history record, and a scan a human just emergency-aborted is
    published as ``pending`` — as work that has not happened yet — while
    carrying the ``run_uid`` of the run that already did. So an item bearing a
    ``result`` is projected through the SAME mapping history uses (aborted →
    ``stopped``, failed → ``error``, anything unrecognized → ``error``,
    fail-closed), and the two views agree.

    A genuinely pending item has no ``result`` at all, and reports ``pending``.
    """
    result = item.get("result")
    if not isinstance(result, dict) or not result.get("exit_status"):
        return STATUS_PENDING
    return _history_status(item)


def is_requeued_after_interruption(item: Any) -> bool:
    """Whether this QUEUE item is a copy upstream pushed back after an interruption.

    True exactly when a queued item carries a ``result`` with an
    ``exit_status`` — which, per the mechanism in `_queue_status`, only happens
    for a plan that already ran and did not finish cleanly. Exported because
    the arming path needs the same question answered: starting a queue whose
    head is an aborted plan would re-run it with no fresh human decision.
    """
    if not isinstance(item, dict):
        return False
    result = item.get("result")
    return isinstance(result, dict) and bool(result.get("exit_status"))


def _error_text(item: dict[str, Any]) -> str:
    """An operator-facing reason for a run that ended badly.

    Prefers the manager's own message; falls back to naming the exit status so
    the record never reports an error with nothing said about it.
    """
    result = item.get("result")
    if isinstance(result, dict):
        msg = result.get("msg")
        if isinstance(msg, str) and msg.strip():
            return msg
        exit_status = result.get("exit_status")
        if exit_status:
            return f"plan ended with exit status {exit_status!r}"
    return "plan did not complete and the manager recorded no reason"


def record_from_item(item: dict[str, Any], status: str) -> dict[str, Any]:
    """One manager item, as the run record the HTTP surface publishes.

    ``run_uid`` is the RunEngine's own uid, which only exists once the worker
    has actually started the plan — present on history entries, absent on a
    pending item. An item requeued after an interruption carries one too, and
    is not reported as pending precisely so the uid and the status agree (see
    `_queue_status`). ``progress`` comes from the document plane and is omitted
    entirely when nothing is known, never reported as a fabricated ``0.0``
    (an unknown denominator is common for agent-authored session plans, and a
    consumer must be able to tell "no idea" from "just started").
    """
    run_id = run_id_of(item)
    record: dict[str, Any] = {
        "id": run_id,
        "status": status,
        "plan_name": item.get("name"),
        "plan_args": _plan_args(item),
    }

    item_uid = item.get("item_uid")
    if isinstance(item_uid, str):
        record["item_uid"] = item_uid

    result = item.get("result")
    if isinstance(result, dict):
        run_uids = result.get("run_uids")
        if isinstance(run_uids, list) and run_uids:
            record["run_uid"] = run_uids[0]

    if status == STATUS_ERROR:
        record["error"] = _error_text(item)

    if run_id is not None:
        progress = document_plane.progress(run_id)
        if progress is not None:
            record["progress"] = progress

    return record


def _ordered_records(
    *,
    running_item: Any,
    queue_items: list[Any] | None,
    history_items: list[Any] | None,
) -> list[dict[str, Any]]:
    """Every OSPREY-enqueued run the manager knows about, most relevant first.

    Order is running, then pending in queue order, then history newest first —
    "what is happening now, what is next, what already happened", which is how
    an operator reads a run list. It is deliberately NOT a single global sort
    by time: a pending item has no start time to sort on.

    A run id is emitted once. Should the same id somehow appear in two of the
    three sources, the earlier (more current) source wins, so a run that has
    just moved from running to history never briefly reports as both.

    The one case where the queue is NOT the more current view is an
    interrupted plan: upstream pushes a copy of it back to the front of the
    queue while also recording it in history, so the queue copy and the
    history entry describe the same finished run. `_queue_status` resolves
    that by projecting a queue item's own ``result`` through the history
    mapping, which makes the two views agree and leaves this precedence rule
    intact rather than making it a special case here.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append(item: Any, status: str) -> None:
        if not isinstance(item, dict) or not item:
            return
        run_id = run_id_of(item)
        if run_id is None or run_id in seen:
            return
        seen.add(run_id)
        records.append(record_from_item(item, status))

    _append(running_item, STATUS_RUNNING)
    for item in queue_items or []:
        _append(item, _queue_status(item) if isinstance(item, dict) else STATUS_PENDING)
    for item in reversed(history_items or []):
        _append(item, _history_status(item) if isinstance(item, dict) else STATUS_ERROR)

    return records


def list_records(
    *,
    running_item: Any,
    queue_items: list[Any] | None,
    history_items: list[Any] | None,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """The run list ``GET /runs`` serves, bounded by ``limit``.

    ``limit`` is clamped into ``1..100`` exactly as the pre-queue registry
    clamped it, so an unbounded or nonsensical query can never make this route
    the way to pull the manager's whole history in one response.
    """
    bound = max(1, min(limit, _MAX_LIMIT))
    return _ordered_records(
        running_item=running_item, queue_items=queue_items, history_items=history_items
    )[:bound]


def find_record(
    run_id: str,
    *,
    running_item: Any,
    queue_items: list[Any] | None,
    history_items: list[Any] | None,
) -> dict[str, Any] | None:
    """The record for *run_id*, or ``None`` if the manager knows no such run.

    ``None`` is also the honest answer for a run whose history the manager has
    since rotated away — nothing in this process remembers runs the manager
    has forgotten, and the caller turns that into a 404.
    """
    for record in _ordered_records(
        running_item=running_item, queue_items=queue_items, history_items=history_items
    ):
        if record["id"] == run_id:
            return record
    return None
