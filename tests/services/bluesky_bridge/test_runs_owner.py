"""The owner an item carries, as the `/runs` surface reports it.

A run record is a projection of the manager's own items, and those items reach
the projection RAW: `runs.record_from_item` never passes through `queue.py`'s
`_public_item`, so the read-side split has to happen here too or the reserved
owner kwarg reaches `GET /runs` as if it were a plan argument.

The three item shapes an operator can see — the running item, a pending queue
item, and a history entry — are all projected by the same function, so each is
driven end to end through `GET /runs` rather than asserted on the projection
alone: what a consumer reads is the response body, and that is what these pin.
Both lane shapes are covered, because the owner reaches an item two ways: the
reserved kwarg on lanes this deployment deploys, and the metadata stamp on
external-worker lanes, whose facility manager would refuse an unknown kwarg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import document_plane, history_removals, live_rows, runs
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.queue_backend import OWNER_META_KEY, QueueBackend
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG


class _ScriptedManager:
    """The smallest `REManagerAPI` stand-in `GET /runs` needs."""

    def __init__(self, queue: dict[str, Any], history: dict[str, Any]) -> None:
        self._queue = queue
        self._history = history

    async def queue_get(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True, **self._queue}

    async def history_get(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True, **self._history}


def _item(
    run_id: str,
    *,
    owner_kwarg: str | None = None,
    owner_meta: str | None = None,
    result: dict | None = None,
) -> dict[str, Any]:
    """One manager item, optionally owned on either lane shape."""
    kwargs: dict[str, Any] = {"readbacks": ["BPM1"]}
    if owner_kwarg is not None:
        kwargs[RESERVED_OWNER_KWARG] = owner_kwarg
    meta: dict[str, Any] = {"osprey_run_id": run_id}
    if owner_meta is not None:
        meta[OWNER_META_KEY] = owner_meta
    item: dict[str, Any] = {
        "item_type": "plan",
        "name": "grid_scan",
        "kwargs": kwargs,
        "item_uid": f"uid-{run_id}",
        "meta": meta,
    }
    if result is not None:
        item["result"] = result
    return item


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(tmp_path / "plans_session"))
    history_removals._clear()
    live_rows._clear()
    document_plane._clear()
    yield
    history_removals._clear()
    live_rows._clear()
    document_plane._clear()
    app_module.set_queue_backend(None)


@pytest.fixture
def bridge():
    """A `TestClient` on the bridge, with a scripted manager behind the queue."""

    def _build(queue: Any = None, history: Any = None) -> TestClient:
        manager = _ScriptedManager(
            queue if queue is not None else {"items": [], "running_item": {}},
            history if history is not None else {"items": []},
        )
        app_module.set_queue_backend(QueueBackend(manager))
        return TestClient(app)

    return _build


def _record(client: TestClient, run_id: str) -> dict[str, Any]:
    with client:
        body = client.get("/runs").json()
    return next(record for record in body if record["id"] == run_id)


# =========================================================================
# The three item shapes, through GET /runs
# =========================================================================


def test_a_running_item_reports_its_owner_and_no_reserved_kwarg(bridge) -> None:
    client = bridge(queue={"items": [], "running_item": _item("live", owner_kwarg="alice")})

    record = _record(client, "live")

    assert record["owner"] == "alice"
    assert record["plan_args"] == {"readbacks": ["BPM1"]}


def test_a_pending_item_reports_its_owner_and_no_reserved_kwarg(bridge) -> None:
    client = bridge(queue={"items": [_item("queued", owner_kwarg="bob")], "running_item": {}})

    record = _record(client, "queued")

    assert record["status"] == "pending"
    assert record["owner"] == "bob"
    assert record["plan_args"] == {"readbacks": ["BPM1"]}


def test_a_history_entry_reports_its_owner_and_no_reserved_kwarg(bridge) -> None:
    client = bridge(
        history={
            "items": [
                _item("done", owner_kwarg="carol", result={"exit_status": "completed"}),
            ]
        }
    )

    record = _record(client, "done")

    assert record["status"] == "completed"
    assert record["owner"] == "carol"
    assert record["plan_args"] == {"readbacks": ["BPM1"]}


def test_a_single_run_lookup_carries_the_owner_too(bridge) -> None:
    """`GET /runs/{id}` projects the same item through the same function; a
    consumer that opened one run must not see a different attribution from the
    list it clicked through."""
    client = bridge(queue={"items": [_item("queued", owner_kwarg="alice")], "running_item": {}})

    with client:
        body = client.get("/runs/queued").json()

    assert body["owner"] == "alice"
    assert RESERVED_OWNER_KWARG not in body["plan_args"]


# =========================================================================
# Both lane shapes, and the absence of an owner
# =========================================================================


def test_an_external_worker_lane_reports_the_owner_from_the_metadata_stamp(bridge) -> None:
    """A facility manager validates every add against the plan's own signature,
    so those lanes carry no reserved kwarg — the stamp is the whole record of
    who enqueued the item."""
    client = bridge(queue={"items": [_item("queued", owner_meta="dave")], "running_item": {}})

    record = _record(client, "queued")

    assert record["owner"] == "dave"
    assert record["plan_args"] == {"readbacks": ["BPM1"]}


def test_the_kwarg_wins_over_the_metadata_stamp(bridge) -> None:
    """The kwarg is the value the worker binds, so it is the one that describes
    the run that actually happened."""
    client = bridge(
        queue={
            "items": [_item("queued", owner_kwarg="alice", owner_meta="dave")],
            "running_item": {},
        }
    )

    assert _record(client, "queued")["owner"] == "alice"


def test_an_unowned_item_carries_no_owner_key_at_all(bridge) -> None:
    """Absent rather than null: a consumer distinguishes "nobody claimed this"
    from an owner whose name failed to project."""
    client = bridge(queue={"items": [_item("queued")], "running_item": {}})

    record = _record(client, "queued")

    assert "owner" not in record
    assert record["plan_args"] == {"readbacks": ["BPM1"]}


def test_an_item_with_no_kwargs_is_projected_without_an_owner(bridge) -> None:
    item = _item("queued")
    del item["kwargs"]
    client = bridge(queue={"items": [item], "running_item": {}})

    record = _record(client, "queued")

    assert record["plan_args"] == {}
    assert "owner" not in record


# =========================================================================
# The projection leaves the manager's document alone
# =========================================================================


def test_projecting_an_item_does_not_strip_the_kwarg_from_the_managers_own_item() -> None:
    """The kwarg is what the worker binds; a projection that mutated the item
    in place would unbind the owner of a run the bridge merely looked at."""
    item = _item("queued", owner_kwarg="alice")

    record = runs.record_from_item(item, runs.STATUS_PENDING)

    assert item["kwargs"][RESERVED_OWNER_KWARG] == "alice"
    assert RESERVED_OWNER_KWARG not in record["plan_args"]
