"""The plan-identity stamp never reaches a queue client (`queue._public_item`).

The enqueue path stamps the plan's name and kwargs into the item's metadata
(`queue_backend.PLAN_META_KEY`) so a run's figure can be rendered from its
start document alone. On a queue READ that stamp duplicates the item's own
top-level ``name``/``kwargs``, on the one surface panels both poll and stream
continuously — so both relays drop it.

What these tests pin is the ASYMMETRY, in both directions at once: the item
the manager holds carries BOTH keys (the stamp has to reach the RunEngine, or
the whole feature is dead), while every item on the wire carries only
``osprey_run_id`` (drop that one and `itemRunId()` can no longer join a queue
row to its run). Both relays are checked against a real response: `GET /queue`
over `TestClient`, and the SSE stream over an actual `TestClient.stream`
capture of the hello frame plus a driven change frame.

The last section pins the other half of the same stamp — what `add_item`
WRITES when the add names an owner. The owner reaches the worker as a reserved
kwarg on the item and is stamped on the item's metadata, but the plan-identity
copy is built without it, because that copy is the one carrier a read-side
strip can never reach: it travels into the run's start document, which the
results table and the live-row recorder both read back.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import document_plane, draft, live_rows, plan_loader, queue
from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.queue_backend import QueueBackend
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"

_GRID_SCAN_ARGS: dict[str, Any] = {
    "readbacks": ["BPM1"],
    "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
}
_RUN_ID = "run-abc"

# The stamp `queue_backend.add_item` puts on an item enqueued with a run id.
_STAMP = {"name": "grid_scan", "kwargs": _GRID_SCAN_ARGS}


class FakeManager:
    """A scripted ``REManagerAPI`` stand-in (same contract as `test_queue_routes.py`'s)."""

    def __init__(self, **responses: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses

    def __getattr__(self, method: str) -> Any:
        async def call(**kwargs: Any) -> Any:
            self.calls.append((method, kwargs))
            response = self.responses.get(method, {"success": True, "msg": ""})
            if isinstance(response, Exception):
                raise response
            return response

        return call

    async def close(self) -> None:  # pragma: no cover - lifespan only
        pass


def status_doc(**overrides: Any) -> dict[str, Any]:
    """A manager status document: idle, environment open, nothing running."""
    doc = {
        "success": True,
        "manager_state": "idle",
        "worker_environment_exists": True,
        "items_in_queue": 1,
        "items_in_history": 0,
        "running_item_uid": None,
        "plan_queue_uid": "q-1",
        "plan_history_uid": "h-1",
        "queue_stop_pending": False,
        "queue_autostart_enabled": False,
    }
    doc.update(overrides)
    return doc


def stamped_item(uid: str = "u1", **overrides: Any) -> dict[str, Any]:
    """A queue item exactly as the manager holds it after a stamped enqueue."""
    item = {
        "item_uid": uid,
        "item_type": "plan",
        "name": "grid_scan",
        "kwargs": dict(_GRID_SCAN_ARGS),
        "meta": {qb.RUN_ID_META_KEY: _RUN_ID, qb.PLAN_META_KEY: dict(_STAMP)},
    }
    item.update(overrides)
    return item


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.delenv(_TOKEN_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    document_plane._clear()
    app_module.set_queue_backend(None)
    yield
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    document_plane._clear()
    app_module.set_queue_backend(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _install(manager: Any) -> QueueBackend:
    backend = QueueBackend(manager)
    app_module.set_queue_backend(backend)
    return backend


def _parse_frame(raw: str) -> dict[str, Any]:
    assert raw.startswith("data: ")
    return json.loads(raw[len("data: ") :])


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


def test_public_item_drops_the_plan_stamp_and_keeps_the_run_id() -> None:
    public = queue._public_item(stamped_item())

    assert public["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}
    # Everything else is the item verbatim — the strip touches metadata only.
    assert public["name"] == "grid_scan"
    assert public["kwargs"] == _GRID_SCAN_ARGS


def test_public_item_leaves_metadata_it_did_not_write_alone() -> None:
    """An item enqueued out of band (a `qserver` CLI, another client) carries
    whatever metadata its author chose; none of it is ours to remove."""
    foreign = {"item_uid": "u9", "name": "count", "meta": {"scan_id": 7, "owner": "beamline"}}

    assert queue._public_item(foreign) is foreign
    assert queue._public_item({"item_uid": "u9"}) == {"item_uid": "u9"}
    assert queue._public_item(None) is None


def test_public_item_does_not_mutate_the_item_it_was_given() -> None:
    """The strip is a projection for the wire; the caller's item — and the
    manager's own response dict it came out of — must be untouched."""
    item = stamped_item()

    queue._public_item(item)

    assert item["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID, qb.PLAN_META_KEY: _STAMP}


# ---------------------------------------------------------------------------
# Relay 1: GET /queue
# ---------------------------------------------------------------------------


def test_get_queue_relays_items_without_the_plan_stamp(client: TestClient) -> None:
    manager = FakeManager(
        status=status_doc(),
        queue_get={
            "success": True,
            "items": [stamped_item("u1")],
            "running_item": stamped_item("u2"),
        },
    )
    _install(manager)

    body = client.get("/queue").json()

    # The manager's own item is the control: it carries BOTH keys, because the
    # stamp has to reach the RunEngine for a figure to be renderable at all.
    (held,) = manager.responses["queue_get"]["items"]
    assert held["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID, qb.PLAN_META_KEY: _STAMP}

    (relayed,) = body["items"]
    assert relayed["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}
    assert relayed["kwargs"] == _GRID_SCAN_ARGS
    assert body["running_item"]["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}


def test_the_running_items_progress_survives_the_strip(client: TestClient) -> None:
    """`_with_progress` and `_public_item` compose on the running item — one
    adds a key, the other removes one, and neither undoes the other."""
    _install(
        FakeManager(
            status=status_doc(running_item_uid="u2"),
            queue_get={"success": True, "items": [], "running_item": stamped_item("u2")},
        )
    )
    document_plane.record_expected_points(_RUN_ID, 3)

    running = client.get("/queue").json()["running_item"]

    assert running["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}
    assert running["progress"] is not None


# ---------------------------------------------------------------------------
# Relay 2: GET /queue/events — the SSE stream
# ---------------------------------------------------------------------------


def test_a_streamed_hello_frame_carries_no_plan_stamp(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame is read off a real SSE response over HTTP, not off `_frame_from`.

    The stream is ended right after the hello frame by pushing the module's own
    ``_DISCONNECT`` sentinel (the slow-consumer path) into the new subscriber:
    Starlette's `TestClient` buffers a response to completion before returning
    it, so a genuinely endless stream would simply hang. Everything up to that
    point is production code — the route, `_frame_from`, `_format_sse`, and the
    ASGI transport.
    """
    _install(
        FakeManager(
            status=status_doc(),
            queue_get={
                "success": True,
                "items": [stamped_item("u1")],
                "running_item": stamped_item("u2"),
            },
        )
    )

    subscribe = queue._subscribe

    async def _subscribe_then_end() -> tuple[asyncio.Queue[Any], dict[str, Any]]:
        subscriber, hello = await subscribe()
        subscriber.put_nowait(queue._DISCONNECT)
        return subscriber, hello

    monkeypatch.setattr(queue, "_subscribe", _subscribe_then_end)

    with client.stream("GET", "/queue/events") as resp:
        assert resp.status_code == 200
        hello = _parse_frame(next(line for line in resp.iter_lines() if line.startswith("data: ")))

    assert hello["type"] == "hello"
    (streamed,) = hello["items"]
    assert streamed["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}
    assert streamed["name"] == "grid_scan"
    assert hello["running_item"]["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}


async def test_a_streamed_change_frame_carries_no_plan_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hello frame is not a special case: every later change frame is
    built by the same `_frame_from`, so the strip has to hold there too."""
    monkeypatch.setattr(queue, "_POLL_INTERVAL_S", 0.01)
    manager = FakeManager(
        status=status_doc(items_in_queue=0),
        queue_get={"success": True, "items": [], "running_item": {}},
    )
    _install(manager)

    resp = await queue.queue_events()
    gen = resp.body_iterator
    try:
        assert _parse_frame(await gen.__anext__())["type"] == "hello"

        # An enqueue lands out of band: the poller notices and pushes a full
        # snapshot, which is the second place a stamp could escape.
        manager.responses["status"] = status_doc(plan_queue_uid="q-2")
        manager.responses["queue_get"] = {
            "success": True,
            "items": [stamped_item("u1")],
            "running_item": {},
        }

        frame = _parse_frame(await asyncio.wait_for(gen.__anext__(), timeout=2))
    finally:
        await gen.aclose()

    assert frame["type"] == "queue"
    (streamed,) = frame["items"]
    assert streamed["meta"] == {qb.RUN_ID_META_KEY: _RUN_ID}
    assert qb.PLAN_META_KEY not in json.dumps(frame)


# ---------------------------------------------------------------------------
# The owner an enqueue stamps
# ---------------------------------------------------------------------------


def _plan_item(**extra_args: Any) -> dict[str, Any]:
    """An item in the shape the add route builds, before any stamping."""
    args = dict(_GRID_SCAN_ARGS)
    args.update(extra_args)
    return {"item_type": "plan", "name": "grid_scan", "kwargs": args}


def _added_item(manager: FakeManager) -> dict[str, Any]:
    """The item dict the one recorded ``item_add`` call actually carried."""
    adds = [kwargs for method, kwargs in manager.calls if method == "item_add"]
    assert len(adds) == 1
    return adds[0]["item"]


def _start_document(item: dict[str, Any], run_uid: str = "run-uid-1") -> dict[str, Any]:
    """The start document a worker publishes for *item*.

    It carries the item's METADATA and the RunEngine's own uid — never the
    item's kwargs, because the worker has no kwargs-to-metadata channel. That
    is why the plan stamp is filtered where it is written: no strip applied on
    a queue read can reach this document.
    """
    return {**item["meta"], "uid": run_uid}


async def test_an_owned_add_sends_the_owner_as_a_kwarg_and_stamps_a_filtered_plan_copy() -> None:
    """The reserved kwarg is the only channel that reaches the plan wrapper, so
    it rides the item's own ``kwargs``; the plan-identity copy is built without
    it, because that copy is what the run's start document republishes as the
    plan its results are rendered as."""
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item(_plan_item(), run_id=_RUN_ID, owner="bob")

    item = _added_item(manager)
    assert item["kwargs"][RESERVED_OWNER_KWARG] == "bob"
    # The plan's own arguments ride beside it, untouched.
    assert {
        key: value for key, value in item["kwargs"].items() if key != RESERVED_OWNER_KWARG
    } == _GRID_SCAN_ARGS
    assert item["meta"][qb.OWNER_META_KEY] == "bob"
    assert item["meta"][qb.PLAN_META_KEY] == {"name": "grid_scan", "kwargs": _GRID_SCAN_ARGS}
    assert item["meta"][qb.RUN_ID_META_KEY] == _RUN_ID


async def test_an_external_worker_add_carries_the_owner_in_metadata_only() -> None:
    """A facility-run RE Manager binds an item against the plan's own
    signature, which has no parameter for an owner — so the kwarg would be
    refused there and the metadata stamp is the whole attribution."""
    manager = FakeManager()
    backend = QueueBackend(manager, external_worker=True)

    await backend.add_item(_plan_item(), run_id=_RUN_ID, owner="bob")

    item = _added_item(manager)
    # Nowhere on the item: not in kwargs, not in the plan stamp, not in meta.
    assert RESERVED_OWNER_KWARG not in json.dumps(item)
    assert item["kwargs"] == _GRID_SCAN_ARGS
    assert item["meta"][qb.OWNER_META_KEY] == "bob"


async def test_an_owned_add_with_no_run_id_still_names_its_owner() -> None:
    """The plan stamp is written only for an item the bridge gave a run id.
    The owner stamp cannot sit behind that condition, or an add that mints no
    run id would reach the queue unattributable."""
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item(_plan_item(), owner="bob")

    item = _added_item(manager)
    assert item["meta"] == {qb.OWNER_META_KEY: "bob"}
    assert item["kwargs"][RESERVED_OWNER_KWARG] == "bob"


async def test_an_owner_less_add_stamps_no_owner_anywhere() -> None:
    """The negative control: without an owner the item is exactly the item a
    stamped enqueue has always produced."""
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item(_plan_item(), run_id=_RUN_ID)

    item = _added_item(manager)
    assert item["kwargs"] == _GRID_SCAN_ARGS
    assert item["meta"] == {
        qb.RUN_ID_META_KEY: _RUN_ID,
        qb.PLAN_META_KEY: {"name": "grid_scan", "kwargs": _GRID_SCAN_ARGS},
    }


async def test_an_item_arriving_with_the_reserved_key_keeps_it_out_of_the_plan_metadata() -> None:
    """The plan copy is filtered, not merely un-injected: the key is reserved
    whoever put it there, and the start document is the one carrier no
    read-side strip can reach."""
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item(_plan_item(**{RESERVED_OWNER_KWARG: "smuggled"}), run_id=_RUN_ID)

    item = _added_item(manager)
    assert item["meta"][qb.PLAN_META_KEY]["kwargs"] == _GRID_SCAN_ARGS


async def test_the_plan_metadata_a_results_table_republishes_carries_no_reserved_key() -> None:
    """A finished run's table payload publishes ``start[PLAN_META_KEY]``
    verbatim as its ``plan`` (`app._tiled_run_snapshot`), so the stamp written
    at enqueue is the whole of what stands between the reserved key and every
    reader of that run."""
    manager = FakeManager()
    backend = QueueBackend(manager)
    await backend.add_item(_plan_item(), run_id=_RUN_ID, owner="bob")

    start = _start_document(_added_item(manager))

    assert start[qb.PLAN_META_KEY] == {"name": "grid_scan", "kwargs": _GRID_SCAN_ARGS}
    assert start[qb.OWNER_META_KEY] == "bob"


async def test_the_live_row_recorders_plan_metadata_carries_no_reserved_key() -> None:
    """The document plane opens each run's recorder with the start document's
    plan stamp and hands that value back on every live-rows read — the second
    reader of the same filtered copy, and the one a panel renders live."""
    manager = FakeManager()
    backend = QueueBackend(manager)
    await backend.add_item(_plan_item(), run_id=_RUN_ID, owner="bob")
    start = _start_document(_added_item(manager))

    try:
        document_plane.RunDocumentRouter()("start", start)
        recorded = live_rows.get(_RUN_ID)
    finally:
        live_rows._clear()

    assert recorded is not None
    assert recorded["plan"] == {"name": "grid_scan", "kwargs": _GRID_SCAN_ARGS}


async def test_the_reserved_key_an_item_arrived_with_never_reaches_the_worker() -> None:
    """The item's own ``kwargs`` are the carrier the worker BINDS, so the
    filter has to hold there and not only on the plan-identity copy.

    A reserved key the enqueuer put on the item is dropped whoever put it
    there: an owner-less add reaches the manager owner-less, and the run is
    judged against nobody's narrowing rather than against a name the bridge
    never stamped.
    """
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item(_plan_item(**{RESERVED_OWNER_KWARG: "smuggled"}), run_id=_RUN_ID)

    item = _added_item(manager)
    assert item["kwargs"] == _GRID_SCAN_ARGS
    assert qb.OWNER_META_KEY not in item["meta"]


async def test_the_stamped_owner_replaces_a_reserved_key_the_item_arrived_with() -> None:
    """The value this call was handed is the single source of the key, so an
    owned add cannot end up naming two people — one to the worker, another on
    the row a queue read renders."""
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item(
        _plan_item(**{RESERVED_OWNER_KWARG: "smuggled"}), run_id=_RUN_ID, owner="bob"
    )

    item = _added_item(manager)
    assert item["kwargs"][RESERVED_OWNER_KWARG] == "bob"
    assert item["meta"][qb.OWNER_META_KEY] == "bob"
    assert "smuggled" not in json.dumps(item)


async def test_an_external_worker_lane_drops_a_reserved_key_it_would_be_refused_for() -> None:
    """A facility RE Manager binds every add against the plan's own signature,
    which has no parameter for the reserved key — so an item arriving with one
    would be refused there. The lane that cannot carry the kwarg carries none,
    whatever the item brought."""
    manager = FakeManager()
    backend = QueueBackend(manager, external_worker=True)

    await backend.add_item(
        _plan_item(**{RESERVED_OWNER_KWARG: "smuggled"}), run_id=_RUN_ID, owner="bob"
    )

    item = _added_item(manager)
    assert RESERVED_OWNER_KWARG not in json.dumps(item)
    assert item["meta"][qb.OWNER_META_KEY] == "bob"


async def test_an_instruction_takes_the_metadata_stamp_alone() -> None:
    """Only a plan has a kwargs surface an owner may ride: a queueserver
    instruction is bound against its own signature, so giving it kwargs would
    turn an attribution into a refused enqueue. The metadata stamp is the whole
    attribution there, the same shape an external-worker lane takes."""
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.add_item({"item_type": "instruction", "name": "queue_stop"}, owner="bob")

    assert _added_item(manager) == {
        "item_type": "instruction",
        "name": "queue_stop",
        "meta": {qb.OWNER_META_KEY: "bob"},
    }
