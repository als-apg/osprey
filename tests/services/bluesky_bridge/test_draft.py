"""Coverage for the shared plan draft (`draft.py`): `GET`/`PATCH`/`DELETE
/draft` and the `GET /draft/events` SSE stream (PROPOSAL.md "Bridge draft
module").

Uses the `orm`/`grid_scan` shipped plans (`plans_core/`) as real schemas —
they're always registered regardless of env (the `shipped` tier is an
in-package directory scan, see `plan_loader.py`) — rather than authoring
throwaway fixture plans. Mirrors `test_plan_source_endpoint.py`'s isolation
fixture, plus resetting the draft singleton (`draft._clear()`).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import draft, plan_loader
from osprey.services.bluesky_bridge.app import app

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"

_ORM_ARGS: dict[str, Any] = {
    "correctors": ["COR1"],
    "detectors": ["BPM1"],
    "span_a": 1.0,
    "num": 3,
}
_GRID_SCAN_ARGS: dict[str, Any] = {
    "detectors": ["BPM1"],
    "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
}


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()
    draft._clear()
    yield
    plan_loader.reset_facility_plans()
    draft._clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _read_frame(lines: Iterator[str]) -> dict[str, Any]:
    """Read the next SSE `data:` frame from an `iter_lines()` iterator, skipping blanks/comments."""
    for line in lines:
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError("SSE stream ended before yielding a frame")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"bridge did not become ready on port {port} within {timeout}s")


@contextmanager
def _live_bridge() -> Iterator[str]:
    """Run the real bridge app on a real TCP port in a background thread.

    A genuine SSE stream needs a real socket: both `httpx.ASGITransport` and
    Starlette's `TestClient` drive the ASGI app to full completion before
    handing back *any* response (they exist to test request/response
    handlers, not infinite generators) — an in-process simulated transport
    would simply hang forever against `GET /draft/events`'s never-ending
    generator. Mirrors `test_read_bounded.py`'s `_live_bridge` helper: same
    module-level `draft`/`plan_loader` singletons as the test process (a
    thread, not a subprocess), but requests travel over a real socket
    through the real ASGI app.

    Cross-loop note: `draft.py`'s module-level asyncio primitives (`_lock`,
    subscriber queues) lazily bind to whichever event loop first touches
    them. This is safe only because this suite's loops are used strictly
    *sequentially* — the uvicorn server thread's loop runs and fully shuts
    down (`server.should_exit = True; t.join(...)`) before the next test's
    loop starts. Overlapping access from two live loops at once would fail
    with "Future ... attached to a different loop".
    """
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    _wait_for_port(port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)


# ---------------------------------------------------------------------------
# GET /draft
# ---------------------------------------------------------------------------


def test_get_draft_returns_revision_when_null(client: TestClient) -> None:
    resp = client.get("/draft")
    assert resp.status_code == 200
    assert resp.json() == {"draft": None, "revision": 0}


def test_get_draft_reflects_current_state(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.get("/draft")
    body = resp.json()
    assert body["revision"] == 1
    assert body["draft"]["plan_name"] == "orm"
    assert body["draft"]["plan_args"] == _ORM_ARGS
    assert body["draft"]["updated_by"] == "agent-1"
    assert "updated_at" in body["draft"]


# ---------------------------------------------------------------------------
# PATCH /draft: creation, merge/remove semantics
# ---------------------------------------------------------------------------


def test_patch_creates_draft_from_plan_name(client: TestClient) -> None:
    resp = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    assert body["plan_name"] == "orm"
    assert sorted(body["changed"]) == sorted(_ORM_ARGS.keys())


def test_patch_merges_into_existing_plan_args(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch("/draft", json={"plan_args_patch": {"num": 5}, "client_id": "agent-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["changed"] == ["num"]
    assert body["revision"] == 2

    full = client.get("/draft").json()["draft"]["plan_args"]
    assert full["num"] == 5
    assert full["correctors"] == ["COR1"]  # untouched keys survive the merge


def test_patch_remove_deletes_keys_and_reports_them_changed(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch("/draft", json={"remove": ["sweep"], "client_id": "agent-1"})
    # "sweep" was never explicitly patched (it has a default), but it exists
    # in the coerced draft's plan_args via full model validation elsewhere;
    # here we only ever store patched/removed keys, so removing a key that
    # isn't present is a true no-op for that key.
    assert resp.status_code == 200

    resp2 = client.patch("/draft", json={"remove": ["num"], "client_id": "agent-1"})
    body2 = resp2.json()
    assert body2["changed"] == ["num"]
    full = client.get("/draft").json()["draft"]["plan_args"]
    assert "num" not in full


def test_plan_name_change_replaces_plan_args(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch(
        "/draft",
        json={
            "plan_name": "grid_scan",
            "plan_args_patch": _GRID_SCAN_ARGS,
            "client_id": "agent-1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan_name"] == "grid_scan"
    # correctors/span_a/num existed under orm and are gone under grid_scan;
    # axes is new; detectors' value is unchanged (["BPM1"] both times).
    assert set(body["changed"]) == {"correctors", "span_a", "num", "axes"}

    full = client.get("/draft").json()["draft"]
    assert full["plan_name"] == "grid_scan"
    assert full["plan_args"] == _GRID_SCAN_ARGS


def test_reaffirming_same_plan_name_is_not_a_replace(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": {"num": 7}, "client_id": "agent-1"},
    )
    body = resp.json()
    assert body["changed"] == ["num"]
    full = client.get("/draft").json()["draft"]["plan_args"]
    assert full["correctors"] == ["COR1"]  # merged, not wiped


# ---------------------------------------------------------------------------
# Per-field validation: constraints, constraint-free fields, coercion
# ---------------------------------------------------------------------------


def test_patch_rejects_empty_detectors_min_length(client: TestClient) -> None:
    resp = client.patch(
        "/draft",
        json={
            "plan_name": "grid_scan",
            "plan_args_patch": {**_GRID_SCAN_ARGS, "detectors": []},
            "client_id": "agent-1",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["field"] == "detectors"


def test_patch_validates_constraint_free_fields_cleanly(client: TestClient) -> None:
    """`snake_axes`/`sweep` carry no `Field()` ge/le/min_length — the
    starred-`Annotated` form would `TypeError` on these; the FieldInfo form
    must validate them cleanly."""
    resp = client.patch(
        "/draft",
        json={
            "plan_name": "grid_scan",
            "plan_args_patch": {**_GRID_SCAN_ARGS, "snake_axes": True},
            "client_id": "agent-1",
        },
    )
    assert resp.status_code == 200

    resp2 = client.patch(
        "/draft",
        json={
            "plan_name": "orm",
            "plan_args_patch": {**_ORM_ARGS, "sweep": "monodirectional"},
            "client_id": "agent-1",
        },
    )
    assert resp2.status_code == 200
    assert client.get("/draft").json()["draft"]["plan_args"]["sweep"] == "monodirectional"


def test_patch_coerces_string_onto_int_and_is_a_noop_when_equal(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    r1 = client.get("/draft").json()["revision"]

    resp = client.patch("/draft", json={"plan_args_patch": {"num": "3"}, "client_id": "agent-1"})
    body = resp.json()
    assert body["changed"] == []
    assert body["revision"] == r1  # no bump: '3' coerces to int 3, which is a no-op

    full = client.get("/draft").json()["draft"]["plan_args"]
    assert full["num"] == 3
    assert isinstance(full["num"], int)


def test_patch_field_constraint_violation_422(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch(
        "/draft",
        json={"plan_args_patch": {"num": 1}, "client_id": "agent-1"},  # ge=3
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["field"] == "num"


# ---------------------------------------------------------------------------
# Unknown plan / unknown field
# ---------------------------------------------------------------------------


def test_unknown_plan_name_422(client: TestClient) -> None:
    resp = client.patch(
        "/draft",
        json={"plan_name": "not_a_real_plan", "client_id": "agent-1"},
    )
    assert resp.status_code == 422
    assert "unknown plan" in str(resp.json()["detail"])


def test_unknown_field_rejected_422(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch(
        "/draft", json={"plan_args_patch": {"bogus_field": 1}, "client_id": "agent-1"}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["field"] == "bogus_field"


# ---------------------------------------------------------------------------
# 409s: no_draft, expected_plan_name mismatch
# ---------------------------------------------------------------------------


def test_patch_with_no_draft_and_no_plan_name_409(client: TestClient) -> None:
    resp = client.patch("/draft", json={"plan_args_patch": {"num": 5}, "client_id": "human-1"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "no_draft"


def test_expected_plan_name_mismatch_409(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp = client.patch(
        "/draft",
        json={
            "plan_args_patch": {"num": 5},
            "expected_plan_name": "grid_scan",
            "client_id": "human-1",
        },
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "plan_name_mismatch"

    # A matching expectation succeeds normally.
    resp2 = client.patch(
        "/draft",
        json={
            "plan_args_patch": {"num": 5},
            "expected_plan_name": "orm",
            "client_id": "human-1",
        },
    )
    assert resp2.status_code == 200


# ---------------------------------------------------------------------------
# No-op suppression + response body shape
# ---------------------------------------------------------------------------


def test_identical_value_patch_is_noop_no_bump(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    r1 = client.get("/draft").json()["revision"]

    resp = client.patch("/draft", json={"plan_args_patch": dict(_ORM_ARGS), "client_id": "agent-1"})
    body = resp.json()
    assert body == {"revision": r1, "changed": [], "plan_name": "orm"}
    assert client.get("/draft").json()["revision"] == r1


@pytest.mark.asyncio
async def test_identical_value_patch_emits_no_frame() -> None:
    """The no-op suppression in `test_identical_value_patch_is_noop_no_bump`
    holds for the SSE frame side too: an identical-value PATCH must not
    broadcast a frame to subscribers, not just skip the revision bump.

    Sequencing note: the sync `client.patch(...)` calls and the
    `await draft._subscribe()` call below never overlap in time (see
    `test_subscribe_hello_is_atomic_with_current_state`'s cross-loop note).
    """
    client = TestClient(app)
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )

    queue, _hello = await draft._subscribe()
    try:
        resp = client.patch(
            "/draft", json={"plan_args_patch": dict(_ORM_ARGS), "client_id": "agent-1"}
        )
        assert resp.json()["changed"] == []
        assert queue.empty()
    finally:
        await draft._unsubscribe(queue)


# ---------------------------------------------------------------------------
# DELETE /draft: idempotence
# ---------------------------------------------------------------------------


def test_delete_on_absent_draft_is_noop(client: TestClient) -> None:
    resp = client.delete("/draft")
    assert resp.status_code == 200
    assert resp.json() == {"revision": 0, "cleared": False}


def test_delete_idempotent_after_real_clear(client: TestClient) -> None:
    client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    resp1 = client.delete("/draft")
    assert resp1.json() == {"revision": 2, "cleared": True}
    assert client.get("/draft").json() == {"draft": None, "revision": 2}

    resp2 = client.delete("/draft")
    assert resp2.json() == {"revision": 2, "cleared": False}


# ---------------------------------------------------------------------------
# Revision monotonicity across clear + plan change
# ---------------------------------------------------------------------------


def test_revision_monotonic_across_clear_and_plan_change(client: TestClient) -> None:
    r0 = client.get("/draft").json()["revision"]
    assert r0 == 0

    r1 = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    ).json()["revision"]
    assert r1 == 1

    r2 = client.patch(
        "/draft",
        json={
            "plan_name": "grid_scan",
            "plan_args_patch": _GRID_SCAN_ARGS,
            "client_id": "agent-1",
        },
    ).json()["revision"]
    assert r2 == 2

    r3 = client.delete("/draft").json()["revision"]
    assert r3 == 3

    # Rebuilding after a clear must never reuse a prior revision.
    r4 = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    ).json()["revision"]
    assert r4 == 4


# ---------------------------------------------------------------------------
# SSE: frame content/order, clear + plan-change frames, atomic hello
# ---------------------------------------------------------------------------


def test_sse_frame_order_and_content_over_real_http() -> None:
    with _live_bridge() as base_url:
        action_client = httpx.Client(base_url=base_url)
        try:
            with httpx.Client(base_url=base_url) as stream_client:
                with stream_client.stream("GET", "/draft/events") as resp:
                    assert resp.status_code == 200
                    assert resp.headers["content-type"].startswith("text/event-stream")
                    lines = resp.iter_lines()

                    hello = _read_frame(lines)
                    assert hello == {"type": "hello", "draft": None, "revision": 0}

                    patch_resp = action_client.patch(
                        "/draft",
                        json={
                            "plan_name": "orm",
                            "plan_args_patch": _ORM_ARGS,
                            "client_id": "agent-1",
                        },
                    )
                    assert patch_resp.status_code == 200
                    frame1 = _read_frame(lines)
                    assert frame1["type"] == "plan-change"
                    assert frame1["revision"] == 1
                    assert frame1["origin"] == "agent-1"
                    assert sorted(frame1["changed"]) == sorted(_ORM_ARGS.keys())
                    assert frame1["draft"]["plan_name"] == "orm"

                    merge_resp = action_client.patch(
                        "/draft", json={"plan_args_patch": {"num": 5}, "client_id": "agent-1"}
                    )
                    assert merge_resp.status_code == 200
                    frame2 = _read_frame(lines)
                    assert frame2["type"] == "change"
                    assert frame2["changed"] == ["num"]
                    assert frame2["revision"] == 2

                    switch_resp = action_client.patch(
                        "/draft",
                        json={
                            "plan_name": "grid_scan",
                            "plan_args_patch": _GRID_SCAN_ARGS,
                            "client_id": "agent-1",
                        },
                    )
                    assert switch_resp.status_code == 200
                    frame3 = _read_frame(lines)
                    assert frame3["type"] == "plan-change"
                    assert frame3["revision"] == 3
                    # The prior orm-only keys (correctors, span_a, num) vanish; axes is new.
                    assert {"correctors", "span_a", "num", "axes"} <= set(frame3["changed"])

                    del_resp = action_client.request("DELETE", "/draft")
                    assert del_resp.status_code == 200
                    frame4 = _read_frame(lines)
                    assert frame4["type"] == "clear"
                    assert frame4["draft"] is None
                    assert frame4["revision"] == 4
                    assert sorted(frame4["changed"]) == sorted(_GRID_SCAN_ARGS.keys())
        finally:
            action_client.close()


@pytest.mark.asyncio
async def test_subscribe_hello_is_atomic_with_current_state() -> None:
    """A subscriber connecting after a mutation gets a hello reflecting it
    exactly once — never a stale hello, and never a duplicate frame for a
    mutation that already happened before it subscribed.

    The sync `client.patch(...)` call and the `await draft._subscribe()`
    call below run on different event loops (`TestClient`'s internal portal
    loop vs. this test's asyncio loop), but never *concurrently* — the
    patch call fully returns before `_subscribe()` starts. That sequencing
    is what keeps `draft.py`'s module-level asyncio primitives (which bind
    lazily to whichever loop first touches them) safe here; genuinely
    overlapping access from two live loops would fail with "attached to a
    different loop".
    """
    client = TestClient(app)
    client.patch(
        "/draft",
        json={"plan_name": "grid_scan", "plan_args_patch": _GRID_SCAN_ARGS, "client_id": "a"},
    )

    queue, hello = await draft._subscribe()
    try:
        assert hello["revision"] == 1
        assert hello["draft"]["plan_name"] == "grid_scan"
        assert queue.empty()
    finally:
        await draft._unsubscribe(queue)


# ---------------------------------------------------------------------------
# Disconnect-on-overflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_disconnects_subscriber_on_full_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(draft, "_QUEUE_MAXSIZE", 2)
    queue, _hello = await draft._subscribe()
    assert queue in draft._subscribers

    async with draft._lock:
        draft._broadcast_locked({"type": "change", "n": 1})
        draft._broadcast_locked({"type": "change", "n": 2})
        # Queue is now full (maxsize=2); this one must trigger disconnect.
        draft._broadcast_locked({"type": "change", "n": 3})

    assert queue not in draft._subscribers

    # The overflowing subscriber is disconnected outright (drained + a single
    # `_DISCONNECT` sentinel) rather than silently dropping just the frame
    # that didn't fit — the client reconnects and gets a fresh hello resync,
    # so the older buffered frames are moot anyway.
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert drained == [draft._DISCONNECT]
