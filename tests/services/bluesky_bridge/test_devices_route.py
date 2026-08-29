"""Wire contract for `GET /devices`, the worker's bounded device-name surface.

A plan's device parameters are strings resolved against the queueserver
worker's namespace, so the set of names that resolve is a thing callers have to
be able to READ — otherwise a wrong name is only discoverable by enqueuing,
starting, and watching the run fail on its first iteration. These tests pin
what that route publishes: a page of the names the manager reports, the
`total` that says whether the page is all of them, the protocol flags that tell
a drivable setpoint from a read-only detector, and nothing else.

They also pin that `prefix`/`limit`/`offset` are clamped rather than rejected.
A caller reaching this route is usually an agent narrowing a namespace it does
not know the size of; a 422 teaches it to fear the parameters, while a clamped
page answers the question it was actually asking.

Driven through a real `QueueBackend` over a scripted manager (same stand-in
shape as `test_run_routes.py`'s), so what is pinned is the whole path — backend
call, projection, response — rather than a stub of the projection.
"""

from __future__ import annotations

from typing import Any

import pytest
from bluesky_queueserver_api.comm_base import RequestTimeoutError
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import queue
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.queue_backend import QueueBackend


class _ScriptedManager:
    """The smallest `REManagerAPI` stand-in this route needs.

    Only `devices_allowed` is ever called on this path; anything else reaching
    here is the route doing more work than it should. Pass an exception to have
    the manager raise instead of answering.
    """

    def __init__(self, reply: Any) -> None:
        self._reply = reply
        self.calls: list[str] = []

    async def devices_allowed(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("devices_allowed")
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


@pytest.fixture(autouse=True)
def _isolated_state():
    yield
    app_module.set_queue_backend(None)


@pytest.fixture
def bridge():
    """A `TestClient` on the bridge, with a scripted manager behind the queue."""

    def _build(reply: Any) -> tuple[TestClient, _ScriptedManager]:
        manager = _ScriptedManager(reply)
        app_module.set_queue_backend(QueueBackend(manager))
        return TestClient(app), manager

    return _build


def _allowed(**devices: Any) -> dict[str, Any]:
    return {"success": True, "devices_allowed": devices}


def _names(payload: dict[str, Any]) -> list[str]:
    return [entry["name"] for entry in payload["devices"]]


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_lists_every_device_the_worker_built_by_name(bridge) -> None:
    """FR-1: an unparameterised read of a namespace smaller than the page size
    is the whole namespace, and says so — `total` equal to the entries
    returned is how a caller knows it does not have to walk."""
    client, manager = bridge(
        _allowed(
            COR1={"is_movable": True, "is_readable": True, "is_flyable": False},
            BPM1={"is_movable": False, "is_readable": True, "is_flyable": False},
        )
    )

    resp = client.get("/devices")

    assert resp.status_code == 200
    # Sorted by name: the answer is a lookup table, and a stable order is what
    # lets a caller (and a diff) find a name in it — and what makes `offset`
    # mean the same thing across two requests.
    assert resp.json() == {
        "devices": [
            {"name": "BPM1", "is_movable": False, "is_readable": True, "is_flyable": False},
            {"name": "COR1", "is_movable": True, "is_readable": True, "is_flyable": False},
        ],
        "total": 2,
        "offset": 0,
        "limit": queue.DEFAULT_DEVICE_PAGE_SIZE,
    }
    assert manager.calls == ["devices_allowed"]


def test_a_page_reports_the_total_it_was_cut_from(bridge) -> None:
    """FR-2: `limit`/`offset` cut a window out of the sorted names, and `total`
    keeps describing the whole match set — the page is what changes, not the
    count the caller pages against."""
    client, _ = bridge(_allowed(**{name: {} for name in ("D1", "D2", "D3", "D4", "D5")}))

    payload = client.get("/devices", params={"limit": 2, "offset": 2}).json()

    assert _names(payload) == ["D3", "D4"]
    assert payload["total"] == 5
    assert payload["offset"] == 2
    assert payload["limit"] == 2


def test_an_offset_past_the_end_is_an_empty_page_not_an_error(bridge) -> None:
    """Walking off the end is how a caller learns the walk is done; a refusal
    there would make the last page indistinguishable from a broken request."""
    client, _ = bridge(_allowed(D1={}, D2={}))

    resp = client.get("/devices", params={"offset": 9})

    assert resp.status_code == 200
    assert resp.json()["devices"] == []
    assert resp.json()["total"] == 2


# ---------------------------------------------------------------------------
# prefix
# ---------------------------------------------------------------------------


def test_prefix_narrows_to_the_names_that_start_with_it(bridge) -> None:
    """FR-3: `total` follows the filter — it counts what matched, not what the
    worker holds, so a caller pages against the set it asked for."""
    client, _ = bridge(_allowed(COR1={}, COR2={}, BPM1={}))

    payload = client.get("/devices", params={"prefix": "COR"}).json()

    assert _names(payload) == ["COR1", "COR2"]
    assert payload["total"] == 2


def test_prefix_is_case_sensitive_and_a_miss_is_an_empty_page(bridge) -> None:
    """FR-3: device names come out of the worker's namespace verbatim, so
    folding case would offer matches that resolve to nothing when a plan names
    them. A prefix nothing starts with is a 200 with no devices — an empty set
    is an answer, and a 404 would read as "the route is missing"."""
    client, _ = bridge(_allowed(COR1={}, COR2={}))

    resp = client.get("/devices", params={"prefix": "cor"})

    assert resp.status_code == 200
    assert resp.json()["devices"] == []
    assert resp.json()["total"] == 0


def test_prefix_keeps_the_flag_keys_on_every_entry_it_returns(bridge) -> None:
    """Filtering and paging change which entries come back, never their shape."""
    client, _ = bridge(_allowed(COR1={"is_movable": True, "is_readable": True}, BPM1={}))

    payload = client.get("/devices", params={"prefix": "COR"}).json()

    assert payload["devices"] == [{"name": "COR1", "is_movable": True, "is_readable": True}]


# ---------------------------------------------------------------------------
# Clamping — never a 422
# ---------------------------------------------------------------------------


def test_clamp_pulls_a_limit_below_one_up_to_a_single_entry(bridge) -> None:
    """FR-4: `limit=0` asks for a page that answers nothing. Clamping to one
    entry keeps the same posture as `runs.list_records`, and the echoed `limit`
    tells the caller which bound it actually got."""
    client, _ = bridge(_allowed(D1={}, D2={}, D3={}))

    resp = client.get("/devices", params={"limit": 0})

    assert resp.status_code == 200
    assert _names(resp.json()) == ["D1"]
    assert resp.json()["limit"] == 1


def test_clamp_holds_a_limit_above_the_cap_down_to_the_cap(bridge, monkeypatch) -> None:
    """FR-4: the page size is the bound this route will not exceed, so asking
    past it cannot turn the route into the way to pull a whole namespace."""
    monkeypatch.setenv(queue.DEVICE_PAGE_SIZE_ENV, "3")
    client, _ = bridge(_allowed(**{f"DEV{index}": {} for index in range(6)}))

    payload = client.get("/devices", params={"limit": 8}).json()

    assert _names(payload) == ["DEV0", "DEV1", "DEV2"]
    assert payload["limit"] == 3
    assert payload["total"] == 6


def test_clamp_lifts_a_negative_offset_to_the_start(bridge) -> None:
    """FR-4: a negative offset would slice from the END in Python. Clamping to
    0 answers from the start, which is what "before the beginning" means."""
    client, _ = bridge(_allowed(D1={}, D2={}, D3={}))

    resp = client.get("/devices", params={"offset": -3})

    assert resp.status_code == 200
    assert _names(resp.json()) == ["D1", "D2", "D3"]
    assert resp.json()["offset"] == 0


# ---------------------------------------------------------------------------
# The cap comes from the environment, per request
# ---------------------------------------------------------------------------


def test_cap_from_the_environment_bounds_an_unparameterised_read(bridge, monkeypatch) -> None:
    """FR-5: with no `limit`, the page size IS the limit — a facility that
    lowers it lowers what this route hands out, and `total` still reports the
    namespace behind the page so the caller knows to walk."""
    monkeypatch.setenv(queue.DEVICE_PAGE_SIZE_ENV, "3")
    client, _ = bridge(_allowed(**{f"DEV{index}": {} for index in range(4)}))

    payload = client.get("/devices").json()

    assert _names(payload) == ["DEV0", "DEV1", "DEV2"]
    assert payload["total"] == 4
    assert payload["offset"] == 0
    assert payload["limit"] == 3


# ---------------------------------------------------------------------------
# Entry shape
# ---------------------------------------------------------------------------


def test_worker_internal_description_detail_stays_off_the_wire(bridge) -> None:
    """`classname`/`module`/`components` describe the object, not the name a
    caller can use — publishing them would only invite an agent to reason about
    a worker's internals it cannot address."""
    client, _ = bridge(
        _allowed(
            COR1={
                "is_movable": True,
                "is_readable": True,
                "is_flyable": False,
                "classname": "EpicsMotor",
                "module": "ophyd.epics_motor",
                "components": {"user_setpoint": {}},
            }
        )
    )

    assert client.get("/devices").json()["devices"] == [
        {"name": "COR1", "is_movable": True, "is_readable": True, "is_flyable": False}
    ]


def test_a_flag_the_manager_did_not_report_stays_absent(bridge) -> None:
    """A flag the manager did not report is not a "no" — a fabricated `False`
    would read as a device that cannot be driven, a different claim entirely."""
    client, _ = bridge(_allowed(COR1={"is_movable": True}))

    assert client.get("/devices").json()["devices"] == [{"name": "COR1", "is_movable": True}]


def test_a_device_the_manager_described_with_a_non_mapping_still_lists(bridge) -> None:
    """The name is the load-bearing half of every entry, so a description this
    route cannot read costs the flags, never the device. Dropping the entry
    would tell a caller the worker lacks a device it has — the one wrong answer
    this route must not give."""
    client, _ = bridge({"success": True, "devices_allowed": {"COR1": None, "BPM1": "readable"}})

    assert client.get("/devices").json()["devices"] == [{"name": "BPM1"}, {"name": "COR1"}]


# ---------------------------------------------------------------------------
# Nothing to report, and nobody to report it
# ---------------------------------------------------------------------------


def test_a_worker_with_no_devices_answers_with_an_empty_page(bridge) -> None:
    client, _ = bridge(_allowed())

    resp = client.get("/devices")

    assert resp.status_code == 200
    assert resp.json() == {
        "devices": [],
        "total": 0,
        "offset": 0,
        "limit": queue.DEFAULT_DEVICE_PAGE_SIZE,
    }


def test_a_reply_carrying_no_device_map_is_empty_not_an_error(bridge) -> None:
    """A manager that answered without the key has no devices to report; that
    is an empty set, not a failure to report one — and it still comes back in
    the envelope, so a caller never has to branch on the response's shape."""
    client, _ = bridge({"success": True})

    payload = client.get("/devices").json()

    assert payload["devices"] == []
    assert payload["total"] == 0


def test_an_unreachable_manager_is_a_503_carrying_the_code(bridge) -> None:
    """Same refusal shape as every other manager-backed read — a consumer
    branches on `detail.code` here exactly as it does on `/runs`."""
    client, _ = bridge(RequestTimeoutError("no answer", {}))

    resp = client.get("/devices")

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "manager_unreachable"
