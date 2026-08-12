"""Wire contract for `GET /devices`, the worker's device-name surface.

A plan's device parameters are strings resolved against the queueserver
worker's namespace, so the set of names that resolve is a thing callers have to
be able to READ — otherwise a wrong name is only discoverable by enqueuing,
starting, and watching the run fail on its first iteration. These tests pin
what that route publishes: the names the manager reports, the protocol flags
that tell a drivable setpoint from a read-only detector, and nothing else.

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


def test_lists_every_device_the_worker_built_by_name(bridge) -> None:
    client, manager = bridge(
        _allowed(
            COR1={"is_movable": True, "is_readable": True, "is_flyable": False},
            BPM1={"is_movable": False, "is_readable": True, "is_flyable": False},
        )
    )

    resp = client.get("/devices")

    assert resp.status_code == 200
    # Sorted by name: the answer is a lookup table, and a stable order is what
    # lets a caller (and a diff) find a name in it.
    assert resp.json() == [
        {"name": "BPM1", "is_movable": False, "is_readable": True, "is_flyable": False},
        {"name": "COR1", "is_movable": True, "is_readable": True, "is_flyable": False},
    ]
    assert manager.calls == ["devices_allowed"]


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

    assert client.get("/devices").json() == [
        {"name": "COR1", "is_movable": True, "is_readable": True, "is_flyable": False}
    ]


def test_a_flag_the_manager_did_not_report_stays_absent(bridge) -> None:
    """A flag the manager did not report is not a "no" — a fabricated `False`
    would read as a device that cannot be driven, a different claim entirely."""
    client, _ = bridge(_allowed(COR1={"is_movable": True}))

    assert client.get("/devices").json() == [{"name": "COR1", "is_movable": True}]


def test_a_device_the_manager_described_with_a_non_mapping_still_lists(bridge) -> None:
    """The name is the load-bearing half of every entry, so a description this
    route cannot read costs the flags, never the device. Dropping the entry
    would tell a caller the worker lacks a device it has — the one wrong answer
    this route must not give."""
    client, _ = bridge({"success": True, "devices_allowed": {"COR1": None, "BPM1": "readable"}})

    assert client.get("/devices").json() == [{"name": "BPM1"}, {"name": "COR1"}]


def test_a_worker_with_no_devices_answers_with_an_empty_list(bridge) -> None:
    client, _ = bridge(_allowed())

    resp = client.get("/devices")

    assert resp.status_code == 200
    assert resp.json() == []


def test_a_reply_carrying_no_device_map_is_empty_not_an_error(bridge) -> None:
    """A manager that answered without the key has no devices to report; that
    is an empty set, not a failure to report one."""
    client, _ = bridge({"success": True})

    assert client.get("/devices").json() == []


def test_an_unreachable_manager_is_a_503_carrying_the_code(bridge) -> None:
    """Same refusal shape as every other manager-backed read — a consumer
    branches on `detail.code` here exactly as it does on `/runs`."""
    client, _ = bridge(RequestTimeoutError("no answer", {}))

    resp = client.get("/devices")

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "manager_unreachable"
