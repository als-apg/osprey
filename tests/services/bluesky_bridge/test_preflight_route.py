"""Wire contract for `POST /plans/{name}/preview` — the read-only pre-flight.

The route is TOTAL: it is read by the launch-approval gate, so every way it can
fail to summarize a trajectory is a 200 carrying the same payload shape with a
machine-readable ``reason`` — never a 4xx, never a 5xx. These tests pin that
shape, every reason on the ladder, and the two things the gate renders: the
worker's capped trajectory and this process's own reading of what channels the
plan declared it would move and read.

Driven through a REAL `QueueBackend` over a scripted manager (the stand-in
shape `test_devices_route.py` uses), so what is exercised is the whole path —
the two-step submit-and-poll the backend hides, the route's error mapping, and
the response — rather than a stub of the middle of it. The plan catalog is
faked by monkeypatching `plan_loader.get_facility_plans`, which is where the
route reads it from, so each declaration under test is constructed exactly.
"""

from __future__ import annotations

from typing import Any

import pytest
from bluesky_queueserver_api.comm_base import RequestFailedError, RequestTimeoutError
from fastapi.testclient import TestClient
from pydantic import BaseModel

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import (
    PREVIEW_REASON_CATALOG_UNAVAILABLE,
    PREVIEW_REASON_FAILED,
    PREVIEW_REASON_PLAN_ERROR,
    PREVIEW_REASON_QUEUE_UNREACHABLE,
    PREVIEW_REASON_REFUSED,
    PREVIEW_REASON_TIMED_OUT,
    PREVIEW_REASON_UNKNOWN_PLAN,
    app,
)
from osprey.services.bluesky_bridge.plan_fields import (
    MovableChannel,
    MovableChannels,
    ReadableChannels,
)
from osprey.services.bluesky_bridge.plan_loader import FacilityPlans
from osprey.services.bluesky_bridge.plan_types import PlanSpec
from osprey.services.bluesky_bridge.queue_backend import QueueBackend

TASK_UID = "task-0001"


# ---------------------------------------------------------------------------
# A plan to preview
# ---------------------------------------------------------------------------


class _Axis(BaseModel):
    """One nested axis, so the movable channel is reached as ``axes[].setpoint``."""

    setpoint: MovableChannel
    start: float = 0.0


class _ScanParams(BaseModel):
    axes: list[_Axis]
    monitors: ReadableChannels


class _NoRoleParams(BaseModel):
    """Parameters that declare nothing — the plan whose channels cannot be read."""

    steps: int = 1
    correctors: list[str] = []


class _FlatParams(BaseModel):
    correctors: MovableChannels
    monitors: ReadableChannels


def _spec(name: str, schema: type[BaseModel]) -> PlanSpec[Any]:
    """One registered plan, with the roles the loader would have recorded for it."""
    from osprey.services.bluesky_bridge.plan_fields import channel_roles

    return PlanSpec(
        name=name,
        plan=lambda _devices, _params: None,
        schema=schema,
        roles=tuple(channel_roles(schema)),
    )


_PARAMS = {"axes": [{"setpoint": "COR1"}, {"setpoint": "COR2"}], "monitors": ["BPM1", "BPM2"]}


# ---------------------------------------------------------------------------
# The manager behind the queue
# ---------------------------------------------------------------------------


class _ScriptedManager:
    """The smallest `REManagerAPI` stand-in the pre-flight needs.

    Only ``function_execute`` and ``task_result`` are ever called on this path;
    anything else reaching here is the route doing more than it should. Either
    reply may be an exception, which the manager raises instead of answering,
    and ``task_replies`` is walked one per poll with the last entry repeating —
    so a task that never completes is scripted as a single ``running`` reply.
    """

    def __init__(self, submit: Any, task_replies: list[Any] | None = None) -> None:
        self._submit = submit
        self._task_replies = task_replies or []
        self.items: list[Any] = []
        self.background: list[Any] = []
        self.task_calls: list[str] = []

    async def function_execute(self, *, item: Any, run_in_background: Any = None) -> Any:
        self.items.append(item)
        self.background.append(run_in_background)
        if isinstance(self._submit, Exception):
            raise self._submit
        return self._submit

    async def task_result(self, *, task_uid: str) -> Any:
        self.task_calls.append(task_uid)
        reply = self._task_replies[min(len(self.task_calls) - 1, len(self._task_replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _refused(msg: str) -> RequestFailedError:
    """The manager answering "no" — what upstream's client raises for it."""
    return RequestFailedError({"method": "function_execute"}, {"success": False, "msg": msg})


def _unreachable() -> RequestTimeoutError:
    """The manager not answering at all."""
    return RequestTimeoutError("timeout occurred", {"method": "function_execute"})


def _accepted() -> dict[str, Any]:
    return {"success": True, "msg": "", "task_uid": TASK_UID}


def _completed(return_value: Any, *, success: bool = True, msg: str = "") -> dict[str, Any]:
    return {
        "success": True,
        "status": "completed",
        "task_uid": TASK_UID,
        "result": {
            "task_uid": TASK_UID,
            "success": success,
            "msg": msg,
            "return_value": return_value,
        },
    }


def _running() -> dict[str, Any]:
    return {
        "success": True,
        "status": "running",
        "task_uid": TASK_UID,
        "result": {"task_uid": TASK_UID, "run_in_background": True},
    }


def _summary(**overrides: Any) -> dict[str, Any]:
    """What the worker's pre-flight returns for a plan that built."""
    return {
        "ok": True,
        "plan": "grid_scan",
        "moves": [{"channel": "COR1", "target": 0.5}, {"channel": "COR2", "target": 1.5}],
        "total_moves": 2,
        "truncated": False,
        "move_cap": 10000,
        **overrides,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state():
    yield
    app_module.set_queue_backend(None)


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    """A registry holding one nested-schema plan, one flat, one declaring nothing."""
    plans = FacilityPlans(
        plans={
            "grid_scan": _spec("grid_scan", _ScanParams),
            "flat_scan": _spec("flat_scan", _FlatParams),
            "mystery": _spec("mystery", _NoRoleParams),
        }
    )
    monkeypatch.setattr(plan_loader, "get_facility_plans", lambda: plans)
    return plans


@pytest.fixture
def bridge():
    """A `TestClient` on the bridge, with a scripted manager behind the queue."""

    def _build(submit: Any, task_replies: list[Any] | None = None) -> Any:
        manager = _ScriptedManager(submit, task_replies)
        app_module.set_queue_backend(
            QueueBackend(manager, function_timeout=0.2, function_poll_interval=0.01)
        )
        return TestClient(app), manager

    return _build


def _preview(client: TestClient, name: str = "grid_scan", params: Any = _PARAMS) -> Any:
    """POST the pre-flight and assert the one thing that is true of every answer."""
    response = client.post(f"/plans/{name}/preview", json=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "ok",
        "plan",
        "channels",
        "moves",
        "total_moves",
        "truncated",
        "move_cap",
        "reason",
        "detail",
    }
    return body


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def test_a_summary_carries_the_trajectory_and_the_declared_channels(bridge) -> None:
    client, _manager = bridge(_accepted(), [_completed(_summary())])

    body = _preview(client)

    assert body["ok"] is True
    assert body["plan"] == "grid_scan"
    assert body["reason"] is None and body["detail"] is None
    # The worker's trajectory, relayed in the order the run would make it.
    assert body["moves"] == [
        {"channel": "COR1", "target": 0.5},
        {"channel": "COR2", "target": 1.5},
    ]
    assert body["total_moves"] == 2
    assert body["truncated"] is False
    assert body["move_cap"] == 10000
    # The bridge's own reading of the declaration: the nested `axes[].setpoint`
    # is found without this route knowing the plan's shape, and the channels a
    # launch would DRIVE lead the list.
    assert body["channels"] == [
        {"channel": "COR1", "role": "movable"},
        {"channel": "COR2", "role": "movable"},
        {"channel": "BPM1", "role": "readable"},
        {"channel": "BPM2", "role": "readable"},
    ]


def test_a_flat_declaration_is_read_the_same_way(bridge) -> None:
    client, _manager = bridge(_accepted(), [_completed(_summary())])

    body = _preview(client, "flat_scan", {"correctors": ["COR9"], "monitors": ["BPM9", "BPM9"]})

    # Repeats collapse: a channel named twice is one channel.
    assert body["channels"] == [
        {"channel": "COR9", "role": "movable"},
        {"channel": "BPM9", "role": "readable"},
    ]


def test_a_plan_declaring_no_channel_roles_reports_no_channels(bridge) -> None:
    client, _manager = bridge(_accepted(), [_completed(_summary())])

    body = _preview(client, "mystery", {"correctors": ["COR1"], "steps": 3})

    # `correctors` is a plain list of strings here — undeclared, so unguessed.
    assert body["ok"] is True
    assert body["channels"] == []


def test_the_plan_name_and_its_parameters_reach_the_worker_verbatim(bridge) -> None:
    client, manager = bridge(_accepted(), [_completed(_summary())])

    _preview(client)

    assert manager.items == [
        {
            "item_type": "function",
            "name": "preview_plan",
            "args": ["grid_scan", _PARAMS],
            "kwargs": {},
        }
    ]
    # In the background, which is what lets an operator ask for a summary while
    # the queue is draining.
    assert manager.background == [True]
    assert manager.task_calls == [TASK_UID]


def test_an_empty_body_previews_the_plan_with_no_parameters(bridge) -> None:
    client, manager = bridge(_accepted(), [_completed(_summary())])

    response = client.post("/plans/grid_scan/preview")

    assert response.status_code == 200
    assert manager.items[0]["args"] == ["grid_scan", {}]


def test_a_capped_trajectory_reports_the_exact_total_it_walked(bridge) -> None:
    capped = _summary(
        moves=[{"channel": "COR1", "target": 0.0}],
        total_moves=41337,
        truncated=True,
        move_cap=1,
    )
    client, _manager = bridge(_accepted(), [_completed(capped)])

    body = _preview(client)

    assert body["ok"] is True
    assert body["moves"] == [{"channel": "COR1", "target": 0.0}]
    assert body["truncated"] is True
    # The count is the walk's, not the list's — the whole point of the cap.
    assert body["total_moves"] == 41337
    assert body["move_cap"] == 1


def test_a_slow_pre_flight_is_polled_until_it_answers(bridge) -> None:
    client, manager = bridge(_accepted(), [_running(), _running(), _completed(_summary())])

    body = _preview(client)

    assert body["ok"] is True
    assert len(manager.task_calls) == 3


# ---------------------------------------------------------------------------
# Every reason there is no summary
# ---------------------------------------------------------------------------


def test_an_unknown_plan_is_a_reason_not_an_error(bridge) -> None:
    client, manager = bridge(_accepted(), [_completed(_summary())])

    body = _preview(client, "no_such_plan")

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_UNKNOWN_PLAN
    assert "no_such_plan" in body["detail"]
    assert body["channels"] == []
    # Nothing was asked of the worker for a plan that does not exist.
    assert manager.items == []


def test_an_unreadable_plan_registry_is_a_reason(bridge, monkeypatch) -> None:
    def _explode() -> Any:
        raise RuntimeError("plan directory is gone")

    monkeypatch.setattr(plan_loader, "get_facility_plans", _explode)
    client, manager = bridge(_accepted(), [_completed(_summary())])

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_CATALOG_UNAVAILABLE
    assert "plan directory is gone" in body["detail"]
    assert manager.items == []


def test_a_queue_server_that_does_not_answer_is_a_reason(bridge) -> None:
    client, _manager = bridge(_unreachable())

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_QUEUE_UNREACHABLE


def test_a_queue_connection_that_cannot_be_built_is_the_same_reason(monkeypatch) -> None:
    # The backend is constructed lazily from the environment, so a deployment
    # whose queue settings are malformed fails BEFORE any of `queue_backend`'s
    # typed failures exist to describe it — upstream raises whatever it raises.
    # That is still "no queue server could be asked", not a 500 in an
    # approver's face.
    def _explode() -> Any:
        raise ValueError("Invalid key 'garbage-key': the key must be a 40 byte z85 string")

    monkeypatch.setattr(app_module, "get_queue_backend", _explode)

    body = _preview(TestClient(app))

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_QUEUE_UNREACHABLE
    assert "garbage-key" in body["detail"]
    # The plan is known, so what it declares is still reported.
    assert body["channels"][0] == {"channel": "COR1", "role": "movable"}


def test_a_deployment_with_no_queue_server_is_the_same_reason() -> None:
    # A browse-only deployment holds no manager at all; the pre-flight is one
    # more thing it cannot do, reported rather than raised.
    app_module.set_queue_backend(QueueBackend(None))

    body = _preview(TestClient(app))

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_QUEUE_UNREACHABLE


@pytest.mark.parametrize(
    "msg",
    [
        # The manager refuses an unpermitted function by name ...
        "Function 'preview_plan' is not allowed",
        # ... and refuses any call when the worker cannot take it.
        "RE Worker environment does not exist",
    ],
)
def test_a_refused_pre_flight_relays_the_queue_servers_own_sentence(bridge, msg: str) -> None:
    client, _manager = bridge(_refused(msg))

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_REFUSED
    # The queue server does not tell a denial from a busy worker in anything
    # this process could read, so its own sentence is what the operator gets.
    assert msg in body["detail"]
    # The plan is known, so what it declares is still reported.
    assert body["channels"][0] == {"channel": "COR1", "role": "movable"}


def test_a_pre_flight_that_never_finishes_is_a_reason(bridge) -> None:
    client, manager = bridge(_accepted(), [_running()])

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_TIMED_OUT
    assert body["moves"] == [] and body["total_moves"] == 0
    # Bounded: the budget closed the poll loop rather than the loop running on.
    assert 1 <= len(manager.task_calls) < 200


def test_a_failed_task_is_a_reason(bridge) -> None:
    client, _manager = bridge(
        _accepted(),
        [_completed("Traceback: TypeError somewhere in the worker", success=False)],
    )

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_FAILED
    assert "Traceback" in body["detail"]


def test_a_result_the_queue_server_no_longer_holds_is_a_reason(bridge) -> None:
    client, _manager = bridge(_accepted(), [{"success": True, "status": "not_found", "result": {}}])

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_FAILED


@pytest.mark.parametrize(
    "submit", [{"success": True, "msg": ""}, {"success": True, "task_uid": ""}]
)
def test_an_acceptance_naming_no_task_is_a_failure_not_a_refusal(bridge, submit) -> None:
    client, _manager = bridge(submit, [_completed(_summary())])

    body = _preview(client)

    assert body["ok"] is False
    # The queue server said yes and then left nowhere to read the result from.
    # Reporting that as `preview_refused` would send an operator looking for a
    # permission problem that does not exist.
    assert body["reason"] == PREVIEW_REASON_FAILED


@pytest.mark.parametrize("return_value", ["not a summary", None, 17, []])
def test_an_answer_that_is_not_a_summary_is_a_reason(bridge, return_value: Any) -> None:
    client, _manager = bridge(_accepted(), [_completed(return_value)])

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_FAILED


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "total_moves": 3},
        {"ok": True, "moves": [], "total_moves": "lots"},
        # `bool` is an `int` subclass, and the one that is never a count.
        {"ok": True, "moves": [], "total_moves": True},
    ],
)
def test_a_summary_missing_its_moves_is_a_reason(bridge, payload: dict[str, Any]) -> None:
    client, _manager = bridge(_accepted(), [_completed(payload)])

    body = _preview(client)

    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_FAILED


def test_a_plan_that_does_not_build_carries_the_workers_own_reason(bridge) -> None:
    failed = {
        "ok": False,
        "error": "ValidationError: axes.0.setpoint - Field required",
        "moves": [],
        "total_moves": 0,
        "truncated": False,
    }
    client, _manager = bridge(_accepted(), [_completed(failed)])

    body = _preview(client, params={"monitors": ["BPM1"]})

    assert body["ok"] is False
    # The plan is what failed, not the pre-flight — and the reason is the same
    # one a launch would have failed with.
    assert body["reason"] == PREVIEW_REASON_PLAN_ERROR
    assert body["detail"] == "ValidationError: axes.0.setpoint - Field required"


def test_a_worker_reason_is_bounded_before_it_reaches_an_approver(bridge) -> None:
    client, _manager = bridge(
        _accepted(), [_completed({"ok": False, "error": "x" * 50_000, "moves": []})]
    )

    body = _preview(client)

    assert body["reason"] == PREVIEW_REASON_PLAN_ERROR
    assert len(body["detail"]) == 2000


# ---------------------------------------------------------------------------
# The property the approval gate depends on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("submit", "task_replies", "plan"),
    [
        (_unreachable(), None, "grid_scan"),
        (_refused("refused"), None, "grid_scan"),
        (_accepted(), [_running()], "grid_scan"),
        (_accepted(), [_completed("boom", success=False)], "grid_scan"),
        (_accepted(), [_completed({"ok": False, "error": "no such device"})], "grid_scan"),
        (_accepted(), [_completed("not a summary")], "grid_scan"),
        (_accepted(), [_completed(_summary())], "no_such_plan"),
        (_accepted(), [{"success": True, "status": "not_found", "result": {}}], "grid_scan"),
    ],
)
def test_no_failure_is_ever_a_server_error(bridge, submit, task_replies, plan) -> None:
    """Every way this can go wrong answers 200 in the one payload shape.

    The launch-approval gate renders this response; a gate that errors is a gate
    that stops a human from seeing what they are about to approve.
    """
    client, _manager = bridge(submit, task_replies)

    body = _preview(client, plan)

    assert body["ok"] is False
    assert isinstance(body["reason"], str) and body["reason"]
    assert isinstance(body["detail"], str) and body["detail"]
    assert body["moves"] == []
    assert body["total_moves"] == 0
    assert body["truncated"] is False
    assert body["move_cap"] is None


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        # Valid JSON that is not an object: nothing here is parameters.
        ("[1, 2]", "application/json"),
        ('"hello"', "application/json"),
        ("5", "application/json"),
        ("true", "application/json"),
        # Not JSON at all.
        ("{not json", "application/json"),
        ("axes=COR1&monitors=BPM1", "application/x-www-form-urlencoded"),
    ],
)
def test_a_body_that_is_not_parameters_is_a_reason_not_a_422(
    bridge, content: str, content_type: str
) -> None:
    """A malformed body is answered in the payload shape, like everything else.

    A declared body type would answer these with FastAPI's own 422, whose shape
    carries no ``ok`` at all — and the gate branches on ``ok`` alone, so a 422
    is the one answer it has no branch for.
    """
    client, manager = bridge(_accepted(), [_completed(_summary())])

    response = client.post(
        "/plans/grid_scan/preview", content=content, headers={"content-type": content_type}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is False
    assert body["reason"] == PREVIEW_REASON_PLAN_ERROR
    assert "JSON object" in body["detail"]
    assert body["moves"] == [] and body["total_moves"] == 0
    assert body["truncated"] is False and body["move_cap"] is None
    # Nothing was asked of the worker for a request that carries no parameters.
    assert manager.items == []


@pytest.mark.parametrize("content", ["", "null"])
def test_a_body_carrying_nothing_previews_the_plan_with_no_parameters(bridge, content: str) -> None:
    # An absent body and an explicit `null` both mean "no parameters" — a plan
    # whose parameters all default is previewed by name alone.
    client, manager = bridge(_accepted(), [_completed(_summary())])

    response = client.post(
        "/plans/grid_scan/preview", content=content, headers={"content-type": "application/json"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert manager.items[0]["args"] == ["grid_scan", {}]
