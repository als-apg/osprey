"""Wire contract for the `/runs` surface after in-process execution retired.

Two halves:

- **The read routes** (`GET /runs`, `GET /runs/{id}`) now derive their answers
  from the queue server rather than from any state the bridge keeps. These
  tests drive them through a real `QueueBackend` against a scripted manager, so
  what is pinned is the whole path — backend call, projection, response — not a
  stub of it. (`test_runs.py` pins the projection itself in isolation.)
- **The retired direct-execute routes** (`POST /runs`,
  `POST /runs/{id}/launch`, `POST /draft/run`, `POST /runs/{id}/stop`), which
  answer with the machine-readable `use_the_queue` refusal. Asserting the
  status code alone would let the body drift: a consumer branches on
  `detail.code`, so every test here asserts that code, and the sentence is
  checked for naming the route that took the capability over.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import document_plane, live_rows
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.queue_backend import QueueBackend

# Every retired route, with the method it answers on and the route an operator
# is supposed to use instead. Parametrized rather than repeated so a route
# added to (or dropped from) the refusal set cannot quietly go untested.
_RETIRED_ROUTES = [
    ("post", "/runs", "POST /queue/items"),
    ("post", "/runs/some-run/launch", "POST /queue/start"),
    ("post", "/draft/run", "POST /queue/items"),
    ("post", "/runs/some-run/stop", "POST /queue/stop"),
]


class _ScriptedManager:
    """The smallest `REManagerAPI` stand-in the run routes need.

    They call `queue_get` and `history_get`; anything else reaching this is a
    route doing more work than it should.
    """

    def __init__(self, queue: dict[str, Any], history: dict[str, Any]) -> None:
        self._queue = queue
        self._history = history
        self.calls: list[str] = []

    async def queue_get(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("queue_get")
        if isinstance(self._queue, Exception):
            raise self._queue
        return {"success": True, **self._queue}

    async def history_get(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("history_get")
        if isinstance(self._history, Exception):
            raise self._history
        return {"success": True, **self._history}


def _item(run_id: str, *, name: str = "grid_scan", result: dict | None = None) -> dict:
    item: dict[str, Any] = {
        "item_type": "plan",
        "name": name,
        "kwargs": {"readables": ["BPM1"]},
        "item_uid": f"uid-{run_id}",
        "meta": {"osprey_run_id": run_id},
    }
    if result is not None:
        item["result"] = result
    return item


@pytest.fixture(autouse=True)
def _isolated_state():
    live_rows._clear()
    document_plane._clear()
    yield
    live_rows._clear()
    document_plane._clear()
    app_module.set_queue_backend(None)


@pytest.fixture
def bridge():
    """A `TestClient` on the bridge, with a scripted manager behind the queue."""

    def _build(queue: Any = None, history: Any = None) -> tuple[TestClient, _ScriptedManager]:
        manager = _ScriptedManager(
            queue if queue is not None else {"items": [], "running_item": {}},
            history if history is not None else {"items": []},
        )
        app_module.set_queue_backend(QueueBackend(manager))
        return TestClient(app), manager

    return _build


# =========================================================================
# GET /runs
# =========================================================================


def test_list_runs_projects_the_managers_own_state(bridge) -> None:
    client, manager = bridge(
        queue={"items": [_item("queued")], "running_item": _item("running")},
        history={"items": [_item("done", result={"exit_status": "completed"})]},
    )
    with client:
        body = client.get("/runs").json()

    assert [(r["id"], r["status"]) for r in body] == [
        ("running", "running"),
        ("queued", "pending"),
        ("done", "completed"),
    ]
    assert manager.calls == ["queue_get", "history_get"]


def test_list_runs_is_empty_on_an_idle_deployment(bridge) -> None:
    client, _ = bridge()
    with client:
        resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_runs_passes_the_limit_through(bridge) -> None:
    client, _ = bridge(
        queue={"items": [_item(f"run-{n}") for n in range(5)], "running_item": {}},
    )
    with client:
        body = client.get("/runs?limit=2").json()
    assert [r["id"] for r in body] == ["run-0", "run-1"]


def test_list_runs_reports_an_unreachable_manager_as_503_with_a_code(bridge) -> None:
    """Same refusal shape as the queue surface: a bare-string detail here would
    make this the one route a consumer has to special-case."""
    from bluesky_queueserver_api.comm_base import RequestTimeoutError

    client, _ = bridge(queue=RequestTimeoutError("no answer", {}))
    with client:
        resp = client.get("/runs")

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "manager_unreachable"


# =========================================================================
# GET /runs/{id}
# =========================================================================


def test_get_run_returns_the_record_for_a_known_run(bridge) -> None:
    client, _ = bridge(
        history={
            "items": [_item("done", result={"exit_status": "completed", "run_uids": ["re-1"]})]
        }
    )
    with client:
        resp = client.get("/runs/done")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "done"
    assert body["status"] == "completed"
    assert body["plan_name"] == "grid_scan"
    assert body["plan_args"] == {"readables": ["BPM1"]}
    assert body["run_uid"] == "re-1"


def test_get_run_404s_a_run_the_manager_has_never_heard_of(bridge) -> None:
    client, _ = bridge()
    with client:
        resp = client.get("/runs/never-existed")
    assert resp.status_code == 404


def test_get_run_carries_progress_once_the_document_plane_has_rows(bridge) -> None:
    document_plane.record_expected_points("running", 2)
    recorder = live_rows.LiveRowRecorder(key="running")
    recorder("start", {"uid": "re-uid"})
    recorder("event", {"data": {"BPM1": 1.0}})

    client, _ = bridge(queue={"items": [], "running_item": _item("running")})
    with client:
        body = client.get("/runs/running").json()

    assert body["progress"]["rows_seen"] == 1
    assert body["progress"]["fraction"] == pytest.approx(0.5)


def test_get_run_reports_an_unreachable_manager_as_503_with_a_code(bridge) -> None:
    from bluesky_queueserver_api.comm_base import RequestTimeoutError

    client, _ = bridge(queue=RequestTimeoutError("no answer", {}))
    with client:
        resp = client.get("/runs/anything")

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "manager_unreachable"


# =========================================================================
# The retired direct-execute routes
# =========================================================================


@pytest.mark.parametrize(("method", "path", "replacement"), _RETIRED_ROUTES)
def test_retired_route_refuses_with_the_use_the_queue_code(
    bridge, method: str, path: str, replacement: str
) -> None:
    client, manager = bridge()
    with client:
        resp = getattr(client, method)(path, json={})

    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert detail["code"] == "use_the_queue"
    assert replacement in detail["detail"]
    # A refusal must be decidable without asking the manager anything.
    assert manager.calls == []


@pytest.mark.parametrize(("method", "path", "replacement"), _RETIRED_ROUTES)
def test_a_launch_token_does_not_buy_a_retired_route_back(
    bridge, method: str, path: str, replacement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These routes are gone for everyone. Holding the token that used to arm
    them changes nothing — there is no in-process execution left to arm."""
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "s3cr3t")  # noqa: S105 - test fixture value
    client, _ = bridge()
    with client:
        resp = getattr(client, method)(path, json={}, headers={"X-Launch-Token": "s3cr3t"})

    assert resp.status_code == 410
    assert resp.json()["detail"]["code"] == "use_the_queue"


def test_every_retired_route_still_answers_rather_than_404ing(bridge) -> None:
    """410, not 404: a vanished route tells a caller nothing about where its
    capability went, and this feature ships with existing clients pointed at
    all four of these paths."""
    client, _ = bridge()
    with client:
        for method, path, _ in _RETIRED_ROUTES:
            assert getattr(client, method)(path, json={}).status_code == 410
