"""Removing one finished run from OSPREY's history view.

The manager can clear its history whole but cannot drop a single entry, so a
per-run removal is an OSPREY-side record (``history_removals.py``) that the
runs surface honours. These tests pin the three things a consumer relies on:

- **The store persists and prunes.** A removal survives a process restart
  (the file is re-read), and an id the manager no longer holds is forgotten
  the next time history is read.
- **The runs surface honours it everywhere.** A removed run is absent from
  ``GET /runs`` and 404s from ``GET /runs/{id}`` — the panel and the agent's
  tools read the same list.
- **Only a finished run can be removed.** A pending or running run answers
  409 with ``run_not_finished`` and names the route that does apply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import history_removals, queue
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.history_removals import RemovedRuns
from osprey.services.bluesky_bridge.queue_backend import QueueBackend


class _ScriptedManager:
    """`queue_get` / `history_get` from fixed documents; records every call."""

    def __init__(self, queue: dict[str, Any], history: dict[str, Any]) -> None:
        self._queue = queue
        self._history = history
        self.calls: list[str] = []

    async def queue_get(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("queue_get")
        return {"success": True, **self._queue}

    async def history_get(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("history_get")
        return {"success": True, **self._history}

    async def status(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("status")
        return {
            "success": True,
            "manager_state": "idle",
            "worker_environment_exists": True,
            "items_in_queue": len(self._queue.get("items") or []),
            "items_in_history": len(self._history.get("items") or []),
            "running_item_uid": None,
            "plan_queue_uid": "q-1",
            "plan_history_uid": "h-1",
            "queue_stop_pending": False,
            "queue_autostart_enabled": False,
        }


def _item(run_id: str, *, result: dict | None = None) -> dict:
    item: dict[str, Any] = {
        "item_type": "plan",
        "name": "grid_scan",
        "kwargs": {},
        "item_uid": f"uid-{run_id}",
        "meta": {"osprey_run_id": run_id},
    }
    if result is not None:
        item["result"] = result
    return item


def _done(run_id: str, exit_status: str = "completed") -> dict:
    return _item(run_id, result={"exit_status": exit_status})


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(tmp_path / "plans_session"))
    history_removals._clear()
    queue._clear()
    yield
    history_removals._clear()
    queue._clear()
    app_module.set_queue_backend(None)


@pytest.fixture
def bridge():
    def _build(
        queue_doc: dict | None = None, history_doc: dict | None = None
    ) -> tuple[TestClient, _ScriptedManager]:
        manager = _ScriptedManager(
            queue_doc if queue_doc is not None else {"items": [], "running_item": {}},
            history_doc if history_doc is not None else {"items": []},
        )
        app_module.set_queue_backend(QueueBackend(manager))
        return TestClient(app), manager

    return _build


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_store_persists_across_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "removed-runs.json"
    store = RemovedRuns(path)
    store.add("run-a")
    store.add("run-b")

    assert json.loads(path.read_text()) == ["run-a", "run-b"]
    reloaded = RemovedRuns(path)
    assert "run-a" in reloaded and "run-b" in reloaded
    assert len(reloaded) == 2


def test_store_prunes_ids_the_history_no_longer_carries(tmp_path: Path) -> None:
    store = RemovedRuns(tmp_path / "removed-runs.json")
    store.add("run-a")
    store.add("run-b")

    store.prune(["run-b", None, "run-c"])

    assert "run-a" not in store
    assert "run-b" in store
    assert json.loads((tmp_path / "removed-runs.json").read_text()) == ["run-b"]


def test_store_starts_empty_on_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "removed-runs.json"
    path.write_text("{not json")
    assert len(RemovedRuns(path)) == 0


# ---------------------------------------------------------------------------
# DELETE /runs/{id} and the runs surface
# ---------------------------------------------------------------------------


def test_removed_run_leaves_the_runs_surface_everywhere(bridge) -> None:
    client, _ = bridge(history_doc={"items": [_done("old"), _done("new", "failed")]})
    with client:
        assert [r["id"] for r in client.get("/runs").json()] == ["new", "old"]

        resp = client.delete("/runs/new")
        assert resp.status_code == 200
        assert resp.json() == {"removed": True, "id": "new"}

        assert [r["id"] for r in client.get("/runs").json()] == ["old"]
        assert client.get("/runs/new").status_code == 404
        assert client.get("/runs/old").status_code == 200
        # The one bridge-owned summary key moved, so SSE subscribers re-read.
        assert client.get("/queue").json()["status"]["runs_removed"] == 1


def test_removing_twice_is_a_404_the_second_time(bridge) -> None:
    client, _ = bridge(history_doc={"items": [_done("r1")]})
    with client:
        assert client.delete("/runs/r1").status_code == 200
        assert client.delete("/runs/r1").status_code == 404


def test_unknown_run_is_404(bridge) -> None:
    client, _ = bridge()
    with client:
        resp = client.delete("/runs/never")
    assert resp.status_code == 404
    assert "never" in resp.json()["detail"]


@pytest.mark.parametrize(
    ("queue_doc", "run_id", "status", "instead"),
    [
        (
            {"items": [_item("waiting")], "running_item": {}},
            "waiting",
            "pending",
            "DELETE /queue/items/{uid}",
        ),
        ({"items": [], "running_item": _item("moving")}, "moving", "running", "POST /queue/abort"),
    ],
    ids=["pending", "running"],
)
def test_a_run_that_has_not_finished_is_refused_with_the_route_that_applies(
    bridge, queue_doc: dict, run_id: str, status: str, instead: str
) -> None:
    client, _ = bridge(queue_doc=queue_doc)
    with client:
        resp = client.delete(f"/runs/{run_id}")
        detail = resp.json()["detail"]
        # Nothing was recorded: the run is still on the surface.
        assert client.get(f"/runs/{run_id}").status_code == 200

    assert resp.status_code == 409
    assert detail["code"] == "run_not_finished"
    assert detail["status"] == status
    assert instead in detail["detail"]
    assert len(history_removals.removed_runs()) == 0


def test_a_removal_survives_a_bridge_restart(bridge) -> None:
    """The store is rebuilt from disk: forgetting the process-level singleton
    stands in for a restart, and the run stays removed."""
    client, _ = bridge(history_doc={"items": [_done("r1"), _done("r2")]})
    with client:
        assert client.delete("/runs/r1").status_code == 200

    history_removals._clear()

    with client:
        assert [r["id"] for r in client.get("/runs").json()] == ["r2"]


def test_a_removal_is_forgotten_once_the_manager_drops_the_run(bridge) -> None:
    """History cleared out-of-band: the id it masked is pruned on the next
    read, so a later run reusing the id would not be born hidden."""
    client, manager = bridge(history_doc={"items": [_done("r1")]})
    with client:
        assert client.delete("/runs/r1").status_code == 200
        assert len(history_removals.removed_runs()) == 1

        manager._history = {"items": []}
        assert client.get("/runs").json() == []

    assert len(history_removals.removed_runs()) == 0
