"""Withdrawing queued work, and the record it leaves.

The manager keeps no record of a removal: a row it drops simply stops existing.
`queue_removals.py` is OSPREY's answer to "who took that off the queue", and
these tests pin what a consumer relies on:

- **The log is bounded, ordered and durable.** Newest first, capped at its own
  constant, re-read from disk after a restart, and starting empty rather than
  failing on a file it cannot parse.
- **A record follows the action, never the request.** The three routes that
  take work off the queue record what the manager actually did — a refused
  removal, an unreadable header and the bridge's own rollback of a just-added
  item all leave the log where it was.
- **An owner-less caller is a first-class one.** Cron and token-only callers
  name nobody, and their withdrawals are recorded with a null owner rather
  than dropped or guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bluesky_queueserver_api.comm_base import RequestFailedError
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import history_removals, queue, queue_removals
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.queue_backend import QueueBackend
from osprey.services.bluesky_bridge.queue_removals import (
    ACTION_ABORT,
    ACTION_CLEAR,
    ACTION_REMOVE,
    QueueRemovals,
    build_record,
)
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG

from .test_queue_routes import FakeManager, _install_abort, status_doc

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    queue._clear()
    queue_removals._clear()
    history_removals._clear()
    app_module.set_queue_backend(None)
    yield
    queue._clear()
    queue_removals._clear()
    history_removals._clear()
    app_module.set_queue_backend(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _install(manager: Any) -> QueueBackend:
    backend = QueueBackend(manager)
    app_module.set_queue_backend(backend)
    return backend


def _item(uid: str, *, name: str = "count_scan", owner: str | None = None) -> dict[str, Any]:
    """A queue item as the manager holds it, owner on the reserved kwarg."""
    item: dict[str, Any] = {
        "item_type": "plan",
        "name": name,
        "kwargs": {"detectors": ["det1"]},
        "item_uid": uid,
        "meta": {"osprey_run_id": f"run-{uid}"},
    }
    if owner is not None:
        item["kwargs"][RESERVED_OWNER_KWARG] = owner
    return item


def _log() -> list[dict[str, Any]]:
    return queue_removals.removal_log().records()


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_a_record_names_the_action_the_owner_and_the_item() -> None:
    record = build_record(ACTION_REMOVE, "anna", _item("u1", owner="bob"))

    assert record["action"] == ACTION_REMOVE
    assert record["owner"] == "anna"
    assert record["uid"] == "u1"
    assert record["name"] == "count_scan"
    assert record["item_type"] == "plan"
    assert record["item_owner"] == "bob"
    assert record["run_id"] == "run-u1"
    assert record["at"].endswith("+00:00")


def test_a_record_never_carries_the_plan_arguments() -> None:
    """Unbounded, and the reserved owner kwarg among them is not a plan
    argument at all — a second place it could leak from is one too many."""
    record = build_record(ACTION_REMOVE, "anna", _item("u1", owner="bob"))

    assert "kwargs" not in record
    assert "det1" not in json.dumps(record)
    assert RESERVED_OWNER_KWARG not in json.dumps(record)


def test_an_item_less_record_reports_nothing_rather_than_guessing() -> None:
    """A clear whose queue read went unanswered: every item key is null, which
    reads as "not recorded" and never as a fabricated row."""
    record = build_record(ACTION_CLEAR, "anna")

    assert record["action"] == ACTION_CLEAR
    assert record["owner"] == "anna"
    assert record["uid"] is None
    assert record["name"] is None
    assert record["item_type"] is None
    assert record["item_owner"] is None
    assert record["run_id"] is None


def test_an_empty_name_is_no_name() -> None:
    """The queue port is the facility's, so an item can arrive from outside
    OSPREY shaped however its enqueuer liked."""
    record = build_record(ACTION_REMOVE, None, {"item_uid": "", "name": "", "item_type": 3})

    assert record["uid"] is None
    assert record["name"] is None
    assert record["item_type"] is None


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_the_log_is_newest_first(tmp_path: Path) -> None:
    store = QueueRemovals(tmp_path / "queue-removals.json")
    store.append(ACTION_REMOVE, "anna", _item("first"))
    store.append(ACTION_REMOVE, "bob", _item("second"))

    assert [record["uid"] for record in store.records()] == ["second", "first"]


def test_the_log_keeps_only_the_newest_records(tmp_path: Path) -> None:
    store = QueueRemovals(tmp_path / "queue-removals.json", max_records=3)
    for index in range(6):
        store.append(ACTION_REMOVE, "anna", _item(f"u{index}"))

    assert [record["uid"] for record in store.records()] == ["u5", "u4", "u3"]
    assert len(store) == 3


def test_the_log_persists_across_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "queue-removals.json"
    store = QueueRemovals(path)
    store.append(ACTION_ABORT, "anna", _item("u1"))

    reloaded = QueueRemovals(path)
    assert [record["uid"] for record in reloaded.records()] == ["u1"]
    assert reloaded.records()[0]["action"] == ACTION_ABORT


def test_a_reload_re_applies_the_bound(tmp_path: Path) -> None:
    """A file written by a build with a larger bound does not raise this one."""
    path = tmp_path / "queue-removals.json"
    path.write_text(
        json.dumps([build_record(ACTION_REMOVE, "anna", _item(f"u{i}")) for i in range(5)])
    )

    assert len(QueueRemovals(path, max_records=2)) == 2


def test_the_log_starts_empty_on_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "queue-removals.json"
    path.write_text("{not json")

    assert QueueRemovals(path).records() == []


def test_a_mangled_entry_costs_itself_and_not_the_log(tmp_path: Path) -> None:
    path = tmp_path / "queue-removals.json"
    path.write_text(json.dumps([build_record(ACTION_REMOVE, "anna", _item("u1")), "junk", None]))

    assert [record["uid"] for record in QueueRemovals(path).records()] == ["u1"]


def test_records_are_copies(tmp_path: Path) -> None:
    store = QueueRemovals(tmp_path / "queue-removals.json")
    store.append(ACTION_REMOVE, "anna", _item("u1"))

    store.records()[0]["owner"] = "mallory"

    assert store.records()[0]["owner"] == "anna"


def test_a_failed_save_leaves_the_log_usable_in_this_process(tmp_path: Path) -> None:
    """Only the restart guarantee is lost; the route still answers success."""
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    store = QueueRemovals(blocked / "queue-removals.json")

    store.append(ACTION_REMOVE, "anna", _item("u1"))

    assert [record["uid"] for record in store.records()] == ["u1"]


# ---------------------------------------------------------------------------
# DELETE /queue/items/{uid}
# ---------------------------------------------------------------------------


def test_a_removal_records_who_asked_for_it(client: TestClient) -> None:
    _install(FakeManager(item_remove={"success": True, "item": _item("u1", owner="bob")}))

    resp = client.delete("/queue/items/u1", headers={OWNER_HEADER: "anna"})

    assert resp.status_code == 200
    (record,) = _log()
    assert record["action"] == ACTION_REMOVE
    assert record["owner"] == "anna"
    assert record["uid"] == "u1"
    assert record["name"] == "count_scan"
    assert record["item_owner"] == "bob"
    assert record["run_id"] == "run-u1"


def test_an_owner_less_removal_is_recorded_with_a_null_owner(client: TestClient) -> None:
    """Cron and token-only callers name nobody, and their withdrawals are
    still the deployment's to know about."""
    _install(FakeManager(item_remove={"success": True, "item": _item("u1")}))

    assert client.delete("/queue/items/u1").status_code == 200

    (record,) = _log()
    assert record["owner"] is None
    assert record["uid"] == "u1"


def test_a_header_the_reader_refuses_costs_the_name_and_not_the_removal(
    client: TestClient,
) -> None:
    """The allowlist is the single reader of this name everywhere; a value it
    will not accept is recorded as nobody, and the item still goes."""
    _install(FakeManager(item_remove={"success": True, "item": _item("u1")}))

    resp = client.delete("/queue/items/u1", headers={OWNER_HEADER: "not a name/../"})

    assert resp.status_code == 200
    assert resp.json()["removed"] is True
    (record,) = _log()
    assert record["owner"] is None


def test_a_refused_removal_records_nothing(client: TestClient) -> None:
    """The log says what happened to the queue, and this did not happen."""
    _install(FakeManager(item_remove=RequestFailedError({}, {"msg": "no such item"})))

    assert client.delete("/queue/items/u1", headers={OWNER_HEADER: "anna"}).status_code == 409

    assert _log() == []


async def test_the_rollback_of_a_just_added_item_records_nothing() -> None:
    """The unarmed enqueue's post-add re-check removes the item it added. That
    is the request undoing itself, not an operator withdrawing work: the caller
    is told the enqueue was refused and never had an item to withdraw."""
    backend = _install(FakeManager(item_remove={"success": True, "item": _item("u1")}))

    assert await queue._remove_item_best_effort(backend, "u1") is True

    assert _log() == []


# ---------------------------------------------------------------------------
# DELETE /queue/items (clear)
# ---------------------------------------------------------------------------


def test_a_clear_records_one_entry_per_item_it_dropped(client: TestClient) -> None:
    _install(
        FakeManager(
            queue_get={
                "success": True,
                "items": [_item("u1", name="count_scan"), _item("u2", name="grid_scan")],
                "running_item": {},
            },
            queue_clear={"success": True, "msg": "cleared"},
        )
    )

    assert client.delete("/queue/items", headers={OWNER_HEADER: "anna"}).status_code == 200

    records = _log()
    assert [record["name"] for record in records] == ["grid_scan", "count_scan"]
    assert {record["action"] for record in records} == {ACTION_CLEAR}
    assert {record["owner"] for record in records} == {"anna"}


def test_clearing_an_empty_queue_records_nothing(client: TestClient) -> None:
    """Nothing was dropped, so nothing was withdrawn."""
    _install(
        FakeManager(
            queue_get={"success": True, "items": [], "running_item": {}},
            queue_clear={"success": True, "msg": ""},
        )
    )

    assert client.delete("/queue/items", headers={OWNER_HEADER: "anna"}).status_code == 200

    assert _log() == []


def test_a_clear_whose_read_went_unanswered_is_recorded_bare(client: TestClient) -> None:
    """The read is the log's, not the clear's: losing it costs the detail."""
    _install(
        FakeManager(
            queue_get=RequestFailedError({}, {"msg": "nope"}),
            queue_clear={"success": True, "msg": "cleared"},
        )
    )

    assert client.delete("/queue/items", headers={OWNER_HEADER: "anna"}).status_code == 200

    (record,) = _log()
    assert record["action"] == ACTION_CLEAR
    assert record["owner"] == "anna"
    assert record["uid"] is None


def test_a_refused_clear_records_nothing(client: TestClient) -> None:
    _install(
        FakeManager(
            queue_get={"success": True, "items": [_item("u1")], "running_item": {}},
            queue_clear=RequestFailedError({}, {"msg": "nope"}),
        )
    )

    assert client.delete("/queue/items", headers={OWNER_HEADER: "anna"}).status_code == 409

    assert _log() == []


# ---------------------------------------------------------------------------
# POST /queue/abort
# ---------------------------------------------------------------------------


def test_an_abort_records_who_stopped_what(client: TestClient) -> None:
    _install_abort(
        FakeManager(
            status=status_doc(manager_state="paused", running_item_uid="u1"),
            queue_get={
                "success": True,
                "items": [],
                "running_item": _item("u1", owner="bob"),
            },
            re_abort={"success": True, "msg": "aborted"},
        )
    )

    resp = client.post("/queue/abort", headers={OWNER_HEADER: "anna"})

    assert resp.status_code == 200
    (record,) = _log()
    assert record["action"] == ACTION_ABORT
    assert record["owner"] == "anna"
    assert record["uid"] == "u1"
    assert record["name"] == "count_scan"
    assert record["item_owner"] == "bob"


def test_the_plan_is_read_after_the_pause_and_before_the_discard(
    client: TestClient,
) -> None:
    """Where the read sits is the safety property, so it is asserted by call
    order. The pause holds the hardware; everything after it delays the unwind
    and not the stop, and the item is still on the manager until ``re_abort``.
    """
    manager = FakeManager(
        status=[
            status_doc(manager_state="executing_queue"),
            status_doc(manager_state="paused"),
            status_doc(manager_state="idle"),
        ],
        queue_get={"success": True, "items": [], "running_item": _item("u1")},
        re_abort={"success": True, "msg": "aborted"},
    )
    _install_abort(manager)

    assert client.post("/queue/abort", headers={OWNER_HEADER: "anna"}).status_code == 200

    names = manager.method_names()
    assert names.index("re_pause") < names.index("queue_get") < names.index("re_abort")
    assert _log()[0]["uid"] == "u1"


def test_nothing_running_issues_no_read_and_records_nothing(client: TestClient) -> None:
    """Nothing was stopped, so nothing is recorded and nothing is asked."""
    manager = FakeManager(
        status=status_doc(manager_state="idle"),
        queue_get={"success": True, "items": [], "running_item": {}},
    )
    _install_abort(manager)

    assert client.post("/queue/abort", headers={OWNER_HEADER: "anna"}).status_code == 409

    assert "queue_get" not in manager.method_names()
    assert _log() == []


def test_a_pause_that_never_lands_issues_no_read_and_records_nothing(
    client: TestClient,
) -> None:
    """NOTHING WAS ABORTED on this path — the plan may still be running — so
    there is no stop to attribute, and the read that would name one is never
    reached."""
    manager = FakeManager(
        status=status_doc(manager_state="executing_queue"),
        queue_get={"success": True, "items": [], "running_item": _item("u1")},
    )
    _install_abort(manager)

    resp = client.post("/queue/abort", headers={OWNER_HEADER: "anna"})

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "abort_pause_timeout"
    assert "queue_get" not in manager.method_names()
    assert "re_abort" not in manager.method_names()
    assert _log() == []


def test_an_abort_whose_read_is_refused_still_halts_and_records(
    client: TestClient,
) -> None:
    """The read names what is being stopped; it can never stop the halt."""
    manager = FakeManager(
        status=status_doc(manager_state="paused", running_item_uid="u1"),
        queue_get=RequestFailedError({}, {"msg": "nope"}),
        re_abort={"success": True, "msg": "aborted"},
    )
    _install_abort(manager)

    resp = client.post("/queue/abort", headers={OWNER_HEADER: "anna"})

    assert resp.status_code == 200
    assert resp.json()["aborted"] is True
    assert "re_abort" in manager.method_names()
    (record,) = _log()
    assert record["action"] == ACTION_ABORT
    assert record["owner"] == "anna"
    assert record["name"] is None


def test_the_abort_response_carries_no_plan_arguments(client: TestClient) -> None:
    """The stopped item feeds the record and reaches no client: it is the
    manager's own object, reserved owner kwarg and all."""
    manager = FakeManager(
        status=status_doc(manager_state="paused", running_item_uid="u1"),
        queue_get={"success": True, "items": [], "running_item": _item("u1", owner="bob")},
        re_abort={"success": True, "msg": "aborted"},
    )
    _install_abort(manager)

    body = client.post("/queue/abort", headers={OWNER_HEADER: "anna"}).json()

    assert set(body) == {"aborted", "abort_pending", "paused_first", "manager_state", "msg"}
    assert "det1" not in json.dumps(body)
    assert RESERVED_OWNER_KWARG not in json.dumps(body)
    # Non-vacuity: the record the response was stripped for did get the item.
    assert _log()[0]["item_owner"] == "bob"


# ---------------------------------------------------------------------------
# GET /queue/removals, and what empties it
# ---------------------------------------------------------------------------


def test_the_read_serves_the_log_newest_first(client: TestClient) -> None:
    _install(
        FakeManager(
            item_remove=[
                {"success": True, "item": _item("u1", name="count_scan")},
                {"success": True, "item": _item("u2", name="grid_scan")},
            ]
        )
    )

    client.delete("/queue/items/u1", headers={OWNER_HEADER: "anna"})
    client.delete("/queue/items/u2", headers={OWNER_HEADER: "bob"})

    body = client.get("/queue/removals").json()
    assert [record["name"] for record in body] == ["grid_scan", "count_scan"]
    assert [record["owner"] for record in body] == ["bob", "anna"]


def test_the_read_answers_while_the_manager_is_down(client: TestClient) -> None:
    """The log is the bridge's own; serving it touches no manager."""
    _install(FakeManager(status=RequestFailedError({}, {"msg": "gone"})))

    assert client.get("/queue/removals").status_code == 200
    assert client.get("/queue/removals").json() == []


def test_a_removal_moves_the_summary_so_subscribers_re_read(client: TestClient) -> None:
    _install(
        FakeManager(
            status=status_doc(),
            queue_get={"success": True, "items": [], "running_item": {}},
            item_remove={"success": True, "item": _item("u1")},
        )
    )
    with client:
        assert client.get("/queue").json()["status"]["queue_removals"] == 0

        assert client.delete("/queue/items/u1").status_code == 200

        assert client.get("/queue").json()["status"]["queue_removals"] == 1


def test_clearing_history_empties_the_withdrawals_listed_with_it(client: TestClient) -> None:
    """One control, one list: the panel lists both under History, so Clear
    must not leave half of it on screen."""
    _install(
        FakeManager(
            item_remove={"success": True, "item": _item("u1")},
            history_clear={"success": True, "msg": "cleared"},
        )
    )

    client.delete("/queue/items/u1", headers={OWNER_HEADER: "anna"})
    assert len(_log()) == 1

    assert client.delete("/history").status_code == 200

    assert _log() == []
    assert client.get("/queue/removals").json() == []
