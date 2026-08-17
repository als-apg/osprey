"""Coverage for the bridge's queue surface (`queue.py`): REST routes, the FR4
arming policy (token-gated start, token-gated enqueue-while-active, one asyncio
lock + post-add re-check), and the `GET /queue/events` SSE stream.

Everything runs against the real `QueueBackend` over a scripted `FakeManager`
stand-in (same shape as `test_queue_backend.py`'s) — these tests pin the ROUTE
layer's policy: which manager calls each route makes, which HTTP refusal each
typed backend failure becomes, and that the arming lock + re-check make the
enqueue gate race-free. The race tests drive the route coroutines directly on
one event loop with event-gated fake-manager calls, so the interleavings are
deterministic, not sleep-and-hope.

Draft-side state is real too (`draft.py`'s module singletons): enqueue consumes
`check_launchable`'s once-per-revision reservation exactly like the launch flow
it replaces, and these tests assert the reservation is released on every
refusal path and consumed on success.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from bluesky_queueserver_api.comm_base import RequestFailedError, RequestTimeoutError
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import draft, plan_loader, queue
from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.plan_fields import (
    MovableChannel,
    MovableChannels,
    ReadableChannels,
    channel_roles,
)
from osprey.services.bluesky_bridge.plan_types import PlanSpec
from osprey.services.bluesky_bridge.queue_backend import QueueBackend
from osprey.services.bluesky_bridge.session_upload import SessionPlanNotReadyError

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"
_TOKEN = "s3cr3t"

# A valid draft for the always-registered shipped `grid_scan` plan
# (mirrors `test_draft.py`).
_GRID_SCAN_ARGS: dict[str, Any] = {
    "readables": ["BPM1"],
    "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
}


class FakeManager:
    """A scripted ``REManagerAPI`` stand-in (same contract as `test_queue_backend.py`'s).

    Records every call as ``(method, kwargs)`` and answers from ``responses``;
    an exception instance is raised instead of returned; a list is consumed one
    call at a time with the tail repeating.
    """

    def __init__(self, **responses: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses
        self.closed = False

    def _answer(self, method: str) -> Any:
        response = self.responses.get(method, {"success": True, "msg": ""})
        if isinstance(response, list):
            value = response[0]
            if len(response) > 1:
                response.pop(0)
        else:
            value = response
        if isinstance(value, Exception):
            raise value
        return value

    def __getattr__(self, method: str) -> Any:
        async def call(**kwargs: Any) -> Any:
            self.calls.append((method, kwargs))
            return self._answer(method)

        return call

    async def close(self) -> None:
        self.closed = True

    def kwargs_for(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def method_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def status_doc(**overrides: Any) -> dict[str, Any]:
    """A manager status document: idle, environment open, nothing running."""
    doc = {
        "success": True,
        "manager_state": "idle",
        "worker_environment_exists": True,
        "items_in_queue": 0,
        "items_in_history": 0,
        "running_item_uid": None,
        "plan_queue_uid": "q-1",
        "plan_history_uid": "h-1",
        "queue_stop_pending": False,
        "queue_autostart_enabled": False,
    }
    doc.update(overrides)
    return doc


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Mirror `test_draft.py`'s isolation, plus the queue module's own state.

    The backend singleton lives in `app.py`; clearing it to `None` here means
    a test that forgets to install a fake would lazily build a managerless
    real backend (fail-closed 503s) rather than seeing another test's fake.
    """
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.delenv(_TOKEN_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    app_module.set_queue_backend(None)
    yield
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    app_module.set_queue_backend(None)


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch):
    """Set (or break) the ``control_system.type`` the capability check resolves."""

    def _set(value: str | Exception) -> None:
        def fake_get_config_value(key: str, default: Any = None) -> Any:
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

    return _set


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _install(manager: Any) -> QueueBackend:
    backend = QueueBackend(manager)
    app_module.set_queue_backend(backend)
    return backend


def _make_draft(client: TestClient) -> int:
    resp = client.patch(
        "/draft",
        json={"plan_name": "grid_scan", "plan_args_patch": _GRID_SCAN_ARGS, "client_id": "test"},
    )
    assert resp.status_code == 200
    return resp.json()["revision"]


async def _make_draft_direct() -> int:
    result = await draft.patch_draft(
        draft.PatchDraftRequest(
            plan_name="grid_scan", plan_args_patch=_GRID_SCAN_ARGS, client_id="test"
        )
    )
    return result["revision"]


# ---------------------------------------------------------------------------
# GET /queue
# ---------------------------------------------------------------------------


def test_get_queue_reports_items_running_item_and_status_summary(client: TestClient) -> None:
    _install(
        FakeManager(
            status=status_doc(items_in_queue=1),
            queue_get={"success": True, "items": [{"item_uid": "a"}], "running_item": {}},
        )
    )

    resp = client.get("/queue")

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == [{"item_uid": "a"}]
    # qserver reports "no running item" as {} — the wire shape is null.
    assert body["running_item"] is None
    assert body["status"]["available"] is True
    assert body["status"]["manager_state"] == "idle"
    assert body["status"]["items_in_queue"] == 1
    # Autostart is observed on the wire, never assumed off.
    assert body["status"]["queue_autostart_enabled"] is False


def test_get_queue_maps_manager_silence_to_503(client: TestClient) -> None:
    _install(FakeManager(status=RequestTimeoutError("no answer", {})))

    resp = client.get("/queue")

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == qb.REASON_MANAGER_UNREACHABLE


# ---------------------------------------------------------------------------
# POST /queue/items — enqueue-from-pinned-draft
# ---------------------------------------------------------------------------


def test_enqueue_on_an_idle_queue_is_ungated_and_threads_the_run_id(
    client: TestClient, connector
) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(), item_add={"success": True, "item": {"item_uid": "u1"}}
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == revision
    assert body["item"] == {"item_uid": "u1"}

    (add_kwargs,) = manager.kwargs_for("item_add")
    assert add_kwargs["item"]["item_type"] == "plan"
    assert add_kwargs["item"]["name"] == "grid_scan"
    assert add_kwargs["item"]["kwargs"] == _GRID_SCAN_ARGS
    # The OSPREY run id and the plan's identity ride the item metadata into
    # start docs.
    assert add_kwargs["item"]["meta"] == {
        qb.RUN_ID_META_KEY: body["run_id"],
        qb.PLAN_META_KEY: {"name": "grid_scan", "kwargs": _GRID_SCAN_ARGS},
    }

    # Arming checks bypass the client's status cache: the capability probe is
    # a plain read, but the pre-check and the unarmed post-add re-check MUST
    # both reload.
    assert manager.kwargs_for("status") == [
        {"reload": False},
        {"reload": True},
        {"reload": True},
    ]


def test_enqueue_consumes_the_draft_revision_exactly_once(client: TestClient, connector) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(), item_add={"success": True, "item": {"item_uid": "u1"}}
    )
    _install(manager)
    revision = _make_draft(client)

    assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200
    second = client.post("/queue/items", json={"draft_revision": revision})

    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "draft_revision_already_launched"
    assert len(manager.kwargs_for("item_add")) == 1


def test_enqueue_with_a_stale_revision_409s_without_touching_the_queue(
    client: TestClient, connector
) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc())
    _install(manager)
    _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": 99})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "stale_draft_revision"
    assert detail["revision"] == 1
    assert "item_add" not in manager.method_names()


@pytest.mark.parametrize("state", sorted(qb.QUEUE_ACTIVE_MANAGER_STATES))
def test_unarmed_enqueue_is_refused_while_the_manager_is_active(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """FR4: enqueue-while-running requires the token — 403, no item, nothing consumed."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(manager_state=state))
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["code"] == "launch_token_required"
    assert detail["manager_state"] == state
    assert "item_add" not in manager.method_names()
    # The reservation was released, so the same revision stays enqueueable.
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


def test_unarmed_enqueue_on_an_unarmed_bridge_is_503_while_active(
    client: TestClient, connector
) -> None:
    """No token configured at all: enqueue-while-active refuses 503, not 403."""
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(manager_state="executing_queue"))
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "launch_token_required"
    assert "item_add" not in manager.method_names()


def test_armed_enqueue_while_active_succeeds(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(manager_state="executing_queue"),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post(
        "/queue/items",
        json={"draft_revision": revision},
        headers={"X-Launch-Token": _TOKEN},
    )

    assert resp.status_code == 200
    assert len(manager.kwargs_for("item_add")) == 1
    assert "item_remove" not in manager.method_names()
    # An armed caller needs no post-add re-check: capability probe + pre-check.
    assert manager.kwargs_for("status") == [{"reload": False}, {"reload": True}]


def test_post_add_recheck_removes_the_item_and_refuses(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manager transitions between the pre-check and the re-check: the
    item is withdrawn and the caller refused — the FR4 defense-in-depth path
    against out-of-band starts the arming lock cannot see."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(),  # capability probe
            status_doc(),  # pre-check: idle, unarmed add allowed
            status_doc(manager_state="starting_queue"),  # post-add re-check
        ],
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["code"] == "launch_token_required"
    # The withdrawal succeeded, so the refusal reports no stranded item.
    assert "item_left_behind" not in detail
    assert manager.kwargs_for("item_remove") == [{"uid": "u1"}]
    # The re-check bypassed the status cache — a stale read here would defeat it.
    assert manager.kwargs_for("status")[-1] == {"reload": True}
    # The revision was not consumed: a failed enqueue never burns the draft.
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


def test_recheck_refusal_reports_a_stranded_item_when_removal_fails(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the withdrawal itself fails, the refusal must say so on the wire —
    an unarmed item left in a draining queue cannot have a container log as
    its only witness."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(),  # capability probe
            status_doc(),  # pre-check
            status_doc(manager_state="starting_queue"),  # post-add re-check
        ],
        item_add={"success": True, "item": {"item_uid": "u1"}},
        item_remove=RequestFailedError({}, {"msg": "cannot remove a running item"}),
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["code"] == "launch_token_required"
    assert detail["item_left_behind"] is True
    assert detail["item_uid"] == "u1"
    assert draft._launching == set()


def test_unarmed_enqueue_is_refused_when_autostart_is_enabled(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Autostart-on means even an idle queue drains a new item without further
    human action — the pre-check observes the flag and demands the token."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(queue_autostart_enabled=True))
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "launch_token_required"
    assert "item_add" not in manager.method_names()
    assert draft._launching == set()


def test_post_enqueue_bookkeeping_failure_releases_the_reservation(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the item is enqueued must not strand the revision in
    the in-flight reservation set forever."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(), item_add={"success": True, "item": {"item_uid": "u1"}}
    )
    _install(manager)
    revision = _make_draft(client)

    async def _boom(*, run_id: str, revision: int) -> None:
        raise RuntimeError("bookkeeping exploded")

    monkeypatch.setattr(queue.draft, "record_and_broadcast_launch", _boom)
    with pytest.raises(RuntimeError, match="bookkeeping exploded"):
        client.post("/queue/items", json={"draft_revision": revision})

    assert draft._launching == set()


def test_recheck_failure_fails_closed_by_withdrawing_the_item(
    client: TestClient, connector
) -> None:
    """If the manager goes silent right after an unarmed add, the state is
    unknowable — the item is withdrawn rather than left behind."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(),  # capability probe
            status_doc(),  # pre-check
            RequestTimeoutError("no answer", {}),  # post-add re-check
        ],
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 503
    assert manager.kwargs_for("item_remove") == [{"uid": "u1"}]
    assert "item_left_behind" not in resp.json()["detail"]
    assert draft._launching == set()


def test_recheck_failure_reports_a_stranded_item_when_withdrawal_also_fails(
    client: TestClient, connector
) -> None:
    """The failure-of-the-failure-path, on the outage branch too: silent
    manager plus a refused withdrawal leaves an unarmed item behind, and the
    503 body says which one."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(),  # capability probe
            status_doc(),  # pre-check
            RequestTimeoutError("no answer", {}),  # post-add re-check
        ],
        item_add={"success": True, "item": {"item_uid": "u1"}},
        item_remove=RequestTimeoutError("no answer either", {}),
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == qb.REASON_MANAGER_UNREACHABLE
    assert detail["item_left_behind"] is True
    assert detail["item_uid"] == "u1"
    assert draft._launching == set()


def test_enqueue_is_refused_on_a_browse_only_deployment(client: TestClient, connector) -> None:
    connector("mock")
    manager = FakeManager(status=status_doc())
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert qb.FLIP_COMMAND in detail["capability"]["detail"]
    # The connector answer wins before anything reaches the manager, and the
    # draft revision is never consumed or even reserved.
    assert manager.calls == []
    assert draft._launching == set()


def test_enqueue_without_a_configured_manager_is_503(client: TestClient, connector) -> None:
    connector("virtual_accelerator")
    app_module.set_queue_backend(QueueBackend(None))
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == qb.REASON_MANAGER_NOT_CONFIGURED
    assert detail["capability"]["can_execute"] is False


def test_validation_gate_refusal_releases_the_reservation(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unchanged launch validation gate runs before anything reaches the
    manager; its refusal must not burn the draft revision."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(), item_add={"success": True, "item": {"item_uid": "u1"}}
    )
    _install(manager)
    revision = _make_draft(client)

    def _reject(request: Any) -> None:
        raise HTTPException(status_code=409, detail="session plan has no passing record")

    monkeypatch.setattr(queue, "_validate_launchable_request", _reject)
    refused = client.post("/queue/items", json={"draft_revision": revision})

    assert refused.status_code == 409
    assert "item_add" not in manager.method_names()
    assert draft._launching == set()

    # Same revision enqueues fine once the gate passes again.
    monkeypatch.undo()
    connector("virtual_accelerator")
    assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200


def test_enqueue_refuses_a_non_admissible_session_plan(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session-plan re-check runs inside the arming lock, before the add:
    a session plan with no current passing record never reaches the manager,
    and the refused enqueue never burns the draft revision."""
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc())
    _install(manager)
    revision = _make_draft(client)

    async def _reject(name: str) -> None:
        raise SessionPlanNotReadyError("no passing record for its current content", plan=name)

    monkeypatch.setattr(queue, "check_session_plan_ready", _reject)
    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "session_plan_unvalidated"
    # The error's native key rides along; "code" is the canonical duplicate.
    assert detail["reason"] == detail["code"]
    assert detail["plan"] == "grid_scan"
    assert "item_add" not in manager.method_names()
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


# ---------------------------------------------------------------------------
# POST /queue/items — add-time device pre-check
# ---------------------------------------------------------------------------


def _devices(*names: str) -> dict[str, Any]:
    """A manager ``devices_allowed`` reply naming exactly ``names``."""
    return {"success": True, "devices_allowed": {name: {"is_movable": True} for name in names}}


def _probe_spec(schema: type[BaseModel]) -> PlanSpec[Any]:
    """A `PlanSpec` around *schema*, with the roles the loader would record."""
    return PlanSpec(
        name="probe",
        plan=lambda devices, params: None,
        schema=schema,
        roles=tuple(channel_roles(schema)),
    )


class _NestedTarget(BaseModel):
    """One movable channel carried alongside a plain string parameter."""

    device: MovableChannel
    mode: str = "fast"


class _NestedParams(BaseModel):
    targets: list[_NestedTarget]


class _SingularParams(BaseModel):
    """A schema whose movable field is named in the SINGULAR."""

    corrector: MovableChannel


class _SplitParams(BaseModel):
    """One field per role, so the two collections can be told apart."""

    correctors: MovableChannels
    bpms: ReadableChannels


class _AbsentFieldParams(BaseModel):
    """A declaration for a field no enqueued params in this file carry."""

    telescopes: MovableChannels


# Representative params for every shipped plan, keyed by plan name, with the
# channel names the pre-check must find under each role. Asserted below to
# cover the whole shipped catalog, so a newly shipped plan fails here until its
# shape is stated — the collection is per-shape, and an unlisted shape is an
# untested one.
_SHIPPED_PLAN_ARGS: dict[str, tuple[dict[str, Any], set[str], set[str]]] = {
    "grid_scan": (_GRID_SCAN_ARGS, {"COR1"}, {"BPM1"}),
    "orm": (
        {
            "correctors": ["COR1"],
            "bpms": ["BPM1"],
            "span_a": 1.0,
            "num": 3,
            "sweep": "bidirectional",
        },
        {"COR1"},
        {"BPM1"},
    ),
}


def test_the_walk_ignores_a_plain_string_beside_a_channel_name() -> None:
    """The nearest ENCLOSING field decides, never an ancestor.

    A role-typed field nested in an object — ``{"targets": [{"device": ...,
    "mode": "fast"}]}`` — must not have ``"fast"`` read as a channel name. A
    walk that stays "inside" a matched field once it enters one collects it and
    refuses the enqueue over a mode string: a false refusal, and one no agent
    can fix, because nothing it does to the channel name makes ``"fast"`` a
    device. Rebinding the key at each dict level collects the declared field's
    value alone, which is what this check has to do.
    """
    params = {"targets": [{"device": "COR1", "mode": "fast"}]}

    movable, readable = queue._referenced_channel_names(_probe_spec(_NestedParams), params)

    assert movable == {"COR1"}
    assert readable == set()


def test_the_walk_matches_a_field_name_exactly() -> None:
    """No singular/plural fuzz: the declared name is the name that matches.

    A schema declaring ``corrector`` collects from ``corrector`` and from
    nothing else — params spelling the field ``correctors`` supply no channel
    name at all, and the enqueue passes through to the worker. Matching a
    near-spelling would mean this check refuses on a name the plan never asked
    for, which is the one direction it must not be wrong in.
    """
    spec = _probe_spec(_SingularParams)

    assert queue._referenced_channel_names(spec, {"corrector": "COR1"}) == ({"COR1"}, set())
    assert queue._referenced_channel_names(spec, {"correctors": ["COR1"]}) == (set(), set())


def test_the_walk_keeps_the_two_roles_apart() -> None:
    """Movable and readable names come back separately, because they are not
    the same mistake: the refusal leads with a movable name when there is one."""
    params = {"correctors": ["COR1", "COR2"], "bpms": ["BPM1"]}

    movable, readable = queue._referenced_channel_names(_probe_spec(_SplitParams), params)

    assert movable == {"COR1", "COR2"}
    assert readable == {"BPM1"}


@pytest.mark.parametrize("plan_name", sorted(_SHIPPED_PLAN_ARGS))
def test_the_walk_reads_every_shipped_plans_declaration(plan_name: str) -> None:
    """Each shipped plan's own declaration, read through the loaded spec.

    This is the same `plan_fields.collect_channels` call the dry run makes when
    it mints a mock per channel, so a name this pre-check would refuse on is by
    construction a name a dry run would have exercised — one declaration, one
    reader, no second walk to drift from it.
    """
    params, expected_movable, expected_readable = _SHIPPED_PLAN_ARGS[plan_name]
    spec = plan_loader.get_facility_plans().plans[plan_name]

    assert queue._referenced_channel_names(spec, params) == (expected_movable, expected_readable)


def test_every_shipped_plan_has_its_param_shape_covered() -> None:
    """Keeps the cross-check above honest as the catalog grows."""
    shipped = {
        name
        for name, spec in plan_loader.get_facility_plans().plans.items()
        if spec.provenance == "shipped"
    }

    assert shipped == set(_SHIPPED_PLAN_ARGS), (
        "a shipped plan's param shape is missing from _SHIPPED_PLAN_ARGS — the "
        "pre-check's channel collection is only tested against the shapes listed there"
    )


def test_enqueue_refuses_a_device_the_worker_did_not_build(client: TestClient, connector) -> None:
    """The one mistake no schema catches: a device name is just a string, and
    it is resolved in the worker on the run's FIRST iteration — so without this
    check the caller learns of it only after an enqueue, a start, and a failed
    run. `COR1` here sits under `grid_scan`'s nested `axes[].setpoint`, the
    field the plan declares movable."""
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(), devices_allowed=_devices("BPM1"))
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["code"] == "unknown_device"
    # The worker's own sentence, word for word, so both layers describe the
    # same event the same way.
    assert detail["detail"] == (
        "plan 'grid_scan' referenced device 'COR1', which this worker did not "
        "build; available devices: ['BPM1']"
    )
    assert detail["devices"] == ["COR1"]
    assert detail["available_devices"] == ["BPM1"]
    assert "item_add" not in manager.method_names()
    # A refused enqueue never burns the draft revision.
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


def test_a_session_tier_plan_is_refused_in_the_session_plans_own_words(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`session_upload.py`'s wrapper opens with "session plan {name}" where the
    catalog wrapper opens with "plan {name}". Whichever would have raised at
    run time, this refusal reads the same — otherwise the earlier, friendlier
    refusal is the one that sounds like a different problem."""
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(), devices_allowed=_devices("BPM1"))
    _install(manager)
    revision = _make_draft(client)
    monkeypatch.setattr(
        plan_loader.get_facility_plans().plans["grid_scan"], "provenance", "session"
    )

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 400
    assert resp.json()["detail"]["detail"].startswith(
        "session plan 'grid_scan' referenced device 'COR1'"
    )


def test_the_refusal_sentence_caps_a_long_device_list(client: TestClient, connector) -> None:
    """A real worker builds hundreds of devices. The sentence is prose someone
    reads, so it summarizes past a cap — while `available_devices` still
    carries every name, which is what a caller picks a correction from."""
    connector("virtual_accelerator")
    built = [f"BPM{n}" for n in range(50)]
    manager = FakeManager(status=status_doc(), devices_allowed=_devices(*built))
    _install(manager)
    revision = _make_draft(client)

    detail = client.post("/queue/items", json={"draft_revision": revision}).json()["detail"]

    assert "(+30 more; full list in available_devices)" in detail["detail"]
    assert len(detail["available_devices"]) == 50
    # Truncation is a rendering choice, never a claim that the set is smaller.
    assert "BPM49" in detail["available_devices"]


def test_a_declared_field_absent_from_the_params_is_not_checked(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared field the params do not carry yields no channel names to
    check, and the manager is never asked. Fail-open: params that supply
    nothing under a role-typed field are params this check has nothing to say
    about, never grounds to refuse an enqueue whose names may be perfectly
    good."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        devices_allowed=_devices("BPM1"),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)
    spec = plan_loader.get_facility_plans().plans["grid_scan"]
    monkeypatch.setattr(spec, "schema", _AbsentFieldParams)
    monkeypatch.setattr(spec, "roles", tuple(channel_roles(_AbsentFieldParams)))

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 200
    assert "devices_allowed" not in manager.method_names()


def test_enqueue_passes_when_the_worker_holds_every_named_device(
    client: TestClient, connector
) -> None:
    """The check must never refuse a name the worker actually has — and asking
    the manager at all is what makes the refusal above non-vacuous."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        devices_allowed=_devices("BPM1", "COR1"),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 200
    assert "devices_allowed" in manager.method_names()
    assert len(manager.kwargs_for("item_add")) == 1


def test_the_precheck_reads_only_the_role_typed_fields(client: TestClient, connector) -> None:
    """`orm` carries a plain string parameter (`sweep`) alongside its channel
    fields. Only the role-typed fields hold channel names; treating every
    string as one would refuse a perfectly good enqueue over `bidirectional`."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        devices_allowed=_devices("BPM1", "COR1"),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    orm_args, _, _ = _SHIPPED_PLAN_ARGS["orm"]
    resp = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": orm_args, "client_id": "test"},
    )
    assert resp.status_code == 200

    added = client.post("/queue/items", json={"draft_revision": resp.json()["revision"]})

    assert added.status_code == 200
    # The check did run — this is a pass-through, not a skipped pre-check.
    assert "devices_allowed" in manager.method_names()


def test_a_plan_declaring_no_channel_roles_is_never_pre_checked(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan's own declaration is what says which params are channel names;
    a schema declaring no role leaves nothing to check, and inventing a rule
    for those plans would refuse enqueues on a guess. The manager is never even
    asked."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        devices_allowed=_devices("BPM1"),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)
    monkeypatch.setattr(plan_loader.get_facility_plans().plans["grid_scan"], "roles", ())

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 200
    assert "devices_allowed" not in manager.method_names()


def test_a_readable_channel_the_worker_lacks_is_refused_too(client: TestClient, connector) -> None:
    """The posture this check has always had for the channels a plan only
    reads, now stated in role terms rather than inherited from a field-name
    list: the worker would raise on the run's very first read, so the mistake
    is just as certain as a movable one and earns the same 400. Nothing is
    loosened and nothing is tightened by the move to declared roles."""
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(), devices_allowed=_devices("COR1"))
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["code"] == "unknown_device"
    assert detail["devices"] == ["BPM1"]
    assert "referenced device 'BPM1'" in detail["detail"]
    assert "item_add" not in manager.method_names()


def test_the_refusal_sentence_leads_with_a_movable_channel(client: TestClient, connector) -> None:
    """When both roles name something the worker lacks, the sentence quotes the
    movable one: that is the reading under which a start would have driven
    hardware toward a channel that does not exist. Every unknown name is on the
    wire either way, under `devices`."""
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(), devices_allowed=_devices("SOMETHING_ELSE"))
    _install(manager)
    revision = _make_draft(client)

    detail = client.post("/queue/items", json={"draft_revision": revision}).json()["detail"]

    assert "referenced device 'COR1'" in detail["detail"]
    assert detail["devices"] == ["BPM1", "COR1"]


async def test_an_unreadable_plan_registry_skips_the_precheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry re-scans the session-plan directory on every call, so it
    can fail on I/O. That must skip the pre-check rather than raise through an
    enqueue whose device names may be perfectly good.

    Driven directly rather than through the route: the launch validation gate
    reads the same registry a step earlier, so a route-level failure would be
    reporting on that call rather than this one.
    """

    def _unreadable() -> Any:
        raise OSError("plan directory is not readable")

    monkeypatch.setattr(plan_loader, "get_facility_plans", _unreadable)
    manager = FakeManager(devices_allowed=_devices("BPM1"))

    await queue._check_devices_exist(QueueBackend(manager), "grid_scan", _GRID_SCAN_ARGS)

    assert manager.method_names() == []


def test_an_unreadable_device_list_does_not_block_the_enqueue(
    client: TestClient, connector
) -> None:
    """Fail-open, deliberately: the worker is still the enforcement point, and
    a convenience gate must never be what costs an operator an enqueue that
    would have run. A manager that is genuinely gone is reported by the add
    itself, not by this check."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        devices_allowed=RequestTimeoutError("no answer", {}),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 200
    assert len(manager.kwargs_for("item_add")) == 1


def test_a_worker_reporting_no_devices_at_all_does_not_block_the_enqueue(
    client: TestClient, connector
) -> None:
    """An empty device list reads as an environment that is not up yet, not as
    a worker on which every name is wrong."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        devices_allowed=_devices(),
        item_add={"success": True, "item": {"item_uid": "u1"}},
    )
    _install(manager)
    revision = _make_draft(client)

    assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200


def test_enqueue_records_no_progress_denominator(client: TestClient, connector) -> None:
    """A queued item carries no progress denominator, deliberately.

    How many points a run will produce is the run's own declaration, read off
    its start document when it begins — so an item that has not started has
    nothing to report, and this route infers nothing from the enqueued
    parameters. `GET /queue` reports that as a null fraction, never a guess.
    """
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(), item_add={"success": True, "item": {"item_uid": "u1"}}
    )
    _install(manager)
    revision = _make_draft(client)

    body = client.post("/queue/items", json={"draft_revision": revision}).json()

    assert queue.document_plane.get_expected_points(body["run_id"]) is None


def test_get_queue_attaches_progress_to_the_running_item(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = {"item_uid": "r1", "name": "grid_scan", "meta": {qb.RUN_ID_META_KEY: "rid-1"}}
    manager = FakeManager(
        status=status_doc(running_item_uid="r1"),
        queue_get={"success": True, "items": [], "running_item": running},
    )
    _install(manager)
    progress = {"rows_seen": 5, "expected_points": 9, "fraction": 5 / 9, "complete": False}
    monkeypatch.setattr(
        queue.document_plane, "progress", lambda run_id: progress if run_id == "rid-1" else None
    )

    body = client.get("/queue").json()

    assert body["running_item"]["item_uid"] == "r1"
    assert body["running_item"]["progress"] == progress


# ---------------------------------------------------------------------------
# Move / remove
# ---------------------------------------------------------------------------


def test_move_forwards_the_single_destination(client: TestClient) -> None:
    manager = FakeManager(item_move={"success": True, "item": {"item_uid": "u1"}})
    _install(manager)

    resp = client.post("/queue/items/u1/move", json={"before_uid": "u2"})

    assert resp.status_code == 200
    assert resp.json()["moved"] is True
    assert manager.kwargs_for("item_move") == [{"uid": "u1", "before_uid": "u2"}]


def test_move_with_no_destination_is_409(client: TestClient) -> None:
    manager = FakeManager()
    _install(manager)

    resp = client.post("/queue/items/u1/move", json={})

    assert resp.status_code == 409
    assert manager.calls == []


def test_remove_forwards_and_maps_an_unknown_uid_to_409(client: TestClient) -> None:
    manager = FakeManager(item_remove={"success": True, "item": {"item_uid": "u1"}})
    _install(manager)

    resp = client.delete("/queue/items/u1")
    assert resp.status_code == 200
    assert resp.json()["removed"] is True
    assert manager.kwargs_for("item_remove") == [{"uid": "u1"}]

    manager.responses["item_remove"] = RequestFailedError({}, {"msg": "unknown uid"})
    missing = client.delete("/queue/items/nope")
    assert missing.status_code == 409
    assert missing.json()["detail"]["code"] == "queue_request_rejected"


# ---------------------------------------------------------------------------
# POST /queue/start / POST /queue/stop
# ---------------------------------------------------------------------------


def test_start_requires_the_launch_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = FakeManager(status=status_doc())
    _install(manager)

    # Unarmed bridge (no token configured): 503, in the queue surface's
    # refusal shape — the arming refusal is exactly the one panels and MCP
    # tools branch on, so a bare-string detail here would break them.
    unarmed = client.post("/queue/start")
    assert unarmed.status_code == 503
    assert unarmed.json()["detail"]["code"] == "launch_token_required"

    # Armed bridge, wrong header: 403, same shape.
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    resp = client.post("/queue/start", headers={"X-Launch-Token": "wrong"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "launch_token_required"

    # The manager was never touched by a refused start.
    assert manager.calls == []


def test_start_ensures_the_environment_then_starts(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc())
    _install(manager)

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 200
    assert resp.json()["started"] is True
    names = manager.method_names()
    assert "queue_start" in names
    # Environment already open: verified, never re-opened.
    assert "environment_open" not in names


def test_start_on_a_browse_only_deployment_carries_the_capability(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("mock")
    manager = FakeManager(status=status_doc())
    _install(manager)

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert qb.FLIP_COMMAND in detail["capability"]["detail"]
    assert "queue_start" not in manager.method_names()


def test_start_gates_every_item_the_start_would_drain(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue-start session gate sees the whole drain set, in queue order.

    The drain set is every PENDING item: a start that found something already
    running never reaches this gate (see the idle check below), so the pending
    queue is the complete set of work the start would put on hardware.
    """
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        queue_get={
            "success": True,
            "items": [{"name": "grid_scan"}, {"name": "session_sweep"}, {"name": "orm"}],
            "running_item": {},
        },
    )
    _install(manager)

    seen: list[list[str]] = []

    async def _record(names: Any) -> None:
        seen.append(list(names))

    monkeypatch.setattr(queue, "check_session_plans_ready", _record)
    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 200
    assert seen == [["grid_scan", "session_sweep", "orm"]]


def test_start_is_refused_when_a_queued_session_plan_is_stale(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stale session plan refuses the whole start — all-or-nothing."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        queue_get={"success": True, "items": [{"name": "session_sweep"}], "running_item": {}},
    )
    _install(manager)

    async def _reject(names: Any) -> None:
        raise SessionPlanNotReadyError("record died with a bridge restart", plan="session_sweep")

    monkeypatch.setattr(queue, "check_session_plans_ready", _reject)
    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "session_plan_unvalidated"
    assert detail["plan"] == "session_sweep"
    assert "queue_start" not in manager.method_names()


# ---------------------------------------------------------------------------
# The interrupted-item start gate.
#
# Upstream does not discard a plan the worker interrupted: it records the run
# in history AND pushes a copy back to the FRONT of the queue with its
# `result`. So the queue an operator leaves behind after an emergency abort has
# the aborted plan at its head, and an armed start would put it straight back
# on the hardware. Found against a real manager by
# tests/e2e/test_bluesky_queue_e2e.py.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_status", ["aborted", "halted", "failed"])
def test_start_refuses_while_an_interrupted_item_sits_in_the_queue(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch, exit_status: str
) -> None:
    """An emergency-aborted plan never re-runs without a fresh human decision."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        queue_get={
            "success": True,
            "items": [
                {
                    "name": "grid_scan",
                    "item_uid": "requeued-uid",
                    "result": {"exit_status": exit_status, "run_uids": ["re-uid"]},
                }
            ],
            "running_item": {},
        },
    )
    _install(manager)

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "interrupted_item_in_queue"
    assert detail["item_uid"] == "requeued-uid"
    assert detail["plan"] == "grid_scan"
    assert detail["exit_status"] == exit_status
    # The refusal must name REMOVAL as the only way on -- this gate is
    # stateless and refuses every start while the copy is queued, so "leave it
    # and start again" is not an exit that exists. Re-running is a step taken
    # AFTER the removal, and the sentence has to read in that order.
    sentence = detail["detail"]
    assert "DELETE /queue/items/requeued-uid" in sentence
    assert "every start is refused" in sentence.lower()
    assert "draft" in sentence
    assert sentence.index("DELETE /queue/items/requeued-uid") < sentence.index("draft"), (
        f"the refusal offers the re-run before the removal it depends on: {sentence}"
    )
    # The load-bearing half: the manager was never asked to start.
    assert "queue_start" not in manager.method_names()


def test_start_is_not_blocked_by_an_ordinary_pending_item(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a queue of normal pending work still starts.

    Without this, the gate above could quietly become "never start", which
    would look identical from the refusal test's point of view.
    """
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        queue_get={
            "success": True,
            "items": [{"name": "grid_scan", "item_uid": "u-1"}, {"name": "orm", "item_uid": "u-2"}],
            "running_item": {},
        },
    )
    _install(manager)

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 200
    assert "queue_start" in manager.method_names()


# ---------------------------------------------------------------------------
# The idle gate: no start while a plan is in motion.
#
# The manager refuses a start from any non-idle state anyway, so this is
# behaviour-preserving on the wire (409 either way). It exists because it is
# what makes the interrupted-item gate above EXHAUSTIVE: a running item carries
# no `result` yet, so a snapshot holding one used to pass that check, and a
# plan that aborted in the gap before `queue_start` would be requeued to the
# front and drained by the very start the gate exists to refuse.
# ---------------------------------------------------------------------------


def test_start_refuses_while_a_plan_is_already_running(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot with a running item never reaches the manager's start."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(manager_state="executing_queue", running_item_uid="running-uid"),
        queue_get={
            "success": True,
            "items": [{"name": "orm", "item_uid": "pending-uid"}],
            "running_item": {"name": "grid_scan", "item_uid": "running-uid"},
        },
    )
    _install(manager)

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "manager_not_idle"
    assert detail["plan"] == "grid_scan"
    assert detail["item_uid"] == "running-uid"
    # The load-bearing half: the manager was never asked to start.
    assert "queue_start" not in manager.method_names()


def test_start_still_starts_when_nothing_is_running(
    client: TestClient, connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the idle gate: an idle snapshot still starts.

    Pairs with the refusal above so the gate cannot quietly become "never
    start" -- the failure mode that would look identical from the refusal
    test's point of view.
    """
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(),
        queue_get={
            "success": True,
            "items": [{"name": "grid_scan", "item_uid": "pending-uid"}],
            "running_item": {},
        },
    )
    _install(manager)

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 200
    assert resp.json()["started"] is True
    assert "queue_start" in manager.method_names()


def test_stop_is_ungated_but_cancelling_a_stop_requires_the_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Halting is always allowed; UN-halting re-arms a draining queue and is
    token-gated — an unarmed caller must not be able to reverse a human's stop."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    manager = FakeManager()
    _install(manager)

    resp = client.post("/queue/stop")
    assert resp.status_code == 200
    assert resp.json()["stop_pending"] is True

    unarmed = client.post("/queue/stop", json={"cancel": True})
    assert unarmed.status_code == 403
    assert unarmed.json()["detail"]["code"] == "launch_token_required"
    # The withdrawal never reached the manager.
    assert manager.method_names() == ["queue_stop"]

    armed = client.post("/queue/stop", json={"cancel": True}, headers={"X-Launch-Token": _TOKEN})
    assert armed.status_code == 200
    assert armed.json()["stop_pending"] is False
    assert manager.method_names() == ["queue_stop", "queue_stop_cancel"]


# ---------------------------------------------------------------------------
# POST /queue/abort — the emergency halt, ungated on every axis
# ---------------------------------------------------------------------------


def _install_abort(manager: Any) -> QueueBackend:
    """Install a backend whose abort pause window closes fast enough to test."""
    backend = QueueBackend(manager, abort_pause_polls=2, abort_poll_interval=0)
    app_module.set_queue_backend(backend)
    return backend


def test_abort_needs_no_token_even_on_an_armed_bridge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the route. A bridge that HAS a token configured is
    the case that would expose an accidental gate: if the route ever grew one,
    this bare call would 403 instead of aborting."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    manager = FakeManager(
        status=[
            status_doc(manager_state="executing_queue"),
            status_doc(manager_state="paused"),
            status_doc(manager_state="idle"),
        ],
        re_abort={"success": True, "msg": "aborted"},
    )
    _install_abort(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 200
    body = resp.json()
    assert body["aborted"] is True
    assert body["abort_pending"] is False
    assert body["paused_first"] is True
    assert body["msg"] == "aborted"
    assert "re_abort" in manager.method_names()


def test_abort_declares_no_launch_token_header(client: TestClient) -> None:
    """Structural, not behavioural: the route takes no token parameter at all,
    so there is nothing for a later "make it consistent with start" edit to
    gate. Read off the OpenAPI schema rather than the source so a header added
    by any means fails this."""
    from osprey.services.bluesky_bridge.app import app as bridge_app

    operation = bridge_app.openapi()["paths"]["/queue/abort"]["post"]
    header_names = {
        param.get("name", "").lower()
        for param in operation.get("parameters", [])
        if param.get("in") == "header"
    }
    assert "x-launch-token" not in header_names, (
        "the emergency abort must not accept a launch token — a halt that can be "
        "refused for a policy reason is a halt with a failure mode"
    )
    # Non-vacuity: the sibling arming route DOES declare one, so this assertion
    # is reading a real distinction rather than an empty parameter list.
    start = bridge_app.openapi()["paths"]["/queue/start"]["post"]
    start_headers = {
        param.get("name", "").lower()
        for param in start.get("parameters", [])
        if param.get("in") == "header"
    }
    assert "x-launch-token" in start_headers


def test_abort_is_not_capability_gated(client: TestClient, connector) -> None:
    """A deployment that somehow has a plan running must be able to stop it,
    whatever its capability record says. `queue_add` refuses on a mock
    connector; the abort deliberately does not."""
    connector("mock")
    manager = FakeManager(
        status=[status_doc(manager_state="paused"), status_doc(manager_state="idle")]
    )
    _install_abort(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 200
    assert resp.json()["aborted"] is True


def test_abort_with_nothing_running_is_a_409_nothing_running(client: TestClient) -> None:
    manager = FakeManager(status=status_doc(manager_state="idle"))
    _install_abort(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "nothing_running"
    assert manager.method_names() == ["status"]


def test_abort_that_never_pauses_is_a_503_that_says_nothing_stopped(
    client: TestClient,
) -> None:
    """The refusal an operator reads while a machine may still be scanning. It
    must carry its own code and say plainly that nothing was aborted."""
    manager = FakeManager(status=status_doc(manager_state="executing_queue"))
    _install_abort(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == "abort_pause_timeout"
    assert "NOTHING WAS ABORTED" in detail["detail"]


def test_abort_on_an_unreachable_manager_is_a_503(client: TestClient) -> None:
    manager = FakeManager(status=RequestTimeoutError("silent", {}))
    _install_abort(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "manager_unreachable"


def test_abort_refused_by_the_manager_is_a_409(client: TestClient) -> None:
    manager = FakeManager(
        status=status_doc(manager_state="paused"),
        re_abort=RequestFailedError("cannot abort", {}),
    )
    _install_abort(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "queue_request_rejected"


async def test_abort_pushes_the_state_change_to_sse_subscribers() -> None:
    """The panel learns what happened from the stream, not from the response.

    An abort moves `manager_state` and `running_item_uid`, both in the poller's
    diffed summary, so no extra broadcast plumbing is needed -- and
    `_notify_change` means the frame does not wait a full ~1s tick. Follows the
    white-box `_subscribe()` precedent (`test_draft_launch_state.py`): one
    event loop, no streaming client to deadlock against.
    """
    manager = FakeManager(
        status=[
            status_doc(manager_state="executing_queue", running_item_uid="u1"),  # hello
            status_doc(manager_state="executing_queue", running_item_uid="u1"),  # abort: first
            status_doc(manager_state="paused", running_item_uid="u1"),  # abort: pause poll
            status_doc(  # abort: final read, and every poll after
                manager_state="idle", running_item_uid=None, plan_history_uid="h-2"
            ),
        ],
        queue_get={"success": True, "items": [], "running_item": {}},
    )
    _install_abort(manager)

    subscriber, hello = await queue._subscribe()
    try:
        assert hello["status"]["manager_state"] == "executing_queue"

        result = await queue.abort_running_plan()
        assert result["aborted"] is True

        frame = await asyncio.wait_for(subscriber.get(), timeout=5)
    finally:
        await queue._unsubscribe(subscriber)

    assert frame["type"] == "queue"
    assert frame["status"]["manager_state"] == "idle"
    assert frame["status"]["running_item_uid"] is None


def test_stop_cancel_on_an_unarmed_bridge_is_503(client: TestClient) -> None:
    manager = FakeManager()
    _install(manager)

    resp = client.post("/queue/stop", json={"cancel": True})

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "launch_token_required"
    assert manager.calls == []


# ---------------------------------------------------------------------------
# Lifespan backend ownership
# ---------------------------------------------------------------------------


def test_lifespan_leaves_a_pre_injected_backend_to_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend set before startup belongs to whoever set it: the lifespan
    neither opens its environment nor closes it on shutdown. Closing another
    owner's handle would break a test (or bespoke deploy wiring) that is still
    using it after the app comes down."""
    monkeypatch.delenv(qb.QSERVER_CONTROL_ADDRESS_ENV, raising=False)
    manager = FakeManager(status=status_doc())
    backend = _install(manager)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert manager.closed is False
    assert "environment_open" not in manager.method_names()
    assert app_module.get_queue_backend() is backend


def test_lifespan_closes_the_backend_it_built_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """The converse: with nothing injected, the backend the lifespan lazily
    built while serving is the lifespan's to close — and the singleton is
    cleared so the next startup builds fresh."""
    monkeypatch.delenv(qb.QSERVER_CONTROL_ADDRESS_ENV, raising=False)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app_module.get_queue_backend() is not None

    assert app_module._queue_backend is None


# ---------------------------------------------------------------------------
# Race tests — the FR4 lock, driven deterministically on one event loop
# ---------------------------------------------------------------------------


class GatedManager:
    """A manager whose armed-path calls block on test-controlled events.

    ``queue_start`` signals ``start_entered`` and then waits for
    ``release_start`` while the caller holds the arming lock — the window the
    race tests interleave an unarmed add into. ``item_add`` optionally waits
    for ``release_add`` the same way. ``manager_state`` flips to
    ``starting_queue`` when a start completes, exactly like a real manager.
    """

    def __init__(self, *, gate_add: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.manager_state = "idle"
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()
        self.add_entered = asyncio.Event()
        self.release_add = asyncio.Event()
        self._gate_add = gate_add

    async def status(self, *, reload: bool = False) -> dict[str, Any]:
        self.calls.append(("status", {"reload": reload}))
        return status_doc(manager_state=self.manager_state)

    async def queue_start(self) -> dict[str, Any]:
        self.calls.append(("queue_start", {}))
        self.start_entered.set()
        await self.release_start.wait()
        self.manager_state = "starting_queue"
        return {"success": True, "msg": ""}

    async def item_add(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("item_add", kwargs))
        self.add_entered.set()
        if self._gate_add:
            await self.release_add.wait()
        return {"success": True, "item": {"item_uid": "u1"}}

    async def item_remove(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("item_remove", kwargs))
        return {"success": True}

    async def queue_get(self) -> dict[str, Any]:
        self.calls.append(("queue_get", {}))
        return {"success": True, "items": [], "running_item": {}}

    async def plans_allowed(self) -> dict[str, Any]:
        self.calls.append(("plans_allowed", {}))
        return {"success": True, "plans_allowed": {}}

    async def close(self) -> None:
        pass

    def method_names(self) -> list[str]:
        return [name for name, _ in self.calls]


async def test_race_unarmed_add_loses_to_an_in_flight_armed_start(
    connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An armed start holds the arming lock; a concurrent unarmed add must
    wait for it and then be refused — 403, no item ever added, revision not
    consumed. The FR4 acceptance race, deterministic via event gates."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = GatedManager()
    app_module.set_queue_backend(QueueBackend(manager))
    revision = await _make_draft_direct()

    start_task = asyncio.create_task(queue.start_queue(x_launch_token=_TOKEN))
    await asyncio.wait_for(manager.start_entered.wait(), timeout=2)
    # The start now holds the arming lock, blocked inside queue_start.

    add_task = asyncio.create_task(
        queue.add_queue_item(queue.QueueAddRequest(draft_revision=revision), x_launch_token="")
    )
    # Give the add every chance to run: it must park on the lock, never reach
    # the manager while the start is in flight.
    for _ in range(10):
        await asyncio.sleep(0)
    assert "item_add" not in manager.method_names()

    manager.release_start.set()
    assert (await start_task)["started"] is True

    with pytest.raises(HTTPException) as excinfo:
        await add_task
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail["code"] == "launch_token_required"
    assert excinfo.value.detail["manager_state"] == "starting_queue"

    assert "item_add" not in manager.method_names()
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


async def test_race_pending_start_waits_for_the_unarmed_add_critical_section(
    connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse interleaving: an unarmed add on an idle queue holds the
    lock; an armed start queues up behind it. The add completes — including
    its post-add re-check — strictly before the start begins, so the add is
    legitimately pre-start (idle at both checks) and both succeed."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    manager = GatedManager(gate_add=True)
    app_module.set_queue_backend(QueueBackend(manager))
    revision = await _make_draft_direct()

    add_task = asyncio.create_task(
        queue.add_queue_item(queue.QueueAddRequest(draft_revision=revision), x_launch_token="")
    )
    await asyncio.wait_for(manager.add_entered.wait(), timeout=2)
    # The add now holds the arming lock, blocked inside item_add.

    start_task = asyncio.create_task(queue.start_queue(x_launch_token=_TOKEN))

    # Let the start run all the way to the arming lock: its pre-lock
    # environment probing stops issuing manager calls once it parks there.
    async def _settled() -> None:
        while True:
            count = len(manager.calls)
            for _ in range(20):
                await asyncio.sleep(0)
            if len(manager.calls) == count:
                return

    await asyncio.wait_for(_settled(), timeout=2)
    assert "queue_start" not in manager.method_names()
    calls_before_release = len(manager.calls)

    manager.release_add.set()
    manager.release_start.set()

    add_result = await add_task
    start_result = await asyncio.wait_for(start_task, timeout=2)

    assert add_result["item"] == {"item_uid": "u1"}
    assert start_result["started"] is True
    assert "item_remove" not in manager.method_names()

    # Serialization on the wire, exactly: everything after the release point
    # is the add's post-add re-check (a reloaded status read), then the
    # start's in-lock section — nothing interleaves. Deleting the re-check
    # makes this fail (the tail would begin with queue_get).
    tail = manager.calls[calls_before_release:]
    assert [name for name, _ in tail] == ["status", "queue_get", "plans_allowed", "queue_start"]
    assert tail[0] == ("status", {"reload": True})

    # The revision was consumed exactly once.
    assert draft._last_launched_revision == revision
    assert draft._launching == set()


async def test_race_concurrent_enqueues_of_one_revision_yield_one_item(connector) -> None:
    """Two unarmed enqueues pinning the same revision race: the reservation
    taken inside `check_launchable`'s critical section admits exactly one."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=status_doc(), item_add={"success": True, "item": {"item_uid": "u1"}}
    )
    app_module.set_queue_backend(QueueBackend(manager))
    revision = await _make_draft_direct()

    async def _try() -> Any:
        try:
            return await queue.add_queue_item(
                queue.QueueAddRequest(draft_revision=revision), x_launch_token=""
            )
        except HTTPException as exc:
            return exc

    first, second = await asyncio.gather(_try(), _try())
    outcomes = {type(first).__name__, type(second).__name__}

    assert outcomes == {"dict", "HTTPException"}
    refusal = first if isinstance(first, HTTPException) else second
    assert refusal.status_code == 409
    assert refusal.detail["code"] == "draft_revision_already_launched"
    assert len(manager.kwargs_for("item_add")) == 1


# ---------------------------------------------------------------------------
# GET /queue/events — SSE
# ---------------------------------------------------------------------------


def _parse_frame(raw: str) -> dict[str, Any]:
    assert raw.startswith("data: ")
    return json.loads(raw[len("data: ") :])


async def test_sse_hello_then_change_frame_on_status_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(queue, "_POLL_INTERVAL_S", 0.01)
    manager = FakeManager(
        status=status_doc(),
        queue_get={"success": True, "items": [], "running_item": {}},
    )
    app_module.set_queue_backend(QueueBackend(manager))

    resp = await queue.queue_events()
    gen = resp.body_iterator
    try:
        hello = _parse_frame(await gen.__anext__())
        assert hello["type"] == "hello"
        assert hello["status"]["available"] is True
        assert hello["items"] == []

        # The queue changes out-of-band (an item appears, the queue uid moves):
        # the poller must notice within a tick and push a full snapshot frame.
        manager.responses["status"] = status_doc(items_in_queue=1, plan_queue_uid="q-2")
        manager.responses["queue_get"] = {
            "success": True,
            "items": [{"item_uid": "a"}],
            "running_item": {},
        }

        frame = _parse_frame(await asyncio.wait_for(gen.__anext__(), timeout=2))
        assert frame["type"] == "queue"
        assert frame["status"]["items_in_queue"] == 1
        assert frame["items"] == [{"item_uid": "a"}]
    finally:
        await gen.aclose()

    # Last subscriber gone: the poller is stopped and the registry empty.
    assert queue._subscribers == set()
    assert queue._poller_task is None


async def test_sse_poller_diffs_on_the_status_summary() -> None:
    """White-box: an unchanged summary broadcasts nothing; a changed one
    broadcasts exactly one full-snapshot frame."""
    manager = FakeManager(
        status=status_doc(), queue_get={"success": True, "items": [], "running_item": {}}
    )
    app_module.set_queue_backend(QueueBackend(manager))
    subscriber: asyncio.Queue[Any] = asyncio.Queue()
    queue._subscribers.add(subscriber)

    await queue._poll_once()
    assert subscriber.qsize() == 1  # first poll always differs from the empty cache
    subscriber.get_nowait()

    await queue._poll_once()
    assert subscriber.qsize() == 0  # nothing changed, nothing broadcast

    manager.responses["status"] = status_doc(manager_state="executing_queue")
    await queue._poll_once()
    assert subscriber.qsize() == 1
    frame = subscriber.get_nowait()
    assert frame["status"]["manager_state"] == "executing_queue"


async def test_sse_reports_manager_outage_as_unavailable() -> None:
    manager = FakeManager(status=RequestTimeoutError("no answer", {}))
    app_module.set_queue_backend(QueueBackend(manager))
    subscriber: asyncio.Queue[Any] = asyncio.Queue()
    queue._subscribers.add(subscriber)

    await queue._poll_once()

    frame = subscriber.get_nowait()
    assert frame["status"]["available"] is False
    assert frame["status"]["reason"] == qb.REASON_MANAGER_UNREACHABLE
    assert frame["items"] == []


def test_bridge_mutations_nudge_the_sse_poller(client: TestClient) -> None:
    manager = FakeManager(item_remove={"success": True})
    _install(manager)
    queue._change_event.clear()

    assert client.delete("/queue/items/u1").status_code == 200

    assert queue._change_event.is_set()
