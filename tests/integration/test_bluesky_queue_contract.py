"""The bridge's queue surface as a WIRE CONTRACT, against a mocked queue server.

This is the CI-lane half of the two-party rule between OSPREY's bridge and
bluesky-queueserver. `tests/services/bluesky_bridge/test_queue_routes.py` pins
the route layer's internals — which manager call each branch makes, which
module state each failure leaves behind. This module pins what a *consumer*
(the sidecar relay, the MCP tools, the panels) can rely on: the JSON bodies,
the refusal codes, the SSE frame sequence, and the safety invariants that hold
across a race. Everything runs against a mocked manager on the standard
`pytest tests/ --ignore=tests/e2e` lane — no containers, no network.

Three things shape how the assertions are written:

**A stateful mock, not a scripted one.** `MockQueueServer` HOLDS the queue:
`item_add` assigns a uid and appends, `item_move`/`item_remove` mutate the same
list, and every mutation moves `plan_queue_uid` exactly as the manager does. So
"the refusal left no item behind" is asserted against the queue itself, not
inferred from which methods were called — the difference between testing the
safety property and testing the code that was written for it.

**Refusals are asserted by body, never by status code alone.** Every refusal on
this surface is `{"code": <machine-readable>, "detail": <sentence>, ...extras}`,
and consumers branch on `detail.code`; a status code that survived a body
rewrite would be a green test over a broken panel. Success paths carry the
matching negative controls — no `code`, no `capability` key on a 200.

**Races are event-gated and pin call ORDER.** No sleeps: the mock blocks inside
a named manager call until the test releases it, so each interleaving is
deterministic, and the assertions name the order the arming lock is supposed to
produce rather than only the outcome it produced this time.

The direct-execute routes that predate the queue are asserted here too: they
still answer, with the same `{"code", "detail"}` refusal body, so a consumer
that finds one of them learns where its capability went instead of reading a
bare 404.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from bluesky_queueserver_api.comm_base import RequestFailedError, RequestTimeoutError
from fastapi import HTTPException
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import draft, plan_loader, queue
from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.plan_fields import (
    CHANNEL_ROLE_KEY,
    MOVABLE_ROLE,
    READABLE_ROLE,
)
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.queue_backend import QueueBackend
from osprey.services.bluesky_bridge.session_upload import (
    REASON_NOT_IN_NAMESPACE,
    REASON_UNVALIDATED,
    SESSION_PLAN_MODULE,
    set_session_uploader,
    upload_after_validation,
)
from osprey.services.bluesky_bridge.validation_record import validation_records
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"
_TOKEN = "contract-token"

_SESSION_PLAN_NAME = "contract_sweep"


def _grid_scan_args(num_points: int = 3) -> dict[str, Any]:
    """A valid draft for the always-registered shipped ``grid_scan`` plan."""
    return {
        "readbacks": ["BPM1"],
        "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": num_points}],
    }


def _session_plan_source(name: str) -> str:
    """A minimal session-tier plan file satisfying the load + upload contract."""
    return (
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A session-tier plan authored for the contract tests.",\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        f'    yield ("noop", {name!r})\n'
    )


# ---------------------------------------------------------------------------
# The mocked queue server
# ---------------------------------------------------------------------------


class MockQueueServer:
    """A stateful stand-in for a queueserver RE manager (``REManagerAPI``'s shape).

    Holds a real queue, a real worker namespace, and real uids, so a test can
    assert what the manager ENDS UP with rather than only which calls it saw.
    Three behaviours are modelled deliberately:

    - every queue mutation moves ``plan_queue_uid`` and every history append
      moves ``plan_history_uid``, which is the whole basis of the SSE poller's
      change detection;
    - :meth:`begin_running` / :meth:`finish_running` advance the worker WITHOUT
      any bridge call, because that is how the manager really moves and the
      poller has to notice it unprompted;
    - :meth:`script_upload` executes the uploaded script the way queueserver
      does (namespace as both globals and locals) and then publishes whatever
      session plans it defined, so the upload/namespace/``plans_allowed`` loop
      is exercised end to end rather than stubbed.

    Any call can be made to fail (``failures``) or to block mid-call until the
    test releases it (:meth:`gate`) — the event gates are what make the race
    tests deterministic.
    """

    def __init__(self, *, environment_exists: bool = True, manager_state: str = "idle") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.items: list[dict[str, Any]] = []
        self.running_item: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.namespace: dict[str, Any] = {}
        self.manager_state = manager_state
        self.environment_exists = environment_exists
        self.autostart_enabled = False
        self.stop_pending = False
        self.closed = False
        self.failures: dict[str, Exception] = {}
        self._uid_counter = 0
        self._queue_uid = 0
        self._history_uid = 0
        self._entered: dict[str, asyncio.Event] = {}
        self._release: dict[str, asyncio.Event] = {}
        self._status_doc: dict[str, Any] | None = None
        self._status_cached = False

    # ------------------------------------------------------------- test hooks

    def cache_status(self) -> None:
        """Serve unreloaded ``status()`` reads from a cache, as the client does.

        ``REManagerAPI`` answers ``status()`` from a ~0.5 s cache and only goes
        to the manager on ``reload=True``. A test that turns this on sees what
        a cached read costs: the document as of the last reloaded read, however
        far the queue has moved since — the mock holds it indefinitely so the
        window is a decision the test makes rather than a clock it races.
        """
        self._status_cached = True

    def gate(self, method: str) -> tuple[asyncio.Event, asyncio.Event]:
        """Block *method* mid-call. Returns ``(entered, release)`` events."""
        entered = asyncio.Event()
        release = asyncio.Event()
        self._entered[method] = entered
        self._release[method] = release
        return entered, release

    def begin_running(self) -> None:
        """The worker picks the head item up — an out-of-band transition."""
        self.running_item = self.items.pop(0)
        self.manager_state = "executing_queue"
        self._queue_uid += 1

    def finish_running(self) -> None:
        """The running item completes and lands in history — also out of band."""
        if self.running_item is not None:
            self.history.append(self.running_item)
        self.running_item = None
        self.manager_state = "idle"
        self._history_uid += 1

    def method_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def kwargs_for(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def item_uids(self) -> list[str]:
        return [item["item_uid"] for item in self.items]

    # ------------------------------------------------------------- plumbing

    async def _enter(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))
        failure = self.failures.get(method)
        if failure is not None:
            raise failure
        entered = self._entered.get(method)
        if entered is not None:
            entered.set()
        release = self._release.get(method)
        if release is not None:
            await release.wait()

    def _take(self, uid: str) -> dict[str, Any]:
        for index, item in enumerate(self.items):
            if item["item_uid"] == uid:
                return self.items.pop(index)
        raise RequestFailedError({}, {"msg": f"item {uid!r} is not in the queue"})

    # --------------------------------------------------------- manager surface

    async def status(self, *, reload: bool = False) -> dict[str, Any]:
        await self._enter("status", reload=reload)
        if self._status_cached and not reload and self._status_doc is not None:
            return dict(self._status_doc)
        running_uid = self.running_item["item_uid"] if self.running_item else None
        self._status_doc = {
            "success": True,
            "manager_state": self.manager_state,
            "worker_environment_exists": self.environment_exists,
            "items_in_queue": len(self.items),
            "items_in_history": len(self.history),
            "running_item_uid": running_uid,
            "plan_queue_uid": f"q-{self._queue_uid}",
            "plan_history_uid": f"h-{self._history_uid}",
            "queue_stop_pending": self.stop_pending,
            "queue_autostart_enabled": self.autostart_enabled,
            # Present so a leak of the raw status document into a wire body
            # would be visible; the bridge publishes an 8-key summary instead.
            "zmq_secret_key": "never-on-the-wire",
        }
        return dict(self._status_doc)

    async def queue_get(self) -> dict[str, Any]:
        await self._enter("queue_get")
        return {
            "success": True,
            "items": [dict(item) for item in self.items],
            # qserver reports "nothing running" as an empty dict, not null.
            "running_item": dict(self.running_item) if self.running_item else {},
        }

    async def history_get(self) -> dict[str, Any]:
        await self._enter("history_get")
        return {"success": True, "items": [dict(item) for item in self.history]}

    async def item_add(self, **kwargs: Any) -> dict[str, Any]:
        await self._enter("item_add", **kwargs)
        self._uid_counter += 1
        stored = {**kwargs["item"], "item_uid": f"item-{self._uid_counter}"}
        self.items.append(stored)
        self._queue_uid += 1
        return {"success": True, "item": dict(stored), "qsize": len(self.items)}

    async def item_move(self, **kwargs: Any) -> dict[str, Any]:
        await self._enter("item_move", **kwargs)
        item = self._take(kwargs["uid"])
        pos_dest = kwargs.get("pos_dest")
        if kwargs.get("before_uid") is not None:
            index = self._index_of(kwargs["before_uid"])
        elif kwargs.get("after_uid") is not None:
            index = self._index_of(kwargs["after_uid"]) + 1
        elif pos_dest == "front":
            index = 0
        elif isinstance(pos_dest, int):
            index = pos_dest
        else:
            index = len(self.items)
        self.items.insert(index, item)
        self._queue_uid += 1
        return {"success": True, "item": dict(item)}

    def _index_of(self, uid: str) -> int:
        for index, item in enumerate(self.items):
            if item["item_uid"] == uid:
                return index
        raise RequestFailedError({}, {"msg": f"item {uid!r} is not in the queue"})

    async def item_remove(self, **kwargs: Any) -> dict[str, Any]:
        await self._enter("item_remove", **kwargs)
        item = self._take(kwargs["uid"])
        self._queue_uid += 1
        return {"success": True, "item": dict(item)}

    async def queue_start(self) -> dict[str, Any]:
        await self._enter("queue_start")
        if not self.environment_exists:
            raise RequestFailedError({}, {"msg": "the worker environment is not open"})
        self.manager_state = "starting_queue"
        self._queue_uid += 1
        return {"success": True, "msg": "queue started"}

    async def queue_stop(self) -> dict[str, Any]:
        await self._enter("queue_stop")
        self.stop_pending = True
        return {"success": True, "msg": "stop pending"}

    async def queue_stop_cancel(self) -> dict[str, Any]:
        await self._enter("queue_stop_cancel")
        self.stop_pending = False
        return {"success": True, "msg": "stop withdrawn"}

    async def queue_autostart(self, *, enable: bool) -> dict[str, Any]:
        await self._enter("queue_autostart", enable=enable)
        self.autostart_enabled = enable
        return {"success": True, "msg": ""}

    async def queue_clear(self) -> dict[str, Any]:
        await self._enter("queue_clear")
        self.items.clear()
        self._queue_uid += 1
        return {"success": True, "msg": "queue cleared"}

    async def history_clear(self) -> dict[str, Any]:
        """Forget every completed item. The pending queue is untouched."""
        await self._enter("history_clear")
        self.history.clear()
        return {"success": True, "msg": "history cleared"}

    async def re_pause(self, **kwargs: Any) -> dict[str, Any]:
        """Pause the Run Engine, refusing when no plan is under way.

        The refusal is the load-bearing half: upstream rejects a pause while
        the queue is only STARTING (the plan has not begun), which is what the
        bridge's abort composition has to survive by retrying.
        """
        await self._enter("re_pause", **kwargs)
        if self.manager_state not in ("executing_queue", "executing_task"):
            raise RequestFailedError({}, {"msg": "no plan is currently running"})
        self.manager_state = "paused"
        return {"success": True, "msg": ""}

    async def re_abort(self) -> dict[str, Any]:
        """Abort the paused plan. Refuses unless the Run Engine IS paused.

        This is the upstream constraint the whole abort route exists to
        compose around, so the mock enforces it rather than accepting an abort
        from any state — a permissive mock here would let the bridge skip the
        pause and still pass.

        AND it puts the item back. Upstream
        (``plan_queue_ops._set_processed_item_as_stopped``) adds the
        interrupted plan to history with its ``exit_status`` AND — for every
        exit status except ``"stopped"`` — pushes a COPY back to the FRONT of
        the queue under a NEW ``item_uid``, keeping the ``result``. That is not
        configurable.

        This mock used to append to history and stop there, which is exactly
        how a real defect stayed invisible: an aborted run came back on the
        wire as ``pending``, and an armed start would have re-run the plan a
        human had just emergency-stopped. The requeue is reproduced here so the
        contract tests below are a two-party proof rather than a
        proof-against-our-own-assumption.
        """
        await self._enter("re_abort")
        if self.manager_state != "paused":
            raise RequestFailedError({}, {"msg": "the Run Engine is not paused"})
        if self.running_item is not None:
            finished = {**self.running_item, "result": {"exit_status": "aborted"}}
            self.history.append(finished)
            # New item_uid on the requeued copy, front of the queue — upstream's
            # shape exactly. The OSPREY run id in `meta` is carried along, which
            # is why the copy and the history entry describe one run.
            self.items.insert(0, {**finished, "item_uid": f"{finished.get('item_uid')}-requeued"})
            self.running_item = None
            self._history_uid += 1
        self.manager_state = "idle"
        return {"success": True, "msg": "plan aborted"}

    async def environment_open(self) -> dict[str, Any]:
        await self._enter("environment_open")
        self.environment_exists = True
        return {"success": True, "msg": ""}

    async def plans_allowed(self, **kwargs: Any) -> dict[str, Any]:
        await self._enter("plans_allowed", **kwargs)
        return {"success": True, "plans_allowed": self._allowed_plans()}

    def _allowed_plans(self) -> dict[str, Any]:
        """What the manager would publish for the current worker namespace."""
        allowed: dict[str, Any] = {}
        for name, value in self.namespace.items():
            if callable(value) and getattr(value, "__module__", None) == SESSION_PLAN_MODULE:
                allowed[name] = {"name": name, "module": SESSION_PLAN_MODULE}
        return allowed

    async def script_upload(self, **kwargs: Any) -> dict[str, Any]:
        """Run the bridge's install script exactly as the worker would.

        Queueserver executes an uploaded script with the worker namespace as
        both globals and locals; doing the same here means the plan that ends
        up in :attr:`namespace` — and therefore in ``plans_allowed`` — is the
        one the real wrapper builds, not a name parsed out of the script text.
        """
        await self._enter("script_upload", **kwargs)
        exec(kwargs["script"], self.namespace, self.namespace)  # noqa: S102
        return {"success": True, "task_uid": f"task-{len(self.kwargs_for('script_upload'))}"}

    async def task_result(self, **kwargs: Any) -> dict[str, Any]:
        await self._enter("task_result", **kwargs)
        return {"success": True, "status": "completed", "result": {"success": True, "msg": ""}}

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A bridge process with no inherited state: no backend, no draft, no records.

    The backend singleton is cleared rather than left alone so a test that
    forgets to install a mock gets a managerless (fail-closed) backend instead
    of the previous test's. `validation_records` is a process-wide singleton
    the whole suite shares, so its contents are saved and restored.
    """
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.delenv(_TOKEN_ENV, raising=False)
    monkeypatch.delenv(qb.QSERVER_CONTROL_ADDRESS_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))

    with validation_records.lock:
        saved_hashes = set(validation_records._passing_hashes)
        validation_records._passing_hashes.clear()

    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    app_module.set_queue_backend(None)
    set_session_uploader(None)
    yield
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    app_module.set_queue_backend(None)
    set_session_uploader(None)
    with validation_records.lock:
        validation_records._passing_hashes.clear()
        validation_records._passing_hashes.update(saved_hashes)


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | Exception], None]:
    """Set — or break — the ``control_system.type`` the capability check reads."""

    def _set(value: str | Exception) -> None:
        def fake_get_config_value(key: str, default: Any = None) -> Any:
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

    return _set


@pytest.fixture
def manager(connector: Callable[[str | Exception], None]) -> MockQueueServer:
    """An executable deployment: EPICS-like connector, reachable manager, env open."""
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))
    return mock


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _draft_revision(client: TestClient, num_points: int = 3) -> int:
    resp = client.patch(
        "/draft",
        json={
            "plan_name": "grid_scan",
            "plan_args_patch": _grid_scan_args(num_points),
            "client_id": "contract",
        },
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["revision"])


async def _draft_revision_direct(num_points: int = 3) -> int:
    """`_draft_revision` for the tests that drive route coroutines directly."""
    result = await draft.patch_draft(
        draft.PatchDraftRequest(
            plan_name="grid_scan",
            plan_args_patch=_grid_scan_args(num_points),
            client_id="contract",
        )
    )
    return int(result["revision"])


def _write_session_plan(name: str = _SESSION_PLAN_NAME, *, validated: bool = True) -> str:
    """Author a session plan on the bridge; optionally record it as passing."""
    from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir

    source = _session_plan_source(name)
    (resolve_session_plan_dir() / f"{name}.py").write_text(source, encoding="utf-8")
    if validated:
        validation_records.record(hash_plan_body(source))
    plan_loader.reset_facility_plans()
    return source


# ---------------------------------------------------------------------------
# The declared plan contract, as a consumer reads it off the catalog
# ---------------------------------------------------------------------------


def test_the_catalog_entry_is_five_keys_with_three_field_metadata(client: TestClient) -> None:
    """`GET /plans` publishes exactly five keys per entry, and a plan's metadata
    is exactly ``name``/``description``/``writes``.

    This is the wire half of the metadata contract: the model forbids extras
    in-process, but what a panel, the MCP tools and the approval hook actually
    read is this payload. `category` and `required_devices` are retired, so an
    entry that still carried either would be a consumer teaching an operator
    about a field the bridge no longer has.
    """
    body = client.get("/plans").json()
    by_name = {entry["name"]: entry for entry in body}
    assert {"orm", "grid_scan"} <= set(by_name), f"shipped plans missing: {sorted(by_name)}"

    for name, entry in by_name.items():
        assert set(entry) == {"name", "description", "schema", "metadata", "provenance"}, (
            f"{name}: unexpected catalog entry keys: {sorted(entry)}"
        )
        assert set(entry["metadata"]) == {"name", "description", "writes"}, (
            f"{name}: plan metadata is no longer the three declared fields: {entry['metadata']}"
        )


def test_the_published_schema_carries_the_plans_own_channel_roles(client: TestClient) -> None:
    """Roles reach a consumer as ``x-channel-role`` on the FIELD, not on its items.

    The role declaration is the whole contract: it is what replaced guessing a
    channel's purpose from its parameter name, and every consumer that has to
    know which channels a launch would MOVE (the approval gate, the pre-flight,
    the default figure) reads it from here. So the JSON path is pinned, not just
    the presence of the annotation — a role that moved into ``items`` would be
    invisible to every one of them while a laxer assertion still passed.

    ``grid_scan`` pins the nested case: its movable is a field of ``GridAxis``,
    which pydantic emits under ``$defs`` and references from ``axes.items``, so
    the role lives one level down and the array itself carries none.
    """
    by_name = {entry["name"]: entry for entry in client.get("/plans").json()}

    orm = by_name["orm"]["schema"]["properties"]
    assert orm["correctors"][CHANNEL_ROLE_KEY] == MOVABLE_ROLE
    assert orm["readbacks"][CHANNEL_ROLE_KEY] == READABLE_ROLE
    # The annotation sits beside `type`/`items`, never inside the item schema.
    assert CHANNEL_ROLE_KEY not in orm["correctors"]["items"], (
        f"the role moved into the item schema: {orm['correctors']}"
    )

    grid = by_name["grid_scan"]["schema"]
    assert grid["properties"]["readbacks"][CHANNEL_ROLE_KEY] == READABLE_ROLE
    assert grid["$defs"]["GridAxis"]["properties"]["setpoint"][CHANNEL_ROLE_KEY] == MOVABLE_ROLE
    assert CHANNEL_ROLE_KEY not in grid["properties"]["axes"], (
        "the axis LIST is not itself a channel field; only GridAxis.setpoint is: "
        f"{grid['properties']['axes']}"
    )


@pytest.mark.parametrize("retired_key", ["category", "required_devices"])
def test_session_authoring_refuses_a_retired_metadata_key(
    client: TestClient, retired_key: str
) -> None:
    """`POST /plans/session` 422s on a retired key rather than storing it.

    The write payload is the same three fields the catalog publishes. A stale
    client sending `category` gets told which key it sent — silently dropping
    it would let an author believe a declaration reached the plan file when
    nothing read it.
    """
    body = {
        "name": "retired_key_probe",
        "description": "probe",
        "writes": False,
        "body": "def build_plan(devices, params):\n    yield ('noop', 'probe')\n",
        retired_key: "scan",
    }

    resp = client.post("/plans/session", json=body)

    assert resp.status_code == 422, resp.text
    assert retired_key in resp.text, f"the refusal does not name the rejected key: {resp.text}"

    # The three-field body is what IS accepted — without this, the refusal above
    # could be passing for an unrelated reason.
    del body[retired_key]
    assert client.post("/plans/session", json=body).status_code == 200


# ---------------------------------------------------------------------------
# Enqueue from a pinned draft revision
# ---------------------------------------------------------------------------


def test_enqueue_puts_the_pinned_draft_in_the_managers_queue(
    client: TestClient, manager: MockQueueServer
) -> None:
    """The success path, body and all — and the negative control for every
    refusal test below: a 200 carries no ``code`` and no ``capability``."""
    revision = _draft_revision(client)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"run_id", "revision", "item", "armed"}
    assert body["revision"] == revision
    assert body["item"]["item_uid"] == "item-1"
    assert body["run_id"]
    # An idle, unarmed manager: the item waits for a start, and the add says so.
    assert body["armed"] is False
    # Negative controls: a success is never shaped like a refusal.
    assert "code" not in body
    assert "capability" not in body

    # The manager holds exactly the plan the DRAFT named, with OSPREY's run id
    # in the item metadata — the key start documents and Tiled results join on —
    # beside the plan-identity stamp a finished run is rendered from. Both keys
    # ride the item into the start document; the queue's own read surfaces relay
    # only the run id (`queue._public_item`).
    (item,) = manager.items
    assert item["name"] == "grid_scan"
    assert item["kwargs"] == _grid_scan_args()
    assert item["meta"] == {
        qb.RUN_ID_META_KEY: body["run_id"],
        qb.PLAN_META_KEY: {"name": "grid_scan", "kwargs": _grid_scan_args()},
    }


def test_a_draft_revision_can_be_enqueued_only_once(
    client: TestClient, manager: MockQueueServer
) -> None:
    """Once-per-revision, on the wire: the second attempt is a 409 naming the
    revision it refused, and the manager's queue still holds one item."""
    revision = _draft_revision(client)
    assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "draft_revision_already_launched"
    assert detail["revision"] == revision
    assert isinstance(detail["detail"], str) and detail["detail"]
    assert manager.item_uids() == ["item-1"]


def test_a_stale_draft_revision_never_reaches_the_manager(
    client: TestClient, manager: MockQueueServer
) -> None:
    revision = _draft_revision(client)
    _draft_revision(client, num_points=7)  # the draft moves on under the caller

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "stale_draft_revision"
    assert "item_add" not in manager.method_names()
    assert manager.items == []


def test_the_queue_read_publishes_a_bounded_status_summary(
    client: TestClient, manager: MockQueueServer
) -> None:
    """`GET /queue` reports the manager's queue, and only the status keys the
    contract names — the raw status document carries 0MQ material that must
    never reach a consumer."""
    revision = _draft_revision(client)
    assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200

    body = client.get("/queue").json()

    assert set(body) == {"status", "items", "running_item"}
    assert [item["item_uid"] for item in body["items"]] == ["item-1"]
    assert body["running_item"] is None
    assert set(body["status"]) == {"available", "runs_removed", *queue._SUMMARY_KEYS}
    assert body["status"]["available"] is True
    assert body["status"]["items_in_queue"] == 1
    assert "zmq_secret_key" not in json.dumps(body)


# ---------------------------------------------------------------------------
# Token gates
# ---------------------------------------------------------------------------


def test_starting_the_queue_requires_the_launch_token(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503 on an unarmed bridge, 403 on a wrong header, 200 with the token —
    and the manager is untouched until the token is right."""
    unarmed = client.post("/queue/start")
    assert unarmed.status_code == 503
    assert unarmed.json()["detail"]["code"] == "launch_token_required"

    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    wrong = client.post("/queue/start", headers={"X-Launch-Token": "not-the-token"})
    assert wrong.status_code == 403
    assert wrong.json()["detail"]["code"] == "launch_token_required"
    assert manager.calls == []

    armed = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})
    assert armed.status_code == 200
    body = armed.json()
    assert body["started"] is True
    assert body["armed"] is True
    assert "code" not in body
    # Nothing was queued, so the start is the arm alone: the manager stays
    # idle and reports autostart on, which is what runs the next item added.
    assert manager.autostart_enabled is True
    assert manager.manager_state == "idle"
    assert "queue_start" not in manager.method_names()


def test_enqueueing_onto_a_running_queue_requires_the_launch_token(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An item added to a draining queue executes with no further human action,
    so it is gated exactly like a start — and the refusal names the manager
    state that made it an armed operation."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    revision = _draft_revision(client)
    manager.manager_state = "executing_queue"

    refused = client.post("/queue/items", json={"draft_revision": revision})

    assert refused.status_code == 403
    detail = refused.json()["detail"]
    assert detail["code"] == "launch_token_required"
    assert detail["manager_state"] == "executing_queue"
    # Nothing was added, so nothing had to be withdrawn.
    assert "item_add" not in manager.method_names()
    assert "item_left_behind" not in detail
    assert manager.items == []

    armed = client.post(
        "/queue/items",
        json={"draft_revision": revision},
        headers={"X-Launch-Token": _TOKEN},
    )
    assert armed.status_code == 200, armed.text
    assert manager.item_uids() == ["item-1"]
    # ...and the add reports that it was the launch: the item is on its way.
    assert armed.json()["armed"] is True


def test_stopping_is_ungated_but_withdrawing_a_stop_requires_the_token(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Halting is always allowed; UN-halting lets the queue keep draining
    toward hardware, so it is an arming action."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)

    stopped = client.post("/queue/stop")
    assert stopped.status_code == 200
    assert stopped.json()["stop_pending"] is True
    assert "code" not in stopped.json()
    assert manager.stop_pending is True

    refused = client.post("/queue/stop", json={"cancel": True})
    assert refused.status_code == 403
    assert refused.json()["detail"]["code"] == "launch_token_required"
    # The withdrawal never reached the manager: the stop still stands.
    assert manager.stop_pending is True
    assert "queue_stop_cancel" not in manager.method_names()

    withdrawn = client.post(
        "/queue/stop", json={"cancel": True}, headers={"X-Launch-Token": _TOKEN}
    )
    assert withdrawn.status_code == 200
    assert withdrawn.json()["stop_pending"] is False
    assert manager.stop_pending is False


@pytest.mark.parametrize(
    ("token_configured", "expected_status"),
    [(False, 503), (True, 403)],
    ids=["unarmed-bridge", "wrong-token"],
)
def test_all_three_arming_routes_refuse_with_one_code(
    client: TestClient,
    manager: MockQueueServer,
    monkeypatch: pytest.MonkeyPatch,
    token_configured: bool,
    expected_status: int,
) -> None:
    """Start, enqueue-while-running, and stop-cancel are three routes but ONE
    refusal for a consumer: same ``launch_token_required`` code, same status
    code for the same cause. A panel branches once, not three times."""
    revision = _draft_revision(client)
    if token_configured:
        monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    headers = {"X-Launch-Token": "not-the-token"} if token_configured else {}
    manager.manager_state = "executing_queue"

    refusals = {
        "start": client.post("/queue/start", headers=headers),
        "enqueue": client.post("/queue/items", json={"draft_revision": revision}, headers=headers),
        "stop-cancel": client.post("/queue/stop", json={"cancel": True}, headers=headers),
    }

    for route, resp in refusals.items():
        assert resp.status_code == expected_status, f"{route}: {resp.text}"
        detail = resp.json()["detail"]
        assert detail["code"] == "launch_token_required", route
        assert isinstance(detail["detail"], str) and detail["detail"], route

    # None of the three arming actions reached the manager.
    names = manager.method_names()
    assert "queue_start" not in names
    assert "item_add" not in names
    assert "queue_stop_cancel" not in names
    assert draft._last_launched_revision == 0


# ---------------------------------------------------------------------------
# POST /queue/abort — the emergency halt, ungated on every axis
# ---------------------------------------------------------------------------


def _fast_abort_backend(mock: MockQueueServer) -> None:
    """Install a backend whose abort pause window closes without real waiting."""
    app_module.set_queue_backend(QueueBackend(mock, abort_pause_polls=4, abort_poll_interval=0))


def test_abort_stops_the_running_plan_against_a_manager_that_demands_a_pause(
    client: TestClient, manager: MockQueueServer
) -> None:
    """The two-party property. The mock enforces upstream's real constraint —
    ``re_abort`` is refused unless the Run Engine is paused — so a bridge that
    skipped the pause would fail here rather than pass against a permissive
    stand-in."""
    _fast_abort_backend(manager)
    manager.items.append({"item_uid": "u1", "name": "grid_scan", "kwargs": {}})
    manager.begin_running()

    resp = client.post("/queue/abort")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["aborted"] is True
    assert body["paused_first"] is True
    assert body["msg"] == "plan aborted"
    # Asserted against the mock's own state, not the calls it saw.
    assert manager.running_item is None
    assert manager.manager_state == "idle"
    # Negative control: a success body carries no refusal code.
    assert "code" not in body


def test_an_aborted_run_reads_as_stopped_and_cannot_silently_re_run(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What an operator sees, and what the machine does, after an emergency abort.

    Upstream leaves the aborted plan at the FRONT of the queue (see the mock's
    ``re_abort``). Two consumer-visible consequences are pinned together here
    because they are one story:

    * ``GET /runs`` must report that run ``stopped`` — the word the record
      contract reserves for "a human stopped it" — and NOT ``pending``, which
      would present a just-halted run as work still to come.
    * ``POST /queue/start`` must REFUSE while that item is queued, so the plan
      cannot go back on the hardware without a fresh, explicit decision.

    The abort itself stays ungated; only the arming action is gated. Asserted
    against the manager's own state, with the mock reproducing upstream's
    requeue, so this is a two-party property rather than a restatement of the
    bridge's assumptions.
    """
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    _fast_abort_backend(manager)
    manager.items.append(
        {
            "item_uid": "u1",
            "name": "grid_scan",
            "kwargs": {},
            "meta": {"osprey_run_id": "run-abort-1"},
        }
    )
    manager.begin_running()

    assert client.post("/queue/abort").status_code == 200

    # The manager really did put it back — otherwise the rest proves nothing.
    assert len(manager.items) == 1, manager.items
    assert manager.items[0]["result"]["exit_status"] == "aborted"

    record = client.get("/runs/run-abort-1")
    assert record.status_code == 200, record.text
    assert record.json()["status"] == "stopped", (
        "an emergency-aborted run must not be published as pending work"
    )

    refused = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "interrupted_item_in_queue"
    assert detail["exit_status"] == "aborted"
    # The load-bearing half: nothing was started.
    assert manager.manager_state == "idle"
    assert manager.running_item is None

    # ... and the operator's documented way out actually works: drop the item,
    # and the queue starts normally again. Without this, "refuse the start"
    # could be an unescapable dead end and the test above would not notice.
    client.delete(f"/queue/items/{manager.items[0]['item_uid']}")
    manager.items.append({"item_uid": "u2", "name": "grid_scan", "kwargs": {}})
    assert client.post("/queue/start", headers={"X-Launch-Token": _TOKEN}).status_code == 200


@pytest.mark.parametrize(
    ("token_configured", "headers"),
    [
        (False, {}),
        (True, {}),
        (True, {"X-Launch-Token": "not-the-token"}),
    ],
    ids=["unarmed-bridge", "armed-bridge-no-header", "armed-bridge-wrong-token"],
)
def test_abort_is_never_refused_for_a_token(
    client: TestClient,
    manager: MockQueueServer,
    monkeypatch: pytest.MonkeyPatch,
    token_configured: bool,
    headers: dict[str, str],
) -> None:
    """The inverse of ``test_all_three_arming_routes_refuse_with_one_code``.

    Those three postures make every arming route answer
    ``launch_token_required``. The abort must answer none of them: it is the
    emergency halt, and a halt that can be refused for a policy reason is a
    halt with a failure mode. A WRONG token is included deliberately — that is
    the posture in which a gate added "for consistency" would bite hardest.
    """
    if token_configured:
        monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    _fast_abort_backend(manager)
    manager.items.append({"item_uid": "u1", "name": "grid_scan", "kwargs": {}})
    manager.begin_running()

    resp = client.post("/queue/abort", headers=headers)

    assert resp.status_code == 200, resp.text
    assert resp.json()["aborted"] is True
    assert manager.running_item is None


def test_abort_survives_the_starting_queue_window(
    client: TestClient, manager: MockQueueServer
) -> None:
    """A pause is refused while the plan has not begun. The composition retries
    across that window rather than reporting a failed halt for a queue that was
    one tick from running."""
    _fast_abort_backend(manager)
    manager.items.append({"item_uid": "u1", "name": "grid_scan", "kwargs": {}})
    manager.manager_state = "starting_queue"

    # The worker picks the item up between the first refused pause and the next
    # poll, exactly as it does on a real deployment.
    original_re_pause = manager.re_pause
    seen: list[str] = []

    async def re_pause(**kwargs: Any) -> dict[str, Any]:
        seen.append(manager.manager_state)
        try:
            return await original_re_pause(**kwargs)
        except RequestFailedError:
            manager.begin_running()
            raise

    manager.re_pause = re_pause  # type: ignore[method-assign]

    resp = client.post("/queue/abort")

    assert resp.status_code == 200, resp.text
    assert seen[0] == "starting_queue"
    assert len(seen) >= 2, "the composition must retry the pause, not give up on one refusal"
    assert manager.manager_state == "idle"


def test_abort_with_nothing_running_refuses_in_the_uniform_shape(
    client: TestClient, manager: MockQueueServer
) -> None:
    """A consumer branches on ``detail.code`` here exactly as everywhere else,
    and nothing is sent to an idle manager."""
    _fast_abort_backend(manager)

    resp = client.post("/queue/abort")

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "nothing_running"
    assert isinstance(detail["detail"], str) and detail["detail"]
    assert "re_pause" not in manager.method_names()
    assert "re_abort" not in manager.method_names()


# ---------------------------------------------------------------------------
# The {check + add} / {start} lock, under concurrent interleaving
# ---------------------------------------------------------------------------


async def test_an_unarmed_add_racing_an_armed_start_is_refused_and_leaves_no_item(
    connector: Callable[[str | Exception], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FR4 acceptance race. An armed start holds the arming lock; an
    unarmed add arrives mid-start. The add must park on the lock (never reach
    the manager while a start is in flight), then be refused against the state
    the start produced — with the manager's queue still empty."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))
    # The arming call — with nothing queued, the start ARMS (autostart on) and
    # sends no `queue_start`; the mock flips to `starting_queue` on the arm so
    # the add's in-lock read sees the state the start produced.
    entered_start, release_start = mock.gate("queue_autostart")
    revision = await _draft_revision_direct()

    start_task = asyncio.create_task(queue.start_queue(x_launch_token=_TOKEN))
    await asyncio.wait_for(entered_start.wait(), timeout=5)
    # The start now holds the arming lock, blocked inside queue_autostart.

    add_task = asyncio.create_task(
        queue.add_queue_item(queue.QueueAddRequest(draft_revision=revision), x_launch_token="")
    )
    # Give the add every chance to run: it must park on the lock, not add.
    for _ in range(20):
        await asyncio.sleep(0)
    assert "item_add" not in mock.method_names()

    release_start.set()
    assert (await asyncio.wait_for(start_task, timeout=5))["started"] is True

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(add_task, timeout=5)
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail["code"] == "launch_token_required"
    # Idle but ARMED: the manager reports autostart on, and that alone is what
    # makes the unarmed add an execution and refuses it.
    assert excinfo.value.detail["manager_state"] == "idle"
    assert mock.autostart_enabled is True

    # The safety property, read off the manager rather than inferred: an
    # unarmed add during a start leaves NOTHING behind, and the revision is
    # still enqueueable once someone arms properly.
    assert mock.items == []
    assert "item_add" not in mock.method_names()
    assert draft._launching == set()
    assert draft._last_launched_revision == 0

    # Call ORDER, which is what the lock actually buys: the add's arming read
    # Call ORDER, which pins the lock geometry the refusal depends on. The add
    # was created after `queue_start` was already logged, so everything after
    # it in the log belongs to the add: its capability probe, then its in-lock
    # arming read — and nothing else. The add never reached the session gate or
    # the add itself while the start held the lock, and its arming read saw the
    # state the start produced rather than the one it replaced.
    names = mock.method_names()
    tail = mock.calls[names.index("queue_autostart") + 1 :]
    assert [name for name, _ in tail] == ["status", "status"]
    assert tail[-1] == ("status", {"reload": True})


async def test_an_out_of_band_start_between_the_add_and_the_recheck_withdraws_the_item(
    connector: Callable[[str | Exception], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense in depth against an actor the bridge's lock cannot serialize.
    The add is unarmed and legitimately pre-start when it enters the manager;
    the queue goes active underneath it anyway. The post-add re-check must
    withdraw the item and refuse — leaving the manager's queue empty."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))
    entered_add, release_add = mock.gate("item_add")
    revision = await _draft_revision_direct()

    add_task = asyncio.create_task(
        queue.add_queue_item(queue.QueueAddRequest(draft_revision=revision), x_launch_token="")
    )
    await asyncio.wait_for(entered_add.wait(), timeout=5)
    # Somebody outside this process started the queue while the add was in the
    # manager — exactly what the re-check exists to catch.
    mock.manager_state = "executing_queue"
    release_add.set()

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(add_task, timeout=5)
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail["code"] == "launch_token_required"
    assert excinfo.value.detail["manager_state"] == "executing_queue"
    # Withdrawal succeeded, so the refusal says nothing about a stranded item.
    assert "item_left_behind" not in excinfo.value.detail

    assert mock.items == []
    names = mock.method_names()
    assert names.index("item_add") < names.index("item_remove")
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


async def test_a_move_racing_a_bound_start_parks_on_the_arming_lock(
    connector: Callable[[str | Exception], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reorder cannot land inside a bound start's critical section.

    The start holds the arming lock across {uid compare … start}. A move
    arriving in that window must park on the same lock and reach the manager
    only afterwards, so the queue the manager is told to drain is the list the
    quoted uid named rather than one that moved under the compare.
    """
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))
    for num_points in (3, 4):
        revision = await _draft_revision_direct(num_points)
        await queue.add_queue_item(
            queue.QueueAddRequest(draft_revision=revision), x_launch_token=""
        )
    approved = await queue.get_queue()
    approved_uid = approved["status"]["plan_queue_uid"]
    assert [item["item_uid"] for item in approved["items"]] == ["item-1", "item-2"]

    # The start blocks inside `queue_start`: past the uid compare, still
    # holding the lock, with the manager not yet draining.
    entered_start, release_start = mock.gate("queue_start")
    start_task = asyncio.create_task(
        queue.start_queue(
            queue.QueueStartRequest(expected_plan_queue_uid=approved_uid),
            x_launch_token=_TOKEN,
        )
    )
    await asyncio.wait_for(entered_start.wait(), timeout=5)

    move_task = asyncio.create_task(
        queue.move_queue_item("item-2", queue.QueueMoveRequest(pos_dest="front"))
    )
    # Give the move every chance to run: it must park on the lock, not reorder.
    for _ in range(20):
        await asyncio.sleep(0)
    assert "item_move" not in mock.method_names()
    assert mock.item_uids() == ["item-1", "item-2"]

    release_start.set()
    assert (await asyncio.wait_for(start_task, timeout=5))["started"] is True
    assert (await asyncio.wait_for(move_task, timeout=5))["moved"] is True

    # Call ORDER, which is what the lock buys: the reorder reached the manager
    # only after the start had, so the started queue is the approved one and
    # the move changes what drains next rather than what was approved.
    names = mock.method_names()
    assert names.index("queue_start") < names.index("item_move")
    assert mock.item_uids() == ["item-2", "item-1"]
    assert mock.autostart_enabled is True


async def test_every_mutation_route_waits_on_the_module_arming_lock_itself(
    connector: Callable[[str | Exception], None],
) -> None:
    """One lock object, not one per route.

    The serialization a bound start relies on is a property of the single
    module-level `_arming_lock` — a per-route lock would read identically at
    every call site and serialize nothing. Holding that one object is enough to
    park a move, a removal and a clear at once, which is what says they wait on
    the same thing the start takes.
    """
    connector("virtual_accelerator")
    mock = MockQueueServer()
    mock.items = [
        {"item_uid": "item-1", "name": "grid_scan"},
        {"item_uid": "item-2", "name": "grid_scan"},
    ]
    app_module.set_queue_backend(QueueBackend(mock))

    await queue._arming_lock.acquire()
    tasks = [
        asyncio.create_task(
            queue.move_queue_item("item-1", queue.QueueMoveRequest(pos_dest="back"))
        ),
        asyncio.create_task(queue.remove_queue_item("item-2")),
        asyncio.create_task(queue.clear_queue_items()),
    ]
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        # Not one of them reached the manager while the lock was held.
        assert mock.method_names() == []
    finally:
        queue._arming_lock.release()

    for task in tasks:
        await asyncio.wait_for(task, timeout=5)
    assert sorted(mock.method_names()) == ["item_move", "item_remove", "queue_clear"]
    assert mock.items == []


async def test_two_enqueues_of_one_revision_yield_exactly_one_item(
    connector: Callable[[str | Exception], None],
) -> None:
    """Concurrent duplicates of one pinned revision: one 200, one 409, and the
    manager holds a single item."""
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))
    revision = await _draft_revision_direct()

    async def _enqueue() -> Any:
        try:
            return await queue.add_queue_item(
                queue.QueueAddRequest(draft_revision=revision), x_launch_token=""
            )
        except HTTPException as exc:
            return exc

    first, second = await asyncio.gather(_enqueue(), _enqueue())

    outcomes = [first, second]
    accepted = [result for result in outcomes if isinstance(result, dict)]
    refused = [result for result in outcomes if isinstance(result, HTTPException)]
    assert len(accepted) == 1 and len(refused) == 1
    assert refused[0].status_code == 409
    assert refused[0].detail["code"] == "draft_revision_already_launched"
    assert mock.item_uids() == ["item-1"]


# ---------------------------------------------------------------------------
# GET /queue/events — the SSE frame sequence
# ---------------------------------------------------------------------------


def _parse_frame(raw: str) -> dict[str, Any]:
    assert raw.startswith("data: "), raw
    return json.loads(raw[len("data: ") :])


def _assert_snapshot_frame(frame: dict[str, Any]) -> None:
    """Every frame on this stream is a full snapshot in one fixed shape."""
    assert set(frame) == {"type", "status", "items", "running_item"}
    assert frame["type"] in ("hello", "queue")
    assert set(frame["status"]) == {"available", "runs_removed", *queue._SUMMARY_KEYS}
    assert frame["status"]["available"] is True
    # Negative control: a snapshot is never a refusal, and never leaks the raw
    # status document's 0MQ material.
    assert "code" not in frame
    assert "zmq_secret_key" not in json.dumps(frame)


async def _frame_where(
    frames: Any, matches: Callable[[dict[str, Any]], bool], *, timeout: float = 5.0
) -> dict[str, Any]:
    """The next ``queue`` frame satisfying *matches*.

    Skips rather than fails on frames that do not match yet: the stream
    promises full snapshots, and it deliberately re-broadcasts one when a
    subscriber joins, so clients absorb duplicates idempotently. What is
    asserted here is the sequence of STATES the stream reports, not a frame
    count nobody can rely on.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    seen: list[Any] = []
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(f"no matching queue frame within {timeout}s; saw {seen}")
        raw = await asyncio.wait_for(frames.__anext__(), timeout=remaining)
        if raw.startswith(":"):  # heartbeat comment
            continue
        frame = _parse_frame(raw)
        _assert_snapshot_frame(frame)
        seen.append((frame["type"], frame["status"]["manager_state"], len(frame["items"])))
        if frame["type"] == "queue" and matches(frame):
            return frame


async def test_the_event_stream_reports_add_reorder_remove_start_and_finish(
    connector: Callable[[str | Exception], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame sequence a panel builds its live view from: a hello snapshot
    on connect, then a full snapshot for every queue change — bridge-side
    mutations and out-of-band worker transitions alike."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    monkeypatch.setattr(queue, "_POLL_INTERVAL_S", 0.01)
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))

    response = await queue.queue_events()
    frames = response.body_iterator
    try:
        hello = _parse_frame(await asyncio.wait_for(frames.__anext__(), timeout=5))
        _assert_snapshot_frame(hello)
        assert hello["type"] == "hello"
        assert hello["items"] == []
        assert hello["running_item"] is None

        # add, twice
        first = await _draft_revision_direct(num_points=3)
        await queue.add_queue_item(queue.QueueAddRequest(draft_revision=first), x_launch_token="")
        frame = await _frame_where(frames, lambda f: len(f["items"]) == 1)
        assert [item["item_uid"] for item in frame["items"]] == ["item-1"]

        second = await _draft_revision_direct(num_points=5)
        await queue.add_queue_item(queue.QueueAddRequest(draft_revision=second), x_launch_token="")
        frame = await _frame_where(frames, lambda f: len(f["items"]) == 2)
        assert [item["item_uid"] for item in frame["items"]] == ["item-1", "item-2"]

        # reorder
        await queue.move_queue_item("item-2", queue.QueueMoveRequest(pos_dest="front"))
        await _frame_where(
            frames, lambda f: [i["item_uid"] for i in f["items"]] == ["item-2", "item-1"]
        )

        # remove
        await queue.remove_queue_item("item-1")
        frame = await _frame_where(frames, lambda f: len(f["items"]) == 1)
        assert [item["item_uid"] for item in frame["items"]] == ["item-2"]

        # start
        await queue.start_queue(x_launch_token=_TOKEN)
        await _frame_where(frames, lambda f: f["status"]["manager_state"] == "starting_queue")

        # The worker picks the item up and finishes it. Neither transition goes
        # through the bridge, so only the poller can surface them.
        mock.begin_running()
        frame = await _frame_where(frames, lambda f: f["running_item"] is not None)
        assert frame["running_item"]["item_uid"] == "item-2"
        assert frame["items"] == []
        assert frame["status"]["manager_state"] == "executing_queue"

        mock.finish_running()
        frame = await _frame_where(frames, lambda f: f["status"]["items_in_history"] == 1)
        assert frame["running_item"] is None
        assert frame["items"] == []
        assert frame["status"]["manager_state"] == "idle"
    finally:
        await frames.aclose()

    # The last subscriber leaving stops the poller: no stream, no polling.
    assert queue._subscribers == set()
    assert queue._poller_task is None


async def test_the_event_stream_reports_a_manager_outage_without_dropping_the_stream(
    connector: Callable[[str | Exception], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manager that stops answering is a status the stream REPORTS, not an
    error that ends it — a panel keeps its connection and shows "unavailable"."""
    monkeypatch.setattr(queue, "_POLL_INTERVAL_S", 0.01)
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))

    response = await queue.queue_events()
    frames = response.body_iterator
    try:
        hello = _parse_frame(await asyncio.wait_for(frames.__anext__(), timeout=5))
        assert hello["type"] == "hello"
        assert hello["status"]["available"] is True

        mock.failures["status"] = RequestTimeoutError("no answer", {})

        while True:
            frame = _parse_frame(await asyncio.wait_for(frames.__anext__(), timeout=5))
            if frame["status"]["available"] is False:
                break
        assert frame["type"] == "queue"
        assert frame["status"]["reason"] == qb.REASON_MANAGER_UNREACHABLE
        assert frame["items"] == []
        assert frame["running_item"] is None
    finally:
        await frames.aclose()


# ---------------------------------------------------------------------------
# Capability fail-closed states
# ---------------------------------------------------------------------------

_FAIL_CLOSED_CASES = [
    pytest.param(qb.REASON_BROWSE_ONLY_CONNECTOR, 409, id="mock-connector"),
    pytest.param(qb.REASON_UNSUPPORTED_CONNECTOR, 409, id="unsupported-connector"),
    pytest.param(qb.REASON_CONFIG_UNREADABLE, 409, id="config-unreadable"),
    pytest.param(qb.REASON_MANAGER_NOT_CONFIGURED, 503, id="manager-not-configured"),
    pytest.param(qb.REASON_MANAGER_UNREACHABLE, 503, id="manager-unreachable"),
]


def _make_incapable(
    reason: str, mock: MockQueueServer, connector: Callable[[str | Exception], None]
) -> None:
    """Put the deployment into the state that yields *reason*.

    The two manager-side reasons are retryable outages (deploy the manager,
    bring it back); the three connector/config reasons are properties of the
    deployment itself, where retrying changes nothing.
    """
    if reason == qb.REASON_BROWSE_ONLY_CONNECTOR:
        connector("mock")
    elif reason == qb.REASON_UNSUPPORTED_CONNECTOR:
        connector("tango")
    elif reason == qb.REASON_CONFIG_UNREADABLE:
        connector(FileNotFoundError("no project config is mounted"))
    elif reason == qb.REASON_MANAGER_NOT_CONFIGURED:
        app_module.set_queue_backend(QueueBackend(None))
    elif reason == qb.REASON_MANAGER_UNREACHABLE:
        mock.failures["status"] = RequestTimeoutError("no answer from the queue server", {})
    else:  # pragma: no cover - guards the parametrization itself
        raise AssertionError(f"unhandled capability reason {reason!r}")


@pytest.mark.parametrize(("reason", "expected_status"), _FAIL_CLOSED_CASES)
def test_a_deployment_that_cannot_execute_refuses_to_hold_queue_items(
    client: TestClient,
    manager: MockQueueServer,
    connector: Callable[[str | Exception], None],
    reason: str,
    expected_status: int,
) -> None:
    """Every fail-closed capability state refuses enqueue, carrying the same
    capability record the status surface publishes — a browse-only deployment
    never holds items it could never run, and never burns the caller's draft
    revision doing so."""
    revision = _draft_revision(client)
    _make_incapable(reason, manager, connector)

    resp = client.post("/queue/items", json={"draft_revision": revision})

    assert resp.status_code == expected_status, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == reason
    assert detail["capability"]["can_execute"] is False
    assert detail["capability"]["reason"] == reason
    assert isinstance(detail["capability"]["detail"], str) and detail["capability"]["detail"]

    assert manager.items == []
    assert "item_add" not in manager.method_names()
    # The refusal is free: the pinned revision is still enqueueable once the
    # deployment is fixed (the sibling success test proves the same fixture
    # enqueues when the deployment is capable).
    assert draft._launching == set()
    assert draft._last_launched_revision == 0


@pytest.mark.parametrize(("reason", "expected_status"), _FAIL_CLOSED_CASES)
def test_health_publishes_the_capability_record_the_refusal_carries(
    client: TestClient,
    manager: MockQueueServer,
    connector: Callable[[str | Exception], None],
    reason: str,
    expected_status: int,
) -> None:
    """The cross-surface half of the contract: whatever `/health` says about
    this deployment is exactly what a refusal carries, and liveness never
    depends on it — a browse-only deployment is a HEALTHY deployment."""
    revision = _draft_revision(client)
    _make_incapable(reason, manager, connector)

    health = client.get("/health")
    refused = client.post("/queue/items", json={"draft_revision": revision})

    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["capability"]["can_execute"] is False
    assert body["capability"]["reason"] == reason
    assert body["capability"] == refused.json()["detail"]["capability"]


def test_an_executable_deployment_advertises_it_on_health(
    client: TestClient, manager: MockQueueServer
) -> None:
    """The positive control for the fail-closed set: `can_execute` is true only
    when a reachable manager sits behind a connector that can drive hardware."""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["capability"]["can_execute"] is True
    assert body["capability"]["reason"] == qb.REASON_EXECUTABLE


def test_a_browse_only_refusal_names_the_command_that_flips_it(
    client: TestClient,
    manager: MockQueueServer,
    connector: Callable[[str | Exception], None],
) -> None:
    """The mock-connector refusal is the one an operator meets most, so it
    carries the remediation, not just the diagnosis."""
    revision = _draft_revision(client)
    connector("mock")

    detail = client.post("/queue/items", json={"draft_revision": revision}).json()["detail"]

    assert detail["code"] == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert qb.FLIP_COMMAND in detail["capability"]["detail"]


def test_a_browse_only_deployment_cannot_be_started_either(
    client: TestClient,
    manager: MockQueueServer,
    connector: Callable[[str | Exception], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability gates the start too — an armed caller on a browse-only
    deployment is refused before the manager is asked to start anything."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    connector("mock")

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert "queue_start" not in manager.method_names()


# ---------------------------------------------------------------------------
# Session-plan admissibility
# ---------------------------------------------------------------------------


def test_a_validated_and_uploaded_session_plan_enqueues(
    client: TestClient, manager: MockQueueServer
) -> None:
    """The positive control for the session gate, through the real upload loop:
    author, validate, upload into the worker namespace, enqueue. Without this,
    the two refusals below could pass for the wrong reason."""
    source = _write_session_plan()
    upload = asyncio.run(upload_after_validation(_SESSION_PLAN_NAME))
    assert upload["uploaded"] is True, upload
    # The mock executed the bridge's install script, so the plan is really in
    # the worker namespace the manager publishes.
    assert manager.namespace[_SESSION_PLAN_NAME].__module__ == SESSION_PLAN_MODULE

    resp = client.patch("/draft", json={"plan_name": _SESSION_PLAN_NAME, "client_id": "contract"})
    assert resp.status_code == 200, resp.text
    revision = int(resp.json()["revision"])

    enqueued = client.post("/queue/items", json={"draft_revision": revision})

    assert enqueued.status_code == 200, enqueued.text
    assert "code" not in enqueued.json()
    (item,) = manager.items
    assert item["name"] == _SESSION_PLAN_NAME
    assert hash_plan_body(source) in validation_records._passing_hashes


def test_enqueue_refuses_a_session_plan_missing_from_the_worker_namespace(
    client: TestClient, manager: MockQueueServer
) -> None:
    """A validated session plan whose bytes are not in the live namespace —
    the state a bridge restart or an environment rebuild leaves behind — is
    refused, never repaired: executing bytes no live validation vouches for is
    the thing this gate exists to prevent."""
    _write_session_plan()
    resp = client.patch("/draft", json={"plan_name": _SESSION_PLAN_NAME, "client_id": "contract"})
    assert resp.status_code == 200, resp.text
    revision = int(resp.json()["revision"])

    refused = client.post("/queue/items", json={"draft_revision": revision})

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == REASON_NOT_IN_NAMESPACE
    # The error's native key rides along; "code" is the canonical duplicate the
    # whole queue surface branches on.
    assert detail["reason"] == detail["code"]
    assert detail["plan"] == _SESSION_PLAN_NAME
    assert "item_add" not in manager.method_names()
    assert manager.items == []
    assert draft._last_launched_revision == 0


def test_starting_the_queue_refuses_an_unvalidated_session_plan(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The start gate re-checks every item the start would drain. One session
    plan with no passing record for its current bytes refuses the WHOLE start —
    all-or-nothing — and the manager is never asked to start."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    _write_session_plan(validated=False)
    manager.items = [
        {"item_uid": "item-1", "name": "grid_scan"},
        {"item_uid": "item-2", "name": _SESSION_PLAN_NAME},
    ]

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == REASON_UNVALIDATED
    assert detail["plan"] == _SESSION_PLAN_NAME
    assert "queue_start" not in manager.method_names()
    assert manager.manager_state == "idle"


def test_enqueueing_an_edited_after_pass_session_plan_refuses_with_a_coded_body(
    client: TestClient, manager: MockQueueServer
) -> None:
    """The pre-lock validation gate re-hashes the file and refuses ahead of the
    namespace check, so for an edited-after-pass plan IT is the refusal a
    consumer receives. It must still carry `session_plan_unvalidated` — a bare
    string here would make that code unreachable at enqueue for exactly the
    case it names, while `detail.code` reads as absent to every consumer."""
    source = _write_session_plan()
    resp = client.patch("/draft", json={"plan_name": _SESSION_PLAN_NAME, "client_id": "contract"})
    assert resp.status_code == 200, resp.text
    revision = int(resp.json()["revision"])

    # Edit the file after it passed, without re-validating: the recorded hash
    # no longer matches the bytes on disk.
    from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir

    edited = source.replace('"description":', '"description":  ', 1) + "\n# edited\n"
    assert hash_plan_body(edited) != hash_plan_body(source)
    (resolve_session_plan_dir() / f"{_SESSION_PLAN_NAME}.py").write_text(edited, encoding="utf-8")

    refused = client.post("/queue/items", json={"draft_revision": revision})

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == REASON_UNVALIDATED
    assert detail["reason"] == detail["code"]
    assert detail["plan"] == _SESSION_PLAN_NAME
    assert "no passing validation record" in detail["detail"]
    # Nothing reached the manager, and the revision was not consumed.
    assert "item_add" not in manager.method_names()
    assert manager.items == []
    assert draft._last_launched_revision == 0


# ---------------------------------------------------------------------------
# The retired direct-execute routes
# ---------------------------------------------------------------------------

# Route, and the queue route that took its capability over. Parametrized so a
# route leaving (or joining) the refusal set cannot go unasserted.
_RETIRED_ROUTES = [
    ("/runs", "POST /queue/items"),
    ("/runs/some-run-id/launch", "POST /queue/start"),
    ("/draft/run", "POST /queue/items"),
    ("/runs/some-run-id/stop", "POST /queue/stop"),
]


@pytest.mark.parametrize(("path", "replacement"), _RETIRED_ROUTES)
def test_a_retired_direct_execute_route_refuses_with_use_the_queue(
    client: TestClient, manager: MockQueueServer, path: str, replacement: str
) -> None:
    """410 Gone with the surface's own refusal body, not a 404: these routes
    used to run plans in the bridge process, and a consumer still pointed at
    one has to learn which route replaced it. 410 rather than the 4xx the
    original brief left open — "the resource is gone" is the precise answer,
    and it cannot be confused with "you asked for a run that does not exist".
    """
    resp = client.post(path, json={})

    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert detail["code"] == "use_the_queue"
    assert replacement in detail["detail"]
    # A refusal that is a property of the deployment's shape, not its state:
    # nothing is asked of the manager, and no item is touched.
    assert manager.method_names() == []
    assert manager.items == []


def test_holding_the_launch_token_does_not_revive_a_retired_route(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control for the arming tests above: every other refusal on
    this surface flips to a 200 for a token holder. These do not — there is no
    in-process execution left to arm, so the token buys nothing here."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)

    for path, _ in _RETIRED_ROUTES:
        resp = client.post(path, json={}, headers={"X-Launch-Token": _TOKEN})
        assert resp.status_code == 410, path
        assert resp.json()["detail"]["code"] == "use_the_queue", path
    assert manager.items == []


def test_starting_a_queue_of_catalog_plans_is_not_gated_on_session_records(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control for the start gate: catalog plans are operator-
    supplied and carry no record gate, so an identical queue without a session
    plan starts."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    manager.items = [{"item_uid": "item-1", "name": "grid_scan"}]

    resp = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 200, resp.text
    assert "code" not in resp.json()
    assert manager.manager_state == "starting_queue"


# ---------------------------------------------------------------------------
# Binding a start to the queue that was approved
# ---------------------------------------------------------------------------


def _queue_uid(client: TestClient) -> str:
    """The manager's ``plan_queue_uid`` as a consumer reads it off `GET /queue`."""
    resp = client.get("/queue")
    assert resp.status_code == 200, resp.text
    uid = resp.json()["status"]["plan_queue_uid"]
    assert isinstance(uid, str) and uid
    return uid


def test_a_start_naming_a_queue_that_has_moved_is_refused_with_the_current_uid(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start carries the uid of the queue its approver read. Re-ordering the
    queue in between moves that uid, and the start is refused rather than
    silently re-bound to a list nobody approved — with the CURRENT uid in the
    body, so the caller can re-read and try again without guessing."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    for num_points in (3, 4):
        revision = _draft_revision(client, num_points)
        assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200
    approved_uid = _queue_uid(client)

    moved = client.post("/queue/items/item-2/move", json={"pos_dest": "front"})
    assert moved.status_code == 200, moved.text
    current_uid = _queue_uid(client)
    assert current_uid != approved_uid

    refused = client.post(
        "/queue/start",
        json={"expected_plan_queue_uid": approved_uid},
        headers={"X-Launch-Token": _TOKEN},
    )

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == "queue_changed_since_approval"
    assert detail["plan_queue_uid"] == current_uid
    assert detail["expected_plan_queue_uid"] == approved_uid
    # The refusal armed nothing and drained nothing: the queue is exactly the
    # re-ordered list, and the manager was never asked to start.
    assert [item["item_uid"] for item in manager.items] == ["item-2", "item-1"]
    assert manager.autostart_enabled is False
    assert "queue_start" not in manager.method_names()


def test_a_start_that_names_no_queue_arms_it_as_before(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control, and the compatibility contract: `expected_plan_
    queue_uid` is optional. A start with no body — and one with an explicit
    null — arms a queue that has moved since it was last read, exactly as every
    shipped script and the e2e driver already expect."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    revision = _draft_revision(client)
    assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200
    stale_uid = _queue_uid(client)
    assert client.post("/queue/items/item-1/move", json={"pos_dest": "front"}).status_code == 200
    assert _queue_uid(client) != stale_uid

    bodyless = client.post("/queue/start", headers={"X-Launch-Token": _TOKEN})

    assert bodyless.status_code == 200, bodyless.text
    assert bodyless.json()["started"] is True
    assert "code" not in bodyless.json()
    assert manager.autostart_enabled is True

    explicit_null = client.post(
        "/queue/start",
        json={"expected_plan_queue_uid": None},
        headers={"X-Launch-Token": _TOKEN},
    )

    assert explicit_null.status_code == 200, explicit_null.text
    assert "code" not in explicit_null.json()


def test_a_uid_taken_before_a_move_earns_one_409_and_the_retry_starts(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole loop a client is meant to run: refuse once, re-read, start.

    The refusal is not sticky and the bridge never re-binds by itself — the
    second attempt succeeds only because the caller quoted the uid the refusal
    handed back, which is the point at which a human has looked at the new
    queue."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    for num_points in (3, 4):
        revision = _draft_revision(client, num_points)
        assert client.post("/queue/items", json={"draft_revision": revision}).status_code == 200
    approved_uid = _queue_uid(client)
    assert client.post("/queue/items/item-2/move", json={"pos_dest": "front"}).status_code == 200

    attempts = [
        client.post(
            "/queue/start",
            json={"expected_plan_queue_uid": approved_uid},
            headers={"X-Launch-Token": _TOKEN},
        )
        for _ in range(2)
    ]

    # Exactly one refusal per attempt on the stale uid — the gate is stateless,
    # so repeating the stale start repeats the same answer and never leaks into
    # an arm.
    assert [resp.status_code for resp in attempts] == [409, 409]
    fresh_uid = attempts[-1].json()["detail"]["plan_queue_uid"]
    assert manager.autostart_enabled is False

    started = client.post(
        "/queue/start",
        json={"expected_plan_queue_uid": fresh_uid},
        headers={"X-Launch-Token": _TOKEN},
    )

    assert started.status_code == 200, started.text
    body = started.json()
    assert set(body) == {"started", "armed", "msg"}
    assert body["started"] is True and body["armed"] is True
    assert manager.autostart_enabled is True
    assert manager.manager_state == "starting_queue"


def test_the_uid_a_read_publishes_names_the_queue_that_read_returned(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GET /queue` answers with the uid of the list it is answering with.

    The route's status read bypasses the client's status cache, so the uid a
    caller stamps an approval with describes the items it rendered beside it.
    A cached uid would be the pre-add one while the list is the post-add one,
    and the bound start built on it would refuse a queue nobody touched: an
    add onto an armed queue takes its one status read BEFORE the add, so the
    cache holds the older document for as long as it lives.
    """
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    # An armed queue — whatever armed it — makes the add below an armed add.
    manager.autostart_enabled = True
    before_uid = _queue_uid(client)
    manager.cache_status()

    revision = _draft_revision(client)
    added = client.post(
        "/queue/items",
        json={"draft_revision": revision},
        headers={"X-Launch-Token": _TOKEN},
    )
    assert added.status_code == 200, added.text

    read = client.get("/queue")
    assert read.status_code == 200, read.text
    body = read.json()
    uid = body["status"]["plan_queue_uid"]
    # The uid moved with the list it is published beside, rather than lagging
    # behind it.
    assert uid != before_uid
    assert [item["item_uid"] for item in body["items"]] == manager.item_uids()

    started = client.post(
        "/queue/start",
        json={"expected_plan_queue_uid": uid},
        headers={"X-Launch-Token": _TOKEN},
    )

    assert started.status_code == 200, started.text
    assert started.json()["started"] is True
    assert manager.manager_state == "starting_queue"


# ---------------------------------------------------------------------------
# Who an enqueued item belongs to
# ---------------------------------------------------------------------------


def test_an_owner_claim_is_trusted_on_its_own_and_gates_nothing(
    client: TestClient, manager: MockQueueServer
) -> None:
    """``X-Osprey-Owner`` is attribution, never authority.

    An idle lane takes an unarmed add, and naming an owner neither arms it nor
    is refused for carrying no launch token: who an item belongs to and what
    an add may do are separate questions, answered by the header and by the
    token against the manager's state. The claim reaches the manager twice —
    as the reserved kwarg the worker binds, and as the metadata stamp that
    still names the owner once the item is history — while the plan-identity
    copy a finished run is rendered from stays the plan's own arguments.
    """
    revision = _draft_revision(client)

    resp = client.post(
        "/queue/items",
        json={"draft_revision": revision},
        headers={OWNER_HEADER: "bob"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["armed"] is False
    (item,) = manager.items
    assert item["kwargs"][RESERVED_OWNER_KWARG] == "bob"
    assert item["meta"][qb.OWNER_META_KEY] == "bob"
    assert item["meta"][qb.PLAN_META_KEY]["kwargs"] == _grid_scan_args()


def test_an_owner_claim_the_guard_refuses_costs_the_attribution_not_the_add(
    client: TestClient, manager: MockQueueServer
) -> None:
    """A claim outside the allowlist — here an unexpanded shell placeholder —
    enqueues as an owner-less add.

    The header is read through the one shared guard, which answers ``None``
    for every shape it will not pass and raises nothing, so a caller with a
    name nobody can render loses the attribution and keeps the enqueue.
    """
    revision = _draft_revision(client)

    resp = client.post(
        "/queue/items",
        json={"draft_revision": revision},
        headers={OWNER_HEADER: "${OSPREY_OWNER}"},
    )

    assert resp.status_code == 200, resp.text
    (item,) = manager.items
    assert item["kwargs"] == _grid_scan_args()
    assert qb.OWNER_META_KEY not in item["meta"]


def test_every_queue_read_surface_names_the_owner_and_never_the_reserved_kwarg(
    client: TestClient, manager: MockQueueServer
) -> None:
    """What a consumer of this surface can rely on, on every body carrying an item.

    A row says who queued it under ``owner``, and the key that carries the name
    to the worker is never on the wire: not on the add echo, not on a read, not
    on the remove echo. A client that replayed one would be claiming to be
    somebody, and it is not a plan argument to render either.
    """
    revision = _draft_revision(client)
    added = client.post(
        "/queue/items", json={"draft_revision": revision}, headers={OWNER_HEADER: "bob"}
    )
    assert added.status_code == 200, added.text
    (held,) = manager.items

    listed = client.get("/queue")
    removed = client.delete(f"/queue/items/{held['item_uid']}")

    # The control: the item the manager holds keeps the kwarg, which is the one
    # path by which the run itself learns whose it is.
    assert held["kwargs"][RESERVED_OWNER_KWARG] == "bob"

    assert added.json()["item"]["owner"] == "bob"
    (row,) = listed.json()["items"]
    assert row["owner"] == "bob"
    assert row["kwargs"] == _grid_scan_args()
    assert removed.json()["item"]["owner"] == "bob"
    for body in (added.text, listed.text, removed.text):
        assert RESERVED_OWNER_KWARG not in body, body


def test_an_external_worker_lane_names_the_owner_from_the_metadata_stamp(
    client: TestClient, manager: MockQueueServer
) -> None:
    """A facility RE Manager validates every add against the facility plan's own
    signature, so such a lane carries no reserved kwarg to lift. The owner is
    stamped on the item's metadata instead, and the row a consumer reads is the
    same shape on either lane."""
    manager.items.append(
        {
            "item_uid": "facility-1",
            "item_type": "plan",
            "name": "count",
            "kwargs": {"num": 3},
            "meta": {qb.OWNER_META_KEY: "bob"},
        }
    )

    (row,) = client.get("/queue").json()["items"]

    assert row["owner"] == "bob"
    assert row["kwargs"] == {"num": 3}


async def test_the_event_stream_names_the_owner_of_the_items_it_carries(
    connector: Callable[[str | Exception], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream is what a live panel renders from, so both rules hold on a
    frame exactly as they hold on a read."""
    monkeypatch.setattr(queue, "_POLL_INTERVAL_S", 0.01)
    connector("virtual_accelerator")
    mock = MockQueueServer()
    app_module.set_queue_backend(QueueBackend(mock))
    revision = await _draft_revision_direct()
    await queue.add_queue_item(
        queue.QueueAddRequest(draft_revision=revision), x_launch_token="", x_osprey_owner="bob"
    )

    response = await queue.queue_events()
    frames = response.body_iterator
    try:
        hello = _parse_frame(await asyncio.wait_for(frames.__anext__(), timeout=5))
    finally:
        await frames.aclose()

    _assert_snapshot_frame(hello)
    (streamed,) = hello["items"]
    assert streamed["owner"] == "bob"
    assert streamed["kwargs"] == _grid_scan_args()
    assert RESERVED_OWNER_KWARG not in json.dumps(hello)


def test_an_owner_named_in_the_request_body_is_not_an_owner(
    client: TestClient, manager: MockQueueServer
) -> None:
    """Attribution is read from the header and from nowhere else.

    A body field named ``owner``, and the reserved spelling beside it, name
    nobody: the enqueued item reaches the manager owner-less on both carriers.
    A second channel would be one nothing guards — the header goes through the
    shared reader that decides what counts as a name.
    """
    revision = _draft_revision(client)

    resp = client.post(
        "/queue/items",
        json={"draft_revision": revision, "owner": "root", RESERVED_OWNER_KWARG: "root"},
    )

    assert resp.status_code == 200, resp.text
    (item,) = manager.items
    assert item["kwargs"] == _grid_scan_args()
    assert qb.OWNER_META_KEY not in item["meta"]
    assert "owner" not in resp.json()["item"]


# ---------------------------------------------------------------------------
# The reserved kwarg on the wire: one walk over the whole surface
# ---------------------------------------------------------------------------

_SURFACE_PREFIXES = ("queue", "history", "runs")
_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")


def _queue_surface_routes() -> set[tuple[str, str]]:
    """Every queue, history and runs route the app publishes, read at test time.

    Taken from the app's own schema rather than listed by hand: a route added
    to this surface later arrives here with no driver in the walk below, and
    the walk fails naming it instead of passing over a hole.
    """
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in _HTTP_METHODS and path.split("/")[1] in _SURFACE_PREFIXES
    }


def _assert_no_reserved_key(payload: Any, where: str, path: str = "") -> None:
    """The reserved owner kwarg appears nowhere in *payload*, at any depth.

    Keys, values and the text of any string are all checked, because all three
    are ways a client could read the name of the channel back off the wire —
    a refusal quoting the kwargs it rejected leaks it as readily as a queue row
    carrying it.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key != RESERVED_OWNER_KWARG, f"{where}: reserved key at {path}.{key}"
            _assert_no_reserved_key(value, where, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_no_reserved_key(value, where, f"{path}[{index}]")
    elif isinstance(payload, str):
        assert RESERVED_OWNER_KWARG not in payload, f"{where}: reserved key in the text at {path}"


def test_no_route_of_this_surface_puts_the_reserved_owner_kwarg_on_the_wire(
    client: TestClient, manager: MockQueueServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one walk: every body this surface can answer with, scanned to the leaf.

    The rule is not that the relays strip the key from the shapes somebody
    thought of — it is that no client of this bridge ever sees it: not on a
    read, a stream frame, a run record, a mutation echo or a refusal, and not
    nested inside one. So the routes are read off the app and every one of them
    is driven here, with an owned item queued, another running and a third in
    history, so the bodies walked are the bodies a live panel renders.

    The control is the manager's own item, which keeps the kwarg throughout:
    that is the single channel by which a run learns whose it is, and it is why
    there is something for each of these bodies to leak. What `add_item` writes
    onto that item — the reserved kwarg, the metadata stamp and the filtered
    plan-identity copy the worker republishes in the run's start document — is
    pinned in `tests/services/bluesky_bridge/test_queue_meta_strip.py`, which
    is the write side of this same rule.
    """
    monkeypatch.setattr(queue, "_POLL_INTERVAL_S", 0.01)
    walked: set[tuple[str, str]] = set()

    def drive(method: str, template: str, *, url: str | None = None, **kwargs: Any) -> Any:
        target = url or template
        resp = client.request(method, target, **kwargs)
        walked.add((method, template))
        label = f"{method} {target} -> {resp.status_code}"
        try:
            body = resp.json()
        except ValueError:  # a non-JSON body (an export download) is scanned raw
            assert RESERVED_OWNER_KWARG not in resp.text, label
            return resp
        _assert_no_reserved_key(body, label)
        return resp

    for num_points in (3, 4, 5):
        added = drive(
            "POST",
            "/queue/items",
            json={"draft_revision": _draft_revision(client, num_points)},
            headers={OWNER_HEADER: "bob"},
        )
        assert added.status_code == 200, added.text
    assert added.json()["item"]["owner"] == "bob"

    # One item through to history, one running, one still pending: the three
    # states a run record is built from, all owned.
    manager.begin_running()
    manager.finish_running()
    manager.begin_running()
    pending_uid = manager.items[0]["item_uid"]
    assert all(item["kwargs"][RESERVED_OWNER_KWARG] == "bob" for item in manager.history)

    listed = drive("GET", "/queue")
    assert [row["owner"] for row in listed.json()["items"]] == ["bob"]
    assert listed.json()["running_item"]["owner"] == "bob"

    records = drive("GET", "/runs").json()
    assert len(records) == 3
    assert {record["owner"] for record in records} == {"bob"}
    run_id = records[0]["id"]

    drive("GET", "/runs/{run_id}", url=f"/runs/{run_id}")
    drive("GET", "/runs/{run_id}/data", url=f"/runs/{run_id}/data")
    drive("GET", "/runs/{run_id}/export", url=f"/runs/{run_id}/export")
    drive("GET", "/runs/{run_id}/figure", url=f"/runs/{run_id}/figure")
    drive("POST", "/runs", json={})
    drive("POST", "/runs/{run_id}/launch", url=f"/runs/{run_id}/launch")
    drive("POST", "/runs/{run_id}/stop", url=f"/runs/{run_id}/stop")

    # The stream a panel renders live, ended after its hello frame by pushing
    # the module's own disconnect sentinel into the new subscriber: `TestClient`
    # buffers a response to completion, so an endless stream would hang.
    subscribe = queue._subscribe

    async def _subscribe_then_end() -> tuple[Any, dict[str, Any]]:
        subscriber, hello = await subscribe()
        subscriber.put_nowait(queue._DISCONNECT)
        return subscriber, hello

    monkeypatch.setattr(queue, "_subscribe", _subscribe_then_end)
    with client.stream("GET", "/queue/events") as stream:
        assert stream.status_code == 200
        raw_frames = [line for line in stream.iter_lines() if line.startswith("data: ")]
    walked.add(("GET", "/queue/events"))
    frames = [_parse_frame(raw) for raw in raw_frames]
    for frame in frames:
        _assert_no_reserved_key(frame, "GET /queue/events")
    assert frames and all(row["owner"] == "bob" for frame in frames for row in frame["items"])

    moved = drive(
        "POST",
        "/queue/items/{uid}/move",
        url=f"/queue/items/{pending_uid}/move",
        json={"pos_dest": "front"},
    )
    assert moved.json()["item"]["owner"] == "bob"
    removed = drive("DELETE", "/queue/items/{uid}", url=f"/queue/items/{pending_uid}")
    assert removed.json()["item"]["owner"] == "bob"

    drive("POST", "/queue/start", json={})
    drive("POST", "/queue/stop", json={})
    drive("POST", "/queue/abort")
    drive("DELETE", "/queue/items")
    drive("DELETE", "/runs/{run_id}", url=f"/runs/{run_id}")
    drive("DELETE", "/history")

    assert walked == _queue_surface_routes()
