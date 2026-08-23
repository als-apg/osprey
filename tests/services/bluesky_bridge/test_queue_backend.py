"""Contract tests for the bridge's queueserver adapter (`queue_backend.py`).

Everything here runs against a scripted stand-in for
``bluesky_queueserver_api``'s ``REManagerAPI`` — no manager, no Redis, no
container. That is the point: this is OSPREY's half of a two-party contract, so
these tests pin the translation (which manager call each method makes, and which
typed error each manager failure becomes) and the two pieces of policy the
adapter owns on top of it — environment lifecycle and capability derivation.
The other half, that a real manager behaves the way this stand-in does, is the
container-bound e2e's job.
"""

from __future__ import annotations

from typing import Any

import pytest
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.comm_base import RequestFailedError, RequestTimeoutError

from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.queue_backend import (
    Capability,
    EnvironmentUnavailableError,
    ExecutionUnavailableError,
    QueueBackend,
    QueueRequestRejectedError,
    QueueUnavailableError,
)


class FakeManager:
    """A scripted ``REManagerAPI`` stand-in.

    Records every call as ``(method, kwargs)`` and answers from ``responses``.
    A response that is an exception instance is raised instead of returned; a
    response that is a list is consumed one call at a time (the tail repeats),
    which is how the environment-open polling loop gets a state sequence.
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
        "running_item_uid": None,
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch):
    """Set (or break) the ``control_system.type`` the adapter resolves.

    Both `_resolve_connector_type` and the shared
    `_resolve_control_system_type` it delegates to import `get_config_value`
    inside the function body, so patching it at its home module covers both.
    """

    def _set(value: str | Exception) -> None:
        def fake_get_config_value(key: str, default: Any = None) -> Any:
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

    return _set


@pytest.fixture
def fast_backend():
    """Build a backend whose environment-open window is short enough to test."""

    def _build(manager: Any, **kwargs: Any) -> QueueBackend:
        kwargs.setdefault("env_open_attempts", 2)
        kwargs.setdefault("env_open_polls", 3)
        kwargs.setdefault("env_poll_interval", 0)
        return QueueBackend(manager, **kwargs)

    return _build


# --------------------------------------------------------------- error mapping


async def test_manager_timeout_becomes_queue_unavailable() -> None:
    manager = FakeManager(status=RequestTimeoutError("no answer", {}))
    with pytest.raises(QueueUnavailableError) as excinfo:
        await QueueBackend(manager).status()
    assert excinfo.value.reason == qb.REASON_MANAGER_UNREACHABLE


async def test_manager_refusal_becomes_request_rejected() -> None:
    manager = FakeManager(item_remove=RequestFailedError({}, {"msg": "unknown uid"}))
    with pytest.raises(QueueRequestRejectedError) as excinfo:
        await QueueBackend(manager).remove("nope")
    assert "unknown uid" in str(excinfo.value)
    assert excinfo.value.reason == "queue_request_rejected"


async def test_every_call_fails_closed_without_a_manager() -> None:
    backend = QueueBackend(None)
    for call in (backend.status(), backend.items(), backend.history(), backend.start()):
        with pytest.raises(QueueUnavailableError):
            await call


async def test_close_is_safe_without_a_manager() -> None:
    await QueueBackend(None).close()

    manager = FakeManager()
    await QueueBackend(manager).close()
    assert manager.closed


# ----------------------------------------------------------------- queue reads


async def test_status_passes_reload_through() -> None:
    manager = FakeManager(status=status_doc())
    backend = QueueBackend(manager)

    await backend.status()
    await backend.status(reload=True)

    assert manager.kwargs_for("status") == [{"reload": False}, {"reload": True}]


async def test_items_and_history_read_the_manager() -> None:
    queue = {"success": True, "items": [{"item_uid": "a"}], "running_item": {}}
    history = {"success": True, "items": [{"item_uid": "old"}]}
    backend = QueueBackend(FakeManager(queue_get=queue, history_get=history))

    assert (await backend.items())["items"] == [{"item_uid": "a"}]
    assert (await backend.history())["items"] == [{"item_uid": "old"}]


# -------------------------------------------------------------- queue mutation


async def test_add_item_threads_the_run_id_through_item_metadata() -> None:
    manager = FakeManager()
    await QueueBackend(manager).add_item({"item_type": "plan", "name": "count"}, run_id="run-7")

    (kwargs,) = manager.kwargs_for("item_add")
    assert kwargs["item"]["meta"] == {
        qb.RUN_ID_META_KEY: "run-7",
        qb.PLAN_META_KEY: {"name": "count", "kwargs": {}},
    }


async def test_add_item_stamps_the_plan_kwargs_as_enqueued() -> None:
    manager = FakeManager()
    item = {"item_type": "plan", "name": "grid_scan", "kwargs": {"num": 5, "readbacks": ["det"]}}
    await QueueBackend(manager).add_item(item, run_id="run-7")

    (kwargs,) = manager.kwargs_for("item_add")
    assert kwargs["item"]["meta"][qb.PLAN_META_KEY] == {
        "name": "grid_scan",
        "kwargs": {"num": 5, "readbacks": ["det"]},
    }


async def test_add_item_preserves_existing_metadata() -> None:
    manager = FakeManager()
    item = {"item_type": "plan", "name": "count", "meta": {"operator": "ada"}}
    await QueueBackend(manager).add_item(item, run_id="run-7")

    (kwargs,) = manager.kwargs_for("item_add")
    assert kwargs["item"]["meta"] == {
        "operator": "ada",
        qb.RUN_ID_META_KEY: "run-7",
        qb.PLAN_META_KEY: {"name": "count", "kwargs": {}},
    }
    # The caller's own dict is never mutated.
    assert item["meta"] == {"operator": "ada"}


async def test_add_item_accepts_a_bplan() -> None:
    manager = FakeManager()
    await QueueBackend(manager).add_item(BPlan("count", ["det"], num=3), run_id="run-9")

    (kwargs,) = manager.kwargs_for("item_add")
    assert kwargs["item"]["name"] == "count"
    assert kwargs["item"]["kwargs"] == {"num": 3}
    assert kwargs["item"]["meta"] == {
        qb.RUN_ID_META_KEY: "run-9",
        qb.PLAN_META_KEY: {"name": "count", "kwargs": {"num": 3}},
    }


async def test_add_item_without_a_run_id_adds_no_metadata() -> None:
    manager = FakeManager()
    await QueueBackend(manager).add_item({"item_type": "plan", "name": "count"})

    (kwargs,) = manager.kwargs_for("item_add")
    assert "meta" not in kwargs["item"]


async def test_add_item_forwards_only_the_placement_it_was_given() -> None:
    manager = FakeManager()
    await QueueBackend(manager).add_item({"name": "count"}, pos="front")

    (kwargs,) = manager.kwargs_for("item_add")
    assert kwargs["pos"] == "front"
    assert "before_uid" not in kwargs and "after_uid" not in kwargs


async def test_add_item_rejects_a_non_item() -> None:
    with pytest.raises(QueueRequestRejectedError):
        await QueueBackend(FakeManager()).add_item("count")


async def test_reorder_moves_the_item() -> None:
    manager = FakeManager()
    await QueueBackend(manager).reorder("uid-1", before_uid="uid-2")

    assert manager.kwargs_for("item_move") == [{"uid": "uid-1", "before_uid": "uid-2"}]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"pos_dest": "front", "before_uid": "uid-2"},
        {"pos_dest": "front", "after_uid": "uid-2"},
    ],
)
async def test_reorder_needs_exactly_one_destination(kwargs: dict[str, Any]) -> None:
    manager = FakeManager()
    with pytest.raises(QueueRequestRejectedError):
        await QueueBackend(manager).reorder("uid-1", **kwargs)
    assert manager.calls == []


async def test_start_and_stop_hit_the_matching_manager_calls() -> None:
    manager = FakeManager()
    backend = QueueBackend(manager)

    await backend.start()
    await backend.stop()
    await backend.stop(cancel=True)

    assert manager.method_names() == ["queue_start", "queue_stop", "queue_stop_cancel"]


# -------------------------------------------------------------- emergency abort
#
# The one composed operation in the adapter. Upstream will not abort a plan that
# is still running -- `re_abort` acts only on a PAUSED Run Engine -- so these
# pin the composition itself: which manager calls are made, in what order, on
# fresh status reads, and that every outcome that did NOT stop the machine says
# so instead of returning a success shape.


@pytest.fixture
def fast_abort_backend():
    """A backend whose abort pause window is short enough to test."""

    def _build(manager: Any, **kwargs: Any) -> QueueBackend:
        kwargs.setdefault("abort_pause_polls", 3)
        kwargs.setdefault("abort_poll_interval", 0)
        return QueueBackend(manager, **kwargs)

    return _build


async def test_abort_pauses_immediately_then_aborts(fast_abort_backend) -> None:
    """The whole composition on the ordinary path: a running plan is paused
    with the IMMEDIATE option (deferred would run it on to the next checkpoint,
    or past the last one to completion) and only then aborted."""
    manager = FakeManager(
        status=[
            status_doc(manager_state="executing_queue"),
            status_doc(manager_state="paused"),
            status_doc(manager_state="idle"),
        ],
        re_abort={"success": True, "msg": "plan aborted"},
    )

    result = await fast_abort_backend(manager).abort()

    assert manager.method_names() == ["status", "re_pause", "status", "re_abort", "status"]
    assert manager.kwargs_for("re_pause") == [{"option": "immediate"}]
    assert result["aborted"] is True
    assert result["paused_first"] is True
    assert result["abort_pending"] is False
    assert result["manager_state"] == "idle"
    assert result["msg"] == "plan aborted"


async def test_abort_reads_status_fresh_at_every_step(fast_abort_backend) -> None:
    """`reload=True` everywhere, because every decision here turns on the live
    manager state: the client caches status for ~0.5s, and a half-second-stale
    read is what makes a halt lie about what it did."""
    manager = FakeManager(
        status=[status_doc(manager_state="paused"), status_doc(manager_state="idle")]
    )

    await fast_abort_backend(manager).abort()

    assert manager.kwargs_for("status") == [{"reload": True}, {"reload": True}]


async def test_abort_skips_the_pause_when_the_engine_is_already_paused(
    fast_abort_backend,
) -> None:
    """A human may have paused from qserver's own console. Re-pausing there is
    at best noise and at worst a refusal on the halt path."""
    manager = FakeManager(
        status=[status_doc(manager_state="paused"), status_doc(manager_state="idle")]
    )

    result = await fast_abort_backend(manager).abort()

    assert manager.method_names() == ["status", "re_abort", "status"]
    assert result["paused_first"] is False


async def test_abort_retries_the_pause_across_the_starting_queue_window(
    fast_abort_backend,
) -> None:
    """A pause is refused while the plan has not begun yet (`starting_queue`),
    so a single attempt would time out on exactly the window an operator is
    most likely to hit -- the seconds right after a start."""
    manager = FakeManager(
        status=[
            status_doc(manager_state="starting_queue"),
            status_doc(manager_state="executing_queue"),
            status_doc(manager_state="paused"),
            status_doc(manager_state="idle"),
        ],
        re_pause=[RequestFailedError("no plan is running", {}), {"success": True, "msg": ""}],
    )

    result = await fast_abort_backend(manager).abort()

    assert manager.method_names() == [
        "status",
        "re_pause",  # refused: the plan has not started
        "status",
        "re_pause",  # retried once it is executing
        "status",
        "re_abort",
        "status",
    ]
    assert result["aborted"] is True


async def test_abort_refuses_when_nothing_is_running(fast_abort_backend) -> None:
    """An honest 'there is nothing to stop', not a failure -- and nothing is
    sent to the manager, so an idle queue cannot be disturbed by a stray abort."""
    manager = FakeManager(status=status_doc(manager_state="idle"))

    with pytest.raises(qb.NothingRunningError) as excinfo:
        await fast_abort_backend(manager).abort()

    assert excinfo.value.reason == "nothing_running"
    assert manager.method_names() == ["status"]


async def test_abort_reports_a_plan_that_ended_while_it_was_being_prepared(
    fast_abort_backend,
) -> None:
    """The race: the plan finishes between the first status read and the pause.
    Nothing was aborted, and the answer must say that rather than claim a halt."""
    manager = FakeManager(
        status=[status_doc(manager_state="executing_queue"), status_doc(manager_state="idle")]
    )

    with pytest.raises(qb.NothingRunningError):
        await fast_abort_backend(manager).abort()

    assert "re_abort" not in manager.method_names()


async def test_abort_that_never_reaches_paused_says_nothing_was_aborted(
    fast_abort_backend,
) -> None:
    """The dangerous outcome. The plan may still be running, so this must NOT
    come back as a success shape and must not reach `re_abort` (which would be
    refused anyway, with a message about the wrong thing)."""
    manager = FakeManager(status=status_doc(manager_state="executing_queue"))

    with pytest.raises(qb.AbortPauseTimeoutError) as excinfo:
        await fast_abort_backend(manager).abort()

    assert excinfo.value.reason == "abort_pause_timeout"
    assert "NOTHING WAS ABORTED" in str(excinfo.value)
    assert "re_abort" not in manager.method_names()
    # One pause per poll plus the opening one: the retry loop is bounded.
    assert manager.method_names().count("re_pause") == 4


async def test_abort_propagates_an_unreachable_manager(fast_abort_backend) -> None:
    manager = FakeManager(status=RequestTimeoutError("no answer", {}))

    with pytest.raises(QueueUnavailableError):
        await fast_abort_backend(manager).abort()


async def test_abort_relays_a_manager_refusal_of_the_abort_itself(fast_abort_backend) -> None:
    manager = FakeManager(
        status=status_doc(manager_state="paused"),
        re_abort=RequestFailedError("cannot abort", {}),
    )

    with pytest.raises(QueueRequestRejectedError):
        await fast_abort_backend(manager).abort()


async def test_abort_survives_a_failed_follow_up_status_read(fast_abort_backend) -> None:
    """The abort is already accepted at that point. Turning a REPORTING failure
    into a raise would tell the operator the machine did not stop when it did."""
    manager = FakeManager(
        status=[status_doc(manager_state="paused"), RequestTimeoutError("gone", {})]
    )

    result = await fast_abort_backend(manager).abort()

    assert result["aborted"] is True
    assert result["manager_state"] is None
    # Unknown state reads as still unwinding, never as settled.
    assert result["abort_pending"] is True


async def test_abort_reports_pending_while_the_manager_is_still_unwinding(
    fast_abort_backend,
) -> None:
    manager = FakeManager(
        status=[status_doc(manager_state="paused"), status_doc(manager_state="executing_queue")]
    )

    result = await fast_abort_backend(manager).abort()

    assert result["aborted"] is True
    assert result["abort_pending"] is True


async def test_abort_without_a_manager_fails_closed() -> None:
    with pytest.raises(QueueUnavailableError):
        await QueueBackend(None).abort()


# ------------------------------------------------------------ worker namespace


async def test_upload_script_refreshes_the_plan_lists_by_default() -> None:
    manager = FakeManager(script_upload={"success": True, "task_uid": "task-1"})
    result = await QueueBackend(manager).upload_script("def my_plan(): ...")

    assert result["task_uid"] == "task-1"
    assert manager.kwargs_for("script_upload") == [
        {"script": "def my_plan(): ...", "update_lists": True}
    ]


async def test_task_result_and_plans_allowed_read_the_manager() -> None:
    manager = FakeManager(
        task_result={"success": True, "status": "completed"},
        plans_allowed={"success": True, "plans_allowed": {"grid_scan": {}}},
    )
    backend = QueueBackend(manager)

    assert (await backend.task_result("task-1"))["status"] == "completed"
    assert "grid_scan" in (await backend.plans_allowed())["plans_allowed"]
    assert manager.kwargs_for("task_result") == [{"task_uid": "task-1"}]


async def test_devices_allowed_reads_the_manager() -> None:
    manager = FakeManager(
        devices_allowed={"success": True, "devices_allowed": {"COR1": {"is_movable": True}}}
    )

    reply = await QueueBackend(manager).devices_allowed()

    assert reply["devices_allowed"]["COR1"] == {"is_movable": True}
    assert manager.method_names() == ["devices_allowed"]


async def test_devices_allowed_without_a_manager_fails_closed() -> None:
    with pytest.raises(QueueUnavailableError):
        await QueueBackend(None).devices_allowed()


# ------------------------------------------------------------------ capability


async def test_capability_is_executable_on_a_virtual_accelerator(connector) -> None:
    connector("virtual_accelerator")
    capability = await QueueBackend(FakeManager(status=status_doc())).capability()

    assert capability.can_execute is True
    assert capability.reason == qb.REASON_EXECUTABLE


async def test_capability_on_mock_names_the_flip_command(connector) -> None:
    connector("mock")
    capability = await QueueBackend(FakeManager(status=status_doc())).capability()

    assert capability.can_execute is False
    assert capability.reason == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert qb.FLIP_COMMAND in capability.detail


async def test_capability_never_probes_the_manager_on_mock(connector) -> None:
    connector("mock")
    manager = FakeManager(status=status_doc())
    await QueueBackend(manager).capability()

    # A browse-only deployment is not expected to have a queue server at all, so
    # the connector answer must win before anything reaches for one.
    assert manager.calls == []


async def test_capability_rejects_an_unsupported_connector(connector) -> None:
    connector("doocs")
    capability = await QueueBackend(FakeManager(status=status_doc())).capability()

    assert capability.can_execute is False
    assert capability.reason == qb.REASON_UNSUPPORTED_CONNECTOR


async def test_capability_fails_closed_when_the_config_is_unreadable(connector) -> None:
    connector(FileNotFoundError("no config mounted"))
    capability = await QueueBackend(FakeManager(status=status_doc())).capability()

    assert capability.can_execute is False
    assert capability.reason == qb.REASON_CONFIG_UNREADABLE


async def test_capability_fails_closed_without_a_configured_manager(connector) -> None:
    connector("virtual_accelerator")
    capability = await QueueBackend(None).capability()

    assert capability.can_execute is False
    assert capability.reason == qb.REASON_MANAGER_NOT_CONFIGURED


async def test_capability_fails_closed_when_the_manager_is_silent(connector) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(status=RequestTimeoutError("no answer", {}))
    capability = await QueueBackend(manager).capability()

    assert capability.can_execute is False
    assert capability.reason == qb.REASON_MANAGER_UNREACHABLE


def test_capability_serializes_to_the_wire_shape() -> None:
    payload = Capability(
        can_execute=False,
        reason="browse_only_connector",
        detail="why",
        lane="bluesky",
        lane_target="live",
    ).to_dict()
    assert payload == {
        "can_execute": False,
        "reason": "browse_only_connector",
        "detail": "why",
        "lane": "bluesky",
        "lane_target": "live",
    }


# -------------------------------------------------------- environment lifecycle


async def test_mock_deployments_never_open_the_environment(connector, fast_backend) -> None:
    connector("mock")
    manager = FakeManager(status=status_doc(worker_environment_exists=False))
    backend = fast_backend(manager)

    assert await backend.ensure_environment() is False
    assert "environment_open" not in manager.method_names()


async def test_ensure_environment_is_a_no_op_when_already_open(connector, fast_backend) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc())

    assert await fast_backend(manager).ensure_environment() is True
    assert "environment_open" not in manager.method_names()


async def test_ensure_environment_opens_and_waits_for_the_worker(connector, fast_backend) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(worker_environment_exists=False),  # capability probe
            status_doc(worker_environment_exists=False),  # pre-open check
            status_doc(worker_environment_exists=False),  # still coming up
            status_doc(worker_environment_exists=True),
        ]
    )

    assert await fast_backend(manager).ensure_environment() is True
    assert manager.method_names().count("environment_open") == 1


async def test_ensure_environment_does_not_reopen_while_one_is_being_created(
    connector, fast_backend
) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(worker_environment_exists=False),
            status_doc(manager_state="creating_environment", worker_environment_exists=False),
            status_doc(worker_environment_exists=True),
        ]
    )

    assert await fast_backend(manager).ensure_environment() is True
    assert "environment_open" not in manager.method_names()


async def test_ensure_environment_gives_up_after_its_bounded_retries(
    connector, fast_backend
) -> None:
    connector("virtual_accelerator")
    manager = FakeManager(status=status_doc(worker_environment_exists=False))
    backend = fast_backend(manager, env_open_attempts=2)

    with pytest.raises(EnvironmentUnavailableError):
        await backend.ensure_environment()
    assert manager.method_names().count("environment_open") == 2


async def test_ensure_environment_survives_a_refused_open(connector, fast_backend) -> None:
    """A concurrent opener wins the race: the refusal is not the answer, the poll is."""
    connector("virtual_accelerator")
    manager = FakeManager(
        status=[
            status_doc(worker_environment_exists=False),
            status_doc(worker_environment_exists=False),
            status_doc(worker_environment_exists=True),
        ],
        environment_open=RequestFailedError({}, {"msg": "Manager state is not idle"}),
    )

    assert await fast_backend(manager).ensure_environment() is True


async def test_ensure_environment_for_execute_refuses_a_browse_only_deployment(
    connector, fast_backend
) -> None:
    connector("mock")
    backend = fast_backend(FakeManager(status=status_doc()))

    with pytest.raises(ExecutionUnavailableError) as excinfo:
        await backend.ensure_environment_for_execute()

    # The refusal carries the capability record so the route layer can put the
    # same reason and remediation on the wire that the status surface publishes.
    assert excinfo.value.reason == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert excinfo.value.capability.can_execute is False
    assert qb.FLIP_COMMAND in str(excinfo.value)


async def test_ensure_environment_for_execute_passes_when_the_worker_is_up(
    connector, fast_backend
) -> None:
    connector("virtual_accelerator")
    await fast_backend(FakeManager(status=status_doc())).ensure_environment_for_execute()


async def test_close_environment_is_refused_while_the_queue_is_active() -> None:
    manager = FakeManager(status=status_doc(manager_state="executing_queue"))
    with pytest.raises(QueueRequestRejectedError) as excinfo:
        await QueueBackend(manager).close_environment()

    assert "Stop the queue first" in str(excinfo.value)
    assert "environment_close" not in manager.method_names()


async def test_close_environment_closes_when_the_queue_is_idle() -> None:
    manager = FakeManager(status=status_doc())
    await QueueBackend(manager).close_environment()

    assert "environment_close" in manager.method_names()


# --------------------------------------------------------------------- helpers


def test_is_environment_open_reads_the_status_document() -> None:
    assert qb.is_environment_open(status_doc()) is True
    assert qb.is_environment_open(status_doc(worker_environment_exists=False)) is False
    assert qb.is_environment_open({}) is False


@pytest.mark.parametrize(
    ("state", "active"),
    [
        ("idle", False),
        ("creating_environment", False),
        ("closing_environment", False),
        ("starting_queue", True),
        ("executing_queue", True),
        ("executing_task", True),
        ("paused", True),
    ],
)
def test_is_queue_active_covers_the_armed_states(state: str, active: bool) -> None:
    assert qb.is_queue_active(status_doc(manager_state=state)) is active


def test_run_id_of_tolerates_items_without_osprey_metadata() -> None:
    assert qb.run_id_of({"meta": {qb.RUN_ID_META_KEY: "run-3"}}) == "run-3"
    assert qb.run_id_of({"meta": {"other": 1}}) is None
    assert qb.run_id_of({"meta": None}) is None
    assert qb.run_id_of({}) is None


# --------------------------------------------------------------- construction


def test_from_env_yields_a_managerless_backend_when_no_address_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(qb.QSERVER_CONTROL_ADDRESS_ENV, raising=False)
    assert QueueBackend.from_env()._manager is None


def test_from_env_builds_a_client_when_an_address_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(qb.QSERVER_CONTROL_ADDRESS_ENV, "tcp://queueserver:60615")
    sentinel = FakeManager()
    monkeypatch.setattr("bluesky_queueserver_api.zmq.aio.REManagerAPI", lambda *a, **kw: sentinel)

    assert QueueBackend.from_env()._manager is sentinel


def test_module_stays_import_clean_of_the_bluesky_stack() -> None:
    """The bridge's import-hygiene boundary (`app.py`'s `_BRIDGE_ONLY_MODULES`).

    `app.py` imports this module, so a top-level `bluesky`/`ophyd`/`tiled`
    import here would breach that boundary from one step away. Must run in a
    fresh interpreter — see `test_app_import_clean.py` for why.
    """
    import subprocess
    import sys

    from osprey.services.bluesky_bridge.app import _BRIDGE_ONLY_MODULES

    code = (
        "import osprey.services.bluesky_bridge.queue_backend, sys; "
        f"leaked = sorted(set({sorted(_BRIDGE_ONLY_MODULES)!r}) & sys.modules.keys()); "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
