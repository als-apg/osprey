"""The action layer: one worker runs each action through admit, precondition, reach, limits, owner and lock.

The gate is the real :class:`AdmitGate` on a real control-context tree, the
run lock is the real :func:`~osprey.runtime.guarded_run.lock` under a
throwaway deployment root, and the connector is a stand-in that records reads.
The limits validator is the one seam stubbed here: which validator
``LimitsValidator.from_config`` builds is the connector package's own business,
so each test says what it returns.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from tests._guarded_run import held_lock, lock_path, reset_run_state

import osprey.runtime._run_state as _run_state
from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY, ActionContext
from osprey.services.trajectory import actions
from osprey.services.trajectory.actions import (
    ACTION_STOPPED,
    GUARDED_RUN_UNAVAILABLE,
    LIMITS_UNAVAILABLE,
    RUN_BUSY,
    SERVICE_STOPPING,
    ActionLayer,
    ActionRefused,
)
from osprey.services.trajectory.admit import (
    NARROWING,
    NO_OPERATOR_IDENTITY,
    WRITES_DISABLED,
    Admission,
    AdmitGate,
    AdmitRefused,
)
from osprey.services.trajectory.reach import TARGET_UNREACHABLE, TRAJECTORY_UNAVAILABLE
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectoryDevice,
    TrajectoryStartup,
)
from osprey_connectors import control_context, posture_store
from osprey_connectors.errors import ChannelReadFailedError
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

OWNER = "alice"
KIND = "panel"
TARGET = "va"
ARMED = {"writes_enabled": True}
COMMANDS = TrajectoryCommands(
    model="SR",
    devices=tuple(
        TrajectoryDevice(
            name=f"SR/BPM{n}",
            x_row=f"SR:BPM{n}:X",
            y_row=f"SR:BPM{n}:Y",
            sum_row=f"SR:BPM{n}:SUM",
            arm="SR:BPM:ARM",
            armed="SR:BPM:ARMED",
        )
        for n in (1, 2, 3)
    ),
    injection_request="SR:INJ:REQ",
    injection_busy="SR:INJ:BUSY",
    trigger_sample=10,
)


class _Connector:
    """Answers every read with *outcome*; records reads and the validator it was handed."""

    def __init__(self, outcome: Any = 0) -> None:
        self.outcome = outcome
        self.reads: list[str] = []
        self._limits_validator: Any = "never-assigned"

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:  # noqa: ARG002
        self.reads.append(address)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Validator:
    """An enforcing validator (``failsafe_reason`` unset) or a fail-safe one."""

    def __init__(self, failsafe_reason: str | None = None) -> None:
        self.failsafe_reason = failsafe_reason


@pytest.fixture(autouse=True)
def _deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "deploy"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text("control_system: {}\n", encoding="utf-8")
    monkeypatch.chdir(root)
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    reset_run_state(monkeypatch)
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


class _Handed(list):
    """What ``from_config`` hands out, in order (the last repeats), and the targets it was asked."""

    def __init__(self) -> None:
        super().__init__([_Validator()])
        self.asked: list[tuple[str | None, str | None]] = []


@pytest.fixture
def validators(monkeypatch: pytest.MonkeyPatch) -> _Handed:
    handed = _Handed()

    def from_config(*, connector_type: str | None = None, target: str | None = None) -> Any:
        handed.asked.append((connector_type, target))
        return handed.pop(0) if len(handed) > 1 else handed[0]

    monkeypatch.setattr(actions.LimitsValidator, "from_config", staticmethod(from_config))
    return handed


def _narrow(owner: str) -> None:
    tree = Path(os.environ[posture_store.CONTROL_CONTEXT_TREE_ENV_VAR])
    control_context.write_record(
        control_context.ControlContext(target=TARGET, generation=1, posture={TARGET: "sandbox"}),
        path=tree / owner / control_context.RECORD_FILENAME,
    )
    posture_store.invalidate_cache()


def _layer(
    connector: Any | None = None,
    *,
    commands: TrajectoryCommands | None = COMMANDS,
    section: dict[str, Any] | None = None,
) -> tuple[ActionLayer, Any]:
    connector = _Connector() if connector is None else connector
    startup = TrajectoryStartup(
        target=TARGET, connector=connector, commands=commands, guarded_run_dir=Path("/unused")
    )
    gate = AdmitGate(target=TARGET, section=ARMED if section is None else section)
    return ActionLayer(startup, gate), connector


async def _refused(coro: Any) -> ActionRefused:
    with pytest.raises(ActionRefused) as caught:
        await coro
    return caught.value


# -- the happy path ------------------------------------------------------------


@pytest.mark.usefixtures("validators")
async def test_an_admitted_action_runs_under_the_owner_and_the_lock() -> None:
    layer, connector = _layer()
    seen: dict[str, Any] = {}

    async def action(admission: Admission, stop: asyncio.Event) -> str:
        seen["admission"] = admission
        seen["owner"] = posture_store.current_owner()
        seen["hold"] = _run_state.STATE.hold
        seen["task"] = asyncio.current_task()
        seen["stop"] = stop
        return "done"

    try:
        assert await layer.submit(OWNER, KIND, action) == "done"
    finally:
        await layer.close()

    assert seen["admission"].owner == OWNER and seen["admission"].kind == KIND
    assert seen["owner"] == OWNER
    hold = seen["hold"]
    assert hold is not None and hold.target == TARGET
    assert hold.context == ActionContext(target=TARGET, owner=OWNER, approved_target=TARGET)
    assert seen["task"] is layer.worker_task
    assert seen["stop"] is layer.stop_event
    # The probe read the busy flag once before the action.
    assert connector.reads == [COMMANDS.injection_busy]
    # Released afterwards, owner back on the floor.
    assert _run_state.STATE.hold is None
    assert posture_store.current_owner() != OWNER


async def test_the_validator_is_rebuilt_for_the_service_target_and_assigned(
    validators: _Handed,
) -> None:
    first, second = _Validator(), _Validator()
    validators[:] = [first, second]
    layer, connector = _layer()
    handed: list[Any] = []

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        handed.append(connector._limits_validator)

    try:
        await layer.submit(OWNER, KIND, action)
        await layer.submit(OWNER, KIND, action)
    finally:
        await layer.close()
    assert handed == [first, second]
    assert validators.asked == [(None, TARGET), (None, TARGET)]


@pytest.mark.usefixtures("validators")
async def test_actions_run_one_at_a_time_in_submission_order() -> None:
    layer, _ = _layer()
    log: list[str] = []
    gate = asyncio.Event()

    def make(name: str, wait: bool = False) -> Any:
        async def action(_admission: Admission, _stop: asyncio.Event) -> str:
            log.append(f"start {name}")
            if wait:
                await gate.wait()
            log.append(f"end {name}")
            return name

        return action

    try:
        first = asyncio.create_task(layer.submit(OWNER, KIND, make("a", wait=True)))
        second = asyncio.create_task(layer.submit(OWNER, KIND, make("b")))
        await asyncio.sleep(0.05)
        assert log == ["start a"]
        gate.set()
        assert await asyncio.gather(first, second) == ["a", "b"]
    finally:
        await layer.close()
    assert log == ["start a", "end a", "start b", "end b"]


@pytest.mark.usefixtures("validators")
async def test_a_failing_action_raises_to_its_caller_and_the_worker_goes_on() -> None:
    layer, _ = _layer()

    async def boom(_admission: Admission, _stop: asyncio.Event) -> None:
        raise OSError("the action's own failure is not the lock's")

    async def fine(_admission: Admission, _stop: asyncio.Event) -> str:
        return "ok"

    try:
        with pytest.raises(OSError, match="action's own failure"):
            await layer.submit(OWNER, KIND, boom)
        assert await layer.submit(OWNER, KIND, fine) == "ok"
    finally:
        await layer.close()
    assert _run_state.STATE.hold is None


# -- the order of the checks ---------------------------------------------------


async def test_admit_runs_first_and_its_refusal_stops_everything(validators: _Handed) -> None:
    layer, connector = _layer()
    order: list[str] = []

    def precondition(_admission: Admission) -> None:
        order.append("precondition")

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("a refused action never runs")

    try:
        with pytest.raises(AdmitRefused) as caught:
            await layer.submit(None, KIND, action, precondition=precondition)
    finally:
        await layer.close()
    assert caught.value.code == NO_OPERATOR_IDENTITY
    assert order == [] and connector.reads == []
    assert validators.asked == []


@pytest.mark.usefixtures("validators")
async def test_admit_reads_the_record_anew_for_every_action() -> None:
    layer, _ = _layer()

    async def action(_admission: Admission, _stop: asyncio.Event) -> str:
        return "ok"

    try:
        assert await layer.submit(OWNER, KIND, action) == "ok"
        _narrow(OWNER)
        with pytest.raises(AdmitRefused) as caught:
            await layer.submit(OWNER, KIND, action)
    finally:
        await layer.close()
    assert caught.value.code == NARROWING


async def test_the_precondition_runs_after_admit_and_before_reach(validators: _Handed) -> None:
    layer, connector = _layer()
    seen: list[Admission] = []

    class Refusal(Exception):
        pass

    def precondition(admission: Admission) -> None:
        seen.append(admission)
        raise Refusal("not the holder")

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        with pytest.raises(Refusal):
            await layer.submit(OWNER, KIND, action, precondition=precondition)
    finally:
        await layer.close()
    assert [a.owner for a in seen] == [OWNER]
    assert connector.reads == []
    assert validators.asked == []


async def test_an_unreachable_target_is_refused_before_limits(validators: _Handed) -> None:
    layer, _ = _layer(_Connector(ChannelReadFailedError("SR:INJ:BUSY", "timeout")))

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (TARGET_UNREACHABLE, 503)
    assert refusal.as_detail()["code"] == TARGET_UNREACHABLE
    assert validators.asked == []


@pytest.mark.usefixtures("validators")
async def test_a_service_with_no_model_is_trajectory_unavailable() -> None:
    layer, _ = _layer(commands=None)

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (TRAJECTORY_UNAVAILABLE, 503)


@pytest.mark.parametrize("handed", [None, _Validator(failsafe_reason="limits database missing")])
async def test_no_enforcing_validator_is_limits_unavailable(
    validators: _Handed, handed: Any
) -> None:
    validators[:] = [handed]
    layer, connector = _layer()

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (LIMITS_UNAVAILABLE, 503)
    assert connector._limits_validator == "never-assigned"
    assert _run_state.STATE.hold is None


def _configure(root: Path, control_system: str) -> None:
    """Write the deployment's ``control_system`` section and drop the cached config."""
    from osprey_connectors.workspace import reset_config_cache

    (root / "config.yml").write_text(f"control_system: {control_system}\n", "utf-8")
    reset_config_cache()


async def _fires(layer: ActionLayer, connector: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def action(_admission: Admission, _stop: asyncio.Event) -> str:
        seen["hold"] = _run_state.STATE.hold
        seen["validator"] = connector._limits_validator
        return "done"

    try:
        assert await layer.submit(OWNER, KIND, action) == "done"
    finally:
        await layer.close()
    return seen


async def _limits_refused(layer: ActionLayer, connector: Any) -> ActionRefused:
    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (LIMITS_UNAVAILABLE, 503)
    assert connector._limits_validator == "never-assigned"
    assert _run_state.STATE.hold is None
    return refusal


async def test_limits_checking_stated_off_fires_with_no_validator(_deployment: Path) -> None:
    _configure(_deployment, "{limits_checking: {enabled: false}}")
    layer, connector = _layer()
    seen = await _fires(layer, connector)
    assert seen["validator"] is None
    assert seen["hold"].context == ActionContext(
        target=TARGET, owner=OWNER, approved_target=TARGET, limits_off=True
    )


async def test_stated_off_replaces_a_validator_an_earlier_action_left(
    _deployment: Path, validators: _Handed
) -> None:
    _configure(_deployment, "{limits_checking: {enabled: false}}")
    validators[:] = [None]
    layer, connector = _layer()
    connector._limits_validator = _Validator()
    seen = await _fires(layer, connector)
    assert seen["validator"] is None
    assert seen["hold"].context.limits_off is True


async def test_limits_checking_unstated_is_limits_unavailable(_deployment: Path) -> None:
    _configure(_deployment, "{limits_checking: {mode: optional}}")
    layer, connector = _layer()
    refusal = await _limits_refused(layer, connector)
    assert "control_system.limits_checking.enabled" in refusal.detail


async def test_an_unreadable_config_is_limits_unavailable() -> None:
    # The deployment root carries no config.yml, so no config can be read.
    layer, connector = _layer()
    refusal = await _limits_refused(layer, connector)
    assert "cannot be read" in refusal.detail


@pytest.mark.parametrize(
    "control_system",
    [
        "{limits_checking: {enabled: 'false'}}",
        "{limits_checking: {enabled: true, database_path: missing.json}}",
    ],
)
async def test_a_fail_safe_posture_is_limits_unavailable(
    _deployment: Path, control_system: str
) -> None:
    _configure(_deployment, control_system)
    layer, connector = _layer()
    await _limits_refused(layer, connector)


async def test_optional_with_no_database_fires_unchecked(_deployment: Path) -> None:
    _configure(_deployment, "{limits_checking: {enabled: true, mode: optional}}")
    layer, connector = _layer()
    seen = await _fires(layer, connector)
    validator = seen["validator"]
    assert validator is not None and validator.failsafe_reason is None
    assert seen["hold"].context.limits_off is False


# -- the lock -------------------------------------------------------------------


@pytest.mark.usefixtures("validators")
async def test_a_held_run_lock_is_run_busy(_deployment: Path) -> None:
    layer, _ = _layer()

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        with held_lock(TARGET, root=_deployment):
            refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (RUN_BUSY, 409)


@pytest.mark.usefixtures("validators")
async def test_an_unusable_lock_dir_is_guarded_run_unavailable(_deployment: Path) -> None:
    lock_file = lock_path(TARGET, _deployment)
    lock_file.parent.mkdir(parents=True)
    lock_file.mkdir()  # a directory at the lock's name
    layer, _ = _layer()

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (GUARDED_RUN_UNAVAILABLE, 503)


@pytest.mark.usefixtures("validators")
async def test_a_readonly_process_is_writes_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    layer, _ = _layer()

    async def action(_admission: Admission, _stop: asyncio.Event) -> None:
        raise AssertionError("never runs")

    try:
        refusal = await _refused(layer.submit(OWNER, KIND, action))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (WRITES_DISABLED, 403)


# -- the control path ------------------------------------------------------------


async def test_the_control_path_acts_at_admission_without_lock_reach_or_queue(
    validators: _Handed,
) -> None:
    layer, connector = _layer(_Connector(ChannelReadFailedError("SR:INJ:BUSY", "down")))
    release = asyncio.Event()
    seen: dict[str, Any] = {}

    async def slow(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    def take_over(admission: Admission) -> str:
        seen["hold"] = _run_state.STATE.hold
        seen["admission"] = admission
        return "taken"

    # Reachable for the in-flight action, unreachable for everything after it.
    connector.outcome = 0
    try:
        in_flight = asyncio.create_task(layer.submit(OWNER, KIND, slow))
        await asyncio.sleep(0.05)
        connector.outcome = ChannelReadFailedError("SR:INJ:BUSY", "down")
        reads_before = list(connector.reads)
        # The worker holds the lock and is busy; the control path still acts now.
        assert layer.control(OWNER, KIND, take_over) == "taken"
        assert connector.reads == reads_before
        release.set()
        await in_flight
    finally:
        await layer.close()
    assert seen["admission"].owner == OWNER
    # The hold seen is the in-flight action's, not one the control path took.
    assert seen["hold"].context.owner == OWNER
    assert validators.asked == [(None, TARGET)]


async def test_the_control_path_refuses_what_admit_refuses() -> None:
    layer, _ = _layer(section={"writes_enabled": False})

    def act(_admission: Admission) -> None:
        raise AssertionError("never acts")

    with pytest.raises(AdmitRefused) as caught:
        layer.control(OWNER, KIND, act)
    assert caught.value.code == WRITES_DISABLED


def test_the_control_path_may_skip_admit_for_a_release() -> None:
    _narrow(OWNER)
    layer, _ = _layer(section={"writes_enabled": False})
    handed: list[Any] = []
    assert layer.control(OWNER, KIND, lambda a: handed.append(a) or "released", admit=False) == (
        "released"
    )
    assert handed == [None]


def test_the_control_path_never_takes_the_lock() -> None:
    layer, _ = _layer()
    holds: list[Any] = []
    layer.control(OWNER, KIND, lambda admission: holds.append(_run_state.STATE.hold))
    assert holds == [None]


# -- the per-holder stop event -----------------------------------------------------


def test_renewing_the_stop_event_sets_the_old_one_and_installs_a_clear_one() -> None:
    layer, _ = _layer()
    old = layer.stop_event
    assert not old.is_set()
    new = layer.renew_stop_event()
    assert old.is_set()
    assert new is layer.stop_event and new is not old
    assert not new.is_set()


@pytest.mark.usefixtures("validators")
async def test_an_in_flight_action_sees_its_own_stop_event_set_on_renewal() -> None:
    layer, _ = _layer()
    started = asyncio.Event()

    async def runs_until_stopped(_admission: Admission, stop: asyncio.Event) -> str:
        started.set()
        await stop.wait()
        return "stopped"

    try:
        task = asyncio.create_task(layer.submit(OWNER, KIND, runs_until_stopped))
        await started.wait()
        fresh = layer.control(OWNER, KIND, lambda admission: layer.renew_stop_event())
        assert await asyncio.wait_for(task, 2) == "stopped"
        assert not fresh.is_set()
    finally:
        await layer.close()


class _SuspendedProbe(_Connector):
    """Holds every read until *release* is set, signalling *reading* when one starts."""

    def __init__(self) -> None:
        super().__init__()
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:
        self.reading.set()
        await self.release.wait()
        return await super().read_channel(address, timeout)


@pytest.mark.usefixtures("validators")
async def test_a_renewal_while_the_probe_is_suspended_refuses_the_action() -> None:
    layer, connector = _layer(_SuspendedProbe())
    ran: list[bool] = []

    async def action(_admission: Admission, stop: asyncio.Event) -> None:
        ran.append(stop.is_set())

    try:
        task = asyncio.create_task(layer.submit(OWNER, KIND, action))
        await asyncio.wait_for(connector.reading.wait(), 2)
        fresh = layer.control(OWNER, KIND, lambda _admission: layer.renew_stop_event())
        connector.release.set()
        refusal = await _refused(asyncio.wait_for(task, 2))
    finally:
        await layer.close()
    assert (refusal.code, refusal.status) == (ACTION_STOPPED, 409)
    assert ran == []
    assert not fresh.is_set()


@pytest.mark.usefixtures("validators")
async def test_an_action_queued_before_a_renewal_is_refused_and_a_later_one_runs() -> None:
    layer, _ = _layer()
    release = asyncio.Event()
    ran: list[str] = []

    async def slow(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    async def queued(_admission: Admission, _stop: asyncio.Event) -> None:
        ran.append("queued")

    async def after(_admission: Admission, stop: asyncio.Event) -> bool:
        ran.append("after")
        return stop.is_set()

    try:
        first = asyncio.create_task(layer.submit(OWNER, KIND, slow))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(layer.submit(OWNER, KIND, queued))
        await asyncio.sleep(0.02)
        layer.renew_stop_event()
        release.set()
        await first
        refusal = await _refused(second)
        assert await layer.submit(OWNER, KIND, after) is False
    finally:
        await layer.close()
    assert refusal.code == ACTION_STOPPED
    assert ran == ["after"]


@pytest.mark.usefixtures("validators")
async def test_an_action_raising_cancelled_error_does_not_end_the_worker() -> None:
    layer, _ = _layer()

    async def cancelled(_admission: Admission, _stop: asyncio.Event) -> None:
        raise asyncio.CancelledError

    async def ok(_admission: Admission, _stop: asyncio.Event) -> str:
        return "ok"

    try:
        with pytest.raises(asyncio.CancelledError):
            await layer.submit(OWNER, KIND, cancelled)
        worker = layer.worker_task
        assert worker is not None and not worker.done()
        assert await asyncio.wait_for(layer.submit(OWNER, KIND, ok), 2) == "ok"
    finally:
        await layer.close()


@pytest.mark.usefixtures("validators")
async def test_a_cancelled_worker_fails_queued_callers_and_the_next_submit_restarts_it() -> None:
    layer, _ = _layer()
    release = asyncio.Event()

    async def slow(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    async def ok(_admission: Admission, _stop: asyncio.Event) -> str:
        return "ok"

    try:
        first = asyncio.create_task(layer.submit(OWNER, KIND, slow))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(layer.submit(OWNER, KIND, ok))
        await asyncio.sleep(0.02)
        worker = layer.worker_task
        assert worker is not None
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 2)
        with pytest.raises(RuntimeError, match="worker ended"):
            await asyncio.wait_for(second, 2)
        assert await asyncio.wait_for(layer.submit(OWNER, KIND, ok), 2) == "ok"
        assert layer.worker_task is not worker
    finally:
        await layer.close()


# -- lifecycle ------------------------------------------------------------------------


@pytest.mark.usefixtures("validators")
async def test_close_ends_the_worker_and_a_later_submit_is_refused() -> None:
    layer, _ = _layer()

    async def action(_admission: Admission, _stop: asyncio.Event) -> str:
        return "ok"

    assert await layer.submit(OWNER, KIND, action) == "ok"
    worker = layer.worker_task
    await layer.close()
    assert worker is not None and worker.done()
    assert layer.closed
    refusal = await _refused(layer.submit(OWNER, KIND, action))
    assert (refusal.code, refusal.status) == (SERVICE_STOPPING, 503)


@pytest.mark.usefixtures("validators")
async def test_close_fails_the_queued_actions_and_lets_the_one_in_flight_finish() -> None:
    layer, _ = _layer()
    started = asyncio.Event()
    release = asyncio.Event()
    ran: list[str] = []

    async def slow(_admission: Admission, _stop: asyncio.Event) -> str:
        started.set()
        await release.wait()
        return "done"

    async def queued(_admission: Admission, _stop: asyncio.Event) -> None:
        ran.append("queued")

    first = asyncio.create_task(layer.submit(OWNER, KIND, slow))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(layer.submit(OWNER, KIND, queued))
    third = asyncio.create_task(layer.submit(OWNER, KIND, queued))
    await asyncio.sleep(0.02)
    closing = asyncio.create_task(layer.close())
    for waiting in (second, third):
        refusal = await asyncio.wait_for(_refused(waiting), 2)
        assert (refusal.code, refusal.status) == (SERVICE_STOPPING, 503)
    assert not closing.done(), "close awaits the action in flight"
    release.set()
    assert await asyncio.wait_for(first, 2) == "done"
    await asyncio.wait_for(closing, 2)
    assert ran == []
    worker = layer.worker_task
    assert worker is not None and worker.done()


@pytest.mark.usefixtures("validators")
async def test_refuse_new_refuses_submits_and_keeps_the_queued_ones() -> None:
    layer, _ = _layer()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_admission: Admission, _stop: asyncio.Event) -> str:
        started.set()
        await release.wait()
        return "slow"

    async def ok(_admission: Admission, _stop: asyncio.Event) -> str:
        return "ok"

    try:
        first = asyncio.create_task(layer.submit(OWNER, KIND, slow))
        await asyncio.wait_for(started.wait(), 2)
        second = asyncio.create_task(layer.submit(OWNER, KIND, ok))
        await asyncio.sleep(0.02)
        layer.refuse_new()
        refusal = await _refused(layer.submit(OWNER, KIND, ok))
        assert refusal.code == SERVICE_STOPPING
        release.set()
        assert await asyncio.wait_for(first, 2) == "slow"
        assert await asyncio.wait_for(second, 2) == "ok"
    finally:
        await layer.close()


async def test_close_without_a_worker_and_close_again_are_no_ops() -> None:
    layer, _ = _layer()
    await layer.close()
    assert layer.worker_task is None
    await layer.close()
    assert layer.closed


@pytest.mark.usefixtures("validators")
async def test_a_caller_that_gives_up_while_queued_never_runs() -> None:
    layer, _ = _layer()
    release = asyncio.Event()
    ran: list[str] = []

    async def slow(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    async def queued(_admission: Admission, _stop: asyncio.Event) -> None:
        ran.append("queued")

    try:
        first = asyncio.create_task(layer.submit(OWNER, KIND, slow))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(layer.submit(OWNER, KIND, queued))
        await asyncio.sleep(0.02)
        second.cancel()
        release.set()
        await first
        with pytest.raises(asyncio.CancelledError):
            await second
        await asyncio.sleep(0.02)
    finally:
        await layer.close()
    assert ran == []
