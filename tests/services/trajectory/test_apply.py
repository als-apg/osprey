"""Apply: the pending step or a put-back, written one setting at a time under the run lock.

Every apply runs through real :class:`Sessions` on a real :class:`ActionLayer`
— real admission on a real control-context tree, the real run lock. The
connector is a stand-in holding channel values whose checked write runs the
limits validator the action turn assigned it, the way every connector does;
``LimitsValidator.from_config`` builds a real :class:`LimitsValidator` whose
corrector records state ``max_step``. The step is the real
:func:`~osprey.services.trajectory.step.compute_step` over a fake pyAML tool.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from tests._guarded_run import held_lock, reset_run_state

from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY, ActionContext, OspreyRunBusy, lock
from osprey.services.trajectory import actions
from osprey.services.trajectory.acquisition import setup_action
from osprey.services.trajectory.actions import ACTION_STOPPED, RUN_BUSY, ActionLayer, ActionRefused
from osprey.services.trajectory.admit import AdmitGate
from osprey.services.trajectory.apply import (
    APPLY,
    BAD_REQUEST,
    DIGEST_MISMATCH,
    LIMITS_VIOLATION,
    MODEL_MISMATCH,
    PLAN_RUNNING,
    STALE_SHOT,
    STEP_CONSUMED,
    ZERO_STEP,
    ApplyFailed,
    ApplyRefused,
    PutBack,
    StepApply,
    apply,
    bridge_plan_running,
    parse_body,
)
from osprey.services.trajectory.sequence import load_sequence
from osprey.services.trajectory.sessions import NOT_SESSION_HOLDER, SessionRefused, Sessions
from osprey.services.trajectory.startup import (
    TrajectoryCommands,
    TrajectorySettings,
    TrajectoryStartup,
)
from osprey.services.trajectory.state import LastAction, Moved, ShotResult, Step
from osprey.services.trajectory.step import (
    MAX_STEP_MARGIN,
    TrajectoryTool,
    compute_step,
    max_steps,
)
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig, LimitsValidator
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

FIXTURE_DIR = Path(__file__).parent / "sequences" / "fixture"

OWNER = "alice"
KIND = "panel"
TARGET = "va"
MODEL = "SR"
ARMED = {"writes_enabled": True}
ARM = "SR:BPM:ARM"
REQUEST = "SR:INJ:REQUEST"
BUSY = "SR:INJ:BUSY"
COMMANDS = TrajectoryCommands(
    model=MODEL, arm=(ARM,), injection_request=REQUEST, injection_busy=BUSY, armed=("SR:BPM:ARMED",)
)
KICKER = "SR:BUNCH:KICKER:ENABLE"
MODES = ("SR01:BPM1:MODE", "SR01:BPM2:MODE")
TAGS = {"bpm_acquisition_mode": MODES}

BPMS = ("BPM1", "BPM2", "BPM3")
X_ROWS = ("SR:BPM1:X", "SR:BPM2:X", "SR:BPM3:X")
Y_ROWS = ("SR:BPM1:Y", "SR:BPM2:Y", "SR:BPM3:Y")
HCM1, HCM2, VCM1 = "SR:HCM1", "SR:HCM2", "SR:VCM1"
H_CORR = (HCM1, HCM2)
V_CORR = (VCM1,)

#: The corrector records: HCM1's ``max_step`` binds the default step.
RECORDS = {
    HCM1: {"min_value": -10.0, "max_value": 10.0, "max_step": 0.5},
    HCM2: {"min_value": -10.0, "max_value": 10.0, "max_step": 1.0},
    VCM1: {"min_value": -10.0, "max_value": 10.0},
}


# -- the fake machine and tool -------------------------------------------------------


class _Gate:
    """Holds a write in flight until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()


class _Machine:
    """Holds channel values; its checked write runs the turn's validator, as a connector does."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {
            HCM1: 1.0,
            HCM2: -2.0,
            VCM1: 0.5,
            MODES[0]: 3,
            MODES[1]: 2,
            KICKER: 0,
            BUSY: 0,
        }
        self.writes: list[tuple[str, Any]] = []
        self.gates: dict[str, _Gate] = {}
        self.fail: set[str] = set()
        self._limits_validator: Any = "never-assigned"

    async def read_channel(self, address: str, timeout: float | None = None) -> ChannelValue:  # noqa: ARG002
        return ChannelValue(value=self.values[address], timestamp=datetime.now(UTC))

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> ChannelWriteResult:
        validator = self._limits_validator
        if validator is not None:
            try:
                validator.validate(address, value, read_current=lambda a: self.values[a])
            except ChannelLimitsViolationError as exc:
                raise ChannelWriteBlockedError(address, "LIMITS", message=str(exc)) from exc
        self.writes.append((address, value))
        gate = self.gates.get(address)
        if gate is not None:
            gate.started.set()
            await gate.release.wait()
        if address in self.fail:
            raise ChannelWriteFailedError(
                address, "UNCONFIRMED", outcome=WriteOutcome.UNCONFIRMED, value_written=value
            )
        self.values[address] = value
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

    def written(self) -> list[str]:
        return [address for address, _ in self.writes]


class _Correct:
    """A fake pyAML tool answering ``correct`` with a set step."""

    def __init__(self, step: list[float], status: str = "") -> None:
        self.step = np.asarray(step, dtype=float)
        self.status = status

    def correct(self, x: Any, y: Any, reference: Any = None, **_: Any) -> Any:  # noqa: ARG002
        inverse = SimpleNamespace(
            no_response_names=(),
            rank_discarded=0,
            singular_values=np.array([1.0]),
            inverse_weights=np.array([1.0]),
            spectrum_kind=(SimpleNamespace(value="LIVE"),),
            n_discard=0,
        )
        return SimpleNamespace(
            step=self.step,
            names=(*H_CORR, *V_CORR),
            reason=SimpleNamespace(value="OK"),
            status=self.status,
            all_zero=not np.any(self.step),
            zeroed_h=(),
            zeroed_v=(),
            pseudo_inverse=inverse,
        )


def _tool(fake: _Correct | None = None) -> TrajectoryTool:
    return TrajectoryTool(
        model=MODEL,
        tool=fake or _Correct([2.0, -1.0, 0.5]),
        layout=SimpleNamespace(
            bpm_names_h=X_ROWS, bpm_names_v=Y_ROWS, hcorr_names=H_CORR, vcorr_names=V_CORR
        ),
        bpm_index=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
        sum_names=("S1", "S2", "S3", "S1", "S2", "S3"),
        hcorr_index=(0.5, 1.5),
        vcorr_index=(2.5,),
        unsolved={},
        alpha=0.0,
        n_discard=0,
        gain=1.0,
    )


def _shot(shot_id: str = "shot-1") -> ShotResult:
    return ShotResult(
        shot_id=shot_id,
        model=MODEL,
        x=dict(zip(X_ROWS, (0.1, 0.2, 0.3), strict=True)),
        y=dict(zip(Y_ROWS, (-0.1, -0.2, -0.3), strict=True)),
        sum={"S1": 1.0, "S2": 1.0, "S3": 1.0},
        bpm_order=list(BPMS),
        lost=[],
        last_bpm=BPMS[-1],
        i0=0,
        window=(0, 10),
        per_shot_lost_at=[None],
        shots_included=1,
        shots_failed=[],
        commands_fired=[],
    )


# -- the deployment -----------------------------------------------------------------


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
        "BLUESKY_BRIDGE_URL",
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


@pytest.fixture(autouse=True)
def validator(monkeypatch: pytest.MonkeyPatch) -> LimitsValidator:
    """The enforcing validator every action turn builds; unlisted channels pass (optional)."""
    built = LimitsValidator(
        {
            address: ChannelLimitsConfig(
                address,
                record["min_value"],
                record["max_value"],
                max_step=record.get("max_step"),
                writable=True,
            )
            for address, record in RECORDS.items()
        },
        {"mode": "optional", "on_violation": "error"},
        {address: {**record, "confirm": True} for address, record in RECORDS.items()},
    )
    monkeypatch.setattr(actions.LimitsValidator, "from_config", staticmethod(lambda **_kw: built))
    return built


class _Plans:
    """The plan probe: answers *running* and counts the asks."""

    def __init__(self, running: bool | None = False, on_ask: Any = None) -> None:
        self.running = running
        self.on_ask = on_ask
        self.asked: list[str] = []

    async def __call__(self, target: str) -> bool | None:
        self.asked.append(target)
        if self.on_ask is not None:
            self.on_ask()
        return self.running


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _sessions(
    *, timeout_s: float = 60.0, tool: TrajectoryTool | None = None, sequence: bool = False
) -> tuple[Sessions, _Machine]:
    machine = _Machine()
    startup = TrajectoryStartup(
        target=TARGET,
        connector=machine,
        commands=COMMANDS,
        guarded_run_dir=Path("/unused"),
        settings=TrajectorySettings(action_timeout_s=timeout_s),
        sequence=load_sequence(FIXTURE_DIR, COMMANDS) if sequence else None,
        tool=tool or _tool(),
    )
    layer = ActionLayer(startup, AdmitGate(target=TARGET, section=ARMED))
    return Sessions(layer), machine


def _pend(
    sessions: Sessions,
    client_id: str,
    validator: LimitsValidator,
    *,
    shot_id: str = "shot-1",
) -> Step:
    """Compute the step for a shot with the tool, bounded by the validator, and make it pending."""
    startup = sessions.layer.startup
    shot = _shot(shot_id)
    step = compute_step(
        startup.tool,  # type: ignore[arg-type]
        shot,
        None,
        target=TARGET,
        client_id=client_id,
        bounds=max_steps(validator),
    )
    sessions.step.replace(step, shot)
    sessions.latest_shot_id = shot_id
    return step


async def _apply(sessions: Sessions, client_id: str, body: Any, **kwargs: Any) -> LastAction:
    kwargs.setdefault("plan_running", _Plans())
    return await apply(sessions, OWNER, KIND, client_id, body, **kwargs)


async def _refused(coro: Any, kind: type[Exception] = ApplyRefused) -> Any:
    with pytest.raises(kind) as caught:
        await coro
    return caught.value


def _lock_is_held() -> bool:
    try:
        with lock(
            TARGET, context=ActionContext(target=TARGET, owner="bob", approved_target=TARGET)
        ):
            return False
    except OspreyRunBusy:
        return True


# -- the body ------------------------------------------------------------------------


def test_a_step_body_and_a_putback_body_parse() -> None:
    assert parse_body({"step_digest": "abc"}) == StepApply(digest="abc")
    assert parse_body(
        {"model": MODEL, "values": [{"name": HCM1, "before": 1.5}, {"name": KICKER, "before": 0}]}
    ) == PutBack(model=MODEL, values=((HCM1, 1.5), (KICKER, 0)))


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"step_digest": "abc", "model": MODEL},
        {"step_digest": "abc", "model": MODEL, "values": [{"name": HCM1, "before": 1.0}]},
        {"step_digest": ""},
        {"step_digest": 3},
        {"model": MODEL},
        {"values": [{"name": HCM1, "before": 1.0}]},
        {"model": MODEL, "values": []},
        {"model": MODEL, "values": [{"name": HCM1}]},
        {"model": MODEL, "values": [{"name": HCM1, "before": 1.0, "outcome": "confirmed"}]},
        {"model": MODEL, "values": [{"name": "", "before": 1.0}]},
        {"model": MODEL, "values": [{"name": HCM1, "before": float("nan")}]},
        {"model": MODEL, "values": [{"name": HCM1, "before": None}]},
        {"model": MODEL, "values": [{"name": HCM1, "before": 1.0}, {"name": HCM1, "before": 2.0}]},
        {"model": "", "values": [{"name": HCM1, "before": 1.0}]},
    ],
)
async def test_a_body_that_is_not_exactly_one_shape_is_400_and_nothing_is_read_or_written(
    body: Any,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        refusal = await _refused(_apply(sessions, client_id, body))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (BAD_REQUEST, 400)
    assert machine.writes == []


# -- the step ------------------------------------------------------------------------


async def test_a_step_scaled_by_a_binding_max_step_passes_the_real_limits_validator(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        result = await _apply(sessions, client_id, {"step_digest": step.digest})
    finally:
        await sessions.layer.close()

    # HCM1 binds: its increment sits just inside its max_step, the others scale with it.
    assert step.factor == pytest.approx(0.5 * (1 - MAX_STEP_MARGIN) / 2.0, rel=1e-15)
    assert abs(step.deltas[HCM1]) < 0.5
    assert abs(step.deltas[HCM1]) == pytest.approx(0.5, rel=1e-8)
    assert machine._limits_validator is validator
    assert machine.writes == [
        (HCM1, 1.0 + step.deltas[HCM1]),
        (HCM2, -2.0 + step.deltas[HCM2]),
        (VCM1, 0.5 + step.deltas[VCM1]),
    ]
    assert result == LastAction(
        name=APPLY,
        stopped=False,
        moved=[
            Moved(name=HCM1, before=1.0, outcome="confirmed"),
            Moved(name=HCM2, before=-2.0, outcome="confirmed"),
            Moved(name=VCM1, before=0.5, outcome="confirmed"),
        ],
    )
    assert sessions.step.consumed is True


async def test_only_correctors_with_a_nonzero_delta_are_written_in_column_order(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions(tool=_tool(_Correct([0.2, 0.0, -0.4])))
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        result = await _apply(sessions, client_id, {"step_digest": step.digest})
    finally:
        await sessions.layer.close()
    assert machine.written() == [HCM1, VCM1]
    assert [moved.name for moved in result.moved] == [HCM1, VCM1]


async def test_a_digest_that_is_not_the_pending_steps_is_409_digest_mismatch(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        _pend(sessions, client_id, validator)
        wrong = await _refused(_apply(sessions, client_id, {"step_digest": "0" * 64}))
        sessions.step.void()
        none = await _refused(_apply(sessions, client_id, {"step_digest": "0" * 64}))
    finally:
        await sessions.layer.close()
    assert (wrong.code, wrong.status) == (DIGEST_MISMATCH, 409)
    assert (none.code, none.status) == (DIGEST_MISMATCH, 409)
    assert machine.writes == []


async def test_a_step_of_a_shot_that_is_not_the_latest_is_409_stale_shot(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        sessions.latest_shot_id = "shot-2"
        refusal = await _refused(_apply(sessions, client_id, {"step_digest": step.digest}))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (STALE_SHOT, 409)
    assert machine.writes == []
    assert sessions.step.consumed is False


async def test_a_step_already_applied_is_409_step_consumed(validator: LimitsValidator) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        await _apply(sessions, client_id, {"step_digest": step.digest})
        written = list(machine.writes)
        refusal = await _refused(_apply(sessions, client_id, {"step_digest": step.digest}))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (STEP_CONSUMED, 409)
    assert machine.writes == written


async def test_an_all_zero_step_is_409_zero_step_with_the_tools_status(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions(tool=_tool(_Correct([0.0, 0.0, 0.0], status="nothing to do")))
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        refusal = await _refused(_apply(sessions, client_id, {"step_digest": step.digest}))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status, refusal.detail) == (ZERO_STEP, 409, "nothing to do")
    assert machine.writes == []


async def test_a_value_outside_its_limits_refuses_the_whole_apply_and_keeps_the_step(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    machine.values[HCM2] = 9.9  # HCM2's increment is positive: 9.9 + δ > max 10
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        assert step.deltas[HCM2] > 0.1
        refusal = await _refused(_apply(sessions, client_id, {"step_digest": step.digest}))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (LIMITS_VIOLATION, 409)
    assert machine.writes == []
    assert sessions.step.step is step and sessions.step.consumed is False


async def test_a_running_plan_is_409_and_a_silent_deployed_bridge_is_503(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    running, silent = _Plans(running=True), _Plans(running=None)
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        busy = await _refused(
            _apply(sessions, client_id, {"step_digest": step.digest}, plan_running=running)
        )
        quiet = await _refused(
            _apply(sessions, client_id, {"step_digest": step.digest}, plan_running=silent)
        )
    finally:
        await sessions.layer.close()
    assert (busy.code, busy.status) == (PLAN_RUNNING, 409)
    assert (quiet.code, quiet.status) == (PLAN_RUNNING, 503)
    assert running.asked == silent.asked == [TARGET]
    assert machine.writes == []
    assert sessions.step.step is step and sessions.step.consumed is False


async def test_a_held_run_lock_is_run_busy_and_leaves_the_step_pending(
    _deployment: Path, validator: LimitsValidator
) -> None:
    sessions, machine = _sessions()
    plans = _Plans()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        with held_lock(TARGET, root=_deployment):
            refusal = await _refused(
                _apply(sessions, client_id, {"step_digest": step.digest}, plan_running=plans),
                ActionRefused,
            )
        result = await _apply(sessions, client_id, {"step_digest": step.digest})
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (RUN_BUSY, 409)
    assert plans.asked == []
    assert len(result.moved) == 3


async def test_a_step_replaced_while_the_apply_waited_for_its_turn_is_never_written(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()

    def take_over() -> None:  # lands after the precondition, before the consume
        sessions.step.void()

    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        refusal = await _refused(
            _apply(
                sessions,
                client_id,
                {"step_digest": step.digest},
                plan_running=_Plans(on_ask=take_over),
            )
        )
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (DIGEST_MISMATCH, 409)
    assert machine.writes == []


async def test_a_stop_before_the_first_write_is_action_stopped_and_keeps_the_step(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        refusal = await _refused(
            _apply(
                sessions,
                client_id,
                {"step_digest": step.digest},
                plan_running=_Plans(on_ask=sessions.layer.renew_stop_event),
            )
        )
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (ACTION_STOPPED, 409)
    assert machine.writes == []
    assert sessions.step.consumed is False


async def test_a_non_holder_is_refused_before_the_apply_checks_and_voids_nothing(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        refusal = await _refused(
            _apply(
                sessions,
                "not-the-holder",
                {"model": MODEL, "values": [{"name": HCM1, "before": 0.0}]},
            ),
            SessionRefused,
        )
    finally:
        await sessions.layer.close()
    assert refusal.code == NOT_SESSION_HOLDER
    assert sessions.step.step is step
    assert machine.writes == []


# -- writing ---------------------------------------------------------------------------


async def test_a_write_blocking_past_the_deadline_keeps_the_lock_and_is_listed(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions(timeout_s=5.0)
    clock = _Clock()
    gate = machine.gates[HCM1] = _Gate()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        task = asyncio.ensure_future(
            _apply(sessions, client_id, {"step_digest": step.digest}, clock=clock)
        )
        await gate.started.wait()
        clock.now += 10.0  # the deadline passes while the write is in flight
        await asyncio.sleep(0)
        assert _lock_is_held()
        assert not task.done()
        gate.release.set()
        failure = await _refused(task, ApplyFailed)
        assert not _lock_is_held()
    finally:
        await sessions.layer.close()
    assert failure.moved == [Moved(name=HCM1, before=1.0, outcome="confirmed")]
    assert machine.written() == [HCM1]


async def test_a_stop_during_a_blocking_write_lists_every_non_refused_write(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    gate = machine.gates[HCM2] = _Gate()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        task = asyncio.ensure_future(_apply(sessions, client_id, {"step_digest": step.digest}))
        await gate.started.wait()
        sessions.layer.renew_stop_event()  # Stop, while HCM2's write is in flight
        gate.release.set()
        result = await task
    finally:
        await sessions.layer.close()
    assert result == LastAction(
        name=APPLY,
        stopped=True,
        moved=[
            Moved(name=HCM1, before=1.0, outcome="confirmed"),
            Moved(name=HCM2, before=-2.0, outcome="confirmed"),
        ],
    )
    assert machine.written() == [HCM1, HCM2]


async def test_a_failed_write_ends_the_apply_and_is_listed_with_its_outcome(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    machine.fail.add(HCM2)
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        failure = await _refused(
            _apply(sessions, client_id, {"step_digest": step.digest}), ApplyFailed
        )
    finally:
        await sessions.layer.close()
    assert failure.moved == [
        Moved(name=HCM1, before=1.0, outcome="confirmed"),
        Moved(name=HCM2, before=-2.0, outcome="unconfirmed"),
    ]
    assert machine.written() == [HCM1, HCM2]


async def test_a_write_the_connector_refuses_is_not_listed(validator: LimitsValidator) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        # HCM2 moves past its max_step once the apply's own reads are taken.
        original = machine.write_channel_checked

        async def drift(address: str, value: Any, **kwargs: Any) -> Any:
            if address == HCM1:
                machine.values[HCM2] = -5.0
            return await original(address, value, **kwargs)

        machine.write_channel_checked = drift  # type: ignore[method-assign]
        failure = await _refused(
            _apply(sessions, client_id, {"step_digest": step.digest}), ApplyFailed
        )
    finally:
        await sessions.layer.close()
    assert failure.moved == [Moved(name=HCM1, before=1.0, outcome="confirmed")]
    assert machine.written() == [HCM1]


# -- put-back --------------------------------------------------------------------------


async def test_a_putback_writes_each_before_and_lists_what_it_moved() -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        result = await _apply(
            sessions,
            client_id,
            {
                "model": MODEL,
                "values": [{"name": VCM1, "before": 0.25}, {"name": HCM1, "before": 0.75}],
            },
        )
    finally:
        await sessions.layer.close()
    assert machine.writes == [(VCM1, 0.25), (HCM1, 0.75)]
    assert result.moved == [
        Moved(name=VCM1, before=0.5, outcome="confirmed"),
        Moved(name=HCM1, before=1.0, outcome="confirmed"),
    ]


async def test_a_putback_then_an_apply_of_the_old_digest_is_409(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        await _apply(
            sessions, client_id, {"model": MODEL, "values": [{"name": HCM1, "before": 0.9}]}
        )
        assert sessions.step.step is None
        refusal = await _refused(_apply(sessions, client_id, {"step_digest": step.digest}))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (DIGEST_MISMATCH, 409)
    assert machine.writes == [(HCM1, 0.9)]


async def test_a_putback_for_another_model_is_409_model_mismatch(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        refusal = await _refused(
            _apply(sessions, client_id, {"model": "BR", "values": [{"name": HCM1, "before": 0.9}]})
        )
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (MODEL_MISMATCH, 409)
    assert machine.writes == []
    assert step is not None and sessions.step.step is None  # the put-back voided it first


async def test_a_putback_of_a_name_that_is_no_rm_column_or_module_slot_is_refused() -> None:
    sessions, machine = _sessions(sequence=True)
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        refusal = await _refused(
            _apply(
                sessions,
                client_id,
                {
                    "model": MODEL,
                    "values": [{"name": HCM1, "before": 0.9}, {"name": BUSY, "before": 1}],
                },
                tag_addresses=TAGS,
            )
        )
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (BAD_REQUEST, 400)
    assert BUSY in refusal.detail
    assert machine.writes == []


async def test_a_putback_of_a_setups_module_slots_restores_them() -> None:
    sessions, machine = _sessions(sequence=True)
    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        setup = await sessions.call(
            OWNER,
            KIND,
            client_id,
            setup_action(sessions.layer.startup, tag_addresses=TAGS, bunch_clean=True),
        )
        assert [moved.name for moved in setup.moved] == [*MODES, KICKER, KICKER]
        # A put-back names each setting once: the earliest before of each.
        first: dict[str, Any] = {}
        for moved in setup.moved:
            first.setdefault(moved.name, moved.before)
        result = await _apply(
            sessions,
            client_id,
            {"model": MODEL, "values": [{"name": n, "before": b} for n, b in first.items()]},
            tag_addresses=TAGS,
        )
    finally:
        await sessions.layer.close()
    assert machine.values[MODES[0]] == 3 and machine.values[MODES[1]] == 2
    assert machine.values[KICKER] == 0
    assert result.moved == [
        Moved(name=MODES[0], before=1, outcome="confirmed"),
        Moved(name=MODES[1], before=1, outcome="confirmed"),
        Moved(name=KICKER, before=0, outcome="confirmed"),
    ]


# -- the bridge probe ----------------------------------------------------------------


async def test_with_no_bridge_deployed_no_plan_runs() -> None:
    assert await bridge_plan_running(TARGET) is False


def _bridge_config(root: Path, services: str) -> None:
    from osprey_connectors.workspace import reset_config_cache

    (root / "build" / "config.yml").write_text(
        f"control_system: {{}}\nservices:\n{services}", encoding="utf-8"
    )
    reset_config_cache()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"status": {"manager_state": "executing_queue"}}, True),
        ({"status": {"manager_state": "idle"}}, False),
        ({"status": {"available": False}}, None),
        (httpx.ConnectError("refused"), None),
    ],
)
async def test_a_deployed_bridge_is_asked_whether_its_queue_runs(
    _deployment: Path, monkeypatch: pytest.MonkeyPatch, answer: Any, expected: bool | None
) -> None:
    _bridge_config(_deployment, "  bluesky:\n    port: 8099\n")
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(200, json=answer)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert await bridge_plan_running(TARGET) is expected
    assert asked == ["http://127.0.0.1:8099/queue"]


async def test_a_bridge_lane_serving_another_target_is_not_asked(
    _deployment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bridge_config(_deployment, "  bluesky_live:\n    port: 8099\n    target: live\n")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"asked {request.url}")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert await bridge_plan_running(TARGET) is False


async def test_a_shot_the_sessions_ran_makes_an_older_step_stale(
    validator: LimitsValidator,
) -> None:
    sessions, machine = _sessions()

    def build(_hook: Any) -> Any:  # a shot whose step was never installed
        async def action(_admission: Any, _stop: asyncio.Event) -> ShotResult:
            return _shot("shot-2")

        return action

    try:
        client_id = (await sessions.open(OWNER, KIND)).client_id
        step = _pend(sessions, client_id, validator)
        await sessions.shot(OWNER, KIND, client_id, None, build)  # type: ignore[arg-type]
        assert sessions.latest_shot_id == "shot-2"
        refusal = await _refused(_apply(sessions, client_id, {"step_digest": step.digest}))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (STALE_SHOT, 409)
    assert machine.writes == []
