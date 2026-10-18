"""Continuous mode: shots one after another on the sessions, until something ends them.

The run drives real :class:`Sessions` over a small action layer that runs one
action at a time in submission order, binds each action to the stop event
current when it was submitted and refuses it ``action_stopped`` when that
event is set before it starts — the turn order of the real layer, without its
probe, limits and run lock. Each shot calls the real post-shot hook, which
computes the real step over a fake pyAML tool whose answer a test sets per
shot. Time is an injected clock; the lease watch sleeps an injected sleep.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from osprey.services.trajectory.acquisition import (
    CONTINUOUS_RUNNING as SETUP_CONTINUOUS_RUNNING,
)
from osprey.services.trajectory.acquisition import (
    AcquisitionRefused,
    ShotStopped,
    setup_action,
)
from osprey.services.trajectory.actions import ACTION_STOPPED, ActionRefused
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.apply import CONTINUOUS_RUNNING as APPLY_CONTINUOUS_RUNNING
from osprey.services.trajectory.apply import DIGEST_MISMATCH, ApplyRefused, apply
from osprey.services.trajectory.continuous import (
    DEFAULT_HEARTBEAT_S,
    DEFAULT_LEASE_S,
    ContinuousRun,
)
from osprey.services.trajectory.sessions import (
    CONTINUOUS_RUNNING,
    NOT_SESSION_HOLDER,
    SessionRefused,
    Sessions,
)
from osprey.services.trajectory.state import ContinuousEndedEvent, ShotResult
from osprey.services.trajectory.step import TrajectoryTool

TARGET = "va"
MODEL = "SR"
OWNER = "alice"
KIND = "panel"
BPMS = ("BPM1", "BPM2", "BPM3")
X_ROWS = ("SR:BPM1:X", "SR:BPM2:X", "SR:BPM3:X")
Y_ROWS = ("SR:BPM1:Y", "SR:BPM2:Y", "SR:BPM3:Y")
H_CORR = ("SR:HCM1", "SR:HCM2")
V_CORR = ("SR:VCM1",)
NONZERO = [2.0, -1.0, 0.5]


# -- fakes ---------------------------------------------------------------------------


class _Layer:
    """Runs one action at a time, in submission order, bound to the stop event at submit."""

    def __init__(self, startup: Any) -> None:
        self.startup = startup
        self._stop = asyncio.Event()
        self._turn = asyncio.Lock()
        self.turns: list[str] = []
        self.in_turn = False

    @property
    def stop_event(self) -> asyncio.Event:
        return self._stop

    def renew_stop_event(self) -> asyncio.Event:
        self._stop.set()
        self._stop = asyncio.Event()
        return self._stop

    def control(self, owner: str, kind: str, fn: Any, *, admit: bool = True) -> Any:
        return fn(_admission(owner, kind) if admit else None)

    async def submit(self, owner: str, kind: str, action: Any, *, precondition: Any = None) -> Any:
        stop = self._stop
        async with self._turn:
            if stop.is_set():
                raise ActionRefused(ACTION_STOPPED, "stopped before it started")
            admission = _admission(owner, kind)
            self.turns.append(getattr(action, "__qualname__", "action"))
            if precondition is not None:
                precondition(admission)
            self.in_turn = True
            try:
                return await action(admission, stop)
            finally:
                self.in_turn = False


def _admission(owner: str, kind: str) -> Admission:
    return Admission(owner=owner, kind=kind, target=TARGET, generation=0)


class _Correct:
    """A fake pyAML tool whose step a test sets between shots."""

    def __init__(self) -> None:
        self.step = np.asarray(NONZERO, dtype=float)
        self.reason = "OK"
        self.status = ""

    def zero(self, reason: str = "ON_REFERENCE", status: str = "on the reference") -> None:
        self.step = np.zeros(3)
        self.reason = reason
        self.status = status

    def correct(self, *_args: Any, **_kwargs: Any) -> Any:
        inverse = SimpleNamespace(
            no_response_names=(),
            rank_discarded=0,
            singular_values=np.array([3.0, 1.0, 0.5]),
            inverse_weights=np.array([0.3, 0.9, 1.0]),
            spectrum_kind=tuple(SimpleNamespace(value="LIVE") for _ in range(3)),
            n_discard=0,
        )
        return SimpleNamespace(
            step=self.step,
            names=(*H_CORR, *V_CORR),
            reason=SimpleNamespace(value=self.reason),
            status=self.status,
            all_zero=not np.any(self.step),
            zeroed_h=(),
            zeroed_v=(),
            pseudo_inverse=inverse,
        )


class _Store:
    """A store with no current reference."""

    async def current(self, model: str) -> str | None:
        assert model == MODEL
        return None


class _Shots:
    """Builds one shot action per iteration; each waits for a permit or the stop event."""

    def __init__(self, *, gated: bool = False) -> None:
        self.permits = asyncio.Semaphore(0) if gated else None
        self.started: list[int] = []
        self.fail: BaseException | None = None
        self.on_shot: Any = None

    def build(self, hook: Any) -> Any:
        async def shot(_admission: Admission, stop: asyncio.Event) -> ShotResult:
            number = len(self.started) + 1
            self.started.append(number)
            if self.permits is not None:
                await _first(self.permits.acquire(), stop.wait())
            if stop.is_set():
                raise ShotStopped([])
            if self.fail is not None:
                raise self.fail
            if self.on_shot is not None:
                self.on_shot(number)
            result = _shot(f"shot-{number}")
            await hook(result)
            return result

        return shot


async def _first(*coros: Any) -> None:
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def _tick(_seconds: float) -> None:
    await asyncio.sleep(0.001)


def _shot(shot_id: str) -> ShotResult:
    return ShotResult(
        shot_id=shot_id,
        model=MODEL,
        x={row: 0.1 * (i + 1) for i, row in enumerate(X_ROWS)},
        y={row: -0.1 * (i + 1) for i, row in enumerate(Y_ROWS)},
        sum={f"S{i + 1}": 1.0 for i in range(len(BPMS))},
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


def _tool(fake: _Correct) -> TrajectoryTool:
    return TrajectoryTool(
        model=MODEL,
        tool=fake,
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


def _sessions(*, max_shots: int = 100, sequence: Any = None) -> tuple[Sessions, _Correct]:
    fake = _Correct()
    startup = SimpleNamespace(
        target=TARGET,
        connector=SimpleNamespace(_limits_validator=None),
        commands=SimpleNamespace(model=MODEL),
        tool=_tool(fake),
        sequence=sequence,
        tag_addresses={},
        settings=SimpleNamespace(continuous_max_shots=max_shots, action_timeout_s=60.0),
    )
    return Sessions(_Layer(startup)), fake  # type: ignore[arg-type]


async def _open(sessions: Sessions, owner: str = OWNER) -> str:
    return (await sessions.open(owner, KIND)).client_id


def _start(sessions: Sessions, client_id: str, shots: _Shots, **kwargs: Any) -> ContinuousRun:
    return sessions.start_continuous(OWNER, KIND, client_id, _Store(), shots.build, **kwargs)  # type: ignore[arg-type]


async def _ended(run: ContinuousRun) -> ContinuousEndedEvent:
    return await asyncio.wait_for(run.wait(), 2)


async def _until(predicate: Any) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 2)


# -- the end conditions --------------------------------------------------------------


async def test_a_run_submits_one_shot_action_per_turn_and_ends_at_max_shots() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    ended: list[ContinuousEndedEvent] = []
    run = _start(sessions, client, _Shots(), max_shots=3, on_end=ended.append)
    event = await _ended(run)
    assert (event.end, event.shots) == ("max_shots", 3)
    assert event.step_reason is None and event.step_status is None
    assert ended == [event] and run.ended == event
    assert sessions.layer.turns.count("Sessions.shot.<locals>.action") == 3  # type: ignore[attr-defined]
    step = sessions.step.step
    assert step is not None and step.shot_id == "shot-3" and not sessions.step.consumed
    assert sessions.continuous is None and not sessions.continuous_running


async def test_a_run_without_a_limit_takes_the_services_continuous_max_shots() -> None:
    sessions, _fake = _sessions(max_shots=2)
    client = await _open(sessions)
    run = _start(sessions, client, _Shots())
    assert run.max_shots == 2
    assert (await _ended(run)).end == "max_shots"


async def test_an_all_zero_step_ends_the_run_with_the_steps_reason_and_status() -> None:
    sessions, fake = _sessions()
    client = await _open(sessions)
    shots = _Shots()
    shots.on_shot = lambda number: fake.zero() if number == 2 else None
    run = _start(sessions, client, shots, max_shots=10)
    event = await _ended(run)
    assert (event.end, event.shots) == ("zero_step", 2)
    assert (event.step_reason, event.step_status) == ("ON_REFERENCE", "on the reference")


async def test_setting_the_stop_event_current_at_the_start_ends_the_run_mid_shot() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=10)
    await _until(lambda: shots.started == [1])
    shots.permits.release()  # type: ignore[union-attr]
    await _until(lambda: shots.started == [1, 2])
    sessions.layer.renew_stop_event()  # a Stop: the holder stays
    event = await _ended(run)
    assert (event.end, event.shots) == ("stopped", 2)
    assert sessions.holder is not None and sessions.holder.client_id == client


async def test_a_shot_queued_after_the_stop_is_refused_and_the_run_ends_stopped() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots()
    release = asyncio.Event()

    async def blocker(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    holding = asyncio.create_task(sessions.layer.submit(OWNER, KIND, blocker))
    await _until(lambda: sessions.layer.in_turn)  # type: ignore[attr-defined]
    run = _start(sessions, client, shots, max_shots=10)
    await asyncio.sleep(0.01)
    sessions.layer.renew_stop_event()
    release.set()
    event = await _ended(run)
    await holding
    assert (event.end, event.shots) == ("stopped", 1)
    assert shots.started == []


async def test_the_holders_stop_reads_through_to_the_layers_current_event() -> None:
    sessions, _fake = _sessions()
    await _open(sessions)
    holder = sessions.holder
    assert holder is not None
    first = holder.stop
    renewed = sessions.layer.renew_stop_event()
    assert first.is_set() and holder.stop is renewed and not holder.stop.is_set()


async def test_a_shot_that_fails_ends_the_run_failed_with_its_error() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots()
    shots.fail = RuntimeError("the connector broke")
    run = _start(sessions, client, shots, max_shots=10)
    event = await _ended(run)
    assert (event.end, event.shots) == ("failed", 1)
    assert isinstance(run.error, RuntimeError)


async def test_cancelling_the_run_ends_it_as_shutdown() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=10)
    await _until(lambda: shots.started == [1])
    event = await asyncio.wait_for(run.cancel(), 2)
    assert event.end == "shutdown"
    assert not sessions.continuous_running


async def test_an_end_named_first_wins_over_the_stop_that_follows() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=10)
    await _until(lambda: shots.started == [1])
    assert run.end("taken_over") is True
    sessions.layer.renew_stop_event()
    assert (await _ended(run)).end == "taken_over"
    assert run.end("stopped") is False


# -- the lease -----------------------------------------------------------------------


async def test_heartbeats_keep_the_run_past_its_lease() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    clock = _Clock()
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=100, clock=clock, sleep=_tick)
    assert run.status().lease_expires_in_s == DEFAULT_LEASE_S
    for _ in range(5):
        clock.now += DEFAULT_HEARTBEAT_S
        await asyncio.sleep(0.01)
        status = sessions.heartbeat(OWNER, KIND, client)
        assert status is not None and status.lease_expires_in_s == DEFAULT_LEASE_S
        shots.permits.release()  # type: ignore[union-attr]
        await asyncio.sleep(0.01)
    assert run.running and sessions.continuous_running
    assert len(shots.started) >= 5
    await asyncio.wait_for(run.cancel(), 2)


async def test_a_lapsed_lease_ends_the_run_voids_the_step_and_renews_the_stop_event() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    clock = _Clock()
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=100, clock=clock, sleep=_tick)
    stop_at_start = sessions.layer.stop_event
    shots.permits.release()  # type: ignore[union-attr]
    await _until(lambda: shots.started == [1, 2])
    assert sessions.step.step is not None
    clock.now += DEFAULT_LEASE_S - 0.5
    await asyncio.sleep(0.01)
    assert run.running
    clock.now += 1.0
    event = await _ended(run)
    assert (event.end, event.shots) == ("lease_expired", 2)
    assert sessions.step.step is None
    assert stop_at_start.is_set() and not sessions.layer.stop_event.is_set()
    assert sessions.holder is not None and sessions.holder.client_id == client


async def test_a_shot_finishing_after_the_lease_lapsed_leaves_no_step() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    clock = _Clock()
    lapse = asyncio.Event()
    finish = asyncio.Event()

    def build(hook: Any) -> Any:
        async def shot(_admission: Admission, _stop: asyncio.Event) -> ShotResult:
            lapse.set()
            await finish.wait()  # ignores the stop: it lands after the expiry
            result = _shot("late")
            await hook(result)
            return result

        return shot

    run = sessions.start_continuous(
        OWNER,
        KIND,
        client,
        _Store(),
        build,
        max_shots=5,
        clock=clock,
        sleep=_tick,  # type: ignore[arg-type]
    )
    await asyncio.wait_for(lapse.wait(), 2)
    clock.now += DEFAULT_LEASE_S + 1
    await _until(lambda: sessions.layer.stop_event is not run._stop)
    finish.set()
    event = await _ended(run)
    assert event.end == "lease_expired"
    assert sessions.step.step is None


async def test_a_heartbeat_from_a_non_holder_is_refused_and_moves_nothing() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    clock = _Clock()
    run = _start(sessions, client, _Shots(gated=True), max_shots=10, clock=clock, sleep=_tick)
    clock.now += 4.0
    with pytest.raises(SessionRefused) as caught:
        sessions.heartbeat(OWNER, KIND, "someone-else")
    assert caught.value.code == NOT_SESSION_HOLDER
    assert run.status().lease_expires_in_s == DEFAULT_LEASE_S - 4.0
    await asyncio.wait_for(run.cancel(), 2)


async def test_a_heartbeat_with_no_run_answers_none_since_only_a_run_holds_a_lease() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    assert sessions.heartbeat(OWNER, KIND, client) is None


def test_a_heartbeat_interval_must_be_shorter_than_the_lease() -> None:
    sessions, _fake = _sessions()
    with pytest.raises(ValueError):
        ContinuousRun(
            sessions,
            OWNER,
            KIND,
            "c",
            _Store(),
            _Shots().build,  # type: ignore[arg-type]
            max_shots=1,
            heartbeat_s=10.0,
            lease_s=10.0,
        )


# -- starting ------------------------------------------------------------------------


async def test_a_second_start_while_a_run_is_in_progress_is_409() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    run = _start(sessions, client, _Shots(gated=True), max_shots=10)
    with pytest.raises(SessionRefused) as caught:
        _start(sessions, client, _Shots(), max_shots=10)
    assert (caught.value.code, caught.value.status) == (CONTINUOUS_RUNNING, 409)
    assert sessions.continuous is run
    await asyncio.wait_for(run.cancel(), 2)


async def test_a_start_from_a_non_holder_is_refused() -> None:
    sessions, _fake = _sessions()
    await _open(sessions)
    with pytest.raises(SessionRefused) as caught:
        _start(sessions, "not-the-holder", _Shots(), max_shots=10)
    assert caught.value.code == NOT_SESSION_HOLDER
    assert sessions.continuous is None


# -- apply and setup during a run ----------------------------------------------------


async def test_an_apply_during_a_run_is_409_and_runs_between_two_shots() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=10)
    await _until(lambda: shots.started == [1])
    applying = asyncio.create_task(apply(sessions, OWNER, KIND, client, {"step_digest": "d"}))
    await asyncio.sleep(0.01)
    assert not applying.done()  # queued behind the shot in its turn
    shots.permits.release()  # type: ignore[union-attr]
    with pytest.raises(ApplyRefused) as caught:
        await asyncio.wait_for(applying, 2)
    assert (caught.value.code, caught.value.status) == (APPLY_CONTINUOUS_RUNNING, 409)
    turns = sessions.layer.turns  # type: ignore[attr-defined]
    shot = "Sessions.shot.<locals>.action"
    assert turns[1:4] == [shot, "apply.<locals>.action", shot]  # between two shots
    await asyncio.wait_for(run.cancel(), 2)


async def test_a_putback_during_a_run_is_409_and_voids_nothing() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=10)
    shots.permits.release()  # type: ignore[union-attr]
    await _until(lambda: shots.started == [1, 2])
    step = sessions.step.step
    assert step is not None
    body = {"model": MODEL, "values": [{"name": H_CORR[0], "before": 1.0}]}
    applying = asyncio.create_task(apply(sessions, OWNER, KIND, client, body))
    shots.permits.release()  # type: ignore[union-attr]
    with pytest.raises(ApplyRefused) as caught:
        await asyncio.wait_for(applying, 2)
    assert caught.value.code == APPLY_CONTINUOUS_RUNNING
    assert sessions.step.step is not None
    await asyncio.wait_for(run.cancel(), 2)


async def test_an_apply_after_the_run_ended_is_not_refused_continuous_running() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    await _ended(_start(sessions, client, _Shots(), max_shots=1))
    with pytest.raises(ApplyRefused) as caught:
        await apply(sessions, OWNER, KIND, client, {"step_digest": "not-the-step"})
    assert caught.value.code == DIGEST_MISMATCH


async def test_a_setup_during_a_run_is_409_and_voids_nothing() -> None:
    module = SimpleNamespace(entry_point=lambda _name: None, path="m.py", metadata={})
    sessions, _fake = _sessions(sequence=module)
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots, max_shots=10)
    shots.permits.release()  # type: ignore[union-attr]
    await _until(lambda: shots.started == [1, 2])
    with pytest.raises(AcquisitionRefused) as caught:
        await sessions.setup(OWNER, KIND, client)
    assert (caught.value.code, caught.value.status) == (SETUP_CONTINUOUS_RUNNING, 409)
    assert sessions.step.step is not None
    await asyncio.wait_for(run.cancel(), 2)
    result = await sessions.setup(OWNER, KIND, client)
    assert result.name == "setup" and sessions.step.step is None


async def test_a_setup_action_refuses_a_run_at_build_and_again_in_its_turn() -> None:
    module = SimpleNamespace(entry_point=lambda _name: None, path="m.py", metadata={})
    startup = SimpleNamespace(sequence=module)
    with pytest.raises(AcquisitionRefused) as caught:
        setup_action(startup, continuous_running=lambda: True)  # type: ignore[arg-type]
    assert (caught.value.code, caught.value.status) == (SETUP_CONTINUOUS_RUNNING, 409)
    running = [False]
    action = setup_action(startup, continuous_running=lambda: running[0])  # type: ignore[arg-type]
    running[0] = True
    with pytest.raises(AcquisitionRefused) as caught:
        await action(_admission(OWNER, KIND), asyncio.Event())
    assert caught.value.code == SETUP_CONTINUOUS_RUNNING
