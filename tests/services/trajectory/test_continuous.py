"""Continuous mode: the chain run again and again on the sessions, until something ends it.

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
    ShotStopped,
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
from osprey.services.trajectory.state import (
    ChainBody,
    ChainResult,
    ContinuousEndedEvent,
    ShotResult,
)
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
        hcorr_index=(0.5, 1.5),
        vcorr_index=(2.5,),
        unsolved={},
        alpha=0.0,
        n_discard=0,
        gain=1.0,
    )


def _sessions(*, min_period_s: float = 0.0, sequence: Any = None) -> tuple[Sessions, _Correct]:
    fake = _Correct()
    startup = SimpleNamespace(
        target=TARGET,
        connector=SimpleNamespace(_limits_validator=None),
        commands=SimpleNamespace(model=MODEL),
        tool=_tool(fake),
        sequence=sequence,
        settings=SimpleNamespace(continuous_min_period_s=min_period_s, action_timeout_s=60.0),
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


class _Chain:
    """Builds one chain action per iteration from the body the run started with.

    Each action re-checks the setup, then injects, then reads out, recording
    each step against its iteration, and answers the steps it ran.
    """

    def __init__(self, body: ChainBody, *, included: int = 1) -> None:
        self.body = body
        self.included = included
        self.log: list[tuple[int, str]] = []
        self.iterations = 0

    def build(self, hook: Any) -> Any:
        steps = tuple(self.body.steps)

        async def chain(_admission: Admission, stop: asyncio.Event) -> ChainResult:
            self.iterations += 1
            number = self.iterations
            if "inject" in steps:
                self.log.append((number, "recheck"))
            for step in steps:
                if stop.is_set():
                    raise ShotStopped([])
                self.log.append((number, step))
                await asyncio.sleep(0)
            shot = None
            if "readout" in steps:
                shot = _shot(f"chain-{number}").model_copy(update={"shots_included": self.included})
                await hook(shot)
            return ChainResult(steps_run=list(steps), commands_fired=[], shot=shot)

        return chain


async def test_a_run_reruns_the_chain_body_it_started_with_one_action_per_turn() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    chain = _Chain(ChainBody(steps=["inject", "readout"]))
    ended: list[ContinuousEndedEvent] = []
    run = sessions.start_continuous(OWNER, KIND, client, _Store(), chain.build, on_end=ended.append)
    await _until(lambda: chain.iterations >= 3)
    sessions.stop(OWNER, KIND)
    event = await _ended(run)
    runs: dict[int, list[str]] = {}
    for number, step in chain.log:
        runs.setdefault(number, []).append(step)
    assert runs[1] == runs[2] == ["recheck", "inject", "readout"]  # same steps, re-check first
    assert event.end == "stopped" and event.shots == chain.iterations
    assert ended == [event] and run.ended == event
    assert sessions.layer.turns.count("Sessions.shot.<locals>.action") == chain.iterations  # type: ignore[attr-defined]
    assert sessions.continuous is None and not sessions.continuous_running


async def test_the_setup_re_check_runs_before_every_iterations_inject() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    chain = _Chain(ChainBody(steps=["arm", "inject", "readout"]))
    run = sessions.start_continuous(OWNER, KIND, client, _Store(), chain.build)
    await _until(lambda: chain.iterations >= 3)
    await asyncio.wait_for(run.cancel(), 2)
    for number in range(1, chain.iterations + 1):
        steps = [step for n, step in chain.log if n == number]
        if "inject" in steps:
            assert steps.index("recheck") < steps.index("inject")


async def test_an_all_zero_step_ends_nothing_and_the_run_goes_on() -> None:
    sessions, fake = _sessions()
    client = await _open(sessions)
    shots = _Shots()
    shots.on_shot = lambda number: fake.zero() if number == 2 else None
    run = _start(sessions, client, shots)
    await _until(lambda: len(shots.started) >= 4)
    assert run.running  # shots 2 onwards solved to a zero step
    event = await asyncio.wait_for(run.cancel(), 2)
    assert event.end == "shutdown"
    assert event.step_reason is None and event.step_status is None


async def test_no_count_of_shots_ends_a_run() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots()
    run = _start(sessions, client, shots)
    await _until(lambda: len(shots.started) >= 150)
    assert run.running
    assert "max_shots" not in run.status().model_dump()
    await asyncio.wait_for(run.cancel(), 2)


async def test_an_iteration_whose_every_injection_failed_ends_the_run_failed() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    chain = _Chain(ChainBody(steps=["inject", "readout"], n_shots=3), included=0)
    run = sessions.start_continuous(OWNER, KIND, client, _Store(), chain.build)
    event = await _ended(run)
    assert (event.end, event.shots) == ("failed", 1)
    assert run.error is None


async def test_a_chain_without_readout_runs_on() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    chain = _Chain(ChainBody(steps=["arm"]))
    run = sessions.start_continuous(OWNER, KIND, client, _Store(), chain.build)
    await _until(lambda: chain.iterations >= 3)
    assert run.running
    await asyncio.wait_for(run.cancel(), 2)


@pytest.mark.parametrize("cause", ["stopped", "taken_over", "released"])
async def test_each_ending_cause_is_published_as_the_runs_end(cause: str) -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    chain = _Chain(ChainBody(steps=["inject", "readout"]))
    ended: list[ContinuousEndedEvent] = []
    run = sessions.start_continuous(OWNER, KIND, client, _Store(), chain.build, on_end=ended.append)
    await _until(lambda: chain.iterations >= 2)
    if cause == "stopped":
        sessions.stop(OWNER, KIND)
    elif cause == "taken_over":
        sessions.take_over("bob", KIND)
    else:
        sessions.release(OWNER, KIND, client)
    event = await _ended(run)
    assert event.end == cause and [e.end for e in ended] == [cause]
    assert not sessions.continuous_running


# -- the minimum period --------------------------------------------------------------


async def test_two_iterations_start_at_least_the_minimum_period_apart() -> None:
    sessions, _fake = _sessions(min_period_s=5.0)
    client = await _open(sessions)
    clock = _Clock()
    starts: list[float] = []
    asked: list[float] = []

    async def sleep(seconds: float) -> None:
        if seconds > 5.0:
            # The lease watcher's wait: the lease never expires here, and the
            # clock moves for the pacing alone.
            await asyncio.Event().wait()
        asked.append(seconds)
        clock.now += seconds
        await asyncio.sleep(0)

    shots = _Shots()

    def on_shot(_number: int) -> None:
        starts.append(clock.now)
        clock.now += 2.0  # the iteration itself takes 2 s

    shots.on_shot = on_shot
    run = _start(sessions, client, shots, clock=clock, sleep=sleep, lease_s=1e9, heartbeat_s=1e8)
    assert run.min_period_s == 5.0  # the service's continuous_min_period_s
    await _until(lambda: len(starts) >= 4)
    await asyncio.wait_for(run.cancel(), 2)
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert gaps and all(gap >= 5.0 for gap in gaps)
    assert 3.0 in asked  # the period less the iteration's own 2 s


async def test_an_iteration_slower_than_the_period_is_followed_at_once() -> None:
    sessions, _fake = _sessions(min_period_s=1.0)
    client = await _open(sessions)
    clock = _Clock()
    asked: list[float] = []

    async def sleep(seconds: float) -> None:
        asked.append(seconds)
        await asyncio.sleep(0)

    shots = _Shots()
    shots.on_shot = lambda _number: setattr(clock, "now", clock.now + 3.0)
    run = _start(sessions, client, shots, clock=clock, sleep=sleep, lease_s=1e9, heartbeat_s=1e8)
    await _until(lambda: len(shots.started) >= 3)
    await asyncio.wait_for(run.cancel(), 2)
    assert [seconds for seconds in asked if seconds < 1e8] == []  # no iteration waited


async def test_a_stop_during_the_wait_between_iterations_ends_the_run_at_once() -> None:
    sessions, _fake = _sessions(min_period_s=1000.0)
    client = await _open(sessions)
    never = asyncio.Event()

    async def sleep(_seconds: float) -> None:
        await never.wait()

    shots = _Shots()
    run = _start(sessions, client, shots, sleep=sleep)
    await _until(lambda: shots.started == [1] and sessions.latest_shot_id == "shot-1")
    await asyncio.sleep(0.01)
    sessions.stop(OWNER, KIND)
    event = await _ended(run)
    assert (event.end, event.shots) == ("stopped", 1)


def test_a_negative_minimum_period_is_refused() -> None:
    sessions, _fake = _sessions()
    with pytest.raises(ValueError):
        ContinuousRun(
            sessions,
            OWNER,
            KIND,
            "c",
            _Store(),
            _Shots().build,  # type: ignore[arg-type]
            min_period_s=-1.0,
        )


async def test_setting_the_stop_event_current_at_the_start_ends_the_run_mid_shot() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots)
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
    run = _start(sessions, client, shots)
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
    run = _start(sessions, client, shots)
    event = await _ended(run)
    assert (event.end, event.shots) == ("failed", 1)
    assert isinstance(run.error, RuntimeError)


async def test_cancelling_the_run_ends_it_as_shutdown() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots)
    await _until(lambda: shots.started == [1])
    event = await asyncio.wait_for(run.cancel(), 2)
    assert event.end == "shutdown"
    assert not sessions.continuous_running


async def test_an_end_named_first_wins_over_the_stop_that_follows() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots)
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
    run = _start(sessions, client, shots, clock=clock, sleep=_tick)
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
    run = _start(sessions, client, shots, clock=clock, sleep=_tick)
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
    run = _start(sessions, client, _Shots(gated=True), clock=clock, sleep=_tick)
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
            heartbeat_s=10.0,
            lease_s=10.0,
        )


# -- starting ------------------------------------------------------------------------


async def test_a_second_start_while_a_run_is_in_progress_is_409() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    run = _start(sessions, client, _Shots(gated=True))
    with pytest.raises(SessionRefused) as caught:
        _start(sessions, client, _Shots())
    assert (caught.value.code, caught.value.status) == (CONTINUOUS_RUNNING, 409)
    assert sessions.continuous is run
    await asyncio.wait_for(run.cancel(), 2)


async def test_a_start_from_a_non_holder_is_refused() -> None:
    sessions, _fake = _sessions()
    await _open(sessions)
    with pytest.raises(SessionRefused) as caught:
        _start(sessions, "not-the-holder", _Shots())
    assert caught.value.code == NOT_SESSION_HOLDER
    assert sessions.continuous is None


# -- apply during a run ----------------------------------------------------


async def test_an_apply_during_a_run_is_409_and_runs_between_two_shots() -> None:
    sessions, _fake = _sessions()
    client = await _open(sessions)
    shots = _Shots(gated=True)
    run = _start(sessions, client, shots)
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
    run = _start(sessions, client, shots)
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
    shots = _Shots()
    run = _start(sessions, client, shots)
    await _until(lambda: len(shots.started) >= 1)
    sessions.stop(OWNER, KIND)
    await _ended(run)
    with pytest.raises(ApplyRefused) as caught:
        await apply(sessions, OWNER, KIND, client, {"step_digest": "not-the-step"})
    assert caught.value.code == DIGEST_MISMATCH
