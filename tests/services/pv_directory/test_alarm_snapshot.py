"""Tests for the alarm snapshot task: one backend's lifetime, its answers passed on.

The snapshot starts the backend on first read, reads it once with what is left of
the caller's deadline and hands the backend's answer back unchanged, ``stale``
included. The backend does the waiting for catch-up.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from osprey_connectors.alarm_state import AlarmStateRecord, AlarmStates
from osprey_connectors.service_answers import BackendSpec, Unavailable

_replay = pytest.importorskip("osprey_connectors.alarm_state._replay")

import osprey_connectors_tests  # noqa: E402
from osprey.services.pv_directory.alarm_snapshot import AlarmSnapshot  # noqa: E402

FIXTURES = Path(osprey_connectors_tests.__file__).parent / "alarm_state" / "fixtures"
OLD_FEED = FIXTURES / "old-feed.ndjson"
STOPPED_FEED = FIXTURES / "stopped-feed.ndjson"
PV = "DEMO:RF1:CAV:Temp-I"
LOADING = Unavailable("alarm feed still loading")
LOAD_MOMENT = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)


def _replay_backend(path: Path):
    spec = BackendSpec(
        kind="alarm_state",
        type="alarm_replay",
        target="va",
        settings={"replay_file": str(path)},
        rehearsal=False,
    )
    return _replay.ReplayAlarmState(spec)


class _StillLoading:
    """A backend whose feed never catches up: ``start`` waits until it is closed.

    Like every backend it honours its deadline: a read waits it out, then answers loading.
    """

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.deadlines: list[float] = []
        self._release = asyncio.Event()

    async def start(self) -> None:
        self.started += 1
        await self._release.wait()

    async def aclose(self) -> None:
        self.closed += 1
        self._release.set()

    async def state_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        return await self._read(deadline, LOADING)

    async def config_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        return await self._read(deadline, LOADING)

    async def states(self, deadline: float):
        return await self._read(deadline, LOADING)

    async def _read(self, deadline: float, answer):
        self.deadlines.append(deadline)
        await asyncio.sleep(max(deadline, 0.0))
        return answer


class _Answering(_StillLoading):
    """A backend that has its answer and gives *answer* to every read at once."""

    def __init__(self, answer) -> None:
        super().__init__()
        self._answer = answer

    async def start(self) -> None:
        self.started += 1

    async def state_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        self.deadlines.append(deadline)
        return self._answer

    async def config_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        self.deadlines.append(deadline)
        return self._answer

    async def states(self, deadline: float):
        self.deadlines.append(deadline)
        return self._answer


def _other_tasks() -> set[asyncio.Task]:
    return {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}


async def test_the_first_read_answers_the_snapshot():
    snapshot = AlarmSnapshot(_replay_backend(OLD_FEED), "va")
    try:
        state = await snapshot.state_for(PV, 2.0)
        config = await snapshot.config_for(PV, 2.0)
    finally:
        await snapshot.aclose()

    assert isinstance(state, AlarmStateRecord)
    assert state.severity == "MAJOR"
    assert state.latched is True
    assert state.stale is False
    assert config.in_tree == "yes"
    assert snapshot.target == "va"


async def test_the_stopped_feed_reads_stale():
    snapshot = AlarmSnapshot(_replay_backend(STOPPED_FEED), "va")
    try:
        state = await snapshot.state_for(PV, 2.0)
    finally:
        await snapshot.aclose()

    assert state.severity == "MAJOR"
    assert state.stale is True


async def test_a_replay_is_not_stale_after_the_window_passes(monkeypatch):
    moment = [LOAD_MOMENT]
    monkeypatch.setattr(_replay, "_now", lambda: moment[0])
    snapshot = AlarmSnapshot(_replay_backend(OLD_FEED), "va")
    try:
        assert (await snapshot.state_for(PV, 2.0)).stale is False
        moment[0] = LOAD_MOMENT + timedelta(hours=1)
        assert (await snapshot.state_for(PV, 2.0)).stale is False
    finally:
        await snapshot.aclose()


async def test_a_feed_still_loading_at_the_deadline_answers_loading():
    backend = _StillLoading()
    snapshot = AlarmSnapshot(backend, "live")
    try:
        assert await snapshot.state_for(PV, 0.05) == LOADING
        assert await snapshot.config_for(PV, 0.05) == LOADING
        assert backend.started == 1
    finally:
        await snapshot.aclose()


async def test_a_snapshot_read_reads_the_backend_once_with_the_deadline_left():
    backend = _Answering(LOADING)
    snapshot = AlarmSnapshot(backend, "live")
    try:
        assert await snapshot.state_for(PV, 0.2) == LOADING
        assert await snapshot.config_for(PV, 0.2) == LOADING
    finally:
        await snapshot.aclose()

    assert len(backend.deadlines) == 2
    assert all(0.0 < deadline <= 0.2 for deadline in backend.deadlines)


@pytest.mark.parametrize(
    "reason",
    [
        "Kafka client not installed (install osprey-framework[alarm-state]; "
        "for an image, through OSPREY_PIP_SPEC)",
        "alarm broker cannot be reached",
    ],
)
async def test_a_feed_that_cannot_answer_is_passed_on_never_no_alarm(reason):
    answer = Unavailable(reason)
    snapshot = AlarmSnapshot(_Answering(answer), "live")
    try:
        assert await snapshot.state_for(PV, 0.5) is answer
        assert await snapshot.config_for(PV, 0.5) is answer
    finally:
        await snapshot.aclose()


async def test_stop_leaves_no_task_running():
    backend = _StillLoading()
    snapshot = AlarmSnapshot(backend, "live")
    before = _other_tasks()
    await snapshot.state_for(PV, 0.01)
    assert _other_tasks() - before

    await snapshot.aclose()
    await snapshot.aclose()

    assert _other_tasks() - before == set()
    assert backend.closed == 1


async def test_stop_of_a_replay_leaves_no_task_running():
    before = _other_tasks()
    snapshot = AlarmSnapshot(_replay_backend(OLD_FEED), "va")
    await snapshot.state_for(PV, 2.0)

    await snapshot.aclose()

    assert _other_tasks() - before == set()


async def test_no_read_means_no_task():
    before = _other_tasks()
    snapshot = AlarmSnapshot(_StillLoading(), "live")

    assert _other_tasks() - before == set()
    await snapshot.aclose()


class _Probed(_StillLoading):
    """A feed that catches up after *load_s* seconds; its probe answers at once."""

    def __init__(self, load_s: float) -> None:
        super().__init__()
        self._load_s = load_s
        self._caught_up = False
        self.probes: list[float] = []

    async def start(self) -> None:
        self.started += 1

    async def state_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        self.deadlines.append(deadline)
        await asyncio.sleep(min(self._load_s, deadline))
        self._caught_up = deadline >= self._load_s
        return None if self._caught_up else LOADING

    async def probe(self, deadline: float):
        self.probes.append(deadline)
        return "read 3 messages; last contact 0 s ago" if self._caught_up else LOADING


async def test_a_probe_starts_the_feed_and_answers_once_it_caught_up():
    backend = _Probed(load_s=0.05)
    snapshot = AlarmSnapshot(backend, "live")
    try:
        assert await snapshot.probe(1.0) == "read 3 messages; last contact 0 s ago"
    finally:
        await snapshot.aclose()

    assert backend.started == 1
    assert len(backend.probes) == 1
    assert 0.0 <= backend.probes[0] < 1.0


async def test_a_probe_answers_loading_only_when_the_deadline_ran_out():
    backend = _Probed(load_s=5.0)
    snapshot = AlarmSnapshot(backend, "live")
    try:
        assert await snapshot.probe(0.05) == LOADING
    finally:
        await snapshot.aclose()


async def test_a_probe_of_a_replay_answers_its_messages():
    snapshot = AlarmSnapshot(_replay_backend(STOPPED_FEED), "va")
    try:
        answer = await snapshot.probe(2.0)
    finally:
        await snapshot.aclose()

    assert answer.startswith("read ")
    assert answer.endswith("from the replay file; the feed stopped")


# -- the list of every channel's state ---------------------------------------------


async def test_the_list_is_the_backends_answer_read_once_after_the_start():
    snapshot = AlarmSnapshot(_replay_backend(OLD_FEED), "va")
    try:
        states = await snapshot.states(2.0)
    finally:
        await snapshot.aclose()

    assert isinstance(states, AlarmStates)
    assert {row.channel for row in states.rows} == {PV, "DEMO:VAC1:GAUGE:Pres-I"}
    assert states.counts == {"MAJOR": 1, "MINOR": 1}


async def test_a_list_read_waits_for_the_start_then_reads_once_with_the_deadline_left():
    backend = _StillLoading()
    snapshot = AlarmSnapshot(backend, "live")
    try:
        assert await snapshot.states(0.05) == LOADING
    finally:
        await snapshot.aclose()

    assert backend.started == 1
    assert len(backend.deadlines) == 1
    assert 0.0 <= backend.deadlines[0] < 0.05


async def test_a_list_the_feed_cannot_answer_is_passed_on():
    answer = Unavailable("alarm broker cannot be reached")
    backend = _Answering(answer)
    snapshot = AlarmSnapshot(backend, "live")
    try:
        assert await snapshot.states(0.5) is answer
    finally:
        await snapshot.aclose()

    assert len(backend.deadlines) == 1
