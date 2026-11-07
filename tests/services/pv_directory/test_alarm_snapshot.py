"""Tests for the alarm snapshot task: one backend's lifetime, its answers passed on.

The snapshot starts the backend on first read, waits for its catch-up up to the
caller's deadline and hands the backend's answer back unchanged, ``stale`` included.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

contract = pytest.importorskip("osprey_connectors.pv_directory.contract")
_replay = pytest.importorskip("osprey_connectors.alarm_state._replay")

import osprey_connectors_tests  # noqa: E402
from osprey.services.pv_directory.alarm_snapshot import AlarmSnapshot  # noqa: E402

FIXTURES = Path(osprey_connectors_tests.__file__).parent / "alarm_state" / "fixtures"
OLD_FEED = FIXTURES / "old-feed.ndjson"
STOPPED_FEED = FIXTURES / "stopped-feed.ndjson"
PV = "DEMO:RF1:CAV:Temp-I"
LOADING = contract.Unavailable("alarm feed still loading")
LOAD_MOMENT = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)


def _replay_backend(path: Path):
    spec = contract.BackendSpec(
        kind="alarm_state",
        type="alarm_replay",
        target="va",
        settings={"replay_file": str(path)},
        rehearsal=False,
    )
    return _replay.ReplayAlarmState(spec)


class _StillLoading:
    """A backend whose feed never catches up: ``start`` waits until it is closed."""

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self._release = asyncio.Event()

    async def start(self) -> None:
        self.started += 1
        await self._release.wait()

    async def aclose(self) -> None:
        self.closed += 1
        self._release.set()

    async def state_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        return LOADING

    async def config_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        return LOADING


class _Answering(_StillLoading):
    """A backend that has caught up and answers *answer* to every read."""

    def __init__(self, answer) -> None:
        super().__init__()
        self._answer = answer

    async def start(self) -> None:
        self.started += 1

    async def state_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
        return self._answer

    async def config_for(self, pv: str, deadline: float):  # noqa: ARG002 - a fake
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

    assert isinstance(state, contract.AlarmStateRecord)
    assert state.severity == "MAJOR"
    assert state.latched is True
    assert state.stale is False
    assert config.in_tree == "yes"
    assert config.label == "live configuration"
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
        moment[0] = LOAD_MOMENT + timedelta(seconds=contract.ALARM_STALE_WINDOW_S + 1)
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


@pytest.mark.parametrize(
    "reason",
    [
        "Kafka client not installed (install osprey-framework[alarm-state]; "
        "for an image, through OSPREY_PIP_SPEC)",
        "alarm broker cannot be reached",
    ],
)
async def test_a_feed_that_cannot_answer_is_passed_on_never_no_alarm(reason):
    answer = contract.Unavailable(reason)
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
