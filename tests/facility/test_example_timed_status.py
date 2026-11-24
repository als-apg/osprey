"""The example's ES1 statuses play their act in real time when the in-process simulator serves them.

A write to the stage's move command or the detector's acquire command makes
its status read its act for the ``for_s`` its seed states, then its ``idle``
again. A subscriber sees both edges; the idle edge arrives on the simulator's
next tick after ``for_s``. The simulator view is the demo build's, served from
a copy that ticks every :data:`TICK_S`.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests._builds import BuiltProject
from tests._live import async_wait_until
from tests.facility.served_tree import in_process_config

pytestmark = [pytest.mark.slow, pytest.mark.wallclock]

#: The simulator's tick in these tests: the idle edge arrives on the tick after ``for_s``.
TICK_S = 0.05
#: How much later than ``for_s`` plus one tick the idle edge may arrive.
LATE_S = 1.0

MOVE, MOVE_STATUS = "ES1:MOT:STAGE:01:MOVE:SP", "ES1:MOT:STAGE:01:STATUS:MOVE"
ACQUIRE, ACQUIRE_STATUS = "ES1:DET:MCA:01:ACQUIRE:SP", "ES1:DET:MCA:01:STATUS:ACQUIRE"


@pytest.fixture
def _armed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The login's switch on for the simulator target, with a write tier that allows."""
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)


@pytest.fixture
def view(built_demo: BuiltProject, tmp_path: Path) -> Path:
    """A copy of the demo's simulator view, its render ticking every :data:`TICK_S`."""
    render = tmp_path / "build"
    copied = render / "data" / "simulator"
    shutil.copytree(built_demo.build_dir / "data" / "simulator", copied)
    (render / "config.yml").write_text(f"simulation:\n  tick_s: {TICK_S}\n")
    return copied


def _status(built: BuiltProject, address: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = [c for c in built.facility["channels"] if c["id"] == address]
    (record,) = records
    return record


class _Act:
    """A subscription to a status and one write to its command, timed on the monotonic clock."""

    def __init__(self, connector: VAInProcessConnector, status: dict[str, Any]) -> None:
        self.connector = connector
        self.options: list[str] = status["options"]
        self.idle: str = status["idle"]
        self.responds: dict[str, Any] = status["simulation"]["responds"]
        self.seen: list[tuple[Any, float]] = []
        self.t0 = 0.0
        self.at_return: list[Any] = []

    def labels(self, values: list[Any]) -> list[str]:
        """The labels of the status's values, which the simulator serves by index."""
        return [self.options[value] for value in values]

    def after_write(self) -> list[str]:
        """The labels a subscriber saw after the write; the first callback is the value before."""
        return self.labels([value for value, _ in self.seen[1:]])


async def _act(view: Path, built: BuiltProject, command: str, status_address: str) -> _Act:
    """Subscribe to the status, write ``TRUE`` to its command and wait for the idle edge.

    The wait's deadline is the edge's upper bound: ``for_s`` plus one tick plus
    :data:`LATE_S` after the write.
    """
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view, response_delay_ms=0))
    act = _Act(connector, _status(built, status_address))
    await connector.subscribe(
        status_address, lambda reading: act.seen.append((reading.value, time.monotonic()))
    )
    await async_wait_until(
        lambda: len(act.seen) >= 1, what=f"the current value of {status_address}"
    )

    act.t0 = time.monotonic()
    await connector.write_channel(command, "TRUE")
    act.at_return = [value for value, _ in act.seen[1:]]
    await async_wait_until(
        lambda: len(act.seen) >= 3,
        timeout=act.responds["for_s"] + TICK_S + LATE_S,
        what=f"the idle edge of {status_address}",
    )
    return act


async def _assert_both_edges(act: _Act, act_label: str) -> None:
    """The subscriber saw ``act_label`` by the write's return, then ``idle`` in its window."""
    try:
        assert act.after_write() == [act_label, act.idle]
        assert act.labels(act.at_return) == [act_label]
        idle_at = act.seen[2][1]
        for_s = act.responds["for_s"]
        assert act.t0 + for_s <= idle_at <= act.t0 + for_s + TICK_S + LATE_S
    finally:
        await act.connector.disconnect()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_a_subscriber_sees_the_detector_busy_then_idle(
    view: Path, built_demo: BuiltProject
) -> None:
    await _assert_both_edges(await _act(view, built_demo, ACQUIRE, ACQUIRE_STATUS), "TRUE")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_a_subscriber_sees_the_stage_moving_then_done(
    view: Path, built_demo: BuiltProject
) -> None:
    await _assert_both_edges(await _act(view, built_demo, MOVE, MOVE_STATUS), "MOVING")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_armed")
async def test_a_read_after_the_idle_edge_reads_idle(view: Path, built_demo: BuiltProject) -> None:
    act = await _act(view, built_demo, ACQUIRE, ACQUIRE_STATUS)
    try:
        reading = await act.connector.read_channel(ACQUIRE_STATUS)
        assert act.labels([reading.value]) == [act.idle]
    finally:
        await act.connector.disconnect()
