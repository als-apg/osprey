"""The in-process simulator asked to evolve runs its time-driven models, shots included.

``evolve: true`` in the connector's settings builds the composite to evolve:
a request write is a client write and starts a shot, whose steps land as the
clock passes their offsets; a ring fed by the chain tracks the shot for the
monitors a write armed. The writes journal is replayed without a client
write, so a journalled request starts no shot in the connector that replays
it, at its connect or when another connector moves the journal.

Every case but the subscription one runs on a fake clock the composite is
built with, moved by hand.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

import osprey_connectors.simulation.composite as composite_module
from osprey_connectors.control_system import _va_in_process_connector as in_process
from osprey_connectors.control_system._base import ChannelValue, WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests.facility import _turn_by_turn_fixture as ring
from tests.facility._injection_fixture import (
    COUNTERS,
    CURRENT,
    OFFSETS,
    PARAMETERS,
    REQUEST,
    injection_view,
)
from tests.facility.served_tree import in_process_config

#: The connector setting under test; a connectors release without it has no evolution to test.
EVOLVE_SETTING: str = getattr(in_process, "EVOLVE_SETTING", "")

pytestmark = pytest.mark.skipif(not EVOLVE_SETTING, reason="connectors floor has no evolve setting")

#: The instant the fake clock starts at, epoch seconds.
T0 = 1_760_000_000.0

#: Past every step offset of one shot, in seconds after its request.
SETTLED_S = max(OFFSETS.values()) + 0.01

#: The stored current one shot adds: charge 1.5 through 0.8 and 0.6, at 2.0 per charge.
SHOT_CURRENT = 1.5 * 0.8 * 0.6 * 2.0

#: A parameter another connector writes, and the value it writes.
PARAMETER = "INJ:GUN:CHARGE:SP"
PARAMETER_VALUE = 1.0


class _Clock:
    """A clock a test moves by hand."""

    def __init__(self) -> None:
        self.t_s = T0

    def __call__(self) -> float:
        return self.t_s


@pytest.fixture(autouse=True)
def _armed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Writes reach the simulator: the switch on and a write tier that allows."""
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """The fake clock every composite the connector builds is handed."""
    fake = _Clock()
    real = composite_module.Composite

    def built(*args, **kwargs):
        kwargs.setdefault("clock", fake)
        return real(*args, **kwargs)

    monkeypatch.setattr(composite_module, "Composite", built)
    return fake


async def _connected(view: Path, **settings: object) -> VAInProcessConnector:
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view, response_delay_ms=0, **settings))
    return connector


async def _read(connector: VAInProcessConnector, addresses) -> dict[str, object]:
    return {a: (await connector.read_channel(a)).value for a in addresses}


async def test_a_request_write_takes_a_shot(tmp_path: Path, clock: _Clock) -> None:
    connector = await _connected(injection_view(tmp_path / "tree"), **{EVOLVE_SETTING: True})
    try:
        stored = PARAMETERS["INJ:RING:CURRENT:SP"]
        assert await _read(connector, (*COUNTERS, CURRENT)) == {
            **dict.fromkeys(COUNTERS, 0.0),
            CURRENT: stored,
        }

        assert (await connector.write_channel(REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED
        clock.t_s = T0 + OFFSETS[COUNTERS[0]] + 1e-3
        assert await _read(connector, COUNTERS) == {COUNTERS[0]: 1.0, COUNTERS[1]: 0.0}

        clock.t_s = T0 + SETTLED_S
        read = await _read(connector, (*COUNTERS, CURRENT))
        assert {a: read[a] for a in COUNTERS} == dict.fromkeys(COUNTERS, 1.0)
        assert read[CURRENT] == pytest.approx(stored + SHOT_CURRENT)
    finally:
        await connector.disconnect()


async def test_without_the_setting_a_request_starts_nothing(tmp_path: Path, clock: _Clock) -> None:
    connector = await _connected(injection_view(tmp_path / "tree"))
    try:
        assert (await connector.write_channel(REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED
        clock.t_s = T0 + SETTLED_S
        assert await _read(connector, COUNTERS) == dict.fromkeys(COUNTERS, 0.0)
    finally:
        await connector.disconnect()


@pytest.mark.parametrize("setting", ["yes", 1, None])
async def test_a_setting_that_is_no_boolean_is_refused(tmp_path: Path, setting: object) -> None:
    connector = VAInProcessConnector()
    config = in_process_config(injection_view(tmp_path / "tree"), **{EVOLVE_SETTING: setting})

    with pytest.raises(ValueError) as raised:
        await connector.connect(config)

    assert str(raised.value) == f"evolve is {setting!r}; use true or false"


async def test_a_connector_that_replays_the_request_at_connect_takes_no_shot(
    tmp_path: Path, clock: _Clock
) -> None:
    view = injection_view(tmp_path / "tree")
    first = await _connected(view, **{EVOLVE_SETTING: True})
    second: VAInProcessConnector | None = None
    try:
        assert (await first.write_channel(REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED
        clock.t_s = T0 + SETTLED_S
        assert await _read(first, COUNTERS) == dict.fromkeys(COUNTERS, 1.0)

        second = await _connected(view, **{EVOLVE_SETTING: True})
        assert await _read(second, [REQUEST]) == {REQUEST: 1.0}
        clock.t_s = T0 + 2 * SETTLED_S + 60.0
        assert await _read(second, COUNTERS) == dict.fromkeys(COUNTERS, 0.0)
        assert await _read(first, COUNTERS) == dict.fromkeys(COUNTERS, 1.0)
    finally:
        await first.disconnect()
        if second is not None:
            await second.disconnect()


async def test_a_replay_another_connectors_write_causes_fires_no_second_shot(
    tmp_path: Path, clock: _Clock
) -> None:
    view = injection_view(tmp_path / "tree")
    first = await _connected(view, **{EVOLVE_SETTING: True})
    second = await _connected(view, **{EVOLVE_SETTING: True})
    try:
        assert (await first.write_channel(REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED
        clock.t_s = T0 + SETTLED_S
        assert await _read(first, COUNTERS) == dict.fromkeys(COUNTERS, 1.0)

        # The second connector's write moves the journal: the first replays it,
        # the request it wrote itself included.
        written = await second.write_channel(PARAMETER, PARAMETER_VALUE)
        assert written.outcome is WriteOutcome.CONFIRMED
        assert await _read(first, [REQUEST, PARAMETER]) == {
            REQUEST: 1.0,
            PARAMETER: PARAMETER_VALUE,
        }
        clock.t_s = T0 + 2 * SETTLED_S + 60.0
        # The replay started the machine again and fired nothing: no counter
        # reads the first shot twice, and none steps as the clock moves on.
        assert await _read(first, COUNTERS) == dict.fromkeys(COUNTERS, 0.0)
        assert await _read(second, COUNTERS) == dict.fromkeys(COUNTERS, 0.0)
    finally:
        await first.disconnect()
        await second.disconnect()


async def test_a_shot_fills_the_armed_monitors_turn_arrays(tmp_path: Path, clock: _Clock) -> None:
    connector = await _connected(
        ring.turn_by_turn_view(tmp_path / "tree"), **{EVOLVE_SETTING: True}
    )
    monitor, idle = ring.MONITORS[0], ring.MONITORS[1]
    try:
        assert np.count_nonzero((await connector.read_channel(ring.TURN_X[monitor])).value) == 0

        assert (await connector.write_channel(ring.RF_SWITCH, 1)).outcome is WriteOutcome.CONFIRMED
        armed = await connector.write_channel(ring.ARMS[monitor], 1)
        assert armed.outcome is WriteOutcome.CONFIRMED
        assert (await connector.read_channel(ring.ARMED[monitor])).value == 1
        assert (await connector.write_channel(ring.REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED

        # A client times the shot on ``armed`` alone: it drops once the arrays are served.
        for step in range(1, 50):
            clock.t_s = T0 + 0.05 * step
            if (await connector.read_channel(ring.ARMED[monitor])).value == 0:
                break
        else:
            pytest.fail("the armed monitor never dropped")

        x = (await connector.read_channel(ring.TURN_X[monitor])).value
        total = (await connector.read_channel(ring.TURN_SUM[monitor])).value
        assert x.shape == total.shape == (ring.SAMPLES,)
        assert np.count_nonzero(x) > 0
        assert np.count_nonzero(total) > 0
        assert (await connector.read_channel(ring.TURNS_TRACKED)).value == ring.TURNS
        assert np.count_nonzero((await connector.read_channel(ring.TURN_X[idle])).value) == 0
    finally:
        await connector.disconnect()


async def test_a_subscription_fires_when_a_step_lands_between_ticks(tmp_path: Path) -> None:
    view = injection_view(tmp_path / "tree")
    # A tick far longer than the test waits: only a deadline of the chain can wake the loop.
    (view.parent.parent / "config.yml").write_text("simulation:\n  tick_s: 600\n", "utf-8")
    connector = await _connected(view, **{EVOLVE_SETTING: True})
    counted: asyncio.Queue[float] = asyncio.Queue()

    def on_count(reading: ChannelValue) -> None:
        counted.put_nowait(float(reading.value))

    try:
        await connector.subscribe(COUNTERS[-1], on_count)
        # A subscription is given the channel's current value before the write.
        assert counted.get_nowait() == 0.0
        assert (await connector.write_channel(REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED

        assert await asyncio.wait_for(counted.get(), timeout=30.0) == 1.0
    finally:
        await connector.disconnect()
