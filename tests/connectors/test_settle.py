"""One settle step: a written setpoint's readback must come within the bound its facility states.

The bound is the setpoint's ``tolerance`` (``absolute``, or ``relative`` times
the demand's magnitude); with none stated, on the simulator only, the
readback's motion band; a caller's floor raises it, never lowers it. The poll
runs on a fake connector with a fake clock, so a deadline passes without
waiting for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import osprey_connectors.settle
from osprey_connectors.errors import ChannelReadFailedError
from osprey_connectors.settle import (
    POLL_INTERVAL_S,
    Landing,
    SettleReadError,
    SettleTimeoutError,
    declared_tolerance,
    settle,
    settle_bound,
)

# --- the bound -------------------------------------------------------------------


def test_an_absolute_tolerance_is_its_magnitude() -> None:
    assert declared_tolerance({"absolute": -0.05}, 100.0) == pytest.approx(0.05)


def test_a_relative_tolerance_is_taken_of_the_demand() -> None:
    assert declared_tolerance({"relative": -1e-3}, -500.0) == pytest.approx(0.5)
    assert declared_tolerance({"relative": 1e-3}, 2.0) == pytest.approx(2e-3)


@pytest.mark.parametrize("record", [None, {}, {"other": 1.0}])
def test_no_tolerance_record_declares_nothing(record: Any) -> None:
    assert declared_tolerance(record, 1.0) is None


def test_the_motion_band_applies_only_on_the_simulator() -> None:
    assert settle_bound(None, 1.0, motion=0.02, simulated=True) == pytest.approx(0.02)
    assert settle_bound(None, 1.0, motion=0.02, simulated=False) is None


def test_a_stated_tolerance_wins_over_the_motion_band() -> None:
    bound = settle_bound({"absolute": 0.1}, 1.0, motion=0.5, simulated=True)
    assert bound == pytest.approx(0.1)


def test_the_floor_is_a_floor_never_a_ceiling() -> None:
    assert settle_bound({"absolute": 0.1}, 1.0, floor=0.01) == pytest.approx(0.1)
    assert settle_bound({"absolute": 0.001}, 1.0, floor=0.01) == pytest.approx(0.01)
    assert settle_bound(None, 1.0, floor=1e-9) == pytest.approx(1e-9)


def test_nothing_declared_and_no_floor_is_no_bound() -> None:
    assert settle_bound(None, 1.0) is None
    assert settle_bound(None, 1.0, motion=0.0, simulated=True) is None


# --- the poll ----------------------------------------------------------------------


@dataclass
class _Reading:
    value: Any


class _Clock:
    """A clock that moves only when the poll pauses; every pause is recorded."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds


class _Connector:
    """Readbacks that reach their target after ``lag`` reads each; every read is logged."""

    def __init__(self, start: dict[str, Any], target: dict[str, Any], lag: int) -> None:
        self.start = start
        self.target = target
        self.lag = lag
        self.reads: list[str] = []

    async def read_channel(self, address: str) -> _Reading:
        self.reads.append(address)
        arrived = self.reads.count(address) > self.lag
        return _Reading((self.target if arrived else self.start)[address])


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


async def test_a_lagging_readback_settles_after_its_lag(clock: _Clock) -> None:
    connector = _Connector({"Q1:RB": 1.0}, {"Q1:RB": 2.04}, lag=3)

    await settle(
        connector,
        [Landing("Q1:SP", "Q1:RB", 2.0, 0.05)],
        timeout_s=5.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert connector.reads == ["Q1:RB"] * 4
    assert clock.pauses == [POLL_INTERVAL_S] * 3


async def test_a_batch_settles_each_landing_on_its_own(clock: _Clock) -> None:
    connector = _Connector({"Q1:RB": 2.0, "Q2:RB": 1.0}, {"Q1:RB": 2.0, "Q2:RB": 3.0}, lag=2)
    landings = [Landing("Q1:SP", "Q1:RB", 2.0, 0.05), Landing("Q2:SP", "Q2:RB", 3.0, 0.05)]

    await settle(connector, landings, timeout_s=5.0, clock=clock, sleep=clock.sleep)

    assert connector.reads == ["Q1:RB", "Q2:RB", "Q2:RB", "Q2:RB"], (
        "a settled landing is not read again"
    )


@pytest.mark.parametrize("reading", [float("nan"), "2.0"], ids=["nan", "string"])
async def test_a_reading_that_is_no_finite_number_never_settles(
    clock: _Clock, reading: Any
) -> None:
    connector = _Connector({"Q1:RB": reading}, {"Q1:RB": reading}, lag=0)

    with pytest.raises(SettleTimeoutError) as info:
        await settle(
            connector,
            [Landing("Q1:SP", "Q1:RB", 2.0, 0.05)],
            timeout_s=0.2,
            clock=clock,
            sleep=clock.sleep,
        )

    assert f"(last read: {reading})" in str(info.value)


async def test_a_readback_that_answers_no_value_is_a_settle_read_error(clock: _Clock) -> None:
    connector = _Connector({"Q1:RB": None}, {"Q1:RB": None}, lag=0)
    landing = Landing("Q1:SP", "Q1:RB", 2.0, 0.05)

    with pytest.raises(SettleReadError) as info:
        await settle(connector, [landing], timeout_s=5.0, clock=clock, sleep=clock.sleep)

    error = info.value
    assert error.landing == landing
    assert error.addresses == ["Q1:RB"]
    assert error.causes == {}
    assert "no value" in str(error)
    assert clock.pauses == []


async def test_a_readback_read_that_raises_is_a_settle_read_error(clock: _Clock) -> None:
    down = ConnectionError("down")

    class _Dead:
        async def read_channel(self, address: str) -> _Reading:
            del address
            raise down

    landing = Landing("Q1:SP", "Q1:RB", 2.0, 0.05)

    with pytest.raises(SettleReadError) as info:
        await settle(_Dead(), [landing], timeout_s=5.0, clock=clock, sleep=clock.sleep)

    error = info.value
    assert isinstance(error, ChannelReadFailedError)
    assert error.landing == landing
    assert error.addresses == ["Q1:RB"]
    assert error.causes["Q1:RB"] is down
    assert error.__cause__ is down
    assert "Q1:RB" in str(error) and "Q1:SP" in str(error)
    assert clock.pauses == []


def test_settle_read_error_is_part_of_the_api() -> None:
    assert "SettleReadError" in osprey_connectors.settle.__all__


async def test_a_timeout_names_the_readback_the_demand_the_bound_and_the_last_reading(
    clock: _Clock,
) -> None:
    connector = _Connector({"Q1:RB": 2.0, "Q2:RB": 1.5}, {"Q1:RB": 2.0, "Q2:RB": 1.5}, lag=0)
    landings = [Landing("Q1:SP", "Q1:RB", 2.0, 0.05), Landing("Q2:SP", "Q2:RB", 3.0, 0.05)]

    with pytest.raises(SettleTimeoutError) as info:
        await settle(connector, landings, timeout_s=1.0, clock=clock, sleep=clock.sleep)

    error = info.value
    assert isinstance(error, TimeoutError)
    assert error.landing == landings[1]
    assert error.last == 1.5
    assert str(error) == "readback Q2:RB did not settle to 3.0 ± 0.05 within 1s (last read: 1.5)"
    assert clock.now >= 1.0


async def test_a_zero_timeout_raises_without_sleeping(clock: _Clock) -> None:
    connector = _Connector({"Q1:RB": 1.0}, {"Q1:RB": 1.0}, lag=0)

    with pytest.raises(SettleTimeoutError):
        await settle(
            connector,
            [Landing("Q1:SP", "Q1:RB", 2.0, 0.05)],
            timeout_s=0.0,
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.pauses == []


async def test_no_landings_reads_nothing(clock: _Clock) -> None:
    connector = _Connector({}, {}, lag=0)

    await settle(connector, [], timeout_s=5.0, clock=clock, sleep=clock.sleep)

    assert connector.reads == []
