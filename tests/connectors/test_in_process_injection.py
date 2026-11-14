"""The in-process simulator serving the injection fixture holds the chain still.

The in-process connector builds its composite without evolution, so a write is
a held value and nothing else: a request write is confirmed and reads back,
and every other channel of the chain keeps the value it read before the write,
both at once and after the clock has moved past every step offset and a whole
cycle more. The clock is moved two ways: a real wait past the last offset, and
a fake clock the composite is built with, stepped far ahead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import osprey_connectors.simulation.composite as composite_module
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests.facility._injection_fixture import (
    OFFSETS,
    PARAMETERS,
    READINGS,
    REQUEST,
    SETPOINTS,
    injection_view,
)
from tests.facility.served_tree import in_process_config

#: The instant the fake clock starts at, epoch seconds.
T0 = 1_760_000_000.0

#: The chain's channels other than the request.
OTHERS = tuple(address for address in (*SETPOINTS, *READINGS) if address != REQUEST)


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


async def _connected(tmp_path: Path) -> VAInProcessConnector:
    connector = VAInProcessConnector()
    view = injection_view(tmp_path / "tree")
    await connector.connect(in_process_config(view, response_delay_ms=0))
    return connector


async def _read(connector: VAInProcessConnector, addresses) -> dict[str, float]:
    return {a: (await connector.read_channel(a)).value for a in addresses}


async def test_every_parameter_reads_its_start_value(tmp_path: Path) -> None:
    connector = await _connected(tmp_path)
    try:
        assert await _read(connector, PARAMETERS) == PARAMETERS
        assert await _read(connector, [REQUEST]) == {REQUEST: 0.0}
    finally:
        await connector.disconnect()


async def test_a_request_write_reads_back_and_moves_nothing_else(tmp_path: Path) -> None:
    connector = await _connected(tmp_path)
    try:
        before = await _read(connector, OTHERS)

        result = await connector.write_channel(REQUEST, 1.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert await _read(connector, [REQUEST]) == {REQUEST: 1.0}
        assert connector._composite.held([REQUEST])[REQUEST] == 1.0
        assert await _read(connector, OTHERS) == before

        await asyncio.sleep(max(OFFSETS.values()) + 0.05)
        assert await _read(connector, [REQUEST]) == {REQUEST: 1.0}
        assert await _read(connector, OTHERS) == before
    finally:
        await connector.disconnect()


async def test_nothing_moves_after_the_clock_passes_every_offset(
    tmp_path: Path, clock: _Clock
) -> None:
    connector = await _connected(tmp_path)
    try:
        before = await _read(connector, OTHERS)
        assert (await connector.write_channel(REQUEST, 1.0)).outcome is WriteOutcome.CONFIRMED

        for offset in sorted(OFFSETS.values()):
            clock.t_s = T0 + offset + 1e-3
            assert await _read(connector, OTHERS) == before
        clock.t_s = T0 + 60.0
        assert await _read(connector, [REQUEST]) == {REQUEST: 1.0}
        assert await _read(connector, OTHERS) == before
        assert connector._composite.held(list(OTHERS)) == before
    finally:
        await connector.disconnect()
