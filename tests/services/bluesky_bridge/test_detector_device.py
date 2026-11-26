"""A detector a plan triggers and records, mediated by the OSPREY connector.

``ConnectorDetector`` fires its acquire command through
``osprey_connectors.command.fire`` and waits for the status to leave and
return to idle; ``read()``/``describe()`` cover every result, one data key per
result, named by its address. It is never set or stopped, and a pause cancels
an unfinished trigger, so nothing is written after an abort. Driven against the
fake connector, whose call log is what these tests assert on.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey.services.bluesky_bridge.devices.connector import (
    PRESS_VALUE,
    ConnectorDetector,
    ConnectorTriggerable,
    build_devices,
)
from osprey.services.bluesky_bridge.devices.specs import DetectorSpec, ReadableSpec
from osprey.services.bluesky_bridge.plan_writes import ChannelWrite
from osprey_connectors.command import CommandBusyError, CommandNoEffectError, CommandTimeoutError
from tests.services.bluesky_bridge._fake_connector import FakeConnector

ACQUIRE, STATUS = "ES1:MCA01:ACQ", "ES1:MCA01:ACQ_STATUS"
COUNTS, SPECTRUM = "ES1:MCA01:COUNTS", "ES1:MCA01:SPECTRUM"
RESULTS = (
    ReadableSpec(COUNTS, COUNTS, unit="counts", value_type="float"),
    ReadableSpec(SPECTRUM, SPECTRUM, shape=(4,), value_type="waveform"),
)

#: The ``timeout_s`` the acquire command states: longer than any act these tests complete.
ACT_TIMEOUT_S = 5.0
#: The ``timeout_s`` stated where the act is meant to run out: the device's own
#: bound ends it, and the test waits on the error.
SHORT_ACT_TIMEOUT_S = ACT_TIMEOUT_S / 100


def _detector(fake: FakeConnector, timeout_s: float = ACT_TIMEOUT_S) -> ConnectorDetector:
    return ConnectorDetector(
        fake, ACQUIRE, STATUS, idle=0, timeout_s=timeout_s, results=RESULTS, name="ES1-MCA01"
    )


def _acquiring(*updates: Any) -> FakeConnector:
    """A connector whose status moves through ``updates`` once the command is written."""
    fake = FakeConnector(readbacks={STATUS: 0, COUNTS: 7.0, SPECTRUM: [1.0, 2.0, 3.0, 4.0]})
    fake.responses[ACQUIRE] = (STATUS, list(updates))
    return fake


async def test_trigger_fires_the_acquire_command_once_and_waits_for_the_status() -> None:
    fake = _acquiring(1, 0)
    device = _detector(fake)

    status = device.trigger()
    await status

    assert status.success
    assert fake.write_calls == [(ACQUIRE, PRESS_VALUE, {"confirm": False})]
    assert fake.subscribers == {}
    assert isinstance(device, ConnectorTriggerable)
    assert device.trigger_writes() == (ChannelWrite(ACQUIRE, PRESS_VALUE),)


async def test_a_busy_status_refuses_the_trigger_and_writes_nothing() -> None:
    fake = _acquiring(1, 0)
    fake.readbacks[STATUS] = 1

    with pytest.raises(CommandBusyError):
        await _detector(fake).trigger()

    assert fake.write_calls == []


async def test_a_status_that_never_leaves_idle_fails_the_trigger() -> None:
    fake = _acquiring()

    with pytest.raises(CommandNoEffectError):
        await _detector(fake, timeout_s=SHORT_ACT_TIMEOUT_S).trigger()

    assert [address for address, _value, _kwargs in fake.write_calls] == [ACQUIRE]


async def test_a_status_that_never_returns_fails_the_trigger_after_its_bound() -> None:
    fake = _acquiring(1)

    with pytest.raises(CommandTimeoutError):
        await _detector(fake, timeout_s=SHORT_ACT_TIMEOUT_S).trigger()

    assert [address for address, _value, _kwargs in fake.write_calls] == [ACQUIRE]
    assert fake.subscribers == {}


async def test_read_reads_every_result_fresh_under_its_address() -> None:
    fake = _acquiring()
    device = _detector(fake)

    first = await device.read()
    fake.readbacks[COUNTS] = 9.0
    second = await device.read()

    assert list(first) == [COUNTS, SPECTRUM]
    assert (first[COUNTS]["value"], second[COUNTS]["value"]) == (7.0, 9.0)
    assert second[SPECTRUM]["value"] == [1.0, 2.0, 3.0, 4.0]
    assert fake.read_calls == [COUNTS, SPECTRUM, COUNTS, SPECTRUM]


async def test_describe_states_each_result_with_its_unit_and_shape_and_reads_nothing() -> None:
    fake = _acquiring()
    device = _detector(fake)

    described = await device.describe()

    assert described == {
        COUNTS: {
            "source": f"connector:{COUNTS}",
            "dtype": "number",
            "shape": [],
            "units": "counts",
        },
        SPECTRUM: {"source": f"connector:{SPECTRUM}", "dtype": "array", "shape": [4]},
    }
    assert fake.read_calls == []
    assert device.hints == {"fields": [COUNTS]}


async def test_a_word_result_is_recorded_as_its_word_and_never_hinted() -> None:
    state = "ES1:MCA01:STATE"
    fake = _acquiring()
    fake.readbacks[state] = 1
    device = ConnectorDetector(
        fake,
        ACQUIRE,
        STATUS,
        idle=0,
        timeout_s=ACT_TIMEOUT_S,
        results=(*RESULTS, ReadableSpec(state, state, value_type="enum", options=("OK", "LOW"))),
        name="ES1-MCA01",
    )

    reading = await device.read()
    described = await device.describe()

    assert reading[state]["value"] == "LOW"
    assert (described[state]["dtype"], described[state]["choices"]) == ("string", ["OK", "LOW"])
    assert device.hints == {"fields": [COUNTS]}


async def test_pause_before_the_write_cancels_the_trigger_and_nothing_is_written() -> None:
    fake = _acquiring(1, 0)
    fake.hold_reads.add(STATUS)
    fake.read_held_here = asyncio.Event()
    device = _detector(fake)
    status = device.trigger()
    await fake.read_held_here.wait()

    device.pause()

    with pytest.raises(asyncio.CancelledError):
        await status
    assert fake.write_calls == []
    assert fake.subscribers == {}
    device.resume()
    assert fake.write_calls == []


class _Announcing(FakeConnector):
    """A fake that sets :attr:`wrote` once a command write has been answered."""

    def __init__(self, readbacks: dict[str, Any]) -> None:
        super().__init__(readbacks)
        self.wrote = asyncio.Event()

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        result = await super().write_channel_checked(channel_address, value, **kwargs)
        self.wrote.set()
        return result


async def test_pause_after_the_write_ends_the_wait_and_writes_nothing_more() -> None:
    fake = _Announcing({STATUS: 0})
    fake.responses[ACQUIRE] = (STATUS, [1])
    device = _detector(fake)
    status = device.trigger()
    await fake.wrote.wait()

    device.pause()

    with pytest.raises(asyncio.CancelledError):
        await status
    assert [address for address, _value, _kwargs in fake.write_calls] == [ACQUIRE]
    assert fake.subscribers == {}


def test_a_detector_has_no_set_and_no_stop() -> None:
    device = _detector(FakeConnector())

    assert not hasattr(device, "set")
    assert not hasattr(device, "stop")
    assert not hasattr(device, "locate")


async def test_build_devices_builds_a_detector_from_its_spec() -> None:
    spec = DetectorSpec("ES1-MCA01", ACQUIRE, STATUS, 0, ACT_TIMEOUT_S, results=RESULTS)

    devices = await build_devices(connector=FakeConnector(), detectors=[spec])

    assert isinstance(devices["ES1-MCA01"], ConnectorDetector)
    assert devices["ES1-MCA01"].channel_addresses() == (ACQUIRE, STATUS, COUNTS, SPECTRUM)
