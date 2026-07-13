"""Unit tests for the connector-mediated ophyd-async device layer (task 1.1).

Runs ONLY in a bluesky-capable environment — `bluesky`/`ophyd-async` are
never installed in the main worktree venv, so every test here is skipped via
`pytest.importorskip` rather than failing, keeping `ci_check` green with no
bluesky installed at all. To actually run this file:

    uv venv /tmp/bluesky-connector-scratch
    /tmp/bluesky-connector-scratch/bin/pip install -e '.[bluesky-bridge]' --python 3.11
    /tmp/bluesky-connector-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_connector_devices.py -q

Exercises `ConnectorSettable`/`ConnectorReadable`/`build_devices` against a
FAKE OSPREY connector (no Channel Access, no Docker, no real IOC): a small
async stand-in exposing `read_channel`/`write_channel_checked`, mirroring
`test_orm_plan_integration.py`'s idiom of driving real bluesky machinery
against a lightweight double rather than a real control system. The whole
point of this device layer is that every read and write is mediated by the
connector, so the fake's call log is itself part of what these tests assert
on.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.errors import ChannelWriteBlockedError, ChannelWriteFailedError  # noqa: E402
from osprey.services.bluesky_bridge.devices import connector as connector_module  # noqa: E402
from osprey.services.bluesky_bridge.devices.connector import (  # noqa: E402
    ConnectorReadable,
    ConnectorSettable,
    build_devices,
)
from osprey.services.bluesky_bridge.devices.specs import ReadableSpec, SettableSpec  # noqa: E402


@dataclass
class _FakeChannelValue:
    """Stand-in for ``osprey.connectors.control_system.base.ChannelValue``."""

    value: Any
    timestamp: float = field(default_factory=time.time)


class FakeConnector:
    """A minimal async double for ``ControlSystemConnector``.

    ``read_channel`` returns whatever ``readbacks[address]`` currently holds
    (a plain mutable dict, so a test can move the simulated readback after
    the fact). ``write_channel_checked`` either records the call and updates
    the readback (default), or raises/behaves however the test configures it
    via ``write_side_effect``/``echo_readback``.
    """

    def __init__(self, readbacks: dict[str, float] | None = None) -> None:
        self.readbacks: dict[str, float] = dict(readbacks or {})
        self.write_calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.read_calls: list[str] = []
        self.write_side_effect: Exception | None = None
        self.echo_readback = True
        # Maps a written channel address to the (possibly separate) readback
        # address it should echo into. Defaults to echoing into the same
        # address that was written.
        self.echo_target: dict[str, str] = {}

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        self.read_calls.append(channel_address)
        return _FakeChannelValue(value=self.readbacks.get(channel_address))

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        self.write_calls.append((channel_address, value, kwargs))
        if self.write_side_effect is not None:
            raise self.write_side_effect
        if self.echo_readback:
            target = self.echo_target.get(channel_address, channel_address)
            self.readbacks[target] = value
        return None


async def test_set_raises_on_blocked_write() -> None:
    """A refused write (writes-disabled/limits) must abort via a raise, not a swallow."""
    fake = FakeConnector(readbacks={"SP:RB": 0.0})
    fake.write_side_effect = ChannelWriteBlockedError("SP:RB", "LIMITS")
    device = ConnectorSettable(fake, "SP:RB", name="motor")

    with pytest.raises(ChannelWriteBlockedError):
        await device.set(5.0)


async def test_set_raises_on_failed_write() -> None:
    """An attempted-but-failed/unverified write must also abort via a raise."""
    fake = FakeConnector(readbacks={"SP:RB": 0.0})
    fake.write_side_effect = ChannelWriteFailedError("SP:RB", "CAPUT_FAILED")
    device = ConnectorSettable(fake, "SP:RB", name="motor")

    with pytest.raises(ChannelWriteFailedError):
        await device.set(5.0)


async def test_set_times_out_when_readback_never_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the readback never echoes the demand, ``set()`` must raise, never hang."""
    monkeypatch.setattr(connector_module, "_READBACK_SETTLE_TIMEOUT_S", 0.1)

    fake = FakeConnector(readbacks={"SP": 0.0})
    fake.echo_readback = False  # write succeeds, but readback never moves
    device = ConnectorSettable(fake, "SP", name="motor")

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await device.set(5.0)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, "set() must not hang past the (monkeypatched) settle timeout"


async def test_set_succeeds_when_write_verified_and_readback_echoes() -> None:
    """A verified write whose readback echoes within deadband must succeed silently."""
    fake = FakeConnector(readbacks={"SP": 0.0})
    device = ConnectorSettable(fake, "SP", name="motor")

    await device.set(3.5)  # must not raise

    assert fake.write_calls == [("SP", 3.5, {"verification_level": "callback"})]
    assert fake.readbacks["SP"] == 3.5


async def test_set_uses_separate_readback_pv_when_given() -> None:
    """Setpoint and readback are separate channels; settle-polling reads the readback PV."""
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.echo_target["SP"] = "RB"
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="motor")

    await device.set(2.0)

    assert fake.write_calls[0][0] == "SP"
    assert "RB" in fake.read_calls
    assert fake.readbacks["RB"] == 2.0


async def test_connector_readable_read_returns_live_connector_value() -> None:
    """``ConnectorReadable.read()`` must reflect the fake's live value, not a cached one."""
    fake = FakeConnector(readbacks={"BPM:1": 42.0})
    device = ConnectorReadable(fake, "BPM:1", name="bpm1")

    reading = await device.read()

    assert reading["bpm1"]["value"] == 42.0
    assert "BPM:1" in fake.read_calls

    # A live re-read after the underlying value changes must pick up the new value.
    fake.readbacks["BPM:1"] = 99.0
    reading2 = await device.read()
    assert reading2["bpm1"]["value"] == 99.0


async def test_connector_readable_describe_shape() -> None:
    """``describe()`` returns a valid scalar-numeric DataKey keyed on the device name."""
    fake = FakeConnector(readbacks={"BPM:1": 1.0})
    device = ConnectorReadable(fake, "BPM:1", name="bpm1")

    described = await device.describe()

    assert described == {
        "bpm1": {"source": "connector:BPM:1", "dtype": "number", "shape": []}
    }


async def test_build_devices_returns_expected_names_and_types() -> None:
    """``build_devices`` builds one device per spec, keyed by name, of the right type."""
    fake = FakeConnector(readbacks={"HCM1:SP": 0.0, "BPM1:I": 0.0})

    devices = await build_devices(
        settables=[SettableSpec(name="hcm1", setpoint_pv="HCM1:SP")],
        readables=[ReadableSpec(name="bpm1", read_pv="BPM1:I")],
        connector=fake,
    )

    assert set(devices) == {"hcm1", "bpm1"}
    assert isinstance(devices["hcm1"], ConnectorSettable)
    assert isinstance(devices["bpm1"], ConnectorReadable)


async def test_build_devices_with_no_specs_returns_empty_mapping() -> None:
    """An empty settables/readables sequence builds an empty (but valid) device set."""
    devices = await build_devices(connector=FakeConnector())
    assert devices == {}


def test_module_does_not_import_raw_channel_access() -> None:
    """The whole point of this module is zero direct-CA paths — verify by source scan."""
    source = inspect.getsource(connector_module)
    for forbidden in ("aioca", "epics_signal_r", "epics_signal_rw", "ophyd_async.epics"):
        assert forbidden not in source, f"found forbidden direct-CA reference: {forbidden!r}"


async def test_set_is_async_status_wrapped() -> None:
    """``set()`` must return an ``AsyncStatus`` (bluesky's ``Movable`` contract)."""
    from ophyd_async.core import AsyncStatus

    fake = FakeConnector(readbacks={"SP": 0.0})
    device = ConnectorSettable(fake, "SP", name="motor")

    status = device.set(1.0)
    assert isinstance(status, AsyncStatus)
    await status
    assert fake.readbacks["SP"] == 1.0
