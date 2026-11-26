"""Unit tests for the connector-mediated ophyd-async device layer (task 1.1).

The bluesky stack is a core dependency, so these run in the normal unit lane.

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

import asyncio
import inspect
import time
from typing import Any

import pytest

from osprey.services.bluesky_bridge.devices import connector as connector_module
from osprey.services.bluesky_bridge.devices.connector import (
    PRESS_VALUE,
    ConnectorPositioner,
    ConnectorReadable,
    ConnectorSettable,
    ConnectorTriggerable,
    build_devices,
)
from osprey.services.bluesky_bridge.devices.specs import (
    ReadableSpec,
    SettableSpec,
    TriggerableSpec,
)
from osprey.services.bluesky_bridge.plan_writes import ChannelWrite
from osprey_connectors.command import CommandAct, CommandBusyError, CommandNoEffectError
from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError
from tests.services.bluesky_bridge._fake_connector import (
    FakeChannelValue,
    FakeConnector,
)


@pytest.mark.parametrize(
    "exc",
    [
        # A refused write (writes-disabled/limits).
        pytest.param(ChannelWriteBlockedError("SP:RB", "LIMITS"), id="blocked-write"),
        # A write the control system did not take.
        pytest.param(ChannelWriteFailedError("SP:RB", "FAILED"), id="failed-write"),
    ],
)
async def test_set_raises_on_a_refused_write(exc: Exception) -> None:
    """A write that did not happen must abort via a raise, not a swallow."""
    fake = FakeConnector(readbacks={"SP:RB": 0.0})
    fake.write_side_effect = exc
    device = ConnectorSettable(fake, "SP:RB", name="motor")

    with pytest.raises(type(exc)):
        await device.set(5.0)


@pytest.mark.parametrize("reason", ["MISMATCH", "UNCONFIRMED"])
async def test_set_aborts_when_the_write_was_not_confirmed(reason: str) -> None:
    """A mismatched or unconfirmed write is a false premise for the rest of the
    scan, so ``write_channel_checked``'s raise must propagate out of ``set()``."""
    fake = FakeConnector(readbacks={"SP:RB": 0.0})
    fake.write_side_effect = ChannelWriteFailedError("SP:RB", reason)
    device = ConnectorSettable(fake, "SP:RB", name="motor")

    with pytest.raises(ChannelWriteFailedError) as excinfo:
        await device.set(5.0)

    assert excinfo.value.reason == reason
    assert fake.read_calls == [], "a raised write must never fall through to the poll loop"


def test_settle_budgets_default_when_the_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing in the environment the module's own defaults are in force."""
    monkeypatch.delenv(connector_module.SETTLE_TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)

    assert connector_module.settle_timeout_s() == connector_module.DEFAULT_SETTLE_TIMEOUT_S
    assert connector_module.settle_tolerance() == connector_module.DEFAULT_SETTLE_TOLERANCE


def test_settle_budgets_read_the_env_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value in force is whatever the environment says now, never at import."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "30")
    monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, "1e-3")

    assert connector_module.settle_timeout_s() == 30.0
    assert connector_module.settle_tolerance() == 1e-3

    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "45")
    assert connector_module.settle_timeout_s() == 45.0


@pytest.mark.parametrize("raw", ["", "abc", "-1", "0"])
def test_settle_timeout_refuses_a_budget_that_is_not_positive(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A zero or unreadable budget is a misconfiguration, not a policy."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, raw)

    with pytest.raises(ValueError, match=connector_module.SETTLE_TIMEOUT_ENV):
        connector_module.settle_timeout_s()


@pytest.mark.parametrize("raw", ["", "abc", "-1e-9"])
def test_settle_tolerance_refuses_a_value_below_zero(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A negative tolerance can never be met, so every move would time out."""
    monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, raw)

    with pytest.raises(ValueError, match=connector_module.SETTLE_TOLERANCE_ENV):
        connector_module.settle_tolerance()


def test_settle_tolerance_accepts_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact equality is a legitimate demand for a readback the IOC echoes."""
    monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, "0")

    assert connector_module.settle_tolerance() == 0.0


async def test_set_settles_within_an_authored_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readback that lands near the demand settles once the facility says so.

    At the default tolerance this same move polls until the budget runs out and
    raises — which is what a facility driving a device that physically moves is
    raising the key to avoid.
    """
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.3")
    monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, "0.01")

    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.echo_target["SP"] = "RB"
    fake.write_outcome = "unrequested"

    async def read_near_miss(channel_address: str, timeout: float | None = None):  # noqa: ARG001 - the connector read signature
        fake.read_calls.append(channel_address)
        return FakeChannelValue(value=5.0 - 0.005)

    monkeypatch.setattr(fake, "read_channel", read_near_miss)
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="motor")

    await device.set(5.0)  # must not raise

    assert fake.read_calls == ["RB"]


def _held_at(fake: FakeConnector, value: float) -> Any:
    """A readback read that always returns ``value``, logging the call."""

    async def read(channel_address: str, timeout: float | None = None):  # noqa: ARG001 - the connector read signature
        fake.read_calls.append(channel_address)
        return FakeChannelValue(value=value)

    return read


#: A device's declared band, wider than the profile floor.
_BAND = 0.011


async def test_set_settles_inside_the_device_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """A readback held at ``demand + 0.9 x band`` settles on its first poll."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.3")
    monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.write_outcome = "unrequested"
    monkeypatch.setattr(fake, "read_channel", _held_at(fake, -3.0 + 0.9 * _BAND))
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", settle_tolerance=_BAND)

    await device.set(-3.0)

    assert fake.read_calls == ["RB"]


async def test_set_times_out_outside_the_device_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """A readback held at ``demand + 1.1 x band`` never settles."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.1")
    monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.write_outcome = "unrequested"
    monkeypatch.setattr(fake, "read_channel", _held_at(fake, -3.0 + 1.1 * _BAND))
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", settle_tolerance=_BAND)

    with pytest.raises(TimeoutError, match="did not settle"):
        await device.set(-3.0)


async def test_a_device_without_a_band_settles_within_the_profile_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No band: the profile value alone decides, so a near miss times out."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.1")
    monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.write_outcome = "unrequested"
    monkeypatch.setattr(fake, "read_channel", _held_at(fake, -3.0 + 0.9 * _BAND))
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m")

    with pytest.raises(TimeoutError):
        await device.set(-3.0)


async def test_the_profile_floor_holds_under_a_tighter_device_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device band never tightens the profile value: the larger of the two governs."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.3")
    monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, "0.1")
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.write_outcome = "unrequested"
    monkeypatch.setattr(fake, "read_channel", _held_at(fake, -3.0 + 0.05))
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", settle_tolerance=_BAND)

    await device.set(-3.0)


async def test_build_devices_hands_each_settable_its_band(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.3")
    monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.write_outcome = "unrequested"
    monkeypatch.setattr(fake, "read_channel", _held_at(fake, -3.0 + 0.9 * _BAND))
    devices = await build_devices(
        settables=[
            SettableSpec(name="m", setpoint_pv="SP", readback_pv="RB", settle_tolerance=_BAND)
        ],
        connector=fake,
    )

    await devices["m"].set(-3.0)


def _unrequested(monkeypatch: pytest.MonkeyPatch, readback: float, timeout_s: str) -> FakeConnector:
    """A fake whose readback is held at ``readback``, its writes unchecked."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, timeout_s)
    monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    fake.write_outcome = "unrequested"
    monkeypatch.setattr(fake, "read_channel", _held_at(fake, readback))
    return fake


async def test_a_live_lane_ignores_the_motion_band(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _unrequested(monkeypatch, -3.0 + 0.9 * _BAND, "0.1")
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", motion_band=_BAND)

    with pytest.raises(TimeoutError, match="did not settle"):
        await device.set(-3.0)


async def test_a_simulated_lane_settles_within_the_motion_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _unrequested(monkeypatch, -3.0 + 0.9 * _BAND, "0.3")
    device = ConnectorSettable(
        fake, "SP", readback_pv="RB", name="m", motion_band=_BAND, simulated=True
    )

    await device.set(-3.0)

    assert fake.read_calls == ["RB"]


@pytest.mark.parametrize("simulated", [False, True], ids=["live", "simulated"])
async def test_a_declared_tolerance_applies_on_every_lane(
    monkeypatch: pytest.MonkeyPatch, simulated: bool
) -> None:
    fake = _unrequested(monkeypatch, -3.0 + 0.9 * _BAND, "0.3")
    device = ConnectorSettable(
        fake,
        "SP",
        readback_pv="RB",
        name="m",
        settle_tolerance=_BAND,
        motion_band=_BAND / 100,
        simulated=simulated,
    )

    await device.set(-3.0)


async def test_a_relative_tolerance_scales_with_the_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    near = _unrequested(monkeypatch, -200.0 + 0.019, "0.3")
    await ConnectorSettable(near, "SP", readback_pv="RB", name="m", settle_relative=1e-4).set(
        -200.0
    )

    far = _unrequested(monkeypatch, -2.0 + 0.019, "0.1")
    with pytest.raises(TimeoutError):
        await ConnectorSettable(far, "SP", readback_pv="RB", name="m", settle_relative=1e-4).set(
            -2.0
        )


async def test_the_floor_is_never_a_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _unrequested(monkeypatch, -3.0 + 0.05, "0.3")
    monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, "0.01")
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", settle_tolerance=0.1)

    await device.set(-3.0)


async def test_build_devices_hands_each_settable_its_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _unrequested(monkeypatch, -3.0 + 0.9 * _BAND, "0.3")
    spec = SettableSpec(name="m", setpoint_pv="SP", readback_pv="RB", motion_band=_BAND)

    simulated = await build_devices(settables=[spec], connector=fake, simulated=True)
    await simulated["m"].set(-3.0)

    live = await build_devices(
        settables=[spec], connector=_unrequested(monkeypatch, -3.0 + 0.9 * _BAND, "0.1")
    )
    with pytest.raises(TimeoutError):
        await live["m"].set(-3.0)


async def test_set_times_out_when_readback_never_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the readback never echoes the demand, ``set()`` must raise, never hang."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.1")

    fake = FakeConnector(readbacks={"SP": 0.0})
    fake.echo_readback = False  # write succeeds, but readback never moves
    fake.write_outcome = "unrequested"  # unchecked, so the poll loop is the only check
    device = ConnectorSettable(fake, "SP", name="motor")

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await device.set(5.0)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, "set() must not hang past the (monkeypatched) settle timeout"


async def test_set_writes_without_forwarding_a_confirm_kwarg() -> None:
    """The channel resolves its own confirm policy: the device states no opinion."""
    fake = FakeConnector(readbacks={"SP": 0.0})
    device = ConnectorSettable(fake, "SP", name="motor")

    await device.set(3.5)  # must not raise

    assert fake.write_calls == [("SP", 3.5, {})]
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


async def test_set_returns_on_a_confirmed_write_to_the_readback_channel() -> None:
    """A confirmed write to the readback channel itself needs no settle poll at all.

    The connector already re-read that exact channel and found the value sent, so
    polling it again could only disagree — the settle tolerance is stricter than
    the connector's comparison rule, so a confirmed write would time out here.
    """
    fake = FakeConnector(readbacks={"SP": 0.0})
    device = ConnectorSettable(fake, "SP", name="motor")

    await device.set(3.5)

    assert fake.read_calls == [], "a confirmed write to the readback channel is settled"


async def test_set_returns_on_a_confirmed_write_the_deadband_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``confirmed`` is the verdict — no arithmetic on ``observed_value`` here.

    The connector confirms within ``isclose(rel_tol=1e-6)``; re-checking that
    value against the 1e-9 absolute deadband would poll the setpoint just written
    for the full settle timeout and then raise on a write that in fact succeeded.
    """
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.1")
    fake = FakeConnector(readbacks={"SP": 0.0})
    fake.echo_readback = False
    fake.readbacks["SP"] = 3.5 + 1e-7  # confirmed by the connector, outside the deadband
    device = ConnectorSettable(fake, "SP", name="motor")

    await device.set(3.5)  # must not raise

    assert fake.read_calls == []


async def test_set_polls_when_the_write_was_unrequested() -> None:
    """A channel configured ``confirm: false`` checks nothing; the poll loop covers it."""
    fake = FakeConnector(readbacks={"SP": 0.0})
    fake.write_outcome = "unrequested"
    device = ConnectorSettable(fake, "SP", name="motor")

    await device.set(3.5)

    assert "SP" in fake.read_calls


async def test_set_polls_an_unrequested_write_until_the_readback_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchecked write settles on the poll loop, not on the result it returned."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.5")
    fake = FakeConnector(readbacks={"SP": 0.0})
    fake.write_outcome = "unrequested"
    fake.echo_readback = False  # the readback moves only under the poll loop
    device = ConnectorSettable(fake, "SP", name="motor")

    async def settle_after_first_poll(channel_address: str, timeout: float | None = None):  # noqa: ARG001 - the connector read signature
        fake.read_calls.append(channel_address)
        fake.readbacks["SP"] = 3.5
        return FakeChannelValue(value=fake.readbacks["SP"])

    fake.read_channel = settle_after_first_poll  # type: ignore[method-assign]

    await device.set(3.5)  # must not raise

    assert fake.read_calls == ["SP"]


async def test_set_never_takes_a_confirmed_setpoint_for_a_separate_readback_pv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmed setpoint write says nothing about a separate readback channel."""
    monkeypatch.setenv(connector_module.SETTLE_TIMEOUT_ENV, "0.1")
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    # The setpoint echoes (so the write confirms) but the readback never moves.
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="motor")

    with pytest.raises(TimeoutError):
        await device.set(2.0)

    assert fake.read_calls and set(fake.read_calls) == {"RB"}


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

    assert described == {"bpm1": {"source": "connector:BPM:1", "dtype": "number", "shape": []}}


async def test_a_readable_with_a_shape_describes_an_array() -> None:
    """A readable given the facility's length describes an array of that length."""
    fake = FakeConnector(readbacks={"W:1": [0.0] * 500})
    device = ConnectorReadable(fake, "W:1", name="w1", unit="A", shape=(500,))

    described = await device.describe()

    assert described == {
        "w1": {"source": "connector:W:1", "dtype": "array", "shape": [500], "units": "A"}
    }
    assert fake.read_calls == []


@pytest.mark.parametrize(
    ("value_type", "shape", "dtype"),
    [
        (None, None, "number"),
        ("float", None, "number"),
        ("int", None, "integer"),
        ("bool", None, "boolean"),
        ("enum", None, "string"),
        ("string", None, "string"),
        ("waveform", (500,), "array"),
    ],
)
async def test_a_readable_describes_its_value_type(
    value_type: str | None, shape: tuple[int] | None, dtype: str
) -> None:
    fake = FakeConnector()
    device = ConnectorReadable(fake, "C:1", name="c", value_type=value_type, shape=shape)

    (key,) = (await device.describe()).values()

    assert key["dtype"] == dtype
    assert fake.read_calls == []


def test_every_facility_value_type_has_a_dtype() -> None:
    from osprey_facility.schema import core

    assert sorted(connector_module.DTYPES) == sorted(member.value for member in core.ValueTypeEnum)


async def test_a_settable_and_a_triggerable_describe_their_channels_value_types() -> None:
    settable = ConnectorSettable(
        FakeConnector(), "SP", "RB", name="m", value_type="int", setpoint_value_type="float"
    )
    triggerable = ConnectorTriggerable(
        FakeConnector(),
        _COMMAND,
        _STATUS,
        idle="DONE",
        timeout_s=5.0,
        name="go",
        value_type="enum",
    )

    described = await settable.describe()

    assert (described["m"]["dtype"], described["m_setpoint"]["dtype"]) == ("integer", "number")
    assert (await triggerable.describe())["go"]["dtype"] == "string"


_WORDS = ("IDLE", "ACQUIRING", "DONE")
"""The words of an enum channel, in option-index order."""


async def test_an_enum_channel_is_described_as_text_with_its_words() -> None:
    fake = FakeConnector()
    worded = ConnectorReadable(fake, "E:1", name="e", value_type="enum", options=_WORDS)
    unworded = ConnectorReadable(fake, "E:2", name="f", value_type="enum")

    (key,) = (await worded.describe()).values()
    (plain,) = (await unworded.describe()).values()

    assert (key["dtype"], key["choices"]) == ("string", list(_WORDS))
    assert (plain["dtype"], "choices" in plain) == ("string", False)
    assert fake.read_calls == []


@pytest.mark.parametrize(
    ("raw", "recorded"),
    [(2, "DONE"), ("ACQUIRING", "ACQUIRING"), (7, "7"), (None, None)],
    ids=["index", "word", "index-without-a-word", "no-value"],
)
async def test_an_enum_channel_is_recorded_as_its_word(raw: Any, recorded: Any) -> None:
    device = ConnectorReadable(
        FakeConnector(readbacks={"E:1": raw}), "E:1", name="e", value_type="enum", options=_WORDS
    )

    assert (await device.read())["e"]["value"] == recorded


async def test_a_switch_and_a_number_are_recorded_as_read() -> None:
    fake = FakeConnector(readbacks={"B:1": True, "N:1": 2})
    switch = ConnectorReadable(fake, "B:1", name="b", value_type="bool", options=("OFF", "ON"))
    number = ConnectorReadable(fake, "N:1", name="n", value_type="int")

    assert (await switch.read())["b"]["value"] is True
    assert (await switch.describe())["b"]["dtype"] == "boolean"
    assert "choices" not in (await switch.describe())["b"]
    assert (await number.read())["n"]["value"] == 2


async def test_every_device_records_an_enum_channel_as_its_word() -> None:
    fake = FakeConnector(readbacks={"SP": 0, "RB": 1, "R": 2, _STATUS: 0})
    devices = await build_devices(
        [
            SettableSpec(
                "m",
                "SP",
                "RB",
                value_type="enum",
                options=_WORDS,
                setpoint_value_type="enum",
                setpoint_options=_WORDS,
            )
        ],
        [ReadableSpec("r", "R", value_type="enum", options=_WORDS)],
        fake,
        triggerables=[
            TriggerableSpec("go", _COMMAND, _STATUS, 0, 5.0, value_type="enum", options=_WORDS)
        ],
    )

    settable = await devices["m"].read()
    assert (settable["m"]["value"], settable["m_setpoint"]["value"]) == ("ACQUIRING", "IDLE")
    assert (await devices["m"].describe())["m_setpoint"]["choices"] == list(_WORDS)
    assert (await devices["r"].read())["r"]["value"] == "DONE"
    assert (await devices["go"].read())["go"]["value"] == "IDLE"
    assert (await devices["go"].describe())["go"]["choices"] == list(_WORDS)


async def test_the_idle_check_reads_the_raw_value_of_an_enum_status() -> None:
    fake = _acting()
    device = ConnectorTriggerable(
        fake,
        _COMMAND,
        _STATUS,
        idle=0,
        timeout_s=5.0,
        name="go",
        value_type="enum",
        options=("DONE", "BUSY"),
    )

    status = device.trigger()
    await status

    assert status.success
    assert fake.readbacks[_STATUS] == 0
    assert (await device.read())["go"]["value"] == "DONE"


async def test_build_devices_hands_each_device_its_value_types() -> None:
    devices = await build_devices(
        [SettableSpec("m", "SP", "RB", value_type="bool", setpoint_value_type="int")],
        [ReadableSpec("r", "R", value_type="string")],
        FakeConnector(),
        triggerables=[TriggerableSpec("go", _COMMAND, _STATUS, 0, 5.0, value_type="bool")],
    )

    assert (await devices["m"].describe())["m"]["dtype"] == "boolean"
    assert (await devices["m"].describe())["m_setpoint"]["dtype"] == "integer"
    assert (await devices["r"].describe())["r"]["dtype"] == "string"
    assert (await devices["go"].describe())["go"]["dtype"] == "boolean"


async def test_build_devices_hands_each_readable_its_shape() -> None:
    fake = FakeConnector(readbacks={"W:1": [0.0] * 500, "S:1": 0.0})

    devices = await build_devices(
        readables=[
            ReadableSpec(name="w1", read_pv="W:1", shape=(500,)),
            ReadableSpec(name="s1", read_pv="S:1"),
        ],
        connector=fake,
    )

    assert (await devices["w1"].describe())["w1"]["shape"] == [500]
    assert (await devices["s1"].describe())["s1"] == {
        "source": "connector:S:1",
        "dtype": "number",
        "shape": [],
    }


def test_a_count_on_a_waveform_readable_records_its_true_shape() -> None:
    """A real RunEngine records the descriptor the facility states and a cell of its length."""
    import numpy as np
    from bluesky import RunEngine
    from bluesky import plans as bp

    fake = FakeConnector(readbacks={"W:1": np.linspace(0.0, 1.0, 500)})
    device = ConnectorReadable(fake, "W:1", name="w1", shape=(500,))
    documents: list[tuple[str, dict[str, Any]]] = []

    RunEngine(context_managers=[])(
        bp.count([device]), lambda name, doc: documents.append((name, doc))
    )

    (descriptor,) = [doc for name, doc in documents if name == "descriptor"]
    (event,) = [doc for name, doc in documents if name == "event"]
    key = descriptor["data_keys"]["w1"]
    assert (key["dtype"], key["shape"]) == ("array", [500])
    assert len(event["data"]["w1"]) == 500


async def test_a_settable_with_a_distinct_readback_reports_its_demand_too() -> None:
    """``<name>`` is where the device is, ``<name>_setpoint`` where it was told to go.

    A plan that decides when to advance on the difference between the two --
    an insertion-device gap that takes ten seconds to arrive -- needs both off
    the one device it drives, without a second device aliasing the setpoint
    channel under a second name.
    """
    fake = FakeConnector(readbacks={"SR04U___GDS1PS_AC00": 25.0, "SR04U___GDS1PS_AM00": 17.3})
    device = ConnectorSettable(
        fake, "SR04U___GDS1PS_AC00", readback_pv="SR04U___GDS1PS_AM00", name="gap"
    )

    reading = await device.read()

    assert device.has_distinct_readback
    assert device.setpoint_key == "gap_setpoint"
    assert reading["gap"]["value"] == 17.3
    assert reading["gap_setpoint"]["value"] == 25.0
    assert set(reading) == {"gap", "gap_setpoint"}
    # Both halves are LIVE reads through the connector, never cached.
    assert set(fake.read_calls) == {"SR04U___GDS1PS_AC00", "SR04U___GDS1PS_AM00"}
    fake.readbacks["SR04U___GDS1PS_AM00"] = 25.0
    assert (await device.read())["gap"]["value"] == 25.0


async def test_a_settable_with_a_distinct_readback_describes_both_channels() -> None:
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="motor")

    assert await device.describe() == {
        "motor": {"source": "connector:RB", "dtype": "number", "shape": []},
        "motor_setpoint": {"source": "connector:SP", "dtype": "number", "shape": []},
    }


async def test_an_aliased_settable_reports_one_key() -> None:
    """Its readback IS its demand: a second key would restate ``<name>``."""
    fake = FakeConnector(readbacks={"SP": 4.0})
    device = ConnectorSettable(fake, "SP", name="motor")

    assert not device.has_distinct_readback
    assert await device.read() == {
        "motor": {"value": 4.0, "timestamp": pytest.approx(time.time(), abs=5)}
    }
    assert set(await device.describe()) == {"motor"}


async def test_a_settable_with_a_distinct_readback_locates_its_demand_and_its_readback() -> None:
    """``locate()`` answers where the device was told to go and where it is, live."""
    fake = FakeConnector(readbacks={"SR04U___GDS1PS_AC00": 25.0, "SR04U___GDS1PS_AM00": 17.3})
    device = ConnectorSettable(
        fake, "SR04U___GDS1PS_AC00", readback_pv="SR04U___GDS1PS_AM00", name="gap"
    )

    assert await device.locate() == {"setpoint": 25.0, "readback": 17.3}
    assert set(fake.read_calls) == {"SR04U___GDS1PS_AC00", "SR04U___GDS1PS_AM00"}
    fake.readbacks["SR04U___GDS1PS_AM00"] = 25.0
    assert await device.locate() == {"setpoint": 25.0, "readback": 25.0}


async def test_an_aliased_settable_locates_one_channel_as_both() -> None:
    """Its readback IS its demand: one read answers both keys."""
    fake = FakeConnector(readbacks={"SP": 4.0})
    device = ConnectorSettable(fake, "SP", name="motor")

    assert await device.locate() == {"setpoint": 4.0, "readback": 4.0}
    assert fake.read_calls == ["SP"]


def test_a_connector_settable_is_bluesky_locatable() -> None:
    """A plan written against Bluesky's protocol locates an OSPREY device."""
    from bluesky import RunEngine
    from bluesky import plan_stubs as bps
    from bluesky.protocols import Locatable

    fake = FakeConnector(readbacks={"SP": 2.5, "RB": 2.4})
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="hcm1")
    located: list[Any] = []

    def _plan():
        located.append((yield from bps.loc(device)))

    assert isinstance(device, Locatable)
    RunEngine(context_managers=[])(_plan())
    assert located == [{"setpoint": 2.5, "readback": 2.4}]


async def test_the_demand_key_is_not_hinted_so_rd_returns_the_readback() -> None:
    """``bps.rd`` locates the device and returns its readback; the one hinted
    field is that readback column, and the demand key ``<name>_setpoint`` is
    never hinted."""
    from bluesky.utils import get_hinted_fields

    fake = FakeConnector(readbacks={"SP": 25.0, "RB": 17.3})
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="gap")

    assert device.hints == {"fields": ["gap"]}
    assert get_hinted_fields(device) == ["gap"]
    assert set(await device.read()) == {"gap", "gap_setpoint"}


def test_connector_settable_hints_declare_the_device_field() -> None:
    """The movable declares its one data key as the hinted field."""
    device = ConnectorSettable(FakeConnector(readbacks={"SP": 0.0}), "SP", name="motor")

    assert device.hints == {"fields": ["motor"]}


def test_connector_readable_hints_declare_the_device_field() -> None:
    """The readable declares its one data key as the hinted field."""
    device = ConnectorReadable(FakeConnector(readbacks={"BPM:1": 1.0}), "BPM:1", name="bpm1")

    assert device.hints == {"fields": ["bpm1"]}


@pytest.mark.parametrize("cls", [ConnectorSettable, ConnectorReadable])
def test_hints_is_a_property_override(cls: type) -> None:
    """Instance assignment raises on the read-only base property, so it must be
    a property on the subclass itself — not an attribute set in ``__init__``."""
    assert isinstance(inspect.getattr_static(cls, "hints"), property)


async def test_hints_do_not_disturb_read_and_describe() -> None:
    """Declaring hints must leave the read/describe contract untouched."""
    fake = FakeConnector(readbacks={"SP": 7.0, "BPM:1": 42.0})
    settable = ConnectorSettable(fake, "SP", name="motor")
    readable = ConnectorReadable(fake, "BPM:1", name="bpm1")

    assert (await settable.read())["motor"]["value"] == 7.0
    assert await settable.describe() == {
        "motor": {"source": "connector:SP", "dtype": "number", "shape": []}
    }
    assert (await readable.read())["bpm1"]["value"] == 42.0
    assert await readable.describe() == {
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


async def test_describe_states_units_exactly_where_the_spec_does() -> None:
    """Each data key carries ``units`` when its spec states one, and no key otherwise;
    ``describe()`` reads nothing through the connector to say so."""
    fake = FakeConnector(readbacks={"P:SP": 0.0, "P:RB": 0.0, "Q:SP": 0.0, "R:SP": 0.0, "T": 0.0})
    paired = SettableSpec(
        name="p", setpoint_pv="P:SP", readback_pv="P:RB", unit="u-rb", setpoint_unit="u-sp"
    )
    unpaired = SettableSpec(name="q", setpoint_pv="Q:SP", unit="u-q")
    unitless = SettableSpec(name="r", setpoint_pv="R:SP")
    readable = ReadableSpec(name="t", read_pv="T", unit="u-t")
    bare = ReadableSpec(name="s", read_pv="T")

    devices = await build_devices(
        settables=[paired, unpaired, unitless], readables=[readable, bare], connector=fake
    )
    described = {name: await device.describe() for name, device in devices.items()}

    assert described["p"]["p"]["units"] == paired.unit
    assert described["p"]["p_setpoint"]["units"] == paired.setpoint_unit
    assert described["q"]["q"]["units"] == unpaired.unit
    assert "units" not in described["r"]["r"]
    assert described["t"]["t"]["units"] == readable.unit
    assert "units" not in described["s"]["s"]
    assert fake.read_calls == []


async def test_a_paired_settable_without_a_setpoint_unit_states_none_for_its_demand() -> None:
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", unit="u-rb")

    described = await device.describe()

    assert described["m"]["units"] == "u-rb"
    assert "units" not in described["m_setpoint"]


async def test_build_devices_with_no_specs_returns_empty_mapping() -> None:
    """An empty settables/readables sequence builds an empty (but valid) device set."""
    devices = await build_devices(connector=FakeConnector())
    assert devices == {}


async def test_build_devices_without_connector_raises_at_build_time() -> None:
    """A missing connector fails here, at the misconfiguration site — not as an
    ``AttributeError`` deep inside a device's ``set()``/``read()`` at run time."""
    with pytest.raises(ValueError, match="requires a connector"):
        await build_devices(settables=[SettableSpec(name="hcm1", setpoint_pv="HCM1:SP")])


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


# --- the worker's bound is the shared settle rule ---------------------------------


@pytest.mark.parametrize(
    ("device", "floor", "simulated", "record", "expected"),
    [
        ({"settle_tolerance": _BAND}, None, False, {"absolute": _BAND}, _BAND),
        ({"settle_relative": 1e-4}, None, False, {"relative": 1e-4}, 0.02),
        ({"motion_band": _BAND}, None, True, None, _BAND),
        ({"motion_band": _BAND}, None, False, None, connector_module.DEFAULT_SETTLE_TOLERANCE),
        ({"settle_tolerance": _BAND}, "0.1", False, {"absolute": _BAND}, 0.1),
    ],
    ids=["absolute", "relative-of-the-demand", "motion-on-the-simulator", "motion-live", "floor"],
)
async def test_the_worker_bound_is_the_shared_rule(
    monkeypatch: pytest.MonkeyPatch,
    device: dict[str, float],
    floor: str | None,
    simulated: bool,
    record: dict[str, float] | None,
    expected: float,
) -> None:
    from osprey_connectors.settle import settle_bound

    if floor is None:
        monkeypatch.delenv(connector_module.SETTLE_TOLERANCE_ENV, raising=False)
    else:
        monkeypatch.setenv(connector_module.SETTLE_TOLERANCE_ENV, floor)
    landings: list[Any] = []

    async def record_landings(connector: Any, batch: Any, *, timeout_s: float) -> None:
        del connector, timeout_s
        landings.extend(batch)

    monkeypatch.setattr(connector_module, "settle", record_landings)
    fake = FakeConnector(readbacks={"SP": 0.0, "RB": 0.0})
    settable = ConnectorSettable(
        fake, "SP", readback_pv="RB", name="m", simulated=simulated, **device
    )

    await settable.set(-200.0)

    (landing,) = landings
    assert (landing.setpoint, landing.readback, landing.demand) == ("SP", "RB", -200.0)
    assert landing.bound == pytest.approx(expected)
    assert landing.bound == pytest.approx(
        settle_bound(
            record,
            -200.0,
            motion=device.get("motion_band", 0.0),
            simulated=simulated,
            floor=connector_module.settle_tolerance(),
        )
    )


async def test_the_timeout_names_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _unrequested(monkeypatch, -3.0 + 1.1 * _BAND, "0.1")
    device = ConnectorSettable(fake, "SP", readback_pv="RB", name="m", settle_tolerance=_BAND)

    with pytest.raises(TimeoutError, match=rf"readback RB did not settle to -3\.0 ± {_BAND:g}"):
        await device.set(-3.0)


# --- a command a plan triggers ------------------------------------------------

_COMMAND = "ES1:MCA01:ACQ"
_STATUS = "ES1:MCA01:ACQ_STATUS"


_ACT_TIMEOUT_S = 5.0
"""The ``timeout_s`` the command states for one act: longer than any act these tests complete."""

_UNANSWERED_ACT_TIMEOUT_S = _ACT_TIMEOUT_S / 100
"""The ``timeout_s`` stated for a command whose status never moves: the device's own
bound, which ends that act with ``CommandNoEffectError``; the test waits on the error."""


def _triggerable(fake: FakeConnector, timeout_s: float = _ACT_TIMEOUT_S) -> ConnectorTriggerable:
    return ConnectorTriggerable(fake, _COMMAND, _STATUS, idle=0, timeout_s=timeout_s, name="mca")


def _acting() -> FakeConnector:
    """A connector whose status leaves idle and returns once the command is written."""
    fake = FakeConnector(readbacks={_STATUS: 0})
    fake.responses[_COMMAND] = (_STATUS, [1, 0])
    return fake


async def test_trigger_fires_through_the_checked_write_unconfirmed_and_waits_for_the_act() -> None:
    fake = _acting()
    device = _triggerable(fake)

    status = device.trigger()
    await status

    assert status.success
    assert fake.write_calls == [(_COMMAND, 1, {"confirm": False})]
    assert fake.readbacks[_STATUS] == 0
    assert fake.subscribers == {}


async def test_a_busy_status_refuses_the_trigger_and_writes_nothing() -> None:
    fake = _acting()
    fake.readbacks[_STATUS] = 1
    device = _triggerable(fake)

    with pytest.raises(CommandBusyError):
        await device.trigger()

    assert fake.write_calls == []


async def test_a_status_that_never_moves_fails_the_trigger() -> None:
    fake = FakeConnector(readbacks={_STATUS: 0})
    device = _triggerable(fake, timeout_s=_UNANSWERED_ACT_TIMEOUT_S)

    with pytest.raises(CommandNoEffectError):
        await device.trigger()

    assert [address for address, _value, _kwargs in fake.write_calls] == [_COMMAND]


def _holding() -> FakeConnector:
    """A connector whose status read waits until the trigger reading it is cancelled."""
    fake = _acting()
    fake.hold_reads.add(_STATUS)
    fake.read_held_here = asyncio.Event()
    return fake


async def test_a_second_trigger_while_one_runs_is_refused() -> None:
    fake = _holding()
    device = _triggerable(fake)
    first = device.trigger()
    await fake.read_held_here.wait()

    with pytest.raises(RuntimeError, match="mca: the previous trigger is still running"):
        await device.trigger()

    device.cancel_trigger()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert fake.write_calls == []


async def test_pause_before_the_write_cancels_and_nothing_is_written() -> None:
    fake = _holding()
    device = _triggerable(fake)
    status = device.trigger()
    await fake.read_held_here.wait()

    device.pause()

    with pytest.raises(asyncio.CancelledError):
        await status
    assert not status.success
    assert fake.write_calls == []
    assert fake.subscribers == {}
    device.resume()
    assert fake.write_calls == []


def test_a_triggerable_has_no_set() -> None:
    device = _triggerable(FakeConnector())

    assert not hasattr(device, "set")
    assert not hasattr(device, "stop")


async def test_trigger_writes_is_what_trigger_sends() -> None:
    fake = _acting()
    device = _triggerable(fake)

    stated = device.trigger_writes()

    assert stated == (ChannelWrite(_COMMAND, 1),)
    assert fake.read_calls == [] and fake.write_calls == []
    await device.trigger()
    assert [(address, value) for address, value, _kwargs in fake.write_calls] == [
        tuple(write) for write in stated
    ]


async def test_a_triggerable_reads_its_status_under_its_name() -> None:
    fake = FakeConnector(readbacks={_STATUS: 0})
    device = _triggerable(fake)

    reading = await device.read()
    described = await device.describe()

    assert list(reading) == ["mca"]
    assert reading["mca"]["value"] == 0
    assert fake.read_calls == [_STATUS]
    assert list(described) == ["mca"]
    assert described["mca"]["source"] == f"connector:{_STATUS}"
    assert device.hints == {"fields": ["mca"]}


async def test_build_devices_builds_a_triggerable_from_its_spec() -> None:
    fake = _acting()
    spec = TriggerableSpec("mca", _COMMAND, _STATUS, 0, 5.0)

    devices = await build_devices(connector=fake, triggerables=[spec])

    assert isinstance(devices["mca"], ConnectorTriggerable)
    assert devices["mca"].trigger_writes() == (ChannelWrite(_COMMAND, 1),)


async def test_build_devices_builds_a_positioner_for_a_started_spec() -> None:
    act = CommandAct(_COMMAND, _STATUS, 0, 5.0)
    started = SettableSpec("stage", "STAGE:SP", "STAGE:RB", started_by=act)
    plain = SettableSpec("quad", "QUAD:SP")

    devices = await build_devices([started, plain], connector=FakeConnector())

    assert isinstance(devices["stage"], ConnectorPositioner)
    assert type(devices["quad"]) is ConnectorSettable
    assert devices["stage"].set_writes(2.0) == (
        ChannelWrite("STAGE:SP", 2.0),
        ChannelWrite(_COMMAND, PRESS_VALUE),
    )
