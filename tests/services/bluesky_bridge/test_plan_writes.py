"""What a plan device states it writes, and the guard that holds a launch to it.

A plan device writes only from ``set()`` and ``trigger()``, and states the
channels each of those writes through ``set_writes`` / ``trigger_writes`` —
pure methods, no I/O. ``plan_writes`` reads that statement for every message a
plan yields: the preview tallies it per channel, and both worker wrappers hold
a launch to its plan's ``writes`` declaration with it, refusing before the
first undeclared write is sent.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from bluesky.utils import Msg

from osprey.services.bluesky_bridge import devices as devices_package
from osprey.services.bluesky_bridge.devices.connector import ConnectorSettable
from osprey.services.bluesky_bridge.devices.mock import MockReadable, MockSettable
from osprey.services.bluesky_bridge.plan_writes import (
    ChannelWrite,
    UndeclaredWriteError,
    UnstatedWriteError,
    WritesTally,
    held_to_declaration,
    message_writes,
)
from tests.services.bluesky_bridge._fake_connector import FakeConnector


class _SilentSettable:
    """A device that can be set but states nothing about what that writes."""

    name = "silent_motor"

    def set(self, value: Any) -> None:  # noqa: ARG002 - the bluesky verb signature
        raise AssertionError("the guard must refuse before the device is driven")


class _SilentTriggerable:
    """A device that can be triggered but states nothing about what that writes."""

    name = "silent_detector"

    def trigger(self) -> None:  # pragma: no cover - never driven
        raise AssertionError("the guard must refuse before the device is driven")


class _StatedSettable:
    """A device whose ``set`` writes its setpoint and a command channel."""

    name = "stage"

    def set(self, value: Any) -> None:  # noqa: ARG002 - the bluesky verb signature
        raise AssertionError("a walk never drives the device")

    def set_writes(self, value: Any) -> tuple[ChannelWrite, ...]:
        return (ChannelWrite("X:SP", value), ChannelWrite("X:GO", 1))


class _Readable:
    """A device with no verb that writes."""

    name = "bpm"

    def read(self) -> dict[str, Any]:  # pragma: no cover - never driven
        return {}


def test_a_set_on_a_device_that_states_no_writes_is_refused() -> None:
    with pytest.raises(UnstatedWriteError) as refusal:
        message_writes(Msg("set", _SilentSettable(), 1.0))

    assert str(refusal.value) == (
        "device 'silent_motor' does not state the channels its set writes; the launch is refused"
    )


def test_a_trigger_on_a_triggerable_that_states_no_writes_is_refused() -> None:
    with pytest.raises(UnstatedWriteError) as refusal:
        message_writes(Msg("trigger", _SilentTriggerable()))

    assert str(refusal.value) == (
        "device 'silent_detector' does not state the channels its trigger writes; "
        "the launch is refused"
    )


def test_a_statement_that_is_not_a_tuple_of_channel_writes_is_refused() -> None:
    class _Misstated(_SilentSettable):
        def set_writes(self, value: Any) -> list[Any]:
            return [("X:SP", value)]

    with pytest.raises(UnstatedWriteError):
        message_writes(Msg("set", _Misstated(), 1.0))


def test_a_message_to_an_object_without_the_verb_writes_nothing() -> None:
    readable = _Readable()

    assert message_writes(Msg("set", readable, 1.0)) == ()
    assert message_writes(Msg("trigger", readable)) == ()
    assert message_writes(Msg("read", readable)) == ()
    assert message_writes(Msg("open_run", None)) == ()


def test_a_connector_settable_states_its_one_setpoint_write() -> None:
    device = ConnectorSettable(FakeConnector(), "SR:SP", readback_pv="SR:RB", name="motor")

    assert message_writes(Msg("set", device, 2.5)) == (ChannelWrite("SR:SP", 2.5),)


def test_a_mock_settable_states_one_write_to_the_channel_it_stands_in_for() -> None:
    device = MockSettable(name="SR:C01:COR:SP")

    assert message_writes(Msg("set", device, 0.5)) == (ChannelWrite("SR:C01:COR:SP", 0.5),)


def test_a_mock_readable_states_that_its_trigger_writes_nothing() -> None:
    assert message_writes(Msg("trigger", MockReadable(name="SR:C01:BPM:X"))) == ()


def _shipped_plan_device_classes() -> list[type]:
    classes: list[type] = []
    for module_info in pkgutil.iter_modules(devices_package.__path__):
        module = importlib.import_module(f"{devices_package.__name__}.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if any(callable(getattr(cls, verb, None)) for verb in ("set", "trigger")):
                classes.append(cls)
    return classes


def test_every_shipped_plan_device_class_states_the_writes_of_each_verb_it_has() -> None:
    classes = _shipped_plan_device_classes()

    assert {ConnectorSettable, MockSettable, MockReadable} <= set(classes)
    for cls in classes:
        for verb, statement in (("set", "set_writes"), ("trigger", "trigger_writes")):
            if callable(getattr(cls, verb, None)):
                assert callable(getattr(cls, statement, None)), (
                    f"{cls.__qualname__} has {verb}() but no {statement}()"
                )


def _drain(plan: Any) -> list[Any]:
    """Drive a plan as a walk does: every message, ``None`` sent back."""
    return list(plan)


def _moving_plan(device: Any) -> Any:
    yield Msg("open_run", None)
    yield Msg("set", device, 1.0)
    yield Msg("set", device, 2.0)
    yield Msg("close_run", None)


def test_a_plan_not_declaring_writes_is_refused_at_its_first_write() -> None:
    yielded: list[Any] = []
    closed: list[bool] = []

    def plan() -> Any:
        try:
            yield from _moving_plan(_StatedSettable())
        finally:
            closed.append(True)

    guard = held_to_declaration("quiet_plan", False, plan())
    with pytest.raises(UndeclaredWriteError) as refusal:
        for message in guard:
            yielded.append(message.command)

    assert yielded == ["open_run"]
    assert closed == [True]
    assert str(refusal.value) == (
        "plan 'quiet_plan' does not declare writes: true and this launch writes X:SP; "
        "state writes: true in PLAN_METADATA to run it"
    )


def test_a_declared_or_undeclared_plan_runs_its_stated_writes() -> None:
    for declared in (True, None):
        guard = held_to_declaration("loud_plan", declared, _moving_plan(_StatedSettable()))
        assert [m.command for m in _drain(guard)] == ["open_run", "set", "set", "close_run"]


def test_an_undeclared_plan_that_writes_nothing_runs() -> None:
    def plan() -> Any:
        yield Msg("open_run", None)
        yield Msg("trigger", _Readable())
        yield Msg("close_run", None)

    assert len(_drain(held_to_declaration("reader", False, plan()))) == 3


def test_metadata_none_still_refuses_an_unstated_write() -> None:
    guard = held_to_declaration("anon", None, _moving_plan(_SilentSettable()))

    with pytest.raises(UnstatedWriteError):
        _drain(guard)


def test_the_refusal_is_raised_out_of_the_guard_not_into_the_plan() -> None:
    swallowed: list[BaseException] = []

    def stubborn() -> Any:
        try:
            yield Msg("set", _StatedSettable(), 1.0)
        except Exception as exc:  # a plan that would swallow anything thrown at it
            swallowed.append(exc)
            yield Msg("set", _StatedSettable(), 2.0)

    guard = held_to_declaration("stubborn", False, stubborn())
    with pytest.raises(UndeclaredWriteError):
        _drain(guard)

    assert swallowed == []


def test_the_guard_forwards_replies_and_exceptions_like_yield_from() -> None:
    seen: list[Any] = []

    def plan() -> Any:
        reply = yield Msg("read", _Readable())
        seen.append(reply)
        try:
            yield Msg("read", _Readable())
        except ValueError as exc:
            seen.append(str(exc))
        return "done"

    guard = held_to_declaration("echo", False, plan())
    assert next(guard).command == "read"
    assert guard.send({"value": 1}).command == "read"
    with pytest.raises(StopIteration) as stop:
        guard.throw(ValueError("thrown in"))

    assert seen == [{"value": 1}, "thrown in"]
    assert stop.value.value == "done"


def test_the_guard_closes_the_plan_when_it_is_closed() -> None:
    closed: list[bool] = []

    def plan() -> Any:
        try:
            yield Msg("read", _Readable())
            yield Msg("read", _Readable())
        finally:
            closed.append(True)

    guard = held_to_declaration("closer", False, plan())
    next(guard)
    guard.close()

    assert closed == [True]


def test_the_tally_counts_each_channel_once_in_first_write_order() -> None:
    tally = WritesTally()
    tally.add("stage", (ChannelWrite("X:SP", 1.0), ChannelWrite("X:GO", 1)))
    tally.add("other", (ChannelWrite("Y:SP", 5.0),))
    tally.add("stage", (ChannelWrite("X:SP", 2.0), ChannelWrite("X:GO", 1)))

    assert tally.entries() == [
        {"channel": "X:SP", "device": "stage", "value": None, "count": 2},
        {"channel": "X:GO", "device": "stage", "value": 1, "count": 2},
        {"channel": "Y:SP", "device": "other", "value": 5.0, "count": 1},
    ]
    assert tally.total() == 5


def test_a_channel_written_with_differing_values_states_no_value() -> None:
    tally = WritesTally()
    tally.add("motor", (ChannelWrite("SR:SP", 1.0),))
    tally.add("motor", (ChannelWrite("SR:SP", 1.0),))
    tally.add("motor", (ChannelWrite("SR:SP", 1.5),))
    tally.add("motor", (ChannelWrite("SR:SP", 1.0),))

    assert tally.entries() == [{"channel": "SR:SP", "device": "motor", "value": None, "count": 4}]


def test_an_empty_tally_states_zero_writes() -> None:
    tally = WritesTally()

    assert tally.entries() == []
    assert tally.total() == 0
