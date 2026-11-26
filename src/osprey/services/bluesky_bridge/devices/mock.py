"""Pure-mock ophyd-async device factory: no CA/EPICS, no scenario/machine state.

``MockSettable``/``MockReadable``/``MockTriggerable`` are in-process devices — there is no
external process backing their values, so nothing here is a "scenario": every
instance starts from the same fixed initial state on every process start, and
there is no shared substrate two callers could observe each other mutating.
That makes this module suitable ONLY for:

- the plan-catalog tests, where the point is to prove a shipped plan body
  actually runs against *some* bluesky-shaped device, not to validate real
  device behavior; and
- ``plan_validation.py``'s stage-3 dry run, which answers "does this authored
  plan body actually execute" without touching an instrument.

It must NEVER be the substrate for the Phase 3 scenario benchmark — that
benchmark's whole point is cross-checking ophyd-async's view of a PV against
OSPREY's own ``channel_read`` on a real (soft-IOC) Channel Access server, and
this module never touches CA. Use
:mod:`osprey.services.bluesky_bridge.devices.connector` for anything that
needs to read/write a real PV — it mediates every read/write through the
OSPREY connector rather than speaking Channel Access directly.

Reimplements the shape of ``ophyd_async.sim.SimMotor``/``SimPointDetector``
(soft position signal with instant "move"; a triggerable soft readout) rather
than importing ``ophyd_async.sim`` directly: that package's ``__init__``
eagerly imports ``SimBlobDetector``, which pulls in ``h5py`` — a heavy
dependency this bridge does not need for a plain setpoint + readback.

A stand-in handed ``data_keys`` — the description the worker gave of the device
it stands in for (``qserver_startup.describe_devices``) — records exactly those
keys, each described as given and holding :func:`stand_in_value`, so a dry run
records the columns, value types and words the run will record. Without one, it
records under its own name.

Imports ophyd-async (a core dependency), so this module (like the rest of
``devices/``) is kept out of the bridge lifecycle core's import path
(``app.py``, ``runs.py``, ``security.py``), which stays import-clean of ophyd.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from bluesky.protocols import DataKey, Hints, Location, Reading
from ophyd_async.core import (
    AsyncStatus,
    StandardReadable,
    soft_signal_r_and_setter,
    soft_signal_rw,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)

from ..plan_fields import SETPOINT_KEY_SUFFIX
from ..plan_writes import ChannelWrite
from ._connect import connect_all
from .connector import data_key

#: The value a stand-in records for a scalar of each non-float ``dtype``.
_STAND_IN_SCALARS: dict[str, Any] = {"string": "", "boolean": False, "integer": 0}


def stand_in_value(described: Mapping[str, Any]) -> Any:
    """The value a stand-in records for the channel *described* states.

    A channel with ``choices`` holds its first word, text holds ``""``, a
    boolean ``False``, an integer ``0`` and any other number ``0.0``; a key
    with a non-empty ``shape`` holds nested lists of that shape (a ragged axis
    one long) of ``0.0``.
    """
    shape = described.get("shape")
    if isinstance(shape, list) and shape:
        value: Any = 0.0
        for size in reversed(shape):
            value = [value] * (size if isinstance(size, int) and size >= 0 else 1)
        return value
    choices = described.get("choices")
    if isinstance(choices, list) and choices:
        return choices[0]
    dtype = described.get("dtype")
    return _STAND_IN_SCALARS.get(dtype, 0.0) if isinstance(dtype, str) else 0.0


class _Recording:
    """What a stand-in records when it is handed the description of the
    device it stands in for: those data keys, each holding a value."""

    def __init__(self, data_keys: Mapping[str, Mapping[str, Any]]) -> None:
        self.data_keys = {key: dict(described) for key, described in data_keys.items()}
        self.values = {key: stand_in_value(described) for key, described in data_keys.items()}

    def read(self) -> dict[str, Reading[Any]]:
        now = time.time()
        return {key: {"value": value, "timestamp": now} for key, value in self.values.items()}

    def describe(self) -> dict[str, DataKey]:
        return {key: dict(described) for key, described in self.data_keys.items()}  # type: ignore[misc]

    def hints(self) -> Hints:
        """The scalar keys that hold a number, as a live plot draws them."""
        return {
            "fields": [
                key
                for key, described in self.data_keys.items()
                if described.get("dtype") in ("number", "integer") and not described.get("shape")
            ]
        }


def _recording(data_keys: Mapping[str, Mapping[str, Any]] | None) -> _Recording | None:
    return None if data_keys is None else _Recording(data_keys)


class MockSettable(StandardReadable):
    """An in-process simulated setpoint: a soft position signal that "moves" instantly.

    ``readback`` is the hinted (primary) signal read into every document;
    ``setpoint`` is the last demand, and ``locate()`` reports both. There is
    no velocity or settle time to simulate — contract tests care that
    ``set()`` completes and the readback reflects the new value, not motion
    realism.

    :meth:`set_writes` states the one write ``set()`` stands in for: the demand
    to the channel this device is named for.

    Handed ``data_keys``, it records those keys instead, and ``set(value)``
    puts ``value`` under its own name and under ``<name>_setpoint`` where it
    records them.
    """

    def __init__(
        self,
        name: str = "",
        initial_value: float = 0.0,
        *,
        data_keys: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.readback, self._set_readback = soft_signal_r_and_setter(float, initial_value)
        self.setpoint = soft_signal_rw(float, initial_value)
        self._recorded = _recording(data_keys)
        super().__init__(name=name)

    def set_writes(self, value: float) -> tuple[ChannelWrite, ...]:
        """The channel writes ``set(value)`` stands in for: ``value`` to this device's name."""
        return (ChannelWrite(self.name, value),)

    @AsyncStatus.wrap
    async def set(self, value: Any) -> None:
        """Move to ``value`` instantly: write the setpoint, then the readback."""
        if self._recorded is not None:
            for key in (self.name, f"{self.name}{SETPOINT_KEY_SUFFIX}"):
                if key in self._recorded.values:
                    self._recorded.values[key] = value
            return
        await self.setpoint.set(value)
        self._set_readback(value)

    async def locate(self) -> Location[Any]:
        """Return the last demand and the current readback."""
        if self._recorded is not None:
            held = self._recorded.values.get(self.name)
            demand = self._recorded.values.get(f"{self.name}{SETPOINT_KEY_SUFFIX}", held)
            return Location(setpoint=demand, readback=held)
        return Location(
            setpoint=await self.setpoint.get_value(),
            readback=await self.readback.get_value(),
        )

    async def read(self) -> dict[str, Reading[Any]]:
        if self._recorded is not None:
            return self._recorded.read()
        return await super().read()

    async def describe(self) -> dict[str, DataKey]:
        if self._recorded is not None:
            return self._recorded.describe()
        return await super().describe()

    @property
    def hints(self) -> Hints:
        if self._recorded is not None:
            return self._recorded.hints()
        return super().hints


class MockReadable(StandardReadable):
    """An in-process simulated readback: a monotonically incrementing counter.

    Deterministic on purpose — a fixed count sequence (1, 2, 3, ...) per
    instance, rather than a random value, so lifecycle/contract tests get
    reproducible documents without needing to seed an RNG.

    :meth:`trigger_writes` states that ``trigger()`` writes no channel.

    Handed ``data_keys``, it records those keys instead.
    """

    def __init__(
        self, name: str = "", *, data_keys: Mapping[str, Mapping[str, Any]] | None = None
    ) -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.value, self._set_value = soft_signal_r_and_setter(int, 0)
        self._count = 0
        self._recorded = _recording(data_keys)
        super().__init__(name=name)

    def trigger_writes(self) -> tuple[ChannelWrite, ...]:
        """The channel writes ``trigger()`` makes: none. No I/O."""
        return ()

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        self._count += 1
        self._set_value(self._count)

    async def read(self) -> dict[str, Reading[Any]]:
        if self._recorded is not None:
            return self._recorded.read()
        return await super().read()

    async def describe(self) -> dict[str, DataKey]:
        if self._recorded is not None:
            return self._recorded.describe()
        return await super().describe()

    @property
    def hints(self) -> Hints:
        if self._recorded is not None:
            return self._recorded.hints()
        return super().hints


class MockTriggerable(StandardReadable):
    """An in-process stand-in for a device a plan triggers.

    ``trigger()`` completes at once and counts (:attr:`trigger_count`).
    ``read()``/``describe()`` report one data key named exactly for the device,
    the column a real triggered device records, holding the trigger count.

    :meth:`trigger_writes` states the write a real triggered device sends: the
    press value to the channel this device is named for.

    Handed ``data_keys``, it records those keys instead: a detector's stand-in
    records the run's own result columns, under their own names.
    """

    def __init__(
        self, name: str = "", *, data_keys: Mapping[str, Mapping[str, Any]] | None = None
    ) -> None:
        self.trigger_count = 0
        self._recorded = _recording(data_keys)
        super().__init__(name=name)

    def trigger_writes(self) -> tuple[ChannelWrite, ...]:
        """The channel writes ``trigger()`` stands in for: 1 to this device's name."""
        return (ChannelWrite(self.name, 1),)

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        self.trigger_count += 1

    async def read(self) -> dict[str, Reading[Any]]:
        """The trigger count under the device's own name."""
        if self._recorded is not None:
            return self._recorded.read()
        return {self.name: {"value": self.trigger_count, "timestamp": time.time()}}

    async def describe(self) -> dict[str, DataKey]:
        """One scalar number under the device's own name."""
        if self._recorded is not None:
            return self._recorded.describe()
        return {self.name: data_key(self.name, scheme="mock")}

    @property
    def hints(self) -> Hints:
        """The one data key, named for the device."""
        if self._recorded is not None:
            return self._recorded.hints()
        return {"fields": [self.name]}


async def build_devices(
    settable_names: Sequence[str] = ("sp1",),
    readable_names: Sequence[str] = ("rb1",),
    triggered_names: Sequence[str] = (),
    described: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and connect a set of mock settables/readables/triggerables, keyed by name.

    Matches the ``get_devices() -> dict[str, Any]`` shape ``plans.py``'s
    built-in plans (and any facility-injected plan, per ``plan_loader.py``)
    resolve device names against. Connection (and why it's an explicit
    ``connect()`` rather than ``init_devices()``) is handled by
    :func:`._connect.connect_all`.

    Args:
        settable_names: Device-mapping keys for the ``MockSettable`` instances
            to build. Defaults to a single ``"sp1"`` for the deploy smoke demo.
        readable_names: Device-mapping keys for the ``MockReadable`` instances
            to build. Defaults to a single ``"rb1"``.
        triggered_names: Device-mapping keys for the ``MockTriggerable``
            instances to build. Defaults to none.
        described: The worker's description of each device a name stands in
            for, ``{name: {"kind", "data_keys"}}``. A described name's
            stand-in records its ``data_keys``. Defaults to none.

    Returns:
        Mapping of device name to connected device instance.
    """
    described = described or {}

    def keys(name: str) -> Mapping[str, Mapping[str, Any]] | None:
        entry = described.get(name)
        return None if entry is None else entry["data_keys"]

    devices: dict[str, Any] = {}
    for name in settable_names:
        devices[name] = MockSettable(name=name, data_keys=keys(name))
    for name in readable_names:
        devices[name] = MockReadable(name=name, data_keys=keys(name))
    for name in triggered_names:
        devices[name] = MockTriggerable(name=name, data_keys=keys(name))
    return await connect_all(devices)
