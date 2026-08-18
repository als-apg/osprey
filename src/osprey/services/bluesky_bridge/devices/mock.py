"""Pure-mock ophyd-async device factory: no CA/EPICS, no scenario/machine state.

``MockSettable``/``MockReadable`` are in-process soft-signal devices — there is no
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

Imports ophyd-async (a core dependency), so this module (like the rest of
``devices/``) is kept out of the bridge lifecycle core's import path
(``app.py``, ``runs.py``, ``security.py``), which stays import-clean of ophyd.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ophyd_async.core import (
    AsyncStatus,
    StandardReadable,
    soft_signal_r_and_setter,
    soft_signal_rw,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)

from ._connect import connect_all


class MockSettable(StandardReadable):
    """An in-process simulated setpoint: a soft position signal that "moves" instantly.

    ``readback`` is the hinted (primary) signal read into every document;
    ``setpoint`` records the last commanded position. There is no velocity or
    settle time to simulate — contract tests care that ``set()`` completes and
    the readback reflects the new value, not motion realism.
    """

    def __init__(self, name: str = "", initial_value: float = 0.0) -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.readback, self._set_readback = soft_signal_r_and_setter(float, initial_value)
        self.setpoint = soft_signal_rw(float, initial_value)
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        """Move to ``value`` instantly: write the setpoint, then the readback."""
        await self.setpoint.set(value)
        self._set_readback(value)


class MockReadable(StandardReadable):
    """An in-process simulated readback: a monotonically incrementing counter.

    Deterministic on purpose — a fixed count sequence (1, 2, 3, ...) per
    instance, rather than a random value, so lifecycle/contract tests get
    reproducible documents without needing to seed an RNG.
    """

    def __init__(self, name: str = "") -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.value, self._set_value = soft_signal_r_and_setter(int, 0)
        self._count = 0
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        self._count += 1
        self._set_value(self._count)


async def build_devices(
    settable_names: Sequence[str] = ("sp1",),
    readable_names: Sequence[str] = ("rb1",),
) -> dict[str, Any]:
    """Build and connect a set of mock settables/readables, keyed by name.

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

    Returns:
        Mapping of device name to connected device instance.
    """
    devices: dict[str, Any] = {}
    for name in settable_names:
        devices[name] = MockSettable(name=name)
    for name in readable_names:
        devices[name] = MockReadable(name=name)
    return await connect_all(devices)
