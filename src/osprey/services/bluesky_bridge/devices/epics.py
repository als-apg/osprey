"""ophyd-async EPICS device factory: Channel Access clients of a real IOC.

Builds bluesky-shaped ``EpicsMotor``/``EpicsDetector`` devices that speak
Channel Access directly to explicit, caller-supplied PV names — no OSPREY
connector is involved (DA ruling: no new ``osprey`` connector; ophyd-async
already speaks CA). This is the factory used for the Phase 3 scenario
benchmark, which reads the same soft-IOC PVs through both this module and
OSPREY's own ``channel_read`` MCP tool as a consistency cross-check — see
``channel_read.py`` (:mod:`osprey.mcp_server.control_system.tools.channel_read`),
which also takes bare PV addresses with no assumed naming convention. Matching
that, ``EpicsMotorSpec``/``EpicsDetectorSpec`` take full PV names verbatim
(never a prefix + guessed suffix), so both sides of the cross-check name the
exact same PV.

Deliberately does NOT use ``ophyd_async.epics.motor.Motor``: that class
assumes a full EPICS motor record (``.VAL``, ``.RBV``, ``.VELO``, ``.ACCL``,
...), which the scenario benchmark's soft IOC does not provide (it seeds
plain scalar PVs, matching what ``channel_read`` reads). ``EpicsMotor`` here
is a minimal setpoint/readback pair over two such scalar PVs instead.

Imports ophyd-async, so this module (like the rest of ``devices/``) lives
behind the optional ``osprey-framework[bluesky-bridge]`` extra — keep it out
of the bridge lifecycle core's import path (``app.py``, ``runs.py``,
``scanner.py``, ``security.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ophyd_async.core import (
    AsyncStatus,
    StandardReadable,
    wait_for_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from ._connect import connect_all

_READBACK_DEADBAND = 1e-9
"""Max ``abs(readback - demand)`` for ``EpicsMotor.set()`` to consider a move
settled. The sp-echo readback on this pythonSoftIOC VA is an exact software
copy of the setpoint (``records.py`` ``on_update=rb.set``), so this is a
float-noise bound, not a physical tolerance."""

_READBACK_SETTLE_TIMEOUT_S = 5.0
"""Bound on how long ``EpicsMotor.set()`` waits for ``readback`` to settle
after the setpoint put completes, before raising ``TimeoutError``."""


@dataclass(frozen=True)
class EpicsMotorSpec:
    """One ``EpicsMotor`` to build: its device-mapping name and backing PV(s).

    ``readback_pv`` defaults to ``setpoint_pv`` when omitted, for a PV that is
    both read and written (no separate readback record).
    """

    name: str
    setpoint_pv: str
    readback_pv: str | None = None


@dataclass(frozen=True)
class EpicsDetectorSpec:
    """One ``EpicsDetector`` to build: its device-mapping name and read-only PV."""

    name: str
    read_pv: str


class EpicsMotor(StandardReadable):
    """A settable/readable pair of Channel Access PVs, shaped as a bluesky motor.

    Not backed by an EPICS motor record — ``readback`` and ``setpoint`` are
    each a single scalar PV, named explicitly by the caller (see
    ``EpicsMotorSpec``), matching the soft-IOC PVs the scenario benchmark seeds.

    ``set()`` does not treat put-completion as set-then-read coherence: on
    this pythonSoftIOC VA, the setpoint's ``on_update`` dispatches the
    readback update on the dispatcher loop *after* the put completes, so a
    ``set()`` immediately followed by ``read()`` can observe a stale
    readback. To close that race, ``set()`` waits, after the put completes,
    until ``readback`` is within ``_READBACK_DEADBAND`` of the demanded
    value or ``_READBACK_SETTLE_TIMEOUT_S`` elapses — uniformly, for every
    plan, with no plan-level knob to disable it.

    When ``readback_pv`` is omitted (see ``EpicsMotorSpec``), ``readback``
    aliases the same PV as ``setpoint``: there is no independent readback to
    wait on, so the deadband wait matches on the first CA get of the PV just
    written and the settle wait degrades to put-completion only.
    """

    def __init__(self, setpoint_pv: str, readback_pv: str | None = None, name: str = "") -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.readback = epics_signal_r(float, readback_pv or setpoint_pv)
        self.setpoint = epics_signal_rw(float, setpoint_pv)
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        """Write ``value`` to the setpoint PV, then wait for ``readback`` to settle.

        Raises:
            TimeoutError: ``readback`` did not come within
                ``_READBACK_DEADBAND`` of ``value`` within
                ``_READBACK_SETTLE_TIMEOUT_S`` seconds. Propagates uncaught,
                through the ``AsyncStatus`` this method is wrapped in — never
                swallowed.
        """
        await self.setpoint.set(value)
        await wait_for_value(
            self.readback,
            lambda rb: abs(rb - value) <= _READBACK_DEADBAND,
            timeout=_READBACK_SETTLE_TIMEOUT_S,
        )


class EpicsDetector(StandardReadable):
    """A single read-only Channel Access PV, shaped as a bluesky detector.

    Deliberately trigger-less: no ``trigger()`` method. ``plans.py``'s
    ``bp.scan``/``bp.count``/``bp.grid_scan`` reach bluesky's
    ``trigger_and_read``, which only triggers objects satisfying the
    ``Triggerable`` protocol; a trigger-less ``EpicsDetector`` is not
    ``Triggerable``, so it falls through to a plain ``read()`` — the same CA
    get ``channel_read`` performs, with no acquire/settle step of its own.
    Settling lives entirely in ``EpicsMotor.set()`` (the readback-deadband
    wait above), never here: this class does not wait, retry, or trigger on
    read.
    """

    def __init__(self, read_pv: str, name: str = "") -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.value = epics_signal_r(float, read_pv)
        super().__init__(name=name)


async def build_devices(
    motors: Sequence[EpicsMotorSpec] = (),
    detectors: Sequence[EpicsDetectorSpec] = (),
) -> dict[str, Any]:
    """Build and connect a set of EPICS-backed motors/detectors, keyed by name.

    Matches the ``get_devices() -> dict[str, Any]`` shape ``plans.py``'s
    built-in plans (and any facility-injected plan, per ``plan_loader.py``)
    resolve device names against. PV names come entirely from ``motors``/
    ``detectors`` — this function has no hardcoded PV prefix or naming
    convention, so the caller (facility config, benchmark harness) decides
    which real IOC address each device connects to. Connection (and why it's
    an explicit ``connect()`` rather than ``init_devices()``) is handled by
    :func:`._connect.connect_all`.

    Args:
        motors: Specs for the ``EpicsMotor`` instances to build.
        detectors: Specs for the ``EpicsDetector`` instances to build.

    Returns:
        Mapping of device name to connected device instance.

    Raises:
        ophyd_async.core.NotConnectedError: If any PV fails to connect (e.g.
            the IOC is unreachable).
    """
    devices: dict[str, Any] = {}
    for motor_spec in motors:
        devices[motor_spec.name] = EpicsMotor(
            motor_spec.setpoint_pv, motor_spec.readback_pv, name=motor_spec.name
        )
    for detector_spec in detectors:
        devices[detector_spec.name] = EpicsDetector(detector_spec.read_pv, name=detector_spec.name)
    return await connect_all(devices)
