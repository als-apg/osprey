"""Control-room-neutral device specs: plain dataclasses, no CA/ophyd imports.

These describe *what* a device is (a settable setpoint/readback pair, or a
read-only value) without naming the control system that backs it. Kept in
their own module, free of ``ophyd_async`` imports and any raw Channel Access
client library, so consumers that only need the shape (e.g. env parsing) can
import it without pulling in ``ophyd_async`` or the rest of the device stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettableSpec:
    """One settable device to build: its device-mapping name and backing PV(s).

    ``readback_pv`` defaults to ``setpoint_pv`` when omitted, for a PV that is
    both read and written (no separate readback record).
    """

    name: str
    setpoint_pv: str
    readback_pv: str | None = None


@dataclass(frozen=True)
class ReadableSpec:
    """One read-only device to build: its device-mapping name and read PV."""

    name: str
    read_pv: str
