"""A pyAML device list whose bulk reads and writes are one OSPREY runtime call each.

A read collects the distinct addresses of every device and issues one
``read_channels``; a write issues one ``write_channels`` carrying every setpoint,
so the limits validator sees the whole batch and refuses it whole before anything
is sent. A write runs only inside a journaled guarded run, which journals every
address first; anywhere else it is refused and nothing is read or written. Connector
exceptions surface as the pyAML exceptions of :mod:`pyaml_cs_osprey.errors`. A
channel refused after the connector sent the rest of the batch surfaces as
:class:`~pyaml_cs_osprey.errors.OspreyWriteFailed` naming the channels sent.

Values cross this list in SI: each device's value is scaled by its own reference's
unit suffix after the read and before the write, so one list may mix suffixes that
share an SI word. The journal and the limits stay in the channels' native units.

An indexed member (``ADDR@i``) reads element ``i`` of its base address's waveform.
A batch reads each base address once however many indexed members share it, so a
BPM array served as one waveform costs one read. Indexed members are read-only: a
list holding one refuses every write.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from pyaml.common.exception import PyAMLException
from pyaml.control.deviceaccess import DeviceAccess
from pyaml.control.deviceaccesslist import DeviceAccessList

import osprey.runtime
from osprey.errors import ChannelWriteBlockedError, ChannelWriteFailedError
from osprey.runtime import journal as runtime_journal
from osprey.runtime.journal import guarded_write
from pyaml_cs_osprey.device import OUTSIDE_A_RUN, OspreyDevice, indexed_element, unknown_unit
from pyaml_cs_osprey.errors import (
    OspreyReadFailed,
    OspreyWriteFailed,
    OspreyWriteRefused,
    _write_reason,
    map_read_error,
    map_write_error,
)
from pyaml_cs_osprey.units import UnitError, factor, si_unit

__all__ = ["OspreyDeviceList"]

#: ``ChannelWriteBlockedError`` reasons that refuse every channel of a batch alike,
#: so a batch refused for one of them wrote nothing. Any other refusal of a
#: multi-channel batch (the control system's, a per-channel validation) comes back
#: after the connector sent the batch's other channels.
_WHOLE_BATCH_REFUSALS = frozenset({"WRITES_DISABLED", "LIMITS"})


class OspreyDeviceList(DeviceAccessList):
    """An ordered group of OSPREY devices exposed as a pyAML :class:`DeviceAccessList`.

    The list starts empty; pyAML's control system fills it with :meth:`add_devices`
    after creating it.
    """

    def __init__(self) -> None:
        self._items: list[DeviceAccess] = []
        self._iter_pos = 0

    def __repr__(self) -> str:
        return f"OspreyDeviceList({', '.join(repr(d) for d in self._items)})"

    # --- collection ------------------------------------------------------------

    def add_devices(self, devices: DeviceAccess | Sequence[DeviceAccess]) -> None:
        """Append one device, or several in the order they should be read."""
        if isinstance(devices, DeviceAccess):
            self._items.append(devices)
        else:
            self._items.extend(devices)

    def get_device_at(self, index: int) -> DeviceAccess:
        """Return the device at ``index``."""
        return self._items[index]

    def len(self) -> int:
        """Return the number of devices."""
        return len(self._items)

    # --- reads -----------------------------------------------------------------

    def get(self) -> npt.NDArray[np.float64]:
        """Read every device's setpoint address in one call, in list order, in SI.

        Raises:
            PyAMLException: A device's reference is write-only; nothing was read.
            OspreyReadFailed: Any channel produced no numeric value, or an indexed
                member's base value is a scalar or too short for its index.
        """
        self._require_readable()
        return self._read_members(readback=False) * self._factors()

    def readback(self) -> npt.NDArray[np.float64]:
        """Read every device's readback address in one call, in list order, in SI.

        Raises:
            PyAMLException: A device's reference is write-only; nothing was read.
            OspreyReadFailed: Any channel produced no numeric value, or an indexed
                member's base value is a scalar or too short for its index.
        """
        self._require_readable()
        return self._read_members(readback=True) * self._factors()

    def check_device_availability(self) -> bool:
        """Return whether one read of every setpoint address succeeds.

        A list holding a write-only reference is never read, so it reports ``False``.
        """
        if any(_is_write_only(d) for d in self._items):
            return False
        try:
            self.get()
        except OspreyReadFailed:
            return False
        return True

    # --- writes ----------------------------------------------------------------

    def set(self, value: Any, *, confirm: bool | None = None) -> None:
        """Write one setpoint per device, in list order, as one batch.

        Every address is journaled into every active guard level first. A value
        that violates any channel's limits refuses the whole batch before anything
        is sent. Devices sharing an address must be given the same value; that
        address is written once.

        Args:
            value: One setpoint per device, in SI; each is scaled to its device's
                native unit before the batch is built.
            confirm: Passed to ``write_channels`` when given; ``None`` lets each
                channel resolve its own confirm default.

        Raises:
            PyAMLException: A member's reference is indexed, so read-only; nothing
                was journaled or written.
            ValueError: ``value`` does not hold one entry per device, or one
                address is given two different values; nothing was written.
            OspreyReadFailed: A prior setpoint could not be read; nothing was written.
            OspreyWriteRefused: The batch was refused whole before anything was
                sent (limits, write gate, control-target change), or no journaled
                guarded run is open (:data:`OUTSIDE_A_RUN`, naming the batch's first
                address); nothing was written.
            OspreyWriteFailed: A write was attempted and not confirmed, or one
                channel of a multi-channel batch was refused after the connector
                sent the others. The message names the failing address and every
                other address of the batch as sent.
        """
        self._require_writable()
        values = np.asarray(value, dtype=float).ravel()
        if values.size != len(self._items):
            raise ValueError(f"expected {len(self._items)} setpoints, got {values.size}")
        batch: dict[str, float] = {}
        natives = values / self._factors()
        for address, v in zip([d.name() for d in self._items], natives, strict=True):
            setpoint = float(v)
            if address in batch and batch[address] != setpoint:
                raise ValueError(
                    f"address {address} given two setpoints: {batch[address]} and {setpoint}"
                )
            batch[address] = setpoint

        def map_error(exc: Exception) -> OspreyWriteRefused | OspreyWriteFailed:
            if isinstance(exc, ChannelWriteFailedError) or (
                isinstance(exc, ChannelWriteBlockedError)
                and len(batch) > 1
                and exc.reason not in _WHOLE_BATCH_REFUSALS
            ):
                return _batch_failure(exc, list(batch))
            return map_write_error(exc)

        try:
            guarded_write(
                list(batch),
                lambda **kwargs: osprey.runtime.write_channels(batch, **kwargs),
                map_error,
                map_read=map_read_error,
                confirm=confirm,
            )
        except OspreyWriteRefused:
            raise
        except runtime_journal.OspreyWriteRefused as exc:
            # The runtime's own refusal, raised before any read: no run is open.
            raise OspreyWriteRefused(OUTSIDE_A_RUN, next(iter(batch), None)) from exc

    def set_and_wait(self, value: Any) -> None:
        """Write the setpoints and confirm each channel holds its value."""
        self.set(value, confirm=True)

    # --- metadata --------------------------------------------------------------

    def unit(self) -> str:
        """Return the SI unit word every device shares, or ``''`` for an empty list.

        Devices whose native suffixes differ but scale to one SI word (mm and um)
        share that word.

        Raises:
            PyAMLException: The devices' SI unit words differ, or a suffix is unknown.
        """
        words = list(dict.fromkeys(_si_word(d) for d in self._items))
        if len(words) > 1:
            raise PyAMLException(
                f"device list mixes SI units {', '.join(repr(w) for w in words)}; "
                "it has no one unit"
            )
        return words[0] if words else ""

    def get_range(self) -> list[float | None]:
        """Return ``[min0, max0, min1, max1, ...]``, ``None`` for an open bound."""
        flat: list[float | None] = []
        for device in self._items:
            flat.extend(device.get_range())
        return flat

    # --- helpers ---------------------------------------------------------------

    def _require_readable(self) -> None:
        """Refuse a read when any device's reference is write-only.

        Raises:
            PyAMLException: A device's reference is write-only.
        """
        for device in self._items:
            if isinstance(device, OspreyDevice):
                device._require_readable()

    def _require_writable(self) -> None:
        """Refuse a write when any device's reference is indexed.

        Raises:
            PyAMLException: A device's reference is indexed, so read-only.
        """
        for device in self._items:
            if _index(device) is not None:
                raise PyAMLException(
                    f"indexed channel reference {device.reference.text!r} is read-only; "
                    "the list writes nothing"
                )

    def _read_members(self, *, readback: bool) -> npt.NDArray[np.float64]:
        """Read every member's address in one batch and return one raw value each.

        An indexed member's base address joins the batch only where it is not
        already there, and the member takes element ``index`` of that value; every
        other member keeps its own position in the batch.

        Raises:
            OspreyReadFailed: The batch read failed, or an indexed member's base
                value is a scalar or too short for its index.
        """
        addresses: list[str] = []
        slots: list[int] = []
        first: dict[str, int] = {}
        for device in self._items:
            address = device.measure_name() if readback else device.name()
            if _index(device) is not None and address in first:
                slots.append(first[address])
                continue
            first.setdefault(address, len(addresses))
            slots.append(len(addresses))
            addresses.append(address)
        if not addresses:
            return np.empty(0, dtype=float)
        try:
            raw = list(osprey.runtime.read_channels(addresses))
        except Exception as exc:
            raise map_read_error(exc, list(dict.fromkeys(addresses))) from exc
        out = np.empty(len(self._items), dtype=float)
        for position, (device, slot) in enumerate(zip(self._items, slots, strict=True)):
            index = _index(device)
            value = (
                raw[slot]
                if index is None
                else indexed_element(device.reference, raw[slot], addresses[slot])
            )
            try:
                out[position] = float(value)
            except Exception as exc:
                raise map_read_error(exc, [addresses[slot]]) from exc
        return out

    def _factors(self) -> npt.NDArray[np.float64]:
        """Each device's native-to-SI factor, in list order.

        Raises:
            PyAMLException: A device's unit suffix is unknown.
        """
        return np.array([_factor(d) for d in self._items], dtype=float)


def _index(device: DeviceAccess) -> int | None:
    """The element index of an indexed OSPREY device, else ``None``."""
    return device.reference.index if isinstance(device, OspreyDevice) else None


def _is_write_only(device: DeviceAccess) -> bool:
    """Whether ``device`` is an OSPREY device built from a write-only reference."""
    return isinstance(device, OspreyDevice) and device.reference.mode == "w"


def _factor(device: DeviceAccess) -> float:
    """The factor turning ``device``'s native value into SI (1 for a foreign device)."""
    if not isinstance(device, OspreyDevice):
        return 1.0
    try:
        return factor(device.reference.unit)
    except UnitError as exc:
        raise unknown_unit(device.reference, exc) from exc


def _si_word(device: DeviceAccess) -> str:
    """The SI unit word of ``device``'s values."""
    if not isinstance(device, OspreyDevice):
        return str(device.unit())
    try:
        return si_unit(device.reference.unit)
    except UnitError as exc:
        raise unknown_unit(device.reference, exc) from exc


def _batch_failure(
    exc: ChannelWriteFailedError | ChannelWriteBlockedError, addresses: list[str]
) -> OspreyWriteFailed:
    """Map a mid-batch failure or refusal, naming every other address as sent."""
    failing = exc.channel_address
    others = [a for a in addresses if a != failing]
    reason = _write_reason(exc)
    if others:
        reason = f"{reason}; also sent in this batch: {', '.join(others)}"
    result = OspreyWriteFailed(reason, failing)
    result.__cause__ = exc
    return result
