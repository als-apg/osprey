"""A pyAML device backed by one OSPREY channel reference.

Every read and write goes through :mod:`osprey.runtime`, so limits, write gates and
the control-target check apply exactly as they do to any other OSPREY write. A write
runs only while a :func:`pyaml_cs_osprey.run_tool.run_tool` call is in progress in
the process, and the prior native setpoint is read first and recorded for the call's
result once the runtime sent or landed the write; outside a ``run_tool`` call, ``execute`` included, it is refused as
:data:`OUTSIDE_A_RUN` and nothing is read or written. Connector exceptions surface as
the pyAML exceptions of :mod:`pyaml_cs_osprey.errors`. The write-only reference
refuses reads; every other reference is read and written.

Values cross this device in SI: reads and ranges are scaled from the reference's unit
suffix to SI, and a set value is scaled back to that native unit before it is written.
The recorded value and ``channel_limits`` stay native.

An indexed reference (``ADDR@i``) reads its base address and returns element ``i`` of
the array value; it is read-only, so every write and range request is refused or open.

A setpoint whose facility record states a ``tolerance`` settles after a
``set_and_wait``, and after every ``set`` inside :func:`settling`: its readback
(the record's ``pair``) is polled until it is within the tolerance of the value
written, and a readback that is not there within :data:`SETTLE_TIMEOUT_S` fails
the write. A setpoint that states no tolerance is not read back.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any

import numpy as np
from pyaml.common.exception import PyAMLException
from pyaml.control.deviceaccess import DeviceAccess

import osprey.runtime
from pyaml_cs_osprey import _run, _settle
from pyaml_cs_osprey._run import OUTSIDE_A_RUN
from pyaml_cs_osprey.catalog import ChannelReference
from pyaml_cs_osprey.errors import (
    OspreyReadFailed,
    OspreyWriteFailed,
    map_read_error,
    map_write_error,
)
from pyaml_cs_osprey.units import UnitError, si_unit, to_native, to_si

__all__ = [
    "OUTSIDE_A_RUN",
    "SETTLE_TIMEOUT_S",
    "OspreyDevice",
    "indexed_element",
    "settle",
    "settling",
    "unknown_unit",
]

#: Seconds a readback has to come within its stated tolerance after a write.
SETTLE_TIMEOUT_S = 5.0

#: Seconds between two readback polls of one settle.
_SETTLE_POLL_S = 0.05

#: The settle loop's pause and clock; tests replace them.
_sleep: Callable[[float], None] = time.sleep
_clock: Callable[[], float] = time.monotonic

#: Whether every ``set`` in this context settles, as ``set_and_wait`` does.
_SETTLING: ContextVar[bool] = ContextVar("pyaml_cs_osprey_settling", default=False)


@contextlib.contextmanager
def settling() -> Iterator[None]:
    """Make every device ``set`` in this context settle after its write."""
    token = _SETTLING.set(True)
    try:
        yield
    finally:
        _SETTLING.reset(token)


def settle(demands: Mapping[str, float]) -> None:
    """Wait until each setpoint's readback is within its stated tolerance of its demand.

    Every readback is read in one ``read_channels`` call per poll. A setpoint whose
    facility record states no tolerance is not read.

    Args:
        demands: Setpoint address to the value written, in its native unit.

    Raises:
        OspreyWriteFailed: A readback was not within its tolerance
            :data:`SETTLE_TIMEOUT_S` seconds after the first poll; names the first
            such setpoint.
        OspreyReadFailed: A readback read failed or produced no numeric value.
    """
    bounds: dict[str, tuple[str, float]] = {}
    for address in demands:
        bound = _settle.settle_bound(address)
        if bound is not None:
            bounds[address] = bound
    if not bounds:
        return
    readbacks = list(dict.fromkeys(pair for pair, _ in bounds.values()))
    deadline = _clock() + SETTLE_TIMEOUT_S
    while True:
        readings = _read_readbacks(readbacks)
        unsettled = [
            address
            for address, (pair, bound) in bounds.items()
            if not abs(readings[pair] - float(demands[address])) <= bound
        ]
        if not unsettled:
            return
        if _clock() >= deadline:
            address = unsettled[0]
            pair, tolerance = bounds[address]
            raise OspreyWriteFailed(
                f"readback {pair} not within {tolerance:g} of {float(demands[address]):g} "
                f"after {SETTLE_TIMEOUT_S:g} s; last read {readings[pair]:g}",
                address,
            )
        _sleep(_SETTLE_POLL_S)


def _read_readbacks(addresses: list[str]) -> dict[str, float]:
    """Read ``addresses`` in one call, each as a float.

    Raises:
        OspreyReadFailed: The read failed or a value is not numeric.
    """
    try:
        raw = list(osprey.runtime.read_channels(addresses))
    except Exception as exc:
        raise map_read_error(exc, addresses) from exc
    readings: dict[str, float] = {}
    for address, value in zip(addresses, raw, strict=True):
        try:
            readings[address] = float(value)
        except Exception as exc:
            raise map_read_error(exc, [address]) from exc
    return readings


class OspreyDevice(DeviceAccess):
    """One scalar channel reference exposed as a pyAML :class:`DeviceAccess`.

    Args:
        reference: The parsed reference. An indexed reference (``ADDR@i``) is
            read-only: reads return element ``i`` of ``ADDR`` and ``set()`` raises.
    """

    def __init__(self, reference: ChannelReference) -> None:
        self._reference = reference

    @property
    def reference(self) -> ChannelReference:
        """The channel reference this device was built from."""
        return self._reference

    def __repr__(self) -> str:
        return self._reference.text

    def name(self) -> str:
        """Return the setpoint address (a bare reference's one address)."""
        return self._reference.address

    def measure_name(self) -> str:
        """Return the readback address, or :meth:`name` when the reference has none."""
        return self._reference.readback or self._reference.address

    def unit(self) -> str:
        """Return the SI unit word of the reference's suffix (``''`` when absent).

        Raises:
            PyAMLException: The suffix has no known SI factor.
        """
        try:
            return si_unit(self._reference.unit)
        except UnitError as exc:
            raise self._unit_error(exc) from exc

    def get(self) -> float:
        """Read the setpoint address (a bare reference's one address), in SI.

        Raises:
            PyAMLException: The reference is write-only; nothing was read.
            OspreyReadFailed: The read raised or produced no numeric value.
        """
        self._require_readable()
        return self._to_si(self._read_value(self._reference.address))

    def readback(self) -> float:
        """Read the readback address, falling back to the setpoint address, in SI.

        Raises:
            PyAMLException: The reference is write-only; nothing was read.
            OspreyReadFailed: The read raised or produced no numeric value.
        """
        self._require_readable()
        return self._to_si(self._read_value(self.measure_name()))

    def set(self, value: Any, *, confirm: bool | None = None) -> None:
        """Write the SI ``value`` to the setpoint address in the reference's native unit.

        The prior native setpoint is read first and recorded for the call's result
        once the runtime sent or landed the write; a refused write records nothing. Outside
        a ``run_tool`` call the write is refused as :data:`OUTSIDE_A_RUN` and
        nothing is read. Inside :func:`settling` the write then settles as in
        :meth:`set_and_wait`.

        Args:
            value: The value to write, in SI.
            confirm: Passed to ``write_channel`` when given; ``None`` lets the
                channel resolve its own confirm default.

        Raises:
            PyAMLException: The reference is indexed (read-only), or the suffix has no
                known SI factor; nothing was recorded or written.
            OspreyReadFailed: The prior setpoint could not be read; nothing was written.
            OspreyWriteRefused: The write was refused, or no ``run_tool`` call is in
                progress; nothing was written.
            OspreyWriteFailed: The write was attempted and not confirmed, or inside
                :func:`settling` its readback did not settle.
        """
        self._write(value, confirm=confirm, settles=_SETTLING.get())

    def set_and_wait(self, value: Any) -> None:
        """Write ``value``, confirm the channel holds it, and wait for its readback.

        The write is ``set(value, confirm=True)``. When the setpoint's facility
        record states a ``tolerance``, its readback is then polled until it is
        within that tolerance of the value written (:func:`settle`).

        Raises:
            OspreyWriteFailed: The write was not confirmed, or the readback was not
                within its tolerance after :data:`SETTLE_TIMEOUT_S`; the write
                stays recorded as moved.
            OspreyReadFailed: The prior setpoint or the readback could not be read.
        """
        self._write(value, confirm=True, settles=True)

    def _write(self, value: Any, *, confirm: bool | None, settles: bool) -> None:
        """Write the SI ``value`` as :meth:`set` does, settling it when ``settles``."""
        self._require_writable()
        address = self._reference.address
        native = self._to_native(value)
        _run.write(
            [address],
            lambda **kwargs: osprey.runtime.write_channel(address, native, **kwargs),
            lambda exc: map_write_error(exc, address),
            map_read=map_read_error,
            confirm=confirm,
        )
        if settles:
            settle({address: native})

    def get_range(self) -> list[float | None]:
        """Return ``[min, max]`` in SI from the channel's limits, ``None`` for an open bound.

        An indexed reference is never written, so its range is open: ``[None, None]``.
        """
        if self._reference.index is not None:
            return [None, None]
        limits = osprey.runtime.channel_limits(self._reference.address)
        if limits is None:
            return [None, None]
        return [
            None if bound is None else self._to_si(bound)
            for bound in (limits.min_value, limits.max_value)
        ]

    def check_device_availability(self) -> bool:
        """Return whether a read of :meth:`get`'s address succeeds.

        A write-only reference is never read, so it reports ``False``.
        """
        if self._reference.mode == "w":
            return False
        try:
            self.get()
        except OspreyReadFailed:
            return False
        return True

    def _require_readable(self) -> None:
        """Refuse a read of a write-only reference.

        Raises:
            PyAMLException: The reference is write-only.
        """
        if self._reference.mode == "w":
            raise PyAMLException(
                f"write-only channel reference {self._reference.text!r} cannot be read"
            )

    def _require_writable(self) -> None:
        """Refuse a write through an indexed (read-only) reference.

        Raises:
            PyAMLException: The reference indexes an array element.
        """
        if self._reference.index is not None:
            raise PyAMLException(
                f"indexed channel reference {self._reference.text!r} is read-only; it cannot be set"
            )

    def _read_value(self, address: str) -> float:
        """Read ``address``: the scalar itself, or element ``index`` of its array.

        Raises:
            OspreyReadFailed: The read raised or produced no numeric value, or an indexed
                reference read a scalar or an array too short for its index.
        """
        if self._reference.index is None:
            return self._read(address)
        return indexed_element(self._reference, self._read_raw(address), address)

    def _to_si(self, value: float) -> float:
        try:
            return to_si(value, self._reference)
        except UnitError as exc:
            raise self._unit_error(exc) from exc

    def _to_native(self, value: Any) -> Any:
        try:
            return to_native(value, self._reference)
        except UnitError as exc:
            raise self._unit_error(exc) from exc

    def _unit_error(self, exc: UnitError) -> PyAMLException:
        return unknown_unit(self._reference, exc)

    @classmethod
    def _read(cls, address: str) -> float:
        value = cls._read_raw(address)
        try:
            return float(value)
        except Exception as exc:
            raise map_read_error(exc, [address]) from exc

    @staticmethod
    def _read_raw(address: str) -> Any:
        try:
            value = osprey.runtime.read_channel(address)
        except Exception as exc:
            raise map_read_error(exc, [address]) from exc
        if value is None:
            raise OspreyReadFailed("no value returned", [address])
        return value


def indexed_element(reference: ChannelReference, value: Any, address: str) -> float:
    """Element ``reference.index`` of the array ``value`` read from ``address``.

    Args:
        reference: An indexed reference (``ADDR@i``).
        value: What the read of its base address returned.
        address: The base address, named by any refusal.

    Raises:
        OspreyReadFailed: ``value`` is not numeric, is a scalar or not
            one-dimensional, or is too short for the index.
    """
    index = reference.index
    text = reference.text
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise OspreyReadFailed(
            f"indexed reference {text!r} needs a numeric array value", [address]
        ) from exc
    if array.ndim == 0:
        raise OspreyReadFailed(
            f"indexed reference {text!r} read a scalar, not an array (no length)",
            [address],
        )
    if array.ndim != 1:
        raise OspreyReadFailed(
            f"indexed reference {text!r} needs a one-dimensional array value, "
            f"got {array.ndim} dimensions",
            [address],
        )
    if index is None or index >= array.shape[0]:
        raise OspreyReadFailed(
            f"index {index} of reference {text!r} is out of range for an array of "
            f"length {array.shape[0]}",
            [address],
        )
    return float(array[index])


def unknown_unit(reference: ChannelReference, exc: UnitError) -> PyAMLException:
    """The pyAML exception for ``reference``'s suffix having no known SI factor."""
    return PyAMLException(
        f"channel reference {reference.text!r} has unknown unit suffix {exc.suffix!r}"
    )
