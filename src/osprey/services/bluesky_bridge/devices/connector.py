"""ophyd-async device factory mediated entirely by the OSPREY connector.

Design reversal (R8): the sibling ``epics.py`` factory was built on an
explicit design ruling that ophyd-async already speaks Channel Access
directly, so no OSPREY connector was needed for the scan device layer.
Phase 4's complete-mediation mandate OVERRIDES that ruling: direct CA from
``epics.py`` is exactly the unmediated second read/write path that
mediation is closing. This module is the replacement device layer — every
scan read and every scan write, for every device built here, goes through
the OSPREY connector
(:class:`osprey.connectors.control_system.base.ControlSystemConnector`):
reads via ``connector.read_channel``, writes via
``connector.write_channel_checked``, which raises on any refused, failed,
or unverified write so a bad write aborts the RunEngine rather than
silently continuing a scan. There is no raw Channel Access client library,
no low-level EPICS signal backend, and no direct PV access anywhere in
this module.

Device-level delegation over the stable ophyd-async public API (design
decision D1): ``ConnectorSettable``/``ConnectorReadable`` are plain
``StandardReadable`` subclasses that call the connector from ``set()``/
``read()``/``describe()``. This is deliberately NOT a custom
``SignalBackend`` — the ophyd-async ``Signal``/backend protocol is an
internal extension point, while ``StandardReadable``'s ``set``/``read``/
``describe``/``connect`` are the stable public device contract that plans
and the RunEngine actually consume.

Imports ophyd-async, so this module (like the rest of ``devices/``) lives
behind the optional ``osprey-framework[bluesky-bridge]`` extra — keep it
out of the bridge lifecycle core's import path (``app.py``, ``runs.py``,
``plan_runner.py``, ``security.py``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from ophyd_async.core import AsyncStatus, StandardReadable

from ._connect import connect_all
from .specs import ReadableSpec, SettableSpec

_READBACK_DEADBAND = 1e-9
"""Max ``abs(readback - demand)`` for ``ConnectorSettable.set()`` to
consider a move settled. Mirrors ``epics.py``'s deadband: a float-noise
bound on a setpoint/readback pair the underlying IOC keeps in exact
software sync, not a physical tolerance."""

_READBACK_SETTLE_TIMEOUT_S = 5.0
"""Bound on how long ``ConnectorSettable.set()`` polls the readback channel
for settlement after the write verifies, before raising ``TimeoutError``.
Referenced by name (not bound as a default argument) inside ``set()`` so
its current module-level value is read at call time — this lets a test
monkeypatch the module attribute to a tiny value without touching the
class."""

_READBACK_POLL_INTERVAL_S = 0.05
"""Sleep between readback polls in ``ConnectorSettable.set()``."""


class ConnectorSettable(StandardReadable):
    """A settable/readable PV pair mediated entirely by the OSPREY connector.

    Replaces ``epics.EpicsMotor``'s direct Channel Access signals: there are
    no ophyd-async EPICS signals declared here at all. ``set()`` writes the
    setpoint through ``connector.write_channel_checked`` — which raises on
    any refusal, failure, or unverified write, aborting the RunEngine — then
    polls the (possibly separate) readback channel through
    ``connector.read_channel`` until it settles within ``_READBACK_DEADBAND``
    of the demanded value, or raises ``TimeoutError`` after
    ``_READBACK_SETTLE_TIMEOUT_S``. ``read()``/``describe()`` are overridden
    to return the *live* readback via the connector on every call — never a
    cached/soft value — so a plan's ``trigger_and_read`` document always
    reflects the current mediated state.

    When ``readback_pv`` is omitted, ``readback`` aliases ``setpoint_pv``:
    there is no independent readback to settle against, so the deadband
    wait matches on the first connector read of the PV just written.

    The OSPREY connector instance is stored as ``self._osprey_connector``,
    not ``self._connector``: ophyd-async's own ``Device.__init__`` already
    owns the ``self._connector`` attribute (its internal
    ``DeviceConnector``, used by ``Device.connect()``); reusing that name
    for the OSPREY connector would silently clobber ophyd-async's connect
    machinery.
    """

    def __init__(
        self,
        connector: Any,
        setpoint_pv: str,
        readback_pv: str | None = None,
        name: str = "",
    ) -> None:
        self._osprey_connector = connector
        self._setpoint_pv = setpoint_pv
        self._readback_pv = readback_pv or setpoint_pv
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        """Write ``value`` through the connector, then wait for readback settle.

        Raises:
            ChannelWriteBlockedError: The reference monitor refused the
                write (writes disabled, limits, validation) — the write was
                never attempted.
            ChannelWriteFailedError: The write was attempted but failed, or
                its callback verification did not succeed.
            ConnectionError: Propagated unchanged from the connector's
                Channel Access layer.
            TimeoutError: Either propagated unchanged from the connector's
                write, or raised directly by this method when ``readback``
                does not settle within ``_READBACK_DEADBAND`` of ``value``
                within ``_READBACK_SETTLE_TIMEOUT_S`` seconds.

            Every one of these propagates uncaught through the
            ``AsyncStatus`` this method is wrapped in, aborting the
            RunEngine — this is the entire safety point of routing writes
            through ``write_channel_checked`` instead of a bare
            ``write_channel``.
        """
        await self._osprey_connector.write_channel_checked(
            self._setpoint_pv, value, verification_level="callback"
        )

        deadline = time.monotonic() + _READBACK_SETTLE_TIMEOUT_S
        while True:
            reading = await self._osprey_connector.read_channel(self._readback_pv)
            if abs(reading.value - value) <= _READBACK_DEADBAND:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Readback '{self._readback_pv}' did not settle to {value} "
                    f"within {_READBACK_SETTLE_TIMEOUT_S}s (last read: {reading.value})"
                )
            await asyncio.sleep(_READBACK_POLL_INTERVAL_S)

    async def read(self) -> dict[str, dict[str, Any]]:
        """Return the live readback value, read fresh through the connector.

        Never returns a cached/soft value: every call issues a new
        ``connector.read_channel`` so the document a plan records reflects
        the current mediated state of the readback channel.
        """
        reading = await self._osprey_connector.read_channel(self._readback_pv)
        return {self.name: {"value": reading.value, "timestamp": time.time()}}

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the live readback channel as a scalar numeric data key."""
        return {
            self.name: {
                "source": f"connector:{self._readback_pv}",
                "dtype": "number",
                "shape": [],
            }
        }


class ConnectorReadable(StandardReadable):
    """A single read-only channel mediated entirely by the OSPREY connector.

    Replaces ``epics.EpicsDetector``: trigger-less (no ``trigger()``
    method), and every ``read()`` performs a fresh ``connector.read_channel``
    call rather than returning a cached/soft value — a soft signal would
    return a stale value, defeating the point of live mediation.

    The OSPREY connector instance is stored as ``self._osprey_connector``,
    for the same reason as :class:`ConnectorSettable`: ``self._connector``
    is already owned by ophyd-async's ``Device.__init__``.
    """

    def __init__(self, connector: Any, read_pv: str, name: str = "") -> None:
        self._osprey_connector = connector
        self._read_pv = read_pv
        super().__init__(name=name)

    async def read(self) -> dict[str, dict[str, Any]]:
        """Return the live value, read fresh through the connector."""
        reading = await self._osprey_connector.read_channel(self._read_pv)
        return {self.name: {"value": reading.value, "timestamp": time.time()}}

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the live channel as a scalar numeric data key."""
        return {
            self.name: {
                "source": f"connector:{self._read_pv}",
                "dtype": "number",
                "shape": [],
            }
        }


async def build_devices(
    settables: Sequence[SettableSpec] = (),
    readables: Sequence[ReadableSpec] = (),
    connector: Any = None,
) -> dict[str, Any]:
    """Build and connect connector-mediated settable/readable devices, keyed by name.

    Matches the ``get_devices() -> dict[str, Any]`` shape ``plans.py``'s
    built-in plans (and any facility-injected plan, per ``plan_loader.py``)
    resolve device names against — the same factory contract
    ``epics.build_devices``/``mock.build_devices`` provide. Connection (and
    why it's an explicit ``connect()`` rather than ``init_devices()``) is
    handled by :func:`._connect.connect_all`; a ``ConnectorSettable``/
    ``ConnectorReadable`` declares no ophyd-async signals, so this connects
    as a no-op per device, same as the other factories in this package.

    Args:
        settables: Specs for the ``ConnectorSettable`` instances to build.
        readables: Specs for the ``ConnectorReadable`` instances to build.
        connector: The OSPREY control-system connector every built device
            delegates its reads/writes to. Every device shares this same
            connector instance.

    Returns:
        Mapping of device name to connected device instance.
    """
    devices: dict[str, Any] = {}
    for settable_spec in settables:
        devices[settable_spec.name] = ConnectorSettable(
            connector,
            settable_spec.setpoint_pv,
            settable_spec.readback_pv,
            name=settable_spec.name,
        )
    for readable_spec in readables:
        devices[readable_spec.name] = ConnectorReadable(
            connector, readable_spec.read_pv, name=readable_spec.name
        )
    return await connect_all(devices)
