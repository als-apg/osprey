"""EPICS Channel Access trigger source for the event dispatcher.

Manages one Channel Access monitor per trigger (a pvapy ``pvaccess.Channel``
opened on the CA provider), applies threshold/edge/cool-down/first-read-
suppression logic, and forwards threshold-crossing events to the asyncio
serving loop via ``run_coroutine_threadsafe``.

pvapy reads the ``EPICS_CA_*`` environment when the first CA channel of the
process is created, not at import, and the settings are then fixed for the
process's lifetime: the dispatcher's environment decides which IOCs every
trigger can reach.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import pvaccess

from osprey.dispatch.pool import QueueFullError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig

logger = logging.getLogger("osprey.dispatch.sources.epics_ca")

#: The three edges ``_PvWatcher._detect_edge`` implements. An unrecognized
#: spelling used to fall through to the widest of them, so ``edge: up`` armed a
#: trigger that fired on both crossings — the opposite of what it asked for, and
#: silently.
VALID_EDGES = frozenset({"rising", "falling", "both"})

#: The fields each monitor asks for. Only ``value`` is thresholded; ``alarm``
#: and ``timeStamp`` are the rest of a CA monitor event and are kept so a debug
#: session inspecting the raw update sees the whole of it.
_MONITOR_REQUEST = "field(value,alarm,timeStamp)"


def _monitor_value(update: Any) -> Any:
    """The scalar a monitor update carries, in the form pyepics used to hand over.

    pvapy delivers a ``PvObject``; its ``value`` is a plain Python scalar for a
    numeric or string record, and for an enum (``bi``/``mbbi``/``bo``/``mbbo``)
    a ``{"index": …, "choices": […]}`` structure. The enum is reduced to its
    index — the number pyepics passed as ``value`` — so a threshold on a
    state PV keeps meaning what it meant before the client changed.

    A one-element array is reduced to its element for the same reason: pyepics
    handed a waveform of one element over as that scalar, so a threshold on
    one kept working. Longer arrays stay arrays and are ignored as non-numeric.
    """
    value = update.toDict().get("value")
    if isinstance(value, dict) and "index" in value:
        return value["index"]
    if not isinstance(value, str | bytes | dict):
        try:
            if len(value) == 1:
                return value[0]
        except TypeError:  # a scalar has no length
            pass
    return value


class _PvWatcher:
    """Monitor one PV and fire a callback when its value crosses a threshold.

    Runs on a pvapy monitor worker thread; hands off to the asyncio serving
    loop via ``run_coroutine_threadsafe``.  Each instance is self-contained so that
    multiple PVs are independent: a cool-down or first-read on one does not
    affect the others.
    """

    def __init__(
        self,
        trigger: TriggerConfig,
        fire_callback: FireCallback,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        cfg = trigger.source_config
        self._pv_name: str = cfg["pv"]
        self._threshold: float = float(cfg.get("threshold", 0.0))
        self._edge: str = cfg.get("edge", "rising")  # rising | falling | both
        self._cool_down: float = float(cfg.get("cool_down_sec", 60.0))
        self._trigger = trigger
        self._fire = fire_callback
        self._loop = loop
        self._channel: pvaccess.Channel | None = None
        # One subscriber per watcher; the name only has to be unique on its
        # channel, and each watcher opens its own channel.
        self._subscriber = f"osprey-dispatch-{trigger.name}"
        self._last_fire_ts: float | None = None  # None = has not fired yet
        self._last_value: float | None = None  # None = first read not yet seen

    def start(self) -> None:
        """Open the channel and start its monitor.

        Neither call waits for the IOC: an unreachable PV arms a monitor that
        begins delivering when the channel connects, so a missing IOC cannot
        stall the dispatcher's lifespan startup. The first update after a
        (re)connect carries the current value.
        """
        channel = pvaccess.Channel(self._pv_name, pvaccess.CA)
        channel.subscribe(self._subscriber, self._on_update)
        channel.startMonitor(_MONITOR_REQUEST)
        self._channel = channel

    def stop(self) -> None:
        if self._channel is not None:
            # stopMonitor first: it takes effect at once, so no update arrives
            # for a subscriber that is about to go away.
            self._channel.stopMonitor()
            self._channel.unsubscribe(self._subscriber)
            self._channel = None

    # ------------------------------------------------------------------
    # Internals (run on the pvapy monitor thread)
    # ------------------------------------------------------------------

    def _detect_edge(self, prev: float, curr: float) -> bool:
        """Whether the step from *prev* to *curr* is the crossing this edge names.

        Only the three edges in :data:`VALID_EDGES` reach here — ``start``
        refuses anything else — so the final branch is ``both``, not a
        catch-all for whatever the config said.
        """
        if self._edge == "rising":
            return prev < self._threshold <= curr
        if self._edge == "falling":
            return prev > self._threshold >= curr
        # "both"
        return (prev < self._threshold <= curr) or (prev > self._threshold >= curr)

    def _on_update(self, update: Any) -> None:
        """pvapy monitor callback. Schedules the fire coroutine on the loop."""
        self._on_change(_monitor_value(update))

    def _on_change(self, value: Any) -> None:
        """Threshold one monitored value; runs on the pvapy monitor thread."""
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            # Non-numeric PV (string record, waveform, …): nothing to threshold
            # on. Log and return rather than raise: pvapy logs
            # an exception from a monitor callback and carries on, but a named
            # warning says which trigger and PV it was.
            logger.warning(
                "EPICS CA trigger '%s' PV '%s' produced non-numeric value %r; ignoring",
                self._trigger.name,
                self._pv_name,
                value,
            )
            return
        if self._last_value is None:
            # Suppress fire on first (connect-time) read — just record the value.
            self._last_value = numeric
            return
        prev, self._last_value = self._last_value, numeric
        if not self._detect_edge(prev, numeric):
            return
        now = time.monotonic()
        # Cool-down only applies between actual fires — never relative to the
        # monotonic epoch, or a large cool_down_sec would suppress the first
        # event whenever monotonic() < cool_down_sec (e.g. a freshly booted host).
        if self._last_fire_ts is not None and now - self._last_fire_ts < self._cool_down:
            return
        self._last_fire_ts = now
        payload = {
            "source": "epics_ca",
            "pv": self._pv_name,
            "value": numeric,
            "previous_value": prev,
            "threshold": self._threshold,
            "edge": self._edge,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        asyncio.run_coroutine_threadsafe(self._dispatch(payload), self._loop)

    async def _dispatch(self, payload: dict[str, Any]) -> None:
        """Coroutine executed on the asyncio loop after an edge is detected."""
        try:
            await self._fire(self._trigger, payload)
        except QueueFullError as exc:
            logger.warning(
                "EPICS CA trigger '%s' dropped: queue full (%s)", self._trigger.name, exc
            )
        except Exception:
            logger.exception("EPICS CA trigger '%s' fire failed", self._trigger.name)


class EpicsCaSource:
    """Event source that fires triggers on EPICS CA PV threshold crossings.

    One CA monitor is armed per trigger in :meth:`start`; all are released in
    :meth:`stop`. Threshold, edge direction, cool-down, and first-read
    suppression are per-trigger and implemented in :class:`_PvWatcher`.

    Expected ``source_config`` keys per trigger:

    * ``pv`` (required): EPICS PV name to monitor.
    * ``threshold`` (default ``0.0``): crossing value.
    * ``edge`` (default ``"rising"``): ``"rising"``, ``"falling"``, or ``"both"``.
    * ``cool_down_sec`` (default ``60.0``): minimum seconds between fires for the same trigger.
    """

    source_type: ClassVar[str] = "epics_ca"

    def __init__(self) -> None:
        self._watchers: list[_PvWatcher] = []

    def register_routes(self, mcp_app: FastMCP) -> None:  # noqa: ARG002 - trigger-source lifecycle signature; a source with no routes registers nothing
        """EPICS CA source has no HTTP routes."""
        return None

    async def start(self, triggers: list[TriggerConfig], fire_callback: FireCallback) -> None:
        """Arm one CA monitor per trigger (lifespan startup, running loop)."""
        loop = asyncio.get_running_loop()
        for trigger in triggers:
            if not trigger.source_config.get("pv"):
                logger.warning(
                    "EPICS CA trigger '%s' has no 'pv' in source_config; skipping",
                    trigger.name,
                )
                continue
            edge = trigger.source_config.get("edge", "rising")
            if edge not in VALID_EDGES:
                # Skip this one trigger, arm the rest: one mistyped edge must
                # not take a dispatcher's whole trigger set down, and it must
                # not quietly become "both" either.
                logger.warning(
                    "EPICS CA trigger '%s' has unknown 'edge' (%r); expected one of %s; skipping",
                    trigger.name,
                    edge,
                    ", ".join(sorted(VALID_EDGES)),
                )
                continue
            watcher = _PvWatcher(trigger, fire_callback, loop)
            watcher.start()
            self._watchers.append(watcher)
        logger.info("EPICS CA source started with %d monitor(s)", len(self._watchers))

    async def stop(self) -> None:
        """Disconnect all CA monitors and release resources."""
        for watcher in self._watchers:
            watcher.stop()
        self._watchers = []
