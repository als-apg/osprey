"""EPICS Channel Access trigger source for the event dispatcher.

Manages one pyepics CA monitor per trigger, applies threshold/edge/cool-down/
first-read-suppression logic, and forwards threshold-crossing events to the
asyncio serving loop via ``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import epics

from osprey.dispatch.pool import QueueFullError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig

logger = logging.getLogger("osprey.dispatch.sources.epics_ca")


class _PvWatcher:
    """Monitor one PV and fire a callback when its value crosses a threshold.

    Runs on pyepics' CA thread; hands off to the asyncio serving loop via
    ``run_coroutine_threadsafe``.  Each instance is self-contained so that
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
        self._pv: epics.PV | None = None
        self._last_fire_ts: float | None = None  # None = has not fired yet
        self._last_value: float | None = None  # None = first read not yet seen

    def start(self) -> None:
        self._pv = epics.PV(self._pv_name, callback=self._on_change)

    def stop(self) -> None:
        if self._pv is not None:
            self._pv.clear_callbacks()
            self._pv.disconnect()
            self._pv = None

    # ------------------------------------------------------------------
    # Internals (run on pyepics CA thread)
    # ------------------------------------------------------------------

    def _detect_edge(self, prev: float, curr: float) -> bool:
        if self._edge == "rising":
            return prev < self._threshold <= curr
        if self._edge == "falling":
            return prev > self._threshold >= curr
        # "both"
        return (prev < self._threshold <= curr) or (prev > self._threshold >= curr)

    def _on_change(self, pvname: str | None = None, value: Any = None, **kw: Any) -> None:
        """pyepics CA-thread callback. Schedules the fire coroutine on the loop."""
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            # Non-numeric PV (enum string, waveform, …): nothing to threshold on.
            # Log and return rather than raise on the CA thread, where pyepics
            # swallows the exception and the watcher would silently die.
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

    def register_routes(self, mcp_app: FastMCP) -> None:
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
            watcher = _PvWatcher(trigger, fire_callback, loop)
            watcher.start()
            self._watchers.append(watcher)
        logger.info("EPICS CA source started with %d monitor(s)", len(self._watchers))

    async def stop(self) -> None:
        """Disconnect all CA monitors and release resources."""
        for watcher in self._watchers:
            watcher.stop()
        self._watchers = []
