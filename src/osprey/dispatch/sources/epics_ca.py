"""EPICS Channel Access trigger source for the event dispatcher.

Watches one channel per trigger through the control-system connector's
``subscribe``, applies threshold/edge/cool-down/first-read-suppression logic,
and hands each threshold crossing to the dispatcher's fire callback on the
asyncio serving loop.

The connector is an ``epics`` one with no gateway of its own: it runs Channel
Access with the ``EPICS_CA_*`` environment the dispatcher was started with (a
deployment hands it over through ``dispatch.env``), on the process's one CA
loop. Going through the connector rather than aioca directly is what keeps a
trigger armed across a repoint of that loop: the connector re-opens its
monitors on the new context.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from osprey.dispatch.pool import QueueFullError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig
    from osprey_connectors.control_system.base import ChannelValue, ControlSystemConnector

logger = logging.getLogger("osprey.dispatch.sources.epics_ca")

#: The three edges ``_PvWatcher._detect_edge`` implements. An unrecognized
#: spelling used to fall through to the widest of them, so ``edge: up`` armed a
#: trigger that fired on both crossings — the opposite of what it asked for, and
#: silently.
VALID_EDGES = frozenset({"rising", "falling", "both"})


#: How long connecting, or arming one monitor, may take. Opening a monitor only
#: queues the channel search, so this bounds a stalled CA loop, not a slow IOC.
_OPEN_TIMEOUT_S = 5.0


async def _connect() -> ControlSystemConnector:
    """Build and connect the Channel Access connector the triggers watch through.

    Built by the connector factory as type ``epics`` with an empty settings
    block, rather than from the deployment's ``control_system`` section. The
    dispatcher is not a session and never switches target, and its gateway is
    the one the deployment hands it in ``EPICS_CA_*`` through ``dispatch.env``:
    a gateway block from the section would rewrite that environment, and could
    name an address only the agent's own network reaches. With no gateway
    configured, ``connect()`` leaves the environment as it stands.
    """
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()  # idempotent; the factory resolves "epics" through it
    return await ConnectorFactory.create_control_system_connector({"type": "epics"})


def _number_setting(
    cfg: Mapping[str, Any], key: str, default: float, *, non_negative: bool = False
) -> float:
    """Read one numeric ``source_config`` setting, or refuse it.

    A threshold or cool-down the watcher cannot compare against is refused,
    never coerced into a trigger that never fires or fires on ``1.0``. An
    absent or blank value takes ``default``; a numeric string reads as its
    number.

    Raises:
        ValueError: The value is a ``bool``, is not readable by ``float()``, is
            not finite, or is negative under ``non_negative``. The message reads
            ``<qualifier> '<key>' (<value>)``.
    """
    value = cfg.get(key)
    if value is None:
        return default
    # ``bool`` is an ``int`` subclass, so ``true`` would otherwise read as 1.0.
    if isinstance(value, bool):
        raise ValueError(f"non-numeric {key!r} ({value!r})")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric {key!r} ({value!r})") from None
    if not math.isfinite(number):
        raise ValueError(f"non-finite {key!r} ({value!r})")
    if non_negative and number < 0:
        raise ValueError(f"negative {key!r} ({value!r})")
    return number


class _PvWatcher:
    """Monitor one PV and fire a callback when its value crosses a threshold.

    The connector delivers each update on the asyncio serving loop, where the
    fire is scheduled as a task. Each instance is self-contained so that
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
        self._threshold: float = _number_setting(cfg, "threshold", 0.0)
        self._edge: str = cfg.get("edge", "rising")  # rising | falling | both
        self._cool_down: float = _number_setting(cfg, "cool_down_sec", 60.0, non_negative=True)
        self._trigger = trigger
        self._fire = fire_callback
        self._loop = loop
        self._connector: ControlSystemConnector | None = None
        self._subscription: str | None = None
        # Fires in flight, held so a task is not collected before it runs.
        self._dispatches: set[asyncio.Task[None]] = set()
        self._last_fire_ts: float | None = None  # None = has not fired yet
        self._last_value: float | None = None  # None = first read not yet seen

    async def start(self, connector: ControlSystemConnector) -> None:
        """Subscribe to the PV through ``connector``.

        Raises ``TimeoutError`` when the monitor does not open within
        :data:`_OPEN_TIMEOUT_S`. The subscribe is cancelled then, and the
        connector closes the monitor behind it whenever the CA loop gets to it,
        so none is left firing with no watcher.
        """
        self._subscription = await asyncio.wait_for(
            connector.subscribe(self._pv_name, self._on_change), _OPEN_TIMEOUT_S
        )
        self._connector = connector

    async def stop(self) -> None:
        if self._connector is not None and self._subscription is not None:
            await self._connector.unsubscribe(self._subscription)
        self._connector = None
        self._subscription = None

    # ------------------------------------------------------------------
    # Internals (run on the serving loop)
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

    def _on_change(self, update: ChannelValue) -> None:
        """Connector monitor callback. Schedules the fire coroutine on the serving loop."""
        value = update.value
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            # Non-numeric PV (string, waveform, …): nothing to threshold on.
            # Log and return rather than raise out of a loop callback, where
            # the exception would only be logged and the update lost unread.
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
            "timestamp": datetime.now(tz=UTC),
        }
        # The connector calls back on this loop, the one the watcher was armed
        # on; once that loop has closed it closes the monitor itself, so no
        # update arrives here after.
        task = self._loop.create_task(self._dispatch(payload))
        self._dispatches.add(task)
        task.add_done_callback(self._dispatches.discard)

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

    One monitor is armed per trigger in :meth:`start`, through one connector
    built for the source; all are released, and the connector disconnected, in
    :meth:`stop`. Threshold, edge direction, cool-down, and first-read
    suppression are per-trigger and implemented in :class:`_PvWatcher`.

    Expected ``source_config`` keys per trigger:

    * ``pv`` (required): EPICS PV name to monitor.
    * ``threshold`` (default ``0.0``): crossing value, a finite number.
    * ``edge`` (default ``"rising"``): ``"rising"``, ``"falling"``, or ``"both"``.
    * ``cool_down_sec`` (default ``60.0``): minimum seconds between fires for the same
      trigger, a number zero or greater.

    A trigger with an unusable ``threshold`` or ``cool_down_sec`` is skipped with a
    warning naming it; its siblings still arm.
    """

    source_type: ClassVar[str] = "epics_ca"

    def __init__(self) -> None:
        self._watchers: list[_PvWatcher] = []
        self._connector: ControlSystemConnector | None = None

    def register_routes(self, mcp_app: FastMCP) -> None:  # noqa: ARG002 - trigger-source lifecycle signature; a source with no routes registers nothing
        """EPICS CA source has no HTTP routes."""
        return None

    async def start(self, triggers: list[TriggerConfig], fire_callback: FireCallback) -> None:
        """Arm one CA monitor per trigger (lifespan startup, running loop).

        Every trigger is read first, and the connector is built only when at
        least one of them is usable: a dispatcher whose ``epics_ca`` triggers
        are all refused never touches Channel Access.
        """
        loop = asyncio.get_running_loop()
        watchers: list[_PvWatcher] = []
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
            try:
                watchers.append(_PvWatcher(trigger, fire_callback, loop))
            except ValueError as exc:
                logger.warning("EPICS CA trigger '%s' has %s; skipping", trigger.name, exc)
                continue
        if not watchers:
            logger.info("EPICS CA source started with 0 monitor(s)")
            return

        try:
            connector = await asyncio.wait_for(_connect(), _OPEN_TIMEOUT_S)
        except TimeoutError:
            logger.warning(
                "EPICS CA source: the CA loop did not connect within %ss; no trigger armed",
                _OPEN_TIMEOUT_S,
            )
            return
        self._connector = connector
        for watcher in watchers:
            try:
                await watcher.start(connector)
            except TimeoutError:
                logger.warning(
                    "EPICS CA trigger '%s': the CA loop did not open a monitor on '%s' "
                    "within %ss; skipping",
                    watcher._trigger.name,
                    watcher._pv_name,
                    _OPEN_TIMEOUT_S,
                )
                continue
            self._watchers.append(watcher)
        logger.info("EPICS CA source started with %d monitor(s)", len(self._watchers))

    async def stop(self) -> None:
        """Disconnect all CA monitors and release the connector."""
        for watcher in self._watchers:
            await watcher.stop()
        self._watchers = []
        connector, self._connector = self._connector, None
        if connector is not None:
            await connector.disconnect()
