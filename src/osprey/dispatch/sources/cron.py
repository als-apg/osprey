"""Cron trigger source for the event dispatcher.

A cron trigger fires on one of two schedules. With ``interval_sec`` it sleeps
the interval, fires, and repeats. With ``at`` (and optional ``days``) it fires
at those wall-clock times, read in the facility zone; a time that passes while
the dispatcher is not running, or not awake, is skipped and never made up.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, ClassVar

from osprey.dispatch.clock_schedule import ClockSchedule, next_fire, previous_fire
from osprey.dispatch.pool import QueueFullError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig

logger = logging.getLogger("osprey.dispatch.sources.cron")

# ``asyncio.sleep`` runs on the monotonic clock, which does not advance while a
# host is suspended, so a clock trigger sleeps in steps no longer than this and
# re-reads the wall clock after each; a step bounds how far the wall clock can
# move unseen.
_MAX_CLOCK_SLEEP_SEC = 60.0
# A wake later than this past its slot is a missed slot: it is skipped, never
# fired late.
_LATE_FIRE_GRACE_SEC = 60.0


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _zone_name(zone: tzinfo) -> str:
    return getattr(zone, "key", None) or str(zone)


class CronSource:
    """Event source that fires triggers on a fixed interval or at clock times.

    Args:
        now: Wall clock returning an aware UTC instant.
        zone: Zone a clock schedule is read in. ``None`` resolves the facility
            zone (``system.timezone``) when the source starts.
    """

    source_type: ClassVar[str] = "cron"

    def __init__(
        self, *, now: Callable[[], datetime] | None = None, zone: tzinfo | None = None
    ) -> None:
        self._tasks: list[asyncio.Task] = []
        self._now: Callable[[], datetime] = now or _utc_now
        self._zone = zone

    def register_routes(self, mcp_app: FastMCP) -> None:  # noqa: ARG002 - trigger-source lifecycle signature; a source with no routes registers nothing
        """Cron has no HTTP routes."""
        return None

    def _resolve_zone(self) -> tzinfo:
        if self._zone is None:
            from osprey.utils.config import get_facility_timezone

            self._zone = get_facility_timezone()
        return self._zone

    async def start(self, triggers: list[TriggerConfig], fire_callback: FireCallback) -> None:
        """Spawn one task per cron trigger with a clock schedule or a valid interval."""
        for trigger in triggers:
            if trigger.schedule is not None:
                zone = self._resolve_zone()
                now = self._now()
                logger.info(
                    "Cron trigger '%s' fires at clock times in %s; slot before start %s "
                    "(not made up); next fire %s",
                    trigger.name,
                    _zone_name(zone),
                    previous_fire(trigger.schedule, now, zone).astimezone(zone).isoformat(),
                    next_fire(trigger.schedule, now, zone).astimezone(zone).isoformat(),
                )
                task = asyncio.create_task(
                    self._run_clock_loop(trigger, trigger.schedule, zone, fire_callback)
                )
                self._tasks.append(task)
                continue
            interval = trigger.source_config.get("interval_sec")
            if (
                not isinstance(interval, (int, float))
                or isinstance(interval, bool)
                or interval <= 0
            ):
                logger.warning(
                    "Cron trigger '%s' has neither a valid 'interval_sec' (%r) nor 'at'; skipping",
                    trigger.name,
                    interval,
                )
                continue
            task = asyncio.create_task(self._run_loop(trigger, float(interval), fire_callback))
            self._tasks.append(task)
        logger.info("Cron source started with %d task(s)", len(self._tasks))

    async def _fire(self, trigger: TriggerConfig, fire_callback: FireCallback) -> None:
        """Fire the trigger once; a full queue drops the tick with a warning."""
        payload = {
            "source": "cron",
            "trigger": trigger.name,
            "timestamp": self._now().isoformat(),
        }
        try:
            await fire_callback(trigger, payload)
        except QueueFullError as exc:
            logger.warning("Cron trigger '%s' dropped: queue full (%s)", trigger.name, exc)
        except Exception:
            logger.exception("Cron trigger '%s' fire failed", trigger.name)

    async def _run_loop(
        self, trigger: TriggerConfig, interval_sec: float, fire_callback: FireCallback
    ) -> None:
        """Sleep for the interval, fire the trigger, repeat until cancelled."""
        while True:
            await asyncio.sleep(interval_sec)
            await self._fire(trigger, fire_callback)

    async def _run_clock_loop(
        self,
        trigger: TriggerConfig,
        schedule: ClockSchedule,
        zone: tzinfo,
        fire_callback: FireCallback,
    ) -> None:
        """Wait for each slot on the wall clock, fire it, repeat until cancelled."""
        target = next_fire(schedule, self._now(), zone)
        while True:
            while (now := self._now()) < target:
                remaining = (target - now).total_seconds()
                await asyncio.sleep(min(remaining, _MAX_CLOCK_SLEEP_SEC))
            late = (now - target).total_seconds()
            if late > _LATE_FIRE_GRACE_SEC:
                logger.warning(
                    "Cron trigger '%s' missed slot %s (woke %.0f s late); not made up; "
                    "next fire %s",
                    trigger.name,
                    target.astimezone(zone).isoformat(),
                    late,
                    next_fire(schedule, max(now, target), zone).astimezone(zone).isoformat(),
                )
            else:
                await self._fire(trigger, fire_callback)
            target = next_fire(schedule, max(self._now(), target), zone)

    async def stop(self) -> None:
        """Cancel all polling tasks and wait for them to settle."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
