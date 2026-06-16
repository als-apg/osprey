"""Cron trigger source for the event dispatcher.

For each cron trigger, spawns an asyncio task that sleeps for the configured
interval and then fires the trigger via the dispatcher's fire callback.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from osprey.dispatch.pool import QueueFullError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig

logger = logging.getLogger("osprey.dispatch.sources.cron")


class CronSource:
    """Event source that fires triggers on a fixed time interval."""

    source_type: ClassVar[str] = "cron"

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task] = []

    def register_routes(self, mcp_app: FastMCP) -> None:
        """Cron has no HTTP routes."""
        return None

    async def start(self, triggers: list[TriggerConfig], fire_callback: FireCallback) -> None:
        """Spawn one polling task per cron trigger with a valid interval."""
        for trigger in triggers:
            interval = trigger.source_config.get("interval_sec")
            if (
                not isinstance(interval, (int, float))
                or isinstance(interval, bool)
                or interval <= 0
            ):
                logger.warning(
                    "Cron trigger '%s' has invalid 'interval_sec' (%r); skipping",
                    trigger.name,
                    interval,
                )
                continue
            task = asyncio.create_task(self._run_loop(trigger, float(interval), fire_callback))
            self._tasks.append(task)
        logger.info("Cron source started with %d task(s)", len(self._tasks))

    async def _run_loop(
        self, trigger: TriggerConfig, interval_sec: float, fire_callback: FireCallback
    ) -> None:
        """Sleep for the interval, fire the trigger, repeat until cancelled."""
        while True:
            await asyncio.sleep(interval_sec)
            payload = {
                "source": "cron",
                "trigger": trigger.name,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            try:
                await fire_callback(trigger, payload)
            except QueueFullError as exc:
                logger.warning("Cron trigger '%s' dropped: queue full (%s)", trigger.name, exc)
            except Exception:
                logger.exception("Cron trigger '%s' fire failed", trigger.name)

    async def stop(self) -> None:
        """Cancel all polling tasks and wait for them to settle."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
