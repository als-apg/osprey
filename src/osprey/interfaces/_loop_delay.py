"""A watch on an event loop: it says so when something held the loop.

Every request of an interface server, and of each panel application served on a
thread of the same process, waits while one piece of work holds the event loop
or the interpreter. :func:`watch_loop_delay` sleeps a fixed step and measures
how late it woke; a wake later than the threshold is logged as a warning with
the delay, so a stall that users feel as a frozen page has a line in the log
with its time and its length.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

__all__ = ["DEFAULT_INTERVAL_S", "DEFAULT_THRESHOLD_S", "watch_loop_delay"]

logger = logging.getLogger(__name__)

#: Seconds between two wakes of the watch.
DEFAULT_INTERVAL_S = 0.5
#: Seconds a wake may be late before it is logged.
DEFAULT_THRESHOLD_S = 0.5


async def watch_loop_delay(
    name: str,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    threshold_s: float = DEFAULT_THRESHOLD_S,
    clock: Callable[[], float] | None = None,
) -> None:
    """Log a warning each time the running loop woke this task late; runs until cancelled.

    Args:
        name: The server the loop belongs to, named in each warning.
        interval_s: Seconds slept between two measurements.
        threshold_s: Seconds a wake may be late before it is logged.
        clock: The monotonic clock read; the running loop's when ``None``.
    """
    now = clock or asyncio.get_running_loop().time
    while True:
        due = now() + interval_s
        await asyncio.sleep(interval_s)
        late_s = now() - due
        if late_s > threshold_s:
            logger.warning(
                "%s: the event loop was held for %.1f s; every request waited that long",
                name,
                late_s,
            )
