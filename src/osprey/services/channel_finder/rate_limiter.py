"""Shared sliding-window rate limiter for LiteLLM call throttling."""

from __future__ import annotations

import asyncio
import collections
import logging
import time

logger = logging.getLogger(__name__)


class _SlidingWindowLimiter:
    """Allow at most ``max_calls`` LiteLLM calls per ``window`` seconds.

    Tracks call timestamps in a deque; ``acquire()`` sleeps until the oldest
    timestamp is outside the window if the bucket is full. Designed for the
    CBORG free-tier 20 req/min cap — we set 18 to leave a small safety margin.
    """

    def __init__(self, max_calls: int, window: float) -> None:
        self.max_calls = max_calls
        self.window = window
        self._calls: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # Drop timestamps outside the window
            while self._calls and now - self._calls[0] >= self.window:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                wait = self.window - (now - self._calls[0]) + 0.05
                logger.info(
                    "Rate limit reached (%d/%d in %.0fs window); sleeping %.1fs",
                    len(self._calls),
                    self.max_calls,
                    self.window,
                    wait,
                )
                await asyncio.sleep(wait)
                # Re-purge after sleep
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


# Module-level singleton, lazily configured by ``configure_rate_limiter`` or
# left as None to disable throttling. The ReAct backend wires this on init.
_RATE_LIMITER: _SlidingWindowLimiter | None = None


def configure_rate_limiter(max_calls: int | None, window: float = 60.0) -> None:
    """Install (or remove) a global rate limiter for LiteLLM calls.

    Pass ``max_calls=None`` to disable throttling. Safe to call repeatedly.
    """
    global _RATE_LIMITER
    if max_calls is None or max_calls <= 0:
        _RATE_LIMITER = None
    else:
        _RATE_LIMITER = _SlidingWindowLimiter(max_calls, window)
        logger.info("Rate limiter armed: %d calls / %.0fs", max_calls, window)


def get_rate_limiter() -> _SlidingWindowLimiter | None:
    return _RATE_LIMITER
