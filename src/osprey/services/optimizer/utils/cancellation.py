"""Cancellation token for thread-safe optimization cancellation."""

from __future__ import annotations

import logging
import time
from typing import TypedDict

from redis import Redis

from osprey.services.optimizer.models.exceptions import CancellationError
from osprey.services.optimizer.utils import redis_client as _redis_client
from osprey.services.optimizer.utils.job_keys import cancellation_key

logger = logging.getLogger(__name__)


class CancellationState(TypedDict):
    """Type definition for the cancellation state stored in Redis."""

    cancelled: bool
    paused: bool
    waiting: bool
    poll_count: int


class CancellationToken:
    """Thread-safe cancellation token using Redis for cross-process coordination."""

    _redis_client: Redis[str]

    def __init__(
        self,
        poll_interval: float = 0.2,
        *,
        redis_client: Redis[str],
        redis_key: str,
    ) -> None:
        """Initialize cancellation token.

        Args:
            poll_interval: Sleep duration (seconds) when waiting while paused.
            redis_client: Redis client for cross-process state coordination.
            redis_key: Redis key to store the cancellation/pause state.
        """
        self._poll_interval = poll_interval
        self._redis_client = redis_client
        self._redis_key = redis_key

        self._ensure_redis_state()

    def _ensure_redis_state(self) -> None:
        client = self._redis_client
        if not client.exists(self._redis_key):
            client.hset(
                self._redis_key,
                mapping={
                    "cancelled": 0,
                    "paused": 0,
                    "waiting": 0,
                    "poll_count": 0,
                    "timestamp": time.time(),
                },
            )

    def _read_state(self) -> CancellationState:
        client = self._redis_client
        data: dict[str, str] = client.hgetall(self._redis_key)
        return {
            "cancelled": bool(int(data.get("cancelled", 0))),
            "paused": bool(int(data.get("paused", 0))),
            "waiting": bool(int(data.get("waiting", 0))),
            "poll_count": int(data.get("poll_count", 0)),
        }

    def _write_state(
        self,
        *,
        cancelled: bool | None = None,
        paused: bool | None = None,
        waiting: bool | None = None,
        poll_count: int | None = None,
    ) -> None:
        client = self._redis_client

        mapping: dict[str | bytes, float] = {"timestamp": time.time()}

        if cancelled is not None:
            mapping["cancelled"] = int(cancelled)
        if paused is not None:
            mapping["paused"] = int(paused)
        if waiting is not None:
            mapping["waiting"] = int(waiting)
        if poll_count is not None:
            mapping["poll_count"] = poll_count

        # HSET only updates the specified fields, leaving others untouched
        client.hset(self._redis_key, mapping=mapping)

    def cancel(self) -> None:
        """Mark the token as cancelled."""
        self._write_state(cancelled=True)

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        state = self._read_state()
        return state["cancelled"]

    def check_cancellation(self) -> None:
        """Check if cancelled and raise CancellationError if so."""
        if self.is_cancelled():
            raise CancellationError("Optimization was cancelled by user")

    def pause(self) -> None:
        """Mark the token as paused."""
        logger.debug("[CancellationToken] pause() called for key=%s", self._redis_key)
        self._write_state(paused=True)
        logger.debug("[CancellationToken] paused, verifying: is_paused=%s", self.is_paused())

    def resume(self) -> None:
        """Clear the paused flag."""
        self._write_state(paused=False)

    def is_paused(self) -> bool:
        """Check if pause has been requested."""
        state = self._read_state()
        return state["paused"]

    # Test-synchronization seam: is_waiting()/get_poll_count() expose the Redis
    # pause-loop state so tests can await pause/resume deterministically without
    # time.sleep (the no-sleeps-in-tests policy). Production never reads these.
    def is_waiting(self) -> bool:
        """Check if the worker is currently waiting in wait_if_paused()."""
        state = self._read_state()
        return state["waiting"]

    def get_poll_count(self) -> int:
        """Get the number of times the worker has polled while waiting."""
        state = self._read_state()
        return state["poll_count"]

    def wait_if_paused(self) -> None:
        """Block while paused, checking periodically for cancellation."""
        first_check = True
        was_paused = False
        poll_count = 0

        try:
            while True:
                state = self._read_state()
                cancelled = state["cancelled"]
                paused = state["paused"]

                if first_check or (paused and not was_paused):
                    logger.debug(
                        "[CancellationToken] wait_if_paused check key=%s cancelled=%s paused=%s",
                        self._redis_key,
                        cancelled,
                        paused,
                    )
                    first_check = False

                if cancelled:
                    raise CancellationError("Optimization was cancelled by user")

                if not paused:
                    if was_paused:
                        logger.debug("[CancellationToken] Resumed! key=%s", self._redis_key)
                    return

                if not was_paused:
                    logger.debug(
                        "[CancellationToken] Optimization is paused, waiting... key=%s",
                        self._redis_key,
                    )
                    self._write_state(waiting=True, poll_count=0)
                    poll_count = 0

                was_paused = True
                time.sleep(self._poll_interval)
                poll_count += 1
                self._write_state(poll_count=poll_count)
        finally:
            if was_paused:
                self._write_state(waiting=False)

    def reset(self) -> None:
        """Reset the cancellation token (for reuse)."""
        self._write_state(cancelled=False, paused=False, waiting=False, poll_count=0)


def get_cancellation_token(job_id: str, poll_interval: float = 0.2) -> CancellationToken:
    """Return a ``CancellationToken`` wired to the current process's Redis client.

    Consumers should prefer this factory over constructing ``CancellationToken``
    directly so Redis client acquisition stays an implementation detail.
    """
    return CancellationToken(
        poll_interval=poll_interval,
        redis_client=_redis_client.get_redis_client(binary=False),
        redis_key=cancellation_key(job_id),
    )
