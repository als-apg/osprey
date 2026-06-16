"""Bounded concurrent dispatch pool with FIFO queuing and result capture."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("osprey.dispatch.pool")


class QueueFullError(Exception):
    """Raised when the dispatch queue exceeds max_queue_depth."""


class DispatchPool:
    """Manages bounded concurrent dispatch with FIFO overflow queuing."""

    def __init__(self, max_concurrent: int, max_queue_depth: int) -> None:
        self._max_concurrent = max_concurrent
        self._max_queue_depth = max_queue_depth
        self._queue: asyncio.Queue = asyncio.Queue()
        # _inflight tracks slots claimed (running + tasks created but not yet running)
        self._inflight = 0
        self._running = 0
        self._results: dict[str, dict[str, Any]] = {}

    async def submit(self, trigger_name: str, dispatch_fn: Callable) -> str:
        """Submit a dispatch function for execution.

        Returns a dispatch_id (UUID string). Runs immediately if capacity
        is available, otherwise enqueues in FIFO order. Raises QueueFullError
        if the queue exceeds max_queue_depth.
        """
        dispatch_id = str(uuid.uuid4())
        self._results[dispatch_id] = {"status": "pending", "trigger_name": trigger_name}

        if self._inflight < self._max_concurrent:
            # Capacity available — claim slot and schedule immediately
            self._inflight += 1
            asyncio.create_task(self._run(dispatch_id, trigger_name, dispatch_fn))
        else:
            # At capacity — check queue depth
            if self._queue.qsize() >= self._max_queue_depth:
                del self._results[dispatch_id]
                raise QueueFullError(
                    f"Queue depth {self._max_queue_depth} exceeded for trigger '{trigger_name}'"
                )
            await self._queue.put((dispatch_id, trigger_name, dispatch_fn))

        return dispatch_id

    async def _run(self, dispatch_id: str, trigger_name: str, dispatch_fn: Callable) -> None:
        """Run dispatch_fn, capture result, then drain one item from the queue."""
        self._running += 1
        try:
            result = await dispatch_fn()
            self._results[dispatch_id] = {
                "status": "completed",
                "result": result,
                "trigger_name": trigger_name,
            }
        except Exception as exc:
            logger.error("Dispatch %s failed: %s", dispatch_id, exc)
            self._results[dispatch_id] = {
                "status": "error",
                "error": str(exc),
                "trigger_name": trigger_name,
            }
        finally:
            self._running -= 1
            self._inflight -= 1
            # Drain next queued item if available
            await self._drain_next()

    async def _drain_next(self) -> None:
        """If queue is non-empty, pull the next item and schedule it."""
        try:
            dispatch_id, trigger_name, dispatch_fn = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        self._inflight += 1
        asyncio.create_task(self._run(dispatch_id, trigger_name, dispatch_fn))

    def get_result(self, dispatch_id: str) -> dict[str, Any] | None:
        """Return the result for a dispatch_id, or None if unknown."""
        return self._results.get(dispatch_id)

    def get_pool_status(self) -> dict:
        """Return current pool status."""
        return {
            "running": self._running,
            "max": self._max_concurrent,
            "queued": self._queue.qsize(),
        }
