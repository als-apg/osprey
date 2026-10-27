"""The one bounded pool search-time provider calls run on.

Embedding a query (and finding the server that embeds it) blocks: an HTTP call,
and for a local server possibly a reachability walk with 2 s probes. Run on the
event loop, a blackholed host would stall every other request the loop serves.
Run on the loop's default executor, it would compete with the work that pool
already carries (qmd's client resolution, ``asyncio.to_thread`` calls) and a
few hung probes could starve it. So search-time provider calls get their own
small pool, which belongs to the service that runs the search: the service
builds it, passes it to each search module that makes such a call, and closes
it when it closes.

The pool is at most :data:`SEARCH_CALL_WORKERS` daemon threads, fed by a queue:
a call waiting for a worker is a queue item, never a blocked thread. A call
given up on (its await timed out or was cancelled) while it runs keeps its
worker until the provider answers, and is counted as held. When every worker is
held, a new call is refused at once with :class:`TimeoutError`, never queued. A
call given up on before a worker took it never runs. Workers are never joined:
a daemon thread does not hold the interpreter open at exit, where a
``ThreadPoolExecutor`` would join every worker.

A child made by ``fork`` inherits the pool object, whose workers do not exist
in the child, so the pool starts its own workers in each process on first use.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

#: Search-time provider calls (query embedding, reachability) that run at once.
SEARCH_CALL_WORKERS = 4

#: Seconds a query-embedding call may take; the step that runs it gets more.
QUERY_EMBEDDING_TIMEOUT_S = 5.0

_thread_numbers = itertools.count()


@dataclass(eq=False)
class _Item:
    """One queued call and where its outcome goes."""

    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future[Any]
    loop: asyncio.AbstractEventLoop
    state: str = "queued"  # queued -> running -> done; abandoned while running


def _deliver(item: _Item, error: BaseException | None, result: Any = None) -> None:
    """Settle the item's future on its loop, unless it already settled."""
    future = item.future

    def _settle() -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    # The loop that awaited an abandoned call may be closed by the time the
    # call returns; there is nobody left to hand the outcome to.
    with contextlib.suppress(RuntimeError):
        item.loop.call_soon_threadsafe(_settle)


class SearchCallPool:
    """The search service's bounded pool for search-time provider calls.

    Args:
        workers: The most calls that run at once.
        thread_name_prefix: The prefix of each worker thread's name.
    """

    def __init__(
        self,
        workers: int = SEARCH_CALL_WORKERS,
        thread_name_prefix: str = "ariel-search-call",
    ) -> None:
        self._size = workers
        self._thread_name_prefix = thread_name_prefix
        self._closed = False
        self._start_process()

    def _start_process(self) -> None:
        """Give this process an empty queue, no workers and no held calls."""
        self._pid = os.getpid()
        self._items: queue.SimpleQueue[_Item | None] = queue.SimpleQueue()
        self._workers: list[threading.Thread] = []
        self._held = 0
        self._lock = threading.Lock()

    def _refusal(self) -> TimeoutError:
        """The error a call meets when the pool no longer admits it."""
        return TimeoutError("search calls closed: the search service has stopped")

    async def run(
        self, fn: Callable[..., T], *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> T:
        """Run ``fn(*args, **kwargs)`` on a worker and await its result.

        See :func:`run_search_call`.
        """
        if self._pid != os.getpid():
            self._start_process()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        item = _Item(fn, args, kwargs, future, loop)
        with self._lock:
            if self._closed:
                raise self._refusal()
            if self._held >= self._size:
                raise TimeoutError(f"search calls saturated: {self._held} calls still running")
            self._items.put(item)
            if len(self._workers) < self._size:
                worker = threading.Thread(
                    target=self._work,
                    name=f"{self._thread_name_prefix}_{next(_thread_numbers)}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()
        try:
            if timeout_s is None:
                return await future
            return await asyncio.wait_for(future, timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            with self._lock:
                if item.state == "running":
                    item.state = "abandoned"
                    self._held += 1
            raise

    def _work(self) -> None:
        """Run queued calls until the pool closes."""
        items = self._items
        while True:
            item = items.get()
            if item is None:
                return
            with self._lock:
                if item.future.done():
                    continue
                if self._closed:
                    item.state = "done"
                    _deliver(item, self._refusal())
                    continue
                item.state = "running"
            try:
                result = item.fn(*item.args, **item.kwargs)
            except BaseException as exc:
                error = (
                    RuntimeError("search call raised StopIteration")
                    if isinstance(exc, StopIteration)
                    else exc
                )
                _deliver(item, error)
            else:
                _deliver(item, None, result)
            with self._lock:
                if item.state == "abandoned":
                    self._held -= 1
                item.state = "done"

    def close(self) -> None:
        """Stop admitting calls and refuse every queued one.

        A running call keeps its worker until it returns; its result still
        reaches its caller. Workers end once the queue is empty and are never
        joined. Closing twice does nothing.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = len(self._workers)
        while True:
            try:
                item = self._items.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            with self._lock:
                if item.state != "queued":
                    continue
                item.state = "done"
            _deliver(item, self._refusal())
        for _ in range(workers):
            self._items.put(None)


async def run_search_call(
    pool: SearchCallPool,
    fn: Callable[..., T],
    *args: Any,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` on ``pool`` and await its result.

    Args:
        pool: The search service's search-call pool; never the loop's default
            executor.
        fn: The blocking callable.
        *args: Positional arguments for ``fn``.
        timeout_s: Seconds to wait before giving up; ``None`` waits for the
            call. A running call given up on keeps its worker until it returns
            and its result is discarded; one given up on while queued never runs.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        What ``fn`` returned.

    Raises:
        TimeoutError: When ``timeout_s`` passes first, or at once, without
            running ``fn``, when every worker is held by a call given up on
            ("search calls saturated: N calls still running") or the pool is
            closed ("search calls closed: …").
        Exception: Whatever ``fn`` raised.
    """
    return await pool.run(fn, *args, timeout_s=timeout_s, **kwargs)
