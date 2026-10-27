"""The bounded pool search-time provider calls run on.

Embedding a query (and finding the server that embeds it) blocks: an HTTP call,
and for a local server possibly a reachability walk with 2 s probes. Run on the
event loop, a blackholed host would stall every other request the loop serves.
Run on the loop's default executor, it would compete with the work that pool
already carries (qmd's client resolution, ``asyncio.to_thread`` calls) and a
few hung probes could starve it. So search-time provider calls get their own
small pool, a :class:`SearchCallPool` the search service owns and closes.

A pool is at most :data:`SEARCH_CALL_WORKERS` daemon threads per process, fed
by a queue: a call waiting for a worker is a queue item, never a blocked
thread. A call given up on (its await timed out or was cancelled) while it runs
keeps its worker until the provider answers, and is counted as held. When every
worker is held, a new call is refused at once with :class:`TimeoutError`, never
queued. A call given up on before a worker took it never runs. Workers are
daemon threads: one still running when the interpreter exits does not hold it
open, where a ``ThreadPoolExecutor`` would join every worker.

A pool's workers belong to the process that started them. A child made by
``fork`` inherits the parent's record of them, whose workers do not exist in
the child, so each process starts its own on first use.

Closing a pool refuses new calls, cancels the queued ones, lets idle workers
end and waits :data:`CLOSE_TIMEOUT_FACTOR` times :data:`QUERY_EMBEDDING_TIMEOUT_S`
for running calls to return; the workers still running then are named.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from osprey_connectors.logger import get_logger

T = TypeVar("T")

#: Search-time provider calls (query embedding, reachability) that run at once.
SEARCH_CALL_WORKERS = 4

#: Seconds a query-embedding call may take; the step that runs it gets more.
QUERY_EMBEDDING_TIMEOUT_S = 5.0

#: The close waits this many times :data:`QUERY_EMBEDDING_TIMEOUT_S` for
#: running calls. The longest search call is a reachability walk (half the
#: embedding timeout) followed by an embedding call (the whole of it).
CLOSE_TIMEOUT_FACTOR = 2

logger = get_logger("ariel")

_THREAD_NAME_PREFIX = "ariel-search-call"

_thread_numbers = itertools.count()


@dataclass(eq=False)
class _Item:
    """One queued call and where its outcome goes."""

    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future[Any]
    loop: asyncio.AbstractEventLoop
    state: str = "queued"  # queued -> running -> done; abandoned while running; cancelled


@dataclass(eq=False)
class _Pool:
    """One process's workers, their queue and the count of held workers."""

    items: queue.SimpleQueue[_Item | None] = field(default_factory=queue.SimpleQueue)
    workers: list[threading.Thread] = field(default_factory=list)
    held: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


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


def _cancel(item: _Item) -> None:
    """Cancel the item's future on its loop, unless it already settled."""
    with contextlib.suppress(RuntimeError):
        item.loop.call_soon_threadsafe(item.future.cancel)


def _work(pool: _Pool) -> None:
    """Run queued calls until the pool's close hands this worker its stop."""
    while True:
        item = pool.items.get()
        if item is None:
            return
        with pool.lock:
            if item.future.done() or item.state == "cancelled":
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
        with pool.lock:
            if item.state == "abandoned":
                pool.held -= 1
            item.state = "done"


def _join(workers: list[threading.Thread], timeout_s: float) -> list[threading.Thread]:
    """Wait up to ``timeout_s`` in all for ``workers`` to end; return those still alive."""
    deadline = time.monotonic() + timeout_s
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    return [worker for worker in workers if worker.is_alive()]


class SearchCallPool:
    """The search service's bounded pool for blocking search-time provider calls.

    Its workers start on first use in each process and end at :meth:`aclose`;
    a closed pool refuses every call.
    """

    def __init__(self) -> None:
        self._pools: dict[int, _Pool] = {}
        self._closed = False

    def _pool(self) -> _Pool:
        """Return this process's workers and queue, created on first use in each process."""
        pid = os.getpid()
        pool = self._pools.get(pid)
        if pool is None:
            pool = self._pools.setdefault(pid, _Pool())
        return pool

    async def run(
        self, fn: Callable[..., T], *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> T:
        """Run ``fn(*args, **kwargs)`` on a search-call worker and await its result.

        Args:
            fn: The blocking callable.
            *args: Positional arguments for ``fn``.
            timeout_s: Seconds to wait before giving up; ``None`` waits for the
                call. A running call given up on keeps its worker until it
                returns and its result is discarded; one given up on while
                queued never runs.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            What ``fn`` returned.

        Raises:
            TimeoutError: When ``timeout_s`` passes first, or at once, without
                running ``fn``, when every worker is held by a call given up on
                ("search calls saturated: N calls still running").
            RuntimeError: At once, without running ``fn``, when the pool is
                closed.
            asyncio.CancelledError: When the pool closes while the call is
                queued.
            Exception: Whatever ``fn`` raised.
        """
        loop = asyncio.get_running_loop()
        pool = self._pool()
        future: asyncio.Future[T] = loop.create_future()
        item = _Item(fn, args, kwargs, future, loop)
        with pool.lock:
            if self._closed:
                raise RuntimeError("search calls closed")
            if pool.held >= SEARCH_CALL_WORKERS:
                raise TimeoutError(f"search calls saturated: {pool.held} calls still running")
            pool.items.put(item)
            if len(pool.workers) < SEARCH_CALL_WORKERS:
                worker = threading.Thread(
                    target=_work,
                    args=(pool,),
                    name=f"{_THREAD_NAME_PREFIX}_{next(_thread_numbers)}",
                    daemon=True,
                )
                pool.workers.append(worker)
                worker.start()
        try:
            if timeout_s is None:
                return await future
            return await asyncio.wait_for(future, timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            with pool.lock:
                if item.state == "running":
                    item.state = "abandoned"
                    pool.held += 1
            raise

    async def aclose(self) -> list[str]:
        """Close the pool; a second call finds nothing left to close.

        Refuses new calls, cancels the queued ones, lets idle workers end and
        waits :data:`CLOSE_TIMEOUT_FACTOR` times :data:`QUERY_EMBEDDING_TIMEOUT_S`
        for the running calls to return.

        Returns:
            The names of the workers still running a call when the wait ended,
            sorted; each is also logged as a warning.
        """
        pool = self._pool()
        with pool.lock:
            self._closed = True
            while True:
                try:
                    item = pool.items.get_nowait()
                except queue.Empty:
                    break
                if item is not None and item.state == "queued":
                    item.state = "cancelled"
                    _cancel(item)
            workers = list(pool.workers)
            pool.workers.clear()
            for _ in workers:
                pool.items.put(None)
        timeout_s = CLOSE_TIMEOUT_FACTOR * QUERY_EMBEDDING_TIMEOUT_S
        survivors = sorted(
            worker.name for worker in await asyncio.to_thread(_join, workers, timeout_s)
        )
        for name in survivors:
            logger.warning("search call still running %g s after close: %s", timeout_s, name)
        return survivors
