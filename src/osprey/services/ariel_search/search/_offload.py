"""The one bounded pool search-time provider calls run on.

Embedding a query (and finding the server that embeds it) blocks: an HTTP call,
and for a local server possibly a reachability walk with 2 s probes. Run on the
event loop, a blackholed host would stall every other request the loop serves.
Run on the loop's default executor, it would compete with the work that pool
already carries (qmd's client resolution, ``asyncio.to_thread`` calls) and a
few hung probes could starve it. So search-time provider calls get their own
small pool.

The pool is at most :data:`SEARCH_CALL_WORKERS` daemon threads per process, fed
by a queue: a call waiting for a worker is a queue item, never a blocked
thread. A call given up on (its await timed out or was cancelled) while it runs
keeps its worker until the provider answers, and is counted as held. When every
worker is held, a new call is refused at once with :class:`TimeoutError`, never
queued. A call given up on before a worker took it never runs. Workers live for
the process and are never joined: a daemon thread does not hold the interpreter
open at exit, where a ``ThreadPoolExecutor`` would join every worker.

The pool belongs to the process that created it. A child made by ``fork``
inherits the parent's pool record, whose workers do not exist in the child, so
each process creates its own on first use.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

#: Search-time provider calls (query embedding, reachability) that run at once.
SEARCH_CALL_WORKERS = 4

#: Seconds a query-embedding call may take; the step that runs it gets more.
QUERY_EMBEDDING_TIMEOUT_S = 5.0

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
    state: str = "queued"  # queued -> running -> done; abandoned while running


@dataclass(eq=False)
class _Pool:
    """One process's workers, their queue and the count of held workers."""

    items: queue.SimpleQueue[_Item] = field(default_factory=queue.SimpleQueue)
    workers: list[threading.Thread] = field(default_factory=list)
    held: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_pools_by_pid: dict[int, _Pool] = {}
_pools_lock = threading.Lock()


def _pool() -> _Pool:
    """Return this process's pool, created on first use in each process."""
    pid = os.getpid()
    pool = _pools_by_pid.get(pid)
    if pool is None:
        with _pools_lock:
            pool = _pools_by_pid.setdefault(pid, _Pool())
    return pool


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


def _work(pool: _Pool) -> None:
    """Run queued calls for the life of the process."""
    while True:
        item = pool.items.get()
        with pool.lock:
            if item.future.done():
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


async def run_search_call(
    fn: Callable[..., T], *args: Any, timeout_s: float | None = None, **kwargs: Any
) -> T:
    """Run ``fn(*args, **kwargs)`` on a search-call worker and await its result.

    Args:
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
            ("search calls saturated: N calls still running").
        Exception: Whatever ``fn`` raised.
    """
    loop = asyncio.get_running_loop()
    pool = _pool()
    future: asyncio.Future[T] = loop.create_future()
    item = _Item(fn, args, kwargs, future, loop)
    with pool.lock:
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
