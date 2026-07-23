"""Bridge synchronous callables onto the asyncio loop via daemon threads.

The health runner is async-native, but several checks wrap blocking synchronous
I/O (sync plugin callables, provider ``check_health``, ``get_chat_completion``).
Those must run off the event loop without ever wedging process exit when a check
hangs.

Why not :func:`asyncio.to_thread` or a stock :class:`concurrent.futures.ThreadPoolExecutor`?
Both route through ``concurrent.futures.thread``, whose module-level
``_python_exit()`` is registered with :mod:`atexit` and *joins every worker
thread unconditionally* at interpreter shutdown. A single hung health check would
therefore block the process from exiting forever. This module instead hand-rolls
``daemon=True`` threads: daemon threads are never joined at interpreter exit, so an
abandoned check can never block the process.

On timeout the awaiter receives :class:`TimeoutError` and the worker thread is
*abandoned* — it keeps running to completion but, being a daemon, never blocks
exit. A thread whose awaiter is *cancelled* (e.g. the runner backstop cancelling a
hung canary) is abandoned the same way, so no hung thread escapes accounting.
:func:`abandoned_count` is a cumulative event counter the CLI uses to decide
whether to fall back to :func:`os._exit`; :func:`abandoned_alive_count` prunes
threads that have since finished and reports only those still running, which the
refresh scheduler's circuit breaker uses to detect a persistently wedged suite.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_abandoned_lock = threading.Lock()
_abandoned = 0
_abandoned_threads: list[threading.Thread] = []


def abandoned_count() -> int:
    """Return the number of worker threads abandoned after a timeout or cancellation.

    A thread is counted at most once, at the moment its awaiter times out or is
    cancelled. Threads that finish before that are never counted, regardless of how
    long they took. The count only ever grows over the life of the process.

    Returns:
        The cumulative number of abandoned worker threads.
    """
    with _abandoned_lock:
        return _abandoned


def abandoned_alive_count() -> int:
    """Return the number of abandoned worker threads that are still running.

    Unlike :func:`abandoned_count` — a monotonic event counter — this reflects the
    live blind-spot: abandoned threads still executing right now. Threads that have
    since finished are pruned on each call, so a slow-but-completing check's thread
    stops being counted once it returns. The refresh scheduler's circuit breaker
    uses this to distinguish a persistently wedged suite from transient slowness.

    Returns:
        The number of abandoned worker threads currently alive.
    """
    with _abandoned_lock:
        _abandoned_threads[:] = [t for t in _abandoned_threads if t.is_alive()]
        return len(_abandoned_threads)


def _mark_abandoned(thread: threading.Thread) -> None:
    """Record that a worker thread was abandoned after a timeout or cancellation."""
    global _abandoned
    with _abandoned_lock:
        _abandoned += 1
        _abandoned_threads.append(thread)


async def run_sync(fn: Callable[..., T], *args: Any, timeout_s: float) -> T:
    """Run a synchronous callable on a daemon thread, bridged to the running loop.

    The callable executes on a freshly spawned ``daemon=True`` thread. Its result or
    exception is delivered back to the awaiter via :meth:`loop.call_soon_threadsafe`
    on an :class:`asyncio.Future`.

    Args:
        fn: The synchronous callable to run off the event loop.
        *args: Positional arguments forwarded to ``fn``.
        timeout_s: Maximum seconds to await a result before abandoning the thread.

    Returns:
        The value returned by ``fn``.

    Raises:
        TimeoutError: If ``fn`` does not complete within ``timeout_s``. The worker
            thread is abandoned (it keeps running but, being a daemon, never blocks
            process exit) and :func:`abandoned_count` is incremented.
        asyncio.CancelledError: If the awaiting task is cancelled while ``fn`` is
            still running. The worker thread is abandoned the same way before the
            cancellation propagates, so a hung thread is never left uncounted.
        Exception: Any exception raised by ``fn`` propagates to the awaiter.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def _set_result(value: T) -> None:
        # The future may already be cancelled by a timed-out ``wait_for``; guard so
        # a late-finishing abandoned thread does not raise InvalidStateError.
        if not future.done():
            future.set_result(value)

    def _set_exception(exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def _worker() -> None:
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - forwarded verbatim to the awaiter
            loop.call_soon_threadsafe(_set_exception, exc)
        else:
            loop.call_soon_threadsafe(_set_result, result)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    try:
        return await asyncio.wait_for(future, timeout=timeout_s)
    except (TimeoutError, asyncio.CancelledError):
        # Timeout or an externally-cancelled await (e.g. the runner backstop) both
        # leave the worker running; abandon it before re-raising so the daemon never
        # blocks exit and never escapes the abandoned-thread accounting.
        _mark_abandoned(thread)
        raise
