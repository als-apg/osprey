"""The one way enhancement module code runs blocking work off the event loop.

A thin wrapper of :func:`osprey.health.offload.run_sync`: the call runs on a
daemon thread, so ``asyncio.run`` never joins it at exit, a cancelled await
returns at once and a late completion is discarded silently. Threads abandoned
here never enter ``osprey health``'s abandoned-thread accounting; they are
tracked per ``key`` instead, so :func:`offload_busy` can tell the catch-up that
a cancelled call to a single-model server is still running.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from osprey.health import offload

T = TypeVar("T")

_lock = threading.Lock()
_orphans: dict[str, list[threading.Thread]] = {}


async def run_blocking(
    fn: Callable[..., T], /, *args: Any, key: str | None = None, **kwargs: Any
) -> T:
    """Run ``fn(*args, **kwargs)`` on a daemon thread and await its result.

    Args:
        fn: The blocking callable.
        *args: Positional arguments for ``fn``.
        key: When given, a thread still running after its await was cancelled
            is recorded under this key until it finishes (see
            :func:`offload_busy`).
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        What ``fn`` returned.

    Raises:
        asyncio.CancelledError: When the await is cancelled; the thread keeps
            running and its result is discarded.
        Exception: Whatever ``fn`` raised.
    """

    def _abandon(thread: threading.Thread) -> None:
        if key is None:
            return
        with _lock:
            _orphans.setdefault(key, []).append(thread)

    return await offload.run_sync(
        functools.partial(fn, *args, **kwargs), timeout_s=None, on_abandon=_abandon
    )


def offload_busy(key: str) -> bool:
    """Return whether a call under ``key`` is still running after its await was cancelled.

    Args:
        key: The key the call was made under.

    Returns:
        True while at least one such thread is alive.
    """
    with _lock:
        alive = [t for t in _orphans.get(key, []) if t.is_alive()]
        if alive:
            _orphans[key] = alive
        else:
            _orphans.pop(key, None)
        return bool(alive)


def reset_offload_state() -> None:
    """Forget every recorded orphan thread. For test isolation; threads keep running."""
    with _lock:
        _orphans.clear()
