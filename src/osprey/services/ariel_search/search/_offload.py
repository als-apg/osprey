"""The one bounded pool search-time provider calls run on.

Embedding a query (and finding the server that embeds it) blocks: an HTTP call,
and for a local server possibly a reachability walk with 2 s probes. Run on the
event loop, a blackholed host would stall every other request the loop serves.
Run on the loop's default executor, it would compete with the work that pool
already carries (qmd's client resolution, ``asyncio.to_thread`` calls) and a
few hung probes could starve it. So search-time provider calls get their own
small pool.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

#: Worker threads for search-time provider calls (query embedding, reachability).
SEARCH_CALL_WORKERS = 4

#: The bounded pool every search-time provider call runs on; never the default executor.
SEARCH_CALL_POOL = ThreadPoolExecutor(
    max_workers=SEARCH_CALL_WORKERS, thread_name_prefix="ariel-search-call"
)


async def run_search_call(
    fn: Callable[..., T], *args: Any, timeout_s: float | None = None, **kwargs: Any
) -> T:
    """Run ``fn(*args, **kwargs)`` on :data:`SEARCH_CALL_POOL` and await its result.

    Args:
        fn: The blocking callable.
        *args: Positional arguments for ``fn``.
        timeout_s: Seconds to wait before giving up; ``None`` waits for the
            call. A call given up on keeps running on its worker and its result
            is discarded.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        What ``fn`` returned.

    Raises:
        TimeoutError: When ``timeout_s`` passes first.
        Exception: Whatever ``fn`` raised.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(SEARCH_CALL_POOL, functools.partial(fn, *args, **kwargs))
    if timeout_s is None:
        return await future
    return await asyncio.wait_for(future, timeout_s)
