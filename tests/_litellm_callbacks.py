"""A thread pool of the test's own for litellm's success handlers.

litellm hands every completed synchronous call's success handler to a
module-level thread pool that it never shuts down. A test that makes a real
litellm call runs those handlers on a pool of its own instead, closed when the
test ends, so no idle worker is left behind for the end-of-run live-thread
report.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

#: The name the scoped pool's workers carry, so a report or a test can tell
#: them from any other pool's.
THREAD_NAME_PREFIX = "litellm-callbacks"


@contextmanager
def litellm_callback_pool() -> Iterator[ThreadPoolExecutor]:
    """Run litellm's success handlers on a pool that is shut down on exit.

    The library's pool is restored before the scoped pool shuts down, so no
    handler submitted after the scope closes can reach a pool that is closing.

    Yields:
        The thread pool litellm submits its success handlers to while the
        scope is open.
    """
    import litellm.utils

    pool = ThreadPoolExecutor(thread_name_prefix=THREAD_NAME_PREFIX)
    library_pool = litellm.utils.executor
    # litellm's ``client`` wrapper looks this attribute up on ``litellm.utils``
    # at call time, so replacing it redirects every handler submitted while the
    # scope is open.
    litellm.utils.executor = pool
    try:
        yield pool
    finally:
        litellm.utils.executor = library_pool
        pool.shutdown(wait=True)
