"""Running synchronous code the way a notebook kernel does."""

from __future__ import annotations

import asyncio


def inside_a_running_loop(call):
    """Run *call* synchronously from inside a coroutine on a fresh loop.

    That is a notebook kernel's shape: the cell's synchronous code executes
    while the kernel's own loop is running on the same thread.
    """

    async def driver():
        return call()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(driver())
    finally:
        loop.close()
