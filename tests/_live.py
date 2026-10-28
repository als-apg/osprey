"""Polling helpers for tests that drive a live process or server.

``wait_until`` and ``async_wait_until`` poll a probe until it returns something
truthy and hand that value back, so a wait that is also a read stays one call::

    status = wait_until(lambda: _status(url) or None, timeout=30, what="the health endpoint")

On timeout they raise :class:`AssertionError` naming *what* was awaited and the
probe's last value, which is what a failing test log needs to show.

For ports, use the ``reserve_port`` fixture or the harness's ``_ports`` helpers,
and ``wait_for_port`` from :mod:`osprey.interfaces._serving` (tests/README.md, "Ports").
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

__all__ = ["async_wait_until", "wait_until"]


def wait_until(
    probe: Callable[[], T],
    *,
    timeout: float = 10.0,
    interval: float = 0.05,
    what: str = "condition",
) -> T:
    """Call *probe* until it returns a truthy value, and return that value.

    Raises:
        AssertionError: *probe* stayed falsy for *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"{what} not reached within {timeout:g} s (last: {value!r})")
        time.sleep(interval)


async def async_wait_until(
    probe: Callable[[], T | Awaitable[T]],
    *,
    timeout: float = 10.0,
    interval: float = 0.05,
    what: str = "condition",
) -> T:
    """:func:`wait_until` for a coroutine test; *probe* may be sync or async."""
    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if inspect.isawaitable(value):
            value = await value
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"{what} not reached within {timeout:g} s (last: {value!r})")
        await asyncio.sleep(interval)
