"""Unit tests for the daemon-thread sync off-loading bridge."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from osprey.health import offload


@pytest.mark.asyncio
async def test_result_delivered() -> None:
    """A sync callable's return value is delivered to the awaiter."""
    result = await offload.run_sync(lambda: 42, timeout_s=1.0)
    assert result == 42


@pytest.mark.asyncio
async def test_positional_args_forwarded() -> None:
    """Positional args are forwarded to the callable."""
    result = await offload.run_sync(lambda a, b: a + b, 2, 3, timeout_s=1.0)
    assert result == 5


@pytest.mark.asyncio
async def test_exception_propagates() -> None:
    """An exception raised by the callable propagates to the awaiter."""

    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        await offload.run_sync(boom, timeout_s=1.0)


@pytest.mark.asyncio
async def test_spawned_thread_is_daemon() -> None:
    """The callable runs on a daemon thread (never joined at interpreter exit)."""
    is_daemon = await offload.run_sync(lambda: threading.current_thread().daemon, timeout_s=1.0)
    assert is_daemon is True


@pytest.mark.asyncio
async def test_timeout_raises_and_returns_promptly() -> None:
    """On timeout the awaiter gets TimeoutError promptly; the thread is abandoned."""
    release = threading.Event()

    def slow() -> str:
        # Blocks well past the timeout; released by the test to avoid leaking work.
        release.wait(timeout=10.0)
        return "late"

    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await offload.run_sync(slow, timeout_s=0.1)
        elapsed = time.monotonic() - start
        # The await returns on the timeout, not when the abandoned thread finishes.
        assert elapsed < 2.0
    finally:
        release.set()


@pytest.mark.asyncio
async def test_abandoned_counter_increments_only_on_abandonment() -> None:
    """The abandoned counter increments on timeout, not for completed work."""
    before = offload.abandoned_count()

    # Completed work must not touch the counter.
    await offload.run_sync(lambda: "done", timeout_s=1.0)
    assert offload.abandoned_count() == before

    release = threading.Event()

    def slow() -> None:
        release.wait(timeout=10.0)

    try:
        with pytest.raises(TimeoutError):
            await offload.run_sync(slow, timeout_s=0.1)
        assert offload.abandoned_count() == before + 1
    finally:
        release.set()


@pytest.mark.asyncio
async def test_event_loop_not_blocked_during_offload() -> None:
    """Off-loaded sync work does not block the event loop."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1

    tick_task = asyncio.ensure_future(ticker())
    result = await offload.run_sync(lambda: time.sleep(0.05) or "ok", timeout_s=1.0)
    await tick_task

    assert result == "ok"
    assert ticks == 5
