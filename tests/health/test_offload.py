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
async def test_alive_count_prunes_completed_abandoned_thread() -> None:
    """A slow-but-completing check's thread self-prunes from the alive count.

    The awaiter times out (so the thread is abandoned and briefly counted alive),
    but the work then finishes — once the worker returns, the next alive-count call
    prunes it. This is what keeps the circuit breaker from tripping on transient
    slowness.
    """
    release = threading.Event()

    def slow() -> None:
        release.wait(timeout=10.0)

    baseline = offload.abandoned_alive_count()
    with pytest.raises(TimeoutError):
        await offload.run_sync(slow, timeout_s=0.1)
    # Abandoned while still blocked → counted as alive.
    assert offload.abandoned_alive_count() == baseline + 1

    # Let the worker finish; the entry self-prunes once the thread terminates.
    release.set()
    deadline = time.monotonic() + 5.0
    while offload.abandoned_alive_count() > baseline and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert offload.abandoned_alive_count() == baseline


@pytest.mark.asyncio
async def test_alive_count_keeps_wedged_thread() -> None:
    """A still-wedged abandoned thread stays in the alive count."""
    release = threading.Event()

    def wedged() -> None:
        release.wait(timeout=10.0)

    baseline = offload.abandoned_alive_count()
    try:
        with pytest.raises(TimeoutError):
            await offload.run_sync(wedged, timeout_s=0.1)
        assert offload.abandoned_alive_count() == baseline + 1
        # Still wedged a beat later — remains counted (no self-prune).
        await asyncio.sleep(0.2)
        assert offload.abandoned_alive_count() == baseline + 1
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancellation_marks_thread_abandoned() -> None:
    """Cancelling the awaiter abandons the worker thread and counts it.

    Closes the blind spot where the runner backstop cancels a hung canary: the
    thread must be marked abandoned even though no TimeoutError fired.
    """
    release = threading.Event()

    def wedged() -> None:
        release.wait(timeout=10.0)

    before = offload.abandoned_count()
    baseline_alive = offload.abandoned_alive_count()
    task = asyncio.ensure_future(offload.run_sync(wedged, timeout_s=10.0))
    try:
        # Let the worker start and the await settle before cancelling externally.
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert offload.abandoned_count() == before + 1
        assert offload.abandoned_alive_count() == baseline_alive + 1
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
