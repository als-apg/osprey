"""Tests for the shared sliding-window rate limiter.

The limiter throttles LiteLLM calls to at most ``max_calls`` per ``window``.
These tests cover the module-singleton wiring (arm / disable / read-back) and
the acquire() gate: it passes freely below the limit and sleeps just long
enough once the window is full. ``asyncio.sleep`` is monkeypatched so no real
wall-clock time is spent.
"""

from __future__ import annotations

import time

import pytest

from osprey.services.channel_finder import rate_limiter
from osprey.services.channel_finder.rate_limiter import (
    _SlidingWindowLimiter,
    configure_rate_limiter,
    get_rate_limiter,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Keep the module-level singleton from leaking across tests (serial lane)."""
    yield
    configure_rate_limiter(None)


class TestConfigure:
    def test_none_disables(self):
        configure_rate_limiter(18, 60.0)
        configure_rate_limiter(None)
        assert get_rate_limiter() is None

    def test_zero_or_negative_disables(self):
        configure_rate_limiter(0)
        assert get_rate_limiter() is None
        configure_rate_limiter(-5)
        assert get_rate_limiter() is None

    def test_positive_arms_with_params(self):
        configure_rate_limiter(18, 30.0)
        limiter = get_rate_limiter()
        assert isinstance(limiter, _SlidingWindowLimiter)
        assert limiter.max_calls == 18
        assert limiter.window == 30.0

    def test_default_window_is_60s(self):
        configure_rate_limiter(5)
        assert get_rate_limiter().window == 60.0


class TestAcquire:
    async def test_under_limit_does_not_sleep(self, monkeypatch: pytest.MonkeyPatch):
        slept: list[float] = []

        async def fake_sleep(secs: float) -> None:
            slept.append(secs)

        monkeypatch.setattr(rate_limiter.asyncio, "sleep", fake_sleep)

        limiter = _SlidingWindowLimiter(max_calls=3, window=100.0)
        for _ in range(3):
            await limiter.acquire()

        assert slept == []
        assert len(limiter._calls) == 3

    async def test_sleeps_once_window_is_full(self, monkeypatch: pytest.MonkeyPatch):
        slept: list[float] = []

        async def fake_sleep(secs: float) -> None:
            slept.append(secs)

        monkeypatch.setattr(rate_limiter.asyncio, "sleep", fake_sleep)

        limiter = _SlidingWindowLimiter(max_calls=2, window=100.0)
        await limiter.acquire()
        await limiter.acquire()
        # Third acquire finds the bucket full and must wait ~one window.
        await limiter.acquire()

        assert len(slept) == 1
        assert 0 < slept[0] <= 100.0 + 0.05

    async def test_expired_timestamps_are_purged(self, monkeypatch: pytest.MonkeyPatch):
        slept: list[float] = []

        async def fake_sleep(secs: float) -> None:  # pragma: no cover - must not run
            slept.append(secs)

        monkeypatch.setattr(rate_limiter.asyncio, "sleep", fake_sleep)

        limiter = _SlidingWindowLimiter(max_calls=1, window=10.0)
        # Seed a timestamp well outside the window; acquire must drop it and
        # admit the new call without sleeping.
        limiter._calls.append(time.monotonic() - 100.0)
        await limiter.acquire()

        assert slept == []
        assert len(limiter._calls) == 1
