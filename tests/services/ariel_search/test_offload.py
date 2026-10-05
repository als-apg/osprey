"""Tests for the enhancement modules' blocking-work bridge, ``enhancement._offload``."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from osprey.health import offload
from osprey.services.ariel_search.enhancement import _offload
from osprey.services.ariel_search.enhancement._offload import offload_busy, run_blocking

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean_orphans():
    _offload.reset_offload_state()
    yield
    _offload.reset_offload_state()


async def test_forwards_positional_and_keyword_arguments() -> None:
    def combine(a, b, *, sep):
        return f"{a}{sep}{b}"

    assert await run_blocking(combine, "x", "y", sep="-") == "x-y"


async def test_key_is_not_forwarded_to_the_callable() -> None:
    def no_kwargs():
        return "ok"

    assert await run_blocking(no_kwargs, key="image_caption") == "ok"


async def test_exception_propagates() -> None:
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        await run_blocking(boom)


async def test_runs_on_a_daemon_thread() -> None:
    assert await run_blocking(lambda: threading.current_thread().daemon) is True


async def test_cancelled_await_returns_at_once_and_marks_the_key_busy() -> None:
    release = threading.Event()
    task = asyncio.ensure_future(run_blocking(release.wait, 10.0, key="image_caption"))
    await asyncio.sleep(0.05)
    before = offload.abandoned_count()
    start = time.monotonic()
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - start < 1.0
        assert offload_busy("image_caption") is True
        assert offload_busy("image_embedding") is False
        # ARIEL orphans never enter osprey health's abandoned-thread accounting.
        assert offload.abandoned_count() == before
    finally:
        release.set()

    deadline = time.monotonic() + 5
    while offload_busy("image_caption") and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert offload_busy("image_caption") is False


async def test_a_completed_call_never_marks_its_key_busy() -> None:
    await run_blocking(lambda: None, key="image_caption")
    assert offload_busy("image_caption") is False


async def test_a_cancelled_call_without_a_key_is_not_tracked() -> None:
    release = threading.Event()
    task = asyncio.ensure_future(run_blocking(release.wait, 10.0))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _offload._orphans == {}
    finally:
        release.set()
