"""The event-loop watch logs a held loop, stays quiet otherwise, and ends when cancelled."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from contextlib import suppress

import pytest

from osprey.interfaces import _loop_delay
from osprey.interfaces._loop_delay import watch_loop_delay
from osprey.interfaces.web_terminal import app as web_terminal_app


async def _watched(
    held_s: float, threshold_s: float, caplog: pytest.LogCaptureFixture
) -> list[str]:
    """The warnings of a watch over a loop that is held once for ``held_s``."""
    with caplog.at_level(logging.WARNING, logger=_loop_delay.__name__):
        task = asyncio.create_task(
            watch_loop_delay("hub", interval_s=0.01, threshold_s=threshold_s)
        )
        await asyncio.sleep(0.03)
        time.sleep(held_s)  # the loop is held: nothing else runs
        await asyncio.sleep(0.03)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert task.cancelled()
    return [record.getMessage() for record in caplog.records]


async def test_a_held_loop_is_logged_with_its_length(caplog: pytest.LogCaptureFixture) -> None:
    messages = await _watched(0.3, 0.1, caplog)

    assert messages
    assert messages[0].startswith("hub: the event loop was held for 0.")


async def test_a_delay_under_the_threshold_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    assert await _watched(0.1, 30.0, caplog) == []


def test_the_web_terminal_starts_the_watch_and_cancels_it() -> None:
    lifespan = inspect.getsource(web_terminal_app._create_lifespan)
    started, _, shutdown = lifespan.partition("\n        yield\n")

    assert 'asyncio.create_task(watch_loop_delay("web terminal"))' in started
    assert "loop_watch_task" in shutdown
