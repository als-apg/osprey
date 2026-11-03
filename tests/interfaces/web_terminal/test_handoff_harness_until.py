"""The hand-off harness's ``until``: a paused process is not the code's delay.

The phase and invariant suites wait on fake-clock looks with a real-time
budget. A long garbage collection on the loop thread stops everything for
seconds; the wait must not spend that pause as the code's time.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.interfaces.web_terminal._handoff_harness import until


async def test_a_pause_of_the_loop_thread_is_not_charged_to_the_condition():
    done: list[bool] = []

    async def three_turns_after_a_pause() -> None:
        time.sleep(0.3)  # nothing on the loop runs, as under a long collection
        for _ in range(3):
            await asyncio.sleep(0)
        done.append(True)

    task = asyncio.create_task(three_turns_after_a_pause())
    await until(lambda: done, timeout=0.1)
    await task


async def test_a_condition_that_never_holds_still_fails():
    with pytest.raises(AssertionError, match="not reached within 0.05 s"):
        await until(lambda: False, timeout=0.05)
