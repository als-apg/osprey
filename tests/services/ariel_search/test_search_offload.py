"""Tests for the pool search-time provider calls run on, ``search._offload``."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from osprey.services.ariel_search.search._offload import SEARCH_CALL_WORKERS, run_search_call
from tests._live import async_wait_until


async def test_search_calls_run_on_their_own_bounded_pool():
    assert SEARCH_CALL_WORKERS == 4
    release = threading.Event()
    lock = threading.Lock()
    running = 0
    peak = 0

    def _call() -> str:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        try:
            release.wait(5.0)
            return threading.current_thread().name
        finally:
            with lock:
                running -= 1

    calls = [asyncio.ensure_future(run_search_call(_call)) for _ in range(SEARCH_CALL_WORKERS + 2)]
    await async_wait_until(lambda: running >= SEARCH_CALL_WORKERS, timeout=5.0, interval=0.01)
    await asyncio.sleep(0.1)
    with lock:
        held = running
    release.set()
    names = await asyncio.gather(*calls)

    assert held == SEARCH_CALL_WORKERS
    assert peak == SEARCH_CALL_WORKERS
    assert all(name.startswith("ariel-search-call") for name in names)


async def test_run_search_call_times_out():
    with pytest.raises(TimeoutError):
        await run_search_call(time.sleep, 0.5, timeout_s=0.05)


async def test_hung_calls_do_not_multiply_threads():
    release = threading.Event()
    lock = threading.Lock()
    entered = 0
    abandoned_ran = 0

    def _hang() -> None:
        nonlocal entered
        with lock:
            entered += 1
        release.wait(30.0)

    def _abandoned() -> None:
        nonlocal abandoned_ran
        with lock:
            abandoned_ran += 1

    before = threading.active_count()
    hung = [asyncio.ensure_future(run_search_call(_hang)) for _ in range(SEARCH_CALL_WORKERS)]
    try:
        await async_wait_until(lambda: entered == SEARCH_CALL_WORKERS, interval=0.01)
        for _ in range(50):
            with pytest.raises(TimeoutError):
                await run_search_call(_abandoned, timeout_s=0.01)
        grown = threading.active_count() - before
    finally:
        release.set()
    await asyncio.gather(*hung)
    assert await run_search_call(lambda: 1, timeout_s=10.0) == 1

    assert grown <= SEARCH_CALL_WORKERS
    assert abandoned_ran == 0


async def test_a_saturated_pool_refuses_at_once():
    release = threading.Event()
    lock = threading.Lock()
    entered = 0
    counted = 0

    def _hang() -> None:
        nonlocal entered
        with lock:
            entered += 1
        release.wait(30.0)

    def _counter() -> None:
        nonlocal counted
        with lock:
            counted += 1

    hung = [asyncio.ensure_future(run_search_call(_hang)) for _ in range(SEARCH_CALL_WORKERS)]
    try:
        await async_wait_until(lambda: entered == SEARCH_CALL_WORKERS, interval=0.01)
        for call in hung:
            call.cancel()
        await asyncio.gather(*hung, return_exceptions=True)

        with pytest.raises(TimeoutError, match="search calls saturated: 4 calls still running"):
            await run_search_call(_counter, timeout_s=5.0)
    finally:
        release.set()

    deadline = time.monotonic() + 10.0
    while True:
        try:
            assert await run_search_call(lambda: 1, timeout_s=5.0) == 1
            break
        except TimeoutError:
            if time.monotonic() > deadline:
                pytest.fail("the pool did not recover after its calls returned")
            await asyncio.sleep(0.01)
    assert counted == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
async def test_a_forked_child_gets_its_own_pool():
    assert await run_search_call(lambda: 1) == 1

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        code = 1
        try:
            result = asyncio.run(run_search_call(lambda: 42, timeout_s=1.0))
            code = 0 if result == 42 else 1
        finally:
            os._exit(code)

    deadline = time.monotonic() + 10.0
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail("the forked child did not exit")
        await asyncio.sleep(0.05)
    assert os.waitstatus_to_exitcode(status) == 0


_ABANDON_ONE_CALL = textwrap.dedent(
    """
    import asyncio
    import threading

    from osprey.services.ariel_search.search._offload import run_search_call


    async def main():
        assert await run_search_call(lambda: 1) == 1
        try:
            await run_search_call(threading.Event().wait, timeout_s=0.05)
        except TimeoutError:
            pass


    asyncio.run(main())
    print("exited")
    """
)


def test_the_pool_does_not_outlive_its_process():
    proc = subprocess.run(
        [sys.executable, "-c", _ABANDON_ONE_CALL], capture_output=True, text=True, timeout=60
    )

    assert proc.returncode == 0, proc.stderr
    assert "exited" in proc.stdout
