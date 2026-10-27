"""Tests for the pool search-time provider calls run on, ``SearchCallPool``."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from osprey.services.ariel_search.search._offload import (
    QUERY_EMBEDDING_TIMEOUT_S,
    SEARCH_CALL_WORKERS,
)
from tests._live import async_wait_until

#: Hang ceiling for a wait on the pool: the close's own budget for a search call.
_CEILING_S = 2 * QUERY_EMBEDDING_TIMEOUT_S


async def test_search_calls_run_on_their_own_bounded_pool(search_calls):
    assert SEARCH_CALL_WORKERS == 4
    lock = threading.Lock()
    release = threading.Event()
    all_started = threading.Event()
    started: list[tuple[threading.Thread, bool]] = []

    def _call() -> str:
        with lock:
            started.append((threading.current_thread(), release.is_set()))
            if len(started) == SEARCH_CALL_WORKERS:
                all_started.set()
        release.wait()
        return threading.current_thread().name

    calls = [asyncio.ensure_future(search_calls.run(_call)) for _ in range(SEARCH_CALL_WORKERS + 1)]
    await asyncio.to_thread(all_started.wait)
    release.set()
    names = await asyncio.gather(*calls)

    # Exactly one worker per call started before any was released; the extra
    # call waited for a worker.
    assert sum(1 for _, released in started if not released) == SEARCH_CALL_WORKERS
    assert len({thread for thread, _ in started}) == SEARCH_CALL_WORKERS
    assert all(name.startswith("ariel-search-call") for name in names)


async def test_a_search_call_times_out(search_calls):
    with pytest.raises(TimeoutError):
        await search_calls.run(time.sleep, 0.5, timeout_s=0.05)


async def test_hung_calls_do_not_multiply_threads(search_calls):
    release = threading.Event()
    lock = threading.Lock()
    entered = 0
    abandoned_ran = 0

    def _hang() -> None:
        nonlocal entered
        with lock:
            entered += 1
        release.wait(_CEILING_S)

    def _abandoned() -> None:
        nonlocal abandoned_ran
        with lock:
            abandoned_ran += 1

    before = threading.active_count()
    hung = [asyncio.ensure_future(search_calls.run(_hang)) for _ in range(SEARCH_CALL_WORKERS)]
    try:
        await async_wait_until(
            lambda: entered == SEARCH_CALL_WORKERS,
            timeout=_CEILING_S,
            interval=0.01,
            what="every worker running a hung call",
        )
        for _ in range(50):
            with pytest.raises(TimeoutError):
                await search_calls.run(_abandoned, timeout_s=0.01)
        grown = threading.active_count() - before
    finally:
        release.set()
    await asyncio.gather(*hung)
    assert await search_calls.run(lambda: 1, timeout_s=_CEILING_S) == 1

    assert grown <= SEARCH_CALL_WORKERS
    assert abandoned_ran == 0


async def test_a_saturated_pool_refuses_at_once(search_calls):
    release = threading.Event()
    lock = threading.Lock()
    entered = 0
    counted = 0

    def _hang() -> None:
        nonlocal entered
        with lock:
            entered += 1
        release.wait(_CEILING_S)

    def _counter() -> None:
        nonlocal counted
        with lock:
            counted += 1

    hung = [asyncio.ensure_future(search_calls.run(_hang)) for _ in range(SEARCH_CALL_WORKERS)]
    try:
        await async_wait_until(
            lambda: entered == SEARCH_CALL_WORKERS,
            timeout=_CEILING_S,
            interval=0.01,
            what="every worker running a hung call",
        )
        for call in hung:
            call.cancel()
        await asyncio.gather(*hung, return_exceptions=True)

        with pytest.raises(TimeoutError, match="search calls saturated: 4 calls still running"):
            await search_calls.run(_counter, timeout_s=_CEILING_S)
    finally:
        release.set()

    async def _runs() -> bool:
        try:
            return await search_calls.run(lambda: 1, timeout_s=_CEILING_S) == 1
        except TimeoutError:
            return False

    await async_wait_until(
        _runs, timeout=_CEILING_S, interval=0.01, what="the pool running calls again"
    )
    assert counted == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
async def test_a_forked_child_gets_its_own_pool(search_calls):
    assert await search_calls.run(lambda: 1) == 1

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        code = 1
        try:
            result = asyncio.run(search_calls.run(lambda: 42, timeout_s=QUERY_EMBEDDING_TIMEOUT_S))
            code = 0 if result == 42 else 1
        finally:
            os._exit(code)

    def _exited() -> tuple[int, int] | None:
        done, status = os.waitpid(pid, os.WNOHANG)
        return (done, status) if done else None

    try:
        _, status = await async_wait_until(
            _exited, timeout=_CEILING_S, interval=0.05, what="the forked child exiting"
        )
    except AssertionError:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        raise
    assert os.waitstatus_to_exitcode(status) == 0


_ABANDON_ONE_CALL = textwrap.dedent(
    """
    import asyncio
    import threading

    from osprey.services.ariel_search.search._offload import SearchCallPool


    async def main():
        search_calls = SearchCallPool()
        assert await search_calls.run(lambda: 1) == 1
        try:
            await search_calls.run(threading.Event().wait, timeout_s=0.05)
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
