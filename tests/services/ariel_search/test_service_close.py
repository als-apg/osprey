"""Closing the ARIEL search service closes everything it opened, by any door.

The service opens three things: the ingestion pool, the read-only pool when its
build hosts the raw-SQL tool, and the pool its search-time provider calls run
on. ``aclose()`` closes all three, once; the context-manager exit, the web app
lifespan, the MCP context shutdown and ``close_ariel_service`` each call it.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.search import _offload
from osprey.services.ariel_search.search._offload import SEARCH_CALL_WORKERS
from osprey.services.ariel_search.service import ARIELSearchService


def _config() -> ARIELConfig:
    return ARIELConfig.from_dict({"database": {"uri": "postgresql://localhost:5432/test"}})


class _RecordingPool:
    """Stands in for an ``AsyncConnectionPool`` that only has to be closed.

    Appends ``name`` to ``log`` on close, and raises ``error`` when one is given.
    """

    def __init__(self, name: str, log: list[str], error: Exception | None = None) -> None:
        self.name = name
        self.log = log
        self.error = error

    @property
    def closed(self) -> bool:
        return self.name in self.log

    async def close(self) -> None:
        self.log.append(self.name)
        if self.error is not None:
            raise self.error


def _service(
    log: list[str] | None = None,
    *,
    readonly: bool = True,
    readonly_error: Exception | None = None,
) -> ARIELSearchService:
    log = [] if log is None else log
    return ARIELSearchService(
        config=_config(),
        pool=_RecordingPool("ingest", log),  # type: ignore[arg-type]
        repository=object(),  # type: ignore[arg-type]
        readonly_pool=(
            _RecordingPool("readonly", log, readonly_error)  # type: ignore[arg-type]
            if readonly
            else None
        ),
    )


def _record_search_calls_close(service: ARIELSearchService, log: list[str]) -> None:
    """Make the service's search-call pool log its close, still closing it."""
    pool = service.search_calls
    real_close = pool.aclose

    async def aclose() -> list[str]:
        survivors = await real_close()
        log.append("search_calls")
        return survivors

    pool.aclose = aclose  # type: ignore[method-assign]


def _spy_on_aclose(service: Any) -> AsyncMock:
    spy = AsyncMock(wraps=service.aclose)
    service.aclose = spy
    return spy


# ---------------------------------------------------------------------------
# The close ends every thread the service started and closes both pools
# ---------------------------------------------------------------------------


async def test_close_ends_every_search_call_thread():
    service = _service()
    all_running = threading.Barrier(SEARCH_CALL_WORKERS)

    def call() -> threading.Thread:
        all_running.wait()
        return threading.current_thread()

    threads = await asyncio.gather(
        *(service.search_calls.run(call) for _ in range(SEARCH_CALL_WORKERS))
    )
    assert len(set(threads)) == SEARCH_CALL_WORKERS
    assert all(thread.is_alive() for thread in threads)

    await service.aclose()

    assert not any(thread.is_alive() for thread in threads)


async def test_close_cancels_queued_calls():
    service = _service()
    lock = threading.Lock()
    started: list[int] = []
    all_started = threading.Event()
    release = threading.Event()
    queued_ran = threading.Event()

    def blocker(index: int) -> int:
        with lock:
            started.append(index)
            if len(started) == SEARCH_CALL_WORKERS:
                all_started.set()
        release.wait()
        return index

    blockers = [
        asyncio.create_task(service.search_calls.run(blocker, index))
        for index in range(SEARCH_CALL_WORKERS)
    ]
    try:
        await asyncio.to_thread(all_started.wait)
        queued = asyncio.create_task(service.search_calls.run(queued_ran.set))
        # One turn of the loop: the queued task submits its call to the pool.
        await asyncio.sleep(0)

        close = asyncio.create_task(service.aclose())
        with pytest.raises(asyncio.CancelledError):
            await queued
    finally:
        # Released whatever failed above, so no blocker outlives the test.
        release.set()
    await close

    assert await asyncio.gather(*blockers) == list(range(SEARCH_CALL_WORKERS))
    assert not queued_ran.is_set()


async def test_close_names_a_call_still_running_past_its_budget(monkeypatch, caplog):
    service = _service()
    # The budget is a multiple of the embedding timeout; shrinking the timeout
    # shrinks the wait without changing the relation.
    monkeypatch.setattr(_offload, "QUERY_EMBEDDING_TIMEOUT_S", 0.01)
    entered = threading.Event()
    release = threading.Event()

    def hang() -> threading.Thread:
        entered.set()
        release.wait()
        return threading.current_thread()

    call = asyncio.create_task(service.search_calls.run(hang))
    try:
        await asyncio.to_thread(entered.wait)
        with caplog.at_level("WARNING", logger="ariel"):
            survivors = await service.search_calls.aclose()
    finally:
        release.set()
    thread = await call

    assert survivors == [thread.name]
    assert thread.name in caplog.text
    await asyncio.to_thread(thread.join)


async def test_a_closed_pool_refuses_new_calls():
    service = _service()
    ran = threading.Event()
    await service.aclose()

    with pytest.raises(RuntimeError, match="search calls closed"):
        await service.search_calls.run(ran.set)

    assert not ran.is_set()


async def test_close_order_and_once():
    log: list[str] = []
    service = _service(log)
    _record_search_calls_close(service, log)

    await service.aclose()
    await service.aclose()

    assert log == ["search_calls", "readonly", "ingest"]


async def test_close_without_a_readonly_pool():
    log: list[str] = []
    service = _service(log, readonly=False)
    _record_search_calls_close(service, log)

    await service.aclose()

    assert log == ["search_calls", "ingest"]


async def test_a_failing_close_still_closes_the_rest():
    log: list[str] = []
    error = RuntimeError("readonly pool refused to close")
    service = _service(log, readonly_error=error)
    _record_search_calls_close(service, log)

    with pytest.raises(RuntimeError) as raised:
        await service.aclose()

    assert raised.value is error
    assert log == ["search_calls", "readonly", "ingest"]


async def test_the_first_close_error_is_the_one_raised():
    log: list[str] = []
    first = RuntimeError("readonly pool refused to close")
    service = _service(log, readonly_error=first)
    service.pool = _RecordingPool("ingest", log, RuntimeError("ingestion pool refused"))  # type: ignore[assignment]
    _record_search_calls_close(service, log)

    with pytest.raises(RuntimeError) as raised:
        await service.aclose()

    assert raised.value is first
    assert log == ["search_calls", "readonly", "ingest"]


# ---------------------------------------------------------------------------
# Every door closes the service through its one close
# ---------------------------------------------------------------------------


async def test_async_with_closes_through_aclose():
    service = _service()
    spy = _spy_on_aclose(service)

    async with service:
        pass

    spy.assert_awaited_once()
    assert service.pool.closed  # type: ignore[attr-defined]


async def test_the_mcp_context_shutdown_closes_through_aclose():
    from osprey.mcp_server.ariel.server_context import ARIELContext

    service = _service()
    spy = _spy_on_aclose(service)
    context = ARIELContext.__new__(ARIELContext)
    context._service = service

    await context.shutdown()

    spy.assert_awaited_once()
    assert service.readonly_pool.closed  # type: ignore[union-attr]
    assert context._service is None


async def test_closing_the_capability_singleton_closes_through_aclose(monkeypatch):
    from osprey.services.ariel_search import capability

    service = _service()
    spy = _spy_on_aclose(service)
    monkeypatch.setattr(capability, "_ariel_service_instance", service)

    await capability.close_ariel_service()

    spy.assert_awaited_once()
    assert capability._ariel_service_instance is None


async def test_the_web_app_lifespan_closes_through_aclose(tmp_path: Path, monkeypatch):
    from osprey.interfaces.ariel.app import create_app

    config_file = tmp_path / "config.yml"
    config_file.write_text(
        yaml.dump({"ariel": {"database": {"uri": "postgresql://localhost:5432/test"}}})
    )
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    service = MagicMock()
    service.health_check = AsyncMock(return_value=(True, "Service healthy"))
    service.aclose = AsyncMock()

    with patch(
        "osprey.services.ariel_search.create_ariel_service", AsyncMock(return_value=service)
    ):
        app = create_app(config_file)
        async with app.router.lifespan_context(app):
            assert app.state.ariel_service is service
            service.aclose.assert_not_awaited()

    service.aclose.assert_awaited_once()
