"""Tests for `HealthRuntimeLifecycle`, the health sidecar's connector owner (task 2.2).

Covers the two concerns the unit owns:

- **Snapshot / re-snapshot** (:meth:`HealthRuntimeLifecycle.reconcile`): lazy
  construction from the first ``control_system`` snapshot; silent re-snapshot
  while no connector was ever constructed (so a broken-config first refresh
  cannot latch an empty mapping); and, once a connector exists, a one-time
  warning log plus a persistent restart-notice row with no in-process swap.
- **Loop-affine, refresh-serialized teardown**: lifespan ``shutdown`` cancels
  the in-flight refresh and disconnects on the owning loop's thread; the
  ``atexit`` hook no-ops without a connector, logs a single traceback-free
  warning when the teardown wedges, and is unregistered on clean shutdown.

The factory-patching spy mirrors ``tests/health/test_runtime.py`` — both names
`get_connector` imports from ``osprey.connectors.factory`` are intercepted.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from osprey.connectors import factory
from osprey.health.models import Status
from osprey.interfaces.health import lifecycle as lifecycle_mod
from osprey.interfaces.health.lifecycle import (
    RESTART_NOTICE_MESSAGE,
    HealthRuntimeLifecycle,
    control_system_snapshot,
)

_LIFECYCLE_LOGGER = "osprey.interfaces.health.lifecycle"


class _SpyConnector:
    """Minimal async-``disconnect``-only stand-in for a control-system connector."""

    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class _ThreadRecordingConnector(_SpyConnector):
    """Spy that records the thread its ``disconnect`` ran on (loop-affinity proof)."""

    def __init__(self) -> None:
        super().__init__()
        self.disconnect_thread_ident: int | None = None

    async def disconnect(self) -> None:
        self.disconnect_thread_ident = threading.get_ident()
        await super().disconnect()


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
    connector: Any,
    construct_calls: list[Any],
) -> None:
    """Spy `register_builtin_connectors` + `create_control_system_connector`."""

    async def fake_create(config: dict[str, Any]) -> Any:
        construct_calls.append(config)
        return connector

    monkeypatch.setattr(factory.ConnectorFactory, "create_control_system_connector", fake_create)
    monkeypatch.setattr(factory, "register_builtin_connectors", lambda: None)


@contextmanager
def _background_loop() -> Iterator[tuple[asyncio.AbstractEventLoop, threading.Thread]]:
    """Run an event loop on a daemon thread, yielding ``(loop, thread)``.

    Teardown cancels any pending tasks and stops the loop so wedged-disconnect
    tests never leave a task pending at close.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    ready.wait()
    try:
        yield loop, thread
    finally:

        def _stop() -> None:
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.stop()

        loop.call_soon_threadsafe(_stop)
        thread.join(timeout=2)
        loop.close()


def _run_on(loop: asyncio.AbstractEventLoop, coro: Any, timeout: float = 5.0) -> Any:
    """Submit a coroutine to a background loop and wait for the result."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)


# -- control_system_snapshot ---------------------------------------------------


def test_snapshot_guard_expression_normalizes_to_empty() -> None:
    # Mirrors the CLI guard `(expanded or {}).get("control_system", {}) or {}`.
    assert control_system_snapshot(None) == {}
    assert control_system_snapshot({}) == {}
    assert control_system_snapshot({"control_system": None}) == {}
    assert control_system_snapshot({"control_system": {"type": "mock"}}) == {"type": "mock"}


# -- reconcile: construction + re-snapshot -------------------------------------


async def test_reconcile_constructs_lazily_from_first_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    _patch_factory(monkeypatch, _SpyConnector(), construct_calls)

    lc = HealthRuntimeLifecycle()
    assert lc.runtime is None

    rows = lc.reconcile({"control_system": {"type": "mock"}})
    assert rows == []
    assert lc.runtime is not None
    # Snapshot stored, nothing constructed until a probe asks for a connector.
    assert lc.runtime.ever_constructed is False
    assert construct_calls == []

    await lc.runtime.get_connector()
    assert construct_calls == [{"type": "mock"}]


async def test_reconcile_unchanged_snapshot_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_factory(monkeypatch, _SpyConnector(), [])

    lc = HealthRuntimeLifecycle()
    lc.reconcile({"control_system": {"type": "mock"}})
    first = lc.runtime

    rows = lc.reconcile({"control_system": {"type": "mock"}})
    assert rows == []
    assert lc.runtime is first  # same instance, no re-snapshot


async def test_reconcile_resnapshots_before_any_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    _patch_factory(monkeypatch, _SpyConnector(), construct_calls)

    lc = HealthRuntimeLifecycle()
    lc.reconcile({"control_system": {"type": "mock"}})
    first = lc.runtime

    # No connector was ever constructed → a changed mapping silently replaces
    # the runtime with one built from the new snapshot, and emits no notice.
    rows = lc.reconcile({"control_system": {"type": "epics"}})
    assert rows == []
    assert lc.runtime is not first
    assert lc.runtime is not None
    assert lc.runtime.ever_constructed is False

    await lc.runtime.get_connector()
    assert construct_calls == [{"type": "epics"}]


async def test_reconcile_broken_first_snapshot_then_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    _patch_factory(monkeypatch, _SpyConnector(), construct_calls)

    lc = HealthRuntimeLifecycle()
    # Broken-config first refresh: no `control_system` section → empty snapshot.
    lc.reconcile({})
    # A later fixed config re-snapshots (empty was never latched into a connector).
    rows = lc.reconcile({"control_system": {"type": "mock"}})
    assert rows == []

    await lc.runtime.get_connector()  # type: ignore[union-attr]
    # The real connector is built from the fixed config, not the broken empty one.
    assert construct_calls == [{"type": "mock"}]


# -- reconcile: restart notice after construction ------------------------------


async def test_reconcile_emits_restart_notice_after_construction(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_factory(monkeypatch, _SpyConnector(), [])

    lc = HealthRuntimeLifecycle()
    lc.reconcile({"control_system": {"type": "mock"}})
    runtime = lc.runtime
    assert runtime is not None
    await runtime.get_connector()  # ever_constructed → True
    assert runtime.ever_constructed is True

    with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
        rows = lc.reconcile({"control_system": {"type": "epics"}})

    # Notice row surfaced; the live runtime is never swapped in-process.
    assert lc.runtime is runtime
    assert len(rows) == 1
    assert rows[0].message == RESTART_NOTICE_MESSAGE
    assert rows[0].status is Status.WARNING
    assert rows[0].category == "configuration"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1


async def test_restart_notice_row_persists_but_warning_logs_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_factory(monkeypatch, _SpyConnector(), [])

    lc = HealthRuntimeLifecycle()
    lc.reconcile({"control_system": {"type": "mock"}})
    await lc.runtime.get_connector()  # type: ignore[union-attr]

    with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
        first = lc.reconcile({"control_system": {"type": "epics"}})
        caplog.clear()
        # Same unapplied change on the next refresh: row again, no second warning.
        second = lc.reconcile({"control_system": {"type": "epics"}})

    assert len(first) == 1
    assert len(second) == 1  # row still rendered on every report
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_reconcile_back_to_live_config_clears_notice_and_relogs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_factory(monkeypatch, _SpyConnector(), [])

    lc = HealthRuntimeLifecycle()
    lc.reconcile({"control_system": {"type": "mock"}})
    await lc.runtime.get_connector()  # type: ignore[union-attr]

    with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
        lc.reconcile({"control_system": {"type": "epics"}})  # diverge → warn + row
        # Config reverts to what the live connector was built from: no notice.
        back = lc.reconcile({"control_system": {"type": "mock"}})
        caplog.clear()
        # Diverging again re-arms the one-time warning (latch was cleared).
        again = lc.reconcile({"control_system": {"type": "epics"}})

    assert back == []
    assert len(again) == 1
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


# -- teardown: loop-affine, refresh-serialized ---------------------------------


def test_shutdown_disconnects_on_owning_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ThreadRecordingConnector()
    _patch_factory(monkeypatch, connector, [])

    with _background_loop() as (loop, thread):
        lc = HealthRuntimeLifecycle()

        async def _setup() -> None:
            lc.bind_loop()
            lc.reconcile({"control_system": {"type": "mock"}})
            await lc.runtime.get_connector()  # type: ignore[union-attr]

        _run_on(loop, _setup())
        _run_on(loop, lc.shutdown())

    # Disconnect ran exactly once, on the loop's thread — not the caller's.
    assert connector.disconnect_calls == 1
    assert connector.disconnect_thread_ident == thread.ident
    assert connector.disconnect_thread_ident != threading.get_ident()


def test_shutdown_cancels_inflight_refresh_before_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _SpyConnector()
    _patch_factory(monkeypatch, connector, [])

    with _background_loop() as (loop, thread):
        state: dict[str, Any] = {}

        async def _setup() -> None:
            lc = HealthRuntimeLifecycle()
            lc.bind_loop()
            lc.reconcile({"control_system": {"type": "mock"}})
            await lc.runtime.get_connector()  # type: ignore[union-attr]

            async def _hang() -> None:
                await asyncio.Event().wait()

            task = asyncio.ensure_future(_hang())
            lc.set_inflight_task_provider(lambda: task)
            state["lc"] = lc
            state["task"] = task

        _run_on(loop, _setup())
        _run_on(loop, state["lc"].shutdown())

        assert state["task"].cancelled()

    assert connector.disconnect_calls == 1


# -- teardown: atexit hook -----------------------------------------------------


def test_atexit_hook_noops_without_a_constructed_connector(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_factory(monkeypatch, _SpyConnector(), [])
    submitted: list[Any] = []
    monkeypatch.setattr(
        lifecycle_mod.asyncio,
        "run_coroutine_threadsafe",
        lambda coro, loop: submitted.append(coro),
    )

    lc = HealthRuntimeLifecycle()
    lc.reconcile({"control_system": {"type": "mock"}})  # snapshot only, no connector

    class _FakeLoop:
        def is_closed(self) -> bool:
            return False

    lc.bind_loop(_FakeLoop())  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
        lc._atexit_shutdown()

    # Never-constructed → nothing submitted to the loop, nothing logged.
    assert submitted == []
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_atexit_wedged_teardown_logs_single_warning_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_factory(monkeypatch, _SpyConnector(), [])

    lc = HealthRuntimeLifecycle()

    async def _build() -> None:
        lc.reconcile({"control_system": {"type": "mock"}})
        await lc.runtime.get_connector()  # type: ignore[union-attr]

    asyncio.run(_build())
    assert lc.runtime.ever_constructed is True  # type: ignore[union-attr]

    class _FakeLoop:
        def is_closed(self) -> bool:
            return False

    lc.bind_loop(_FakeLoop())  # type: ignore[arg-type]

    class _WedgedFuture:
        def result(self, timeout: float | None = None) -> None:
            raise TimeoutError("teardown wedged")

    def _fake_submit(coro: Any, loop: Any) -> _WedgedFuture:
        coro.close()  # avoid "coroutine was never awaited"
        return _WedgedFuture()

    monkeypatch.setattr(lifecycle_mod.asyncio, "run_coroutine_threadsafe", _fake_submit)

    with caplog.at_level(logging.WARNING, logger=_LIFECYCLE_LOGGER):
        lc._atexit_shutdown()  # must not raise

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is None  # one clean line, no exit-time traceback


def test_atexit_hook_unregistered_after_lifespan_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[Any] = []
    unregistered: list[Any] = []
    monkeypatch.setattr(lifecycle_mod.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(lifecycle_mod.atexit, "unregister", lambda fn: unregistered.append(fn))

    lc = HealthRuntimeLifecycle()
    lc.register_atexit()
    assert lc.atexit_registered is True
    assert registered == [lc._atexit_shutdown]

    asyncio.run(lc.shutdown())

    assert lc.atexit_registered is False
    assert unregistered == [lc._atexit_shutdown]


def test_register_and_unregister_atexit_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[Any] = []
    unregistered: list[Any] = []
    monkeypatch.setattr(lifecycle_mod.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(lifecycle_mod.atexit, "unregister", lambda fn: unregistered.append(fn))

    lc = HealthRuntimeLifecycle()
    lc.register_atexit()
    lc.register_atexit()  # second call is a no-op
    lc.unregister_atexit()
    lc.unregister_atexit()  # second call is a no-op

    assert len(registered) == 1
    assert len(unregistered) == 1
