"""Tests for `HealthRuntime`, the health suite's connector-lifecycle owner (task 1.4).

`HealthRuntime` lazily constructs at most one control-system connector and
disconnects it exactly once on exit. Mirrors the spy-connector shape from
``tests/services/bluesky_bridge/test_lifespan_connector.py``:

- construct-once/disconnect-once: two ``get_connector`` calls construct one
  connector; context exit disconnects it exactly once;
- never-constructed: a runtime whose ``get_connector`` is never called neither
  registers connector types nor constructs anything, and disconnects nothing;
- best-effort teardown: a ``disconnect()`` that raises is swallowed so context
  exit never propagates it;
- serialized construction: probes run concurrently inside a health category, so
  two gathered ``get_connector``/``get_archive`` calls must still construct a
  single connector rather than orphaning the loser of the race.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey.health.runtime import HealthRuntime
from osprey_connectors import factory
from osprey_connectors.service_answers import BackendSpec
from tests.health._fakes import SpyConnector

#: An archive's spec, as the archive row of ``service_targets`` names one.
_ARCHIVE_SPEC = BackendSpec(
    kind="archive",
    type="epics_archiver",
    target="live",
    settings={"url": "https://arch.example"},
    rehearsal=False,
)


class _RaisingDisconnectConnector(SpyConnector):
    """Spy whose ``disconnect`` raises, to prove teardown swallows the error."""

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        raise RuntimeError("boom during disconnect")


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
    connector: Any,
    construct_calls: list[Any],
) -> None:
    """Spy `create_control_system_connector`.

    `HealthRuntime.get_connector` imports the factory from
    ``osprey_connectors.factory`` at call time, so patching it there
    intercepts the build.
    """

    async def fake_create(config: Any, *, control_target: str | None = None) -> Any:  # noqa: ARG001 - stands in for the connector factory, whose control_target is keyword-only
        construct_calls.append(dict(config.raw))
        return connector

    monkeypatch.setattr(
        factory.ConnectorFactory,
        "create_control_system_connector",
        fake_create,
    )


async def test_get_connector_constructs_once_and_disconnects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    config = {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": {"serving": "in_process"}},
    }

    async with HealthRuntime(config) as runtime:
        first = await runtime.get_connector()
        second = await runtime.get_connector()

        # Both calls return the same cached instance, built exactly once from
        # the configured section — and not disconnected while the suite runs.
        assert first is spy
        assert second is spy
        assert construct_calls == [config]
        assert spy.disconnect_calls == 0

    # Disconnected exactly once on context exit.
    assert spy.disconnect_calls == 1


async def test_never_constructs_connector_when_get_connector_unused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    # A connector that would explode if constructed — it must never be.
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    async with HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ):
        pass  # no channel_read-style probe ever asks for a connector

    # Nothing constructed, nothing to disconnect.
    assert construct_calls == []
    assert spy.disconnect_calls == 0


async def test_disconnect_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = _RaisingDisconnectConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    # Context exit must not propagate the disconnect failure.
    async with HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ) as runtime:
        await runtime.get_connector()

    assert spy.disconnect_calls == 1


async def test_explicit_shutdown_disconnects_once_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )
    await runtime.get_connector()
    await runtime.shutdown()
    await runtime.shutdown()  # second call disconnects nothing

    assert spy.disconnect_calls == 1


async def test_get_connector_refuses_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )
    await runtime.get_connector()
    await runtime.shutdown()

    # A closed runtime refuses rather than reconstructing a torn-down connector.
    with pytest.raises(RuntimeError):
        await runtime.get_connector()

    # The refusal built nothing new: still a single construction, no re-register.
    assert construct_calls == [
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ]


async def test_get_connector_refuses_after_shutdown_without_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    # shutdown() closes the runtime even when a connector was never built.
    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )
    await runtime.shutdown()

    with pytest.raises(RuntimeError):
        await runtime.get_connector()

    assert construct_calls == []


async def test_flag_transitions_across_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )
    # Freshly built: nothing constructed, not closed.
    assert runtime.ever_constructed is False
    assert runtime.closed is False

    await runtime.get_connector()
    # Construction history recorded; still open.
    assert runtime.ever_constructed is True
    assert runtime.closed is False

    await runtime.shutdown()
    # Closed, but construction history is sticky (unlike `_connector is None`).
    assert runtime.closed is True
    assert runtime.ever_constructed is True


async def test_ever_constructed_stays_false_when_get_connector_unused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    async with HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ) as runtime:
        assert runtime.ever_constructed is False
        assert runtime.closed is False

    # Context exit closes the runtime; nothing was ever constructed.
    assert runtime.ever_constructed is False
    assert runtime.closed is True


async def test_double_shutdown_is_safe_after_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_factory(monkeypatch, spy, construct_calls)

    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )
    await runtime.get_connector()
    await runtime.shutdown()
    await runtime.shutdown()  # idempotent: still closed, disconnected once

    assert runtime.closed is True
    assert runtime.ever_constructed is True
    assert spy.disconnect_calls == 1


class _SpyArchiver:
    """Minimal async-``aclose``-only stand-in for an archive."""

    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _patch_archiver_factory(
    monkeypatch: pytest.MonkeyPatch,
    archive: Any,
    construct_calls: list[Any],
) -> None:
    """Spy `create_service`, yielding once inside the construction.

    The ``await asyncio.sleep(0)`` is the whole point: it hands control back to
    the event loop while the cache slot is still empty, which is exactly the
    window a second concurrent probe used to slip through.
    """

    async def fake_create(spec: BackendSpec, **_kwargs: Any) -> Any:
        construct_calls.append(spec)
        await asyncio.sleep(0)
        return archive

    monkeypatch.setattr(factory.ConnectorFactory, "create_service", fake_create)


def _patch_slow_factory(
    monkeypatch: pytest.MonkeyPatch,
    connector: Any,
    construct_calls: list[Any],
) -> None:
    """Like `_patch_factory`, but yields to the loop mid-construction.

    Without the runtime's lock, the yield lets a second gathered caller observe
    the still-empty cache and build a connector of its own — the orphan
    ``shutdown()`` would never disconnect.
    """

    async def fake_create(config: Any, *, control_target: str | None = None) -> Any:  # noqa: ARG001 - stands in for the connector factory, whose control_target is keyword-only
        construct_calls.append(dict(config.raw))
        await asyncio.sleep(0)
        return connector

    monkeypatch.setattr(
        factory.ConnectorFactory,
        "create_control_system_connector",
        fake_create,
    )


async def test_concurrent_get_connector_constructs_exactly_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_slow_factory(monkeypatch, spy, construct_calls)

    async with HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ) as runtime:
        first, second = await asyncio.gather(
            runtime.get_connector(),
            runtime.get_connector(),
        )

        # One construction between them, and the waiter got the winner's
        # instance rather than an orphan of its own.
        assert construct_calls == [
            {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"serving": "in_process"}},
            }
        ]
        assert first is spy
        assert second is spy

    # The single connector is disconnected once; nothing was left behind.
    assert spy.disconnect_calls == 1


async def test_many_concurrent_get_connector_calls_construct_exactly_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_slow_factory(monkeypatch, spy, construct_calls)

    # A realistic category fan-out: several channel_read-style probes at once.
    async with HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ) as runtime:
        results = await asyncio.gather(*(runtime.get_connector() for _ in range(8)))

    assert len(construct_calls) == 1
    assert all(result is spy for result in results)


async def test_shutdown_during_construction_does_not_orphan_the_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The closed check runs before the lock, so it can be passed and then
    overtaken: ``shutdown()`` finds the slot still empty, disconnects nothing,
    and the construction completes into a runtime that has already torn down.
    Nothing would ever disconnect that connector — the orphan the lock exists
    to prevent, reached through a different window. The construction has to
    re-check inside the lock and close what it just built."""
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_slow_factory(monkeypatch, spy, construct_calls)
    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )

    getter = asyncio.create_task(runtime.get_connector())
    await asyncio.sleep(0)  # getter passes the closed check and enters construction
    assert construct_calls == [
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ]
    await runtime.shutdown()

    with pytest.raises(RuntimeError, match="closed while its connector was being constructed"):
        await getter
    assert spy.disconnect_calls == 1
    assert runtime.closed
    assert not runtime.ever_constructed


async def test_concurrent_get_archive_constructs_exactly_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = _SpyArchiver()
    _patch_archiver_factory(monkeypatch, spy, construct_calls)

    async with HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    ) as runtime:
        first, second = await asyncio.gather(
            runtime.get_archive(_ARCHIVE_SPEC),
            runtime.get_archive(_ARCHIVE_SPEC),
        )

        assert construct_calls == [_ARCHIVE_SPEC]
        assert first is spy
        assert second is spy
        assert runtime.archive_ever_constructed is True

    assert spy.aclose_calls == 1


async def test_closed_runtime_refuses_without_waiting_on_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_calls: list[Any] = []
    spy = SpyConnector()
    _patch_slow_factory(monkeypatch, spy, construct_calls)

    runtime = HealthRuntime(
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        }
    )
    await runtime.shutdown()

    # The closed guard sits outside the lock, so the refusal does not depend on
    # the lock being free: it is raised even while another task holds it.
    await runtime._lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(runtime.get_connector(), timeout=1)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(runtime.get_archive(_ARCHIVE_SPEC), timeout=1)
    finally:
        runtime._lock.release()

    assert construct_calls == []
