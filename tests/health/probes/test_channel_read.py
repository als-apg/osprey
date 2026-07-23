"""Tests for the ``channel_read`` probe (task 2.4).

Pins the connector-mediated read contract: bands (ok/warn/error, expect
equality, non-numeric guard), units appended from ``ChannelValue.metadata``,
read-failure → error, the explicit ``timeout_s`` pass-through, and — the safety
crux — that the probe *only ever reads*: a spy connector fails loudly on any
``write_channel``/``subscribe``, and a spy runtime proves the connector is
acquired lazily via ``ctx.runtime.get_connector()``. One test drives the real
in-tree :class:`MockConnector` end to end through a real ``HealthRuntime``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from osprey.connectors.control_system.base import ChannelMetadata, ChannelValue
from osprey.health.models import Status
from osprey.health.probes import ProbeContext, get_probe
from osprey.health.probes.channel_read import run
from osprey.health.runtime import HealthRuntime


class _SpyConnector:
    """Records reads, returns a preset value, and forbids any write/subscribe."""

    def __init__(self, value: Any = 1.0, units: str = "", raise_exc: Exception | None = None):
        self._value = value
        self._units = units
        self._raise_exc = raise_exc
        self.read_calls: list[tuple[str, Any]] = []
        self.write_calls = 0
        self.subscribe_calls = 0

    async def read_channel(self, address: str, timeout: float | None = None) -> ChannelValue:
        self.read_calls.append((address, timeout))
        if self._raise_exc is not None:
            raise self._raise_exc
        return ChannelValue(
            value=self._value,
            timestamp=datetime(2026, 1, 1),
            metadata=ChannelMetadata(units=self._units),
        )

    async def write_channel(self, *args: Any, **kwargs: Any) -> Any:
        self.write_calls += 1
        raise AssertionError("channel_read probe must never write")

    async def subscribe(self, *args: Any, **kwargs: Any) -> Any:
        self.subscribe_calls += 1
        raise AssertionError("channel_read probe must never subscribe")

    async def disconnect(self) -> None:
        pass


class _SpyRuntime:
    """Lazy connector owner: hands out the connector only when asked."""

    def __init__(self, connector: Any):
        self._connector = connector
        self.get_connector_calls = 0

    async def get_connector(self) -> Any:
        self.get_connector_calls += 1
        return self._connector


def _ctx(connector: Any) -> tuple[ProbeContext, _SpyRuntime]:
    runtime = _SpyRuntime(connector)
    return ProbeContext(runtime=runtime), runtime  # type: ignore[arg-type]


async def _run(spec: dict[str, Any], connector: Any):
    ctx, _runtime = _ctx(connector)
    return await run(spec, ctx)


# --- Registry wiring --------------------------------------------------------


def test_probe_is_registered() -> None:
    assert get_probe("channel_read") is run


# --- Bands ------------------------------------------------------------------


async def test_ok_when_no_bands() -> None:
    result = await _run({"address": "BEAM:CURRENT"}, _SpyConnector(value=1.0))
    assert result.status is Status.OK
    assert result.name == "channel_read"
    assert result.category == "channel_read"
    assert "1.0" in result.message
    assert result.latency_ms > 0  # latency measured on the success branch


async def test_units_appended_to_value() -> None:
    result = await _run({"address": "BEAM:CURRENT"}, _SpyConnector(value=401.2, units="mA"))
    assert result.status is Status.OK
    assert result.value == "401.2 mA"


async def test_inside_ok_range_is_ok() -> None:
    result = await _run(
        {"address": "X", "ok_range": [0, 10], "warn_range": [-5, 20]},
        _SpyConnector(value=5),
    )
    assert result.status is Status.OK


async def test_outside_ok_inside_warn_is_warning() -> None:
    result = await _run(
        {"address": "X", "ok_range": [0, 10], "warn_range": [0, 20]},
        _SpyConnector(value=12),
    )
    assert result.status is Status.WARNING
    assert "outside ok range" in result.message


async def test_outside_warn_is_error() -> None:
    result = await _run(
        {"address": "X", "ok_range": [0, 10], "warn_range": [0, 20]},
        _SpyConnector(value=25),
    )
    assert result.status is Status.ERROR
    assert "outside warn range" in result.message


async def test_only_ok_range_outside_is_warning() -> None:
    result = await _run({"address": "X", "ok_range": [0, 10]}, _SpyConnector(value=12))
    assert result.status is Status.WARNING


async def test_only_warn_range_outside_is_error() -> None:
    result = await _run({"address": "X", "warn_range": [0, 20]}, _SpyConnector(value=25))
    assert result.status is Status.ERROR


async def test_range_boundaries_are_inclusive() -> None:
    result = await _run({"address": "X", "ok_range": [0, 10]}, _SpyConnector(value=10))
    assert result.status is Status.OK


async def test_non_numeric_value_with_range_is_error() -> None:
    result = await _run({"address": "X", "ok_range": [0, 10]}, _SpyConnector(value="oops"))
    assert result.status is Status.ERROR
    assert "not numeric" in result.message


# --- Expect equality --------------------------------------------------------


async def test_expect_match_is_ok() -> None:
    result = await _run({"address": "S", "expect": "ENABLED"}, _SpyConnector(value="ENABLED"))
    assert result.status is Status.OK


async def test_expect_mismatch_is_error() -> None:
    result = await _run({"address": "S", "expect": "ENABLED"}, _SpyConnector(value="DISABLED"))
    assert result.status is Status.ERROR
    assert "expected ENABLED" in result.message


# --- Read failure -----------------------------------------------------------


async def test_read_failure_is_error_with_details() -> None:
    connector = _SpyConnector(raise_exc=TimeoutError("read timed out after 5s"))
    result = await _run({"address": "BEAM:CURRENT"}, connector)
    assert result.status is Status.ERROR
    assert "read BEAM:CURRENT failed" in result.message
    assert "read timed out after 5s" in result.details
    assert result.latency_ms > 0  # latency measured on the read-failure branch


# --- Safety contract: read-only, lazy acquisition ---------------------------


async def test_probe_never_writes_or_subscribes() -> None:
    connector = _SpyConnector(value=1.0)
    await _run({"address": "X", "ok_range": [0, 10]}, connector)
    assert connector.write_calls == 0
    assert connector.subscribe_calls == 0
    assert len(connector.read_calls) == 1


async def test_timeout_passed_to_read_channel() -> None:
    connector = _SpyConnector(value=1.0)
    await _run({"address": "X", "timeout_s": 2.5}, connector)
    assert connector.read_calls == [("X", 2.5)]


async def test_connector_acquired_lazily() -> None:
    connector = _SpyConnector(value=1.0)
    ctx, runtime = _ctx(connector)
    assert runtime.get_connector_calls == 0  # nothing acquired before the run
    await run({"address": "X"}, ctx)
    assert runtime.get_connector_calls == 1  # acquired exactly once, on demand


# --- Real in-tree MockConnector end to end ----------------------------------


async def test_real_mock_connector_read_is_ok() -> None:
    runtime = HealthRuntime({"type": "mock", "connector": {"mock": {"noise_level": 0}}})
    try:
        ctx = ProbeContext(runtime=runtime)
        result = await run({"address": "BEAM:CURRENT", "name": "beam"}, ctx)
        assert result.status is Status.OK
        assert result.name == "beam"
        assert result.value  # a reading was captured
    finally:
        await runtime.shutdown()
