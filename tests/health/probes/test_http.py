"""Tests for the ``http`` health probe (task 2.1).

Drives :func:`osprey.health.probes.http.run` through an injected
:class:`httpx.MockTransport` so no real network is touched:

- status match → ``ok``; mismatch → ``error`` carrying the observed code;
- latency banding against ``warn_latency_ms`` / ``error_latency_ms`` ceilings;
- request timeout → configured ``timeout_status`` (default ``error``, opt-in
  ``warning``);
- transport failure (connection refused) → ``error`` with ``str(exc)`` details;
- ``latency_ms`` is set on every branch.

Also exercises the lazy registry (:func:`osprey.health.probes.get_probe`).
"""

from __future__ import annotations

import httpx
import pytest

from osprey.health.models import Status
from osprey.health.probes import ProbeContext, get_probe
from osprey.health.probes.http import run
from osprey.health.runtime import HealthRuntime


def _ctx() -> ProbeContext:
    """A probe context with a real (never-constructed) runtime; http ignores it."""
    return ProbeContext(runtime=HealthRuntime({}))


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_status_match_is_ok() -> None:
    transport = _transport(lambda req: httpx.Response(200, text="pong"))
    result = await run({"url": "http://svc/health"}, _ctx(), transport=transport)

    assert result.status is Status.OK
    assert result.value == "200"
    assert result.latency_ms > 0


async def test_custom_expect_status_match() -> None:
    transport = _transport(lambda req: httpx.Response(204))
    result = await run(
        {"url": "http://svc/health", "expect_status": 204},
        _ctx(),
        transport=transport,
    )

    assert result.status is Status.OK
    assert result.value == "204"


async def test_status_mismatch_is_error() -> None:
    transport = _transport(lambda req: httpx.Response(503))
    result = await run({"url": "http://svc/health"}, _ctx(), transport=transport)

    assert result.status is Status.ERROR
    assert result.value == "503"
    assert "expected 200" in result.message
    assert result.latency_ms > 0


async def test_latency_within_ceilings_is_ok() -> None:
    transport = _transport(lambda req: httpx.Response(200))
    result = await run(
        {"url": "http://svc/", "warn_latency_ms": 1e9, "error_latency_ms": 1e9},
        _ctx(),
        transport=transport,
    )

    assert result.status is Status.OK


async def test_latency_over_warn_ceiling_is_warning() -> None:
    # A zero warn ceiling is exceeded by any real elapsed time.
    transport = _transport(lambda req: httpx.Response(200))
    result = await run(
        {"url": "http://svc/", "warn_latency_ms": 0.0},
        _ctx(),
        transport=transport,
    )

    assert result.status is Status.WARNING
    assert "warn ceiling" in result.message


async def test_latency_over_error_ceiling_is_error() -> None:
    transport = _transport(lambda req: httpx.Response(200))
    result = await run(
        {"url": "http://svc/", "error_latency_ms": 0.0},
        _ctx(),
        transport=transport,
    )

    assert result.status is Status.ERROR
    assert "error ceiling" in result.message


async def test_error_ceiling_takes_precedence_over_warn() -> None:
    transport = _transport(lambda req: httpx.Response(200))
    result = await run(
        {"url": "http://svc/", "warn_latency_ms": 0.0, "error_latency_ms": 0.0},
        _ctx(),
        transport=transport,
    )

    assert result.status is Status.ERROR


async def test_timeout_defaults_to_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=req)

    result = await run(
        {"url": "http://svc/", "timeout_s": 0.5},
        _ctx(),
        transport=_transport(handler),
    )

    assert result.status is Status.ERROR
    assert "timed out" in result.message
    assert result.details
    assert result.latency_ms > 0


async def test_timeout_status_warning_opt_in() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=req)

    result = await run(
        {"url": "http://svc/", "timeout_status": "warning"},
        _ctx(),
        transport=_transport(handler),
    )

    assert result.status is Status.WARNING


async def test_connection_refused_is_error_with_details() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=req)

    result = await run({"url": "http://svc/"}, _ctx(), transport=_transport(handler))

    assert result.status is Status.ERROR
    assert "failed" in result.message
    assert "Connection refused" in result.details
    assert result.latency_ms > 0


async def test_custom_name_and_category_flow_through() -> None:
    transport = _transport(lambda req: httpx.Response(200))
    result = await run(
        {"url": "http://svc/", "name": "web.terminal", "category": "web"},
        _ctx(),
        transport=transport,
    )

    assert result.name == "web.terminal"
    assert result.category == "web"


async def test_get_probe_resolves_http_lazily() -> None:
    probe = get_probe("http")
    assert probe is run


def test_get_probe_unknown_type_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_probe("nope")
