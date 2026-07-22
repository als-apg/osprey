"""Tests for the core ``openobserve`` health category (task 4.5).

Drives the category's async healthz probe through an injected
:class:`httpx.MockTransport`, and exercises the deploy gate, retention banding,
and endpoint construction from ``bind_address`` + ``port``.
"""

from __future__ import annotations

import httpx

from osprey.health.core.openobserve import openobserve
from osprey.health.models import CheckResult, Status


async def _run(config, *, transport=None) -> dict[str, CheckResult]:
    results = await openobserve(config, transport=transport)()
    assert isinstance(results, list)
    return {r.name: r for r in results}


def _ok_transport(captured: list[str] | None = None) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(str(req.url))
        return httpx.Response(200)

    return httpx.MockTransport(handler)


async def test_no_rows_when_not_deployed() -> None:
    by_name = await _run({"deployed_services": ["web"]}, transport=_ok_transport())
    assert by_name == {}


async def test_no_rows_when_config_none() -> None:
    by_name = await _run(None, transport=_ok_transport())
    assert by_name == {}


async def test_deployed_emits_both_rows() -> None:
    config = {"deployed_services": ["openobserve"]}
    by_name = await _run(config, transport=_ok_transport())
    assert set(by_name) == {"openobserve_healthz", "openobserve_retention"}
    assert all(r.category == "openobserve" for r in by_name.values())


async def test_healthz_ok_on_200() -> None:
    config = {"deployed_services": ["openobserve"]}
    row = (await _run(config, transport=_ok_transport()))["openobserve_healthz"]
    assert row.status is Status.OK
    assert "ready" in row.message


async def test_healthz_warns_on_non_200() -> None:
    config = {"deployed_services": ["openobserve"]}
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    row = (await _run(config, transport=transport))["openobserve_healthz"]
    assert row.status is Status.WARNING
    assert "503" in row.message


async def test_healthz_warns_when_unreachable() -> None:
    config = {"deployed_services": ["openobserve"]}

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=req)

    row = (await _run(config, transport=httpx.MockTransport(handler)))["openobserve_healthz"]
    assert row.status is Status.WARNING
    assert "unreachable" in row.message
    assert row.details  # carries the remediation hint


async def test_healthz_url_uses_bind_and_port() -> None:
    config = {
        "deployed_services": ["openobserve"],
        "services": {"openobserve": {"port": 9999}},
        "deployment": {"bind_address": "10.0.0.5"},
    }
    captured: list[str] = []
    await _run(config, transport=_ok_transport(captured))
    assert captured == ["http://10.0.0.5:9999/healthz"]


async def test_healthz_url_defaults() -> None:
    config = {"deployed_services": ["openobserve"]}
    captured: list[str] = []
    await _run(config, transport=_ok_transport(captured))
    assert captured == ["http://127.0.0.1:5080/healthz"]


async def test_retention_default_is_ok() -> None:
    config = {"deployed_services": ["openobserve"]}
    row = (await _run(config, transport=_ok_transport()))["openobserve_retention"]
    assert row.status is Status.OK
    assert "14" in row.message


async def test_retention_below_floor_warns() -> None:
    config = {
        "deployed_services": ["openobserve"],
        "services": {"openobserve": {"retention_days": 1}},
    }
    row = (await _run(config, transport=_ok_transport()))["openobserve_retention"]
    assert row.status is Status.WARNING
    assert "floor of 3" in row.message
    assert row.details


async def test_retention_at_floor_is_ok() -> None:
    config = {
        "deployed_services": ["openobserve"],
        "services": {"openobserve": {"retention_days": 3}},
    }
    row = (await _run(config, transport=_ok_transport()))["openobserve_retention"]
    assert row.status is Status.OK
