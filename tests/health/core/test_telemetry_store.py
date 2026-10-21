"""Tests for the core ``telemetry_store`` health category.

The store is stood in for by an :class:`httpx.MockTransport`; the tests pin the
deploy gate and the address the row dials in both network shapes — the compose
network (declared by ``OSPREY_OTEL_GATEWAY_HOST=telemetry_gateway``) and the
host namespace (a host launch, or a container declaring a host address).
"""

from __future__ import annotations

import httpx

from osprey.deployment import telemetry_config as tc
from osprey.health.core.telemetry_store import CATEGORY, on_compose_network, telemetry_store
from osprey.health.models import CheckResult, Status

DEPLOYED = {"deployed_services": ["telemetry_store", "telemetry_gateway"]}
BRIDGE_ENV = {tc.GATEWAY_HOST_ENV: "telemetry_gateway", tc.GATEWAY_PORT_ENV: "4318"}
HOST_NET_ENV = {tc.GATEWAY_HOST_ENV: "10.0.0.5", tc.GATEWAY_PORT_ENV: "9051"}


def _transport(status: int = 200, seen: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(status, text="Ok.\n")

    return httpx.MockTransport(handler)


async def _run(config, environ, transport) -> list[CheckResult]:
    return await telemetry_store(config, transport=transport, environ=environ)()


async def test_not_deployed_contributes_no_rows():
    seen: list[str] = []
    assert await _run({"deployed_services": ["web"]}, {}, _transport(seen=seen)) == []
    assert await _run(None, {}, _transport(seen=seen)) == []
    assert seen == []


async def test_bridge_container_dials_the_service_name_and_container_port():
    seen: list[str] = []
    rows = await _run(DEPLOYED, BRIDGE_ENV, _transport(seen=seen))
    assert seen == [f"http://telemetry_store:{tc.STORE_HTTP_CONTAINER_PORT}/ping"]
    assert [(r.name, r.category, r.status) for r in rows] == [
        ("telemetry_store_ping", CATEGORY, Status.OK)
    ]


async def test_host_launch_dials_loopback_on_the_published_port():
    seen: list[str] = []
    config = {**DEPLOYED, "services": {"telemetry_store": {"port": 18123}}}
    rows = await _run(config, {}, _transport(seen=seen))
    assert seen == ["http://127.0.0.1:18123/ping"]
    assert rows[0].status == Status.OK


async def test_host_network_container_dials_loopback_whatever_the_bind_address():
    seen: list[str] = []
    config = {**DEPLOYED, "deployment": {"bind_address": "10.0.0.5"}}
    await _run(config, HOST_NET_ENV, _transport(seen=seen))
    assert seen == [f"http://127.0.0.1:{tc.store_port(config)}/ping"]


async def test_non_200_is_a_warning_naming_the_status():
    rows = await _run(DEPLOYED, {}, _transport(status=503))
    assert rows[0].status == Status.WARNING
    assert "503" in rows[0].message


async def test_unreachable_store_is_a_warning_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rows = await _run(DEPLOYED, BRIDGE_ENV, httpx.MockTransport(handler))
    assert rows[0].status == Status.WARNING
    assert "unreachable" in rows[0].message


def test_network_shape_is_read_from_the_declared_gateway_host():
    assert on_compose_network(BRIDGE_ENV)
    assert not on_compose_network(HOST_NET_ENV)
    assert not on_compose_network({})
