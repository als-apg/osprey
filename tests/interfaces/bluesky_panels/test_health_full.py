"""Unit tests for the bluesky panels sidecar's full dependency healthcheck
(task 1.4, ``GET /health/full``).

Builds a local FastAPI app (not the real sidecar app) with the health router
mounted directly, so the tests are independent of the app skeleton's lifespan
and panel mounts. HTTP dependencies (bridge, Tiled) are mocked via
``httpx.MockTransport`` on ``app.state.client``; the VA IOC's raw TCP probe
is mocked by monkeypatching ``asyncio.open_connection``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_panels import health


class _FakeWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


async def _fake_open_connection_ok(_host: str, _port: int) -> tuple[object, _FakeWriter]:
    return object(), _FakeWriter()


async def _fake_open_connection_refused(_host: str, _port: int) -> tuple[object, object]:
    raise OSError("Connection refused")


async def _fake_open_connection_timeout(_host: str, _port: int) -> tuple[object, object]:
    raise TimeoutError("connect timed out")


def _make_app(
    handler: Callable[[httpx.Request], httpx.Response],
    bridge_url: str = "http://bridge.test",
) -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    app.state.bridge_url = bridge_url
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def _all_ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/healthz":
        return httpx.Response(200, text="ok")
    raise AssertionError(f"unexpected probe URL: {request.url}")


def test_all_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    app = _make_app(_all_ok_handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    assert response.status_code == 200
    body = response.json()
    assert body["rollup"] == "ok"
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses == {"bridge": "ok", "tiled": "ok", "va_ioc": "ok"}
    names = {entry["name"] for entry in body["services"]}
    assert names == {"bridge", "tiled", "va_ioc"}
    for entry in body["services"]:
        assert isinstance(entry["latency_ms"], float)
        assert entry["latency_ms"] >= 0
        assert isinstance(entry["detail"], str) and entry["detail"]


def test_bridge_down_via_non_200_degrades_only_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503, text="service unavailable")
        if request.url.path == "/healthz":
            return httpx.Response(200, text="ok")
        raise AssertionError(f"unexpected probe URL: {request.url}")

    app = _make_app(handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    assert response.status_code == 200
    body = response.json()
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses["bridge"] == "unhealthy"
    assert statuses["tiled"] == "ok"
    assert statuses["va_ioc"] == "ok"
    assert body["rollup"] == "unhealthy"


def test_bridge_down_via_connect_error_degrades_only_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/healthz":
            return httpx.Response(200, text="ok")
        raise AssertionError(f"unexpected probe URL: {request.url}")

    app = _make_app(handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    assert response.status_code == 200
    body = response.json()
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses["bridge"] == "unhealthy"
    assert body["rollup"] == "unhealthy"


def test_va_tcp_connect_ok_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    app = _make_app(_all_ok_handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    body = response.json()
    va_entry = next(entry for entry in body["services"] if entry["name"] == "va_ioc")
    assert va_entry["status"] == "ok"
    assert body["rollup"] == "ok"


@pytest.mark.parametrize(
    "fake_connect",
    [_fake_open_connection_refused, _fake_open_connection_timeout],
)
def test_va_tcp_connect_failure_degrades_rollup(
    monkeypatch: pytest.MonkeyPatch,
    fake_connect: Callable,
) -> None:
    monkeypatch.setattr(asyncio, "open_connection", fake_connect)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    app = _make_app(_all_ok_handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    assert response.status_code == 200
    body = response.json()
    va_entry = next(entry for entry in body["services"] if entry["name"] == "va_ioc")
    assert va_entry["status"] == "unhealthy"
    assert body["rollup"] == "unhealthy"
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses["bridge"] == "ok"
    assert statuses["tiled"] == "ok"


def test_endpoint_never_500s_when_all_probes_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_refused)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/healthz":
            return httpx.Response(500, text="internal error")
        raise AssertionError(f"unexpected probe URL: {request.url}")

    app = _make_app(handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    assert response.status_code == 200
    body = response.json()
    assert body["rollup"] == "unhealthy"
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses == {"bridge": "unhealthy", "tiled": "unhealthy", "va_ioc": "unhealthy"}


def test_rollup_is_worst_of_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    # One unhealthy service anywhere in the trio must degrade the rollup,
    # regardless of which service it is.
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/healthz":
            return httpx.Response(503, text="unavailable")
        raise AssertionError(f"unexpected probe URL: {request.url}")

    app = _make_app(handler)
    with TestClient(app) as client:
        response = client.get("/health/full")

    body = response.json()
    assert body["rollup"] == "unhealthy"
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses["tiled"] == "unhealthy"
    assert statuses["bridge"] == "ok"
    assert statuses["va_ioc"] == "ok"


def test_uses_app_state_bridge_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bridge probe must use request.app.state.bridge_url verbatim (it is
    # already env-resolved by app.py's lifespan) rather than re-resolving
    # BLUESKY_BRIDGE_URL itself.
    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/healthz":
            return httpx.Response(200, text="ok")
        raise AssertionError(f"unexpected probe URL: {request.url}")

    app = _make_app(handler, bridge_url="http://custom-bridge.test:9999")
    with TestClient(app) as client:
        client.get("/health/full")

    assert "http://custom-bridge.test:9999/health" in seen_urls
