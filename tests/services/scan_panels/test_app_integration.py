"""Integration tests for the COMPOSED scan panels sidecar app (task 4.1).

The per-router unit tests in this directory (``test_read_proxy.py``,
``test_execute.py``, ``test_health.py``, ``test_health_full.py``) each mount
a single router onto a locally-built ``FastAPI()`` instance. This module
instead exercises the package-level ``osprey.services.scan_panels.app:app``
-- the object actually served in production -- to catch wiring bugs that a
per-router test can't see: router composition, static-mount registration,
and the shared design-system/fonts assets.

``TestClient(app)`` is entered as a context manager so the app's real
``_lifespan`` runs (it sets ``app.state.client``/``app.state.bridge_url`` from
env/config resolution, mirroring production startup). Once inside the
context, each test overwrites ``app.state.client`` with an
``httpx.AsyncClient(transport=httpx.MockTransport(...))`` and
``app.state.bridge_url`` with a fixed test URL, since every route reads those
two attributes off ``request.app.state`` at request time (not at import or
lifespan time) -- see ``read_proxy._forward_get``, ``execute.execute_run``,
and ``health._probe_bridge_http``. The lifespan's own ``finally: await
client.aclose()`` still closes the ORIGINAL client object it created (it
holds a local reference, not ``app.state.client``), so no double-close or
leak occurs when a test swaps the app-state client out from under it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.services.scan_panels.app import app

TOKEN = "s3cr3t-promote-token"  # noqa: S105 - test fixture value, not a real secret
RUN_ID = "run-xyz789"
_BRIDGE_URL = "http://bridge.test"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Mirrors test_execute.py's isolation fixture: point config resolution at
    # a config.yml that does not exist, so ambient repo/user config can never
    # leak a real promote token (or bridge URL) into these tests.
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)


def _wire_mock_bridge(handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Overwrite the (already-lifespan-started) composed app's bridge client.

    Must be called from inside a ``with TestClient(app) as client:`` block.
    """
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.bridge_url = _BRIDGE_URL


def _refusing_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"bridge must not be called: {request.method} {request.url}")


# ---------------------------------------------------------------------------
# Read-proxy round-trips through the wired (composed) app
# ---------------------------------------------------------------------------


def test_list_plans_round_trips_through_composed_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/plans"
        return httpx.Response(200, json=[{"name": "grid_scan", "provenance": "shipped"}])

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/plans")

    assert response.status_code == 200
    assert response.json() == [{"name": "grid_scan", "provenance": "shipped"}]


def test_get_plan_source_round_trips_through_composed_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/plans/grid_scan/source"
        return httpx.Response(200, json={"name": "grid_scan", "source": "def grid_scan(): ..."})

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/plans/grid_scan/source")

    assert response.status_code == 200
    assert response.json()["source"] == "def grid_scan(): ..."


def test_list_runs_round_trips_through_composed_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs"
        return httpx.Response(200, json=[{"run_id": "abc123", "status": "completed"}])

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/runs")

    assert response.status_code == 200
    assert response.json() == [{"run_id": "abc123", "status": "completed"}]


def test_get_run_status_round_trips_through_composed_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/abc123"
        return httpx.Response(200, json={"run_id": "abc123", "status": "completed"})

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/runs/abc123")

    assert response.status_code == 200
    assert response.json()["run_id"] == "abc123"


def test_read_run_data_round_trips_through_composed_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/abc123/data"
        return httpx.Response(
            200,
            json={
                "run_uid": "uid-1",
                "columns": ["x"],
                "rows": [[1]],
                "row_count": 1,
                "truncated": False,
            },
        )

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/runs/abc123/data")

    assert response.status_code == 200
    assert response.json()["row_count"] == 1


# ---------------------------------------------------------------------------
# Execute end-to-end on the composed app
# ---------------------------------------------------------------------------


def test_execute_armed_end_to_end_on_composed_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json={"id": RUN_ID, "status": "intent"})
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/promote":
            assert request.headers.get("x-promote-token") == TOKEN
            return httpx.Response(200, json={"id": RUN_ID, "status": "running"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.post(
            "/runs/execute", json={"plan_name": "response_matrix", "plan_args": {}}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == RUN_ID
    assert data["status"] == "running"

    # The token must never leak into the response body or headers.
    assert TOKEN not in response.text
    for header_name, header_value in response.headers.items():
        assert TOKEN not in header_value, f"token leaked in header {header_name!r}"

    assert len(calls) == 2


def test_execute_unarmed_on_composed_app_is_inert() -> None:
    with TestClient(app) as client:
        _wire_mock_bridge(_refusing_handler)
        response = client.post(
            "/runs/execute", json={"plan_name": "response_matrix", "plan_args": {}}
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }


# ---------------------------------------------------------------------------
# /health/full on the wired app (bridge + tiled mocked, VA TCP monkeypatched)
# ---------------------------------------------------------------------------


class _FakeWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


async def _fake_open_connection_ok(_host: str, _port: int) -> tuple[object, _FakeWriter]:
    return object(), _FakeWriter()


def test_health_full_rollup_on_composed_app(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setattr(asyncio, "open_connection", _fake_open_connection_ok)
    monkeypatch.setenv("SCAN_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("SCAN_PANELS_VA_ADDR", "va.test:5064")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/healthz":
            return httpx.Response(200, text="ok")
        raise AssertionError(f"unexpected probe URL: {request.url}")

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/health/full")

    assert response.status_code == 200
    body = response.json()
    assert body["rollup"] == "ok"
    statuses = {entry["name"]: entry["status"] for entry in body["services"]}
    assert statuses == {"bridge": "ok", "tiled": "ok", "va_ioc": "ok"}


# ---------------------------------------------------------------------------
# Negative safety assertions on the FULL route table of the composed app
# ---------------------------------------------------------------------------


def test_no_route_ends_with_stop() -> None:
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            assert not path.endswith("/stop"), f"composed app must not expose a stop route: {path}"


def test_only_post_under_runs_is_execute() -> None:
    post_run_paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if path and path.startswith("/runs") and "POST" in methods:
            post_run_paths.add(path)
    assert post_run_paths == {"/runs/execute"}, (
        "composed app must expose exactly one POST route under /runs "
        f"(/runs/execute), found: {post_run_paths}"
    )


# ---------------------------------------------------------------------------
# Panel-asset wiring (shared design-system + fonts, panel mounts registered)
# ---------------------------------------------------------------------------


def test_design_system_and_fonts_are_served() -> None:
    with TestClient(app) as client:
        css_response = client.get("/design-system/css/tokens.css")
        fonts_response = client.get("/static/fonts/fonts.css")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert fonts_response.status_code == 200
    assert "text/css" in fonts_response.headers["content-type"]


def test_panel_mounts_are_registered_on_composed_app() -> None:
    mounted_paths = {route.path for route in app.routes if hasattr(route, "path")}
    for mount_path in ("/plan", "/results", "/health-panel"):
        assert mount_path in mounted_paths
