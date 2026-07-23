"""Integration tests for the COMPOSED bluesky panels sidecar app (task 4.1).

The per-router unit tests in this directory (``test_read_proxy.py``,
``test_launch.py``, ``test_health.py``, ``test_health_full.py``) each mount
a single router onto a locally-built ``FastAPI()`` instance. This module
instead exercises the package-level ``osprey.interfaces.bluesky_panels.app:app``
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
lifespan time) -- see ``read_proxy._forward_get``, ``launch.launch_run``,
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

from osprey.interfaces.bluesky_panels.app import app

TOKEN = "s3cr3t-launch-token"  # noqa: S105 - test fixture value, not a real secret
RUN_ID = "run-xyz789"
_BRIDGE_URL = "http://bridge.test"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Mirrors test_launch.py's isolation fixture: point config resolution at
    # a config.yml that does not exist, so ambient repo/user config can never
    # leak a real launch token (or bridge URL) into these tests.
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)


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


def test_get_run_round_trips_through_composed_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/abc123"
        return httpx.Response(200, json={"run_id": "abc123", "status": "completed"})

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.get("/runs/abc123")

    assert response.status_code == 200
    assert response.json()["run_id"] == "abc123"


def test_get_run_data_round_trips_through_composed_app() -> None:
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
# Launch end-to-end on the composed app
# ---------------------------------------------------------------------------


def test_launch_armed_end_to_end_on_composed_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json={"id": RUN_ID, "status": "pending"})
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/launch":
            assert request.headers.get("x-launch-token") == TOKEN
            return httpx.Response(200, json={"id": RUN_ID, "status": "running"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.post("/runs/launch", json={"plan_name": "orm", "plan_args": {}})

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == RUN_ID
    assert data["status"] == "running"

    # The token must never leak into the response body or headers.
    assert TOKEN not in response.text
    for header_name, header_value in response.headers.items():
        assert TOKEN not in header_value, f"token leaked in header {header_name!r}"

    assert len(calls) == 2


def test_launch_unarmed_on_composed_app_is_inert() -> None:
    with TestClient(app) as client:
        _wire_mock_bridge(_refusing_handler)
        response = client.post("/runs/launch", json={"plan_name": "orm", "plan_args": {}})

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
    monkeypatch.setenv("BLUESKY_PANELS_TILED_URL", "http://tiled.test")
    monkeypatch.setenv("BLUESKY_PANELS_VA_ADDR", "va.test:5064")

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


def _served_routes() -> dict[str, set[str]]:
    """Return ``{path: {METHOD, ...}}`` from the composed app's OpenAPI schema.

    Enumerating the OpenAPI schema rather than ``app.routes`` keeps these
    negative safety assertions robust to Starlette's internal route
    representation: since Starlette 1.0, ``include_router`` stores opaque
    wrapper objects on the parent app instead of flattening the child routes,
    so ``app.routes`` no longer exposes a per-route ``.path``/``.methods`` for
    router-mounted endpoints (they surface only as method-less wrappers). The
    execute/read-proxy/health routes are all attached via ``include_router``,
    so an ``app.routes`` scan silently sees *zero* of them — passing the
    ``/stop`` check vacuously and failing the ``/runs/execute`` check. The
    OpenAPI schema is the public, version-stable contract for what the app
    actually serves. See ``test_scaffold_routes_registration._registered_paths``.
    """
    schema = app.openapi()
    return {
        path: {method.upper() for method in operations}
        for path, operations in schema["paths"].items()
    }


def test_no_route_ends_with_stop() -> None:
    for path in _served_routes():
        assert not path.endswith("/stop"), f"composed app must not expose a stop route: {path}"


def test_only_post_under_runs_is_launch() -> None:
    # PATCH/DELETE /draft are draft-scratch edits, not run launches, so they
    # deliberately live outside /runs and are exempt from this check; see
    # test_write_surface_is_exactly_launch_and_draft below for the full
    # cross-router write-surface invariant.
    post_run_paths = {
        path
        for path, methods in _served_routes().items()
        if path.startswith("/runs") and "POST" in methods
    }
    assert post_run_paths == {"/runs/launch"}, (
        "composed app must expose exactly one POST route under /runs "
        f"(/runs/launch), found: {post_run_paths}"
    )


def test_draft_routes_registered_on_composed_app() -> None:
    # Task 3.1 (sidecar-draft-relay): GET/PATCH/DELETE /draft plus the SSE
    # relay at /draft/events must be wired onto the composed app.
    paths = app.openapi()["paths"]
    assert "/draft" in paths
    assert "/draft/events" in paths
    assert {"get", "patch", "delete"} <= set(paths["/draft"].keys())
    assert set(paths["/draft/events"].keys()) == {"get"}


def test_write_surface_is_exactly_launch_and_draft() -> None:
    # The full non-GET/HEAD/OPTIONS route surface across every router
    # composed onto the sidecar app. /runs/launch is the sole run-launch
    # path (gated by the launch token + writes-enabled, see test_launch.py);
    # PATCH/DELETE /draft are draft-scratch writes relayed verbatim to the
    # bridge -- they never arm or launch a run. No other write verb may exist
    # anywhere in the composed app.
    write_paths = {
        (path, method)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method not in ("get", "head", "options")
    }
    assert write_paths == {
        ("/runs/launch", "post"),
        ("/draft", "patch"),
        ("/draft", "delete"),
    }, f"unexpected write surface: {write_paths}"


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


def test_every_response_forbids_browser_caching() -> None:
    """Panel assets ship unversioned filenames (panel.js), so any cached copy
    silently survives a container rebuild; the no-cache header is what makes a
    redeploy actually reach the operator's browser. Blanket: static panel
    bundles, design-system assets, and live JSON alike."""
    with TestClient(app) as client:
        for path in ("/plan/", "/health", "/design-system/css/tokens.css"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate", path
