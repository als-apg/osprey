"""Unit tests for the bluesky panels sidecar's read-proxy router (task 1.2).

Exercises `read_proxy.router` mounted on a LOCAL FastAPI app (never the
package-level `osprey.interfaces.bluesky_panels.app.app`, which does not include
this router yet -- that wiring is a separate integration task). The bridge
HTTP layer is faked with `httpx.MockTransport` so no real network call is
made and no real Bluesky bridge process needs to be running.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_panels import read_proxy

_BRIDGE_URL = "http://bridge.test"


def _build_app(handler: Callable[[httpx.Request], httpx.Response]) -> FastAPI:
    """Build a local FastAPI app with the read-proxy router mounted, backed
    by a mock-transport client standing in for the real bridge.
    """
    app = FastAPI()
    app.include_router(read_proxy.router)
    app.state.bridge_url = _BRIDGE_URL
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def _json_response(status_code: int, body: object) -> httpx.Response:
    return httpx.Response(status_code, json=body)


# ---------------------------------------------------------------------------
# Round-trip passthrough for each of the 5 GET endpoints
# ---------------------------------------------------------------------------


def test_list_plans_round_trips_body_and_status() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(200, [{"name": "grid_scan", "provenance": "shipped"}])

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/plans")

    assert response.status_code == 200
    assert response.json() == [{"name": "grid_scan", "provenance": "shipped"}]
    assert len(seen) == 1
    assert str(seen[0].url) == f"{_BRIDGE_URL}/plans"


def test_get_plan_source_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/plans/grid_scan/source"
        return _json_response(
            200,
            {
                "name": "grid_scan",
                "provenance": "shipped",
                "validated": True,
                "truncated": False,
                "source": "def grid_scan(): ...",
            },
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/plans/grid_scan/source")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "grid_scan"
    assert body["source"] == "def grid_scan(): ..."


def test_list_runs_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, [{"run_id": "abc123", "status": "completed"}])

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs")

    assert response.status_code == 200
    assert response.json() == [{"run_id": "abc123", "status": "completed"}]


def test_get_run_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/abc123"
        return _json_response(
            200, {"run_id": "abc123", "status": "completed", "plan_name": "grid_scan"}
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/abc123")

    assert response.status_code == 200
    assert response.json()["run_id"] == "abc123"


def test_get_run_data_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/abc123/data"
        return _json_response(
            200,
            {
                "run_uid": "uid-1",
                "columns": ["x", "y"],
                "rows": [[1, 2]],
                "row_count": 1,
                "truncated": False,
            },
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/abc123/data")

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 1
    assert body["truncated"] is False


# ---------------------------------------------------------------------------
# Error passthrough (verbatim body + status, never recomputed)
# ---------------------------------------------------------------------------


def test_unknown_run_404_passes_through_verbatim() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(404, {"detail": "unknown run 'nope'"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown run 'nope'"}


def test_run_data_409_passes_through_verbatim() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(409, {"detail": "run 'abc123' has not started; no data yet"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/abc123/data")

    assert response.status_code == 409
    assert response.json() == {"detail": "run 'abc123' has not started; no data yet"}


def test_plan_source_404_passes_through_verbatim() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(404, {"detail": "no source file found for plan 'nope'"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/plans/nope/source")

    assert response.status_code == 404
    assert response.json() == {"detail": "no source file found for plan 'nope'"}


# ---------------------------------------------------------------------------
# Query-param forwarding
# ---------------------------------------------------------------------------


def test_list_runs_forwards_limit_param() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(200, [])

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs", params={"limit": "5"})

    assert response.status_code == 200
    assert seen[0].url.params["limit"] == "5"


def test_get_run_data_forwards_max_rows_offset_tail_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(
            200, {"run_uid": "uid-1", "columns": [], "rows": [], "row_count": 0, "truncated": False}
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get(
            "/runs/abc123/data", params={"max_rows": "50", "offset": "10", "tail": "true"}
        )

    assert response.status_code == 200
    assert seen[0].url.params["max_rows"] == "50"
    assert seen[0].url.params["offset"] == "10"
    assert seen[0].url.params["tail"] == "true"


# ---------------------------------------------------------------------------
# Bridge unreachable -> 502, never an uncaught 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/plans", "/plans/grid_scan/source", "/runs", "/runs/abc123", "/runs/abc123/data"],
)
def test_bridge_unreachable_returns_502(path: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}
