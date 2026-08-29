"""Unit tests for the bluesky-web sidecar's read-proxy router (task 1.2).

Exercises `read_proxy.router` mounted on a LOCAL FastAPI app (never the
package-level `osprey.interfaces.bluesky_web.app.app`, which does not include
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

from osprey.interfaces.bluesky_web import read_proxy

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
# Round-trip passthrough for each GET endpoint
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


def test_list_devices_round_trips_body_and_status() -> None:
    """The operator half of device discovery: a panel reads the worker's device
    names through the same relay the agent's tool reads them through. The
    bridge answers a paginated envelope, and the proxy hands it back whole --
    counts and window included, not just the page of entries."""
    envelope = {
        "devices": [{"name": "COR1", "is_movable": True}],
        "total": 1,
        "offset": 0,
        "limit": 500,
    }
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(200, envelope)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/devices")

    assert response.status_code == 200
    assert response.json() == envelope
    assert str(seen[0].url) == f"{_BRIDGE_URL}/devices"


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


def test_get_run_figure_round_trips() -> None:
    """The figure body is the bridge's `Figure.model_dump()` and nothing else --
    the proxy relays every field, panels and marks included, untouched."""
    figure = {
        "panels": [
            {
                "title": "COR1 vs BPM1",
                "x_label": "COR1",
                "y_label": "BPM1",
                "x_units": "A",
                "y_units": "mm",
                "annotations": ["slope = 1.2 mm/A"],
                "mark": {
                    "kind": "lines",
                    "series": [
                        {
                            "label": "BPM1",
                            "points": [[0.0, 1.0], [1.0, 2.2]],
                            "decimated": False,
                            "source_points": 2,
                        }
                    ],
                },
            }
        ],
        "partial": False,
        "source": "tiled",
        "reason": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/abc123/figure"
        return _json_response(200, figure)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/abc123/figure")

    assert response.status_code == 200
    assert response.json() == figure


def test_get_run_figure_relays_reason_and_partial_unreshaped() -> None:
    """A figure the plan's own `render` could not draw still comes back 200 with
    a `reason` and empty panels -- the proxy must not turn that into an error or
    invent panels for it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {
                "panels": [],
                "partial": True,
                "source": "tiled",
                "reason": "source_unavailable",
            },
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/abc123/figure")

    assert response.status_code == 200
    assert response.json() == {
        "panels": [],
        "partial": True,
        "source": "tiled",
        "reason": "source_unavailable",
    }


def test_get_run_figure_quotes_run_id() -> None:
    """Run ids reach the bridge escaped into a single path segment, exactly as
    they do for `/data`. `#` is the sharp case: unquoted it would truncate the
    outgoing URL to `/runs/a` and swallow the route entirely."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(
            200, {"panels": [], "partial": False, "source": "live", "reason": None}
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/a%23b/figure")

    assert response.status_code == 200
    assert seen[0].url.raw_path == b"/runs/a%23b/figure"


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


def test_run_figure_404_passes_through_verbatim() -> None:
    """The unknown-run 404 body must survive the relay unchanged: the MCP tool
    maps it to `unknown_run` by the same string `/data`'s 404 carries."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/runs/nope/figure"
        return _json_response(404, {"detail": "unknown run 'nope'"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/nope/figure")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown run 'nope'"}


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


def test_list_devices_forwards_prefix_limit_offset_params() -> None:
    """All three pagination knobs reach the bridge: a panel that pages through
    a large device list is only as good as the params the relay carries."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(200, {"devices": [], "total": 0, "offset": 1, "limit": 2})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/devices", params={"prefix": "COR", "limit": "2", "offset": "1"})

    assert response.status_code == 200
    assert seen[0].url.params["prefix"] == "COR"
    assert seen[0].url.params["limit"] == "2"
    assert seen[0].url.params["offset"] == "1"


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
    [
        "/plans",
        "/plans/grid_scan/source",
        "/runs",
        "/runs/abc123",
        "/runs/abc123/data",
        "/runs/abc123/figure",
    ],
)
def test_bridge_unreachable_returns_502(path: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}
