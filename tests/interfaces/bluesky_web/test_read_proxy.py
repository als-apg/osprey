"""Unit tests for the bluesky-web sidecar's read-proxy router (task 1.2).

Exercises `read_proxy.router` mounted on a LOCAL FastAPI app (never the
package-level `osprey.interfaces.bluesky_web.app.app`, which does not include
this router yet -- that wiring is a separate integration task). The bridge
HTTP layer is faked with `httpx.MockTransport` so no real network call is
made and no real Bluesky bridge process needs to be running.
"""

from __future__ import annotations

from functools import partial

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_web import read_proxy
from tests.interfaces.bluesky_web._relay import BRIDGE_URL, build_router_app

_build_app = partial(build_router_app, read_proxy.router)


# ---------------------------------------------------------------------------
# Round-trip passthrough for each GET endpoint
# ---------------------------------------------------------------------------

_FIGURE = {
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


@pytest.mark.parametrize(
    ("path", "body"),
    [
        pytest.param("/plans", [{"name": "grid_scan", "provenance": "shipped"}], id="plans"),
        # The operator half of device discovery: a panel reads the worker's
        # device names through the same relay the agent's tool reads them
        # through. The bridge answers a paginated envelope, and the proxy hands
        # it back whole -- counts and window included, not just the page.
        pytest.param(
            "/devices",
            {
                "devices": [{"name": "COR1", "is_movable": True}],
                "total": 1,
                "offset": 0,
                "limit": 500,
            },
            id="devices",
        ),
        pytest.param(
            "/plans/grid_scan/source",
            {
                "name": "grid_scan",
                "provenance": "shipped",
                "validated": True,
                "truncated": False,
                "source": "def grid_scan(): ...",
            },
            id="plan-source",
        ),
        pytest.param("/runs", [{"run_id": "abc123", "status": "completed"}], id="runs"),
        pytest.param(
            "/runs/abc123",
            {"run_id": "abc123", "status": "completed", "plan_name": "grid_scan"},
            id="run",
        ),
        pytest.param(
            "/runs/abc123/data",
            {
                "run_uid": "uid-1",
                "columns": ["x", "y"],
                "rows": [[1, 2]],
                "row_count": 1,
                "truncated": False,
            },
            id="run-data",
        ),
        # The figure body is the bridge's `Figure.model_dump()` and nothing
        # else -- the proxy relays every field, panels and marks included.
        pytest.param("/runs/abc123/figure", _FIGURE, id="run-figure"),
        # A figure the plan's own `render` could not draw still comes back 200
        # with a `reason` and empty panels -- the proxy must not turn that into
        # an error or invent panels for it.
        pytest.param(
            "/runs/abc123/figure",
            {"panels": [], "partial": True, "source": "tiled", "reason": "source_unavailable"},
            id="run-figure-with-reason",
        ),
    ],
)
def test_get_round_trips_body_and_status(path: str, body: object) -> None:
    """Each GET reaches the same bridge path and relays its body untouched."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json() == body
    assert len(seen) == 1
    assert str(seen[0].url) == f"{BRIDGE_URL}{path}"


def test_get_run_figure_quotes_run_id() -> None:
    """Run ids reach the bridge escaped into a single path segment, exactly as
    they do for `/data`. `#` is the sharp case: unquoted it would truncate the
    outgoing URL to `/runs/a` and swallow the route entirely."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"panels": [], "partial": False, "source": "live", "reason": None}
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/runs/a%23b/figure")

    assert response.status_code == 200
    assert seen[0].url.raw_path == b"/runs/a%23b/figure"


# ---------------------------------------------------------------------------
# Error passthrough (verbatim body + status, never recomputed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "status", "body"),
    [
        pytest.param("/runs/nope", 404, {"detail": "unknown run 'nope'"}, id="unknown-run"),
        pytest.param(
            "/runs/abc123/data",
            409,
            {"detail": "run 'abc123' has not started; no data yet"},
            id="run-data-not-started",
        ),
        # The MCP tool maps this to `unknown_run` by the same string `/data`'s
        # 404 carries.
        pytest.param(
            "/runs/nope/figure", 404, {"detail": "unknown run 'nope'"}, id="run-figure-unknown"
        ),
        pytest.param(
            "/plans/nope/source",
            404,
            {"detail": "no source file found for plan 'nope'"},
            id="plan-source-unknown",
        ),
    ],
)
def test_bridge_error_passes_through_verbatim(path: str, status: int, body: object) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get(path)

    assert seen[0].url.path == path
    assert response.status_code == status
    assert response.json() == body


# ---------------------------------------------------------------------------
# Query-param forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "params", "body"),
    [
        # All three pagination knobs reach the bridge: a panel that pages
        # through a large device list is only as good as the params the relay
        # carries.
        pytest.param(
            "/devices",
            {"prefix": "COR", "limit": "2", "offset": "1"},
            {"devices": [], "total": 0, "offset": 1, "limit": 2},
            id="devices",
        ),
        pytest.param("/runs", {"limit": "5"}, [], id="runs"),
        pytest.param(
            "/runs/abc123/data",
            {"max_rows": "50", "offset": "10", "tail": "true"},
            {"run_uid": "uid-1", "columns": [], "rows": [], "row_count": 0, "truncated": False},
            id="run-data",
        ),
    ],
)
def test_query_params_are_forwarded(path: str, params: dict[str, str], body: object) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get(path, params=params)

    assert response.status_code == 200
    assert {key: seen[0].url.params[key] for key in params} == params


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
