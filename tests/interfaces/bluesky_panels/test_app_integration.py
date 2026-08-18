"""Integration tests for the COMPOSED bluesky panels sidecar app.

The per-router unit tests in this directory (``test_read_proxy.py``,
``test_queue_relay.py``, ``test_health.py``) each mount
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
lifespan time) -- see ``read_proxy._forward_get`` and the queue relay's write
routes. The lifespan's own ``finally: await
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
    # Mirrors test_queue_relay.py's isolation fixture: point config resolution
    # at a config.yml that does not exist, so ambient repo/user config can never
    # leak a real launch token (or bridge URL) into these tests.
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)


def _wire_mock_bridge(handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Overwrite the (already-lifespan-started) composed app's bridge client.

    Must be called from inside a ``with TestClient(app) as client:`` block.
    """
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.bridge_url = _BRIDGE_URL


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
# Enqueue end-to-end on the composed app
# ---------------------------------------------------------------------------


def test_enqueue_armed_end_to_end_on_composed_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """The token is resolved in-process, reaches the bridge, and never comes back.

    Composed-app coverage of the sidecar's one arming-relevant hop. The
    per-route token rules live in ``test_queue_relay.py``; what this pins is
    that they survive router composition -- that the app actually served
    resolves the token off the environment and attaches it, rather than the
    property holding only on a hand-built single-router app.
    """
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/queue/items":
            assert request.headers.get("x-launch-token") == TOKEN
            return httpx.Response(200, json={"run_id": RUN_ID, "revision": 7})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    with TestClient(app) as client:
        _wire_mock_bridge(handler)
        response = client.post("/queue/items", json={"draft_revision": 7})

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID

    # The token must never leak into the response body or headers.
    assert TOKEN not in response.text
    for header_name, header_value in response.headers.items():
        assert TOKEN not in header_value, f"token leaked in header {header_name!r}"

    assert len(calls) == 1


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


def test_no_per_run_stop_route() -> None:
    # The route this forbids is a per-run stop (``POST /runs/{run_id}/stop``):
    # it would let the browser halt one operator's run without going through
    # the agent's own tooling, which is why the sidecar has never relayed it.
    #
    # ``POST /queue/stop`` is exempt, and is the opposite direction: it halts
    # the QUEUE after the running item finishes, and the bridge leaves a plain
    # stop UNGATED precisely because halting is always allowed. Only its
    # ``cancel: true`` form -- which withdraws a human's pending stop and lets
    # the queue keep draining toward hardware -- is an arming action, and the
    # bridge gates that one on the launch token. The sidecar relays both
    # verbatim and reads ``cancel`` itself never (see queue_relay.py).
    run_paths = [path for path in _served_routes() if path.startswith("/runs")]
    # Non-vacuity: narrowing this loop to /runs means an empty /runs surface
    # would make it pass without checking anything.
    assert run_paths, "no /runs routes to check — this guard has gone vacuous"
    for path in run_paths:
        assert not path.endswith("/stop"), f"composed app must not expose a per-run stop: {path}"


def test_no_post_route_under_runs() -> None:
    # /runs is a READ surface on the sidecar: the run list, one run, and its
    # data. Nothing starts a plan through it. The single relay that once did
    # (POST /runs/launch, composing the bridge's pending-run create + launch)
    # is gone with the bridge primitives it called -- those answer an
    # unconditional 410 use_the_queue now, so the relay could only ever hand
    # the operator a dead route. Enqueueing goes through POST /queue/items and
    # the queue's own arming gates.
    #
    # PATCH/DELETE /draft are draft-scratch edits, not run launches, so they
    # deliberately live outside /runs; see
    # test_write_surface_is_exactly_draft_and_queue below for the full
    # cross-router write-surface invariant.
    run_paths = {path for path in _served_routes() if path.startswith("/runs")}
    # Non-vacuity: an empty /runs surface would satisfy the check below without
    # checking anything (the same guard test_no_per_run_stop_route carries).
    assert run_paths, "no /runs routes to check — this guard has gone vacuous"
    post_run_paths = {
        path
        for path, methods in _served_routes().items()
        if path.startswith("/runs") and "POST" in methods
    }
    assert post_run_paths == set(), (
        f"composed app must expose no POST route under /runs, found: {post_run_paths}"
    )


def test_draft_routes_registered_on_composed_app() -> None:
    # The draft relay: GET/PATCH/DELETE /draft plus the SSE
    # relay at /draft/events must be wired onto the composed app.
    paths = app.openapi()["paths"]
    assert "/draft" in paths
    assert "/draft/events" in paths
    assert {"get", "patch", "delete"} <= set(paths["/draft"].keys())
    assert set(paths["/draft/events"].keys()) == {"get"}


def test_write_surface_is_exactly_draft_and_queue() -> None:
    # The full non-GET/HEAD/OPTIONS route surface across every router
    # composed onto the sidecar app. PATCH/DELETE /draft are draft-scratch
    # writes relayed verbatim to the bridge -- they never arm or launch a run.
    # No other write verb may exist anywhere in the composed app.
    #
    # The /queue writes are the sidecar's relay of the bridge's queue
    # surface (see test_queue_relay.py), and are the ONLY way anything this
    # sidecar serves can put work in front of hardware. They add no policy
    # here: the launch token is resolved in-process and attached to every one
    # of them, and which of them actually ARM anything -- /queue/start always,
    # /queue/stop only on cancel:true, /queue/abort never -- is the bridge's
    # decision to make and enforce, not a copy kept in the sidecar.
    #
    # /queue/abort is a write in the HTTP sense only: it is the emergency halt
    # for a plan already moving hardware, the bridge gates it on nothing, and
    # it is listed here for the same reason as the rest -- so a new write route
    # cannot appear without someone justifying it in this list.
    #
    # Enumerated deliberately: a new write route must fail this test until
    # someone justifies it in this list.
    write_paths = {
        (path, method)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method not in ("get", "head", "options")
    }
    assert write_paths == {
        ("/draft", "patch"),
        ("/draft", "delete"),
        ("/queue/items", "post"),
        ("/queue/items/{uid}/move", "post"),
        ("/queue/items/{uid}", "delete"),
        ("/queue/start", "post"),
        ("/queue/stop", "post"),
        ("/queue/abort", "post"),
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
    # /results is the deprecated alias of /bluesky (same bundle, one release);
    # both must be reachable on the composed app, not just the new name.
    for mount_path in ("/plan", "/bluesky", "/results"):
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
