"""Unit tests for the bluesky panels sidecar's launch route.

Exercises ``osprey.services.bluesky_panels.launch.router`` mounted onto a local
FastAPI app (not the full sidecar app) against a mocked Bluesky bridge via
``httpx.MockTransport`` (respx is not installed in this environment). Covers
the write-safety contract: the launch token is resolved in-process, never
accepted from the request, never echoed back, and an unarmed deployment
(locally or bridge-side) is an inert 200, not an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.bluesky_panels import launch

TOKEN = "s3cr3t-launch-token"
RUN_ID = "run-abc123"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Point config resolution at a config.yml that does not exist by default,
    # so ambient repo/user config can never leak a real launch token into a
    # test that expects the "unarmed" path. Individual tests that need a
    # config-sourced token overwrite this env var with a real temp file.
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)


def _make_app(client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI()
    app.include_router(launch.router)
    app.state.bridge_url = "http://bridge.test"
    app.state.client = client
    return app


def _refusing_transport() -> httpx.MockTransport:
    """A transport that fails the test if the bridge is ever contacted."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"bridge must not be called: {request.method} {request.url}")

    return httpx.MockTransport(handler)


def _armed_transport(
    expected_token: str,
    *,
    launch_status: int = 200,
    launch_body: dict | None = None,
    create_status: int = 200,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    calls: list[httpx.Request] = []
    body = (
        launch_body
        if launch_body is not None
        else {"id": RUN_ID, "status": "running", "tiled_degraded": False}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            if create_status >= 400:
                return httpx.Response(create_status, json={"detail": "bad request"})
            return httpx.Response(
                create_status, json={"id": RUN_ID, "status": "pending", "tiled_degraded": False}
            )
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/launch":
            assert request.headers.get("x-launch-token") == expected_token
            return httpx.Response(launch_status, json=body)
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    return httpx.MockTransport(handler), calls


def _post_launch(app: FastAPI, **json_body: object) -> httpx.Response:
    payload = {"plan_name": "orm", "plan_args": {}}
    payload.update(json_body)
    with TestClient(app) as client:
        return client.post("/runs/launch", json=payload)


# ---------------------------------------------------------------------------
# (a) armed, full happy path
# ---------------------------------------------------------------------------
def test_launch_armed_success_returns_run_id_and_never_leaks_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, calls = _armed_transport(TOKEN)
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_launch(app)

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == RUN_ID
    assert data["status"] == "running"

    # Token must never appear anywhere in the response.
    assert TOKEN not in response.text
    for header_name, header_value in response.headers.items():
        assert TOKEN not in header_value, f"token leaked in header {header_name!r}"

    # Bridge received exactly the two calls, and the launch call carried
    # the token header.
    assert len(calls) == 2
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/runs"
    assert calls[1].method == "POST"
    assert calls[1].url.path == f"/runs/{RUN_ID}/launch"
    assert calls[1].headers["x-launch-token"] == TOKEN


def test_launch_armed_via_config_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No env var token -- resolved from bluesky.launch_token in config.yml instead.
    config_path = tmp_path / "config.yml"
    config_path.write_text(f'bluesky:\n  launch_token: "{TOKEN}"\n', encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)

    transport, calls = _armed_transport(TOKEN)
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_launch(app)

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# (b) unarmed locally -- no token resolved at all
# ---------------------------------------------------------------------------
def test_launch_unarmed_locally_is_inert_and_makes_no_bridge_call() -> None:
    app = _make_app(httpx.AsyncClient(transport=_refusing_transport()))

    response = _post_launch(app)

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }


# ---------------------------------------------------------------------------
# (c) bridge itself reports unarmed (503) or token mismatch (403) on launch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bridge_status", [503, 403])
def test_launch_bridge_unarmed_or_mismatch_is_inert(
    monkeypatch: pytest.MonkeyPatch, bridge_status: int
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, calls = _armed_transport(
        TOKEN, launch_status=bridge_status, launch_body={"detail": "unarmed or mismatched"}
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_launch(app)

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }
    # Both bridge calls still happened (pending-run create + launch attempt).
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# (d) launch conflict is surfaced as-is
# ---------------------------------------------------------------------------
def test_launch_conflict_surfaces_409(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, _ = _armed_transport(
        TOKEN,
        launch_status=409,
        launch_body={"detail": "run already launched"},
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_launch(app)

    assert response.status_code == 409
    assert response.json()["detail"] == "run already launched"


# ---------------------------------------------------------------------------
# launch 404 / 500 pass through with the bridge's status + detail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bridge_status", [404, 500])
def test_launch_unknown_or_scanner_failure_passes_through(
    monkeypatch: pytest.MonkeyPatch, bridge_status: int
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, _ = _armed_transport(
        TOKEN, launch_status=bridge_status, launch_body={"detail": "bridge says no"}
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_launch(app)

    assert response.status_code == bridge_status
    assert response.json()["detail"] == "bridge says no"


# ---------------------------------------------------------------------------
# bridge unreachable entirely
# ---------------------------------------------------------------------------
def test_launch_bridge_unreachable_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_launch(app)

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


# ---------------------------------------------------------------------------
# (e) negative route-surface assertions
# ---------------------------------------------------------------------------
def test_router_exposes_exactly_one_route() -> None:
    assert len(launch.router.routes) == 1


def test_router_has_no_stop_route() -> None:
    for route in launch.router.routes:
        path = getattr(route, "path", "")
        assert not path.endswith("/stop"), f"launch router must not expose a stop route: {path}"


def test_router_has_no_bare_unbounded_runs_create_route() -> None:
    for route in launch.router.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if "POST" in methods:
            assert path != "/runs", "launch router must not expose a bare POST /runs passthrough"


def test_router_launch_route_shape() -> None:
    paths_and_methods: list[tuple[str, set]] = [
        (getattr(route, "path", ""), getattr(route, "methods", set()) or set())
        for route in launch.router.routes
    ]
    assert ("/runs/launch", {"POST"}) in paths_and_methods


# ---------------------------------------------------------------------------
# (f) draft mode: launched via the bridge's single POST /draft/run primitive
#
# The sidecar no longer reads the draft and then races a separate
# create + launch against it. Draft mode is now one bridge call, POST
# {bridge}/draft/run, and the sidecar only maps that response: the bridge owns
# pinning, minting, launching, recording, and the 409 discriminator. These
# tests exercise the sidecar's mapping of each bridge outcome.
# ---------------------------------------------------------------------------


def _draft_run_transport(
    expected_token: str,
    *,
    status: int = 200,
    body: dict | None = None,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A mock bridge transport serving only ``POST /draft/run``.

    Any other bridge path is an assertion failure: draft mode must make
    exactly the one launch call, never a snapshot read or a bare
    create + launch pair.
    """
    calls: list[httpx.Request] = []
    response_body = (
        body
        if body is not None
        else {"id": RUN_ID, "status": "running", "launched_by": "draft", "run_uid": "uid-1"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/draft/run":
            assert request.headers.get("x-launch-token") == expected_token
            return httpx.Response(status, json=response_body)
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    return httpx.MockTransport(handler), calls


def test_launch_draft_mode_launches_via_draft_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, calls = _draft_run_transport(TOKEN)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 5})

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == RUN_ID
    assert data["status"] == "running"

    # Exactly one bridge call: POST /draft/run carrying the pinned revision in
    # the body and the in-process token in the header -- never a snapshot read
    # or a bare create + launch, and never plan args from the request body.
    assert [(call.method, call.url.path) for call in calls] == [("POST", "/draft/run")]
    assert json.loads(calls[0].content) == {"draft_revision": 5}
    assert calls[0].headers["x-launch-token"] == TOKEN


def test_launch_draft_mode_unarmed_locally_never_contacts_bridge() -> None:
    # The "no bridge call at all when unarmed" invariant extends to draft
    # mode: the arming check happens strictly before the /draft/run call, so
    # an unarmed deployment never contacts the bridge.
    app = _make_app(httpx.AsyncClient(transport=_refusing_transport()))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 1})

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }


@pytest.mark.parametrize("bridge_status", [503, 403])
def test_launch_draft_mode_bridge_unarmed_or_mismatch_is_inert(
    monkeypatch: pytest.MonkeyPatch, bridge_status: int
) -> None:
    # A bridge 503 (arming disabled) or 403 (token mismatch) on /draft/run
    # maps to the same inert writes_not_armed 200 as manual mode's launch
    # path -- for the human and the agent alike, not a 500.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, calls = _draft_run_transport(
        TOKEN, status=bridge_status, body={"detail": "unarmed or mismatched"}
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 5})

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }
    assert len(calls) == 1


def test_launch_draft_mode_stale_revision_unwrapped_to_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bridge owns stale detection now and returns a FastAPI-NESTED 409
    # ({"detail": {"code", ...}}), but the panel client classifies on a
    # TOP-LEVEL ``code`` (draft-client.js). The sidecar unwraps the nested
    # detail to the panel's top-level shape, preserving the ``revision`` field
    # the panel resyncs on.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    inner = {
        "code": "stale_draft_revision",
        "detail": "draft revision 5 is stale",
        "revision": 9,
    }
    bridge_body = {"detail": inner}
    transport, calls = _draft_run_transport(TOKEN, status=409, body=bridge_body)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 5})

    assert response.status_code == 409
    # Unwrapped: the panel sees a top-level {"code", "detail", "revision"}.
    assert response.json() == inner
    assert response.json()["code"] == "stale_draft_revision"
    assert response.json()["revision"] == 9
    assert len(calls) == 1


def test_launch_draft_mode_already_launched_unwrapped_to_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other bridge 409 code -- a replayed/in-flight revision -- gets the
    # same unwrap to the panel's top-level shape; the sidecar does not branch
    # on the code (the panel client's classify handling for this new code
    # lands in a later task).
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    inner = {
        "code": "draft_revision_already_launched",
        "detail": "draft revision 5 was already launched",
        "revision": 5,
    }
    bridge_body = {"detail": inner}
    transport, calls = _draft_run_transport(TOKEN, status=409, body=bridge_body)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 5})

    assert response.status_code == 409
    assert response.json() == inner
    assert response.json()["code"] == "draft_revision_already_launched"
    assert response.json()["revision"] == 5
    assert len(calls) == 1


@pytest.mark.parametrize("bridge_status", [404, 500])
def test_launch_draft_mode_other_bridge_error_passes_through(
    monkeypatch: pytest.MonkeyPatch, bridge_status: int
) -> None:
    # Any other non-200 (a 500 post-mint launch failure, a 404) is relayed
    # with the bridge's status and detail message -- same convention as manual
    # mode's launch passthrough.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    transport, _ = _draft_run_transport(
        TOKEN, status=bridge_status, body={"detail": "bridge says no"}
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 5})

    assert response.status_code == bridge_status
    assert response.json()["detail"] == "bridge says no"


def test_launch_draft_mode_bridge_unreachable_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 1})

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}
