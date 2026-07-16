"""Unit tests for the bluesky panels sidecar's execute route (task 1.3).

Exercises ``osprey.services.bluesky_panels.execute.router`` mounted onto a local
FastAPI app (not the full sidecar app) against a mocked Bluesky bridge via
``httpx.MockTransport`` (respx is not installed in this environment). Covers
the write-safety contract: the promote token is resolved in-process, never
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

from osprey.services.bluesky_panels import execute

TOKEN = "s3cr3t-promote-token"
RUN_ID = "run-abc123"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Point config resolution at a config.yml that does not exist by default,
    # so ambient repo/user config can never leak a real promote token into a
    # test that expects the "unarmed" path. Individual tests that need a
    # config-sourced token overwrite this env var with a real temp file.
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)


def _make_app(client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI()
    app.include_router(execute.router)
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
    promote_status: int = 200,
    promote_body: dict | None = None,
    create_status: int = 200,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    calls: list[httpx.Request] = []
    body = (
        promote_body
        if promote_body is not None
        else {"id": RUN_ID, "status": "running", "tiled_degraded": False}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            if create_status >= 400:
                return httpx.Response(create_status, json={"detail": "bad request"})
            return httpx.Response(
                create_status, json={"id": RUN_ID, "status": "intent", "tiled_degraded": False}
            )
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/promote":
            assert request.headers.get("x-promote-token") == expected_token
            return httpx.Response(promote_status, json=body)
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    return httpx.MockTransport(handler), calls


def _post_execute(app: FastAPI, **json_body: object) -> httpx.Response:
    payload = {"plan_name": "orm", "plan_args": {}}
    payload.update(json_body)
    with TestClient(app) as client:
        return client.post("/runs/execute", json=payload)


# ---------------------------------------------------------------------------
# (a) armed, full happy path
# ---------------------------------------------------------------------------
def test_execute_armed_success_returns_run_id_and_never_leaks_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    transport, calls = _armed_transport(TOKEN)
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_execute(app)

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == RUN_ID
    assert data["status"] == "running"

    # Token must never appear anywhere in the response.
    assert TOKEN not in response.text
    for header_name, header_value in response.headers.items():
        assert TOKEN not in header_value, f"token leaked in header {header_name!r}"

    # Bridge received exactly the two calls, and the promote call carried
    # the token header.
    assert len(calls) == 2
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/runs"
    assert calls[1].method == "POST"
    assert calls[1].url.path == f"/runs/{RUN_ID}/promote"
    assert calls[1].headers["x-promote-token"] == TOKEN


def test_execute_armed_via_config_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No env var token -- resolved from bluesky.promote_token in config.yml instead.
    config_path = tmp_path / "config.yml"
    config_path.write_text(f'bluesky:\n  promote_token: "{TOKEN}"\n', encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)

    transport, calls = _armed_transport(TOKEN)
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_execute(app)

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# (b) unarmed locally -- no token resolved at all
# ---------------------------------------------------------------------------
def test_execute_unarmed_locally_is_inert_and_makes_no_bridge_call() -> None:
    app = _make_app(httpx.AsyncClient(transport=_refusing_transport()))

    response = _post_execute(app)

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }


# ---------------------------------------------------------------------------
# (c) bridge itself reports unarmed (503) or token mismatch (403) on promote
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bridge_status", [503, 403])
def test_execute_bridge_unarmed_or_mismatch_is_inert(
    monkeypatch: pytest.MonkeyPatch, bridge_status: int
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    transport, calls = _armed_transport(
        TOKEN, promote_status=bridge_status, promote_body={"detail": "unarmed or mismatched"}
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_execute(app)

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }
    # Both bridge calls still happened (intent create + promote attempt).
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# (d) promote conflict is surfaced as-is
# ---------------------------------------------------------------------------
def test_execute_promote_conflict_surfaces_409(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    transport, _ = _armed_transport(
        TOKEN,
        promote_status=409,
        promote_body={"detail": "run already promoted"},
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_execute(app)

    assert response.status_code == 409
    assert response.json()["detail"] == "run already promoted"


# ---------------------------------------------------------------------------
# promote 404 / 500 pass through with the bridge's status + detail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bridge_status", [404, 500])
def test_execute_promote_unknown_or_scanner_failure_passes_through(
    monkeypatch: pytest.MonkeyPatch, bridge_status: int
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    transport, _ = _armed_transport(
        TOKEN, promote_status=bridge_status, promote_body={"detail": "bridge says no"}
    )
    app = _make_app(httpx.AsyncClient(transport=transport))

    response = _post_execute(app)

    assert response.status_code == bridge_status
    assert response.json()["detail"] == "bridge says no"


# ---------------------------------------------------------------------------
# bridge unreachable entirely
# ---------------------------------------------------------------------------
def test_execute_bridge_unreachable_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_execute(app)

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


# ---------------------------------------------------------------------------
# (e) negative route-surface assertions
# ---------------------------------------------------------------------------
def test_router_exposes_exactly_one_route() -> None:
    assert len(execute.router.routes) == 1


def test_router_has_no_stop_route() -> None:
    for route in execute.router.routes:
        path = getattr(route, "path", "")
        assert not path.endswith("/stop"), f"execute router must not expose a stop route: {path}"


def test_router_has_no_bare_unbounded_runs_create_route() -> None:
    for route in execute.router.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if "POST" in methods:
            assert path != "/runs", "execute router must not expose a bare POST /runs passthrough"


def test_router_execute_route_shape() -> None:
    paths_and_methods: list[tuple[str, set]] = [
        (getattr(route, "path", ""), getattr(route, "methods", set()) or set())
        for route in execute.router.routes
    ]
    assert ("/runs/execute", {"POST"}) in paths_and_methods


# ---------------------------------------------------------------------------
# (f) draft-mode revision gate (task 3.2: execute-revision-gate)
# ---------------------------------------------------------------------------

DRAFT_PLAN_NAME = "grid_scan"
DRAFT_PLAN_ARGS = {
    "detectors": ["bpm1"],
    "axes": [{"channel": "SR01C:BEND:SP", "start": 0, "stop": 1, "num": 3}],
}


def _draft_armed_transport(
    expected_token: str,
    *,
    draft_body: dict,
    promote_status: int = 200,
    promote_body: dict | None = None,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A mock bridge transport that also serves ``GET /draft`` for draft-mode tests."""
    calls: list[httpx.Request] = []
    body = promote_body if promote_body is not None else {"id": RUN_ID, "status": "running"}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET" and request.url.path == "/draft":
            return httpx.Response(200, json=draft_body)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json={"id": RUN_ID, "status": "intent"})
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/promote":
            assert request.headers.get("x-promote-token") == expected_token
            return httpx.Response(promote_status, json=body)
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    return httpx.MockTransport(handler), calls


def test_execute_draft_mode_match_launches_snapshot_args_never_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    draft_body = {
        "draft": {
            "plan_name": DRAFT_PLAN_NAME,
            "plan_args": DRAFT_PLAN_ARGS,
            "updated_by": "mcp-agent",
            "updated_at": "2026-07-16T00:00:00+00:00",
        },
        "revision": 5,
    }
    transport, calls = _draft_armed_transport(TOKEN, draft_body=draft_body)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 5})

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == RUN_ID
    assert data["status"] == "running"

    # Exactly GET /draft, then POST /runs, then POST promote -- in that order.
    assert [(call.method, call.url.path) for call in calls] == [
        ("GET", "/draft"),
        ("POST", "/runs"),
        ("POST", f"/runs/{RUN_ID}/promote"),
    ]
    create_payload = json.loads(calls[1].content)
    assert create_payload == {"plan_name": DRAFT_PLAN_NAME, "plan_args": DRAFT_PLAN_ARGS}


def test_execute_draft_mode_stale_revision_returns_409_with_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    draft_body = {
        "draft": {
            "plan_name": DRAFT_PLAN_NAME,
            "plan_args": DRAFT_PLAN_ARGS,
            "updated_by": "mcp-agent",
            "updated_at": "2026-07-16T00:00:00+00:00",
        },
        "revision": 9,
    }
    transport, calls = _draft_armed_transport(TOKEN, draft_body=draft_body)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 5})

    assert response.status_code == 409
    assert response.json()["code"] == "stale_draft_revision"
    # Only the snapshot read happened -- create/promote never attempted
    # against a stale pin.
    assert [(call.method, call.url.path) for call in calls] == [("GET", "/draft")]


def test_execute_draft_mode_stale_after_clear_returns_409_with_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The draft was cleared since the panel's revision was pinned: the
    # bridge's GET /draft now reports {draft: null, revision: <bumped>} --
    # PROPOSAL.md's "a pinned revision from a cleared draft must 409 against
    # {draft: null, revision}".
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    draft_body = {"draft": None, "revision": 12}
    transport, calls = _draft_armed_transport(TOKEN, draft_body=draft_body)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 11})

    assert response.status_code == 409
    assert response.json()["code"] == "stale_draft_revision"
    assert [(call.method, call.url.path) for call in calls] == [("GET", "/draft")]


def test_execute_draft_mode_null_draft_at_matching_revision_still_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Defensive case: even if a null draft's revision happened to numerically
    # match the pinned draft_revision, there is no plan_name/plan_args to
    # launch -- this must still 409, never proceed with a null snapshot.
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    draft_body = {"draft": None, "revision": 0}
    transport, calls = _draft_armed_transport(TOKEN, draft_body=draft_body)
    app = _make_app(httpx.AsyncClient(transport=transport))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 0})

    assert response.status_code == 409
    assert response.json()["code"] == "stale_draft_revision"
    assert [(call.method, call.url.path) for call in calls] == [("GET", "/draft")]


def test_execute_draft_mode_bridge_unreachable_on_snapshot_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 1})

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


def test_execute_draft_mode_unarmed_locally_never_contacts_bridge() -> None:
    # The same "no bridge call at all when unarmed" invariant extends to
    # draft mode: the arming check happens strictly before the draft
    # snapshot read, so an unarmed deployment never even reads the draft.
    app = _make_app(httpx.AsyncClient(transport=_refusing_transport()))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 1})

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }


# ---------------------------------------------------------------------------
# (g) review remediation (task 3.2 follow-up): GET /draft non-200 must be
# relayed as a bridge error, never misread as a stale-revision 409.
# ---------------------------------------------------------------------------
def test_execute_draft_mode_bridge_500_on_draft_is_not_stale_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET" and request.url.path == "/draft":
            return httpx.Response(500, json={"detail": "draft store unavailable"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 5})

    # The bridge's own 500 must be relayed, not misreported as a 409
    # stale_draft_revision -- a non-200 GET /draft parses (via _safe_json)
    # to a dict with no "draft" key, which looks identical to a cleared
    # draft unless the status is checked first.
    assert response.status_code == 500
    assert response.json().get("code") != "stale_draft_revision"
    assert response.json()["detail"] == "draft store unavailable"
    # Only the snapshot read happened -- create/promote never attempted.
    assert [(call.method, call.url.path) for call in calls] == [("GET", "/draft")]


def test_execute_draft_mode_snapshot_missing_plan_name_is_bridge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-null draft snapshot that matches the pinned revision but lacks a
    # plan_name is malformed bridge data -- it must not be forwarded to
    # create-intent as {"plan_name": null, ...}, nor treated as a stale
    # revision (the revision DID match).
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    draft_body = {
        "draft": {"plan_name": None, "plan_args": {}, "updated_by": "mcp-agent"},
        "revision": 5,
    }
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET" and request.url.path == "/draft":
            return httpx.Response(200, json=draft_body)
        raise AssertionError(
            f"create/promote must never be called for an unlaunchable snapshot: "
            f"{request.method} {request.url}"
        )

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    with TestClient(app) as client:
        response = client.post("/runs/execute", json={"draft_revision": 5})

    assert response.status_code == 502
    assert response.json().get("code") != "stale_draft_revision"
    assert "plan_name" in response.json()["detail"]
    assert [(call.method, call.url.path) for call in calls] == [("GET", "/draft")]
