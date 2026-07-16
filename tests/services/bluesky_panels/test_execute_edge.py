"""Edge-case tests for the bluesky panels sidecar's execute route (task 4.1).

``test_execute.py`` covers the happy path, unarmed states, and the
promote-side error passthrough. This module closes three gaps a review
flagged that ``test_execute.py`` doesn't exercise:

(a) The bridge's CREATE call (``POST {bridge}/runs``) itself failing (500) --
    distinct from the PROMOTE call failing, which ``test_execute.py`` already
    covers.
(b) The bridge's CREATE call succeeding (200) but returning a body with no
    ``id`` -- the "no run id" guard, and that ``promote`` is never called in
    that case.
(c) A request body that supplies a ``promote_token`` field is ignored
    entirely on the (locally) unarmed path: no bridge call is made, and the
    inert response is unaffected by the attacker-supplied field. This locks
    in the "token is never read from the request" invariant against future
    refactors of ``ExecuteRequest``.

Mirrors ``test_execute.py``'s harness: a local ``FastAPI()`` with only
``execute.router`` mounted, backed by ``httpx.MockTransport`` (respx is not
installed in this environment), with config/env isolated via an autouse
fixture so ambient repo/user config can never leak a real promote token.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.bluesky_panels import execute

TOKEN = "s3cr3t-promote-token"  # noqa: S105 - test fixture value, not a real secret
RUN_ID = "run-abc123"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)


def _make_app(client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI()
    app.include_router(execute.router)
    app.state.bridge_url = "http://bridge.test"
    app.state.client = client
    return app


def _post_execute(app: FastAPI, **json_body: object) -> httpx.Response:
    payload = {"plan_name": "orm", "plan_args": {}}
    payload.update(json_body)
    with TestClient(app) as client:
        return client.post("/runs/execute", json=payload)


# ---------------------------------------------------------------------------
# (a) CREATE-side error: bridge POST /runs itself returns 500
# ---------------------------------------------------------------------------


def test_execute_create_error_mirrors_bridge_status_and_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(500, json={"detail": "scanner subsystem unavailable"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_execute(app)

    # The response must mirror the bridge's status + detail, not silently
    # succeed or invent a generic error.
    assert response.status_code == 500
    assert response.json()["detail"] == "scanner subsystem unavailable"

    # Only the create call happened -- promote must never be attempted after
    # a failed create.
    assert len(calls) == 1
    assert calls[0].url.path == "/runs"


# ---------------------------------------------------------------------------
# (b) NO-RUN-ID guard: bridge POST /runs returns 200 with a body lacking "id"
# ---------------------------------------------------------------------------


def test_execute_no_run_id_in_create_body_returns_502_and_never_promotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            # 200 status, but the body has no "id" key at all.
            return httpx.Response(200, json={"status": "intent"})
        # If the route ever calls promote here, that is exactly the bug this
        # test exists to catch -- fail loudly rather than returning a
        # plausible-looking response.
        raise AssertionError(
            f"promote must never be called when create returned no run id "
            f"(got {request.method} {request.url})"
        )

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_execute(app)

    assert response.status_code == 502
    assert "no run id" in response.json()["detail"].lower()

    # Exactly one bridge call (create) -- promote was never attempted.
    assert len(calls) == 1
    assert calls[0].url.path == "/runs"


@pytest.mark.parametrize(
    "create_body",
    [
        {"status": "intent"},  # no "id" key at all
        {"id": None, "status": "intent"},  # "id" present but null
        {"id": "", "status": "intent"},  # "id" present but empty/falsy
    ],
)
def test_execute_falsy_run_id_variants_all_guard(
    monkeypatch: pytest.MonkeyPatch, create_body: dict
) -> None:
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json=create_body)
        raise AssertionError(f"promote must never be called: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_execute(app)

    assert response.status_code == 502
    assert "no run id" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# (c) REQUEST-BODY-TOKEN-IGNORED: an attacker-supplied promote_token field is
# never read, on the (locally) unarmed path.
# ---------------------------------------------------------------------------


def test_execute_request_body_promote_token_is_ignored_when_unarmed() -> None:
    # No BLUESKY_PROMOTE_TOKEN env var and an isolated (nonexistent) config
    # file -- this deployment is unarmed. Even though the request body
    # supplies a promote_token, ExecuteRequest has no such field, so it must
    # never reach _resolve_promote_token or the bridge.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"bridge must not be called -- request-body token must never arm "
            f"the route: {request.method} {request.url}"
        )

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_execute(app, promote_token="attacker")  # type: ignore[arg-type]

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }
    # The attacker-supplied token must not leak back either.
    assert "attacker" not in response.text


def test_execute_request_body_promote_token_is_ignored_when_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even when the deployment IS armed (a real token resolved in-process),
    # a request-body promote_token must be dropped by validation and never
    # substituted for the real, in-process-resolved token.
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json={"id": RUN_ID, "status": "intent"})
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/promote":
            # The header must carry the real in-process token, never the
            # attacker-supplied body field.
            assert request.headers.get("x-promote-token") == TOKEN
            return httpx.Response(200, json={"id": RUN_ID, "status": "running"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_execute(app, promote_token="attacker")  # type: ignore[arg-type]

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert len(calls) == 2
    assert calls[1].headers["x-promote-token"] == TOKEN
