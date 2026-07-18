"""Edge-case tests for the bluesky panels sidecar's launch route.

``test_launch.py`` covers the happy path, unarmed states, and the
launch-side error passthrough. This module closes three gaps a review
flagged that ``test_launch.py`` doesn't exercise:

(a) The bridge's CREATE call (``POST {bridge}/runs``) itself failing (500) --
    distinct from the LAUNCH call failing, which ``test_launch.py`` already
    covers.
(b) The bridge's CREATE call succeeding (200) but returning a body with no
    ``id`` -- the "no run id" guard, and that the launch is never attempted
    in that case.
(c) A request body that supplies a ``launch_token`` field is ignored
    entirely on the (locally) unarmed path: no bridge call is made, and the
    inert response is unaffected by the attacker-supplied field. This locks
    in the "token is never read from the request" invariant against future
    refactors of ``LaunchRequest``.

Mirrors ``test_launch.py``'s harness: a local ``FastAPI()`` with only
``launch.router`` mounted, backed by ``httpx.MockTransport`` (respx is not
installed in this environment), with config/env isolated via an autouse
fixture so ambient repo/user config can never leak a real launch token.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.bluesky_panels import launch

TOKEN = "s3cr3t-launch-token"  # noqa: S105 - test fixture value, not a real secret
RUN_ID = "run-abc123"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)


def _make_app(client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI()
    app.include_router(launch.router)
    app.state.bridge_url = "http://bridge.test"
    app.state.client = client
    return app


def _post_launch(app: FastAPI, **json_body: object) -> httpx.Response:
    payload = {"plan_name": "orm", "plan_args": {}}
    payload.update(json_body)
    with TestClient(app) as client:
        return client.post("/runs/launch", json=payload)


# ---------------------------------------------------------------------------
# (a) CREATE-side error: bridge POST /runs itself returns 500
# ---------------------------------------------------------------------------


def test_launch_create_error_mirrors_bridge_status_and_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(500, json={"detail": "scanner subsystem unavailable"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_launch(app)

    # The response must mirror the bridge's status + detail, not silently
    # succeed or invent a generic error.
    assert response.status_code == 500
    assert response.json()["detail"] == "scanner subsystem unavailable"

    # Only the create call happened -- the launch must never be attempted
    # after a failed create.
    assert len(calls) == 1
    assert calls[0].url.path == "/runs"


# ---------------------------------------------------------------------------
# (b) NO-RUN-ID guard: bridge POST /runs returns 200 with a body lacking "id"
# ---------------------------------------------------------------------------


def test_launch_no_run_id_in_create_body_returns_502_and_never_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            # 200 status, but the body has no "id" key at all.
            return httpx.Response(200, json={"status": "pending"})
        # If the route ever calls launch here, that is exactly the bug this
        # test exists to catch -- fail loudly rather than returning a
        # plausible-looking response.
        raise AssertionError(
            f"launch must never be called when create returned no run id "
            f"(got {request.method} {request.url})"
        )

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_launch(app)

    assert response.status_code == 502
    assert "no run id" in response.json()["detail"].lower()

    # Exactly one bridge call (create) -- the launch was never attempted.
    assert len(calls) == 1
    assert calls[0].url.path == "/runs"


@pytest.mark.parametrize(
    "create_body",
    [
        {"status": "pending"},  # no "id" key at all
        {"id": None, "status": "pending"},  # "id" present but null
        {"id": "", "status": "pending"},  # "id" present but empty/falsy
    ],
)
def test_launch_falsy_run_id_variants_all_guard(
    monkeypatch: pytest.MonkeyPatch, create_body: dict
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json=create_body)
        raise AssertionError(f"launch must never be called: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_launch(app)

    assert response.status_code == 502
    assert "no run id" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# (c) REQUEST-BODY-TOKEN-IGNORED: an attacker-supplied launch_token field is
# never read, on the (locally) unarmed path.
# ---------------------------------------------------------------------------


def test_launch_request_body_launch_token_is_ignored_when_unarmed() -> None:
    # No BLUESKY_LAUNCH_TOKEN env var and an isolated (nonexistent) config
    # file -- this deployment is unarmed. Even though the request body
    # supplies a launch_token, LaunchRequest has no such field, so it must
    # never reach _resolve_launch_token or the bridge.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"bridge must not be called -- request-body token must never arm "
            f"the route: {request.method} {request.url}"
        )

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_launch(app, launch_token="attacker")  # type: ignore[arg-type]

    assert response.status_code == 200
    assert response.json() == {
        "status": "writes_not_armed",
        "detail": "writes are not armed on this deployment",
    }
    # The attacker-supplied token must not leak back either.
    assert "attacker" not in response.text


def test_launch_request_body_launch_token_is_ignored_when_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even when the deployment IS armed (a real token resolved in-process),
    # a request-body launch_token must be dropped by validation and never
    # substituted for the real, in-process-resolved token.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json={"id": RUN_ID, "status": "pending"})
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/launch":
            # The header must carry the real in-process token, never the
            # attacker-supplied body field.
            assert request.headers.get("x-launch-token") == TOKEN
            return httpx.Response(200, json={"id": RUN_ID, "status": "running"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    response = _post_launch(app, launch_token="attacker")  # type: ignore[arg-type]

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert len(calls) == 2
    assert calls[1].headers["x-launch-token"] == TOKEN


# ---------------------------------------------------------------------------
# (d) draft-revision XOR shape violations (the launch revision gate)
#
# ``LaunchRequest`` is ``draft_revision`` XOR (``plan_name`` + optional
# ``plan_args``): exactly one launch mode. Both modes, or neither, must 422
# before the bridge is ever contacted -- there is no "ignored but present"
# body field that could act as a second, silently-dropped source of launched
# args.
# ---------------------------------------------------------------------------


def _refusing_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(
        f"bridge must not be called for a 422 XOR violation: {request.method} {request.url}"
    )


def test_launch_xor_both_draft_revision_and_plan_name_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(_refusing_handler)))

    with TestClient(app) as client:
        response = client.post(
            "/runs/launch",
            json={"draft_revision": 5, "plan_name": "orm", "plan_args": {}},
        )

    assert response.status_code == 422
    assert "exactly one" in str(response.json()).lower()


def test_launch_xor_neither_draft_revision_nor_plan_name_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(_refusing_handler)))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={})

    assert response.status_code == 422
    assert "exactly one" in str(response.json()).lower()


def test_launch_xor_plan_args_alone_does_not_establish_manual_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # plan_args alone (no plan_name, no draft_revision) does not complete
    # either arm -- plan_name is what completes the manual arm.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(_refusing_handler)))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"plan_args": {"foo": 1}})

    assert response.status_code == 422


def test_launch_xor_draft_revision_with_stray_plan_args_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stray plan_args alongside draft_revision (with no plan_name) is still
    # a second source of launch args and must be rejected outright, not
    # silently dropped -- this is what guarantees request-body args can never
    # leak into a draft-mode launch.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(_refusing_handler)))

    with TestClient(app) as client:
        response = client.post(
            "/runs/launch",
            json={"draft_revision": 5, "plan_args": {"stale": "value"}},
        )

    assert response.status_code == 422


def test_launch_xor_draft_revision_with_explicit_null_plan_name_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review remediation (task 3.2 follow-up): an explicitly-present
    # ``plan_name: null`` alongside ``draft_revision`` looked identical by
    # value to plain draft mode (both are "no plan_name"), so the XOR check
    # missed it when it only value-checked plan_name instead of also
    # consulting model_fields_set the way the plan_args check does. This
    # must 422 just like the stray-plan_args case above.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(_refusing_handler)))

    with TestClient(app) as client:
        response = client.post(
            "/runs/launch",
            json={"draft_revision": 5, "plan_name": None},
        )

    assert response.status_code == 422
    assert "exactly one" in str(response.json()).lower()


def test_launch_xor_draft_revision_omitting_plan_name_entirely_still_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Companion to the explicit-null case above: a body that omits
    # plan_name entirely (never present in model_fields_set) alongside
    # draft_revision must keep validating -- only an explicitly-present
    # plan_name (null or not) should trip the XOR check. A valid draft-mode
    # request rides the bridge's single POST /draft/run primitive.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/draft/run":
            return httpx.Response(200, json={"id": RUN_ID, "status": "running"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"draft_revision": 5})

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID


def test_launch_xor_manual_mode_plan_name_only_still_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Manual mode's plan_args stays optional: plan_name alone (no plan_args,
    # no draft_revision) must still be accepted -- this is the pre-task-3.2
    # shape of a manual-mode request and must keep working unchanged.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(200, json={"id": RUN_ID, "status": "pending"})
        if request.method == "POST" and request.url.path == f"/runs/{RUN_ID}/launch":
            assert request.headers.get("x-launch-token") == TOKEN
            return httpx.Response(200, json={"id": RUN_ID, "status": "running"})
        raise AssertionError(f"unexpected bridge call: {request.method} {request.url}")

    app = _make_app(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as client:
        response = client.post("/runs/launch", json={"plan_name": "orm"})

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
