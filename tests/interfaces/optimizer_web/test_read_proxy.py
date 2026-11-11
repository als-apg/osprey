"""Unit tests for the optimizer-web sidecar's read proxy.

The router is mounted on a local app with the optimizer faked by
``httpx.MockTransport``. Three contracts: the allowlist (every listed GET is
relayed to the matching service path, nothing else and no other verb is), the
verbatim relay (body, status and query string), and the ``X-Config-Revision``
header reaching the browser.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.optimizer_web import read_proxy
from tests.interfaces.optimizer_web._relay import (
    OPTIMIZER_URL,
    build_router_app,
    raising_handler,
    recording_app,
)

#: (sidecar path, service path) for every relayed read.
_ALLOWLIST = [
    ("/service/health", "/health"),
    ("/environments", "/environments"),
    ("/environments/booster", "/environments/booster"),
    ("/environments/booster/status", "/environments/booster/status"),
    ("/environments/booster/variables", "/environments/booster/variables"),
    ("/environments/booster/presets", "/environments/booster/presets"),
    ("/environments/booster/initial-bounds", "/environments/booster/initial-bounds"),
    ("/jobs/current", "/jobs/current"),
    ("/jobs/job-1", "/jobs/job-1"),
    ("/jobs/job-1/config", "/jobs/job-1/config"),
    ("/models", "/models"),
]


@pytest.mark.parametrize(("path", "upstream"), _ALLOWLIST)
def test_every_listed_read_is_relayed_to_its_service_path(path: str, upstream: str) -> None:
    body = {"answer": path}
    app, seen, _ = recording_app(read_proxy.router, 200, body)

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json() == body
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert str(seen[0].url).split("?")[0] == f"{OPTIMIZER_URL}{upstream}"


@pytest.mark.parametrize(("path", "_upstream"), _ALLOWLIST)
@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_no_write_verb_reaches_the_service_through_a_read_path(
    path: str, _upstream: str, method: str
) -> None:
    app, seen, _ = recording_app(read_proxy.router)

    with TestClient(app) as client:
        response = client.request(method, path)

    assert response.status_code == 405
    assert seen == []


@pytest.mark.parametrize(
    "path",
    [
        "/jobs",
        "/results",
        "/environments/booster/observables",
        "/results/booster/2026-11-09/restore-preview",
        "/results/booster/2026-11-09",
        "/environments/booster/pv",
        "/jobs/job-1/start",
        "/optimizations",
    ],
)
def test_an_unlisted_read_is_not_relayed(path: str) -> None:
    app, seen, _ = recording_app(read_proxy.router)

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code in (404, 405)
    assert seen == []


def test_the_query_string_is_forwarded_as_it_arrived() -> None:
    app, seen, _ = recording_app(read_proxy.router)

    with TestClient(app) as client:
        client.get("/environments/booster/variables?target=va&names=a&names=b")
        client.get("/jobs/job-1?detail=full")

    assert seen[0].url.params.multi_items() == [("target", "va"), ("names", "a"), ("names", "b")]
    assert seen[1].url.params.multi_items() == [("detail", "full")]


def test_a_path_segment_is_escaped_before_it_reaches_the_service() -> None:
    app, seen, _ = recording_app(read_proxy.router)

    with TestClient(app) as client:
        client.get("/environments/a%20b%2Bc/status")

    assert seen[0].url.raw_path == b"/environments/a%20b%2Bc/status"


@pytest.mark.parametrize("status_code", [404, 409, 422, 503])
def test_a_service_refusal_is_relayed_with_body_and_status(status_code: int) -> None:
    body = {"detail": f"refused with {status_code}"}
    app, _, _ = recording_app(read_proxy.router, status_code, body)

    with TestClient(app) as client:
        response = client.get("/jobs/job-1")

    assert response.status_code == status_code
    assert response.json() == body


def test_the_config_revision_header_reaches_the_browser() -> None:
    revision = "a" * 64
    app, _, _ = recording_app(
        read_proxy.router,
        200,
        {"environment_name": "booster"},
        headers={"X-Config-Revision": revision, "X-Other": "dropped"},
    )

    with TestClient(app) as client:
        response = client.get("/jobs/job-1/config")

    assert response.headers["x-config-revision"] == revision
    assert "x-other" not in response.headers
    assert response.json() == {"environment_name": "booster"}


def test_the_config_revision_header_survives_the_panel_proxy_filter() -> None:
    """The browser reaches this sidecar through the web-terminal panel proxy,
    whose response filter is a denylist; the revision must not be in it."""
    from osprey.interfaces.web_terminal.routes.proxy import _filter_response_headers

    kept = _filter_response_headers(httpx.Headers({"X-Config-Revision": "rev"}))

    assert kept == {"x-config-revision": "rev"}


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_an_unreachable_service_is_a_502_with_a_fixed_body(error: type) -> None:
    app = build_router_app(read_proxy.router, raising_handler(error))

    with TestClient(app) as client:
        response = client.get("/jobs/current")

    assert response.status_code == 502
    assert response.json() == {"detail": "optimizer service unreachable"}


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout])
def test_a_service_that_does_not_answer_in_time_is_a_504_not_unreachable(error: type) -> None:
    app = build_router_app(read_proxy.router, raising_handler(error))

    with TestClient(app) as client:
        response = client.get("/environments/booster/status")

    assert response.status_code == 504
    assert "may still be working" in response.json()["detail"]
    assert "unreachable" not in response.json()["detail"]


def test_a_read_carries_no_credential_header() -> None:
    app, seen, _ = recording_app(read_proxy.router)

    with TestClient(app) as client:
        client.get(
            "/jobs/job-1/config",
            headers={"X-Launch-Token": "browser-token", "X-Osprey-Owner": "bob"},
        )

    assert "x-launch-token" not in seen[0].headers
    assert "x-osprey-owner" not in seen[0].headers
