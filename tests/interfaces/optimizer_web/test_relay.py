"""Unit tests for the optimizer-web sidecar's write relay.

The router is mounted on a local app with the optimizer faked by
``httpx.MockTransport``. Asserted on what the service actually receives:

- **The token rides start and resume only**, read from this process's
  environment, and a browser-supplied one is never forwarded.
- **The owner comes from the auth gate**, rides every relayed write, and a
  browser-supplied ``X-Osprey-Owner`` is dropped.
- **Start binds to the browser's revision**: forwarded verbatim, and a start
  without one is refused 400 before the service is called.
- **The session route reports only the gate's owner**, behind the gate, and
  never a browser-supplied one.
- **Bodies and outcomes pass through verbatim**; an unreachable service is a
  502 and one that does not answer in time a 504.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, WebAuthMiddleware
from osprey.interfaces.optimizer_web import relay
from osprey.interfaces.web_auth import BIND_HOST_ENV, TERMINAL_USER_ENV, WebCredentials
from osprey.utils.owner_header import OWNER_HEADER
from tests.interfaces.optimizer_web._relay import (
    OPTIMIZER_URL,
    build_router_app,
    raising_handler,
    recording_app,
)

TOKEN = "s3cr3t-optimizer-launch-token"
REVISION = "f" * 64

#: (method, sidecar path, service path, body, armed) for every relayed write.
_WRITES = [
    ("POST", "/relay/optimizations", "/optimizations", {"environment_name": "booster"}, False),
    ("PUT", "/relay/jobs/job-1/config", "/jobs/job-1/config", {"n_iterations": 5}, False),
    (
        "POST",
        f"/relay/jobs/job-1/start?config_revision={REVISION}",
        "/jobs/job-1/start",
        None,
        True,
    ),
    ("POST", "/relay/jobs/job-1/resume", "/jobs/job-1/resume", None, True),
    ("POST", "/relay/jobs/job-1/pause", "/jobs/job-1/pause", None, False),
    ("POST", "/relay/jobs/job-1/cancel", "/jobs/job-1/cancel", None, False),
]
_WRITE_IDS = [w[1].split("?")[0] for w in _WRITES]


@pytest.fixture(autouse=True)
def _launch_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIMIZER_LAUNCH_TOKEN", TOKEN)


def _send(client: TestClient, method: str, path: str, body: dict | None, **headers: str):
    return client.request(method, path, json=body, headers=headers or None)


@pytest.mark.parametrize(("method", "path", "upstream", "body", "_armed"), _WRITES, ids=_WRITE_IDS)
def test_each_write_reaches_its_service_route_with_its_body_verbatim(
    method: str, path: str, upstream: str, body: dict | None, _armed: bool
) -> None:
    answer = {"job_id": "job-1", "status": "created"}
    app, seen, contents = recording_app(relay.router, 200, answer)

    with TestClient(app) as client:
        response = _send(client, method, path, body)

    assert response.status_code == 200
    assert response.json() == answer
    assert len(seen) == 1
    assert seen[0].method == method
    assert str(seen[0].url).split("?")[0] == f"{OPTIMIZER_URL}{upstream}"
    if body is None:
        assert contents[0] == b""
    else:
        assert json.loads(contents[0]) == body


@pytest.mark.parametrize(("method", "path", "_upstream", "body", "armed"), _WRITES, ids=_WRITE_IDS)
def test_the_token_rides_start_and_resume_only(
    method: str, path: str, _upstream: str, body: dict | None, armed: bool
) -> None:
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        _send(client, method, path, body)

    if armed:
        assert seen[0].headers.get_list("X-Launch-Token") == [TOKEN]
    else:
        assert "x-launch-token" not in seen[0].headers


@pytest.mark.parametrize(("method", "path", "_upstream", "body", "_armed"), _WRITES, ids=_WRITE_IDS)
def test_a_browser_token_is_never_forwarded(
    method: str, path: str, _upstream: str, body: dict | None, _armed: bool, monkeypatch
) -> None:
    monkeypatch.delenv("OPTIMIZER_LAUNCH_TOKEN")
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        _send(client, method, path, body, **{"X-Launch-Token": "browser-token"})

    assert "x-launch-token" not in seen[0].headers


def test_the_browser_token_is_replaced_not_merged_on_an_armed_route() -> None:
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        client.post("/relay/jobs/job-1/resume", headers={"X-Launch-Token": "browser-token"})

    assert seen[0].headers.get_list("X-Launch-Token") == [TOKEN]


def test_the_token_is_never_echoed_back() -> None:
    app, _, _ = recording_app(relay.router, 403, {"detail": "invalid or missing launch token"})

    with TestClient(app) as client:
        response = client.post(f"/relay/jobs/job-1/start?config_revision={REVISION}")

    assert response.status_code == 403
    assert TOKEN not in response.text
    assert all(TOKEN not in value for value in response.headers.values())


# ---------------------------------------------------------------------------
# Start binds to the revision the browser names
# ---------------------------------------------------------------------------


def test_start_forwards_the_browsers_revision_verbatim() -> None:
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        client.post(f"/relay/jobs/job-1/start?config_revision={REVISION}")

    assert seen[0].url.params.get_list("config_revision") == [REVISION]


def test_start_forwards_only_the_revision() -> None:
    """A flag the panel never showed (``override_unhealthy``) cannot ride a start."""
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        client.post(f"/relay/jobs/job-1/start?config_revision={REVISION}&override_unhealthy=true")

    assert seen[0].url.params.multi_items() == [("config_revision", REVISION)]


@pytest.mark.parametrize("query", ["", "?config_revision=", "?override_unhealthy=true"])
def test_start_without_a_revision_is_400_and_never_reaches_the_service(query: str) -> None:
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        response = client.post(f"/relay/jobs/job-1/start{query}")

    assert response.status_code == 400
    assert "config_revision" in response.json()["detail"]
    assert seen == []


def test_a_stale_revision_is_the_services_409_relayed_intact() -> None:
    body = {"detail": "Job job-1's draft changed since it was shown; re-read it and start again."}
    app, _, _ = recording_app(relay.router, 409, body)

    with TestClient(app) as client:
        response = client.post(f"/relay/jobs/job-1/start?config_revision={REVISION}")

    assert response.status_code == 409
    assert response.json() == body


def test_an_unset_token_omits_the_header_and_lets_the_service_answer(monkeypatch) -> None:
    monkeypatch.delenv("OPTIMIZER_LAUNCH_TOKEN")
    body = {"detail": "launch disabled: OPTIMIZER_LAUNCH_TOKEN is not configured"}
    app, seen, _ = recording_app(relay.router, 503, body)

    with TestClient(app) as client:
        response = client.post(f"/relay/jobs/job-1/start?config_revision={REVISION}")

    assert "x-launch-token" not in seen[0].headers
    assert response.status_code == 503
    assert response.json() == body


@pytest.mark.parametrize(
    "path",
    [
        "/relay/results/booster/2026-11-09/restore",
        "/relay/results/booster/2026-11-09/apply-best",
        "/relay/environments/booster/variables",
        "/relay/environments/booster/evaluate",
        "/relay/models",
        "/relay/jobs/current",
        "/relay/environments/booster/resolve-bounds",
    ],
)
@pytest.mark.parametrize("method", ["POST", "PUT"])
def test_writes_outside_this_relay_are_not_relayed(path: str, method: str) -> None:
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        response = client.request(method, path, json={})

    assert response.status_code in (404, 405)
    assert seen == []


def test_an_unreachable_service_is_a_502_with_a_fixed_body() -> None:
    app = build_router_app(relay.router, raising_handler(httpx.ConnectError))

    with TestClient(app) as client:
        response = client.post("/relay/jobs/job-1/cancel")

    assert response.status_code == 502
    assert response.json() == {"detail": "optimizer service unreachable"}


def test_a_start_the_service_did_not_answer_in_time_is_a_504() -> None:
    """The start may have gone through: the operator is told to check, not retry."""
    app = build_router_app(relay.router, raising_handler(httpx.ReadTimeout))

    with TestClient(app) as client:
        response = client.post(f"/relay/jobs/job-1/start?config_revision={REVISION}")

    assert response.status_code == 504
    assert "check the job before retrying" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Owner: minted from the credential the gate matched, never from the wire
# ---------------------------------------------------------------------------

_OWN_SECRET = "deployment-wide-secret"
_OP1_SECRET = "op1-roster-secret"
_UNNAMED_SECRET = "roster-secret-nobody-owns"


def _gated_recording_app() -> tuple[FastAPI, list, list]:
    """``recording_app`` behind the real auth gate, with a known roster."""
    app, seen, contents = recording_app(relay.router)
    app.add_middleware(WebAuthMiddleware)
    app.state.web_credentials = WebCredentials(
        operator_secret=_OWN_SECRET,
        panel_token="panel-token",
        roster_secrets=(_OP1_SECRET, _UNNAMED_SECRET),
        roster_owners=("op1", ""),
    )
    return app, seen, contents


@pytest.mark.no_auth_seam
@pytest.mark.parametrize(("method", "path", "_upstream", "body", "_armed"), _WRITES, ids=_WRITE_IDS)
def test_every_write_carries_the_gates_owner_and_drops_the_browsers(
    method: str, path: str, _upstream: str, body: dict | None, _armed: bool
) -> None:
    app, seen, _ = _gated_recording_app()

    with TestClient(app) as client:
        response = _send(
            client, method, path, body, **{OPERATOR_SECRET_HEADER: _OP1_SECRET, OWNER_HEADER: "bob"}
        )

    assert response.status_code == 200
    assert seen[0].headers.get_list(OWNER_HEADER) == ["op1"]


@pytest.mark.no_auth_seam
def test_no_owner_header_when_the_credential_named_nobody() -> None:
    app, seen, _ = _gated_recording_app()

    with TestClient(app) as client:
        response = client.post(
            f"/relay/jobs/job-1/start?config_revision={REVISION}",
            headers={OPERATOR_SECRET_HEADER: _UNNAMED_SECRET, OWNER_HEADER: "bob"},
        )

    assert response.status_code == 200
    assert OWNER_HEADER not in seen[0].headers


def test_no_owner_header_when_nothing_stamped_the_connection() -> None:
    app, seen, _ = recording_app(relay.router)

    with TestClient(app) as client:
        client.post("/relay/jobs/job-1/pause", headers={OWNER_HEADER: "bob"})

    assert OWNER_HEADER not in seen[0].headers


# ---------------------------------------------------------------------------
# Session: the owner the panel's writes would carry, before Start
# ---------------------------------------------------------------------------


@pytest.mark.no_auth_seam
@pytest.mark.parametrize(
    ("secret", "owner"),
    [(_OP1_SECRET, "op1"), (_UNNAMED_SECRET, None), (_OWN_SECRET, None)],
    ids=["roster-login", "unnamed-roster-secret", "sidecar-deployment-secret"],
)
def test_the_session_reports_the_gates_owner_and_ignores_a_forged_header(
    secret: str, owner: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sidecar's compose shape: a bind host and no terminal user, so the
    # deployment-wide secret authorises without naming anyone.
    monkeypatch.setenv(BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    app, seen, _ = _gated_recording_app()

    with TestClient(app) as client:
        response = client.get(
            "/relay/session", headers={OPERATOR_SECRET_HEADER: secret, OWNER_HEADER: "bob"}
        )

    assert response.status_code == 200
    assert response.json() == {"owner": owner}
    assert seen == []


@pytest.mark.no_auth_seam
def test_the_session_is_behind_the_gate() -> None:
    app, seen, _ = _gated_recording_app()

    with TestClient(app) as client:
        response = client.get("/relay/session", headers={OWNER_HEADER: "bob"})

    assert response.status_code == 401
    assert "bob" not in response.text
    assert seen == []


def test_the_session_names_nobody_when_nothing_stamped_the_connection() -> None:
    app, _, _ = recording_app(relay.router)

    with TestClient(app) as client:
        response = client.get("/relay/session", headers={OWNER_HEADER: "bob"})

    assert response.json() == {"owner": None}


# ---------------------------------------------------------------------------
# Single-user alias: the environment the build hands a roster-less sidecar
# ---------------------------------------------------------------------------


def _alias_shaped_app(monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, list]:
    """``recording_app`` behind the gate, credentials populated as the sidecar's are.

    The environment ``services/_panel_grant.j2`` renders for a roster-less
    deployment: the deployment-wide secret as the sidecar's own, the same value
    again under the operator's roster variable, the accept flag, and the owners
    map naming that variable's account.
    """
    from osprey.interfaces.web_auth import (
        OPERATOR_SECRET_ENV,
        ROSTER_ACCEPT_ENV,
        ROSTER_OWNERS_ENV,
        ROSTER_SECRET_ENV_PREFIX,
        get_web_credentials,
    )

    monkeypatch.setenv(BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv(OPERATOR_SECRET_ENV, _OWN_SECRET)
    monkeypatch.setenv(f"{ROSTER_SECRET_ENV_PREFIX}OPERATOR", _OWN_SECRET)
    monkeypatch.setenv(ROSTER_ACCEPT_ENV, "1")
    monkeypatch.setenv(ROSTER_OWNERS_ENV, "OPERATOR=operator")
    app, seen, _ = recording_app(relay.router)
    app.add_middleware(WebAuthMiddleware)
    app.state.web_credentials = get_web_credentials()
    return app, seen


@pytest.mark.no_auth_seam
def test_the_alias_names_the_operator_behind_the_deployment_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-user Start carries the operator the host terminal signs in as."""
    app, seen = _alias_shaped_app(monkeypatch)

    with TestClient(app) as client:
        session = client.get(
            "/relay/session", headers={OPERATOR_SECRET_HEADER: _OWN_SECRET, OWNER_HEADER: "bob"}
        )
        start = client.post(
            f"/relay/jobs/job-1/start?config_revision={REVISION}",
            headers={OPERATOR_SECRET_HEADER: _OWN_SECRET, OWNER_HEADER: "bob"},
        )

    assert session.json() == {"owner": "operator"}
    assert start.status_code == 200
    assert seen[0].headers.get_list(OWNER_HEADER) == ["operator"]


@pytest.mark.no_auth_seam
def test_the_alias_admits_no_other_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    app, seen = _alias_shaped_app(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/relay/session", headers={OPERATOR_SECRET_HEADER: "guess"})

    assert response.status_code == 401
    assert seen == []


# ---------------------------------------------------------------------------
# The agent never gets the token
# ---------------------------------------------------------------------------


def test_an_agent_child_environment_never_carries_the_token(monkeypatch) -> None:
    """Every agent child path starts from ``build_base_child_env``, whose
    sensitive-name strip drops ``*_LAUNCH_TOKEN``: the agent can draft a run
    through the MCP server but cannot arm a start."""
    from osprey.agent_runner.clean_env import build_base_child_env

    monkeypatch.setenv("OPTIMIZER_LAUNCH_TOKEN", TOKEN)

    assert "OPTIMIZER_LAUNCH_TOKEN" not in build_base_child_env()
