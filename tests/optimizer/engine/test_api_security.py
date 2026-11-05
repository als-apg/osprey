"""The deployment's gates: the launch token on the routes that move hardware,
write what a start will run, or accept code, and the targets a deployment's
optimizer may drive.

Both are deployment credentials rather than route behaviour, so the rest of the
suite runs past a stand-in for the token (``conftest._launch_token_armed``);
these tests meet the real gate, through the real app.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from osprey_optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.api.main import app
from osprey_optimizer.api.security import (
    LAUNCH_TOKEN_ENV,
    LAUNCH_TOKEN_HEADER,
    TARGETS_ENV,
    allowed_targets,
)

TOKEN = "s3cret-launch-token"

#: Every gated route, as (method, path, request kwargs). Unknown names and jobs
#: on purpose: a request the gate admits fails further in, never by moving
#: anything.
GATED = [
    pytest.param(
        "put",
        "/environments/nope/variables?target=va",
        {"json": {"variables": {}}},
        id="set_variables",
    ),
    pytest.param(
        "post",
        "/environments/nope/evaluate?target=va",
        {"json": {"variables": {}, "observables": []}},
        id="evaluate",
    ),
    pytest.param("post", "/optimizations", {"json": {}}, id="create_draft"),
    pytest.param("put", "/jobs/nope/config", {"json": {}}, id="update_draft"),
    pytest.param("put", "/jobs/current", {"json": {"job_id": "nope"}}, id="current_job"),
    pytest.param("post", "/jobs/nope/start", {}, id="start"),
    pytest.param("post", "/jobs/nope/resume", {}, id="resume"),
    pytest.param("post", "/results/env/ts/restore", {}, id="restore"),
    pytest.param("post", "/models", {"files": {"file": ("m.joblib", b"x")}}, id="upload_model"),
]


@pytest.fixture
def gate_client(mock_registry: MagicMock) -> Generator[TestClient, None, None]:
    tracker = MagicMock(spec=JobTracker)
    tracker.get_active_job.return_value = None
    tracker.get_job.return_value = None
    app.dependency_overrides[get_environment_registry] = lambda: mock_registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_environment_registry, None)
        app.dependency_overrides.pop(get_job_tracker, None)


def _call(client: TestClient, method: str, path: str, kwargs: dict, **headers: str):
    return getattr(client, method)(path, headers=headers or None, **kwargs)


@pytest.mark.launch_token_gate
@pytest.mark.parametrize(("method", "path", "kwargs"), GATED)
def test_with_no_token_configured_the_gated_routes_refuse(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str, kwargs: dict
) -> None:
    monkeypatch.delenv(LAUNCH_TOKEN_ENV, raising=False)
    resp = _call(gate_client, method, path, kwargs, **{LAUNCH_TOKEN_HEADER: TOKEN})
    assert resp.status_code == 503
    assert LAUNCH_TOKEN_ENV in resp.json()["detail"]


@pytest.mark.launch_token_gate
@pytest.mark.parametrize("sent", [None, "wrong"], ids=["missing", "wrong"])
@pytest.mark.parametrize(("method", "path", "kwargs"), GATED)
def test_a_missing_or_wrong_token_is_forbidden(
    gate_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    kwargs: dict,
    sent: str | None,
) -> None:
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    headers = {LAUNCH_TOKEN_HEADER: sent} if sent else {}
    resp = _call(gate_client, method, path, kwargs, **headers)
    assert resp.status_code == 403
    assert TOKEN not in resp.text


@pytest.mark.launch_token_gate
@pytest.mark.parametrize(("method", "path", "kwargs"), GATED)
def test_the_right_token_is_admitted(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str, kwargs: dict
) -> None:
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    resp = _call(gate_client, method, path, kwargs, **{LAUNCH_TOKEN_HEADER: TOKEN})
    assert resp.status_code not in (403, 503), resp.text


@pytest.mark.launch_token_gate
@pytest.mark.parametrize("action", ["cancel", "pause"])
def test_stopping_a_run_needs_no_token(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    monkeypatch.delenv(LAUNCH_TOKEN_ENV, raising=False)
    resp = gate_client.post(f"/jobs/nope/{action}")
    assert resp.status_code not in (403, 503), resp.text


@pytest.mark.launch_token_gate
def test_a_non_ascii_token_is_forbidden_not_an_error(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headers decode as latin-1; the compare must refuse such a token, not raise."""
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    resp = gate_client.post(
        "/jobs/nope/start", headers={LAUNCH_TOKEN_HEADER: "\u00e9".encode("latin-1")}
    )
    assert resp.status_code == 403


@pytest.mark.launch_token_gate
def test_an_upload_without_the_token_is_refused_before_its_body_is_read(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The body cap's upload exemption is for token holders only.

    The declared length is far past any cap, and no such body is sent: a
    refusal that read it first would hang or fail differently.
    """
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    resp = gate_client.post("/models", content=b"", headers={"content-length": str(10 * 1024**3)})
    assert resp.status_code == 403


@pytest.mark.launch_token_gate
def test_an_upload_past_the_model_cap_is_refused_by_its_declared_length(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    resp = gate_client.post(
        "/models",
        content=b"",
        headers={LAUNCH_TOKEN_HEADER: TOKEN, "content-length": str(10 * 1024**3)},
    )
    assert resp.status_code == 413


# ── allowed targets ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, None, id="unset_is_unrestricted"),
        pytest.param("va", frozenset({"va"}), id="one"),
        pytest.param(" va , standin ", frozenset({"va", "standin"}), id="spaced"),
        pytest.param("", frozenset(), id="empty_allows_none"),
    ],
)
def test_allowed_targets_parse(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: frozenset[str] | None
) -> None:
    if raw is None:
        monkeypatch.delenv(TARGETS_ENV, raising=False)
    else:
        monkeypatch.setenv(TARGETS_ENV, raw)
    assert allowed_targets() == expected


def test_a_machine_route_on_a_target_not_allowed_is_forbidden(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TARGETS_ENV, "va")
    resp = gate_client.get("/environments/sim/variables", params={"target": "standin"})
    assert resp.status_code == 403
    assert "'standin'" in resp.json()["detail"]


def test_a_machine_route_on_an_allowed_target_passes_the_check(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TARGETS_ENV, "va,standin")
    resp = gate_client.get("/environments/sim/variables", params={"target": "standin"})
    assert resp.status_code != 403, resp.text


def test_a_draft_on_a_target_not_allowed_is_forbidden(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TARGETS_ENV, "va")
    resp = gate_client.post(
        "/optimizations",
        json={"environment_name": "sim", "control_target": "standin", "objective_name": "obj"},
    )
    assert resp.status_code == 403
