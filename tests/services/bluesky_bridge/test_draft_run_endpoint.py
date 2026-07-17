"""Coverage for `POST /draft/run` (`app.py`) — the bridge's single
launch-from-draft primitive (PROPOSAL.md, Task 2.2), composing task 2.1's
launch-state primitives (`draft.check_launchable` /
`draft.record_and_broadcast_launch`) with the run registry and launch gate.

Contract under test: token verification before any state is touched (403/503
exactly like `POST /runs/{id}/launch`); every 409/403/503 mints NOTHING; a
replayed revision never fires a second run; a post-mint failure stamps the
record ``error`` (never an eternal pre-launch state); success emits a
``launched`` SSE frame. Isolation mirrors `test_draft_launch_state.py` (draft
singleton, plan loader) plus `test_launch_validation_gate.py` (validation
records, run registry, runner factory). The `launched`-frame test follows the
white-box `draft._subscribe()` precedent — `TestClient` drains streaming
responses, so the SSE generator can't be driven in-process.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import Response

from osprey.services.bluesky_bridge import draft, plan_loader
from osprey.services.bluesky_bridge.app import app, set_runner_factory
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.runs import registry
from osprey.services.bluesky_bridge.validation_record import validation_records

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_LAUNCH_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"
_TOKEN = "s3cr3t"

_ORM_ARGS: dict[str, Any] = {
    "correctors": ["COR1"],
    "detectors": ["BPM1"],
    "span_a": 1.0,
    "num": 3,
}


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh draft singleton, plan-loader cache, validation-record store, run
    registry, and runner factory per test — all process-wide singletons in the
    real bridge (union of `test_draft_launch_state.py`'s and
    `test_launch_validation_gate.py`'s isolation fixtures).
    """
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()
    draft._clear()

    with validation_records.lock:
        saved_hashes = set(validation_records._passing_hashes)
        validation_records._passing_hashes.clear()

    registry._runs.clear()
    set_runner_factory(FakePlanRunner)

    yield

    plan_loader.reset_facility_plans()
    draft._clear()
    with validation_records.lock:
        validation_records._passing_hashes.clear()
        validation_records._passing_hashes.update(saved_hashes)
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv(_LAUNCH_TOKEN_ENV, _TOKEN)
    return TestClient(app)


@pytest.fixture
def unarmed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv(_LAUNCH_TOKEN_ENV, raising=False)
    return TestClient(app)


def _seed_orm_draft(client: TestClient, client_id: str = "panel-1") -> int:
    """Create an `orm` draft (shipped tier — always registered) and return its revision."""
    resp = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": client_id},
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["revision"])


def _launch(client: TestClient, revision: int, token: str | None = _TOKEN) -> Response:
    headers = {} if token is None else {"X-Launch-Token": token}
    return client.post("/draft/run", json={"draft_revision": revision}, headers=headers)


def _session_plan_source(name: str) -> str:
    """A minimal, bluesky-free session-tier plan file satisfying the load contract."""
    return (
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A session-tier test plan.",\n'
        '    "category": "accelerator",\n'
        '    "required_devices": [],\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        f'    return {{"plan": {name!r}}}\n'
    )


def _write_session_plan(tmp_path: Path, name: str, source: str) -> Path:
    session_dir = tmp_path / "plans_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.py"
    path.write_text(source)
    return path


# ---------------------------------------------------------------------------
# Happy path: draft snapshot minted, launched, and recorded as launched
# ---------------------------------------------------------------------------


def test_happy_path_launches_the_draft_snapshot(client: TestClient) -> None:
    revision = _seed_orm_draft(client)

    resp = _launch(client, revision)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"
    assert body["launched_by"] == "draft"
    assert body["run_uid"]

    # Exactly one run, minted from the SNAPSHOT (never the request body).
    assert len(registry._runs) == 1
    run = registry.get(body["id"])
    assert run.request.plan_name == "orm"
    assert run.request.plan_args == _ORM_ARGS


def test_happy_path_leaves_the_draft_intact(client: TestClient) -> None:
    """Launch never clears the draft or bumps its revision (a later PATCH re-arms)."""
    revision = _seed_orm_draft(client)
    _launch(client, revision)

    body = client.get("/draft").json()
    assert body["draft"] is not None
    assert body["draft"]["plan_name"] == "orm"
    assert body["revision"] == revision


@pytest.mark.asyncio
async def test_happy_path_emits_a_launched_frame(client: TestClient) -> None:
    """The `launched` SSE frame carries the minted run's id and the pinned
    revision. White-box `draft._subscribe()` precedent (module docstring):
    the sync `client.post(...)` and the `await draft.*` calls never overlap
    in time, so `draft.py`'s lazily-loop-bound primitives stay safe.
    """
    revision = _seed_orm_draft(client)

    queue, _hello = await draft._subscribe()
    try:
        resp = _launch(client, revision)
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["id"]

        frame = queue.get_nowait()
        assert frame == {"type": "launched", "run_id": run_id, "revision": revision}
        assert queue.empty()
    finally:
        await draft._unsubscribe(queue)


# ---------------------------------------------------------------------------
# 409 stale_draft_revision: revision mismatch / never-set / cleared — all
# mint nothing
# ---------------------------------------------------------------------------


def test_stale_revision_after_patch_mints_nothing(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    client.patch("/draft", json={"plan_args_patch": {"num": 5}, "client_id": "panel-1"})

    resp = _launch(client, revision)

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "stale_draft_revision"
    assert detail["revision"] == revision + 1  # fresh baseline for resync
    assert registry._runs == {}


def test_never_set_draft_mints_nothing(client: TestClient) -> None:
    resp = _launch(client, 0)

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "stale_draft_revision"
    assert registry._runs == {}


def test_cleared_draft_mints_nothing(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    client.delete("/draft")

    resp = _launch(client, revision)

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "stale_draft_revision"
    assert registry._runs == {}


# ---------------------------------------------------------------------------
# 409 draft_revision_already_launched: a replay never fires a second run
# ---------------------------------------------------------------------------


def test_replayed_revision_mints_no_second_run(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    assert _launch(client, revision).status_code == 200

    replay = _launch(client, revision)

    assert replay.status_code == 409
    detail = replay.json()["detail"]
    assert detail["code"] == "draft_revision_already_launched"
    assert detail["revision"] == revision
    assert len(registry._runs) == 1  # exactly the first launch's record


@pytest.mark.asyncio
async def test_concurrent_same_revision_launches_mint_exactly_one_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two OVERLAPPING launches at the SAME revision: exactly one run mints,
    the loser 409s while the winner is still mid-launch.

    This is the window the in-flight reservation exists for: the winner sits
    suspended in the unlocked threadpool mint/launch (past `check_launchable`,
    before `record_and_broadcast_launch`), so a tail-committed guard alone
    would let the loser's own check pass too. The sync `TestClient`
    serializes requests and can never observe this, so the app is driven
    through `httpx.ASGITransport` on one event loop, with a runner that
    blocks inside `reinitialize` until the test releases it.
    """
    monkeypatch.setenv(_LAUNCH_TOKEN_ENV, _TOKEN)

    winner_mid_launch = threading.Event()
    finish_launch = threading.Event()

    class BlockingRunner(FakePlanRunner):
        def reinitialize(self, exec_config: Any) -> bool:
            winner_mid_launch.set()
            assert finish_launch.wait(timeout=10), "test never released the blocked launch"
            return super().reinitialize(exec_config)

    set_runner_factory(BlockingRunner)

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://bridge") as client:
            seed = await client.patch(
                "/draft",
                json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "panel-1"},
            )
            assert seed.status_code == 200, seed.text
            revision = int(seed.json()["revision"])

            headers = {"X-Launch-Token": _TOKEN}
            winner = asyncio.create_task(
                client.post("/draft/run", json={"draft_revision": revision}, headers=headers)
            )
            # Wait (off-loop) until the winner is INSIDE the unlocked
            # mint/launch window — exactly where the race lived.
            assert await asyncio.to_thread(winner_mid_launch.wait, 10)

            loser = await client.post(
                "/draft/run", json={"draft_revision": revision}, headers=headers
            )
            assert loser.status_code == 409
            assert loser.json()["detail"]["code"] == "draft_revision_already_launched"

            finish_launch.set()
            winner_resp = await winner
            assert winner_resp.status_code == 200, winner_resp.text
    finally:
        finish_launch.set()  # never leave the threadpool thread blocked

    assert len(registry._runs) == 1  # the winner's run, and only the winner's


# ---------------------------------------------------------------------------
# Token gate: 403/503 exactly like the launch route, before ANY state is
# touched — mints nothing, consumes nothing
# ---------------------------------------------------------------------------


def test_missing_token_is_403_and_mints_nothing(client: TestClient) -> None:
    revision = _seed_orm_draft(client)

    resp = _launch(client, revision, token=None)

    assert resp.status_code == 403
    assert registry._runs == {}
    # Launchability was never consumed: the same revision launches fine.
    assert _launch(client, revision).status_code == 200


def test_wrong_token_is_403_and_mints_nothing(client: TestClient) -> None:
    revision = _seed_orm_draft(client)

    resp = _launch(client, revision, token="wrong-token")

    assert resp.status_code == 403
    assert registry._runs == {}


def test_unarmed_bridge_is_503_and_mints_nothing(unarmed_client: TestClient) -> None:
    revision = _seed_orm_draft(unarmed_client)

    resp = _launch(unarmed_client, revision)

    assert resp.status_code == 503
    assert registry._runs == {}


# ---------------------------------------------------------------------------
# Validation-gate rejection: mints NOTHING (no eternal pre-launch record)
# ---------------------------------------------------------------------------


def test_validation_gate_rejection_mints_nothing(tmp_path: Path, client: TestClient) -> None:
    """A session plan whose file was edited after validation (the launch-time
    race the gate exists for) 409s with the registry untouched — the gate
    runs BEFORE `registry.add` on this path, so there is no record to go
    eternal. The revision is not consumed either: re-validating the edited
    content lets the SAME pinned revision launch.
    """
    source = _session_plan_source("draft_gated_plan")
    path = _write_session_plan(tmp_path, "draft_gated_plan", source)
    validation_records.record(hash_plan_body(source))

    resp = client.patch(
        "/draft", json={"plan_name": "draft_gated_plan", "client_id": "panel-1"}
    )
    assert resp.status_code == 200, resp.text
    revision = int(resp.json()["revision"])

    # Edit the file (still well-formed) without re-validating it.
    edited = source.replace(
        '"description": "A session-tier test plan."', '"description": "edited"'
    )
    path.write_text(edited)

    rejected = _launch(client, revision)
    assert rejected.status_code == 409
    assert "no passing validation record" in str(rejected.json()["detail"])
    assert registry._runs == {}
    assert draft._launching == set()  # reservation released, not leaked

    # Re-validate the current content: the pinned revision was never marked
    # launched, so the same launch request now succeeds.
    validation_records.record(hash_plan_body(edited))
    assert _launch(client, revision).status_code == 200


# ---------------------------------------------------------------------------
# Post-mint launch failure: record stamped `error`, revision not consumed
# ---------------------------------------------------------------------------


def test_post_mint_launch_failure_stamps_error(client: TestClient) -> None:
    set_runner_factory(lambda: FakePlanRunner(reinitialize_fails=True))
    revision = _seed_orm_draft(client)

    resp = _launch(client, revision)

    assert resp.status_code == 500
    assert len(registry._runs) == 1
    (run,) = registry._runs.values()
    assert run.status == "error"
    assert run.error


def test_failed_launch_does_not_consume_the_revision(client: TestClient) -> None:
    """`record_and_broadcast_launch` only runs on success — after a failed
    launch the same revision is retryable once the runner is healthy."""
    set_runner_factory(lambda: FakePlanRunner(reinitialize_fails=True))
    revision = _seed_orm_draft(client)
    assert _launch(client, revision).status_code == 500
    assert draft._launching == set()  # failure released the reservation

    set_runner_factory(FakePlanRunner)
    retry = _launch(client, revision)

    assert retry.status_code == 200
    assert retry.json()["status"] == "running"
    # Two records total: the error-stamped first attempt plus the retry.
    assert len(registry._runs) == 2


# ---------------------------------------------------------------------------
# PATCH-then-launch: a post-launch edit re-arms the new revision
# ---------------------------------------------------------------------------


def test_patch_after_launch_re_arms_the_new_revision(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    assert _launch(client, revision).status_code == 200

    new_revision = int(
        client.patch(
            "/draft", json={"plan_args_patch": {"num": 5}, "client_id": "panel-1"}
        ).json()["revision"]
    )
    assert new_revision == revision + 1

    resp = _launch(client, new_revision)

    assert resp.status_code == 200, resp.text
    assert len(registry._runs) == 2
    run = registry.get(resp.json()["id"])
    assert run.request.plan_args["num"] == 5


# ---------------------------------------------------------------------------
# Route safety: OpenAPI paths, never composed `app.routes`
# ---------------------------------------------------------------------------


def test_draft_run_route_is_post_only() -> None:
    paths = app.openapi()["paths"]
    assert "/draft/run" in paths
    assert set(paths["/draft/run"].keys()) == {"post"}


def test_existing_draft_and_launch_routes_are_untouched() -> None:
    paths = app.openapi()["paths"]
    assert {"get", "patch", "delete"} <= set(paths["/draft"].keys())
    assert set(paths["/draft/events"].keys()) == {"get"}
    assert "post" in paths["/runs/{run_id}/launch"]
