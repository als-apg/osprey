"""Coverage for the draft's launch-state primitives (`draft.py`):
`check_launchable`, `record_and_broadcast_launch`, and the `launched` SSE
frame — the substrate for `POST /draft/run` (PROPOSAL.md "Bridge draft
module", Task 2.1).

Like `test_draft.py`, uses the shipped `orm` plan as a real schema (the
`shipped` tier is an in-package directory scan, always registered) and
mirrors its isolation fixture, resetting the draft singleton via
`draft._clear()`. The `launched`-frame delivery test follows the white-box
`draft._subscribe()` precedent (`TestClient`/`ASGITransport` drain streaming
responses, so the SSE generator can't be driven in-process) rather than the
live-uvicorn `_live_bridge` route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import draft, plan_loader
from osprey.services.bluesky_bridge.app import app

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"

_ORM_ARGS: dict[str, Any] = {
    "correctors": ["COR1"],
    "detectors": ["BPM1"],
    "span_a": 1.0,
    "num": 3,
}


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()
    draft._clear()
    yield
    plan_loader.reset_facility_plans()
    draft._clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_orm_draft(client: TestClient) -> int:
    """Create an `orm` draft and return its revision."""
    resp = client.patch(
        "/draft",
        json={"plan_name": "orm", "plan_args_patch": _ORM_ARGS, "client_id": "agent-1"},
    )
    assert resp.status_code == 200
    return int(resp.json()["revision"])


# ---------------------------------------------------------------------------
# check_launchable: success on match + not-yet-launched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_launchable_returns_snapshot_on_match(client: TestClient) -> None:
    revision = _seed_orm_draft(client)

    result = await draft.check_launchable(revision)

    assert isinstance(result, draft.LaunchSnapshot)
    assert result.revision == revision
    assert result.plan_name == "orm"
    assert result.plan_args == _ORM_ARGS


@pytest.mark.asyncio
async def test_snapshot_plan_args_is_a_defensive_copy(client: TestClient) -> None:
    """Mutating the returned snapshot's plan_args must not touch the live draft."""
    revision = _seed_orm_draft(client)

    result = await draft.check_launchable(revision)
    assert isinstance(result, draft.LaunchSnapshot)
    result.plan_args["num"] = 999

    assert client.get("/draft").json()["draft"]["plan_args"]["num"] == 3


# ---------------------------------------------------------------------------
# check_launchable: typed stale failure (revision mismatch / no draft)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_launchable_stale_on_revision_mismatch(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    # A subsequent edit bumps the revision; the caller's pinned one is now stale.
    client.patch("/draft", json={"plan_args_patch": {"num": 5}, "client_id": "agent-1"})

    result = await draft.check_launchable(revision)

    assert isinstance(result, draft.LaunchRejected)
    assert result.code == "stale_draft_revision"
    assert result.revision == revision + 1


@pytest.mark.asyncio
async def test_check_launchable_stale_on_empty_draft(client: TestClient) -> None:
    # No draft has ever been created; revision is still 0.
    result = await draft.check_launchable(0)

    assert isinstance(result, draft.LaunchRejected)
    assert result.code == "stale_draft_revision"
    assert result.revision == 0


@pytest.mark.asyncio
async def test_check_launchable_stale_after_clear(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    client.delete("/draft")  # draft is now None, revision bumped

    result = await draft.check_launchable(revision)

    assert isinstance(result, draft.LaunchRejected)
    assert result.code == "stale_draft_revision"


# ---------------------------------------------------------------------------
# check_launchable: typed already-launched failure on replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_launchable_already_launched_on_replay(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    await draft.record_and_broadcast_launch(run_id="run-1", revision=revision)

    result = await draft.check_launchable(revision)

    assert isinstance(result, draft.LaunchRejected)
    assert result.code == "draft_revision_already_launched"
    assert result.revision == revision


# ---------------------------------------------------------------------------
# record_launch then PATCH re-arms launch (new revision is launchable)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_re_arms_launch_after_record(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    await draft.record_and_broadcast_launch(run_id="run-1", revision=revision)

    # The draft persists after launch; a PATCH bumps its revision and re-arms.
    new_revision = int(
        client.patch("/draft", json={"plan_args_patch": {"num": 5}, "client_id": "agent-1"}).json()[
            "revision"
        ]
    )
    assert new_revision == revision + 1

    result = await draft.check_launchable(new_revision)
    assert isinstance(result, draft.LaunchSnapshot)
    assert result.revision == new_revision
    assert result.plan_args["num"] == 5


# ---------------------------------------------------------------------------
# In-flight reservation: check_launchable reserves the revision at the HEAD
# of the unlocked launch window; record consumes it, release_launch drops it
# without recording. This is what makes "exactly one launch per revision"
# hold for CONCURRENT callers, not just sequential replays.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_launchable_reserves_the_revision(client: TestClient) -> None:
    """A second check at the same revision is refused while the first launch
    is in flight (reserved, not yet recorded or released)."""
    revision = _seed_orm_draft(client)

    first = await draft.check_launchable(revision)
    assert isinstance(first, draft.LaunchSnapshot)

    second = await draft.check_launchable(revision)
    assert isinstance(second, draft.LaunchRejected)
    assert second.code == "draft_revision_already_launched"
    assert "in progress" in second.detail
    assert second.revision == revision


@pytest.mark.asyncio
async def test_release_launch_re_arms_the_same_revision(client: TestClient) -> None:
    """Releasing without recording (the failure path) makes the SAME revision
    immediately launchable again — a failed launch never consumes it."""
    revision = _seed_orm_draft(client)
    assert isinstance(await draft.check_launchable(revision), draft.LaunchSnapshot)

    await draft.release_launch(revision)

    retry = await draft.check_launchable(revision)
    assert isinstance(retry, draft.LaunchSnapshot)
    assert retry.revision == revision


@pytest.mark.asyncio
async def test_record_consumes_the_reservation(client: TestClient) -> None:
    """Success settles the reservation into `_last_launched_revision`: the
    in-flight set empties and a replayed check gets the already-launched
    refusal (now via the recorded state, not the reservation)."""
    revision = _seed_orm_draft(client)
    assert isinstance(await draft.check_launchable(revision), draft.LaunchSnapshot)

    await draft.record_and_broadcast_launch(run_id="run-1", revision=revision)

    assert draft._launching == set()
    replay = await draft.check_launchable(revision)
    assert isinstance(replay, draft.LaunchRejected)
    assert replay.code == "draft_revision_already_launched"


@pytest.mark.asyncio
async def test_release_launch_tolerates_an_unreserved_revision() -> None:
    await draft.release_launch(123)  # idempotent no-op, must not raise
    assert draft._launching == set()


@pytest.mark.asyncio
async def test_clear_resets_the_reservation_set(client: TestClient) -> None:
    revision = _seed_orm_draft(client)
    assert isinstance(await draft.check_launchable(revision), draft.LaunchSnapshot)

    draft._clear()

    assert draft._launching == set()


# ---------------------------------------------------------------------------
# launched frame reaches subscribers (white-box _subscribe precedent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launched_frame_reaches_subscribers(client: TestClient) -> None:
    """`record_and_broadcast_launch` emits a `launched` frame to every SSE
    subscriber via the same `_broadcast_locked` mechanism as `change` frames.

    Follows the white-box `_subscribe()` precedent (see `test_draft.py`'s
    no-op-frame test): the sync `client.patch(...)` and the `await
    draft.*` calls never overlap in time, so `draft.py`'s lazily-loop-bound
    module primitives stay safe.
    """
    revision = _seed_orm_draft(client)

    queue, _hello = await draft._subscribe()
    try:
        await draft.record_and_broadcast_launch(run_id="run-42", revision=revision)
        frame = queue.get_nowait()
        assert frame == {"type": "launched", "run_id": "run-42", "revision": revision}
        assert queue.empty()
    finally:
        await draft._unsubscribe(queue)


@pytest.mark.asyncio
async def test_record_launch_does_not_clear_draft(client: TestClient) -> None:
    """The draft persists after launch (a later PATCH re-arms it)."""
    revision = _seed_orm_draft(client)
    await draft.record_and_broadcast_launch(run_id="run-1", revision=revision)

    body = client.get("/draft").json()
    assert body["draft"] is not None
    assert body["draft"]["plan_name"] == "orm"
    assert body["revision"] == revision  # launch itself does not bump the revision
