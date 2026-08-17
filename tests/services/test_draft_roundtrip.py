"""Cross-service integration roundtrip locking the agent-plan-draft contract:
PATCH draft -> SSE frame -> a pinned revision that only the queue will take.

Here, the real ``osprey.interfaces.bluesky_panels.app`` sidecar is wired to the
real ``osprey.services.bluesky_bridge.app`` bridge via ``httpx.ASGITransport``
(no real bridge process, no container) -- so a PATCH sent through the sidecar's
``/draft`` relay actually lands on the bridge's draft singleton, and the SSE
frame it broadcasts is observed on the bridge itself. That cross-service hop is
this module's whole subject.

Launching is not part of that chain: plans execute in the queue server, so
``POST /draft/run`` launches nothing and answers the machine-readable
``use_the_queue`` refusal instead. This module asserts exactly that -- a caller
who follows the PATCH with a launch call learns where the capability lives. It
deliberately stops there. The draft-revision-to-enqueue
path (reservation, arming, what reaches the manager) is
``tests/integration/test_bluesky_queue_contract.py``'s subject, against a
stateful mock queue server; duplicating it here would mean two places to update
and one of them silently going stale.

Everything else is covered per-router elsewhere:

- ``tests/interfaces/bluesky_panels/test_draft_relay.py`` covers the sidecar's
  routers in isolation (mocked bridge).
- ``tests/services/bluesky_bridge/test_draft.py`` covers the bridge draft
  module's SSE wire format and per-field validation directly.
- ``tests/services/bluesky_bridge/test_run_routes.py`` covers every retired
  route's refusal against the bridge directly.

SSE observation note: ``TestClient``/``httpx.ASGITransport`` cannot stream an
infinite ``text/event-stream`` response (both drain the response body to
completion before returning it), so this test does not connect to
``GET /draft/events`` over HTTP at all. Instead it uses the bridge draft
module's own white-box subscription hook (``draft._subscribe()``), exactly as
``tests/services/bluesky_bridge/test_draft.py`` does for its deterministic
frame assertions. Subscribing directly against the bridge module is legitimate
here specifically because the sidecar and the bridge are the same importable
Python objects in this test process (connected by ``ASGITransport`` rather
than a real socket) -- the SSE wire format itself is already locked by
``test_draft.py``; this test's job is only the cross-service chain around it.

The subscribe-then-patch sequencing below never has the bridge's asyncio
``_lock``/subscriber ``Queue`` contended from two event loops at once (the
synchronous sidecar ``TestClient`` call fully completes on its own portal
loop before this test's own coroutine resumes) -- the same non-concurrent
cross-loop pattern already relied on by
``test_draft.py::test_subscribe_hello_is_atomic_with_current_state``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_panels.app import app as sidecar_app
from osprey.services.bluesky_bridge import app as bridge_app_module
from osprey.services.bluesky_bridge import draft as bridge_draft
from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge import queue as bridge_queue
from osprey.services.bluesky_bridge.app import app as bridge_app

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_LAUNCH_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"

_TOKEN = "s3cr3t-integration-roundtrip-token"  # noqa: S105 - test fixture value, not a real secret
_BRIDGE_URL = "http://bridge.test"

# `grid_scan` is a shipped-tier plan (`plans_core/`) -- always registered
# regardless of env (mirrors `test_draft.py`'s own choice of real schema over a
# throwaway fixture plan).
_PLAN_NAME = "grid_scan"
_PLAN_ARGS: dict[str, Any] = {
    "readables": ["BPM1"],
    "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
}


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every process-wide singleton this roundtrip touches.

    Mirrors `test_plan_source_endpoint.py`'s plan-loader isolation plus
    `test_draft.py`'s draft-singleton reset.
    """
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.setenv(_LAUNCH_TOKEN_ENV, _TOKEN)
    plan_loader.reset_facility_plans()
    bridge_draft._clear()
    bridge_queue._clear()
    bridge_app_module.set_queue_backend(None)

    yield

    plan_loader.reset_facility_plans()
    bridge_draft._clear()
    bridge_queue._clear()
    bridge_app_module.set_queue_backend(None)


@pytest.fixture
def bridge_client() -> Iterator[httpx.AsyncClient]:
    """A direct async client on the real bridge app, over ASGI."""
    yield httpx.AsyncClient(transport=httpx.ASGITransport(app=bridge_app), base_url=_BRIDGE_URL)


@pytest.fixture
def sidecar_client(bridge_client: httpx.AsyncClient) -> Iterator[TestClient]:
    """The composed sidecar app, with its bridge client rewired onto the real
    bridge app via `ASGITransport` instead
    of a real network hop -- mirrors `test_app_integration.py`'s
    `_wire_mock_bridge` pattern, but targets the real bridge ASGI app rather
    than an `httpx.MockTransport` handler.
    """
    with TestClient(sidecar_app) as client:
        sidecar_app.state.client = bridge_client
        sidecar_app.state.bridge_url = _BRIDGE_URL
        yield client


@pytest.mark.asyncio
async def test_patch_through_the_sidecar_reaches_the_bridge_and_broadcasts(
    sidecar_client: TestClient,
) -> None:
    """PATCH (sidecar relay) -> the bridge's own draft state -> the SSE frame
    it broadcasts, all in one chain across the two services."""
    # Subscribe before the PATCH so the queue captures the live change frame,
    # not just a hello snapshot (module docstring: sequencing, not concurrency,
    # is what keeps this cross-loop-safe).
    frames, hello = await bridge_draft._subscribe()
    assert hello == {"type": "hello", "draft": None, "revision": 0}

    try:
        patch_response = sidecar_client.patch(
            "/draft",
            json={
                "plan_name": _PLAN_NAME,
                "plan_args_patch": _PLAN_ARGS,
                "client_id": "roundtrip-test",
            },
        )
        assert patch_response.status_code == 200, patch_response.text
        patch_body = patch_response.json()
        assert patch_body["plan_name"] == _PLAN_NAME
        assert patch_body["revision"] == 1
        assert set(patch_body["changed"]) == set(_PLAN_ARGS)

        frame = await asyncio.wait_for(frames.get(), timeout=2.0)
    finally:
        await bridge_draft._unsubscribe(frames)

    # The SSE frame the PATCH broadcast carries the same revision/changed[]
    # the sidecar's own PATCH response reported.
    assert frame["type"] == "plan-change"
    assert frame["revision"] == patch_body["revision"]
    assert set(frame["changed"]) == set(patch_body["changed"])
    assert frame["draft"]["plan_name"] == _PLAN_NAME

    # And the bridge really holds it, read back through the sidecar's relay.
    current = sidecar_client.get("/draft").json()
    assert current["draft"]["plan_name"] == _PLAN_NAME
    assert current["revision"] == patch_body["revision"]


@pytest.mark.asyncio
async def test_launching_the_pinned_revision_directly_is_refused(
    sidecar_client: TestClient, bridge_client: httpx.AsyncClient
) -> None:
    """The third step of the original chain, as it answers today.

    A caller that follows the PATCH with the old direct-launch call gets the
    machine-readable refusal naming the route that replaced it -- not a 404,
    and not a silent success. The draft is untouched: refusing to launch must
    not consume the revision a caller would then enqueue.
    """
    patch_response = sidecar_client.patch(
        "/draft",
        json={
            "plan_name": _PLAN_NAME,
            "plan_args_patch": _PLAN_ARGS,
            "client_id": "roundtrip-test",
        },
    )
    assert patch_response.status_code == 200, patch_response.text
    revision = patch_response.json()["revision"]

    refused = await bridge_client.post(
        "/draft/run",
        json={"draft_revision": revision},
        headers={"X-Launch-Token": _TOKEN},
    )

    assert refused.status_code == 410
    detail = refused.json()["detail"]
    assert detail["code"] == "use_the_queue"
    assert "POST /queue/items" in detail["detail"]

    # Nothing consumed: the revision still stands and the draft still holds the
    # plan, so the caller can go straight to the queue with the same pin.
    current = sidecar_client.get("/draft").json()
    assert current["revision"] == revision
    assert current["draft"]["plan_name"] == _PLAN_NAME
    assert bridge_draft._last_launched_revision == 0
