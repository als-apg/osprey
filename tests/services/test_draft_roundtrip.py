"""Cross-service integration roundtrip locking the agent-plan-draft contract
(PROPOSAL.md "Tests": "One non-agentic integration roundtrip: PATCH draft ->
SSE frame -> launch pinned revision").

This is the ONE test that exercises the full chain end to end -- everything
else (per-router unit tests, per-field validation, launch-token gating) is
covered elsewhere:

- ``tests/services/bluesky_panels/test_draft_relay.py`` / ``test_launch.py``
  cover the sidecar's routers in isolation (mocked bridge).
- ``tests/services/bluesky_bridge/test_draft.py`` covers the bridge draft
  module's SSE wire format and per-field validation directly.
- ``tests/services/bluesky_bridge/test_launch_validation_gate.py`` covers
  the launch-time session-plan validation gate.

Here, the real ``osprey.services.bluesky_panels.app`` sidecar is wired to the
real ``osprey.services.bluesky_bridge.app`` bridge via
``httpx.ASGITransport`` (no real bridge process, no container) -- so a PATCH
sent through the sidecar's ``/draft`` relay actually lands on the bridge's
draft singleton, and a ``POST /runs/launch`` actually creates and launches a
run on the bridge's real run registry (with the bridge's default
``FakePlanRunner`` standing in for a real bluesky ``RunEngine`` -- see
``plan_runner.py``; PLAN.md's task 5.1 acceptance is "run created (mock
runner)", not a real scan).

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

from osprey.services.bluesky_bridge import draft as bridge_draft
from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app as bridge_app
from osprey.services.bluesky_bridge.app import set_runner_factory
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import registry as bridge_run_registry
from osprey.services.bluesky_panels.app import app as sidecar_app

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_LAUNCH_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"

_TOKEN = "s3cr3t-integration-roundtrip-token"  # noqa: S105 - test fixture value, not a real secret
_BRIDGE_URL = "http://bridge.test"

# `grid_scan` is a shipped-tier plan (`plans_core/`) -- always registered
# regardless of env, and never subject to the session-tier launch-validation
# gate (mirrors `test_draft.py`'s own choice of real schema over a throwaway
# fixture plan).
_PLAN_NAME = "grid_scan"
_PLAN_ARGS: dict[str, Any] = {
    "detectors": ["BPM1"],
    "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
}


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every process-wide singleton this roundtrip touches.

    Mirrors `test_plan_source_endpoint.py`'s plan-loader isolation plus
    `test_draft.py`'s draft-singleton reset and
    `test_launch_validation_gate.py`'s run-registry/runner-factory reset --
    this test is the union of all three surfaces.
    """
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.setenv(_LAUNCH_TOKEN_ENV, _TOKEN)
    plan_loader.reset_facility_plans()
    bridge_draft._clear()
    bridge_run_registry._runs.clear()
    set_runner_factory(FakePlanRunner)

    yield

    plan_loader.reset_facility_plans()
    bridge_draft._clear()
    bridge_run_registry._runs.clear()
    set_runner_factory(FakePlanRunner)


@pytest.fixture
def sidecar_client() -> Iterator[TestClient]:
    """The composed sidecar app, with its bridge client rewired onto the real
    bridge app via `ASGITransport` (task 5.1's cross-service wiring) instead
    of a real network hop -- mirrors
    `test_app_integration.py`'s `_wire_mock_bridge` pattern, but targets the
    real bridge ASGI app rather than an `httpx.MockTransport` handler.
    """
    with TestClient(sidecar_app) as client:
        sidecar_app.state.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=bridge_app), base_url=_BRIDGE_URL
        )
        sidecar_app.state.bridge_url = _BRIDGE_URL
        yield client


@pytest.mark.asyncio
async def test_patch_sse_launch_roundtrip_through_sidecar_and_bridge(
    sidecar_client: TestClient,
) -> None:
    """PATCH (sidecar relay) -> SSE frame (bridge white-box) -> launch (sidecar
    launch route, pinned to the PATCH's revision) -> a real run launched on
    the bridge's registry with a mock (`FakePlanRunner`) runner.
    """
    # Subscribe before the PATCH so the queue captures the live change frame,
    # not just a hello snapshot (module docstring: sequencing, not concurrency,
    # is what keeps this cross-loop-safe).
    queue, hello = await bridge_draft._subscribe()
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

        frame = await asyncio.wait_for(queue.get(), timeout=2.0)
    finally:
        await bridge_draft._unsubscribe(queue)

    # (b) the SSE frame the PATCH broadcast carries the same revision/changed[]
    # the sidecar's own PATCH response reported.
    assert frame["type"] == "plan-change"
    assert frame["revision"] == patch_body["revision"]
    assert set(frame["changed"]) == set(patch_body["changed"])
    assert frame["draft"]["plan_name"] == _PLAN_NAME

    # (c) launch the pinned revision through the sidecar; the launched args
    # come from the bridge's draft snapshot, never from this request body.
    launch_response = sidecar_client.post(
        "/runs/launch", json={"draft_revision": patch_body["revision"]}
    )
    assert launch_response.status_code == 200, launch_response.text
    launch_body = launch_response.json()
    assert launch_body["status"] == "running"
    run_id = launch_body["run_id"]

    # The bridge really did create and launch a run with this exact plan --
    # observed through the sidecar's own read-proxy, closing the loop.
    run_response = sidecar_client.get(f"/runs/{run_id}")
    assert run_response.status_code == 200, run_response.text
    run_body = run_response.json()
    assert run_body["status"] == "running"
    assert run_body["plan_name"] == _PLAN_NAME
    assert run_body["plan_args"] == _PLAN_ARGS


def test_launch_with_a_stale_draft_revision_is_409(sidecar_client: TestClient) -> None:
    """(d) A pin from before the draft was cleared can never match again --
    the bridge's revision counter bumps on `DELETE` and never resets."""
    patch_response = sidecar_client.patch(
        "/draft",
        json={
            "plan_name": _PLAN_NAME,
            "plan_args_patch": _PLAN_ARGS,
            "client_id": "roundtrip-test",
        },
    )
    assert patch_response.status_code == 200, patch_response.text
    stale_revision = patch_response.json()["revision"]

    delete_response = sidecar_client.request("DELETE", "/draft")
    assert delete_response.status_code == 200, delete_response.text
    assert delete_response.json()["cleared"] is True

    launch_response = sidecar_client.post("/runs/launch", json={"draft_revision": stale_revision})

    assert launch_response.status_code == 409
    assert launch_response.json()["code"] == "stale_draft_revision"
