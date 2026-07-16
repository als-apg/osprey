"""RunEngine integration test for `BlueskyPlanRunner` (task 2.7).

Runs ONLY in a bluesky-capable environment — `bluesky`/`ophyd-async` are
never installed in the main worktree venv, so every test here is skipped via
`pytest.importorskip` rather than failing, keeping `ci_check` green with no
bluesky installed at all. To actually run this file:

    uv venv /tmp/bluesky-runengine-scratch
    /tmp/bluesky-runengine-scratch/bin/pip install -e '.[bluesky-bridge]' --python 3.11
    /tmp/bluesky-runengine-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_runengine_integration.py -q

Proves the full chain end to end: a real bluesky `RunEngine` running a plan
against mock ophyd-async devices (`devices/mock.py`) produces documents, the
run reaches `completed`, its `error_message` stays unset, and
`read_run_data` (via the real bridge route, not a patched one) returns the
buffered rows.

The v1 hand-built plan set that used to carry `scan`/`count` is gone (see
`plan_loader.get_facility_plans`, now the sole plan registry) — this file's
own point is exercising the bridge's plumbing (`reinitialize`/
`start_run_thread`/the live-row buffer), not a
specific shipped plan's device motion (the shipped `orm`/`grid_scan` plans'
own RunEngine round trip is `test_exemplar_plans.py`'s job), so a trivial
local `buffer_probe` fixture plan (read a detector N times, move nothing)
stands in below — registered in a small local registry rather than resolved
through `get_facility_plans()`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from bluesky import plans as bp  # noqa: E402

from osprey.services.bluesky_bridge import live_rows  # noqa: E402
from osprey.services.bluesky_bridge.app import app, set_runner_factory  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import build_devices  # noqa: E402
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner  # noqa: E402
from osprey.services.bluesky_bridge.plan_runner_bluesky import BlueskyPlanRunner  # noqa: E402
from osprey.services.bluesky_bridge.plan_types import PlanSpec  # noqa: E402
from osprey.services.bluesky_bridge.runs import do_promote, registry  # noqa: E402


class _CountParams(BaseModel):
    """A trivial no-motor plan's params: read `detectors` `num` times."""

    detectors: list[str] = Field(..., min_length=1)
    num: int = Field(..., ge=1)


def _count_plan(devices: dict[str, Any], params: _CountParams) -> Any:
    detector_devices = [devices[name] for name in params.detectors]
    return bp.count(detector_devices, num=params.num)


# The local registry every test below resolves plan names against — a
# trivial no-motor buffering/read smoke, standing in for the removed
# built-in `count`/`scan` (see module docstring).
_TEST_PLANS: dict[str, PlanSpec[Any]] = {
    "buffer_probe": PlanSpec(
        name="buffer_probe", plan=_count_plan, schema=_CountParams, description="test-only"
    ),
}


def _wait_until_idle(runner: BlueskyPlanRunner, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.is_run_active():
        if time.monotonic() > deadline:
            raise AssertionError("scan did not finish within the timeout")
        time.sleep(0.05)


@pytest.fixture(autouse=True)
def _isolated_state():
    registry._runs.clear()
    live_rows._clear()
    set_runner_factory(FakePlanRunner)
    yield
    registry._runs.clear()
    live_rows._clear()
    set_runner_factory(FakePlanRunner)


@pytest.fixture(scope="module")
def mock_devices() -> dict:
    """One connected mock motor + detector, shared across this module's tests."""
    return asyncio.run(build_devices(motor_names=["motor1"], detector_names=["det1"]))


# =========================================================================
# Direct runner-level: RunEngine actually runs the no-motor fixture plan
# against mock devices
# =========================================================================


def test_buffer_probe_plan_runs_to_completion_and_buffers_rows(mock_devices: dict) -> None:
    runner = BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS)
    exec_config = {"plan_name": "buffer_probe", "plan_args": {"detectors": ["det1"], "num": 3}}

    assert runner.reinitialize(exec_config) is True
    assert runner.current_state == "armed"

    runner.start_run_thread()
    _wait_until_idle(runner)

    assert runner.error_message is None
    assert runner.current_state == "completed"
    assert runner.last_run_uid is not None
    assert runner.estimate_current_completion() == 1.0

    buf = live_rows.get(runner.last_run_uid)
    assert buf is not None
    assert buf["partial"] is False
    assert buf["total_seen"] == 3  # 3 readings
    assert len(buf["rows"]) == 3
    assert "det1" in buf["columns"] or any("det1" in col for col in buf["columns"])


def test_buffer_probe_plan_against_mock_devices(mock_devices: dict) -> None:
    runner = BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS)
    exec_config = {"plan_name": "buffer_probe", "plan_args": {"detectors": ["det1"], "num": 4}}

    assert runner.reinitialize(exec_config) is True
    runner.start_run_thread()
    _wait_until_idle(runner)

    assert runner.error_message is None
    buf = live_rows.get(runner.last_run_uid)
    assert buf is not None
    assert buf["total_seen"] == 4


def test_reinitialize_bridges_an_async_device_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """`devices/mock.py`'s `build_devices` is `async def` — a real deploy wiring
    would pass it (or an equivalent facility factory) directly rather than a
    pre-resolved dict. `reinitialize()` must bridge that, not assume sync.
    """
    runner = BlueskyPlanRunner(
        devices=lambda: build_devices(motor_names=["motor1"], detector_names=["det1"]),
        plans=_TEST_PLANS,
    )
    exec_config = {"plan_name": "buffer_probe", "plan_args": {"detectors": ["det1"], "num": 2}}

    assert runner.reinitialize(exec_config) is True

    runner.start_run_thread()
    _wait_until_idle(runner)

    assert runner.error_message is None
    assert runner.current_state == "completed"
    buf = live_rows.get(runner.last_run_uid)
    assert buf is not None
    assert buf["total_seen"] == 2


def test_stop_run_thread_aborts_a_running_scan(mock_devices: dict) -> None:
    runner = BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS)
    exec_config = {"plan_name": "buffer_probe", "plan_args": {"detectors": ["det1"], "num": 1000}}

    assert runner.reinitialize(exec_config) is True
    runner.start_run_thread()
    time.sleep(0.1)  # let the scan actually get going before aborting it

    runner.stop_run_thread()

    assert runner.is_run_active() is False
    assert runner.current_state == "stopped"  # terminal state distinguishes stop from completed
    assert runner.error_message is None  # an intentional stop is not an error


def test_run_thread_sets_error_state_when_the_plan_raises_mid_run(mock_devices: dict) -> None:
    """A plan that fails only once the RunEngine drives it in the daemon thread
    (not during `reinitialize()`) must land the runner in the 'error' terminal
    state with `error_message` set. This exercises `_run()`'s thread-level
    failure branch — distinct from `reinitialize()`'s pre-thread validation
    failures (unknown plan/device) covered above, where no thread ever starts.
    """
    from pydantic import BaseModel

    from osprey.services.bluesky_bridge.plan_types import PlanSpec

    class _NoParams(BaseModel):
        pass

    def _boom_plan(devices: dict, params: _NoParams):
        # `yield` makes this a generator function, so the body runs only when
        # the RunEngine iterates it inside `_run()` — never during reinitialize.
        raise RuntimeError("boom mid-plan")
        yield  # pragma: no cover - unreachable; present only to make this a generator

    runner = BlueskyPlanRunner(
        devices=mock_devices,
        plans={"boom": PlanSpec(name="boom", plan=_boom_plan, schema=_NoParams)},
    )

    # Resolution succeeds — the failure is deferred to the RunEngine thread.
    assert runner.reinitialize({"plan_name": "boom", "plan_args": {}}) is True
    assert runner.current_state == "armed"

    runner.start_run_thread()
    _wait_until_idle(runner)

    assert runner.current_state == "error"
    assert runner.error_message is not None
    assert runner.estimate_current_completion() != 1.0  # a failed run never reports 100%


# =========================================================================
# Contract failures: reinitialize() returns False, never raises
# =========================================================================


def test_reinitialize_returns_false_for_unknown_plan_name(mock_devices: dict) -> None:
    runner = BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS)
    ok = runner.reinitialize({"plan_name": "does_not_exist", "plan_args": {}})

    assert ok is False
    assert runner.error_message is not None
    assert runner.current_state == "error"


def test_reinitialize_returns_false_for_an_unknown_device_name(mock_devices: dict) -> None:
    runner = BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS)
    ok = runner.reinitialize(
        {"plan_name": "buffer_probe", "plan_args": {"detectors": ["no_such_detector"], "num": 1}}
    )

    assert ok is False
    assert runner.error_message is not None


# =========================================================================
# Full lifecycle: do_promote + the real GET /runs/{id}/data route
# =========================================================================


def test_promoted_run_read_run_data_returns_the_buffered_rows(
    mock_devices: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the real bridge route, not a patched buffer."""
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "s3cr3t")
    set_runner_factory(lambda: BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS))

    client = TestClient(app)
    create_resp = client.post(
        "/runs",
        json={
            "plan_name": "buffer_probe",
            "plan_args": {"detectors": ["det1"], "num": 3},
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    run_id = create_resp.json()["id"]

    promote_resp = client.post(f"/runs/{run_id}/promote", headers={"X-Promote-Token": "s3cr3t"})
    assert promote_resp.status_code == 200, promote_resp.text

    deadline = time.monotonic() + 15.0
    while client.get(f"/runs/{run_id}").json()["status"] == "running":
        if time.monotonic() > deadline:
            raise AssertionError("promoted run did not complete within the timeout")
        time.sleep(0.05)

    status_body = client.get(f"/runs/{run_id}").json()
    assert status_body["status"] == "completed"

    data_resp = client.get(f"/runs/{run_id}/data")
    assert data_resp.status_code == 200, data_resp.text
    data_body = data_resp.json()
    assert data_body["run_uid"] == status_body["run_uid"]
    assert data_body["row_count"] == 3
    assert len(data_body["rows"]) == 3
    assert "partial" not in data_body


def test_do_promote_stamps_osprey_run_id_onto_the_start_doc(mock_devices: dict) -> None:
    """`do_promote` (runs.py) sets `runner.osprey_run_id = run.id`, and `_run`
    (plan_runner_bluesky.py) must thread it onto the RunEngine start doc as
    metadata — not nested under an `md` key — so a Tiled-persisted run can be
    found again by `run.id` after the in-memory registry (and `run_uid` with
    it) is gone.
    """
    docs: list[tuple[str, dict]] = []

    def runner_factory() -> BlueskyPlanRunner:
        runner = BlueskyPlanRunner(devices=mock_devices, plans=_TEST_PLANS)
        runner.RE.subscribe(lambda name, doc: docs.append((name, dict(doc))))
        return runner

    run = registry.add(
        request={"plan_name": "buffer_probe", "plan_args": {"detectors": ["det1"], "num": 2}}
    )

    do_promote(run, runner_factory)
    _wait_until_idle(run.runner)

    assert run.runner.osprey_run_id == run.id  # type: ignore[union-attr]

    start_docs = [doc for name, doc in docs if name == "start"]
    assert len(start_docs) == 1
    assert start_docs[0]["osprey_run_id"] == run.id
    assert "md" not in start_docs[0]
