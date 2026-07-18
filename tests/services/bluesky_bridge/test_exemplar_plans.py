"""Coverage for the shipped accelerator plans (task 1.5):
``plans_core/orm.py`` and ``plans_core/grid_scan.py``.

Runs ONLY in a bluesky-capable environment — `bluesky`/`ophyd-async` are
never installed in the main worktree venv, so every test here is skipped via
`pytest.importorskip` rather than failing, keeping `ci_check` green with no
bluesky installed at all. To actually run this file:

    uv venv /tmp/bluesky-exemplar-scratch
    /tmp/bluesky-exemplar-scratch/bin/pip install -e '.[bluesky-bridge]' --python 3.11
    /tmp/bluesky-exemplar-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_exemplar_plans.py -q

Two things are proven for each plan: (1) it registers through the real
layered-directory loader (`plan_loader.get_facility_plans`) with valid
metadata and `provenance == "shipped"` — i.e. the file satisfies the catalog
contract, not a hand-built `PlanSpec`; and (2) its `build_plan` drives a real
bluesky `RunEngine` (via `BlueskyPlanRunner`, mirroring
`test_runengine_integration.py`'s harness) to completion against mock
devices, emitting documents. `orm`'s own restore-in-`finally` abort-safety
(the FAILURE path) is covered separately by `test_builtin_plans.py`, which
drives its generator directly rather than through this registration path.
"""

from __future__ import annotations

import asyncio
import time

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.services.bluesky_bridge import live_rows, plan_loader  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import build_devices  # noqa: E402
from osprey.services.bluesky_bridge.plan_runner_bluesky import BlueskyPlanRunner  # noqa: E402


def _wait_until_idle(runner: BlueskyPlanRunner, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.is_run_active():
        if time.monotonic() > deadline:
            raise AssertionError("scan did not finish within the timeout")
        time.sleep(0.05)


@pytest.fixture(autouse=True)
def _isolated_state():
    live_rows._clear()
    plan_loader.reset_facility_plans()
    yield
    live_rows._clear()
    plan_loader.reset_facility_plans()


# =========================================================================
# Registration through the real layered-directory loader
# =========================================================================


def test_orm_and_grid_scan_register_as_shipped_with_valid_metadata() -> None:
    facility = plan_loader.get_facility_plans()

    assert "orm" in facility.plans
    assert "grid_scan" in facility.plans

    orm_spec = facility.plans["orm"]
    gs_spec = facility.plans["grid_scan"]

    assert orm_spec.provenance == "shipped"
    assert gs_spec.provenance == "shipped"

    assert orm_spec.metadata is not None
    assert gs_spec.metadata is not None
    assert orm_spec.metadata.writes is True
    assert gs_spec.metadata.writes is True

    assert orm_spec.name == "orm"
    assert gs_spec.name == "grid_scan"


# =========================================================================
# RunEngine dry-run against mock devices: a real run, not a no-op
# =========================================================================


@pytest.fixture
def orm_devices() -> dict:
    return asyncio.run(
        build_devices(
            motor_names=["hcm1", "hcm2"],
            detector_names=["bpm1", "bpm2"],
        )
    )


def test_orm_plan_runs_to_completion_and_buffers_rows(orm_devices: dict) -> None:
    facility = plan_loader.get_facility_plans()
    runner = BlueskyPlanRunner(devices=orm_devices, plans=facility.plans)
    exec_config = {
        "plan_name": "orm",
        "plan_args": {
            "correctors": ["hcm1", "hcm2"],
            "detectors": ["bpm1", "bpm2"],
            "span_a": 2.0,
            "num": 3,
        },
    }

    assert runner.reinitialize(exec_config) is True
    assert runner.current_state == "armed"

    runner.start_run_thread()
    _wait_until_idle(runner)

    assert runner.error_message is None
    assert runner.current_state == "completed"
    assert runner.last_run_uid is not None

    buf = live_rows.get(runner.last_run_uid)
    assert buf is not None
    # 2 correctors x 3 points each = one event (row) per (corrector, current).
    assert buf["total_seen"] == 6
    assert len(buf["rows"]) == 6
    for row in buf["rows"]:
        assert all(value is not None for value in row)

    # Each corrector restored to 0 A after its own sweep.
    assert asyncio.run(orm_devices["hcm1"].readback.get_value()) == 0.0
    assert asyncio.run(orm_devices["hcm2"].readback.get_value()) == 0.0


@pytest.fixture
def gs_devices() -> dict:
    return asyncio.run(
        build_devices(
            motor_names=["motor1", "motor2"],
            detector_names=["det1"],
        )
    )


def test_grid_scan_plan_runs_to_completion_and_buffers_rows(gs_devices: dict) -> None:
    facility = plan_loader.get_facility_plans()
    runner = BlueskyPlanRunner(devices=gs_devices, plans=facility.plans)
    exec_config = {
        "plan_name": "grid_scan",
        "plan_args": {
            "detectors": ["det1"],
            "axes": [
                {"setpoint": "motor1", "start": 0.0, "stop": 1.0, "num_points": 2},
                {"setpoint": "motor2", "start": 0.0, "stop": 1.0, "num_points": 3},
            ],
        },
    }

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
    # 2 x 3 grid = 6 total points.
    assert buf["total_seen"] == 6
    assert len(buf["rows"]) == 6
    assert any("det1" in col for col in buf["columns"])
