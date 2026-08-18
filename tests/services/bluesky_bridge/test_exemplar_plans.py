"""Coverage for the shipped accelerator plans: ``plans_core/orm.py``,
``plans_core/grid_scan.py``, and ``plans_core/orbit_bump_sweep.py``.

Runs ONLY in a bluesky-capable environment — `bluesky`/`ophyd-async` are
never installed in the main worktree venv, so every test here is skipped via
`pytest.importorskip` rather than failing, keeping `ci_check` green with no
bluesky installed at all. To actually run this file:

    uv venv /tmp/bluesky-exemplar-scratch
    /tmp/bluesky-exemplar-scratch/bin/pip install -e . --python 3.11
    /tmp/bluesky-exemplar-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_exemplar_plans.py -q

Two things are proven for each plan: (1) it registers through the real
layered-directory loader (`plan_loader.get_facility_plans`) with valid
metadata and `provenance == "shipped"` — i.e. the file satisfies the catalog
contract, not a hand-built `PlanSpec`; and (2) it drives a real bluesky
`RunEngine` to completion against mock devices, emitting documents — through
`bluesky_plan_drive.run_plan`, which wraps each plan exactly as the queueserver
worker's namespace does, so what runs here is the production call shape. `orm`'s own restore-in-`finally` abort-safety
(the FAILURE path) is covered separately by `test_builtin_plans.py`, which
drives its generator directly rather than through this registration path.
"""

from __future__ import annotations

import asyncio

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.services.bluesky_bridge import live_rows, plan_loader  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import build_devices  # noqa: E402

from .bluesky_plan_drive import run_plan  # noqa: E402


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


def test_orbit_bump_sweep_registers_as_shipped_with_valid_metadata() -> None:
    facility = plan_loader.get_facility_plans()

    assert "orbit_bump_sweep" in facility.plans

    spec = facility.plans["orbit_bump_sweep"]

    assert spec.provenance == "shipped"
    assert spec.metadata is not None
    assert spec.metadata.writes is True
    assert spec.name == "orbit_bump_sweep"


# =========================================================================
# RunEngine dry-run against mock devices: a real run, not a no-op
# =========================================================================


@pytest.fixture
def orm_devices() -> dict:
    return asyncio.run(
        build_devices(
            settable_names=["hcm1", "hcm2"],
            readable_names=["bpm1", "bpm2"],
        )
    )


def test_orm_plan_runs_to_completion_and_buffers_rows(orm_devices: dict) -> None:
    facility = plan_loader.get_facility_plans()
    run_uid = run_plan(
        "orm",
        {
            "correctors": ["hcm1", "hcm2"],
            "readbacks": ["bpm1", "bpm2"],
            "span_a": 2.0,
            "num": 3,
        },
        devices=orm_devices,
        plans=facility.plans,
    )

    buf = live_rows.get(run_uid)
    assert buf is not None
    # 2 correctors x 3 points each = one event (row) per (corrector, current).
    assert buf["total_seen"] == 6
    assert len(buf["rows"]) == 6
    for row in buf["rows"]:
        assert all(value is not None for value in row)

    # Each corrector restored to its pre-plan working point after its own
    # sweep. These mock motors start at 0 A, so here that value is 0 A.
    assert asyncio.run(orm_devices["hcm1"].readback.get_value()) == 0.0
    assert asyncio.run(orm_devices["hcm2"].readback.get_value()) == 0.0


@pytest.fixture
def gs_devices() -> dict:
    return asyncio.run(
        build_devices(
            settable_names=["motor1", "motor2"],
            readable_names=["det1"],
        )
    )


def test_grid_scan_plan_runs_to_completion_and_buffers_rows(gs_devices: dict) -> None:
    facility = plan_loader.get_facility_plans()
    run_uid = run_plan(
        "grid_scan",
        {
            "readbacks": ["det1"],
            "axes": [
                {"setpoint": "motor1", "start": 0.0, "stop": 1.0, "num_points": 2},
                {"setpoint": "motor2", "start": 0.0, "stop": 1.0, "num_points": 3},
            ],
        },
        devices=gs_devices,
        plans=facility.plans,
    )

    buf = live_rows.get(run_uid)
    assert buf is not None
    # 2 x 3 grid = 6 total points.
    assert buf["total_seen"] == 6
    assert len(buf["rows"]) == 6
    assert any("det1" in col for col in buf["columns"])


@pytest.fixture
def bump_devices() -> dict:
    """Every device the bump touches, all as mock motors.

    The BPMs are motors, not detectors, deliberately: `MockDetector` counts up
    on every trigger, so a "BPM" built from one would read a different value on
    each of the baseline reads and the plan would (rightly) refuse the run as
    unverifiable at the noise floor. `MockMotor`'s readback is constant, which
    is the only mock orbit a physics-free dry run can honestly claim. It is
    also how `plan_validation._collect_device_names` buckets these same names
    for its own stage-3 dry run, so both doctrine sites drive the plan against
    the same device set.
    """
    return asyncio.run(
        build_devices(
            motor_names=["hcm1", "hcm2", "hcm3", "bpm1", "bpm2", "bpm3"],
            detector_names=[],
        )
    )


def test_orbit_bump_sweep_plan_runs_to_completion_and_buffers_rows(bump_devices: dict) -> None:
    """A zero-offset bump: the whole plan structure, no orbit response needed.

    Mock devices move no beam, so the probe fits an identically zero response
    and the run takes the demand gate's trivially-converged path — every
    step's solution is zero, `solve_offsets` is never called. The profile is
    still walked in full, which is what makes the row count below the real
    pinned layout and not a shortened one.
    """
    facility = plan_loader.get_facility_plans()
    run_uid = run_plan(
        "orbit_bump_sweep",
        {
            "correctors": ["hcm1", "hcm2", "hcm3"],
            "targets": [{"bpm": "bpm1", "value": 0.0}],
            "closure_bpms": ["bpm2", "bpm3"],
            "bpms": [],
            "num": 2,
            "baseline_reads": 3,
            "probe_amplitude": 0.1,
            "tolerance": 0.01,
            "max_trim_iterations": 1,
            "best_effort": True,
            "settle_s": 0.0,
        },
        devices=bump_devices,
        plans=facility.plans,
    )

    buf = live_rows.get(run_uid)
    assert buf is not None
    # The pinned row layout: `baseline_reads` baseline rows, then exactly one
    # row per amplitude step — `2 * num` of them for a monodirectional sweep
    # (up to the peak and back down to zero). The probe and convergence reads
    # are taken off the record and must contribute no rows at all.
    assert buf["total_seen"] == 3 + 2 * 2
    assert len(buf["rows"]) == 7
    assert any("bpm1" in col for col in buf["columns"])

    # Every corrector back on its pre-scan working point — 0 A for these mock
    # motors — including via the terminal scale-0 step, which commands the
    # recorded working points verbatim.
    for name in ("hcm1", "hcm2", "hcm3"):
        assert asyncio.run(bump_devices[name].readback.get_value()) == 0.0
