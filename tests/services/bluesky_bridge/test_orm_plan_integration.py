"""RunEngine integration test for the shipped `orm` plan.

The bluesky stack is a core dependency, so these run in the normal unit
lane; the `pytest.importorskip` guards below only skip them in a slimmed
install where `bluesky`/`ophyd-async` are absent.

Drives a real bluesky `RunEngine` through the real `orm` plan
(`plans_core/orm.py`, resolved through the layered directory loader —
`plan_loader.get_facility_plans`, the sole plan registry) against mock
ophyd-async devices (`devices/mock.py`) — no EPICS, no container, and via
`bluesky_plan_drive.run_plan`, which wraps the plan exactly as the queueserver
worker's namespace does. Proves the
generator built there actually runs to completion and the live-row buffer
ends up with one row per (corrector, current) pair, every BPM column
present, and no hang.
"""

from __future__ import annotations

import asyncio

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.services.bluesky_bridge import live_rows, plan_loader  # noqa: E402
from osprey.services.bluesky_bridge.devices._connect import connect_all  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import (  # noqa: E402
    MockReadable,
    MockSettable,
    build_devices,
)
from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix  # noqa: E402

from .bluesky_plan_drive import run_plan  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state():
    live_rows._clear()
    plan_loader.reset_facility_plans()
    yield
    live_rows._clear()
    plan_loader.reset_facility_plans()


@pytest.fixture
def orm_devices() -> dict:
    """Two mock correctors + three mock BPM readables, fresh per test.

    Unlike `test_runengine_integration.py`'s module-scoped `mock_devices`,
    this is function-scoped: the `orm` plan kicks each corrector away from
    where it found it and back, so a stale setpoint position carried over from
    a previous test would move the working point these tests assert against
    out from under them.
    """
    return asyncio.run(
        build_devices(
            settable_names=["hcm1", "hcm2"],
            readable_names=["bpm1", "bpm2", "bpm3"],
        )
    )


def test_orm_plan_runs_to_completion_and_buffers_one_row_per_point(
    orm_devices: dict,
) -> None:
    run_uid = run_plan(
        "orm",
        {
            "correctors": ["hcm1", "hcm2"],
            "readbacks": ["bpm1", "bpm2", "bpm3"],
            "span_a": 2.0,
            "num": 5,
        },
        devices=orm_devices,
        plans=plan_loader.get_facility_plans().plans,
    )

    buf = live_rows.get(run_uid)
    assert buf is not None
    assert buf["partial"] is False

    # 2 correctors x 5 points each = one event (one row) per (corrector, current).
    assert buf["total_seen"] == 10
    assert len(buf["rows"]) == 10

    # Every BPM's reading is a column, alongside the driven corrector's own
    # (readback == current, per FR10's echo).
    for bpm_name in ("bpm1", "bpm2", "bpm3"):
        assert any(bpm_name in col for col in buf["columns"])
    assert any("hcm1" in col or "hcm2" in col for col in buf["columns"])

    # No row has a missing BPM reading — every point read all three readbacks.
    for row in buf["rows"]:
        assert all(value is not None for value in row)


def test_orm_plan_restores_each_corrector_through_the_real_loader_path(
    orm_devices: dict,
) -> None:
    """Restore survives the round trip through the real plan loader and
    `run_plan` wrapper, not just a directly-built generator.

    These mock settables start at 0 A, so the working point each corrector is
    put back to is 0 A here; `test_builtin_plans.py` covers the nonzero case
    that distinguishes "put back where it was" from "driven to zero".
    """
    run_plan(
        "orm",
        {
            "correctors": ["hcm1", "hcm2"],
            "readbacks": ["bpm1"],
            "span_a": 3.0,
            "num": 3,
        },
        devices=orm_devices,
        plans=plan_loader.get_facility_plans().plans,
    )

    hcm1_readback = asyncio.run(orm_devices["hcm1"].readback.get_value())
    hcm2_readback = asyncio.run(orm_devices["hcm2"].readback.get_value())
    assert hcm1_readback == 0.0
    assert hcm2_readback == 0.0


def test_orm_plan_output_is_fittable_by_the_analysis_it_feeds() -> None:
    """The seam: rows the plan actually emits must be rows
    `orm_analysis.build_response_matrix` can actually fit.

    The two halves hold the same invariant from opposite sides — the plan
    sweeps each corrector symmetrically about its own working point, and the
    fit needs every idle sample to sit at the fit's x-mean — and nothing but
    a test spanning both notices when one side moves. Drive the real plan
    against correctors idling at nonzero working points and hand its own
    buffered rows straight to the fit: before the relative sweep, this raised
    `DegenerateFitError` on the plan's own output.

    The mock readable is a counter, not a machine, so the fitted *values* are
    meaningless here and go unasserted; that the fit accepts the rows at all
    is the whole claim.
    """
    correctors = {"hcm1": 2.5, "hcm2": -1.25}
    devices = asyncio.run(
        connect_all(
            {
                **{name: MockSettable(name, initial_value=v) for name, v in correctors.items()},
                "bpm1": MockReadable("bpm1"),
                "bpm2": MockReadable("bpm2"),
            }
        )
    )
    readbacks = ["bpm1", "bpm2"]
    run_uid = run_plan(
        "orm",
        {
            "correctors": list(correctors),
            "readbacks": readbacks,
            "span_a": 2.0,
            "num": 5,
        },
        devices=devices,
        plans=plan_loader.get_facility_plans().plans,
    )

    buf = live_rows.get(run_uid)
    assert buf is not None
    rows = [dict(zip(buf["columns"], row, strict=True)) for row in buf["rows"]]

    matrix = build_response_matrix(rows, list(correctors), readbacks)

    assert matrix.shape == (len(readbacks), len(correctors))


def test_orm_plan_single_corrector_produces_exactly_num_rows(orm_devices: dict) -> None:
    run_uid = run_plan(
        "orm",
        {
            "correctors": ["hcm1"],
            "readbacks": ["bpm1", "bpm2"],
            "span_a": 1.0,
            "num": 4,
        },
        devices=orm_devices,
        plans=plan_loader.get_facility_plans().plans,
    )

    buf = live_rows.get(run_uid)
    assert buf is not None
    assert buf["total_seen"] == 4
    assert len(buf["rows"]) == 4
