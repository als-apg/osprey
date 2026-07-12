"""RunEngine integration test for the `orm` built-in plan (task 3.8).

Runs ONLY in a bluesky-capable environment — `bluesky`/`ophyd-async` are
never installed in the main worktree venv, so every test here is skipped via
`pytest.importorskip` rather than failing, keeping `ci_check` green with no
bluesky installed at all. To actually run this file:

    uv venv /tmp/bluesky-orm-scratch
    /tmp/bluesky-orm-scratch/bin/pip install -e '.[bluesky-bridge]' --python 3.11
    /tmp/bluesky-orm-scratch/bin/python -m pytest \
        tests/services/bluesky_bridge/test_orm_plan_integration.py -q

Mirrors `test_runengine_integration.py`'s idiom: drives a real `BlueskyScanner`
(a real bluesky `RunEngine` in a daemon thread) through the real `orm` plan
against mock ophyd-async devices (`devices/mock.py`) — no EPICS, no container.
Proves the generator built in `plans.py` actually runs to completion and the
live-row buffer ends up with one row per (corrector, current) pair, every BPM
column present, and no hang.
"""

from __future__ import annotations

import asyncio
import time

import pytest

bluesky = pytest.importorskip("bluesky")
ophyd_async = pytest.importorskip("ophyd_async")

from osprey.services.bluesky_bridge import live_rows  # noqa: E402
from osprey.services.bluesky_bridge.devices.mock import build_devices  # noqa: E402
from osprey.services.bluesky_bridge.plans import BUILTIN_PLANS  # noqa: E402
from osprey.services.bluesky_bridge.scanner_bluesky import BlueskyScanner  # noqa: E402


def _wait_until_idle(scanner: BlueskyScanner, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while scanner.is_scanning_active():
        if time.monotonic() > deadline:
            raise AssertionError("scan did not finish within the timeout")
        time.sleep(0.05)


@pytest.fixture(autouse=True)
def _isolated_state():
    live_rows._clear()
    yield
    live_rows._clear()


@pytest.fixture
def orm_devices() -> dict:
    """Two mock correctors + three mock BPM detectors, fresh per test.

    Unlike `test_runengine_integration.py`'s module-scoped `mock_devices`,
    this is function-scoped: the `orm` plan drives each corrector to a
    nonzero current and back, so a stale motor position from a previous test
    would silently pass a bug (starting-from-zero is part of what "restore to
    0 after its sweep" proves).
    """
    return asyncio.run(
        build_devices(
            motor_names=["hcm1", "hcm2"],
            detector_names=["bpm1", "bpm2", "bpm3"],
        )
    )


def test_orm_plan_runs_to_completion_and_buffers_one_row_per_point(
    orm_devices: dict,
) -> None:
    scanner = BlueskyScanner(devices=orm_devices, plans=BUILTIN_PLANS)
    exec_config = {
        "plan_name": "orm",
        "plan_args": {
            "correctors": ["hcm1", "hcm2"],
            "detectors": ["bpm1", "bpm2", "bpm3"],
            "span_a": 2.0,
            "num": 5,
        },
    }

    assert scanner.reinitialize(exec_config) is True
    assert scanner.current_state == "armed"

    scanner.start_scan_thread()
    _wait_until_idle(scanner)

    assert scanner.error_message is None
    assert scanner.current_state == "completed"
    assert scanner.last_run_uid is not None
    assert scanner.estimate_current_completion() == 1.0  # non-adaptive: binary 0/1

    buf = live_rows.get(scanner.last_run_uid)
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

    # No row has a missing BPM reading — every point read all three detectors.
    for row in buf["rows"]:
        assert all(value is not None for value in row)


def test_orm_plan_restores_each_corrector_to_zero_after_its_sweep(
    orm_devices: dict,
) -> None:
    scanner = BlueskyScanner(devices=orm_devices, plans=BUILTIN_PLANS)
    exec_config = {
        "plan_name": "orm",
        "plan_args": {
            "correctors": ["hcm1", "hcm2"],
            "detectors": ["bpm1"],
            "span_a": 3.0,
            "num": 3,
        },
    }

    assert scanner.reinitialize(exec_config) is True
    scanner.start_scan_thread()
    _wait_until_idle(scanner)

    assert scanner.error_message is None
    assert scanner.current_state == "completed"

    hcm1_readback = asyncio.run(orm_devices["hcm1"].readback.get_value())
    hcm2_readback = asyncio.run(orm_devices["hcm2"].readback.get_value())
    assert hcm1_readback == 0.0
    assert hcm2_readback == 0.0


def test_orm_plan_single_corrector_produces_exactly_num_rows(orm_devices: dict) -> None:
    scanner = BlueskyScanner(devices=orm_devices, plans=BUILTIN_PLANS)
    exec_config = {
        "plan_name": "orm",
        "plan_args": {
            "correctors": ["hcm1"],
            "detectors": ["bpm1", "bpm2"],
            "span_a": 1.0,
            "num": 4,
        },
    }

    assert scanner.reinitialize(exec_config) is True
    scanner.start_scan_thread()
    _wait_until_idle(scanner)

    assert scanner.error_message is None
    buf = live_rows.get(scanner.last_run_uid)
    assert buf is not None
    assert buf["total_seen"] == 4
    assert len(buf["rows"]) == 4
