"""Coverage for the `session`-tier directory layer and its LOAD-TIME gate
(task 2.4): `plan_loader.py`'s CF-1 core enforcement point.

A `session`-tier plan file (written by task 2.3's `POST /plans/session`,
never itself exec'd by that route) must never be `exec_module`'d, registered,
or discoverable via `get_facility_plans()` unless its *current* on-disk
content hash has a passing record in `validation_record.validation_records`.
This is proven directly — not just "the plan is absent from `facility.plans`"
but "the file's own top-level code never ran at all" — via a sentinel file
write a gated body would perform at import time.

All plan files here are pure pydantic/stdlib (no bluesky import), matching
`test_plan_loader_layered.py`'s bluesky-less lane; the one exception (shipped
exemplar regression) is guarded by `pytest.importorskip`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"


@pytest.fixture(autouse=True)
def _isolated_plan_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every test gets its own session-plan directory, a clean loader cache,
    and a clean validation-record store (the module-level singleton is shared
    process-wide, so it must be snapshotted and restored like any other
    global test double)."""
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()

    with validation_records.lock:
        saved_hashes = set(validation_records._passing_hashes)
        validation_records._passing_hashes.clear()

    yield

    plan_loader.reset_facility_plans()
    with validation_records.lock:
        validation_records._passing_hashes.clear()
        validation_records._passing_hashes.update(saved_hashes)


def _sentinel_plan_source(name: str, sentinel_path: Path) -> str:
    """A session-tier plan file whose top-level code writes ``sentinel_path``
    on import, so a test can assert the file was (or was never) `exec_module`'d
    without relying solely on absence from the resolved plan set."""
    return (
        "from pathlib import Path\n\n\n"
        f"Path({str(sentinel_path)!r}).write_text('executed')\n\n\n"
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


def _write_session_plan(tmp_path: Path, filename: str, source: str) -> Path:
    session_dir = tmp_path / "plans_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / filename
    path.write_text(source)
    return path


def test_unvalidated_session_file_is_never_exec_moduled_or_discovered(
    tmp_path: Path,
) -> None:
    sentinel_path = tmp_path / "sentinel_unvalidated.txt"
    source = _sentinel_plan_source("unvalidated_plan", sentinel_path)
    _write_session_plan(tmp_path, "unvalidated_plan.py", source)
    # Deliberately NOT recording a passing validation for this content hash.

    facility = plan_loader.get_facility_plans()

    assert not sentinel_path.exists(), "the gated file's top-level code ran"
    assert "unvalidated_plan" not in facility.plans


def test_same_file_once_hash_is_recorded_loads_and_registers_as_session(
    tmp_path: Path,
) -> None:
    sentinel_path = tmp_path / "sentinel_validated.txt"
    source = _sentinel_plan_source("validated_plan", sentinel_path)
    _write_session_plan(tmp_path, "validated_plan.py", source)
    validation_records.record(hash_plan_body(source))

    facility = plan_loader.get_facility_plans()

    assert sentinel_path.exists(), "a validated file's top-level code never ran"
    assert "validated_plan" in facility.plans
    assert facility.plans["validated_plan"].provenance == "session"


def test_session_layer_rescans_on_write_without_reset(tmp_path: Path) -> None:
    """A session plan written and validated *after* the first
    `get_facility_plans()` call must appear on the next call with no
    `reset_facility_plans()` in between — the live-authoring contract."""
    first = plan_loader.get_facility_plans()
    assert "new_session_plan" not in first.plans

    source = _sentinel_plan_source("new_session_plan", tmp_path / "unused_sentinel.txt")
    _write_session_plan(tmp_path, "new_session_plan.py", source)
    validation_records.record(hash_plan_body(source))

    second = plan_loader.get_facility_plans()
    assert "new_session_plan" in second.plans
    assert second.plans["new_session_plan"].provenance == "session"


def test_startup_layers_stay_cached_across_session_rescans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expensive startup layers (shipped/preset/facility, plus legacy
    device construction) are built once, not re-scanned on every
    `get_facility_plans()` call — only the session layer is live."""
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    (shipped_dir / "startup_plan.py").write_text(
        "PLAN_METADATA = {\n"
        '    "name": "startup_plan",\n'
        '    "description": "A startup-tier plan.",\n'
        '    "category": "accelerator",\n'
        '    "required_devices": [],\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        '    return {"plan": "startup_plan"}\n'
    )
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    first = plan_loader.get_facility_plans()
    assert "startup_plan" in first.plans
    startup_layers_after_first_call = plan_loader._startup_layers
    assert startup_layers_after_first_call is not None

    # Write and validate a session plan, forcing a fresh session-layer scan.
    source = _sentinel_plan_source("another_session_plan", tmp_path / "unused.txt")
    _write_session_plan(tmp_path, "another_session_plan.py", source)
    validation_records.record(hash_plan_body(source))

    second = plan_loader.get_facility_plans()
    assert "startup_plan" in second.plans
    assert "another_session_plan" in second.plans
    # The startup registry object itself was never rebuilt.
    assert plan_loader._startup_layers is startup_layers_after_first_call


def test_malformed_session_file_is_quarantined_and_siblings_still_register(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One malformed session file must never break discovery of its siblings,
    regardless of *which* failure path it hits: a validated-but-structurally-
    broken file (missing `build_plan`) still exec's (it passed the gate) but
    is quarantined by the existing `except (Exception, SystemExit)` path; an
    unvalidated file never even reaches that far — it is skipped by the gate
    itself (see `test_unvalidated_session_file_is_never_exec_moduled_or_discovered`).
    Both must leave a well-formed, validated sibling plan registered."""
    good_source = _sentinel_plan_source("good_session_plan", tmp_path / "good_sentinel.txt")
    _write_session_plan(tmp_path, "good_session_plan.py", good_source)
    validation_records.record(hash_plan_body(good_source))

    # Syntactically valid, PLAN_METADATA present, but no `build_plan` — and
    # its hash IS recorded, so this exercises the post-gate quarantine path
    # (not the gate itself).
    missing_build_plan_source = (
        "PLAN_METADATA = {\n"
        '    "name": "missing_build_plan",\n'
        '    "description": "d",\n'
        '    "category": "accelerator",\n'
        '    "required_devices": [],\n'
        '    "writes": False,\n'
        "}\n"
    )
    _write_session_plan(tmp_path, "missing_build_plan.py", missing_build_plan_source)
    validation_records.record(hash_plan_body(missing_build_plan_source))

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        facility = plan_loader.get_facility_plans()

    assert "good_session_plan" in facility.plans
    assert "missing_build_plan" not in facility.plans
    assert any(r.levelno == logging.WARNING and "quarantining" in r.message for r in caplog.records)


def test_shipped_plans_are_not_gated() -> None:
    """Regression: the shipped plans (`orm`/`grid_scan`, in `plans_core/`)
    carry no validation record and are `provenance="shipped"`, not
    `session`/`unreviewed` — the session gate must never touch them, so they
    load unconditionally."""
    pytest.importorskip("bluesky")

    facility = plan_loader.get_facility_plans()
    assert "orm" in facility.plans
    assert "grid_scan" in facility.plans
    assert facility.plans["orm"].provenance == "shipped"
    assert facility.plans["grid_scan"].provenance == "shipped"
