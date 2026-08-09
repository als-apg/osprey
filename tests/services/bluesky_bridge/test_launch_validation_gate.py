"""Coverage for the session-plan validation gate.

`validation._validate_launchable_request` 409s a request for a
session/unreviewed plan whose CURRENT on-disk content hash has no passing
record in `validation_record.validation_records` — defense-in-depth alongside
the session-layer LOAD gate (`plan_loader.py`), re-hashing fresh at request
time rather than trusting an earlier snapshot.

This is the gate the enqueue path runs (`queue.py`'s `POST /queue/items`
calls it in a thread before anything reaches the manager) -- and, since it runs
ahead of `session_upload`'s own admissibility check, it is the refusal an
edited-after-pass session plan actually receives. So its body must be the queue
surface's contract shape, not a bare string; that is asserted below alongside
each rejection reason.

Every plan file here is pure pydantic/stdlib (no bluesky import), matching
`test_session_load_gate.py`'s bluesky-less lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.session_upload import REASON_UNVALIDATED
from osprey.services.bluesky_bridge.validation import _validate_launchable_request
from osprey.services.bluesky_bridge.validation_record import validation_records

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fresh session-plan directory, loader cache, and validation-record
    store for every test — each is a process-wide singleton in the real
    bridge, same as `test_session_load_gate.py`.
    """
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


# =========================================================================
# The gate itself
# =========================================================================


class _Snapshot:
    """A request carrying `plan_name` as an attribute — the shape the enqueue
    path passes (`draft.LaunchSnapshot`), alongside the plain-dict shape."""

    def __init__(self, plan_name: str) -> None:
        self.plan_name = plan_name
        self.plan_args: dict = {}


def _request(plan_name: str) -> dict:
    return {"plan_name": plan_name, "plan_args": {}}


def test_gate_blocks_a_session_plan_with_no_passing_record(tmp_path: Path) -> None:
    source = _session_plan_source("unvalidated_plan")
    _write_session_plan(tmp_path, "unvalidated_plan", source)
    # Deliberately not recording a passing validation for this content hash.

    with pytest.raises(HTTPException) as excinfo:
        _validate_launchable_request(_request("unvalidated_plan"))
    assert excinfo.value.status_code == 409
    detail = excinfo.value.detail
    assert detail["code"] == REASON_UNVALIDATED
    assert detail["plan"] == "unvalidated_plan"
    assert "no passing validation record" in detail["detail"]


def test_the_refusal_body_matches_the_session_upload_gate_it_pre_empts(tmp_path: Path) -> None:
    """This gate fires BEFORE `session_upload.check_session_plan_ready` on the
    enqueue path, so for an edited-after-pass plan it is the only refusal a
    caller sees. If its body drifted from that gate's, `session_plan_unvalidated`
    would be unreachable at enqueue and every consumer branching on
    `detail.code` would break on exactly this case."""
    from osprey.services.bluesky_bridge.session_upload import SessionPlanNotReadyError

    _write_session_plan(tmp_path, "shape_probe", _session_plan_source("shape_probe"))
    equivalent = SessionPlanNotReadyError("some sentence", plan="shape_probe")
    expected_keys = {*equivalent.to_dict(), "code"}

    with pytest.raises(HTTPException) as excinfo:
        _validate_launchable_request(_request("shape_probe"))
    assert set(excinfo.value.detail) == expected_keys
    # `code` and `reason` carry the same value, as `_session_refusal` does.
    assert excinfo.value.detail["code"] == excinfo.value.detail["reason"]


def test_gate_reads_plan_name_off_an_attribute_shaped_request(tmp_path: Path) -> None:
    """The enqueue path hands the gate a `LaunchSnapshot`, not a dict."""
    source = _session_plan_source("snapshot_plan")
    _write_session_plan(tmp_path, "snapshot_plan", source)

    with pytest.raises(HTTPException) as excinfo:
        _validate_launchable_request(_Snapshot("snapshot_plan"))
    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["plan"] == "snapshot_plan"


def test_gate_allows_a_session_plan_once_validated(tmp_path: Path) -> None:
    source = _session_plan_source("validated_plan")
    _write_session_plan(tmp_path, "validated_plan", source)
    validation_records.record(hash_plan_body(source))

    assert _validate_launchable_request(_request("validated_plan")) is None


def test_gate_reblocks_after_the_session_file_is_edited(tmp_path: Path) -> None:
    """A post-validation edit changes the content hash: the fresh re-hash at
    request time catches it even though the file still exists under the same
    name.
    """
    original = _session_plan_source("edited_plan")
    path = _write_session_plan(tmp_path, "edited_plan", original)
    validation_records.record(hash_plan_body(original))

    # Passes right after validation.
    assert _validate_launchable_request(_request("edited_plan")) is None

    # Edit the file (still a well-formed plan, just different content) without
    # re-validating it.
    edited = original.replace(
        '"description": "A session-tier test plan."', '"description": "edited"'
    )
    assert edited != original
    path.write_text(edited)

    with pytest.raises(HTTPException) as excinfo:
        _validate_launchable_request(_request("edited_plan"))
    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["code"] == REASON_UNVALIDATED


def test_gate_ignores_a_non_session_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shipped-tier plan (no validation record, never gated) is unaffected."""
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    (shipped_dir / "startup_plan.py").write_text(_session_plan_source("startup_plan"))
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    facility = plan_loader.get_facility_plans()
    assert facility.plans["startup_plan"].provenance == "shipped"

    assert _validate_launchable_request(_request("startup_plan")) is None


def test_gate_ignores_a_plan_name_with_no_session_file_and_no_registration() -> None:
    """A genuinely unknown plan name is left to the manager's own "not in the
    allowed namespace" rejection — the gate never 409s for it."""
    assert _validate_launchable_request(_request("nonexistent_plan")) is None


def test_gate_ignores_a_request_with_no_plan_name() -> None:
    assert _validate_launchable_request({}) is None
