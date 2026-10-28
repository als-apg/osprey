"""Unit tests for `contribution.py` (task 2.9): the thin glue that hands a
validated session-tier plan off to the existing `/osprey:contribute` skill for
a contribution PR/MR.

`prepare_contribution`/`stage_contribution` never open a PR themselves and never
touch `plan_loader.py`'s trust order — these tests cover only the glue's own
contract: refuse an unknown or unvalidated plan, and copy validated bytes
byte-for-byte into the target catalog directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.bluesky_bridge.contribution import (
    ContributionRequest,
    UnknownSessionPlanError,
    UnvalidatedPlanError,
    prepare_contribution,
    stage_contribution,
)
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records
from tests.services.bluesky_bridge._session_plans import write_session_plan

_PLAN_BODY = "PLAN_METADATA = {'name': 'orbit_bump'}\n"


def test_unknown_plan_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownSessionPlanError):
        prepare_contribution("nonexistent_plan", tmp_path / "catalog")


def test_unvalidated_plan_raises(tmp_path: Path) -> None:
    write_session_plan("orbit_bump", _PLAN_BODY)
    # No validation_records.record() call — hash was never recorded as passing.
    with pytest.raises(UnvalidatedPlanError):
        prepare_contribution("orbit_bump", tmp_path / "catalog")


def test_validated_plan_prepares_request(tmp_path: Path) -> None:
    source_path = write_session_plan("orbit_bump", _PLAN_BODY)
    validation_records.record(hash_plan_body(_PLAN_BODY))

    catalog_dir = tmp_path / "catalog"
    request = prepare_contribution("orbit_bump", catalog_dir)

    assert isinstance(request, ContributionRequest)
    assert request.name == "orbit_bump"
    assert request.body == _PLAN_BODY
    assert request.content_hash == hash_plan_body(_PLAN_BODY)
    assert request.source_path == source_path
    assert request.target_path == catalog_dir / "orbit_bump.py"
    assert "orbit_bump" in request.suggested_branch
    assert "orbit_bump" in request.suggested_pr_title
    assert request.content_hash in request.suggested_pr_body


def test_editing_after_validation_re_raises_unvalidated(tmp_path: Path) -> None:
    """Matches the 2.4/2.5 re-check contract: an edit after validation drops
    the hash's passing record until re-validated."""
    write_session_plan("orbit_bump", _PLAN_BODY)
    validation_records.record(hash_plan_body(_PLAN_BODY))

    # Edit the file after validation — its content hash changes.
    write_session_plan("orbit_bump", _PLAN_BODY + "# tweak\n")

    with pytest.raises(UnvalidatedPlanError):
        prepare_contribution("orbit_bump", tmp_path / "catalog")


def test_stage_contribution_copies_bytes_identically(tmp_path: Path) -> None:
    write_session_plan("orbit_bump", _PLAN_BODY)
    validation_records.record(hash_plan_body(_PLAN_BODY))

    catalog_dir = tmp_path / "catalog" / "nested"  # doesn't exist yet
    request = prepare_contribution("orbit_bump", catalog_dir)
    written_path = stage_contribution(request)

    assert written_path == catalog_dir / "orbit_bump.py"
    assert written_path.read_text() == _PLAN_BODY
    # The staged copy re-hashes identically to the session file it came from.
    assert hash_plan_body(written_path.read_text()) == request.content_hash
