"""Coverage for `GET /plans/{name}/source` (task 2.6).

This route is the data source for the launch-approval hook's plan-source
excerpt — the human backstop for the plan validator's documented, accepted
obfuscation residual (see `plan_validation.py`'s module docstring): an
approver who can actually SEE a plan's source has a chance to refuse an
obfuscated body even where the earlier automated stages could not catch it.

Mirrors `test_promote_validation_gate.py`'s isolation fixture and session-plan
helpers — every plan file here is pure pydantic/stdlib (no bluesky import).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import (
    _SOURCE_TRUNCATE_CHARS,
    app,
)
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _session_plan_source(name: str, *, body: str = "") -> str:
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
        f"{body}"
    )


def _write_session_plan(tmp_path: Path, name: str, source: str) -> Path:
    session_dir = tmp_path / "plans_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.py"
    path.write_text(source)
    return path


def test_session_plan_with_no_passing_record_is_unvalidated(
    tmp_path: Path, client: TestClient
) -> None:
    source = _session_plan_source("unvalidated_plan")
    _write_session_plan(tmp_path, "unvalidated_plan", source)

    resp = client.get("/plans/unvalidated_plan/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"] == "session"
    assert body["validated"] is False
    assert body["truncated"] is False
    assert body["source"] == source


def test_session_plan_with_a_passing_record_is_validated(
    tmp_path: Path, client: TestClient
) -> None:
    source = _session_plan_source("validated_plan")
    _write_session_plan(tmp_path, "validated_plan", source)
    validation_records.record(hash_plan_body(source))

    resp = client.get("/plans/validated_plan/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"] == "session"
    assert body["validated"] is True


def test_session_plan_source_is_truncated_beyond_the_bound(
    tmp_path: Path, client: TestClient
) -> None:
    oversized_body = "\n# padding\n" * (_SOURCE_TRUNCATE_CHARS // 5)
    source = _session_plan_source("huge_plan", body=oversized_body)
    assert len(source) > _SOURCE_TRUNCATE_CHARS
    _write_session_plan(tmp_path, "huge_plan", source)

    resp = client.get("/plans/huge_plan/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["truncated"] is True
    assert len(body["source"]) == _SOURCE_TRUNCATE_CHARS
    assert body["source"] == source[:_SOURCE_TRUNCATE_CHARS]


def test_unknown_plan_name_is_404(client: TestClient) -> None:
    resp = client.get("/plans/does_not_exist_anywhere/source")
    assert resp.status_code == 404


def test_non_identifier_name_is_400(client: TestClient) -> None:
    """`name` is sanitized the SAME way `/plans/session` and `/plans/validate`
    already do (task 2.3's `_sanitize_plan_name`) before it ever touches a
    filesystem path — a file-reading route must not rely implicitly on
    FastAPI's no-slash-in-path-param behavior for traversal safety. A bare
    single-segment name that isn't a valid identifier reaches the handler
    (no embedded slash to trip routing) and must 400 from the sanitizer."""
    resp = client.get("/plans/..%2E%2E/source")
    assert resp.status_code == 400

    resp = client.get("/plans/not-a-valid-identifier/source")
    assert resp.status_code == 400


def test_embedded_slash_name_never_reaches_the_handler(client: TestClient) -> None:
    """An encoded-slash traversal attempt (`..%2F..%2Fetc%2Fpasswd`) doesn't
    even match the single-segment `{name}` path parameter — FastAPI's own
    routing 404s it before `_sanitize_plan_name` ever runs. Documented here
    so a routing change that started letting such paths through would be
    caught by a test, not just an implicit assumption."""
    resp = client.get("/plans/..%2F..%2Fetc%2Fpasswd/source")
    assert resp.status_code == 404

    resp = client.get("/plans/not a valid identifier/source")
    assert resp.status_code == 400


def test_shipped_plan_is_located_by_declared_name_not_file_stem(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shipped-tier file may declare a `PLAN_METADATA["name"]` different
    from its own filename (e.g. `plans_core/grid_scan.py` -> "grid_scan_nd")
    — the source route must resolve by the declared name, not the stem."""
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    (shipped_dir / "some_file_stem.py").write_text(
        _session_plan_source("declared_plan_name")
    )
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    resp = client.get("/plans/declared_plan_name/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"] == "shipped"
    # Shipped/preset/facility tiers carry no validation-record gate at all —
    # trusted by construction, not by a passing record.
    assert body["validated"] is True
    assert "declared_plan_name" in body["source"]


def test_re_authored_session_plan_re_reflects_unvalidated_after_edit(
    tmp_path: Path, client: TestClient
) -> None:
    """A previously-validated session file, edited afterward, is honestly
    reported as unvalidated again — mirrors the load/promote gates' own
    re-hash-on-every-check behavior (never a stale cached verdict)."""
    original = _session_plan_source("edited_plan")
    path = _write_session_plan(tmp_path, "edited_plan", original)
    validation_records.record(hash_plan_body(original))
    assert client.get("/plans/edited_plan/source").json()["validated"] is True

    edited = _session_plan_source("edited_plan", body="\n# an edit\n")
    path.write_text(edited)

    resp = client.get("/plans/edited_plan/source")
    body = resp.json()
    assert body["validated"] is False
    assert body["source"] == edited
