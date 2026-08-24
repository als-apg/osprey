"""Coverage for `DELETE /plans/session/{name}`.

The counterpart of `POST /plans/session`. Authoring a session-tier plan was
always a one-way door: short of restarting the bridge (session state is
ephemeral by design — see `session_dir.py`) there was no way to un-author one,
so a plan written once stayed in `GET /plans` for the life of the container.

Mirrors `test_plan_source_endpoint.py`'s isolation fixture and session-plan
helpers — every plan file here is pure pydantic/stdlib (no bluesky import).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app
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


def _session_plan_source(name: str) -> str:
    return (
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A session-tier test plan.",\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        f'    return {{"plan": {name!r}}}\n'
    )


def _author_and_validate(tmp_path: Path, name: str) -> Path:
    """Put a session plan on disk AND make it discoverable.

    A session file with no passing validation record is invisible to
    `get_facility_plans()` by design (the load gate), so a test that wants to
    prove the DELETE actually retires a *registered* plan has to record the
    pass — otherwise it would be asserting against a plan the registry never
    listed in the first place, and would keep passing if the route did
    nothing at all.
    """
    session_dir = tmp_path / "plans_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.py"
    source = _session_plan_source(name)
    path.write_text(source)
    validation_records.record(hash_plan_body(source))
    plan_loader.reset_facility_plans()
    return path


def _registered_names(client: TestClient) -> set[str]:
    response = client.get("/plans")
    assert response.status_code == 200, response.text
    return {spec["name"] for spec in response.json()}


def test_delete_retires_the_plan_from_the_registry_and_disk(
    tmp_path: Path, client: TestClient
) -> None:
    """The whole point: after a DELETE the name is gone from `GET /plans`.

    Asserted through the route rather than the filesystem because that is the
    surface every other consumer resolves a plan name through — the draft's
    plan-name check, the load gate, the enqueue gate. `plan_loader` re-scans
    the directory layers on each call, so the removal has to take effect with
    no restart and no cache reset.
    """
    path = _author_and_validate(tmp_path, "retired_plan")
    assert "retired_plan" in _registered_names(client)

    response = client.delete("/plans/session/retired_plan")

    assert response.status_code == 200, response.text
    assert response.json() == {"name": "retired_plan", "deleted": True}
    assert not path.exists()
    assert "retired_plan" not in _registered_names(client)


def test_delete_of_an_absent_name_reports_deleted_false(client: TestClient) -> None:
    """Idempotent, like `DELETE /draft` — "already absent" is success.

    A cleanup loop (the scan-stack E2E fixture is the first caller) races
    against nothing in particular and must not have to distinguish "I removed
    it" from "it was never there"; both leave the caller in the state it
    asked for.
    """
    response = client.delete("/plans/session/never_written")

    assert response.status_code == 200, response.text
    assert response.json() == {"name": "never_written", "deleted": False}


def test_delete_removes_only_the_named_plan(tmp_path: Path, client: TestClient) -> None:
    keep = _author_and_validate(tmp_path, "keep_me")
    _author_and_validate(tmp_path, "drop_me")
    assert {"keep_me", "drop_me"} <= _registered_names(client)

    assert client.delete("/plans/session/drop_me").status_code == 200

    assert keep.exists()
    registered = _registered_names(client)
    assert "keep_me" in registered
    assert "drop_me" not in registered


@pytest.mark.parametrize("name", ["_leading_underscore", "not-an-identifier", "a" * 101])
def test_delete_refuses_a_name_that_is_not_a_plan_name(name: str, client: TestClient) -> None:
    """Same gate as the write route, for the same reason.

    The name is used verbatim as a file stem, so anything that is not a plain
    identifier fails closed in `_sanitize_plan_name` rather than resolving to
    a path. Asserted as a 400 specifically: a 404 here would mean the request
    never reached the handler, which is a different (and, for a name shaped
    like this one, accidental) protection.
    """
    response = client.delete(f"/plans/session/{name}")

    assert response.status_code == 400, response.text
    assert "invalid plan name" in response.json()["detail"]


@pytest.mark.parametrize("name", ["../escape", "..%2Fescape", "sub/dir"])
def test_delete_traversal_shapes_never_reach_the_handler(
    tmp_path: Path, name: str, client: TestClient
) -> None:
    """A path-shaped name is refused by routing, before any name check.

    DELETE is the one verb where a traversal would be destructive rather than
    merely wrong, so this pins the outer of the two defences: a separator in
    the name means the URL does not match the single-segment route at all.
    The sentinel proves it — a file one level up from the session directory,
    exactly where `../escape` would land, is still there afterwards.
    """
    outside = tmp_path / "escape.py"
    outside.write_text("# must survive\n")

    response = client.delete(f"/plans/session/{name}")

    assert response.status_code == 404, response.text
    assert outside.exists()


def test_delete_cannot_reach_a_directory_layer_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A `facility`-tier file is not this route's to remove.

    The route resolves names inside `resolve_session_plan_dir()` and nowhere
    else, so a name that belongs to an operator-supplied layer simply has no
    session file behind it. The operator's file must still be on disk
    afterwards — the trust tiers are read-only from the bridge's point of
    view, and a route that could delete one would be a hole in that.
    """
    facility_dir = tmp_path / "facility_plans"
    facility_dir.mkdir(parents=True, exist_ok=True)
    facility_file = facility_dir / "operator_plan.py"
    facility_file.write_text(_session_plan_source("operator_plan"))
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(facility_dir))
    plan_loader.reset_facility_plans()
    assert "operator_plan" in _registered_names(client)

    response = client.delete("/plans/session/operator_plan")

    assert response.status_code == 200, response.text
    assert response.json() == {"name": "operator_plan", "deleted": False}
    assert facility_file.exists()
    assert "operator_plan" in _registered_names(client)
