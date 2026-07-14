"""Coverage for the promote-time session-plan validation gate (task 2.5).

Two parts, per the task split:

- **Part A**: `app.py`'s `_epics_scanner_factory` now resolves plan names
  through the gated registry (``plans=None`` -> `BlueskyScanner.reinitialize`
  falls back to `_default_plan_registry()`, which merges built-ins with the
  re-scanned, re-gated `get_facility_plans().plans`) instead of a fixed
  `BUILTIN_PLANS` snapshot — so a validated session/facility plan is
  launchable on the connector-mediated path, and an unvalidated one is not.
- **Part B**: `app.py`'s `_promote_validation_gate`, dependency-injected into
  `runs.do_promote` as its `validator` keyword, 409s a promote attempt for a
  session/unreviewed plan whose CURRENT on-disk content hash has no passing
  record in `validation_record.validation_records` — defense-in-depth
  alongside task 2.4's session-layer LOAD gate, re-hashing fresh at promote
  time rather than trusting an earlier snapshot.

All Part B plan files here are pure pydantic/stdlib (no bluesky import),
matching `test_session_load_gate.py`'s bluesky-less lane; `FakeScanner` is
used throughout so these tests never need bluesky either — the gate runs
entirely before any scanner is built.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app, set_scanner_factory
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.runs import Run, do_promote, registry
from osprey.services.bluesky_bridge.scanner import FakeScanner
from osprey.services.bluesky_bridge.validation_record import validation_records

_SESSION_PLAN_DIR_ENV = "BLUESKY_SESSION_PLAN_DIR"
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"
_PROMOTE_TOKEN_ENV = "BLUESKY_PROMOTE_TOKEN"
_TOKEN = "s3cr3t"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fresh session-plan directory, loader cache, validation-record store,
    run registry, and scanner factory for every test — every one of these is
    a process-wide singleton in the real bridge, same as `test_session_load_gate.py`.
    """
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    monkeypatch.setenv(_SESSION_PLAN_DIR_ENV, str(tmp_path / "plans_session"))
    plan_loader.reset_facility_plans()

    with validation_records.lock:
        saved_hashes = set(validation_records._passing_hashes)
        validation_records._passing_hashes.clear()

    registry._runs.clear()
    set_scanner_factory(FakeScanner)

    yield

    plan_loader.reset_facility_plans()
    with validation_records.lock:
        validation_records._passing_hashes.clear()
        validation_records._passing_hashes.update(saved_hashes)
    registry._runs.clear()
    set_scanner_factory(FakeScanner)


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
# Part B: `_promote_validation_gate` unit coverage
# =========================================================================


def _run_for(plan_name: str) -> Run:
    return registry.add(request={"plan_name": plan_name, "plan_args": {}})


def test_gate_blocks_a_session_plan_with_no_passing_record(tmp_path: Path) -> None:
    source = _session_plan_source("unvalidated_plan")
    _write_session_plan(tmp_path, "unvalidated_plan", source)
    # Deliberately not recording a passing validation for this content hash.

    run = _run_for("unvalidated_plan")

    with pytest.raises(HTTPException) as excinfo:
        app_module._promote_validation_gate(run)
    assert excinfo.value.status_code == 409
    assert "unvalidated_plan" in excinfo.value.detail
    assert "no passing validation record" in excinfo.value.detail


def test_gate_allows_a_session_plan_once_validated(tmp_path: Path) -> None:
    source = _session_plan_source("validated_plan")
    _write_session_plan(tmp_path, "validated_plan", source)
    validation_records.record(hash_plan_body(source))

    run = _run_for("validated_plan")

    assert app_module._promote_validation_gate(run) is None


def test_gate_reblocks_after_the_session_file_is_edited(tmp_path: Path) -> None:
    """A post-validation edit changes the content hash: the fresh re-hash at
    promote catches it even though the file still exists under the same name.
    """
    original = _session_plan_source("edited_plan")
    path = _write_session_plan(tmp_path, "edited_plan", original)
    validation_records.record(hash_plan_body(original))

    # Passes right after validation.
    assert app_module._promote_validation_gate(_run_for("edited_plan")) is None

    # Edit the file (still a well-formed plan, just different content) without
    # re-validating it.
    edited = original.replace(
        '"description": "A session-tier test plan."', '"description": "edited"'
    )
    assert edited != original
    path.write_text(edited)

    with pytest.raises(HTTPException) as excinfo:
        app_module._promote_validation_gate(_run_for("edited_plan"))
    assert excinfo.value.status_code == 409


def test_gate_ignores_a_non_session_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shipped-tier plan (no validation record, never gated) is unaffected."""
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    (shipped_dir / "startup_plan.py").write_text(_session_plan_source("startup_plan"))
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    facility = plan_loader.get_facility_plans()
    assert facility.plans["startup_plan"].provenance == "shipped"

    run = _run_for("startup_plan")
    assert app_module._promote_validation_gate(run) is None


def test_gate_ignores_a_plan_name_with_no_session_file_and_no_registration() -> None:
    """A genuinely unknown plan name is left to `Scanner.reinitialize`'s own
    "unknown plan" handling — the gate never 409s for it."""
    run = _run_for("nonexistent_plan")
    assert app_module._promote_validation_gate(run) is None


def test_gate_ignores_a_run_with_no_plan_name() -> None:
    run = registry.add(request={})
    assert app_module._promote_validation_gate(run) is None


# =========================================================================
# Part B: end-to-end through `POST /runs/{id}/promote`
# =========================================================================


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv(_PROMOTE_TOKEN_ENV, _TOKEN)
    return TestClient(app)


def _create_run(client: TestClient, plan_name: str) -> str:
    resp = client.post("/runs", json={"plan_name": plan_name, "plan_args": {}})
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _promote(client: TestClient, run_id: str):
    return client.post(f"/runs/{run_id}/promote", headers={"X-Promote-Token": _TOKEN})


def test_promote_409s_an_unvalidated_session_plan(tmp_path: Path, client: TestClient) -> None:
    source = _session_plan_source("http_unvalidated")
    _write_session_plan(tmp_path, "http_unvalidated", source)

    run_id = _create_run(client, "http_unvalidated")
    resp = _promote(client, run_id)

    assert resp.status_code == 409
    assert "no passing validation record" in resp.json()["detail"]
    assert client.get(f"/runs/{run_id}").json()["status"] == "intent"


def test_promote_succeeds_once_the_session_plan_is_validated(
    tmp_path: Path, client: TestClient
) -> None:
    source = _session_plan_source("http_validated")
    _write_session_plan(tmp_path, "http_validated", source)
    validation_records.record(hash_plan_body(source))

    run_id = _create_run(client, "http_validated")
    resp = _promote(client, run_id)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "running"


def test_promote_reblocks_after_a_post_validation_edit(tmp_path: Path, client: TestClient) -> None:
    original = _session_plan_source("http_edited")
    path = _write_session_plan(tmp_path, "http_edited", original)
    validation_records.record(hash_plan_body(original))

    edited = original.replace(
        '"description": "A session-tier test plan."', '"description": "edited"'
    )
    path.write_text(edited)

    run_id = _create_run(client, "http_edited")
    resp = _promote(client, run_id)

    assert resp.status_code == 409
    assert "no passing validation record" in resp.json()["detail"]


def test_promote_of_a_shipped_plan_is_unaffected(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    (shipped_dir / "startup_plan.py").write_text(_session_plan_source("startup_plan"))
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    run_id = _create_run(client, "startup_plan")
    resp = _promote(client, run_id)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "running"


# =========================================================================
# `do_promote(..., validator=None)` — the default preserves every existing
# caller that never passes one (contract-test path unbroken).
# =========================================================================


def test_do_promote_without_a_validator_is_unaffected_by_the_gate(tmp_path: Path) -> None:
    """An unvalidated session plan is only refused when a validator is wired
    in — `runs.py` itself stays import-clean of `plan_loader`/bluesky, and
    every pre-2.5 caller that never passes `validator` sees no new behavior.
    """
    source = _session_plan_source("no_validator_plan")
    _write_session_plan(tmp_path, "no_validator_plan", source)

    run = _run_for("no_validator_plan")
    scanner = FakeScanner()

    result = do_promote(run, lambda: scanner)

    assert result is run
    assert run.promoted is True
    assert scanner.reinitialize_calls == 1


def test_do_promote_with_validator_none_explicit_matches_default() -> None:
    run = registry.add(request={})
    scanner = FakeScanner()

    result = do_promote(run, lambda: scanner, validator=None)

    assert result is run
    assert run.promoted is True


# =========================================================================
# Part A: `_epics_scanner_factory` resolves plans through the gated registry
# =========================================================================


@pytest.fixture
def _clean_epics_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "BLUESKY_EPICS_SUBSTRATE",
        "BLUESKY_DEMO_SCANNER",
        "BLUESKY_EPICS_MOTORS",
        "BLUESKY_EPICS_DETECTORS",
    ):
        monkeypatch.delenv(var, raising=False)
    app_module._connector = None
    yield
    app_module._connector = None


def test_epics_scanner_factory_resolves_a_validated_session_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _clean_epics_env: None
) -> None:
    """`_epics_scanner_factory` no longer pins `plans=BUILTIN_PLANS` — a
    validated session plan (re-gated by `_default_plan_registry()` on every
    `reinitialize()` call) is resolvable through the connector-mediated
    launch path, and an unvalidated one is not.
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    from osprey.connectors.factory import ConnectorFactory

    class _SpyConnector:
        async def disconnect(self) -> None:
            pass

    async def fake_create_control_system_connector(config):
        return _SpyConnector()

    monkeypatch.setattr(
        ConnectorFactory, "create_control_system_connector", fake_create_control_system_connector
    )
    monkeypatch.setenv("BLUESKY_EPICS_SUBSTRATE", "true")

    validated_source = _session_plan_source("gated_validated_plan")
    _write_session_plan(tmp_path, "gated_validated_plan", validated_source)
    validation_records.record(hash_plan_body(validated_source))

    unvalidated_source = _session_plan_source("gated_unvalidated_plan")
    _write_session_plan(tmp_path, "gated_unvalidated_plan", unvalidated_source)
    # Deliberately not recording a passing validation for this one.

    with TestClient(app):
        pass

    assert app_module._scanner_factory is not FakeScanner
    scanner = app_module._scanner_factory()
    # `plans=None` — resolution happens lazily via `_default_plan_registry()`,
    # not a fixed built-in snapshot, on every `reinitialize()` call.
    assert scanner._plans is None

    ok = scanner.reinitialize({"plan_name": "gated_validated_plan", "plan_args": {}})
    assert ok is True, scanner.error_message

    other_scanner = app_module._scanner_factory()
    not_ok = other_scanner.reinitialize({"plan_name": "gated_unvalidated_plan", "plan_args": {}})
    assert not_ok is False
    assert "unknown plan" in (other_scanner.error_message or "")
