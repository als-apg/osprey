"""Tests for the fail-OPEN startup assertion (task 3.1).

`app.py`'s `_lifespan` hook refuses to start *writable* against an unreadable
limits source: it raises IFF ALL of `control_system.writes_enabled` is true,
`control_system.limits_checking.enabled` is true, AND the limits database at
`control_system.limits_checking.database_path` is missing, unreadable, or
unparseable. Every other combination starts normally — most importantly,
writes disabled must start read-only REGARDLESS of limits readability, and
must never even probe the database. See `_assert_limits_readable_if_writable`'s
docstring in `app.py` for the full condition and rationale.

Exercised here:

- writes_enabled + limits_enabled + DB missing -> entering the app lifespan
  raises.
- writes_enabled + limits_enabled + DB readable (valid channel_limits JSON)
  -> starts, `/health` 200.
- writes_enabled + limits_enabled disabled -> starts without any database
  configured at all.
- writes_enabled=False -> starts read-only even with a missing database, and
  the probe (`LimitsValidator._load_limits_database`) is never called.
- The refusal message names the failing config keys but never leaks the
  database file's contents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.app import app, set_runner_factory
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import registry

_SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"
_DEMO_ENV = "BLUESKY_DEMO_RUNNER"
_MOTORS_ENV = "BLUESKY_EPICS_MOTORS"
_DETECTORS_ENV = "BLUESKY_EPICS_DETECTORS"
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean flag set, registry, runner factory, and connector global.

    The EPICS-substrate branch (where `_assert_limits_readable_if_writable`
    lives) only runs when `BLUESKY_EPICS_SUBSTRATE` is truthy, so this
    fixture enables it for every test in this file.
    """
    for var in (
        _SUBSTRATE_ENV,
        _DEMO_ENV,
        _MOTORS_ENV,
        _DETECTORS_ENV,
        _TILED_URI_ENV,
        _TILED_API_KEY_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)
    app_module._connector = None
    yield
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)
    app_module._connector = None


@pytest.fixture(autouse=True)
def _spy_connector_factory(monkeypatch: pytest.MonkeyPatch):
    """Stub out the real CA connector — this file probes the limits guard, not the connector.

    Requires bluesky/ophyd-async importable (same guard `_lifespan`'s
    EPICS-substrate branch itself requires); skipped, not failed, when the
    extra is absent.
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")
    from osprey.connectors.factory import ConnectorFactory

    class _SpyConnector:
        async def disconnect(self) -> None:
            return None

    async def fake_create_control_system_connector(config):
        return _SpyConnector()

    monkeypatch.setattr(
        ConnectorFactory,
        "create_control_system_connector",
        fake_create_control_system_connector,
    )


def _patch_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writes_enabled: bool,
    limits_enabled: bool | None = None,
    db_path: str | None = None,
    project_root: str | None = None,
) -> None:
    """Patch `osprey.utils.config.get_config_value` for the three keys the guard reads.

    `_assert_limits_readable_if_writable` does its own
    `from osprey.utils.config import get_config_value` inside the function
    body (never at module import time), so patching the underlying
    `osprey.utils.config` attribute — the same convention
    `test_epics_gateway_selection.py` uses for `EPICSConnector.connect` —
    takes effect on the next call.
    """

    def fake_get_config_value(key: str, default=None):
        if key == "control_system.writes_enabled":
            return writes_enabled
        if key == "control_system.limits_checking.enabled":
            return limits_enabled if limits_enabled is not None else default
        if key == "control_system.limits_checking.database_path":
            return db_path if db_path is not None else default
        if key == "project_root":
            return project_root if project_root is not None else default
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


def _valid_limits_db(tmp_path: Path) -> Path:
    db = tmp_path / "channel_limits.json"
    db.write_text(json.dumps({"TEST:MOTOR:01:SP": {"min_value": 0.0, "max_value": 10.0}}))
    return db


# =========================================================================
# The one unsafe combination: writable + limits enabled + DB unreadable
# =========================================================================


def test_writable_with_missing_limits_db_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist.json"
    _patch_config(monkeypatch, writes_enabled=True, limits_enabled=True, db_path=str(missing))

    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    message = str(excinfo.value)
    assert "writes_enabled" in message
    assert "limits_checking.enabled" in message


def test_writable_with_unparseable_limits_db_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad_json = tmp_path / "channel_limits.json"
    bad_json.write_text("{not valid json")
    _patch_config(monkeypatch, writes_enabled=True, limits_enabled=True, db_path=str(bad_json))

    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


# =========================================================================
# Every other combination starts normally
# =========================================================================


def test_writable_with_readable_limits_db_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = _valid_limits_db(tmp_path)
    _patch_config(monkeypatch, writes_enabled=True, limits_enabled=True, db_path=str(db))

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def test_writable_with_relative_db_path_resolves_via_config_file_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R4 container fix: a relative database_path resolves against the
    CONFIG_FILE directory, NOT project_root.

    Simulates the container deploy: CONFIG_FILE points at the mounted
    config.yml under /app/project (here, a temp dir containing a readable
    config.yml and data/channel_limits.json), while project_root is a
    bogus/nonexistent HOST path. Before the fix, the guard resolved the
    relative database_path against project_root and raised (the host path
    doesn't exist in-container); after the fix it resolves against the
    CONFIG_FILE directory and starts normally.
    """
    container_dir = tmp_path / "app_project"
    container_dir.mkdir()
    (container_dir / "config.yml").write_text("control_system: {}\n")
    data_dir = container_dir / "data"
    data_dir.mkdir()
    (data_dir / "channel_limits.json").write_text(
        json.dumps({"TEST:MOTOR:01:SP": {"min_value": 0.0, "max_value": 10.0}})
    )
    monkeypatch.setenv("CONFIG_FILE", str(container_dir / "config.yml"))

    bogus_project_root = tmp_path / "host_build_path_does_not_exist"
    _patch_config(
        monkeypatch,
        writes_enabled=True,
        limits_enabled=True,
        db_path="data/channel_limits.json",
        project_root=str(bogus_project_root),
    )

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def test_writable_with_limits_checking_disabled_starts_without_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """writes enabled + limits checking disabled: no database required at all."""
    _patch_config(monkeypatch, writes_enabled=True, limits_enabled=False, db_path=None)

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def test_writes_disabled_starts_readonly_even_with_missing_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read-only posture starts regardless of limits readability, and never probes."""
    missing = tmp_path / "does_not_exist.json"
    _patch_config(monkeypatch, writes_enabled=False, limits_enabled=True, db_path=str(missing))

    from osprey.connectors.control_system.limits_validator import LimitsValidator

    probe_calls: list[str] = []
    original_load = LimitsValidator._load_limits_database

    def spy(db_path: str):
        probe_calls.append(db_path)
        return original_load(db_path)

    monkeypatch.setattr(LimitsValidator, "_load_limits_database", staticmethod(spy))

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    # Writes disabled -> the guard returns before even resolving/probing the
    # database path; the readability probe must never run.
    assert probe_calls == []


# =========================================================================
# The refusal message never leaks the database's contents
# =========================================================================


def test_refusal_message_never_leaks_db_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "channel_limits.json"
    db.write_text('{"SECRET_MARKER_VALUE_DO_NOT_LEAK": not valid json')
    _patch_config(monkeypatch, writes_enabled=True, limits_enabled=True, db_path=str(db))

    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    assert "SECRET_MARKER_VALUE_DO_NOT_LEAK" not in str(excinfo.value)
