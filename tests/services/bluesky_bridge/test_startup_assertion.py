"""Tests for the fail-OPEN startup assertion.

`app.py`'s `_lifespan` hook refuses to start *writable* against an unreadable
limits source: it raises IFF ALL of the write tier allows writes on a lane
that is not degraded, `control_system.limits_checking.enabled` is true, AND the
limits database at `control_system.limits_checking.database_path` is missing,
unreadable, or unparseable. Every other combination starts normally — most
importantly, writes disabled must start read-only REGARDLESS of limits
readability, and must never even probe the database. See
`_assert_limits_readable_if_writable`'s docstring in `validation.py` for the
full condition and rationale.

"Could write" is the write tier (`tier_allows`), which is target-blind: on a
deployment whose tier allows writes, every lane that is not degraded must
refuse, whichever connector block armed the tier. A degraded lane
(`OSPREY_BLUESKY_LANE` declaring a target the config cannot resolve) can
never write and must start.

The guard runs on EVERY startup, not only when some wiring flag is set: the
posture it refuses is a property of the project config, so a deployment that
never touches the queue must be checked exactly like one that does. That is
what `test_guard_runs_with_no_bluesky_env_set_at_all` pins.

Exercised here:

- tier allowed + limits_enabled + DB missing -> entering the app lifespan
  raises.
- tier allowed + limits_enabled + DB readable (valid channel_limits JSON)
  -> starts, `/health` 200.
- tier allowed + limits_enabled disabled -> starts without any database
  configured at all.
- tier never -> starts read-only even with a missing database, and
  the probe (`LimitsValidator._load_limits_database`) is never called.
- A mixed per-type config -> both lanes refuse, because the write tier is
  target-blind; a degraded lane and a read-only run need no database at all.
- The refusal message names the resolved limits key and the lane it refused
  for, and no writes key, but never leaks the database file's contents.
- The other boot-time refusals in the same hook: a `BLUESKY_DEVICE_PAGE_SIZE`,
  `BLUESKY_LIVE_MAX_RUNS` or `BLUESKY_LIVE_MAX_ROWS_PER_RUN` that is not a
  whole number >= 1 fails the start rather than the first request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import queue
from osprey.services.bluesky_bridge.app import app, set_queue_backend

_DEVICES_FILE_ENV = "BLUESKY_DEVICES_FILE"
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"
_LANE_ENV = "OSPREY_BLUESKY_LANE"
_EXECUTION_MODE_ENV = "OSPREY_EXECUTION_MODE"
_DEVICE_PAGE_SIZE_ENV = "BLUESKY_DEVICE_PAGE_SIZE"
_LIVE_MAX_RUNS_ENV = "BLUESKY_LIVE_MAX_RUNS"
_LIVE_MAX_ROWS_ENV = "BLUESKY_LIVE_MAX_ROWS_PER_RUN"


class _InertBackend:
    """A pre-injected queue backend, so the lifespan owns no queue lifecycle.

    Injecting one is what tells `_lifespan` a caller owns the backend's whole
    lifecycle, which skips the background environment open entirely — this file
    is about the limits guard, and a startup probe of a manager that isn't
    there would only add noise to it.
    """

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test starts from a clean env and an inert, pre-injected backend.

    The device-file/Tiled variables are cleared rather than set: the guard is no
    longer reached through any of them, and leaving one set from the ambient
    environment would only obscure that.
    """
    for var in (
        _DEVICES_FILE_ENV,
        _TILED_URI_ENV,
        _TILED_API_KEY_ENV,
        _LANE_ENV,
        _EXECUTION_MODE_ENV,
        _DEVICE_PAGE_SIZE_ENV,
        _LIVE_MAX_RUNS_ENV,
        _LIVE_MAX_ROWS_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    set_queue_backend(_InertBackend())
    yield
    set_queue_backend(None)


def _patch_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writes_allowed: bool | None = None,
    section: dict[str, Any] | None = None,
    lane_targets: dict[str, str] | None = None,
    limits_enabled: bool | None = None,
    db_path: str | None = None,
    project_root: str | None = None,
) -> None:
    """Patch `osprey_connectors.config.get_config_value` for the keys the guard reads.

    Write posture is per connector type, so the guard reads the whole
    `control_system` SECTION (its `writes` tier key), plus the
    `services.<lane>.target` its lane declares. Pass `writes_allowed` for a
    deployment whose only posture is the tier (`writes: allowed` or
    `writes: never`); pass `section`
    (and `lane_targets`) for a config that arms one connector type and not
    another.

    Limits checking is per connector type too, and the guard reads it out of
    the same SECTION, so `limits_enabled` is folded into the section's
    `limits_checking` block rather than answered behind a dotted key. The copy
    is what keeps a module-level section constant (`_MIXED_SECTION`) from
    carrying one test's posture into the next. Only `database_path` stays
    deployment-wide, since the deployment mounts one limits database.

    `_assert_limits_readable_if_writable` does its own
    `from osprey_connectors.config import get_config_value` inside the function
    body (never at module import time), so patching the underlying
    `osprey_connectors.config` attribute — the same convention
    `test_epics_gateway_selection.py` uses for `EPICSConnector.connect` —
    takes effect on the next call.
    """
    control_system: dict[str, Any] = (
        {"writes": "allowed" if writes_allowed else "never"} if section is None else dict(section)
    )
    if limits_enabled is not None:
        control_system["limits_checking"] = {
            **control_system.get("limits_checking", {}),
            "enabled": limits_enabled,
        }
    targets = lane_targets or {}

    def fake_get_config_value(key: str, default=None):
        if key == "control_system":
            return control_system
        if key == "control_system.type":
            return control_system.get("type", default)
        if key == "control_system.limits_checking.database_path":
            return db_path if db_path is not None else default
        if key == "project_root":
            return project_root if project_root is not None else default
        if key.startswith("services.") and key.endswith(".target"):
            return targets.get(key.split(".")[1], default)
        return default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", fake_get_config_value)


def _valid_limits_db(tmp_path: Path) -> Path:
    db = tmp_path / "channel_limits.json"
    db.write_text(
        json.dumps({"TEST:COR:01:SP": {"writable": True, "min_value": 0.0, "max_value": 10.0}})
    )
    return db


def _spy_on_limits_probe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every `LimitsValidator._load_limits_database` call, still doing it.

    The returned list staying empty is the assertion that a non-writable
    posture never even reached for the database.
    """
    from osprey_connectors.control_system._limits_validator import LimitsValidator

    probe_calls: list[str] = []
    original_load = LimitsValidator._load_limits_database

    def spy(db_path: str):
        probe_calls.append(db_path)
        return original_load(db_path)

    monkeypatch.setattr(LimitsValidator, "_load_limits_database", staticmethod(spy))
    return probe_calls


# =========================================================================
# The one unsafe combination: writable + limits enabled + DB unreadable
# =========================================================================


def test_writable_with_missing_limits_db_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist.json"
    _patch_config(monkeypatch, writes_allowed=True, limits_enabled=True, db_path=str(missing))

    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    message = str(excinfo.value)
    assert "the write tier allows writes" in message
    assert "writes_enabled" not in message
    assert "limits_checking.enabled" in message


def test_guard_runs_with_no_bluesky_env_set_at_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard is unconditional, not gated on any device-wiring flag.

    It once lived inside the device-substrate runner branch, so a deployment
    that wired up no devices — today, one that never sets
    `BLUESKY_DEVICES_FILE` — was never checked, and a writable project with an
    unreadable limits database came up clean. The autouse fixture clears every
    wiring variable, so this test failing means the guard has been re-gated on
    something.
    """
    for var in (_DEVICES_FILE_ENV, _TILED_URI_ENV):
        assert var not in os.environ

    missing = tmp_path / "does_not_exist.json"
    _patch_config(monkeypatch, writes_allowed=True, limits_enabled=True, db_path=str(missing))

    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_writable_with_unparseable_limits_db_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad_json = tmp_path / "channel_limits.json"
    bad_json.write_text("{not valid json")
    _patch_config(monkeypatch, writes_allowed=True, limits_enabled=True, db_path=str(bad_json))

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
    _patch_config(monkeypatch, writes_allowed=True, limits_enabled=True, db_path=str(db))

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
        json.dumps({"TEST:COR:01:SP": {"writable": True, "min_value": 0.0, "max_value": 10.0}})
    )
    monkeypatch.setenv("CONFIG_FILE", str(container_dir / "config.yml"))

    bogus_project_root = tmp_path / "host_build_path_does_not_exist"
    _patch_config(
        monkeypatch,
        writes_allowed=True,
        limits_enabled=True,
        db_path="data/channel_limits.json",
        project_root=str(bogus_project_root),
    )

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def test_writable_with_an_unreadable_limits_leaf_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A leaf that is present but not a literal boolean is a startup refusal.

    The connector and the hook would refuse every write from the same posture;
    the gate turns that into a refusal at startup naming the line to fix,
    rather than a lane that starts writable and then denies everything.
    """
    # Arrange
    _patch_config(
        monkeypatch,
        section={"type": "epics", "writes": "allowed", "limits_checking": {"enabled": "true"}},
        db_path=str(_valid_limits_db(tmp_path)),
    )
    probe_calls = _spy_on_limits_probe(monkeypatch)

    # Act
    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    # Assert
    message = str(excinfo.value)
    assert "control_system.limits_checking.enabled" in message
    # An ABSENT deployment-wide leaf is unset, not unreadable, so it is not named.
    assert "control_system.limits_checking.mode" not in message
    assert "literal true or false" in message
    assert probe_calls == []


def test_writable_with_limits_checking_disabled_starts_without_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """writes enabled + limits checking disabled: no database required at all."""
    _patch_config(monkeypatch, writes_allowed=True, limits_enabled=False, db_path=None)

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def test_writable_optional_with_no_db_path_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """writes enabled + optional limits + no path: legal, nothing to load (q-0001/1, q-0002)."""
    probe_calls = _spy_on_limits_probe(monkeypatch)
    _patch_config(
        monkeypatch,
        section={"writes_enabled": True, "limits_checking": {"mode": "optional"}},
        limits_enabled=True,
        db_path=None,
    )

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
    assert probe_calls == []


def test_writable_exclusive_with_no_db_path_refuses_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusive mode needs the database to know what is writable at all."""
    _patch_config(
        monkeypatch,
        section={"writes_enabled": True, "limits_checking": {"mode": "exclusive"}},
        limits_enabled=True,
        db_path=None,
    )

    with pytest.raises(RuntimeError, match="database_path is not configured"):
        with TestClient(app):
            pass


def test_writable_optional_with_missing_db_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configured path must load whatever the mode."""
    _patch_config(
        monkeypatch,
        section={"writes_enabled": True, "limits_checking": {"mode": "optional"}},
        limits_enabled=True,
        db_path=str(tmp_path / "does_not_exist.json"),
    )

    with pytest.raises(RuntimeError, match="could not be read or parsed"):
        with TestClient(app):
            pass


def test_writes_disabled_starts_readonly_even_with_missing_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read-only posture starts regardless of limits readability, and never probes."""
    missing = tmp_path / "does_not_exist.json"
    _patch_config(monkeypatch, writes_allowed=False, limits_enabled=True, db_path=str(missing))

    probe_calls = _spy_on_limits_probe(monkeypatch)

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
    _patch_config(monkeypatch, writes_allowed=True, limits_enabled=True, db_path=str(db))

    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    assert "SECRET_MARKER_VALUE_DO_NOT_LEAK" not in str(excinfo.value)


# =========================================================================
# The tier is target-blind; only a degraded lane is exempt
# =========================================================================

#: A live-baseline deployment with a virtual accelerator beside it, whose tier
#: allows writes. The tier is target-blind (ruling PR-3): it arms every
#: machine, and the person's switch picks which one at run time, so a lane
#: serving `live` could write as well as one serving `va`.
_MIXED_SECTION: dict[str, Any] = {
    "type": "epics",
    "writes": "allowed",
    "connector": {
        "epics": {"gateway_address": "epics-gateway.example"},
        "virtual_accelerator": {},
    },
}

_VA_LANE = "bluesky_va"
_LIVE_LANE = "bluesky_live"
_MIXED_LANE_TARGETS = {_VA_LANE: "va", _LIVE_LANE: "live"}


def test_va_lane_requires_limits_db_when_the_tier_allows_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The VA lane of a tier-allowed deployment refuses without its database."""
    # Arrange
    monkeypatch.setenv(_LANE_ENV, _VA_LANE)
    missing = tmp_path / "does_not_exist.json"
    _patch_config(
        monkeypatch,
        section=_MIXED_SECTION,
        lane_targets=_MIXED_LANE_TARGETS,
        limits_enabled=True,
        db_path=str(missing),
    )

    # Act
    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    # Assert
    message = str(excinfo.value)
    assert "the write tier allows writes" in message
    assert "writes_enabled" not in message
    assert f"lane {_VA_LANE} serves target va" in message


def test_live_lane_requires_limits_db_too_because_the_tier_is_target_blind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other lane of the same config refuses as well.

    The write tier is target-blind: a deployment whose tier allows writes
    allows them on every machine, and the person's switch picks which one at
    run time. A live lane whose own block arms nothing can still be switched on
    after it starts, so it needs working limits enforcement from the start.
    """
    # Arrange
    monkeypatch.setenv(_LANE_ENV, _LIVE_LANE)
    missing = tmp_path / "does_not_exist.json"
    _patch_config(
        monkeypatch,
        section=_MIXED_SECTION,
        lane_targets=_MIXED_LANE_TARGETS,
        limits_enabled=True,
        db_path=str(missing),
    )

    # Act
    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(app):
            pass

    # Assert
    assert f"lane {_LIVE_LANE} serves target live" in str(excinfo.value)


def test_a_degraded_lane_does_not_require_the_limits_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lane whose target this deployment cannot resolve can never write.

    The connector refuses each of its writes for that reason, so the guard
    counts it as unable to write and never probes the database, whatever the
    tier allows.
    """
    # Arrange: a VA baseline with no live block, and a lane that declares live.
    monkeypatch.setenv(_LANE_ENV, _LIVE_LANE)
    missing = tmp_path / "does_not_exist.json"
    _patch_config(
        monkeypatch,
        section={
            "type": "virtual_accelerator",
            "writes": "allowed",
            "connector": {"virtual_accelerator": {}},
        },
        lane_targets=_MIXED_LANE_TARGETS,
        limits_enabled=True,
        db_path=str(missing),
    )
    probe_calls = _spy_on_limits_probe(monkeypatch)

    # Act
    with TestClient(app) as client:
        resp = client.get("/health")

    # Assert
    assert resp.status_code == 200
    assert probe_calls == []


def test_readonly_run_never_requires_the_limits_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A read-only run cannot write, so the armed lane needs no database either."""
    # Arrange
    monkeypatch.setenv(_LANE_ENV, _VA_LANE)
    monkeypatch.setenv(_EXECUTION_MODE_ENV, "readonly")
    missing = tmp_path / "does_not_exist.json"
    _patch_config(
        monkeypatch,
        section=_MIXED_SECTION,
        lane_targets=_MIXED_LANE_TARGETS,
        limits_enabled=True,
        db_path=str(missing),
    )
    probe_calls = _spy_on_limits_probe(monkeypatch)

    # Act
    with TestClient(app) as client:
        resp = client.get("/health")

    # Assert
    assert resp.status_code == 200
    assert probe_calls == []


# =========================================================================
# The device page size is parsed at boot, not at the first request
# =========================================================================


@pytest.mark.parametrize("bad_value", ["abc", "0", "-1"])
def test_malformed_device_page_size_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """A page size that is not a whole number >= 1 fails the boot.

    Read-only posture, so the limits guard passes and the only thing left that
    can refuse the lifespan is the page-size parse.
    """
    # Arrange
    _patch_config(monkeypatch, writes_allowed=False)
    monkeypatch.setenv(_DEVICE_PAGE_SIZE_ENV, bad_value)

    # Act
    with pytest.raises(ValueError) as excinfo:
        with TestClient(app):
            pass

    # Assert
    message = str(excinfo.value)
    assert _DEVICE_PAGE_SIZE_ENV in message
    assert bad_value in message


def test_device_page_size_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset is the overwhelmingly common case, and it is not an error."""
    monkeypatch.delenv(_DEVICE_PAGE_SIZE_ENV, raising=False)

    assert queue.device_page_size() == 500


def test_device_page_size_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value is read from `os.environ` on each call, never cached."""
    monkeypatch.setenv(_DEVICE_PAGE_SIZE_ENV, "200")

    assert queue.device_page_size() == 200


# =========================================================================
# The live-row buffer caps are parsed at boot too
# =========================================================================


@pytest.mark.parametrize("env_var", [_LIVE_MAX_RUNS_ENV, _LIVE_MAX_ROWS_ENV])
@pytest.mark.parametrize("bad_value", ["abc", "0", "-1"])
def test_malformed_live_row_cap_refuses_startup(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
) -> None:
    """A live-row cap that is not a whole number >= 1 fails the boot.

    The recorder that reads these caps handles every document inside a
    try/except, so a value it cannot parse would drop run buffers silently for
    the life of the container. Parsing at boot is what turns that into a
    refusal an operator can see.
    """
    # Arrange
    _patch_config(monkeypatch, writes_allowed=False)
    monkeypatch.setenv(env_var, bad_value)

    # Act
    with pytest.raises(ValueError) as excinfo:
        with TestClient(app):
            pass

    # Assert
    message = str(excinfo.value)
    assert env_var in message
    assert bad_value in message
