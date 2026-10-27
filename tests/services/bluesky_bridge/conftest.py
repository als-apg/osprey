"""Shared isolation for the Bluesky bridge suites."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from osprey.services.bluesky_bridge import (
    analysis,
    document_plane,
    draft,
    figure_cache,
    history_removals,
    live_rows,
    plan_loader,
    queue,
    queue_removals,
)
from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.validation_record import validation_records
from tests._control_system import section_for

#: Variables a test sets itself when it needs them. Left to the ambient
#: environment, an exported ``BLUESKY_TILED_URI`` would send a test that falls
#: through to the real ``_from_tiled`` at a real Tiled server instead of the
#: "not configured" branch it claims to exercise.
_AMBIENT_ENV = (
    "BLUESKY_PLAN_DIRS",
    "BLUESKY_PLAN_MODULE",
    "BLUESKY_TILED_URI",
    "BLUESKY_TILED_API_KEY",
    "BLUESKY_LAUNCH_TOKEN",
)


def _reset_bridge_singletons() -> None:
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    queue_removals._clear()
    history_removals._clear()
    document_plane._clear()
    live_rows._clear()
    analysis._clear()
    figure_cache._clear()
    app_module.set_queue_backend(None)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts and ends with the bridge's process-wide state empty.

    The plan-loader cache, draft, queue, removal stores, live buffers, analysis
    and figure caches and the queue-backend override are module singletons in
    the real bridge, so one test's leftovers would otherwise be the next one's
    starting point. The validation-record store is snapshotted and restored
    rather than only cleared, since suites outside this directory share it.
    The writable plan directory is already per test (``tests/conftest.py``).
    """
    for var in _AMBIENT_ENV:
        monkeypatch.delenv(var, raising=False)
    _reset_bridge_singletons()
    with validation_records.lock:
        saved_hashes = set(validation_records._passing_hashes)
        validation_records._passing_hashes.clear()

    yield

    _reset_bridge_singletons()
    with validation_records.lock:
        validation_records._passing_hashes.clear()
        validation_records._passing_hashes.update(saved_hashes)


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | Exception], None]:
    """Set (or break) the ``control_system.type`` the bridge resolves.

    Both `_resolve_connector_type` and the shared `_resolve_control_system_type`
    it delegates to import `get_config_value` inside the function body, so
    patching it at its home module covers both.
    """

    def _set(value: str | Exception) -> None:
        def fake_get_config_value(key: str, default: Any = None) -> Any:
            if isinstance(value, Exception):
                raise value
            # Only the control system is what this fixture is about. Every
            # other key falls through to its default, because the bridge reads
            # more than one (the lifespan's limits guard reads three) and
            # answering all of them with the connector type would stage a
            # nonsense config: a `writes_enabled` of "epics" is truthy.
            section = section_for(value)
            if key == "control_system.type":
                return section["type"]
            if key == "control_system":
                return section
            return default

        monkeypatch.setattr("osprey_connectors.config.get_config_value", fake_get_config_value)

    return _set


@pytest.fixture
def limits_probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every `LimitsValidator._load_limits_database` call, still doing it.

    The list staying empty is the assertion that a posture never even reached
    for the database.
    """
    from osprey_connectors.control_system._limits_validator import LimitsValidator

    probe_calls: list[str] = []
    original_load = LimitsValidator._load_limits_database

    def spy(db_path: str):
        probe_calls.append(db_path)
        return original_load(db_path)

    monkeypatch.setattr(LimitsValidator, "_load_limits_database", staticmethod(spy))
    return probe_calls
