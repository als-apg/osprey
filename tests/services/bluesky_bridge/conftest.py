"""Shared isolation for the Bluesky bridge suites."""

from __future__ import annotations

from collections.abc import Iterator

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
