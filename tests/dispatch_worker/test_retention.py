"""Tests for the opt-in dispatch artifact retention sweep.

Hermetic: ``tmp_path`` for the ArtifactStore, an injected ``now`` for all age
math (no real clock), and the periodic loop driven one iteration via a
monkeypatched ``asyncio.sleep`` (no real sleeps).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from osprey.mcp_server.dispatch_worker import retention
from osprey.stores.artifact_store import ArtifactStore

_DAY = retention._SECONDS_PER_DAY
_NOW = 1_000_000_000.0  # fixed reference epoch for every age computation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _save_artifact(store: ArtifactStore, title: str, *, run_id: str = "") -> str:
    """Save a trivial artifact; return its id. run_id tags the created-by run."""
    import os

    prev = os.environ.get("OSPREY_DISPATCH_RUN_ID")
    if run_id:
        os.environ["OSPREY_DISPATCH_RUN_ID"] = run_id
    else:
        os.environ.pop("OSPREY_DISPATCH_RUN_ID", None)
    try:
        entry = store.save_file(
            file_content=b"x",
            filename="a.txt",
            artifact_type="text",
            title=title,
            mime_type="text/plain",
            tool_source="test",
        )
    finally:
        if prev is None:
            os.environ.pop("OSPREY_DISPATCH_RUN_ID", None)
        else:
            os.environ["OSPREY_DISPATCH_RUN_ID"] = prev
    return entry.id


def _set_artifact_age_days(store: ArtifactStore, artifact_id: str, days_old: float) -> None:
    """Rewrite an artifact's stored timestamp to ``days_old`` days before _NOW."""
    ts = datetime.fromtimestamp(_NOW - days_old * _DAY, tz=UTC).isoformat()
    for e in store._entries:
        if e.id == artifact_id:
            e.timestamp = ts
    store._save_index()


# ---------------------------------------------------------------------------
# retention_days_from_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 0),
        ("", 0),
        ("   ", 0),
        ("0", 0),
        ("-5", 0),
        ("abc", 0),
        ("1.5", 0),
        ("7", 7),
        (" 30 ", 30),
    ],
)
def test_retention_days_from_env(raw, expected):
    env = {} if raw is None else {"RETENTION_DAYS": raw}
    assert retention.retention_days_from_env(env) == expected


def test_retention_days_reads_os_environ(monkeypatch):
    monkeypatch.setenv("RETENTION_DAYS", "14")
    assert retention.retention_days_from_env() == 14


# ---------------------------------------------------------------------------
# sweep_artifacts
# ---------------------------------------------------------------------------


def test_artifact_age_boundary(tmp_path):
    store = ArtifactStore(workspace_root=tmp_path)
    young = _save_artifact(store, "young")
    old = _save_artifact(store, "old")
    _set_artifact_age_days(store, young, 4)  # N-1
    _set_artifact_age_days(store, old, 6)  # N+1

    deleted = retention.sweep_artifacts(store, retention_days=5, now=_NOW)

    assert deleted == 1
    assert store.get_entry(young) is not None
    assert store.get_entry(old) is None


def test_artifact_of_in_flight_run_survives(tmp_path):
    store = ArtifactStore(workspace_root=tmp_path)
    art = _save_artifact(store, "live-art", run_id="live-run")
    _set_artifact_age_days(store, art, 100)

    deleted = retention.sweep_artifacts(
        store, retention_days=5, now=_NOW, in_flight_run_ids={"live-run"}
    )

    assert deleted == 0
    assert store.get_entry(art) is not None


def test_artifact_sweep_deletes_run_under_the_system_actor(tmp_path):
    """Retention deletes are maintenance: the store-level activity listener
    must be able to tell them apart from agent deletes."""
    from osprey.stores.artifact_store import (
        current_artifact_mutation_actor,
        register_artifact_delete_listener,
        unregister_artifact_delete_listener,
    )

    store = ArtifactStore(workspace_root=tmp_path)
    old = _save_artifact(store, "old")
    _set_artifact_age_days(store, old, 100)

    actors = []

    def record_actor(_entry):
        actors.append(current_artifact_mutation_actor())

    register_artifact_delete_listener(record_actor)
    try:
        deleted = retention.sweep_artifacts(store, retention_days=5, now=_NOW)
    finally:
        unregister_artifact_delete_listener(record_actor)

    assert deleted == 1
    assert actors == ["system"]


def test_artifact_sweep_disabled(tmp_path):
    store = ArtifactStore(workspace_root=tmp_path)
    art = _save_artifact(store, "old")
    _set_artifact_age_days(store, art, 100)

    assert retention.sweep_artifacts(store, retention_days=0, now=_NOW) == 0
    assert store.get_entry(art) is not None


def test_retention_keeps_no_run_records():
    """Run history is the agent record's to age out, not this sweep's."""
    for gone in ("record_is_deletable", "delete_run_records", "sweep_dispatch_runs"):
        assert not hasattr(retention, gone)


# ---------------------------------------------------------------------------
# retention_loop — driven a few iterations, no real sleeps
# ---------------------------------------------------------------------------


def _scoped_sleep(monkeypatch, cycles: int) -> list[float]:
    """Patch the retention module's own ``asyncio.sleep`` to stop after *cycles*.

    ``retention_loop`` awaits ``asyncio.sleep(...)`` through its own module-level
    ``asyncio`` name. Patch THAT name, never ``asyncio.sleep`` itself: the
    function is shared with every other event loop alive in the process, and a
    daemon thread left by an earlier test in the same worker would tick through
    the fake.
    """
    calls: list[float] = []

    async def fake_sleep(interval):
        calls.append(interval)
        if len(calls) > cycles:
            raise asyncio.CancelledError

    class _AsyncioForRetention:
        sleep = staticmethod(fake_sleep)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    monkeypatch.setattr(retention, "asyncio", _AsyncioForRetention())
    return calls


def _drive_loop(store_factory, in_flight=lambda: frozenset()) -> None:
    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await retention.retention_loop(
                store_factory,
                retention_days=5,
                in_flight_run_ids=in_flight,
                interval_sec=123.0,
            )

    asyncio.run(drive())


def test_retention_loop_runs_one_iteration(tmp_path, monkeypatch):
    store = ArtifactStore(workspace_root=tmp_path)
    art = _save_artifact(store, "old")
    _set_artifact_age_days(store, art, 10)
    calls = _scoped_sleep(monkeypatch, cycles=1)

    _drive_loop(lambda: store)

    assert calls == [123.0, 123.0]
    assert store.get_entry(art) is None  # the one iteration swept it


def test_retention_loop_builds_its_store_and_reads_in_flight_runs_per_cycle(tmp_path, monkeypatch):
    """Both are resolved per cycle, not once at start.

    The loop starts during application lifespan, before anything has
    established that the bind-mounted config is readable, so a store rooted
    once there could stay on the fallback root for the life of the process;
    and a run that starts mid-interval must be protected on the next cycle.
    """
    store = ArtifactStore(workspace_root=tmp_path)
    built: list[int] = []
    asked: list[int] = []

    def _factory():
        built.append(1)
        return store

    def _in_flight():
        asked.append(1)
        return frozenset()

    _scoped_sleep(monkeypatch, cycles=2)

    _drive_loop(_factory, _in_flight)

    assert len(built) == 2
    assert len(asked) == 2
