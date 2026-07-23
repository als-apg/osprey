"""Tests for the opt-in dispatch retention sweep.

Hermetic: ``tmp_path`` for the run-record log dir and the ArtifactStore, an
injected ``now`` for all age math (no real clock), and the periodic loop driven
one iteration via a monkeypatched ``asyncio.sleep`` (no real sleeps).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.mcp_server.dispatch_worker import retention
from osprey.stores.artifact_store import ArtifactStore

_DAY = retention._SECONDS_PER_DAY
_NOW = 1_000_000_000.0  # fixed reference epoch for every age computation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_run(
    log_dir: Path,
    run_id: str,
    *,
    status: str = "completed",
    completed_at: float | None = None,
    created_at: float | None = None,
) -> Path:
    """Write a persisted run record shaped like dispatch_api._persist_run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"run_id": run_id, "status": status}
    if completed_at is not None:
        record["completed_at"] = completed_at
    if created_at is not None:
        record["created_at"] = created_at
    path = log_dir / f"{run_id}.json"
    path.write_text(json.dumps(record))
    return path


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
# sweep_dispatch_runs — age boundary + in-flight protection
# ---------------------------------------------------------------------------


def test_run_age_boundary_unit_pinned(tmp_path):
    """N-1 days survives, exactly N survives, N+1 days is deleted."""
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "younger", completed_at=_NOW - 4 * _DAY)  # N-1
    _write_run(log_dir, "boundary", completed_at=_NOW - 5 * _DAY)  # exactly N
    _write_run(log_dir, "older", completed_at=_NOW - 6 * _DAY)  # N+1

    deleted = retention.sweep_dispatch_runs(log_dir, retention_days=5, now=_NOW)

    assert deleted == 1
    assert (log_dir / "younger.json").exists()
    assert (log_dir / "boundary.json").exists()
    assert not (log_dir / "older.json").exists()


def test_run_sweep_uses_created_at_when_no_completed_at(tmp_path):
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "old", status="error", created_at=_NOW - 10 * _DAY)

    deleted = retention.sweep_dispatch_runs(log_dir, retention_days=5, now=_NOW)

    assert deleted == 1
    assert not (log_dir / "old.json").exists()


def test_nonterminal_run_never_swept_regardless_of_age(tmp_path):
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "pending", status="pending", created_at=_NOW - 100 * _DAY)

    deleted = retention.sweep_dispatch_runs(log_dir, retention_days=5, now=_NOW)

    assert deleted == 0
    assert (log_dir / "pending.json").exists()


def test_in_flight_id_never_swept(tmp_path):
    """A terminal-looking record whose id is in-flight is still protected."""
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "live", status="completed", completed_at=_NOW - 100 * _DAY)

    deleted = retention.sweep_dispatch_runs(
        log_dir, retention_days=5, now=_NOW, in_flight_run_ids={"live"}
    )

    assert deleted == 0
    assert (log_dir / "live.json").exists()


def test_unreadable_run_file_skipped(tmp_path):
    log_dir = tmp_path / "dispatch"
    log_dir.mkdir(parents=True)
    (log_dir / "broken.json").write_text("{not json")
    _write_run(log_dir, "old", completed_at=_NOW - 10 * _DAY)

    deleted = retention.sweep_dispatch_runs(log_dir, retention_days=5, now=_NOW)

    assert deleted == 1
    assert (log_dir / "broken.json").exists()
    assert not (log_dir / "old.json").exists()


def test_run_sweep_disabled_and_missing_dir(tmp_path):
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "old", completed_at=_NOW - 10 * _DAY)
    # Disabled → no-op
    assert retention.sweep_dispatch_runs(log_dir, retention_days=0, now=_NOW) == 0
    assert (log_dir / "old.json").exists()
    # Missing dir → no-op
    assert retention.sweep_dispatch_runs(tmp_path / "nope", retention_days=5, now=_NOW) == 0


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


def test_artifact_sweep_disabled(tmp_path):
    store = ArtifactStore(workspace_root=tmp_path)
    art = _save_artifact(store, "old")
    _set_artifact_age_days(store, art, 100)

    assert retention.sweep_artifacts(store, retention_days=0, now=_NOW) == 0
    assert store.get_entry(art) is not None


# ---------------------------------------------------------------------------
# run_sweep (combined)
# ---------------------------------------------------------------------------


def test_run_sweep_combined_counts(tmp_path):
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "old", completed_at=_NOW - 10 * _DAY)
    _write_run(log_dir, "recent", completed_at=_NOW - 1 * _DAY)

    store = ArtifactStore(workspace_root=tmp_path)
    old_art = _save_artifact(store, "old")
    _set_artifact_age_days(store, old_art, 10)
    young_art = _save_artifact(store, "young")
    _set_artifact_age_days(store, young_art, 1)

    counts = retention.run_sweep(log_dir, store, retention_days=5, now=_NOW)

    assert counts == {"runs": 1, "artifacts": 1}
    assert not (log_dir / "old.json").exists()
    assert (log_dir / "recent.json").exists()
    assert store.get_entry(old_art) is None
    assert store.get_entry(young_art) is not None


def test_run_sweep_disabled_is_noop(tmp_path):
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "old", completed_at=_NOW - 10 * _DAY)
    store = ArtifactStore(workspace_root=tmp_path)

    counts = retention.run_sweep(log_dir, store, retention_days=0, now=_NOW)

    assert counts == {"runs": 0, "artifacts": 0}
    assert (log_dir / "old.json").exists()


# ---------------------------------------------------------------------------
# retention_loop — one iteration, no real sleeps
# ---------------------------------------------------------------------------


def test_retention_loop_runs_one_iteration(tmp_path, monkeypatch):
    log_dir = tmp_path / "dispatch"
    _write_run(log_dir, "old", completed_at=_NOW - 10 * _DAY)
    store = ArtifactStore(workspace_root=tmp_path)

    calls: list[float] = []

    async def fake_sleep(interval):
        # Let the first sleep return so one sweep runs, then cancel the loop.
        calls.append(interval)
        if len(calls) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await retention.retention_loop(
                log_dir,
                lambda: store,
                retention_days=5,
                in_flight_run_ids=lambda: frozenset(),
                interval_sec=123.0,
            )

    asyncio.run(drive())

    assert calls == [123.0, 123.0]
    assert not (log_dir / "old.json").exists()  # the one iteration swept it
