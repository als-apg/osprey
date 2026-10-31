"""The record's jobs: export then retention, at start-up and daily.

Pins the job order (export, then retention; retention skipped when the export
fails), the contract signatures both jobs are called with, ``last_error``
kept from a failed export and cleared by a good one, the start-up run before
the first daily slot, the clock override feeding every run's ``now`` while
the wall clock decides when a run is due, a failed run never ending the loop,
a cancel that waits for a job thread still running, and both jobs answering
with the contract's return types while no day has closed.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import archive, retention, scheduler
from osprey.agent_record import contract as c
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.services.archive.run import INCOMING_DIRNAME

NOW = datetime(2026, 10, 23, 0, 0, tzinfo=UTC)
ROOT = Path("/archive-root")


class _Jobs:
    """Fakes for both jobs, recording each call with its arguments."""

    def __init__(self, monkeypatch, *, fail_export=lambda n: False):
        self.calls: list[tuple] = []
        self.exports = 0
        self._fail_export = fail_export
        monkeypatch.setattr(archive, "run_export", self.export)
        monkeypatch.setattr(retention, "run_retention", self.retention)

    def export(self, record, now, archive_root):
        self.exports += 1
        self.calls.append(("export", record, now, archive_root))
        if self._fail_export(self.exports):
            raise RuntimeError("archive unreachable")
        return []

    def retention(self, record, now, retention_days, archive_root):
        self.calls.append(("retention", record, now, retention_days, archive_root))
        return 0

    @property
    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


def test_next_run_is_the_next_daily_slot():
    at = scheduler.DAILY_AT
    before = datetime(2026, 10, 23, 0, 0, tzinfo=UTC)
    assert scheduler.next_run(before) == datetime(2026, 10, 23, at.hour, at.minute, tzinfo=UTC)
    on_it = datetime(2026, 10, 23, at.hour, at.minute, tzinfo=UTC)
    assert scheduler.next_run(on_it) == on_it + timedelta(days=1)


def test_the_jobs_run_export_then_retention_with_the_contract_arguments(monkeypatch):
    jobs = _Jobs(monkeypatch)
    record = object()
    scheduler.run_jobs(record, NOW, archive_root=ROOT, retention_days=7)
    assert jobs.calls == [
        ("export", record, NOW, ROOT),
        ("retention", record, NOW, 7, ROOT),
    ]


def test_a_failed_export_skips_retention_and_is_kept_as_last_error(monkeypatch):
    jobs = _Jobs(monkeypatch, fail_export=lambda n: n == 1)
    state = scheduler.JobsState()
    with pytest.raises(RuntimeError):
        scheduler.run_jobs(object(), NOW, archive_root=ROOT, retention_days=7, state=state)
    assert jobs.names == ["export"]
    assert state.last_error == "RuntimeError: archive unreachable"
    scheduler.run_jobs(object(), NOW, archive_root=ROOT, retention_days=7, state=state)
    assert state.last_error is None, "a good export clears the last error"
    assert jobs.names == ["export", "export", "retention"]


async def test_jobs_run_at_start_then_daily_and_a_failed_run_never_ends_the_loop(monkeypatch):
    jobs = _Jobs(monkeypatch, fail_export=lambda n: n == 1)
    wall = [NOW]
    waits: list[float] = []
    state = scheduler.JobsState()

    async def sleep(seconds: float) -> None:
        assert state.ready, "the start-up run finished before the first wait"
        waits.append(seconds)
        wall[0] += timedelta(seconds=seconds)
        if len(waits) > 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await scheduler.run(
            object(),
            archive_root=ROOT,
            retention_days=3,
            state=state,
            clock=lambda: wall[0],
            wall=lambda: wall[0],
            sleep=sleep,
        )
    assert waits[0] == 30 * 60 and waits[1] == 24 * 3600
    exports = [call[2] for call in jobs.calls if call[0] == "export"]
    assert exports == [NOW, NOW + timedelta(minutes=30), NOW + timedelta(days=1, minutes=30)]
    assert jobs.names == ["export", "export", "retention", "export", "retention"]
    assert state.last_error is None


async def test_the_clock_gives_now_while_the_wall_clock_gives_the_slot(monkeypatch):
    jobs = _Jobs(monkeypatch)
    fixed = datetime(2030, 1, 2, tzinfo=UTC)
    wall = [NOW]
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        wall[0] += timedelta(seconds=seconds)
        if len(waits) > 1:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await scheduler.run(
            object(),
            archive_root=ROOT,
            retention_days=3,
            clock=scheduler.fixed_clock(fixed),
            wall=lambda: wall[0],
            sleep=sleep,
        )
    assert {call[2] for call in jobs.calls} == {fixed}
    assert waits == [30 * 60, 24 * 3600], "the override never changes how often the jobs run"


def test_a_fixed_clock_reads_aware_utc():
    assert scheduler.fixed_clock(datetime(2030, 1, 2))() == datetime(2030, 1, 2, tzinfo=UTC)


async def test_a_cancel_waits_for_the_job_thread_still_running(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    events: list[str] = []

    def export(_record, _now, _root):
        started.set()
        release.wait(10)
        events.append("export done")
        return []

    monkeypatch.setattr(archive, "run_export", export)
    monkeypatch.setattr(retention, "run_retention", lambda *_a: 0)
    state = scheduler.JobsState()
    task = asyncio.create_task(
        scheduler.run(object(), archive_root=ROOT, retention_days=3, state=state)
    )
    assert await asyncio.to_thread(started.wait, 10)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done(), "the cancelled run still waits on its job thread"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    events.append("run returned")
    assert events == ["export done", "run returned"]
    assert not state.ready, "a run cancelled during start-up never reports ready"


def test_the_jobs_take_the_contract_arguments_and_change_nothing_before_a_day_closes(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        entry = c.Entry(
            key=c.spool_key("alice", "a1"),
            kind=c.KIND_AUDIT,
            ts=NOW,
            identity="alice",
            body={"what": "a1"},
            inserted_at=NOW,
        )
        backend.append([entry])
        assert archive.run_export(backend, NOW, root) == []
        assert retention.run_retention(backend, NOW, 1, root) == 0
        assert len(backend.select(c.ScopeFilter(), c.Query())) == 1
    # The export takes the archive lock on its staging directory; it stages
    # nothing and places no day while today is still open.
    assert [p.name for p in root.iterdir()] == [INCOMING_DIRNAME]
    assert list((root / INCOMING_DIRNAME).iterdir()) == []
