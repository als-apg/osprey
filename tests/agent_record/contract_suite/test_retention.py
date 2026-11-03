"""The record's daily retention, on every backend.

Retention removes an arrival day only when it is past the window
(``retention_days`` days before the run's day) and the archive holds it; the
removal stops before the oldest day the archive does not hold. A removed day
never comes back: a rewritten transcript's expired lines are skipped and the
export does not visit the removed days again.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import archive, retention
from osprey.agent_record import contract as c
from osprey.agent_record.ingest.tailer import TranscriptTailer
from osprey.agent_record.ingest.transcripts import parse_line
from osprey.services.archive.manifest import MANIFEST_NAME

from . import _kit as k

DAY = 24 * 3600
D0 = k.T0.date()


def _day(n: int) -> date:
    return D0 + timedelta(days=n)


def _now(day: date) -> datetime:
    """Noon UTC of *day*: every day before it is closed."""
    return k.T0 + timedelta(days=(day - D0).days)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    os.chmod(root, 0o700)
    return root


def _feedback(key: str, day: int) -> c.Entry:
    t = day * DAY + 5
    return k.entry(k.ALICE, key, c.KIND_FEEDBACK, t, {"text": key}, arrived=t)


def _keys(backend) -> set[str]:
    return {row.key for row in k.rows(backend)}


def _purge(root: Path, day: date) -> None:
    manifest = root / day.isoformat() / MANIFEST_NAME
    with manifest.open("a") as handle:
        handle.write(json.dumps({"kind": "agent_record_purged", "day": day.isoformat()}) + "\n")


@pytest.mark.parametrize("retention_days", [1, 14, 90])
def test_exported_days_past_the_window_are_removed(backend, settle, tmp_path, retention_days):
    # Today is day R + 2, so the window starts at day 2: days 0 and 1 are past it.
    today = _day(retention_days + 2)
    backend.append(
        [
            _feedback("sp:alice:d0", 0),
            _feedback("sp:alice:d1", 1),
            _feedback("sp:alice:d2", 2),
            _feedback("sp:alice:last", retention_days + 1),
            _feedback("sp:alice:today", retention_days + 2),
        ]
    )
    settle()
    root = _root(tmp_path)
    archive.run_export(backend, _now(today), root)

    assert retention.run_retention(backend, _now(today), retention_days, root) == 2
    settle()

    assert _keys(backend) == {"sp:alice:d2", "sp:alice:last", "sp:alice:today"}
    assert backend.health().oldest_arrival_day == _day(2)
    # A second run the same day finds nothing more to remove.
    assert retention.run_retention(backend, _now(today), retention_days, root) == 0


def test_the_last_moment_of_the_day_still_keeps_the_window_start(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:d0", 0), _feedback("sp:alice:d1", 1)])
    settle()
    root = _root(tmp_path)
    late = _now(_day(2)) + timedelta(hours=11, minutes=59)
    archive.run_export(backend, late, root)

    assert retention.run_retention(backend, late, 1, root) == 1
    settle()

    assert _keys(backend) == {"sp:alice:d1"}


def test_an_export_outage_longer_than_the_window_removes_nothing(backend, settle, tmp_path):
    backend.append([_feedback(f"sp:alice:d{n}", n) for n in range(5)])
    settle()
    root = _root(tmp_path)

    # The export never ran: no day is marked, and every day is far past a 1-day window.
    assert retention.run_retention(backend, _now(_day(30)), 1, root) == 0
    settle()

    assert _keys(backend) == {f"sp:alice:d{n}" for n in range(5)}


def test_an_unexported_day_stops_the_cutoff(backend, settle, tmp_path):
    backend.append([_feedback(f"sp:alice:d{n}", n) for n in range(4)])
    settle()
    root = _root(tmp_path)
    today = _day(4)
    archive.run_export(backend, _now(today), root)
    _purge(root, _day(1))
    assert archive.exported_days(root) == {_day(0), _day(2), _day(3)}

    # Days 0..2 are past a 1-day window, but day 1 is not in the archive.
    assert retention.run_retention(backend, _now(today), 1, root) == 1
    settle()
    assert _keys(backend) == {"sp:alice:d1", "sp:alice:d2", "sp:alice:d3"}

    # Once the export holds day 1 again, the rest past the window goes.
    assert archive.run_export(backend, _now(today), root) == [_day(1)]
    assert retention.run_retention(backend, _now(today), 1, root) == 2
    settle()
    assert _keys(backend) == {"sp:alice:d3"}


def test_an_empty_record_removes_nothing(backend, tmp_path):
    root = _root(tmp_path)
    assert retention.run_retention(backend, _now(_day(30)), 1, root) == 0


def test_a_retention_below_one_day_is_refused(backend, tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        retention.run_retention(backend, _now(_day(3)), 0, _root(tmp_path))


def _arrived_as_written(backend, path: Path, data: bytes) -> None:
    """Store *data*'s entries as the tailer parses them, each arriving at its own time.

    The tailer stamps arrival with the wall clock; the first ingest is stored
    directly so the transcript's facts span several arrival days.
    """
    entries: list[c.Entry] = []
    for line in data.splitlines(keepends=True):
        for parsed in parse_line(k.ALICE, k.relpath(path.stem), line):
            entries.append(dataclasses.replace(parsed, inserted_at=parsed.ts))
    backend.append(entries)


def test_a_transcript_rewritten_after_retention_is_not_inserted_or_archived_again(
    backend, settle, tmp_path
):
    mounts = tmp_path / "mounts"
    root = _root(tmp_path)
    session = "s-rewrite"
    old = k.user_line(session, "old question", uuid="u1", when=k.T0) + k.text_line(
        session, "old answer", uuid="u2", when=k.T0 + timedelta(seconds=30)
    )
    young = k.user_line(session, "young question", uuid="u3", when=_now(_day(2)))
    path = k.write_transcript(mounts, k.ALICE, session, old + young)
    _arrived_as_written(backend, path, old + young)
    settle()
    held = _keys(backend)
    young_keys = {row.key for row in k.rows(backend) if row.ts >= _now(_day(2))}
    assert young_keys and held - young_keys

    today = _day(3)
    assert archive.run_export(backend, _now(today), root) == [_day(0), _day(1), _day(2)]
    assert retention.run_retention(backend, _now(today), 1, root) == len(held - young_keys)
    settle()
    assert _keys(backend) == young_keys
    manifests = {
        day: (root / day.isoformat() / MANIFEST_NAME).read_bytes()
        for day in (_day(0), _day(1), _day(2))
    }

    # The transcript is read again from its first line, as after a rewrite.
    path.unlink()
    path.write_bytes(old + young)
    tailer = TranscriptTailer(
        backend, {k.ROOTS[k.ALICE]: mounts / k.ROOTS[k.ALICE]}, k.CONFIG, retention_days=1
    )
    report = tailer.poll_once(_now(today) + timedelta(minutes=1))
    assert not report.errors, report.errors
    settle()

    assert report.appended == 0
    assert report.skipped_retention == 2
    assert _keys(backend) == young_keys
    # The removed days are not exported again; only the newly closed day is.
    assert archive.run_export(backend, _now(today + timedelta(days=1)), root) == [today]
    for day, before in manifests.items():
        assert (root / day.isoformat() / MANIFEST_NAME).read_bytes() == before
