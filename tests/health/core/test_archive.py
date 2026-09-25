"""The ``archive`` health category reads the archive's own manifests."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from osprey.health.core.archive import archive
from osprey.health.models import Status

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)


def _config(*deployed: str, interval: int | None = None) -> dict:
    services: dict = {"archive": {}}
    if interval is not None:
        services["archive"]["interval_seconds"] = interval
    return {"deployed_services": list(deployed), "services": services}


def _archive_root(repo: Path) -> Path:
    root = repo / "var" / "archive"
    root.mkdir(parents=True)
    os.chmod(root, 0o700)
    return root


def _write(root: Path, day: str, *records: dict) -> None:
    (root / day).mkdir(exist_ok=True)
    with (root / day / "MANIFEST.jsonl").open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _pass(completed_at: str, errors: list | None = None) -> dict:
    return {
        "kind": "pass",
        "started_at": completed_at,
        "completed_at": completed_at,
        "files": 1,
        "bytes": 1,
        "errors": errors or [],
    }


def _rows(config: dict, repo: Path) -> dict:
    return {row.name: row for row in archive(config, cwd=repo, now=lambda: NOW)()}


def test_silent_when_not_deployed(tmp_path):
    assert archive(_config("openobserve"), cwd=tmp_path, now=lambda: NOW)() == []


def test_a_missing_archive_warns(tmp_path):
    rows = _rows(_config("archive"), tmp_path)

    assert rows["archive_last_pass"].status is Status.WARNING
    assert "No archive" in rows["archive_last_pass"].message


def test_an_archive_with_no_pass_warns(tmp_path):
    _archive_root(tmp_path)

    rows = _rows(_config("archive"), tmp_path)

    assert rows["archive_last_pass"].status is Status.WARNING
    assert "no pass" in rows["archive_last_pass"].message


def test_a_stale_pass_warns_at_twice_the_interval(tmp_path):
    root = _archive_root(tmp_path)
    _write(root, "2026-09-24", _pass("2026-09-24T09:00:00Z"))

    fresh = _rows(_config("archive", interval=7200), tmp_path)["archive_last_pass"]
    stale = _rows(_config("archive", interval=3600), tmp_path)["archive_last_pass"]

    assert fresh.status is Status.OK
    assert stale.status is Status.WARNING
    assert "3 h" in stale.message


def test_a_pass_with_errors_warns_naming_the_first_source(tmp_path):
    root = _archive_root(tmp_path)
    _write(
        root,
        "2026-09-24",
        _pass(
            "2026-09-24T11:00:00Z",
            errors=[{"source": "terminals/alice", "error": "Permission denied"}],
        ),
    )

    row = _rows(_config("archive"), tmp_path)["archive_last_pass"]

    assert row.status is Status.WARNING
    assert "terminals/alice" in row.message


def test_a_recent_clean_pass_is_ok(tmp_path):
    root = _archive_root(tmp_path)
    _write(root, "2026-09-23", _pass("2026-09-23T12:00:00Z", errors=[{"source": "x"}]))
    _write(root, "2026-09-24", _pass("2026-09-24T11:30:00Z"))

    rows = _rows(_config("archive"), tmp_path)

    assert rows["archive_last_pass"].status is Status.OK
    assert rows["archive_last_pass"].value == "30 min"
    assert "archive_telemetry_day" not in rows


def test_the_telemetry_row_warns_when_no_day_was_exported(tmp_path):
    root = _archive_root(tmp_path)
    _write(root, "2026-09-24", _pass("2026-09-24T11:30:00Z"))

    row = _rows(_config("archive", "openobserve"), tmp_path)["archive_telemetry_day"]

    assert row.status is Status.WARNING


def test_the_telemetry_row_warns_when_the_newest_day_is_old(tmp_path):
    root = _archive_root(tmp_path)
    _write(
        root,
        "2026-09-24",
        {"kind": "telemetry_day", "day": "2026-09-21", "rows": {}},
        _pass("2026-09-24T11:30:00Z"),
    )

    row = _rows(_config("archive", "openobserve"), tmp_path)["archive_telemetry_day"]

    assert row.status is Status.WARNING
    assert "2026-09-21" in row.message


def test_the_telemetry_row_is_ok_for_the_day_before_yesterday(tmp_path):
    root = _archive_root(tmp_path)
    _write(
        root,
        "2026-09-24",
        {"kind": "telemetry_day", "day": "2026-09-22", "rows": {}},
        _pass("2026-09-24T11:30:00Z"),
    )

    row = _rows(_config("archive", "openobserve"), tmp_path)["archive_telemetry_day"]

    assert row.status is Status.OK
    assert row.value == "2026-09-22"
