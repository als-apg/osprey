"""The archive's state is folded from its append-only manifests, never stored."""

from __future__ import annotations

import json
from pathlib import Path

from osprey.services.archive.manifest import MANIFEST_NAME, load_state
from osprey.services.archive.run import run_pass


def _write(day: Path, *records: dict) -> None:
    day.mkdir(parents=True, exist_ok=True)
    with (day / MANIFEST_NAME).open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _file(source: str, sha: str, size: int = 1) -> dict:
    return {
        "kind": "file",
        "source": source,
        "path": f"x/{source}",
        "size": size,
        "sha256": sha,
        "source_mtime_ns": 5,
    }


def test_state_folds_every_day_in_order(tmp_path):
    _write(tmp_path / "2026-09-02", _file("terminals/a/projects/s.jsonl", "new", 2))
    _write(
        tmp_path / "2026-09-01",
        _file("terminals/a/projects/s.jsonl", "old"),
        {"kind": "pass", "started_at": "first"},
    )
    _write(tmp_path / "2026-09-02", {"kind": "pass", "started_at": "second"})

    state = load_state(tmp_path)

    assert state.sources["terminals/a/projects/s.jsonl"].sha256 == "new"
    assert state.sources["terminals/a/projects/s.jsonl"].size == 2
    assert state.last_pass == {"kind": "pass", "started_at": "second"}


def test_a_line_of_an_unknown_kind_is_ignored(tmp_path):
    _write(
        tmp_path / "2026-09-01",
        _file("terminal_agent_data/a/artifacts/x.json", "abc"),
        {"kind": "telemetry_day", "day": "2026-08-31", "rows": {}},
        {"kind": "pass", "started_at": "p"},
    )

    state = load_state(tmp_path)

    assert set(state.sources) == {"terminal_agent_data/a/artifacts/x.json"}
    assert state.agent_record_days == set()
    assert state.last_pass == {"kind": "pass", "started_at": "p"}
    assert not hasattr(state, "telemetry_days")


def test_agent_record_days_fold_with_the_latest_line_winning(tmp_path):
    _write(
        tmp_path / "2026-10-21",
        {"kind": "agent_record_day", "day": "2026-10-21", "rows": 3},
        {"kind": "agent_record_purged", "day": "2026-10-21"},
    )
    _write(
        tmp_path / "2026-10-22",
        {"kind": "agent_record_day", "day": "2026-10-22", "rows": 0},
    )
    _write(
        tmp_path / "2026-10-23",
        {"kind": "agent_record_day", "day": "2026-10-23", "rows": 1},
        {"kind": "agent_record_purged", "day": "2026-10-23"},
        {"kind": "agent_record_day", "day": "2026-10-23", "rows": 2},
        {"kind": "agent_record_day"},
    )

    state = load_state(tmp_path)

    assert state.agent_record_days == {"2026-10-22", "2026-10-23"}
    assert state.sources == {}


def test_agent_record_lines_leave_the_copy_state_alone(tmp_path):
    _write(
        tmp_path / "2026-10-22",
        _file("terminals/a/projects/s.jsonl", "abc"),
        {"kind": "agent_record_day", "day": "2026-10-22", "rows": 0},
        {"kind": "pass", "started_at": "p"},
        {"kind": "agent_record_purged", "day": "2026-10-22"},
    )

    state = load_state(tmp_path)

    assert set(state.sources) == {"terminals/a/projects/s.jsonl"}
    assert state.last_pass == {"kind": "pass", "started_at": "p"}
    assert state.agent_record_days == set()


def test_a_torn_last_line_is_skipped(tmp_path, caplog):
    day = tmp_path / "2026-09-01"
    _write(day, _file("audit/x.jsonl", "abc"))
    with (day / MANIFEST_NAME).open("a") as handle:
        handle.write('{"kind": "file", "source": "audit/y.js')

    state = load_state(tmp_path)

    assert set(state.sources) == {"audit/x.jsonl"}
    assert "unreadable line" in caplog.text


def test_no_state_file_is_ever_written(tmp_path):
    sources = tmp_path / "src"
    artifacts = sources / "terminal_agent_data" / "alice" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "artifacts.json").write_text("[]")
    dest = tmp_path / "dest"
    dest.mkdir(mode=0o700)

    run_pass(sources, dest)
    run_pass(sources, dest)

    top = sorted(p.name for p in dest.iterdir())
    assert top[0] == ".incoming"
    assert all(name == ".incoming" or len(name) == 10 for name in top)
    files = [p for p in dest.rglob("*") if p.is_file()]
    names = {p.name for p in files}
    assert names == {"MANIFEST.jsonl", "artifacts.json"}
