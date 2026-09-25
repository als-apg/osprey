"""A pass copies what changed, verbatim and append-only."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.services.archive import run as archive_run
from osprey.services.archive.manifest import MANIFEST_NAME
from osprey.services.archive.run import ArchiveDestinationError, run_pass

DAY1 = datetime(2026, 9, 24, 14, 22, 33, tzinfo=UTC)
LATER = datetime(2026, 9, 24, 18, 0, 0, tzinfo=UTC)


def _put(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    return path


def _lines(dest: Path, day: str = "2026-09-24") -> list[dict]:
    text = (dest / day / MANIFEST_NAME).read_text()
    return [json.loads(line) for line in text.splitlines()]


@pytest.fixture
def trees(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    os.chmod(dest, 0o700)
    return sources, dest


def test_first_pass_copies_every_included_file_verbatim(trees):
    sources, dest = trees
    payloads = {
        "terminals/alice/projects/p/s.jsonl": b'{"a":1}\n',
        "terminal_agent_data/alice/artifacts/artifacts.json": b"[]",
        "terminal_agent_data/alice/artifacts/files/plot.png": b"\x89PNG",
        "dispatch/shared/claude-config/projects/p/r.jsonl": b"run\n",
        "dispatch/shared/dispatch/abc.json": b"{}",
        "dispatch/shared/artifacts/artifacts.json": b"[1]",
        "bluesky/lane/dump.rdb": b"REDIS",
        "bluesky/lane/appendonlydir/a.aof": b"*1\r\n",
        "audit/ident/ledger.jsonl": b"{}\n",
    }
    for rel, data in payloads.items():
        _put(sources / rel, data)

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == len(payloads)
    assert result.errors == []
    for rel, data in payloads.items():
        copy = dest / "2026-09-24" / rel
        assert copy.read_bytes() == data
    records = {r["source"]: r for r in _lines(dest) if r["kind"] == "file"}
    for rel, data in payloads.items():
        assert records[rel]["sha256"] == hashlib.sha256(data).hexdigest()
        assert records[rel]["size"] == len(data)


def test_a_second_pass_with_nothing_changed_copies_nothing(trees):
    sources, dest = trees
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x\n")

    run_pass(sources, dest, now=DAY1)
    before = _lines(dest)
    result = run_pass(sources, dest, now=LATER)
    after = _lines(dest)

    assert result.files == 0
    assert after[: len(before)] == before
    assert [r["kind"] for r in after[len(before) :]] == ["pass"]


def test_a_grown_transcript_is_copied_whole_again_under_a_suffixed_name_the_same_day(trees):
    sources, dest = trees
    src = _put(sources / "terminals/alice/projects/p/abc.jsonl", "one\n")
    run_pass(sources, dest, now=DAY1)
    with src.open("a") as handle:
        handle.write("two\n")
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10_000_000))

    result = run_pass(sources, dest, now=LATER)

    assert result.files == 1
    base = dest / "2026-09-24/terminals/alice/projects/p"
    assert (base / "abc.jsonl").read_text() == "one\n"
    assert (base / "abc.T180000Z.jsonl").read_text() == "one\ntwo\n"


def test_a_touched_but_unchanged_file_is_not_copied(trees):
    sources, dest = trees
    src = _put(sources / "terminals/alice/projects/p/abc.jsonl", "one\n")
    run_pass(sources, dest, now=DAY1)
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10_000_000))

    result = run_pass(sources, dest, now=LATER)

    assert result.files == 0
    assert not list((dest / ".incoming").iterdir())


def test_nothing_outside_the_include_table_is_copied(trees):
    sources, dest = trees
    root = sources / "terminals" / "alice"
    _put(root / ".credentials.json", "secret")
    _put(root / "settings.json", "{}")
    _put(root / ".claude.json", "{}")
    _put(root / "projects/p/s.jsonl", "x\n")
    _put(sources / "dispatch/shared/workspace/notebook.ipynb", "{}")
    _put(sources / "dispatch/shared/dispatch/nested/a.json", "{}")

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == 1
    copied = [p for p in (dest / "2026-09-24").rglob("*") if p.is_file()]
    assert sorted(p.name for p in copied) == ["MANIFEST.jsonl", "s.jsonl"]


def test_an_unknown_kind_is_skipped_with_a_warning(trees, caplog):
    sources, dest = trees
    _put(sources / "mystery/x/projects/a.jsonl", "x")

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == 0
    assert "mystery" in caplog.text


def test_symlinks_are_not_followed(trees, tmp_path):
    sources, dest = trees
    outside = _put(tmp_path / "outside/secret.jsonl", "secret")
    (tmp_path / "outside/dir").mkdir()
    _put(tmp_path / "outside/dir/inner.jsonl", "inner")
    projects = sources / "terminals/alice/projects"
    projects.mkdir(parents=True)
    (projects / "link.jsonl").symlink_to(outside)
    (projects / "linkdir").symlink_to(tmp_path / "outside/dir")

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == 0
    assert "terminals/alice/projects/link.jsonl" in result.skipped
    assert "terminals/alice/projects/linkdir" in result.skipped
    assert not (dest / "2026-09-24/terminals").exists()


def test_a_deleted_source_leaves_its_archive_copy(trees):
    sources, dest = trees
    src = _put(sources / "dispatch/shared/dispatch/run.json", "{}")
    run_pass(sources, dest, now=DAY1)
    src.unlink()

    run_pass(sources, dest, now=LATER)

    assert (dest / "2026-09-24/dispatch/shared/dispatch/run.json").read_text() == "{}"
    assert any(r.get("source") == "dispatch/shared/dispatch/run.json" for r in _lines(dest))


@pytest.mark.parametrize(
    ("root_mode", "dir_mode", "file_mode"), [(0o700, 0o700, 0o600), (0o750, 0o750, 0o640)]
)
def test_modes_follow_the_archive_root(trees, root_mode, dir_mode, file_mode):
    sources, dest = trees
    os.chmod(dest, root_mode)
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x\n")

    run_pass(sources, dest, now=DAY1)

    day = dest / "2026-09-24"
    for path in [day, *day.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (dir_mode if path.is_dir() else file_mode), path


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_source_error_is_recorded_and_the_pass_goes_on(trees):
    sources, dest = trees
    bad = _put(sources / "terminals/alice/projects/p/a.jsonl", "a")
    _put(sources / "terminals/alice/projects/p/b.jsonl", "b")
    os.chmod(bad, 0)
    try:
        result = run_pass(sources, dest, now=DAY1)
    finally:
        os.chmod(bad, 0o600)

    assert result.files == 1
    assert [e["source"] for e in result.errors] == ["terminals/alice/projects/p/a.jsonl"]
    assert _lines(dest)[-1]["errors"] == result.errors


def test_a_missing_destination_refuses_before_copying(tmp_path):
    sources = tmp_path / "sources"
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x")

    with pytest.raises(ArchiveDestinationError):
        run_pass(sources, tmp_path / "absent", now=DAY1)

    assert not (tmp_path / "absent").exists()


def test_a_pass_line_that_cannot_be_written_refuses_the_pass(trees, monkeypatch):
    sources, dest = trees
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x")
    append = archive_run.ArchiveTree.append

    def _full_disk_on_pass_line(self, record):
        if record["kind"] == "pass":
            raise OSError(28, "No space left on device")
        append(self, record)

    monkeypatch.setattr(archive_run.ArchiveTree, "append", _full_disk_on_pass_line)

    with pytest.raises(ArchiveDestinationError):
        run_pass(sources, dest, now=DAY1)


def test_incoming_is_empty_after_a_pass(trees):
    sources, dest = trees
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x")
    _put(dest / ".incoming/leftover-from-a-crash", "torn")

    run_pass(sources, dest, now=DAY1)

    assert list((dest / ".incoming").iterdir()) == []


def test_a_pass_waits_while_another_pass_holds_the_archive(trees):
    sources, dest = trees
    _put(sources / "terminals/alice/projects/p/s.jsonl", "x")
    holder = archive_run.ArchiveTree(dest, DAY1)
    staged = _put(dest / ".incoming/staged-by-the-running-pass", "half")
    second = threading.Thread(target=run_pass, args=(sources, dest), kwargs={"now": LATER})

    try:
        second.start()
        second.join(timeout=0.5)
        assert second.is_alive()
        assert staged.read_text() == "half"
    finally:
        holder.close()
    second.join(timeout=10)

    assert not second.is_alive()
    assert list((dest / ".incoming").iterdir()) == []
