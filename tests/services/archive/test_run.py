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

#: One artifact file of a web terminal, under the sources tree and in the day.
ARTIFACT = "terminal_agent_data/alice/artifacts/files"


def _put(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    return path


def _lines(dest: Path, day: str = "2026-09-24") -> list[dict]:
    text = (dest / day / MANIFEST_NAME).read_text()
    return [json.loads(line) for line in text.splitlines()]


def test_first_pass_copies_every_included_file_verbatim(trees):
    sources, dest = trees
    # Source path under the sources tree -> its path in the day.
    payloads = {
        "terminal_agent_data/alice/artifacts/artifacts.json": (
            "terminal_agent_data/alice/artifacts/artifacts.json",
            b"[]",
        ),
        "terminal_agent_data/alice/artifacts/files/plot.png": (
            "terminal_agent_data/alice/artifacts/files/plot.png",
            b"\x89PNG",
        ),
        "dispatch/shared/artifacts/artifacts.json": (
            "dispatch/shared/artifacts/artifacts.json",
            b"[1]",
        ),
        "bluesky/lane/dump.rdb": ("bluesky/lane/dump.rdb", b"REDIS"),
        "bluesky/lane/appendonlydir/a.aof": ("bluesky/lane/appendonlydir/a.aof", b"*1\r\n"),
    }
    for rel, (_key, data) in payloads.items():
        _put(sources / rel, data)

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == len(payloads)
    assert result.errors == []
    for key, data in payloads.values():
        copy = dest / "2026-09-24" / key
        assert copy.read_bytes() == data
    records = {r["source"]: r for r in _lines(dest) if r["kind"] == "file"}
    for key, data in payloads.values():
        assert records[key]["sha256"] == hashlib.sha256(data).hexdigest()
        assert records[key]["size"] == len(data)


def test_a_second_pass_with_nothing_changed_copies_nothing(trees):
    sources, dest = trees
    _put(sources / f"{ARTIFACT}/s.jsonl", "x\n")

    run_pass(sources, dest, now=DAY1)
    before = _lines(dest)
    result = run_pass(sources, dest, now=LATER)
    after = _lines(dest)

    assert result.files == 0
    assert after[: len(before)] == before
    assert [r["kind"] for r in after[len(before) :]] == ["pass"]


def test_a_grown_file_is_copied_whole_again_under_a_suffixed_name_the_same_day(trees):
    sources, dest = trees
    src = _put(sources / f"{ARTIFACT}/abc.jsonl", "one\n")
    run_pass(sources, dest, now=DAY1)
    with src.open("a") as handle:
        handle.write("two\n")
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10_000_000))

    result = run_pass(sources, dest, now=LATER)

    assert result.files == 1
    base = dest / f"2026-09-24/{ARTIFACT}"
    assert (base / "abc.jsonl").read_text() == "one\n"
    assert (base / "abc.T180000Z.jsonl").read_text() == "one\ntwo\n"


def test_a_touched_but_unchanged_file_is_not_copied(trees):
    sources, dest = trees
    src = _put(sources / f"{ARTIFACT}/abc.jsonl", "one\n")
    run_pass(sources, dest, now=DAY1)
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10_000_000))

    result = run_pass(sources, dest, now=LATER)

    assert result.files == 0
    assert not list((dest / ".incoming").iterdir())


def test_nothing_outside_the_include_table_is_copied(trees):
    sources, dest = trees
    _put(sources / f"{ARTIFACT}/s.jsonl", "x\n")
    # What else a dispatch worker's agent-data volume holds: session state,
    # transcripts, the workspace and the run records.
    _put(sources / "dispatch/shared/claude-config/.credentials.json", "secret")
    _put(sources / "dispatch/shared/claude-config/projects/p/old.jsonl", "old\n")
    _put(sources / "dispatch/shared/workspace/notebook.ipynb", "{}")
    _put(sources / "dispatch/shared/dispatch/abc.json", "{}")
    _put(sources / "dispatch/shared/dispatch/nested/a.json", "{}")
    # What a web terminal's agent-data volume holds beside its artifacts.
    _put(sources / "terminal_agent_data/alice/notes/todo.md", "x")

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == 1
    copied = [p for p in (dest / "2026-09-24").rglob("*") if p.is_file()]
    assert sorted(p.name for p in copied) == ["MANIFEST.jsonl", "s.jsonl"]


@pytest.mark.parametrize("kind", ["terminals", "dispatch_projects", "audit"])
def test_transcripts_and_audit_ledgers_are_not_a_kind_the_archive_copies(trees, caplog, kind):
    sources, dest = trees
    _put(sources / f"{kind}/alice/p/s.jsonl", "{}\n")

    result = run_pass(sources, dest, now=DAY1)

    assert kind not in archive_run.SOURCE_TABLE
    assert result.files == 0
    assert kind in caplog.text
    assert not (dest / "2026-09-24" / kind).exists()


def test_the_archive_copies_artifact_stores_and_plan_queue_history_only():
    assert archive_run.SOURCE_TABLE == {
        "terminal_agent_data": ("artifacts/**",),
        "dispatch": ("artifacts/**",),
        "bluesky": ("appendonlydir/**", "dump.rdb"),
    }


def test_a_pass_line_names_no_telemetry(trees):
    sources, dest = trees
    _put(sources / f"{ARTIFACT}/s.jsonl", "x\n")

    run_pass(sources, dest, now=DAY1)

    assert set(_lines(dest)[-1]) == {
        "kind",
        "started_at",
        "completed_at",
        "files",
        "bytes",
        "errors",
        "skipped",
    }


def test_a_dot_named_file_follows_its_kinds_table(trees):
    sources, dest = trees
    _put(sources / "terminal_agent_data/alice/artifacts/.state.json", "{}")

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == 1
    assert (dest / "2026-09-24/terminal_agent_data/alice/artifacts/.state.json").exists()


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
    artifacts = sources / "terminal_agent_data/alice/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "link.jsonl").symlink_to(outside)
    (artifacts / "linkdir").symlink_to(tmp_path / "outside/dir")

    result = run_pass(sources, dest, now=DAY1)

    assert result.files == 0
    assert "terminal_agent_data/alice/artifacts/link.jsonl" in result.skipped
    assert "terminal_agent_data/alice/artifacts/linkdir" in result.skipped
    assert not (dest / "2026-09-24/terminal_agent_data").exists()


def test_a_deleted_source_leaves_its_archive_copy(trees):
    sources, dest = trees
    src = _put(sources / "dispatch/shared/artifacts/files/plot.png", "png")
    run_pass(sources, dest, now=DAY1)
    src.unlink()

    run_pass(sources, dest, now=LATER)

    assert (dest / "2026-09-24/dispatch/shared/artifacts/files/plot.png").read_text() == "png"
    assert any(r.get("source") == "dispatch/shared/artifacts/files/plot.png" for r in _lines(dest))


@pytest.mark.parametrize(
    ("root_mode", "dir_mode", "file_mode"), [(0o700, 0o700, 0o600), (0o750, 0o750, 0o640)]
)
def test_modes_follow_the_archive_root(trees, root_mode, dir_mode, file_mode):
    sources, dest = trees
    os.chmod(dest, root_mode)
    _put(sources / f"{ARTIFACT}/s.jsonl", "x\n")

    run_pass(sources, dest, now=DAY1)

    day = dest / "2026-09-24"
    for path in [day, *day.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (dir_mode if path.is_dir() else file_mode), path


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_source_error_is_recorded_and_the_pass_goes_on(trees):
    sources, dest = trees
    bad = _put(sources / f"{ARTIFACT}/a.jsonl", "a")
    _put(sources / f"{ARTIFACT}/b.jsonl", "b")
    os.chmod(bad, 0)
    try:
        result = run_pass(sources, dest, now=DAY1)
    finally:
        os.chmod(bad, 0o600)

    assert result.files == 1
    assert [e["source"] for e in result.errors] == [f"{ARTIFACT}/a.jsonl"]
    assert _lines(dest)[-1]["errors"] == result.errors


def test_a_missing_destination_refuses_before_copying(tmp_path):
    sources = tmp_path / "sources"
    _put(sources / f"{ARTIFACT}/s.jsonl", "x")

    with pytest.raises(ArchiveDestinationError):
        run_pass(sources, tmp_path / "absent", now=DAY1)

    assert not (tmp_path / "absent").exists()


def test_a_pass_line_that_cannot_be_written_refuses_the_pass(trees, monkeypatch):
    sources, dest = trees
    _put(sources / f"{ARTIFACT}/s.jsonl", "x")
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
    _put(sources / f"{ARTIFACT}/s.jsonl", "x")
    _put(dest / ".incoming/leftover-from-a-crash", "torn")

    run_pass(sources, dest, now=DAY1)

    assert list((dest / ".incoming").iterdir()) == []


def test_a_pass_waits_while_another_pass_holds_the_archive(trees):
    sources, dest = trees
    _put(sources / f"{ARTIFACT}/s.jsonl", "x")
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


def test_a_directory_handed_to_the_roots_owner_keeps_the_setgid_bit(tmp_path, monkeypatch):
    """A chown clears setgid, so the mode is set after it: a second writer under
    another uid then files into the root's group, which the deployer reads."""
    dest = tmp_path / "archive"
    dest.mkdir()
    os.chmod(dest, 0o2770)
    tree = archive_run.ArchiveTree(dest, DAY1)
    try:
        # What a rootful runtime does: hand the path over, losing setgid.
        def chown(path, uid, gid, *, follow_symlinks=True):  # noqa: ARG001
            os.chmod(path, stat.S_IMODE(os.lstat(path).st_mode) & ~stat.S_ISGID)

        monkeypatch.setattr(tree, "_owner", (os.getuid(), os.getgid()))
        monkeypatch.setattr(archive_run.os, "chown", chown)

        tree.makedirs(dest / "2026-01-01" / "agent_record")

        for directory in (dest / "2026-01-01", dest / "2026-01-01" / "agent_record"):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o2770, oct(directory.stat().st_mode)
    finally:
        tree.close()


def test_a_writer_that_is_not_root_hands_what_it_makes_to_the_roots_group(tmp_path, monkeypatch):
    """A parent another writer left without setgid would give a new directory
    the writer's own group; the root's group is set on it outright."""
    dest = tmp_path / "archive"
    dest.mkdir()
    tree = archive_run.ArchiveTree(dest, DAY1)
    try:
        calls: list[tuple[str, int, int]] = []
        monkeypatch.setattr(tree, "_owner", None)
        monkeypatch.setattr(tree, "_group", os.getgid() + 1)
        monkeypatch.setattr(
            archive_run.os,
            "chown",
            lambda path, uid, gid, *, follow_symlinks=True: calls.append(
                (Path(path).name, uid, gid)
            ),
        )

        tree.makedirs(dest / "2026-01-01" / "agent_record")

        assert calls == [
            ("2026-01-01", -1, os.getgid() + 1),
            ("agent_record", -1, os.getgid() + 1),
        ]
    finally:
        tree.close()
