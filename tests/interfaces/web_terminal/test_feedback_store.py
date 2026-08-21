"""Tests for the web-terminal feedback record store.

Covers the three guarantees the on-disk layout rests on:

* ids are unique even when two submissions land in the same millisecond,
* the context document is durable on disk *before* the header that points at
  it (a crash may leave an orphan context, never a dangling header),
* in-flight temporary files can never be picked up by the ``fb-*.json`` /
  ``ctx-*.json`` globs the CLI reads the store with,
* pruning to the store ceiling drops bulk (contexts) oldest-first and never
  drops history (headers).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

from osprey.interfaces.web_terminal import feedback_store

ID_RE = re.compile(r"^fb-\d+-[0-9a-f]{8}$")


def _record_globs_in(directory: Path) -> list[str]:
    """Names in *directory* that the store's public globs would pick up."""
    return [
        entry.name
        for entry in sorted(directory.iterdir())
        if fnmatch.fnmatch(entry.name, feedback_store.HEADER_GLOB)
        or fnmatch.fnmatch(entry.name, feedback_store.CONTEXT_GLOB)
    ]


# ── ids ────────────────────────────────────────────────────────────────────


def test_new_record_id_has_the_documented_shape() -> None:
    record_id = feedback_store.new_record_id()
    assert ID_RE.match(record_id), record_id


def test_new_record_id_is_unique_within_one_frozen_millisecond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_store, "_now_ms", lambda: 1755712345678)
    ids = {feedback_store.new_record_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("fb-1755712345678-") for i in ids)


def test_filename_helpers_pair_a_header_and_context_for_one_record_id() -> None:
    record_id = "fb-1755712345678-3fa9c1d2"
    assert feedback_store.header_filename(record_id) == "fb-1755712345678-3fa9c1d2.json"
    assert feedback_store.context_filename(record_id) == "ctx-1755712345678-3fa9c1d2.json"
    assert fnmatch.fnmatch(feedback_store.header_filename(record_id), feedback_store.HEADER_GLOB)
    assert fnmatch.fnmatch(feedback_store.context_filename(record_id), feedback_store.CONTEXT_GLOB)


# ── write_record ───────────────────────────────────────────────────────────


def test_write_record_creates_both_documents_and_returns_the_id(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "nested" / "feedback"

    record_id = feedback_store.write_record(
        feedback_dir,
        {"channel": "local", "excerpt": "it broke"},
        {"context": {"panels": []}, "scrollback": "$ osprey\n"},
    )

    assert ID_RE.match(record_id), record_id
    header_path = feedback_dir / feedback_store.header_filename(record_id)
    context_path = feedback_dir / feedback_store.context_filename(record_id)
    assert header_path.is_file()
    assert context_path.is_file()

    header = json.loads(header_path.read_text())
    assert header["id"] == record_id
    assert header["context_file"] == context_path.name
    assert header["channel"] == "local"
    assert header["excerpt"] == "it broke"

    context = json.loads(context_path.read_text())
    assert context["id"] == record_id
    assert context["context"] == {"panels": []}
    assert context["scrollback"] == "$ osprey\n"


def test_write_record_puts_the_context_on_disk_before_the_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feedback_dir = tmp_path / "feedback"
    real_replace = os.replace
    destinations: list[str] = []
    committed_before_each: list[list[str]] = []

    def spy_replace(src, dst, *args, **kwargs):  # noqa: ANN001, ANN202 — stdlib passthrough
        committed_before_each.append(_record_globs_in(Path(dst).parent))
        destinations.append(Path(dst).name)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(feedback_store.os, "replace", spy_replace)

    record_id = feedback_store.write_record(feedback_dir, {"channel": "local"}, {"context": {}})

    assert destinations == [
        feedback_store.context_filename(record_id),
        feedback_store.header_filename(record_id),
    ]
    # Nothing matching the globs existed before the context landed, and only the
    # context existed when the header was committed.
    assert committed_before_each[0] == []
    assert committed_before_each[1] == [feedback_store.context_filename(record_id)]


def test_write_record_temp_files_never_match_the_record_globs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feedback_dir = tmp_path / "feedback"
    real_mkstemp = tempfile.mkstemp
    temp_names: list[str] = []

    def spy_mkstemp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 — stdlib passthrough
        fd, name = real_mkstemp(*args, **kwargs)
        temp_names.append(Path(name).name)
        return fd, name

    monkeypatch.setattr(feedback_store.tempfile, "mkstemp", spy_mkstemp)
    real_replace = os.replace
    mid_write_listings: list[list[str]] = []

    def spy_replace(src, dst, *args, **kwargs):  # noqa: ANN001, ANN202 — stdlib passthrough
        # The temp file is still present here; anything it matched would be
        # read as a half-written record by the CLI.
        mid_write_listings.append(sorted(p.name for p in Path(dst).parent.iterdir()))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(feedback_store.os, "replace", spy_replace)

    feedback_store.write_record(feedback_dir, {"channel": "local"}, {"context": {}})

    assert len(temp_names) == 2
    for name in temp_names:
        assert name.startswith(".")
        assert name.endswith(".tmp")
        assert not fnmatch.fnmatch(name, feedback_store.HEADER_GLOB)
        assert not fnmatch.fnmatch(name, feedback_store.CONTEXT_GLOB)
    # The mid-write listing of the context step is the temp file alone.
    assert all(n.startswith(".") for n in mid_write_listings[0])


def test_write_record_does_not_mutate_the_caller_dicts(tmp_path: Path) -> None:
    header = {"channel": "local"}
    context = {"context": {}}

    feedback_store.write_record(tmp_path / "feedback", header, context)

    assert header == {"channel": "local"}
    assert context == {"context": {}}


def test_write_record_id_and_pointer_win_over_caller_supplied_values(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"

    record_id = feedback_store.write_record(
        feedback_dir,
        {"id": "spoofed", "context_file": "../escape.json"},
        {"id": "spoofed"},
    )

    header = json.loads((feedback_dir / feedback_store.header_filename(record_id)).read_text())
    context = json.loads((feedback_dir / feedback_store.context_filename(record_id)).read_text())
    assert header["id"] == record_id
    assert header["context_file"] == feedback_store.context_filename(record_id)
    assert context["id"] == record_id


def test_write_record_files_under_a_caller_supplied_id(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    allocated = "fb-1755712345678-3fa9c1d2"

    record_id = feedback_store.write_record(
        feedback_dir,
        {"channel": "local", "id": "spoofed"},
        {"context": {}, "id": "spoofed"},
        record_id=allocated,
    )

    assert record_id == allocated
    assert _record_globs_in(feedback_dir) == [
        "ctx-1755712345678-3fa9c1d2.json",
        "fb-1755712345678-3fa9c1d2.json",
    ]
    header = json.loads((feedback_dir / feedback_store.header_filename(allocated)).read_text())
    context = json.loads((feedback_dir / feedback_store.context_filename(allocated)).read_text())
    assert header["id"] == allocated
    assert header["context_file"] == feedback_store.context_filename(allocated)
    assert context["id"] == allocated


@pytest.mark.parametrize(
    "bad_id", ["1755712345678-3fa9c1d2", "ctx-1755712345678-3fa9c1d2", "../escape", ""]
)
def test_write_record_refuses_an_id_without_the_header_prefix(tmp_path: Path, bad_id: str) -> None:
    # The id IS the header's filename stem: without the prefix the header files
    # somewhere HEADER_GLOB cannot see, so the submission disappears from
    # list_headers and the pruner while its context becomes a prunable orphan.
    feedback_dir = tmp_path / "feedback"

    with pytest.raises(ValueError, match="must start with"):
        feedback_store.write_record(
            feedback_dir, {"channel": "local"}, {"context": {}}, record_id=bad_id
        )

    # Refused before the directory is even created, so a bad id leaves nothing.
    assert not feedback_dir.exists()


def test_write_record_mints_its_own_id_when_none_is_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feedback_dir = tmp_path / "feedback"
    monkeypatch.setattr(feedback_store, "_now_ms", lambda: 1755712345678)

    record_id = feedback_store.write_record(feedback_dir, {"channel": "local"}, {"context": {}})

    assert ID_RE.match(record_id), record_id
    assert record_id.startswith("fb-1755712345678-")
    assert (feedback_dir / feedback_store.header_filename(record_id)).is_file()


def test_atomic_write_unlinks_the_temp_file_when_serialization_fails(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    target = feedback_dir / "fb-1-aaaaaaaa.json"

    with pytest.raises(TypeError):
        feedback_store._atomic_write(target, {"ok": 1, ("bad", "key"): 2})

    assert list(feedback_dir.iterdir()) == []


# ── list_headers ───────────────────────────────────────────────────────────


def test_list_headers_tolerates_a_missing_directory(tmp_path: Path) -> None:
    assert feedback_store.list_headers(tmp_path / "never-created") == []


def test_list_headers_reads_only_headers_ignoring_contexts_and_temp_files(
    tmp_path: Path,
) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    (feedback_dir / "fb-1755712345678-aaaaaaaa.json").write_text(
        json.dumps({"id": "fb-1755712345678-aaaaaaaa", "channel": "local"})
    )
    (feedback_dir / "ctx-1755712345678-aaaaaaaa.json").write_text(
        json.dumps({"id": "fb-1755712345678-aaaaaaaa", "context": {"big": "payload"}})
    )
    (feedback_dir / ".fb-1755712345679-bbbbbbbb.json.xyz.tmp").write_text("{partial")
    (feedback_dir / "notes.txt").write_text("unrelated")

    headers = feedback_store.list_headers(feedback_dir)

    assert [h["id"] for h in headers] == ["fb-1755712345678-aaaaaaaa"]
    assert headers[0]["channel"] == "local"


def test_list_headers_sorts_by_id(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    for stem in (
        "fb-1755712345680-cccccccc",
        "fb-1755712345678-aaaaaaaa",
        "fb-1755712345679-bbbbbbbb",
    ):
        (feedback_dir / f"{stem}.json").write_text(json.dumps({"id": stem}))

    assert [h["id"] for h in feedback_store.list_headers(feedback_dir)] == [
        "fb-1755712345678-aaaaaaaa",
        "fb-1755712345679-bbbbbbbb",
        "fb-1755712345680-cccccccc",
    ]


def test_list_headers_skips_unreadable_documents(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    (feedback_dir / "fb-1755712345678-aaaaaaaa.json").write_text(
        json.dumps({"id": "fb-1755712345678-aaaaaaaa"})
    )
    (feedback_dir / "fb-1755712345679-bbbbbbbb.json").write_text("{ truncated")
    (feedback_dir / "fb-1755712345680-cccccccc.json").write_text(json.dumps(["not", "a", "dict"]))

    assert [h["id"] for h in feedback_store.list_headers(feedback_dir)] == [
        "fb-1755712345678-aaaaaaaa"
    ]


def test_list_headers_returns_empty_for_a_file_where_the_directory_should_be(
    tmp_path: Path,
) -> None:
    not_a_dir = tmp_path / "feedback"
    not_a_dir.write_text("oops")

    assert feedback_store.list_headers(not_a_dir) == []


# ── prune_store ────────────────────────────────────────────────────────────


def _seed_record(
    feedback_dir: Path,
    milliseconds: int,
    suffix: str,
    *,
    context: bool = True,
    header: dict | None = None,
) -> str:
    """Write one on-disk record by hand and return its id.

    Built without :func:`write_record` so the timestamp component of the id —
    the ordering prune relies on — is chosen by the test rather than the clock.
    The context payload is far larger than the header, so a ceiling expressed
    as "the store minus N contexts" cannot be met by header bytes alone.
    """
    feedback_dir.mkdir(parents=True, exist_ok=True)
    record_id = f"fb-{milliseconds}-{suffix}"
    header_doc = {
        "id": record_id,
        "context_file": feedback_store.context_filename(record_id),
        "channel": "local",
    }
    header_doc.update(header or {})
    (feedback_dir / feedback_store.header_filename(record_id)).write_text(json.dumps(header_doc))
    if context:
        (feedback_dir / feedback_store.context_filename(record_id)).write_text(
            json.dumps({"id": record_id, "scrollback": "x" * 2000})
        )
    return record_id


def _store_bytes(directory: Path) -> int:
    """Total size of the files the store ceiling is measured over."""
    return sum(
        entry.stat().st_size
        for entry in directory.iterdir()
        if fnmatch.fnmatch(entry.name, feedback_store.HEADER_GLOB)
        or fnmatch.fnmatch(entry.name, feedback_store.CONTEXT_GLOB)
    )


def test_prune_store_is_a_no_op_below_the_ceiling(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    ids = [_seed_record(feedback_dir, 1755712345678 + i, f"{i:08x}") for i in range(3)]

    assert feedback_store.prune_store(feedback_dir, 10_000_000) == []

    assert _record_globs_in(feedback_dir) == sorted(
        [feedback_store.header_filename(i) for i in ids]
        + [feedback_store.context_filename(i) for i in ids]
    )


def test_prune_store_deletes_the_oldest_contexts_until_under_the_ceiling(
    tmp_path: Path,
) -> None:
    feedback_dir = tmp_path / "feedback"
    # Seeded out of chronological order: the deletion order must come from the
    # id timestamp, not from directory or insertion order. 999999999999 is the
    # oldest of these *numerically* and the newest lexicographically — a plain
    # filename sort prunes the wrong record.
    ids = {
        ms: _seed_record(feedback_dir, ms, suffix)
        for ms, suffix in (
            (1755712345680, "cccccccc"),
            (1755712345678, "aaaaaaaa"),
            (999999999999, "dddddddd"),
            (1755712345679, "bbbbbbbb"),
        )
    }
    oldest, second = 999999999999, 1755712345678
    context_size = (feedback_dir / feedback_store.context_filename(ids[oldest])).stat().st_size
    ceiling = _store_bytes(feedback_dir) - 2 * context_size + 500

    pruned = feedback_store.prune_store(feedback_dir, ceiling)

    assert pruned == [ids[oldest], ids[second]]
    surviving_contexts = sorted(p.name for p in feedback_dir.glob(feedback_store.CONTEXT_GLOB))
    assert surviving_contexts == sorted(
        feedback_store.context_filename(ids[ms]) for ms in (1755712345679, 1755712345680)
    )
    assert _store_bytes(feedback_dir) <= ceiling


def test_prune_store_keeps_every_header_and_flags_the_pruned_ones(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    old = _seed_record(feedback_dir, 1755712345678, "aaaaaaaa", header={"excerpt": "it broke"})
    new = _seed_record(feedback_dir, 1755712345679, "bbbbbbbb")
    context_size = (feedback_dir / feedback_store.context_filename(old)).stat().st_size
    ceiling = _store_bytes(feedback_dir) - context_size + 500

    assert feedback_store.prune_store(feedback_dir, ceiling) == [old]

    old_header = json.loads((feedback_dir / feedback_store.header_filename(old)).read_text())
    assert old_header["context_pruned"] is True
    # The rest of the record survives verbatim — history is what headers are for.
    assert old_header["id"] == old
    assert old_header["excerpt"] == "it broke"
    assert old_header["context_file"] == feedback_store.context_filename(old)

    new_header = json.loads((feedback_dir / feedback_store.header_filename(new)).read_text())
    assert "context_pruned" not in new_header


def test_prune_store_rewrites_headers_through_the_temp_file_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feedback_dir = tmp_path / "feedback"
    old = _seed_record(feedback_dir, 1755712345678, "aaaaaaaa")
    _seed_record(feedback_dir, 1755712345679, "bbbbbbbb")
    context_size = (feedback_dir / feedback_store.context_filename(old)).stat().st_size
    ceiling = _store_bytes(feedback_dir) - context_size + 500

    real_replace = os.replace
    replacements: list[tuple[str, str]] = []

    def spy_replace(src, dst, *args, **kwargs):  # noqa: ANN001, ANN202 — stdlib passthrough
        replacements.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(feedback_store.os, "replace", spy_replace)

    feedback_store.prune_store(feedback_dir, ceiling)

    assert len(replacements) == 1
    source, destination = replacements[0]
    assert source.startswith(".")
    assert source.endswith(".tmp")
    assert not fnmatch.fnmatch(source, feedback_store.HEADER_GLOB)
    assert destination == feedback_store.header_filename(old)


def test_prune_store_deletes_an_orphan_context_without_writing_a_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feedback_dir = tmp_path / "feedback"
    orphan_id = "fb-1755712345678-aaaaaaaa"
    feedback_dir.mkdir(parents=True)
    (feedback_dir / feedback_store.context_filename(orphan_id)).write_text(
        json.dumps({"id": orphan_id, "scrollback": "x" * 2000})
    )
    kept = _seed_record(feedback_dir, 1755712345679, "bbbbbbbb")
    orphan_size = (feedback_dir / feedback_store.context_filename(orphan_id)).stat().st_size
    ceiling = _store_bytes(feedback_dir) - orphan_size + 500

    def refuse_replace(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("an orphan context has no header to rewrite")

    monkeypatch.setattr(feedback_store.os, "replace", refuse_replace)

    assert feedback_store.prune_store(feedback_dir, ceiling) == [orphan_id]

    assert not (feedback_dir / feedback_store.context_filename(orphan_id)).exists()
    assert not (feedback_dir / feedback_store.header_filename(orphan_id)).exists()
    assert (feedback_dir / feedback_store.context_filename(kept)).is_file()


def test_prune_store_never_deletes_headers_when_nothing_is_left_to_prune(
    tmp_path: Path,
) -> None:
    feedback_dir = tmp_path / "feedback"
    # Every record is already pruned: header-only, no context to reclaim.
    ids = [
        _seed_record(
            feedback_dir,
            1755712345678 + i,
            f"{i:08x}",
            context=False,
            header={"context_pruned": True},
        )
        for i in range(3)
    ]

    assert feedback_store.prune_store(feedback_dir, 10) == []

    assert sorted(p.name for p in feedback_dir.glob(feedback_store.HEADER_GLOB)) == sorted(
        feedback_store.header_filename(i) for i in ids
    )


def test_prune_store_measures_only_record_files(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    record_id = _seed_record(feedback_dir, 1755712345678, "aaaaaaaa")
    ceiling = _store_bytes(feedback_dir) + 10
    # Unrelated debris — an editor scratch file and an in-flight temp file —
    # must not push the store over its ceiling.
    (feedback_dir / "notes.txt").write_text("y" * 100_000)
    (feedback_dir / ".fb-1755712345679-bbbbbbbb.json.xyz.tmp").write_text("z" * 100_000)

    assert feedback_store.prune_store(feedback_dir, ceiling) == []

    assert (feedback_dir / feedback_store.context_filename(record_id)).is_file()


def test_prune_store_tolerates_a_missing_directory(tmp_path: Path) -> None:
    assert feedback_store.prune_store(tmp_path / "never-created", 0) == []


def test_prune_store_leaves_a_damaged_header_untouched(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    record_id = "fb-1755712345678-aaaaaaaa"
    feedback_dir.mkdir(parents=True)
    (feedback_dir / feedback_store.header_filename(record_id)).write_text("{ truncated")
    (feedback_dir / feedback_store.context_filename(record_id)).write_text(
        json.dumps({"id": record_id, "scrollback": "x" * 2000})
    )

    assert feedback_store.prune_store(feedback_dir, 100) == [record_id]

    assert not (feedback_dir / feedback_store.context_filename(record_id)).exists()
    assert (feedback_dir / feedback_store.header_filename(record_id)).read_text() == "{ truncated"
