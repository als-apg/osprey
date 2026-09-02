"""Tests for the shared atomic-JSON helper the web-terminal stores are built on.

Covers the two guarantees every consumer leans on:

* a write is either wholly visible or not visible at all — a reader racing a
  rewrite sees the previous document, never a truncation, and a failed write
  leaves neither a damaged target nor temp-file debris,
* a read never raises: a missing, unreadable, unparseable or non-object
  document all degrade to ``None`` so the caller can fall back to its default
  instead of taking the page down.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import tempfile
from pathlib import Path

import pytest

from osprey.interfaces.web_terminal import _json_store

LOGGER_NAME = "osprey.interfaces.web_terminal._json_store"


# ── write_json_atomic ──────────────────────────────────────────────────────


def test_write_json_atomic_serializes_the_document(tmp_path: Path) -> None:
    target = tmp_path / "doc.json"

    _json_store.write_json_atomic(target, {"b": 1, "a": ["x", None, True]})

    assert json.loads(target.read_text()) == {"b": 1, "a": ["x", None, True]}
    # Indented, so an operator reading the file out of a volume can follow it.
    assert "\n" in target.read_text()


def test_write_json_atomic_stringifies_what_json_cannot_encode(tmp_path: Path) -> None:
    target = tmp_path / "doc.json"

    _json_store.write_json_atomic(target, {"path": Path("/var/agent_data")})

    assert json.loads(target.read_text()) == {"path": "/var/agent_data"}


def test_write_json_atomic_lands_by_replacing_a_hidden_sibling_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "prefs.json"
    real_mkstemp = tempfile.mkstemp
    temp_paths: list[Path] = []

    def spy_mkstemp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 — stdlib passthrough
        fd, name = real_mkstemp(*args, **kwargs)
        temp_paths.append(Path(name))
        return fd, name

    real_replace = os.replace
    replacements: list[tuple[str, str]] = []

    def spy_replace(src, dst, *args, **kwargs):  # noqa: ANN001, ANN202 — stdlib passthrough
        replacements.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)
    monkeypatch.setattr(os, "replace", spy_replace)

    _json_store.write_json_atomic(target, {"ok": True})

    assert len(temp_paths) == 1
    # Same directory, so os.replace stays within one filesystem and is atomic.
    assert temp_paths[0].parent == target.parent
    # Hidden and .tmp-suffixed: the globs a store reads itself with are written
    # against committed names and must never match a document mid-write.
    assert temp_paths[0].name.startswith(".")
    assert temp_paths[0].name.endswith(".tmp")
    assert not fnmatch.fnmatch(temp_paths[0].name, "*.json")
    assert replacements == [(temp_paths[0].name, target.name)]
    assert not temp_paths[0].exists()


def test_write_json_atomic_leaves_the_previous_document_readable_until_the_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "prefs.json"
    _json_store.write_json_atomic(target, {"generation": 1})
    real_replace = os.replace
    seen_mid_write: list[dict] = []

    def spy_replace(src, dst, *args, **kwargs):  # noqa: ANN001, ANN202 — stdlib passthrough
        # A concurrent reader arriving here reads the whole previous document.
        seen_mid_write.append(json.loads(Path(dst).read_text()))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy_replace)

    _json_store.write_json_atomic(target, {"generation": 2})

    assert seen_mid_write == [{"generation": 1}]
    assert json.loads(target.read_text()) == {"generation": 2}


def test_write_json_atomic_removes_the_temp_file_and_reraises_when_serialization_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "prefs.json"
    _json_store.write_json_atomic(target, {"generation": 1})

    with pytest.raises(TypeError):
        _json_store.write_json_atomic(target, {"ok": 1, ("bad", "key"): 2})

    # No debris, and the document that was already there is untouched.
    assert [entry.name for entry in tmp_path.iterdir()] == ["prefs.json"]
    assert json.loads(target.read_text()) == {"generation": 1}


def test_write_json_atomic_survives_a_filesystem_that_cannot_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "prefs.json"

    def refuse_fsync(_fd: int) -> None:
        raise OSError("fsync unsupported on this mount")

    monkeypatch.setattr(os, "fsync", refuse_fsync)

    _json_store.write_json_atomic(target, {"ok": True})

    assert json.loads(target.read_text()) == {"ok": True}


def test_write_json_atomic_requires_the_parent_directory_to_exist(tmp_path: Path) -> None:
    # Documented contract: creating the directory belongs to the store, which
    # knows whether an absent directory is a first write or a broken mount.
    with pytest.raises(OSError):
        _json_store.write_json_atomic(tmp_path / "never-created" / "prefs.json", {"ok": True})


# ── read_json_object ───────────────────────────────────────────────────────


def test_read_json_object_returns_the_document(tmp_path: Path) -> None:
    target = tmp_path / "prefs.json"
    target.write_text(json.dumps({"items": ["clock"], "version": 1}))

    assert _json_store.read_json_object(target) == {"items": ["clock"], "version": 1}


def test_read_json_object_round_trips_a_written_document(tmp_path: Path) -> None:
    target = tmp_path / "prefs.json"
    document = {"version": 1, "items": [{"id": "clock", "locked": True}]}

    _json_store.write_json_atomic(target, document)

    assert _json_store.read_json_object(target) == document


def test_read_json_object_reports_an_absent_document_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # An unused store is a normal state, not damage: no log line for it.
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert _json_store.read_json_object(tmp_path / "never-written.json") is None

    assert caplog.records == []


def test_read_json_object_reports_a_truncated_document_as_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "prefs.json"
    target.write_text("{ truncated")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert _json_store.read_json_object(target) is None

    assert "prefs.json" in caplog.text


def test_read_json_object_rejects_a_document_that_is_not_an_object(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    array = tmp_path / "array.json"
    array.write_text(json.dumps(["not", "an", "object"]))
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps(7))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert _json_store.read_json_object(array) is None
        assert _json_store.read_json_object(scalar) is None

    assert "array.json" in caplog.text
    assert "scalar.json" in caplog.text


def test_read_json_object_reports_an_unreadable_path_as_absent(tmp_path: Path) -> None:
    directory = tmp_path / "prefs.json"
    directory.mkdir()

    assert _json_store.read_json_object(directory) is None


def test_read_json_object_never_raises_on_the_failure_modes_it_documents(
    tmp_path: Path,
) -> None:
    # One assertion for the contract itself: every degraded input is None, and
    # the caller can therefore fall back to its default without a try block.
    (tmp_path / "empty.json").write_text("")
    (tmp_path / "null.json").write_text("null")
    (tmp_path / "text.json").write_text("just some text")

    assert [
        _json_store.read_json_object(tmp_path / name)
        for name in ("empty.json", "null.json", "text.json", "missing.json")
    ] == [None, None, None, None]
