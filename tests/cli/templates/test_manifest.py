"""Tests for :func:`calculate_file_checksums`: what it skips and what it need not read.

A ``.duckdb`` file is a derived binary artifact that carries its own content
digest in its ``meta`` table, so checksumming it would turn every rebuilt
index into apparent render drift.

A file whose writer noted its digest (:func:`osprey.build.manifest.note_sha256`)
is listed with that digest while it is as the writer left it, and read otherwise.
"""

import hashlib
import os
from pathlib import Path

import pytest

from osprey.build import manifest as build_manifest
from osprey.build.manifest import note_sha256, noted_sha256
from osprey.cli.templates.manifest import calculate_file_checksums


def test_duckdb_files_are_skipped(tmp_path: Path):
    project = tmp_path / "proj"
    (project / "data" / "channel_databases").mkdir(parents=True)

    (project / "data" / "channel_databases" / "graph.duckdb").write_bytes(b"duckdb-bytes")
    (project / "data" / "channel_databases" / "build_index.py").write_text("# builder\n")

    checksums = calculate_file_checksums(project)

    assert "data/channel_databases/build_index.py" in checksums
    assert "data/channel_databases/graph.duckdb" not in checksums


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_noted_file_is_listed_without_being_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    (project / "data").mkdir(parents=True)
    noted = project / "data" / "view.json"
    noted.write_bytes(b"view")
    note_sha256(noted, _sha256(b"view"))
    (project / "config.yml").write_bytes(b"config")
    read: list[str] = []

    def reading(path: Path) -> str:
        read.append(Path(path).name)
        return _sha256(Path(path).read_bytes())

    monkeypatch.setattr("osprey.cli.templates.manifest.sha256_file", reading)
    checksums = calculate_file_checksums(project)

    assert read == ["config.yml"]
    assert checksums == {
        "data/view.json": f"sha256:{_sha256(b'view')}",
        "config.yml": f"sha256:{_sha256(b'config')}",
    }


def test_a_noted_file_changed_since_is_read(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    resized, retimed, gone = project / "resized", project / "retimed", project / "gone"
    for path in (resized, retimed, gone):
        path.write_bytes(b"first")
        note_sha256(path, _sha256(b"first"))
    resized.write_bytes(b"longer than it was")
    stat = retimed.stat()
    retimed.write_bytes(b"other")
    os.utime(retimed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    gone.unlink()

    assert noted_sha256(resized) is None
    assert noted_sha256(retimed) is None
    assert noted_sha256(gone) is None
    assert calculate_file_checksums(project) == {
        "resized": f"sha256:{_sha256(b'longer than it was')}",
        "retimed": f"sha256:{_sha256(b'other')}",
    }


def test_a_file_nobody_noted_has_no_noted_digest(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"bytes")

    assert noted_sha256(path) is None
    assert str(path) not in build_manifest._noted
