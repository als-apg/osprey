"""Tests for :func:`calculate_file_checksums`'s binary-suffix skip list.

A ``.duckdb`` file is a derived binary artifact that carries its own content
digest in its ``meta`` table, so checksumming it would turn every
``osprey knowledge build-index`` into apparent render drift.
"""

from pathlib import Path

from osprey.cli.templates.manifest import calculate_file_checksums


def test_duckdb_files_are_skipped(tmp_path: Path):
    project = tmp_path / "proj"
    (project / "data" / "channel_databases").mkdir(parents=True)

    (project / "data" / "channel_databases" / "graph.duckdb").write_bytes(b"duckdb-bytes")
    (project / "data" / "channel_databases" / "build_index.py").write_text("# builder\n")

    checksums = calculate_file_checksums(project)

    assert "data/channel_databases/build_index.py" in checksums
    assert "data/channel_databases/graph.duckdb" not in checksums
