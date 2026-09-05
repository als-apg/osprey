"""Tests for opening a built index read-only, and for refusing to.

Every index here is a real one: the corpora are parsed, the rows derived and
the file written by the builder, so what the reader opens is what a build
produces rather than a hand-made table. The absences are the point of the
module — a deployment meets a missing index far more often than a corrupt one,
and each of the three reasons drives a different sentence on a different
surface, so each is pinned on a file that really is in that state.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest
from tests.services.channel_finder.graph_index import corpora

from osprey.services.channel_finder.graph_index.builder import (
    CALLER_META_KEYS,
    ParsedCorpus,
    build_from_rows,
    channels_from_rows,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.reader import (
    GraphIndex,
    GraphIndexAbsence,
    GraphIndexMeta,
    open_graph_index,
)
from osprey.services.channel_finder.graph_index.schema import META_KEYS, SCHEMA_VERSION

CORPUS_SHA = "a" * 64
CORPUS_FILENAME = "demo_machine.ttl"


def _meta(parsed: ParsedCorpus) -> dict:
    """The ``meta`` mapping the corpus build states for ``parsed``."""
    values = {
        "corpus_sha256": CORPUS_SHA,
        "corpus_filename": CORPUS_FILENAME,
        "binding_count": len(parsed.binding_rows),
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }
    assert set(values) == set(CALLER_META_KEYS)
    return values


def _build(parsed: ParsedCorpus, index_path: Path) -> dict:
    """Write ``parsed`` to ``index_path``; return the ``meta`` it was given."""
    values = _meta(parsed)
    build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels_from_rows(parsed.binding_rows),
        index_path,
        values,
    )
    return values


@pytest.fixture(scope="module")
def parsed() -> ParsedCorpus:
    return parse_corpus(corpora.SUBCLASS_CHAIN)


@pytest.fixture
def index(parsed: ParsedCorpus, tmp_path: Path) -> Path:
    """A freshly built index file. The reader never writes, so this is reusable."""
    path = tmp_path / "graph.duckdb"
    _build(parsed, path)
    return path


class TestOpen:
    """A built index opens and carries the row the build wrote."""

    def test_open_returns_a_graph_index(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        try:
            assert opened.path == index
            assert opened.closed is False
        finally:
            opened.close()

    def test_meta_is_the_row_the_build_wrote(self, index: Path, parsed: ParsedCorpus):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            assert opened.meta == GraphIndexMeta(schema_version=SCHEMA_VERSION, **_meta(parsed))

    def test_meta_carries_every_column_of_the_meta_table(self):
        assert tuple(GraphIndexMeta.__dataclass_fields__) == META_KEYS

    def test_meta_is_frozen(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened, pytest.raises(Exception):
            opened.meta.binding_count = 0  # type: ignore[misc]

    def test_a_cursor_reads_the_rows(self, index: Path, parsed: ParsedCorpus):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            count = opened.cursor().execute("SELECT count(*) FROM bindings").fetchone()
        assert count == (len(parsed.binding_rows),)

    def test_each_call_returns_a_distinct_cursor(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            assert opened.cursor() is not opened.cursor()

    def test_an_empty_corpus_opens_and_reports_zero_bindings(self, tmp_path: Path):
        empty = parse_corpus(corpora.NO_BINDINGS)
        path = tmp_path / "empty.duckdb"
        _build(empty, path)
        opened = open_graph_index(path)
        assert isinstance(opened, GraphIndex)
        with opened:
            assert opened.meta.binding_count == 0


class TestAbsences:
    """Each of the three reasons, on a file that really is in that state."""

    def test_a_path_with_nothing_at_it_is_missing(self, tmp_path: Path):
        path = tmp_path / "not-built-yet.duckdb"
        absence = open_graph_index(path)
        assert isinstance(absence, GraphIndexAbsence)
        assert absence.reason == "missing"
        assert absence.path == path
        assert str(path) in absence.detail

    def test_a_directory_at_the_path_is_unreadable(self, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        path.mkdir()
        absence = open_graph_index(path)
        assert isinstance(absence, GraphIndexAbsence)
        assert absence.reason == "unreadable"
        assert absence.path == path
        assert str(path) in absence.detail

    def test_a_garbage_file_is_unreadable(self, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        path.write_bytes(b"not a duckdb file, just some bytes\n" * 64)
        absence = open_graph_index(path)
        assert isinstance(absence, GraphIndexAbsence)
        assert absence.reason == "unreadable"
        assert str(path) in absence.detail

    def test_a_duckdb_file_without_our_tables_is_unreadable(self, tmp_path: Path):
        path = tmp_path / "someone-elses.duckdb"
        con = duckdb.connect(str(path))
        try:
            con.execute("CREATE TABLE unrelated (x INTEGER)")
        finally:
            con.close()
        absence = open_graph_index(path)
        assert isinstance(absence, GraphIndexAbsence)
        assert absence.reason == "unreadable"
        assert "meta" in absence.detail

    def test_an_index_carrying_no_meta_row_is_unreadable(self, index: Path, tmp_path: Path):
        path = tmp_path / "no-meta.duckdb"
        shutil.copy(index, path)
        con = duckdb.connect(str(path))
        try:
            con.execute("DELETE FROM meta")
        finally:
            con.close()
        absence = open_graph_index(path)
        assert isinstance(absence, GraphIndexAbsence)
        assert absence.reason == "unreadable"
        assert "meta row" in absence.detail

    def test_another_schema_version_is_a_schema_mismatch(self, index: Path, tmp_path: Path):
        path = tmp_path / "stale.duckdb"
        shutil.copy(index, path)
        stale = SCHEMA_VERSION + 1
        con = duckdb.connect(str(path))
        try:
            con.execute("UPDATE meta SET schema_version = ?", [stale])
        finally:
            con.close()

        absence = open_graph_index(path)
        assert isinstance(absence, GraphIndexAbsence)
        assert absence.reason == "schema_mismatch"
        assert absence.path == path
        assert str(stale) in absence.detail
        assert str(SCHEMA_VERSION) in absence.detail

    def test_an_absence_holds_no_connection_open(self, tmp_path: Path):
        """A refused open must not leave the file locked against a rebuild."""
        path = tmp_path / "graph.duckdb"
        con = duckdb.connect(str(path))
        try:
            con.execute("CREATE TABLE unrelated (x INTEGER)")
        finally:
            con.close()

        assert isinstance(open_graph_index(path), GraphIndexAbsence)
        # A read-write connection is refused while a read-only one is still open.
        writable = duckdb.connect(str(path))
        writable.close()


class TestConcurrency:
    """One connection per process, one cursor per call, threads in parallel."""

    def test_two_cursors_serve_two_threads(self, index: Path, parsed: ParsedCorpus):
        expected = (len(parsed.binding_rows),)
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            cursors = [opened.cursor(), opened.cursor()]

            def scan(cursor) -> list:
                return [
                    cursor.execute("SELECT count(*) FROM bindings").fetchone() for _ in range(20)
                ]

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(scan, cursors))

        assert results == [[expected] * 20, [expected] * 20]

    def test_many_threads_each_take_their_own_cursor(self, index: Path, parsed: ParsedCorpus):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:

            def scan(_: int):
                return opened.cursor().execute("SELECT count(*) FROM bindings").fetchone()

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(scan, range(24)))

        assert results == [(len(parsed.binding_rows),)] * 24


class TestReadOnly:
    """The handle is read-only: a write through it is refused by the driver."""

    def test_an_insert_through_a_cursor_fails(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            cursor = opened.cursor()
            with pytest.raises(duckdb.Error):
                cursor.execute("INSERT INTO channels VALUES ('SR:NEW', 'read', NULL)")

    def test_the_rows_are_unchanged_after_a_refused_write(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            before = opened.cursor().execute("SELECT count(*) FROM channels").fetchone()
            with pytest.raises(duckdb.Error):
                opened.cursor().execute("DELETE FROM channels")
            assert opened.cursor().execute("SELECT count(*) FROM channels").fetchone() == before


class TestClose:
    """Closing releases the file and every later use says so."""

    def test_cursor_after_close_raises_cleanly(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        opened.close()
        assert opened.closed is True
        with pytest.raises(RuntimeError, match="closed"):
            opened.cursor()

    def test_close_is_idempotent(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        opened.close()
        opened.close()
        assert opened.closed is True

    def test_the_context_manager_closes_on_the_way_out(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with opened:
            assert opened.closed is False
        assert opened.closed is True

    def test_the_context_manager_closes_after_an_exception(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        with pytest.raises(ZeroDivisionError), opened:
            raise ZeroDivisionError
        assert opened.closed is True

    def test_meta_and_path_survive_close(self, index: Path, parsed: ParsedCorpus):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        opened.close()
        assert opened.path == index
        assert opened.meta.binding_count == len(parsed.binding_rows)

    def test_close_releases_the_file_for_a_rebuild(self, index: Path):
        opened = open_graph_index(index)
        assert isinstance(opened, GraphIndex)
        opened.close()
        writable = duckdb.connect(str(index))
        writable.close()


class TestImportIsolation:
    """``duckdb`` must not be imported by importing the reader."""

    def test_importing_the_reader_does_not_import_duckdb(self):
        script = textwrap.dedent(
            """
            import sys
            from pathlib import Path

            import osprey.services.channel_finder.graph_index.reader as reader

            forbidden = sorted(
                name
                for name in ("duckdb", "rdflib", "neo4j", "osprey.services.qmd")
                if name in sys.modules
            )
            assert not forbidden, forbidden
            assert callable(reader.open_graph_index)
            assert reader.GraphIndexAbsence("missing", Path("x"), "detail").reason == "missing"
            assert "duckdb" not in sys.modules
            """
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_a_missing_index_is_answered_without_importing_duckdb(self):
        """The roster and the health check ask about paths that do not exist."""
        script = textwrap.dedent(
            """
            import sys
            from pathlib import Path

            from osprey.services.channel_finder.graph_index import open_graph_index

            absence = open_graph_index(Path("/nonexistent/graph.duckdb"))
            assert absence.reason == "missing", absence
            assert "duckdb" not in sys.modules
            """
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_the_package_exports_the_reader_lazily(self):
        from osprey.services.channel_finder import graph_index

        assert graph_index.open_graph_index is open_graph_index
        assert graph_index.GraphIndex is GraphIndex
        assert graph_index.GraphIndexAbsence is GraphIndexAbsence
        assert graph_index.GraphIndexMeta is GraphIndexMeta
