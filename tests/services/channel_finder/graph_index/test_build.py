"""Tests for the whole build: a Turtle corpus in, a DuckDB index file out.

The pieces below it have their own lanes -- ``test_parse_corpus.py`` for the
rows, ``test_channels.py`` for the roster, ``test_writer.py`` for the file. What
is only true of the entry point is pinned here: that the census it states is the
one the seeded store would count, that the digest it stamps is the seeder's own
so a store and an index filled from one corpus agree they hold one corpus, and
that a corpus which is missing and a corpus which is malformed arrive as
different exceptions.

Every count is read back through a fresh read-only connection, the way the
reader and the health check will open the file.
"""

from __future__ import annotations

import hashlib
import logging
import time
from importlib.resources import as_file, files
from pathlib import Path

import duckdb
import pytest
from tests.services.channel_finder.graph_index import corpora

from osprey.channel_roster import RosterSource, RosterSourceKind
from osprey.channel_roster.graph import read_graph_roster
from osprey.services.channel_finder.core.exceptions import GraphIndexBuildError
from osprey.services.channel_finder.graph_index.builder import (
    IndexBuildReport,
    build_graph_index,
    channels_from_corpus,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.schema import META_KEYS, SCHEMA_VERSION
from osprey.services.channel_finder.graph_index.taxonomy import prune_device_taxonomy
from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

#: The shipped demo corpus, as the store counts its populations. ``bindings``,
#: ``devices`` and ``classes`` are the numbers ``test_parse_corpus.py`` pins
#: against the parity lane; ``signals`` is what ``GRAPH_SIGNAL_COUNT_CYPHER``
#: counts, and ``sections`` what ``GRAPH_SECTION_COUNT_CYPHER`` does.
DEMO_BINDINGS = 2908
DEMO_DEVICES = 512
DEMO_CLASSES = 19
DEMO_SIGNALS = 113

#: Derived from the corpus (three distinct ``narad_p:sectionCode`` literals),
#: not read off a store-backed assertion -- no census test seeds this corpus and
#: asks the store for its section count. It is corroborated by the store-backed
#: search facet in ``tests/integration/test_graph_mcp.py``, which asserts the
#: seeded demo store answers ``{"SR", "BR", "BTS"}`` for the section facet.
DEMO_SECTIONS = 3

#: The whole build -- read, parse, derive, write -- for the shipped corpus.
BUILD_BUDGET_SECONDS = 5.0


@pytest.fixture(scope="module")
def demo_path():
    """The packaged demo corpus, materialised on disk."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield path


def _read(index_path: Path, sql: str) -> list[tuple]:
    """Run ``sql`` against a fresh read-only connection, as the reader will."""
    con = duckdb.connect(str(index_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _meta_row(index_path: Path) -> dict:
    """The written ``meta`` row, keyed by column name."""
    row = _read(index_path, f"SELECT {', '.join(META_KEYS)} FROM meta")
    assert len(row) == 1, row
    return dict(zip(META_KEYS, row[0], strict=True))


def _write(path: Path, text: str, *, crlf: bool = False) -> Path:
    """Write ``text`` byte-for-byte, optionally with CRLF line endings."""
    payload = text.replace("\n", "\r\n") if crlf else text
    path.write_bytes(payload.encode("utf-8"))
    return path


class TestDemoCorpus:
    """The shipped corpus, built end to end."""

    @pytest.fixture(scope="class")
    def built(self, demo_path: Path, tmp_path_factory) -> tuple[IndexBuildReport, Path, float]:
        index_path = tmp_path_factory.mktemp("demo") / "graph.duckdb"
        started = time.perf_counter()
        report = build_graph_index(demo_path, index_path)
        return report, index_path, time.perf_counter() - started

    def test_the_report_states_the_store_s_census(self, built, demo_path: Path):
        report, index_path, _ = built

        assert report.path == index_path
        assert report.binding_count == DEMO_BINDINGS
        assert report.device_count == DEMO_DEVICES
        assert report.class_count == DEMO_CLASSES
        assert report.signal_count == DEMO_SIGNALS
        assert report.section_count == DEMO_SECTIONS
        assert report.corpus_sha256 == ttl_sha256(demo_path.read_text(encoding="utf-8"))

    def test_the_meta_row_carries_the_same_census(self, built, demo_path: Path):
        _, index_path, _ = built

        assert _meta_row(index_path) == {
            "schema_version": SCHEMA_VERSION,
            "corpus_sha256": ttl_sha256(demo_path.read_text(encoding="utf-8")),
            "corpus_filename": "demo_machine.ttl",
            "binding_count": DEMO_BINDINGS,
            "device_count": DEMO_DEVICES,
            "class_count": DEMO_CLASSES,
            "signal_count": DEMO_SIGNALS,
            "section_count": DEMO_SECTIONS,
        }

    def test_the_tables_hold_the_rows_the_report_counted(self, built):
        report, index_path, _ = built

        assert _read(index_path, "SELECT count(*) FROM bindings") == [(DEMO_BINDINGS,)]
        assert _read(index_path, "SELECT count(*) FROM classes") == [(DEMO_CLASSES,)]
        assert _read(index_path, "SELECT count(*) FROM channels") == [(report.channel_count,)]
        assert _read(index_path, "SELECT count(DISTINCT device_uri) FROM bindings") == [
            (DEMO_DEVICES,)
        ]

    def test_the_device_count_is_the_bound_subjects_the_census_counts(self, built, demo_path: Path):
        """``GRAPH_DEVICE_COUNT_CYPHER``: a DISTINCT ``:Resource`` with a binding."""
        report, _, _ = built
        parsed = parse_corpus(demo_path.read_text(encoding="utf-8"))

        assert report.device_count == len({row.device_uri for row in parsed.binding_rows})

    def test_the_class_count_is_the_pruned_taxonomy(self, built, demo_path: Path):
        """The classes written are the ones the explorer draws, not every ``:Class``."""
        report, _, _ = built
        parsed = parse_corpus(demo_path.read_text(encoding="utf-8"))
        raw = [
            {
                "uri": row.uri,
                "altLabel": row.alt_labels,
                "parents": row.parents,
                "rollup": row.rollup_devices,
                "direct": row.direct_devices,
            }
            for row in parsed.class_rows
        ]

        assert report.class_count == len(parsed.class_rows)
        assert report.class_count == len(prune_device_taxonomy(raw))

    def test_the_channels_table_is_the_roster_the_reader_answers(
        self, built, demo_path: Path
    ) -> None:
        """The rows the build wrote are the records the roster hands back.

        The oracle is the CORPUS, not the file: the channels the derivation
        rules read out of the same Turtle are what a consumer has to get back
        out of the index, in that order, with each direction and readback
        intact. Comparing the reader against a query over the file it just read
        would pass on any index the writer and the reader agreed to truncate
        together. The census the report states is asserted alongside it.
        """
        report, index_path, _ = built
        expected = channels_from_corpus(
            parse_corpus(demo_path.read_text(encoding="utf-8")),
            RosterSource(kind=RosterSourceKind.GRAPH, path=demo_path),
        )

        records = read_graph_roster(
            RosterSource(kind=RosterSourceKind.GRAPH, path=index_path)
        ).records

        assert [(r.address, r.direction, r.readback) for r in records] == [
            (row.address, row.direction, row.readback) for row in expected
        ]
        assert len(records) == report.channel_count == DEMO_BINDINGS

    def test_the_whole_build_stays_inside_its_budget(self, built):
        _, _, elapsed = built

        logging.getLogger(__name__).info(
            "build_graph_index over the demo corpus took %.2f s", elapsed
        )
        assert elapsed < BUILD_BUDGET_SECONDS, (
            f"building the demo index took {elapsed:.2f} s, budget {BUILD_BUDGET_SECONDS} s"
        )

    def test_the_build_logs_one_line_naming_its_counts(
        self, demo_path: Path, tmp_path: Path, caplog
    ):
        """One DEBUG line with the counts and the timing; nothing at INFO.

        The callers own the operator-facing sentence (the build's progress
        line, the ``build-index`` verb's summary), and a build keeps absolute
        paths out of its INFO view, so the builder's own line stays at DEBUG.
        """
        builder_logger = "osprey.services.channel_finder.graph_index.builder"
        with caplog.at_level(logging.DEBUG, logger=builder_logger):
            build_graph_index(demo_path, tmp_path / "graph.duckdb")

        records = [record for record in caplog.records if record.name == builder_logger]
        assert [record.levelno for record in records] == [logging.DEBUG], records
        line = records[0].getMessage()
        assert "2908 bindings" in line
        assert "512 devices" in line
        assert " s: " in line, line

    def test_the_package_exports_the_entry_point_lazily(self):
        from osprey.services.channel_finder import graph_index

        assert graph_index.build_graph_index is build_graph_index


class TestCorpusDigest:
    """The digest is the seeder's, over the text ``read_text`` returns."""

    def test_a_crlf_corpus_hashes_as_the_text_not_as_the_bytes(self, tmp_path: Path):
        path = _write(tmp_path / "crlf.ttl", corpora.SUBCLASS_CHAIN, crlf=True)
        raw = path.read_bytes()
        assert b"\r\n" in raw, "the fixture must actually be CRLF on disk"

        report = build_graph_index(path, tmp_path / "graph.duckdb")

        assert report.corpus_sha256 == ttl_sha256(path.read_text(encoding="utf-8"))
        assert report.corpus_sha256 != hashlib.sha256(raw).hexdigest()

    def test_the_lf_and_crlf_copies_of_one_corpus_agree(self, tmp_path: Path):
        lf = _write(tmp_path / "lf.ttl", corpora.SUBCLASS_CHAIN)
        crlf = _write(tmp_path / "crlf.ttl", corpora.SUBCLASS_CHAIN, crlf=True)
        assert lf.read_bytes() != crlf.read_bytes()

        lf_report = build_graph_index(lf, tmp_path / "lf.duckdb")
        crlf_report = build_graph_index(crlf, tmp_path / "crlf.duckdb")

        assert lf_report.corpus_sha256 == crlf_report.corpus_sha256
        assert lf_report.binding_count == crlf_report.binding_count
        assert _meta_row(tmp_path / "lf.duckdb")["corpus_sha256"] == crlf_report.corpus_sha256

    def test_the_meta_row_names_the_corpus_file(self, tmp_path: Path):
        path = _write(tmp_path / "facility.ttl", corpora.SUBCLASS_CHAIN)

        build_graph_index(path, tmp_path / "graph.duckdb")

        assert _meta_row(tmp_path / "graph.duckdb")["corpus_filename"] == "facility.ttl"


class TestFailures:
    """A corpus that is not there and one that is broken are different faults."""

    def test_a_missing_corpus_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            build_graph_index(tmp_path / "absent.ttl", tmp_path / "graph.duckdb")

        assert not (tmp_path / "graph.duckdb").exists()

    def test_invalid_turtle_raises_a_build_error(self, tmp_path: Path):
        path = _write(tmp_path / "broken.ttl", corpora.INVALID_TURTLE)

        with pytest.raises(GraphIndexBuildError, match="not valid Turtle"):
            build_graph_index(path, tmp_path / "graph.duckdb")

        assert not (tmp_path / "graph.duckdb").exists()

    def test_a_missing_index_directory_raises_a_build_error(self, tmp_path: Path):
        path = _write(tmp_path / "corpus.ttl", corpora.SUBCLASS_CHAIN)

        with pytest.raises(GraphIndexBuildError, match="does not exist"):
            build_graph_index(path, tmp_path / "nowhere" / "graph.duckdb")

    def test_a_failed_build_leaves_no_temporary_behind(self, tmp_path: Path):
        path = _write(tmp_path / "broken.ttl", corpora.INVALID_TURTLE)

        with pytest.raises(GraphIndexBuildError):
            build_graph_index(path, tmp_path / "graph.duckdb")

        assert [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name] == []


class TestCorpusWithoutBindings:
    """An unseeded corpus is a staging gap, and the index says so out loud."""

    def test_it_writes_an_empty_index_and_warns(self, tmp_path: Path, caplog):
        path = _write(tmp_path / "empty.ttl", corpora.NO_BINDINGS)
        index_path = tmp_path / "graph.duckdb"

        with caplog.at_level(logging.WARNING):
            report = build_graph_index(path, index_path)

        assert index_path.exists()
        assert report.binding_count == 0
        assert report.device_count == 0
        assert report.channel_count == 0
        assert _read(index_path, "SELECT count(*) FROM bindings") == [(0,)]
        assert any("bound no channels" in record.getMessage() for record in caplog.records)

    def test_the_meta_row_is_still_written(self, tmp_path: Path):
        path = _write(tmp_path / "empty.ttl", corpora.NO_BINDINGS)

        build_graph_index(path, tmp_path / "graph.duckdb")

        meta = _meta_row(tmp_path / "graph.duckdb")
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["binding_count"] == 0
        assert meta["corpus_sha256"] == ttl_sha256(path.read_text(encoding="utf-8"))
