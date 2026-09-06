"""Tests for the writer that turns derived rows into a DuckDB index file.

Every assertion here reads the file back through a *fresh* read-only
connection, the way the reader and the health check will open it: what the
writing connection believed is not evidence, and a column order or a NULL that
only holds inside the build is a bug the explorer would find first.

The atomicity pins are the reason this module exists at all. An index is
rebuilt in place under a running deployment, so a failed build must leave the
previous file byte-identical and no temporary file behind.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import duckdb
import pytest
from tests.services.channel_finder.graph_index import corpora

from osprey.services.channel_finder.core.exceptions import GraphIndexBuildError
from osprey.services.channel_finder.graph_index import schema as schema_module
from osprey.services.channel_finder.graph_index.builder import (
    CALLER_META_KEYS,
    BindingRow,
    ChannelRow,
    ClassRow,
    IndexBuildReport,
    ParsedCorpus,
    build_from_rows,
    channels_from_rows,
    parse_corpus,
)
from osprey.services.channel_finder.graph_index.schema import META_KEYS, SCHEMA_VERSION


def _meta(parsed: ParsedCorpus, channels: list[ChannelRow], **overrides) -> dict:
    """A ``meta`` mapping for ``parsed``, as the corpus build will state it."""
    values = {
        "corpus_sha256": "0" * 64,
        "corpus_filename": "corpus.ttl",
        "binding_count": len(parsed.binding_rows),
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }
    values.update(overrides)
    return values


def _build(parsed: ParsedCorpus, index_path: Path, **overrides) -> IndexBuildReport:
    """Write ``parsed`` to ``index_path`` with a consistent ``meta`` row."""
    channels = channels_from_rows(parsed.binding_rows)
    return build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels,
        index_path,
        _meta(parsed, channels, **overrides),
    )


def _read(index_path: Path, sql: str) -> list[tuple]:
    """Run ``sql`` against a fresh read-only connection, as the reader will."""
    con = duckdb.connect(str(index_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _temp_files(directory: Path) -> list[Path]:
    """Every build temporary left in ``directory`` — none may survive a build."""
    return sorted(p for p in directory.iterdir() if ".tmp-" in p.name)


@pytest.fixture
def chain() -> ParsedCorpus:
    return parse_corpus(corpora.SUBCLASS_CHAIN)


class TestRoundTrip:
    """The rows go in and come back out unchanged, in column order."""

    @pytest.fixture
    def index_path(self, chain: ParsedCorpus, tmp_path: Path) -> Path:
        path = tmp_path / "graph.duckdb"
        _build(chain, path)
        return path

    def test_the_file_exists_and_carries_the_four_tables(self, index_path: Path):
        con = duckdb.connect(str(index_path), read_only=True)
        try:
            names = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        finally:
            con.close()
        assert names == {"bindings", "classes", "channels", "meta"}

    def test_binding_rows_round_trip_in_column_order(self, index_path: Path, chain: ParsedCorpus):
        read_back = _read(index_path, "SELECT * FROM bindings")
        expected = [
            (
                row.binding_uri,
                row.full_pv,
                row.description,
                row.device_uri,
                row.device_name,
                row.section,
                row.system,
                row.edges,
                row.signal_uris,
                row.signal_names,
                row.class_uris,
                row.haystack,
            )
            for row in chain.binding_rows
        ]
        assert read_back == expected

    def test_rows_are_written_in_the_order_given(self, index_path: Path, chain: ParsedCorpus):
        # parse_corpus sorts by (full_pv, device_uri, binding_uri); the writer
        # must not reorder, so the reader's ORDER BY is the only sort in play.
        read_back = _read(index_path, "SELECT full_pv FROM bindings")
        assert [pv for (pv,) in read_back] == [row.full_pv for row in chain.binding_rows]

    def test_list_columns_survive_as_lists(self, index_path: Path):
        rows = _read(
            index_path,
            "SELECT edges, signal_uris, signal_names, class_uris FROM bindings "
            "WHERE full_pv = 'SR:MAG:QF1:CURRENT:SP'",
        )
        edges, signal_uris, signal_names, class_uris = rows[0]
        assert edges == ["WRITESSIGNAL"]
        assert signal_names == ["quad_current_sp"]
        assert isinstance(signal_uris, list) and len(signal_uris) == 1
        assert class_uris == sorted(class_uris) and len(class_uris) == 3

    def test_an_empty_list_column_is_an_empty_list_not_null(self, index_path: Path):
        rows = _read(
            index_path,
            "SELECT edges, signal_uris FROM bindings WHERE full_pv = 'SR:MAG:QF1:NOTE'",
        )
        assert rows == [([], [])]

    def test_absent_scalars_are_null(self, tmp_path: Path):
        parsed = parse_corpus(corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM)
        path = tmp_path / "graph.duckdb"
        _build(parsed, path)
        assert _read(path, "SELECT device_name, section, system FROM bindings") == [
            (None, None, None)
        ]

    def test_class_rows_round_trip_with_their_counts(self, index_path: Path, chain: ParsedCorpus):
        read_back = _read(index_path, "SELECT * FROM classes")
        assert read_back == [
            (
                row.uri,
                row.name,
                row.alt_labels,
                row.parents,
                row.direct_devices,
                row.rollup_devices,
            )
            for row in chain.class_rows
        ]

    def test_channel_rows_round_trip(self, index_path: Path):
        assert _read(index_path, "SELECT * FROM channels ORDER BY address") == [
            ("SR:MAG:QF1:CURRENT:RB", "read", None),
            ("SR:MAG:QF1:CURRENT:SP", "write", None),
            ("SR:MAG:QF1:NOTE", None, None),
        ]

    def test_the_meta_row_is_exact(self, index_path: Path, chain: ParsedCorpus):
        row = _read(index_path, f"SELECT {', '.join(META_KEYS)} FROM meta")
        assert row == [
            (
                SCHEMA_VERSION,
                "0" * 64,
                "corpus.ttl",
                len(chain.binding_rows),
                1,
                len(chain.class_rows),
                chain.signal_count,
                1,
            )
        ]

    def test_exactly_one_meta_row(self, index_path: Path):
        assert _read(index_path, "SELECT count(*) FROM meta") == [(1,)]

    def test_no_temporary_file_survives(self, index_path: Path):
        assert _temp_files(index_path.parent) == []

    def test_the_index_opens_read_only_twice(self, index_path: Path):
        # The reader keeps one read-only connection per process; a lock left by
        # the build would make the second process fail.
        for _ in range(2):
            assert _read(index_path, "SELECT count(*) FROM bindings") == [(3,)]


class TestReadbackNull:
    def test_an_empty_readback_is_stored_as_null(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        channels = [ChannelRow(address="A", direction=None, readback="")]
        build_from_rows(
            chain.binding_rows, chain.class_rows, channels, path, _meta(chain, channels)
        )
        assert _read(path, "SELECT readback FROM channels") == [(None,)]
        assert _read(path, "SELECT count(*) FROM channels WHERE readback IS NULL") == [(1,)]

    def test_a_stated_readback_is_kept(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        channels = [ChannelRow(address="A:SP", direction="write", readback="A:RB")]
        build_from_rows(
            chain.binding_rows, chain.class_rows, channels, path, _meta(chain, channels)
        )
        assert _read(path, "SELECT * FROM channels") == [("A:SP", "write", "A:RB")]


class TestReport:
    def test_fields_and_values(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        report = _build(chain, path, corpus_sha256="a" * 64, corpus_filename="demo.ttl")
        assert report == IndexBuildReport(
            path=path,
            corpus_sha256="a" * 64,
            binding_count=3,
            device_count=1,
            class_count=len(chain.class_rows),
            signal_count=chain.signal_count,
            section_count=1,
            channel_count=3,
        )

    def test_the_counts_are_what_the_meta_row_carries(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        report = _build(chain, path)
        stated = _read(
            path,
            "SELECT binding_count, device_count, class_count, signal_count, section_count "
            "FROM meta",
        )[0]
        assert stated == (
            report.binding_count,
            report.device_count,
            report.class_count,
            report.signal_count,
            report.section_count,
        )

    def test_path_is_the_target_not_the_temporary(self, chain: ParsedCorpus, tmp_path: Path):
        report = _build(chain, tmp_path / "graph.duckdb")
        assert report.path == tmp_path / "graph.duckdb"
        assert report.path.exists()


class TestEmptyCorpus:
    @pytest.fixture
    def parsed(self) -> ParsedCorpus:
        return parse_corpus(corpora.NO_BINDINGS)

    def test_the_index_is_still_written(self, parsed: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        _build(parsed, path)
        assert path.exists()
        assert _read(path, "SELECT count(*) FROM bindings") == [(0,)]

    def test_the_meta_row_says_zero_bindings(self, parsed: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        _build(parsed, path)
        assert _read(path, "SELECT binding_count FROM meta") == [(0,)]

    def test_it_warns(self, parsed: ParsedCorpus, tmp_path: Path, caplog):
        path = tmp_path / "graph.duckdb"
        with caplog.at_level("WARNING"):
            _build(parsed, path)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert str(path) in warnings[0].getMessage()

    def test_a_corpus_with_bindings_does_not_warn(
        self, chain: ParsedCorpus, tmp_path: Path, caplog
    ):
        with caplog.at_level("WARNING"):
            _build(chain, tmp_path / "graph.duckdb")
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []


@dataclass(slots=True)
class _ShortRow:
    """A row of the wrong width, to fail an insert mid-build."""

    binding_uri: str


class TestAtomicity:
    """A failed build leaves the previous index untouched and no temp file."""

    @pytest.fixture
    def existing(self, chain: ParsedCorpus, tmp_path: Path) -> tuple[Path, bytes]:
        path = tmp_path / "graph.duckdb"
        _build(chain, path)
        return path, path.read_bytes()

    def test_a_row_of_the_wrong_width_leaves_the_old_index_byte_identical(
        self, existing: tuple[Path, bytes], chain: ParsedCorpus
    ):
        path, before = existing
        channels = channels_from_rows(chain.binding_rows)
        rows = [*chain.binding_rows, _ShortRow("https://example.org/binding/short")]
        # The columnar load would pad a short row out with nulls, so the writer
        # counts the values itself and names the row rather than letting one
        # through: the error is the build's, not the database's.
        with pytest.raises(GraphIndexBuildError, match="Row 3 of bindings carries 1 values"):
            build_from_rows(rows, chain.class_rows, channels, path, _meta(chain, channels))
        assert path.read_bytes() == before
        assert _temp_files(path.parent) == []

    def test_a_failing_replace_leaves_the_old_index_and_removes_the_temporary(
        self, existing: tuple[Path, bytes], chain: ParsedCorpus, monkeypatch
    ):
        path, before = existing

        def boom(src, dst):
            raise OSError("replace refused")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="replace refused"):
            _build(chain, path)
        assert path.read_bytes() == before
        assert _temp_files(path.parent) == []

    def test_a_bad_meta_row_writes_nothing_at_all(self, tmp_path: Path, chain: ParsedCorpus):
        path = tmp_path / "graph.duckdb"
        channels = channels_from_rows(chain.binding_rows)
        with pytest.raises(GraphIndexBuildError):
            build_from_rows(chain.binding_rows, chain.class_rows, channels, path, {})
        assert not path.exists()
        assert _temp_files(tmp_path) == []

    def test_a_stale_temporary_from_a_killed_build_does_not_block_the_next_one(
        self, tmp_path: Path, chain: ParsedCorpus
    ):
        path = tmp_path / "graph.duckdb"
        stale = tmp_path / f"graph.duckdb.tmp-{os.getpid()}"
        stale.write_bytes(b"not a database")
        _build(chain, path)
        assert _read(path, "SELECT count(*) FROM bindings") == [(3,)]
        assert _temp_files(tmp_path) == []

    def test_the_temporary_is_named_beside_the_target_with_the_pid(
        self, tmp_path: Path, chain: ParsedCorpus, monkeypatch
    ):
        path = tmp_path / "graph.duckdb"
        seen: list[str] = []

        real_replace = os.replace

        def record(src, dst):
            seen.append(str(src))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", record)
        _build(chain, path)
        assert seen == [str(tmp_path / f"graph.duckdb.tmp-{os.getpid()}")]


class TestOverwrite:
    def test_an_existing_index_is_replaced(self, tmp_path: Path, chain: ParsedCorpus):
        path = tmp_path / "graph.duckdb"
        _build(chain, path)
        smaller = parse_corpus(corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM)
        report = _build(smaller, path, corpus_filename="second.ttl")
        assert _read(path, "SELECT full_pv FROM bindings") == [("NOWHERE:RB",)]
        assert _read(path, "SELECT corpus_filename FROM meta") == [("second.ttl",)]
        assert report.binding_count == 1
        assert _temp_files(tmp_path) == []

    def test_a_file_that_is_not_a_database_is_replaced_too(
        self, tmp_path: Path, chain: ParsedCorpus
    ):
        # The build never opens the target, so whatever is there is overwritten.
        path = tmp_path / "graph.duckdb"
        path.write_bytes(b"junk")
        _build(chain, path)
        assert _read(path, "SELECT count(*) FROM bindings") == [(3,)]


class TestMetaValidation:
    @pytest.fixture
    def parts(self, chain: ParsedCorpus):
        channels = channels_from_rows(chain.binding_rows)
        return chain, channels

    def test_caller_keys_are_the_meta_columns_minus_schema_version(self):
        assert set(CALLER_META_KEYS) == set(META_KEYS) - {"schema_version"}
        assert "schema_version" not in CALLER_META_KEYS

    def test_a_missing_key_is_refused_and_named(self, parts, tmp_path: Path):
        chain, channels = parts
        meta = _meta(chain, channels)
        del meta["signal_count"]
        with pytest.raises(GraphIndexBuildError, match="missing signal_count"):
            build_from_rows(
                chain.binding_rows, chain.class_rows, channels, tmp_path / "i.duckdb", meta
            )

    def test_an_unknown_key_is_refused_and_named(self, parts, tmp_path: Path):
        chain, channels = parts
        meta = _meta(chain, channels, device_countt=1)
        with pytest.raises(GraphIndexBuildError, match="unknown device_countt"):
            build_from_rows(
                chain.binding_rows, chain.class_rows, channels, tmp_path / "i.duckdb", meta
            )

    def test_the_caller_may_not_choose_the_schema_version(self, parts, tmp_path: Path):
        chain, channels = parts
        meta = _meta(chain, channels, schema_version=99)
        with pytest.raises(GraphIndexBuildError, match="unknown schema_version"):
            build_from_rows(
                chain.binding_rows, chain.class_rows, channels, tmp_path / "i.duckdb", meta
            )

    def test_the_written_schema_version_is_the_modules(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        _build(chain, path)
        assert _read(path, "SELECT schema_version FROM meta") == [(schema_module.SCHEMA_VERSION,)]

    def test_a_binding_count_that_miscounts_the_rows_is_refused(self, parts, tmp_path: Path):
        chain, channels = parts
        meta = _meta(chain, channels, binding_count=99)
        with pytest.raises(GraphIndexBuildError, match="binding_count=99 but 3 rows"):
            build_from_rows(
                chain.binding_rows, chain.class_rows, channels, tmp_path / "i.duckdb", meta
            )

    def test_a_class_count_that_miscounts_the_rows_is_refused(self, parts, tmp_path: Path):
        chain, channels = parts
        meta = _meta(chain, channels, class_count=0)
        with pytest.raises(GraphIndexBuildError, match="class_count=0"):
            build_from_rows(
                chain.binding_rows, chain.class_rows, channels, tmp_path / "i.duckdb", meta
            )


class TestMissingDirectory:
    def test_the_error_names_the_directory(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "nested" / "graph.duckdb"
        with pytest.raises(GraphIndexBuildError, match="does not exist"):
            _build(chain, path)

    def test_it_is_a_build_error_not_a_database_error(self, chain: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "nested" / "graph.duckdb"
        with pytest.raises(GraphIndexBuildError) as excinfo:
            _build(chain, path)
        assert str(path.parent) in str(excinfo.value)


class TestSharedFullPvAndTwoDevices:
    """The shapes that make ``binding_uri`` non-unique must survive the write."""

    def test_one_binding_under_two_devices_is_two_rows(self, tmp_path: Path):
        parsed = parse_corpus(corpora.BINDING_UNDER_TWO_DEVICES)
        path = tmp_path / "graph.duckdb"
        _build(parsed, path)
        rows = _read(path, "SELECT binding_uri, section FROM bindings ORDER BY section")
        assert [section for _, section in rows] == ["BR", "SR"]
        assert len({uri for uri, _ in rows}) == 1

    def test_two_bindings_sharing_a_full_pv_are_one_channel(self, tmp_path: Path):
        parsed = parse_corpus(corpora.SHARED_FULL_PV)
        path = tmp_path / "graph.duckdb"
        _build(parsed, path)
        assert _read(path, "SELECT count(*) FROM bindings") == [(2,)]
        assert _read(path, "SELECT * FROM channels") == [("SR:MAG:SHARED:CURRENT", None, None)]


class TestImportIsolation:
    """``duckdb`` must not be imported by importing the builder."""

    def test_importing_the_builder_does_not_import_duckdb(self):
        script = textwrap.dedent(
            """
            import sys

            import osprey.services.channel_finder.graph_index.builder as builder

            forbidden = sorted(
                name
                for name in ("duckdb", "rdflib", "neo4j", "osprey.services.qmd")
                if name in sys.modules
            )
            assert not forbidden, forbidden
            assert callable(builder.build_from_rows)
            """
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_the_package_exports_the_writer_lazily(self):
        from osprey.services.channel_finder import graph_index

        assert graph_index.build_from_rows is build_from_rows
        assert graph_index.IndexBuildReport is IndexBuildReport


class TestDemoCorpus:
    @pytest.fixture(scope="class")
    def demo(self) -> ParsedCorpus:
        resource = (
            files("osprey.templates")
            .joinpath("apps")
            .joinpath("control_assistant")
            .joinpath("data")
            .joinpath("demo_machine.ttl")
        )
        with as_file(resource) as path:
            return parse_corpus(path.read_text(encoding="utf-8"))

    def test_the_shipped_corpus_writes_its_rows_well_inside_the_budget(
        self, demo: ParsedCorpus, tmp_path: Path
    ):
        path = tmp_path / "graph.duckdb"
        started = time.perf_counter()
        report = _build(demo, path, corpus_filename="demo_machine.ttl")
        elapsed = time.perf_counter() - started

        assert report.binding_count == 2908
        assert report.class_count == 19
        assert _read(path, "SELECT count(*) FROM bindings") == [(2908,)]
        assert _read(path, "SELECT count(*) FROM classes") == [(19,)]
        assert _read(path, "SELECT count(*) FROM channels") == [(report.channel_count,)]
        # The whole build (parse included) is budgeted at 5 s; the write alone
        # is a fraction of that, and a regression that made it the larger half
        # would show here first.
        assert elapsed < 5.0, f"writing the demo index took {elapsed:.2f} s"

    def test_every_binding_row_survives_with_its_lists(self, demo: ParsedCorpus, tmp_path: Path):
        path = tmp_path / "graph.duckdb"
        _build(demo, path)
        read_back = _read(path, "SELECT binding_uri, edges, class_uris, haystack FROM bindings")
        expected = [
            (row.binding_uri, row.edges, row.class_uris, row.haystack) for row in demo.binding_rows
        ]
        assert read_back == expected


def test_row_types_are_the_dataclasses_the_parser_produces():
    """A reminder that the writer inserts these three, in field order."""
    assert list(BindingRow.__dataclass_fields__)[0] == "binding_uri"
    assert list(ClassRow.__dataclass_fields__)[0] == "uri"
    assert list(ChannelRow.__dataclass_fields__) == ["address", "direction", "readback"]
