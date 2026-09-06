"""Tests for the graph search index's table layout.

The columns are pinned literally: the builder writes them positionally and the
reader's SQL names them, so a rename or a retype that slips through here is a
silent disagreement between two modules that never import each other.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import duckdb
import pytest

from osprey.services.channel_finder.core.exceptions import (
    ChannelFinderError,
    GraphIndexBuildError,
)
from osprey.services.channel_finder.graph_index import (
    META_KEYS,
    SCHEMA_VERSION,
    create_tables,
)
from osprey.services.channel_finder.graph_index import schema as schema_module

# (column name, DuckDB type) in table order, exactly as PROPOSAL Requirement 2
# lists them.
EXPECTED_COLUMNS = {
    "bindings": [
        ("binding_uri", "VARCHAR"),
        ("full_pv", "VARCHAR"),
        ("description", "VARCHAR"),
        ("device_uri", "VARCHAR"),
        ("device_name", "VARCHAR"),
        ("section", "VARCHAR"),
        ("system", "VARCHAR"),
        ("edges", "VARCHAR[]"),
        ("signal_uris", "VARCHAR[]"),
        ("signal_names", "VARCHAR[]"),
        ("class_uris", "VARCHAR[]"),
        ("haystack", "VARCHAR"),
    ],
    "classes": [
        ("uri", "VARCHAR"),
        ("name", "VARCHAR"),
        ("alt_labels", "VARCHAR[]"),
        ("parents", "VARCHAR[]"),
        ("direct_devices", "BIGINT"),
        ("rollup_devices", "BIGINT"),
    ],
    "channels": [
        ("address", "VARCHAR"),
        ("direction", "VARCHAR"),
        ("readback", "VARCHAR"),
    ],
    "meta": [
        ("schema_version", "INTEGER"),
        ("corpus_sha256", "VARCHAR"),
        ("corpus_filename", "VARCHAR"),
        ("binding_count", "BIGINT"),
        ("device_count", "BIGINT"),
        ("class_count", "BIGINT"),
        ("signal_count", "BIGINT"),
        ("section_count", "BIGINT"),
    ],
}


@pytest.fixture
def con():
    """An in-memory connection carrying the index tables."""
    connection = duckdb.connect(":memory:")
    create_tables(connection)
    try:
        yield connection
    finally:
        connection.close()


def _describe(connection: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str]]:
    return [(row[0], row[1]) for row in connection.execute(f"DESCRIBE {table}").fetchall()]


class TestSchemaVersion:
    def test_is_one(self):
        assert SCHEMA_VERSION == 1

    def test_is_an_int_not_a_string(self):
        # The reader compares it against meta.schema_version, an INTEGER column.
        assert isinstance(SCHEMA_VERSION, int)

    def test_re_exported_from_the_package_and_the_module(self):
        assert SCHEMA_VERSION is schema_module.SCHEMA_VERSION


class TestTables:
    def test_exactly_the_four_tables_are_created(self, con):
        names = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert names == {"bindings", "classes", "channels", "meta"}

    @pytest.mark.parametrize("table", sorted(EXPECTED_COLUMNS))
    def test_columns_and_types(self, con, table):
        assert _describe(con, table) == EXPECTED_COLUMNS[table]

    def test_list_columns_are_varchar_lists(self, con):
        list_columns = {
            ("bindings", "edges"),
            ("bindings", "signal_uris"),
            ("bindings", "signal_names"),
            ("bindings", "class_uris"),
            ("classes", "alt_labels"),
            ("classes", "parents"),
        }
        for table, column in list_columns:
            types = dict(_describe(con, table))
            assert types[column] == "VARCHAR[]", f"{table}.{column}"

    def test_create_tables_runs_every_statement(self):
        assert len(schema_module.CREATE_TABLE_STATEMENTS) == 4

    def test_a_second_create_fails_rather_than_reusing_a_half_built_file(self, con):
        # The builder writes each index into a fresh file; nothing migrates.
        with pytest.raises(duckdb.CatalogException):
            create_tables(con)


class TestChannelsReadback:
    def test_readback_is_nullable(self, con):
        con.execute("INSERT INTO channels VALUES ('SR:BPM1:X', 'R', NULL)")
        assert con.execute("SELECT readback FROM channels").fetchall() == [(None,)]

    def test_direction_is_nullable(self, con):
        # channels_from_rows collapses a fullPv bound twice to direction NULL.
        con.execute("INSERT INTO channels VALUES ('SR:BPM1:X', NULL, NULL)")
        assert con.execute("SELECT direction FROM channels").fetchall() == [(None,)]

    def test_address_is_not_nullable(self, con):
        with pytest.raises(duckdb.ConstraintException):
            con.execute("INSERT INTO channels VALUES (NULL, 'R', NULL)")

    def test_readback_holds_an_address_when_the_corpus_names_one(self, con):
        con.execute("INSERT INTO channels VALUES ('SR:QF1:SP', 'W', 'SR:QF1:RB')")
        assert con.execute("SELECT * FROM channels").fetchall() == [("SR:QF1:SP", "W", "SR:QF1:RB")]


class TestMetaKeys:
    def test_matches_the_meta_table_column_order(self, con):
        assert list(META_KEYS) == [name for name, _ in _describe(con, "meta")]

    def test_is_a_tuple(self):
        assert isinstance(META_KEYS, tuple)

    def test_a_meta_row_round_trips_by_key(self, con):
        values = (SCHEMA_VERSION, "abc123", "demo_machine.ttl", 2908, 700, 41, 88, 12)
        placeholders = ", ".join("?" for _ in META_KEYS)
        con.execute(f"INSERT INTO meta VALUES ({placeholders})", values)
        row = con.execute(f"SELECT {', '.join(META_KEYS)} FROM meta").fetchone()
        assert dict(zip(META_KEYS, row, strict=True)) == dict(zip(META_KEYS, values, strict=True))


class TestBindingsRow:
    def test_a_row_with_list_columns_round_trips(self, con):
        row = (
            "http://example.org/binding/1",
            "SR:BPM1:X",
            "Horizontal position",
            "http://example.org/device/bpm1",
            "BPM1",
            "SR01",
            "Diagnostics",
            ["readsSignal"],
            ["http://example.org/signal/position"],
            ["position"],
            ["http://example.org/class/BPM"],
            "sr:bpm1:x horizontal position bpm1 position bpm",
        )
        placeholders = ", ".join("?" for _ in row)
        con.execute(f"INSERT INTO bindings VALUES ({placeholders})", row)
        assert con.execute("SELECT * FROM bindings").fetchall() == [row]

    def test_nullable_device_columns_accept_a_device_without_section_or_system(self, con):
        con.execute(
            "INSERT INTO bindings VALUES (?, ?, NULL, ?, ?, NULL, NULL, [], [], [], [], ?)",
            ["u", "PV", "d", "DEV", "pv dev"],
        )
        assert con.execute("SELECT section, system FROM bindings").fetchall() == [(None, None)]

    def test_haystack_is_not_nullable(self, con):
        with pytest.raises(duckdb.ConstraintException):
            con.execute(
                "INSERT INTO bindings VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, "
                "[], [], [], [], NULL)",
                ["u", "PV"],
            )


class TestClassesRow:
    def test_counts_and_lists_round_trip(self, con):
        row = ("http://example.org/class/BPM", "BPM", ["monitor"], ["http://x/Device"], 7, 9)
        con.execute("INSERT INTO classes VALUES (?, ?, ?, ?, ?, ?)", list(row))
        assert con.execute("SELECT * FROM classes").fetchall() == [row]

    def test_device_counts_are_not_nullable(self, con):
        with pytest.raises(duckdb.ConstraintException):
            con.execute("INSERT INTO classes VALUES ('u', 'n', [], [], NULL, 0)")


class TestGraphIndexBuildError:
    def test_is_a_channel_finder_error(self):
        assert issubclass(GraphIndexBuildError, ChannelFinderError)

    def test_re_exported_from_the_index_package(self):
        from osprey.services.channel_finder import graph_index

        assert graph_index.GraphIndexBuildError is GraphIndexBuildError


class TestImportIsolation:
    """Importing the package must not drag a graph stack into the process.

    The roster reader and the health check import it on paths where ``rdflib``
    or ``neo4j`` appearing in ``sys.modules`` is the regression, so the guard
    runs in a subprocess where nothing else has imported them first.
    """

    def test_no_heavy_dependency_is_imported(self):
        script = textwrap.dedent(
            """
            import sys

            import osprey.services.channel_finder.graph_index as gi

            forbidden = sorted(
                name
                for name in ("duckdb", "rdflib", "neo4j", "osprey.services.qmd")
                if name in sys.modules
            )
            assert not forbidden, forbidden
            assert gi.SCHEMA_VERSION == 1
            assert gi.META_KEYS[0] == "schema_version"
            assert callable(gi.create_tables)
            assert issubclass(gi.GraphIndexBuildError, Exception)
            assert "duckdb" not in sys.modules
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_unknown_attribute_raises_attribute_error(self):
        from osprey.services.channel_finder import graph_index

        with pytest.raises(AttributeError):
            graph_index.no_such_name

    def test_dir_lists_the_public_names(self):
        from osprey.services.channel_finder import graph_index

        assert {
            "SCHEMA_VERSION",
            "META_KEYS",
            "create_tables",
            "GraphIndexBuildError",
        } <= set(dir(graph_index))
