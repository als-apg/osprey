"""Tests for the shared columnar DuckDB loader.

``bulk_insert`` loads rows through one registered frame instead of a per-row
``executemany``. These tests pin that the values arrive unchanged (``None``,
lists, strings, timestamps), that a short row is refused rather than padded,
and that a table constraint still raises.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

duckdb = pytest.importorskip("duckdb")

from osprey.services.channel_finder.databases.duckdb_bulk import bulk_insert  # noqa: E402


@pytest.fixture()
def con():
    connection = duckdb.connect()
    try:
        yield connection
    finally:
        connection.close()


class TestBulkInsert:
    def test_round_trips_none_list_str_and_timestamp(self, con):
        con.execute(
            "CREATE TABLE t (name TEXT, tags TEXT[], note TEXT, stamp TIMESTAMP, n INTEGER)"
        )
        stamp = datetime(2026, 1, 2, 3, 4, 5, 678901)
        rows = [
            ("a", ["x", "y"], None, stamp, 1),
            ("b", [], "text", None, None),
            ("c", None, "", stamp, 3),
        ]

        written = bulk_insert(con, "t", ("name", "tags", "note", "stamp", "n"), iter(rows))

        assert written == 3
        assert con.execute("SELECT * FROM t ORDER BY name").fetchall() == rows

    def test_aware_timestamp_lands_as_a_row_by_row_insert_lands_it(self, con):
        con.execute("CREATE TABLE t (name TEXT, stamp TIMESTAMP)")
        stamp = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC)
        con.execute("INSERT INTO t (name, stamp) VALUES (?, ?)", ("ref", stamp))
        bulk_insert(con, "t", ("name", "stamp"), [("bulk", stamp)])
        stamps = dict(con.execute("SELECT name, stamp FROM t").fetchall())
        assert stamps["bulk"] == stamps["ref"]

    def test_all_none_column_takes_the_table_type(self, con):
        con.execute("CREATE TABLE t (name TEXT, n INTEGER)")
        bulk_insert(con, "t", ("name", "n"), [("a", None), ("b", None)])
        assert con.execute("SELECT name, n FROM t ORDER BY name").fetchall() == [
            ("a", None),
            ("b", None),
        ]

    def test_no_rows_writes_nothing(self, con):
        con.execute("CREATE TABLE t (name TEXT)")
        assert bulk_insert(con, "t", ("name",), []) == 0
        assert con.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)

    def test_unnamed_columns_keep_their_defaults(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("CREATE TABLE t (row_id BIGINT DEFAULT nextval('s'), name TEXT)")
        bulk_insert(con, "t", ("name",), [("a",), ("b",)])
        assert con.execute("SELECT row_id, name FROM t ORDER BY row_id").fetchall() == [
            (1, "a"),
            (2, "b"),
        ]

    def test_short_row_is_refused(self, con):
        con.execute("CREATE TABLE bindings (a TEXT, b TEXT)")
        with pytest.raises(
            ValueError,
            match=r"Row 3 of bindings carries 1 values, but the table has 2 columns: a, b\.",
        ):
            bulk_insert(con, "bindings", ("a", "b"), [("1", "2")] * 3 + [("1",)])
        assert con.execute("SELECT COUNT(*) FROM bindings").fetchone() == (0,)

    def test_unique_violation_still_raises(self, con):
        con.execute("CREATE TABLE t (name TEXT, UNIQUE (name))")
        with pytest.raises(duckdb.ConstraintException):
            bulk_insert(con, "t", ("name",), [("a",), ("a",)])

    def test_view_is_released_after_a_failure(self, con):
        con.execute("CREATE TABLE t (name TEXT, UNIQUE (name))")
        with pytest.raises(duckdb.ConstraintException):
            bulk_insert(con, "t", ("name",), [("a",), ("a",)])
        assert bulk_insert(con, "t", ("name",), [("b",)]) == 1
