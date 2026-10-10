"""Tests for the shared columnar DuckDB loader.

``bulk_insert`` loads rows through one registered frame instead of a per-row
``executemany``. These tests pin that the values arrive unchanged (``None``,
lists, strings, timestamps), that a short row is refused rather than padded,
that a table constraint still raises, and that the middle-layer import writes
exactly the rows a row-by-row load would.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from osprey.services.channel_finder.databases import duckdb_import as dimp  # noqa: E402
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


def _row_by_row(con, table, columns, rows):
    """The reference load: one parameterised INSERT per row."""
    rows = [tuple(row) for row in rows]
    if rows:
        con.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            rows,
        )
    return len(rows)


@pytest.fixture()
def multi_membership_json(tmp_path: Path) -> str:
    """A middle layer where one channel sits in two families and two systems."""
    data = {
        "SR": {
            "_description": "Storage",
            "BPM": {
                "_description": "Beam position monitors",
                "Monitor": {
                    "ChannelNames": ["SR01:BPM:X", "SHARED:PV"],
                    "Units": "Hardware",
                    "HWUnits": "mm",
                    "DataType": "double",
                    "MemberOf": ["BPM", "Diagnostics"],
                },
                "Setpoint": {"X": {"ChannelNames": ["SR01:BPM:XSet"]}},
                "setup": {"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["BPM1", "BPM2"]},
            },
            "HCM": {
                "_description": "Correctors",
                "Setpoint": {"ChannelNames": ["SR01:HCM:SP", "SHARED:PV"], "Units": "A"},
                "setup": {"DeviceList": [[2, 1]]},
            },
        },
        "BTS": {
            "_description": "Transfer line",
            "BPM": {
                "Monitor": {"ChannelNames": ["SHARED:PV"]},
            },
        },
    }
    path = tmp_path / "middle_layer.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def _dump(path: str) -> dict:
    con = duckdb.connect(path)
    try:
        return {
            "systems": con.execute("SELECT * FROM systems ORDER BY name").fetchall(),
            "families": con.execute("SELECT * FROM families ORDER BY system, name").fetchall(),
            "channels": con.execute(
                "SELECT channel_name, system, family, field, subfield, description, units, "
                "data_type, mode, member_of, source FROM channels "
                "ORDER BY channel_name, system, family"
            ).fetchall(),
            "row_ids": con.execute("SELECT row_id FROM channels ORDER BY row_id").fetchall(),
            "device_map": con.execute(
                "SELECT * FROM device_map ORDER BY system, family, device_index"
            ).fetchall(),
        }
    finally:
        con.close()


class TestMiddleLayerImport:
    @pytest.fixture(autouse=True)
    def _no_fts(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(dimp, "ensure_fts", lambda con: None)
        monkeypatch.setattr(dimp, "_create_fts_index", lambda con: None)

    def test_matches_a_row_by_row_load(
        self, multi_membership_json: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        bulk_path = str(tmp_path / "bulk.duckdb")
        bulk_stats = dimp.import_to_duckdb(multi_membership_json, bulk_path)

        monkeypatch.setattr(dimp, "bulk_insert", _row_by_row)
        ref_path = str(tmp_path / "ref.duckdb")
        ref_stats = dimp.import_to_duckdb(multi_membership_json, ref_path)

        for stats in (bulk_stats, ref_stats):
            stats.pop("duckdb_path")
        assert bulk_stats == ref_stats
        assert bulk_stats["channels"] == 6

        bulk, ref = _dump(bulk_path), _dump(ref_path)
        assert bulk == ref
        assert [row[:3] for row in bulk["channels"] if row[0] == "SHARED:PV"] == [
            ("SHARED:PV", "BTS", "BPM"),
            ("SHARED:PV", "SR", "BPM"),
            ("SHARED:PV", "SR", "HCM"),
        ]
        assert len(bulk["row_ids"]) == len(set(bulk["row_ids"])) == 6
