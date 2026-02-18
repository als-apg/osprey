"""Tests for the MATLAB MML indexer — schema, data loading, and stats."""

import json
import sqlite3

import pytest

from osprey.mcp_server.matlab.indexer import build_index


class TestBuildIndex:
    def test_builds_db(self, tmp_path, sample_data_file):
        db_path = tmp_path / "test.db"
        result = build_index(data_file=sample_data_file, db_path=db_path)
        assert result == db_path
        assert db_path.exists()

    def test_correct_function_count(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) as c FROM functions").fetchone()
        assert row["c"] == 8
        conn.close()

    def test_fts_populated(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM functions_fts WHERE functions_fts MATCH 'orbit correction'"
        ).fetchall()
        assert len(rows) >= 1
        conn.close()

    def test_dependencies_indexed(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) as c FROM dependencies").fetchone()
        # 10 edges total, 1 self-loop filtered = 9 edges
        assert row["c"] == 9
        conn.close()

    def test_self_loops_filtered(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM dependencies WHERE caller = callee"
        ).fetchall()
        assert len(rows) == 0
        conn.close()

    def test_stats_materialized(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM stats WHERE key='total_functions'"
        ).fetchone()
        assert row is not None
        assert int(row["value"]) == 8
        conn.close()

    def test_group_indexed(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT function_name FROM functions WHERE group_name = 'StorageRing'"
        ).fetchall()
        assert len(rows) == 3  # getbpm, setsp, orbitcorrection
        conn.close()

    def test_missing_data_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_index(
                data_file=tmp_path / "nonexistent.json",
                db_path=tmp_path / "test.db",
            )

    def test_source_code_null_when_file_missing(self, indexed_db):
        """Source code should be None when .m files don't exist."""
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT source_code FROM functions WHERE function_name = 'getbpm'"
        ).fetchone()
        assert row["source_code"] is None
        conn.close()

    def test_stats_groups_breakdown(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM stats WHERE key='groups'"
        ).fetchone()
        groups = json.loads(row["value"])
        assert "StorageRing" in groups
        assert groups["StorageRing"] == 3
        conn.close()
