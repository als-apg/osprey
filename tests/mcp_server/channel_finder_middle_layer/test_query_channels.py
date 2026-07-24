"""Tests for the Middle Layer channel-finder ``query_channels`` DuckDB tool.

Exercises the tool against a real on-disk DuckDB database (duckdb + fts are
first-party deps and work offline), covering the happy path, the row cap /
truncation flag, the SELECT-only guard, the not-configured guard, and the
SQL-error and internal-error envelopes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from tests.mcp_server.channel_finder_middle_layer.conftest import get_tool_fn
from tests.mcp_server.conftest import assert_raises_error

_TOOL_MODULE = "osprey.mcp_server.channel_finder_middle_layer.tools.query_channels"


def _make_duckdb(path, rows):
    """Create a minimal channels DuckDB at ``path`` with ``rows`` (name, desc)."""
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE channels (channel_name VARCHAR, description VARCHAR)")
    con.executemany("INSERT INTO channels VALUES (?, ?)", rows)
    con.close()


def _run(sql: str, duckdb_path):
    """Invoke the tool fn with get_cf_ml_context patched to expose duckdb_path."""
    from osprey.mcp_server.channel_finder_middle_layer.tools.query_channels import query_channels

    ctx = MagicMock()
    ctx.duckdb_path = duckdb_path
    with patch(f"{_TOOL_MODULE}.get_cf_ml_context", return_value=ctx):
        return get_tool_fn(query_channels)(sql=sql)


@pytest.mark.unit
def test_query_returns_columns_rows_and_count(tmp_path):
    """Happy path: a SELECT returns columns, dict rows, count, and truncated=False."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [("SR:BPM1:X", "horizontal position"), ("SR:HCM1", "corrector")])

    result = _run("SELECT channel_name, description FROM channels ORDER BY channel_name", str(db))
    data = json.loads(result)

    assert data["columns"] == ["channel_name", "description"]
    assert data["row_count"] == 2
    assert data["truncated"] is False
    assert data["rows"][0] == {"channel_name": "SR:BPM1:X", "description": "horizontal position"}


@pytest.mark.unit
def test_query_caps_and_flags_truncation(tmp_path):
    """More than 500 matching rows are capped at 500 and flagged truncated."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [(f"PV:{i}", f"desc {i}") for i in range(600)])

    result = _run("SELECT channel_name FROM channels", str(db))
    data = json.loads(result)

    assert data["row_count"] == 500
    assert len(data["rows"]) == 500
    assert data["truncated"] is True


@pytest.mark.unit
def test_non_select_query_is_rejected(tmp_path):
    """Only SELECT is allowed; a mutating statement raises invalid_query."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [("PV:1", "x")])

    with assert_raises_error(error_type="invalid_query") as ctx:
        _run("DROP TABLE channels", str(db))
    assert "SELECT" in ctx["envelope"]["error_message"]


@pytest.mark.unit
def test_not_configured_when_no_duckdb_path():
    """A context without a DuckDB path yields a not_configured envelope."""
    from osprey.mcp_server.channel_finder_middle_layer.tools.query_channels import query_channels

    ctx = MagicMock()
    ctx.duckdb_path = None
    with patch(f"{_TOOL_MODULE}.get_cf_ml_context", return_value=ctx):
        with assert_raises_error(error_type="not_configured"):
            get_tool_fn(query_channels)(sql="SELECT 1")


@pytest.mark.unit
def test_sql_error_returns_sql_error_envelope(tmp_path):
    """A DuckDB execution error is classified as sql_error with guidance."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [("PV:1", "x")])

    with assert_raises_error(error_type="sql_error") as ctx:
        _run("SELECT * FROM table_that_does_not_exist", str(db))
    assert ctx["envelope"]["suggestions"]  # actionable hints present


@pytest.mark.unit
def test_unexpected_error_is_internal_error(tmp_path):
    """A non-DuckDB exception falls through to the internal_error envelope."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [("PV:1", "x")])

    with patch("duckdb.connect", side_effect=RuntimeError("disk gone")):
        with assert_raises_error(error_type="internal_error"):
            _run("SELECT 1", str(db))
