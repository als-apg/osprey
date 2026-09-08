"""Tests for the Middle Layer channel-finder ``run_sql`` DuckDB tool.

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

from osprey.mcp_server.channel_finder_middle_layer.server_context import (
    DEFAULT_QUERY_MAX_ROWS,
    QUERY_MAX_ROWS_CONFIG_KEY,
)
from tests.mcp_server.channel_finder_middle_layer.conftest import get_tool_fn
from tests.mcp_server.conftest import assert_raises_error

_TOOL_MODULE = "osprey.mcp_server.channel_finder_middle_layer.tools.run_sql"


def _make_duckdb(path, rows):
    """Create a minimal channels DuckDB at ``path`` with ``rows`` (name, desc)."""
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE channels (channel_name VARCHAR, description VARCHAR)")
    con.executemany("INSERT INTO channels VALUES (?, ?)", rows)
    con.close()


def _run(sql: str, duckdb_path, query_max_rows: int = DEFAULT_QUERY_MAX_ROWS):
    """Invoke the tool fn with get_cf_ml_context patched to expose duckdb_path.

    ``query_max_rows`` is what the deployment's `channel_finder.query_max_rows`
    resolved to; the default here is the context's own fallback, so a test that
    does not care about the cap exercises the shipped one.
    """
    from osprey.mcp_server.channel_finder_middle_layer.tools.run_sql import run_sql

    ctx = MagicMock()
    ctx.duckdb_path = duckdb_path
    ctx.query_max_rows = query_max_rows
    with patch(f"{_TOOL_MODULE}.get_cf_ml_context", return_value=ctx):
        return get_tool_fn(run_sql)(sql=sql)


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
    """More rows than the cap are capped at it and flagged truncated.

    The number comes from the deployment's `channel_finder.query_max_rows`, so
    this reads the default rather than pinning a literal a facility may raise.
    """
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [(f"PV:{i}", f"desc {i}") for i in range(DEFAULT_QUERY_MAX_ROWS + 100)])

    result = _run("SELECT channel_name FROM channels", str(db))
    data = json.loads(result)

    assert data["row_count"] == DEFAULT_QUERY_MAX_ROWS
    assert len(data["rows"]) == DEFAULT_QUERY_MAX_ROWS
    assert data["truncated"] is True


@pytest.mark.unit
def test_the_cap_is_the_configured_one(tmp_path):
    """A facility that lowers the key gets fewer rows, not the shipped default."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [(f"PV:{i}", f"desc {i}") for i in range(20)])

    data = json.loads(_run("SELECT channel_name FROM channels", str(db), query_max_rows=5))

    assert data["row_count"] == 5
    assert data["truncated"] is True


@pytest.mark.unit
def test_a_truncated_answer_names_the_key_and_the_number(tmp_path):
    """The agent is told what cut the list, so it narrows instead of guessing."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [(f"PV:{i}", f"desc {i}") for i in range(20)])

    data = json.loads(_run("SELECT channel_name FROM channels", str(db), query_max_rows=5))

    guidance = " ".join(data["guidance"])
    assert QUERY_MAX_ROWS_CONFIG_KEY in guidance
    assert "5" in guidance


@pytest.mark.unit
def test_an_untruncated_answer_carries_no_guidance(tmp_path):
    """Nothing to say when the whole result fits."""
    db = tmp_path / "chan.duckdb"
    _make_duckdb(db, [("PV:1", "x")])

    data = json.loads(_run("SELECT channel_name FROM channels", str(db)))

    assert "guidance" not in data


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
    from osprey.mcp_server.channel_finder_middle_layer.tools.run_sql import run_sql

    ctx = MagicMock()
    ctx.duckdb_path = None
    with patch(f"{_TOOL_MODULE}.get_cf_ml_context", return_value=ctx):
        with assert_raises_error(error_type="not_configured"):
            get_tool_fn(run_sql)(sql="SELECT 1")


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
