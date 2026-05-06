"""Tests for the sql_query MCP tool and sql_query validation."""

import json
from unittest.mock import AsyncMock, patch
from tests.mcp_server.conftest import assert_error, assert_raises_error, extract_response_dict

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.services.ariel_search.search.sql_query import validate_sql_query
from tests.mcp_server.ariel.conftest import get_tool_fn

# ---------------------------------------------------------------------------
# validate_sql_query unit tests
# ---------------------------------------------------------------------------


class TestValidateSqlQuery:
    """Tests for the SQL validation function."""

    def test_valid_select(self):
        """Basic SELECT is accepted."""
        validate_sql_query("SELECT * FROM enhanced_entries LIMIT 10")

    def test_valid_cte(self):
        """WITH (CTE) + SELECT is accepted."""
        validate_sql_query(
            "WITH recent AS (SELECT * FROM enhanced_entries WHERE timestamp > '2024-01-01') "
            "SELECT * FROM recent"
        )

    def test_reject_insert(self):
        """INSERT is rejected."""
        with pytest.raises(ValueError, match="INSERT"):
            validate_sql_query("INSERT INTO enhanced_entries VALUES ('x')")

    def test_reject_update(self):
        """UPDATE is rejected."""
        with pytest.raises(ValueError, match="UPDATE"):
            validate_sql_query("UPDATE enhanced_entries SET author = 'x'")

    def test_reject_delete(self):
        """DELETE is rejected."""
        with pytest.raises(ValueError, match="DELETE"):
            validate_sql_query("DELETE FROM enhanced_entries")

    def test_reject_drop(self):
        """DROP is rejected."""
        with pytest.raises(ValueError, match="DROP"):
            validate_sql_query("DROP TABLE enhanced_entries")

    def test_reject_truncate(self):
        """TRUNCATE is rejected."""
        with pytest.raises(ValueError, match="TRUNCATE"):
            validate_sql_query("TRUNCATE enhanced_entries")

    def test_reject_multi_statement(self):
        """Semicolons in query body are rejected."""
        with pytest.raises(ValueError, match="Multi-statement"):
            validate_sql_query("SELECT 1; DROP TABLE enhanced_entries")

    def test_trailing_semicolon_ok(self):
        """Trailing semicolon is stripped, not rejected."""
        validate_sql_query("SELECT * FROM enhanced_entries;")

    def test_case_insensitive_rejection(self):
        """Forbidden keywords detected case-insensitively."""
        with pytest.raises(ValueError, match="DROP"):
            validate_sql_query("SELECT * FROM enhanced_entries WHERE 1=1 DROP TABLE foo")

    def test_reject_table_not_in_allowlist(self):
        """Tables not in allowlist are rejected."""
        with pytest.raises(ValueError, match="not in the allowlist"):
            validate_sql_query("SELECT * FROM users")

    def test_text_embeddings_table_allowed(self):
        """text_embeddings_* tables are allowed."""
        validate_sql_query("SELECT * FROM text_embeddings_nomic_embed_text LIMIT 5")

    def test_empty_query(self):
        """Empty query is rejected."""
        with pytest.raises(ValueError, match="empty"):
            validate_sql_query("")

    def test_must_start_with_select_or_with(self):
        """Query must start with SELECT or WITH."""
        with pytest.raises(ValueError, match="Only SELECT"):
            validate_sql_query("EXPLAIN SELECT * FROM enhanced_entries")

    def test_cte_with_insert_rejected(self):
        """CTE followed by INSERT is rejected."""
        with pytest.raises(ValueError, match="INSERT"):
            validate_sql_query(
                "WITH data AS (SELECT 1) INSERT INTO enhanced_entries SELECT * FROM data"
            )

    def test_comment_injection(self):
        """SQL comments don't bypass keyword detection."""
        # The query body still contains DROP after the comment
        with pytest.raises(ValueError, match="DROP"):
            validate_sql_query("SELECT * FROM enhanced_entries -- DROP TABLE foo\n DROP TABLE bar")


# ---------------------------------------------------------------------------
# sql_query MCP tool tests
# ---------------------------------------------------------------------------


def _get_sql_query():
    from osprey.mcp_server.ariel.tools.sql_query import sql_query

    return get_tool_fn(sql_query)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    (tmp_path / "config.yml").write_text(config)
    initialize_ariel_context()


@pytest.mark.unit
async def test_sql_query_valid(tmp_path, monkeypatch):
    """Valid SELECT query executes and returns rows."""
    _setup_registry(tmp_path, monkeypatch)

    mock_rows = [
        {"entry_id": "e1", "author": "Alice"},
        {"entry_id": "e2", "author": "Bob"},
    ]

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(return_value=mock_rows),
        ),
    ):
        fn = _get_sql_query()
        result = await fn(query="SELECT entry_id, author FROM enhanced_entries LIMIT 2")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["row_count"] == 2
    assert data["rows"][0]["entry_id"] == "e1"


@pytest.mark.unit
async def test_sql_query_rejected_dml():
    """DML queries return validation error without touching the database."""
    fn = _get_sql_query()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(query="INSERT INTO enhanced_entries VALUES ('x')")

    data = _exc_ctx["envelope"]
    assert "INSERT" in data["error_message"]


@pytest.mark.unit
async def test_sql_query_empty():
    """Empty query returns validation error."""
    fn = _get_sql_query()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(query="")

    data = _exc_ctx["envelope"]


@pytest.mark.unit
async def test_sql_query_row_limit(tmp_path, monkeypatch):
    """max_rows > 200 is capped at 200 inside sql_query()."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    mock_sql = AsyncMock(return_value=[])

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=mock_sql,
        ),
    ):
        fn = _get_sql_query()
        await fn(query="SELECT * FROM enhanced_entries", max_rows=500)

    # Tool passes max_rows through; capping happens inside sql_query()
    mock_sql.assert_called_once()


@pytest.mark.unit
async def test_sql_query_service_error(tmp_path, monkeypatch):
    """Database error returns internal_error."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    with (
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=mock_service),
        ),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(side_effect=RuntimeError("Connection refused")),
        ),
    ):
        fn = _get_sql_query()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(query="SELECT * FROM enhanced_entries")

    data = _exc_ctx["envelope"]
    assert "Connection refused" in data["error_message"]
