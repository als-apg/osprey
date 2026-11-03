"""Tests for the sql_query MCP tool and sql_query validation."""

from unittest.mock import AsyncMock, patch

import pytest

from osprey.services.ariel_search.search.sql_query import validate_sql_query
from tests.mcp_server.ariel.conftest import get_tool_fn, patch_ariel_service
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict

# ---------------------------------------------------------------------------
# validate_sql_query unit tests
# ---------------------------------------------------------------------------


class TestValidateSqlQuery:
    """Tests for the SQL validation function."""

    @pytest.mark.parametrize(
        "query",
        [
            # Basic SELECT is accepted.
            pytest.param("SELECT * FROM enhanced_entries LIMIT 10", id="select"),
            # WITH (CTE) + SELECT is accepted.
            pytest.param(
                "WITH recent AS (SELECT * FROM enhanced_entries WHERE timestamp > '2024-01-01') "
                "SELECT * FROM recent",
                id="cte",
            ),
            # Trailing semicolon is stripped, not rejected.
            pytest.param("SELECT * FROM enhanced_entries;", id="trailing-semicolon"),
            # text_embeddings_* tables are allowed.
            pytest.param(
                "SELECT * FROM text_embeddings_nomic_embed_text LIMIT 5",
                id="text-embeddings-table",
            ),
        ],
    )
    def test_a_read_only_query_over_allowed_tables_is_accepted(self, query):
        validate_sql_query(query)

    @pytest.mark.parametrize(
        ("query", "match"),
        [
            # A server-side file read names no table, so the allowlist saw nothing.
            pytest.param(
                "SELECT pg_read_file('/etc/passwd')",
                "reads no allowlisted table",
                id="query-that-reads-no-table",
            ),
            # A query whose only references are its own CTEs reads no real table.
            pytest.param(
                "WITH x AS (SELECT 1 AS n) SELECT * FROM x",
                "reads no allowlisted table",
                id="cte-only-query",
            ),
            pytest.param("INSERT INTO enhanced_entries VALUES ('x')", "INSERT", id="insert"),
            pytest.param("UPDATE enhanced_entries SET author = 'x'", "UPDATE", id="update"),
            pytest.param("DELETE FROM enhanced_entries", "DELETE", id="delete"),
            pytest.param("DROP TABLE enhanced_entries", "DROP", id="drop"),
            pytest.param("TRUNCATE enhanced_entries", "TRUNCATE", id="truncate"),
            # Semicolons in query body are rejected.
            pytest.param(
                "SELECT 1; DROP TABLE enhanced_entries", "Multi-statement", id="multi-statement"
            ),
            # Forbidden keywords detected case-insensitively.
            pytest.param(
                "SELECT * FROM enhanced_entries WHERE 1=1 DROP TABLE foo",
                "DROP",
                id="case-insensitive-rejection",
            ),
            # Tables not in allowlist are rejected.
            pytest.param(
                "SELECT * FROM users", "not in the allowlist", id="table-not-in-allowlist"
            ),
            pytest.param("", "empty", id="empty-query"),
            # Query must start with SELECT or WITH.
            pytest.param(
                "EXPLAIN SELECT * FROM enhanced_entries",
                "Only SELECT",
                id="must-start-with-select-or-with",
            ),
            # CTE followed by INSERT is rejected.
            pytest.param(
                "WITH data AS (SELECT 1) INSERT INTO enhanced_entries SELECT * FROM data",
                "INSERT",
                id="cte-with-insert",
            ),
            # A comment cannot hide a keyword: the comment itself is refused first.
            pytest.param(
                "SELECT * FROM enhanced_entries -- DROP TABLE foo\n DROP TABLE bar",
                "Line comments are not allowed",
                id="comment-injection",
            ),
        ],
    )
    def test_a_disallowed_query_is_rejected_with_its_reason(self, query, match):
        with pytest.raises(ValueError, match=match):
            validate_sql_query(query)


# ---------------------------------------------------------------------------
# sql_query MCP tool tests
# ---------------------------------------------------------------------------


def _get_sql_query():
    from osprey.mcp_server.ariel.tools.sql_query import sql_query

    return get_tool_fn(sql_query)


async def test_sql_query_valid(ariel_registry):
    """Valid SELECT query executes and returns rows."""
    ariel_registry()

    mock_rows = [
        {"entry_id": "e1", "author": "Alice"},
        {"entry_id": "e2", "author": "Bob"},
    ]

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(return_value=mock_rows),
        ),
    ):
        fn = _get_sql_query()
        result = await fn(sql="SELECT entry_id, author FROM enhanced_entries LIMIT 2")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["row_count"] == 2
    assert data["rows"][0]["entry_id"] == "e1"


async def test_sql_query_rejected_dml():
    """DML queries return validation error without touching the database."""
    fn = _get_sql_query()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(sql="INSERT INTO enhanced_entries VALUES ('x')")

    data = _exc_ctx["envelope"]
    assert "INSERT" in data["error_message"]


async def test_sql_query_empty():
    """Empty query returns validation error."""
    fn = _get_sql_query()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(sql="")

    _exc_ctx["envelope"]


async def test_sql_query_row_limit(ariel_registry):
    """max_rows > 200 is capped at 200 inside sql_query()."""
    ariel_registry()

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    mock_sql = AsyncMock(return_value=[])

    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=mock_sql,
        ),
    ):
        fn = _get_sql_query()
        await fn(sql="SELECT * FROM enhanced_entries", max_rows=500)

    # Tool passes max_rows through; capping happens inside sql_query()
    mock_sql.assert_called_once()


async def test_sql_query_service_error(ariel_registry):
    """Database error returns internal_error."""
    ariel_registry()

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()

    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=AsyncMock(side_effect=RuntimeError("Connection refused")),
        ),
    ):
        fn = _get_sql_query()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(sql="SELECT * FROM enhanced_entries")

    data = _exc_ctx["envelope"]
    assert "Connection refused" in data["error_message"]


async def test_sql_query_uses_the_readonly_pool(ariel_registry):
    """The tool queries through the SELECT-only role where the deployment has
    one, so a dangerous function is refused by Postgres rather than by a
    pattern match over the query text."""
    ariel_registry()

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()
    mock_service.readonly_pool = AsyncMock()

    mock_sql = AsyncMock(return_value=[])

    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=mock_sql,
        ),
    ):
        fn = _get_sql_query()
        await fn(sql="SELECT entry_id FROM enhanced_entries")

    assert mock_sql.call_args.args[0] is mock_service.readonly_pool


async def test_sql_query_falls_back_to_the_ingestion_pool(ariel_registry):
    """A store whose data volume predates the role has no read-only pool. The
    tool keeps working on the connection it always used; the service logs the
    one warning about it at start-up rather than here, per query."""
    ariel_registry()

    mock_service = AsyncMock()
    mock_service.pool = AsyncMock()
    mock_service.readonly_pool = None

    mock_sql = AsyncMock(return_value=[])

    with (
        patch_ariel_service(mock_service),
        patch(
            "osprey.mcp_server.ariel.tools.sql_query.execute_sql_query",
            new=mock_sql,
        ),
    ):
        fn = _get_sql_query()
        await fn(sql="SELECT entry_id FROM enhanced_entries")

    assert mock_sql.call_args.args[0] is mock_service.pool


async def test_sql_query_binary_table_refusal_hints_at_text_columns():
    """The tool-level hint names the attachment text columns an agent may query instead."""
    fn = _get_sql_query()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(sql="SELECT data FROM attachment_files")

    data = _exc_ctx["envelope"]
    assert "attachment_files" in data["error_message"]
    hints = " ".join(data["suggestions"])
    assert "binary tables are not queryable" in hints.lower()
    assert "attachment_text" in hints


def test_sql_query_docstring_lists_attachment_text_columns():
    """The agent reads the docstring as the schema; the attachment text columns are in it."""
    from osprey.mcp_server.ariel.tools.sql_query import sql_query

    doc = get_tool_fn(sql_query).__doc__ or ""
    assert "attachment_text (text)" in doc
    assert "attachment_captions (jsonb)" in doc
    assert "attachment_files" in doc
