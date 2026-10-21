"""MCP tool: sql_query — raw SQL query against the ARIEL database.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Facility-customizable: table schema, metadata JSONB keys
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.ariel.server import (
    build_entry_url,
    check_sql_tool_offered,
    make_error,
    mcp,
)
from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.services.ariel_search.search.sql_query import (
    format_sql_result,
    validate_sql_query,
)
from osprey.services.ariel_search.search.sql_query import (
    sql_query as execute_sql_query,
)

logger = logging.getLogger("osprey.mcp_server.ariel.tools.sql_query")


@mcp.tool()
async def sql_query(
    sql: str,
    max_rows: int = 100,
) -> str:
    """Execute a read-only SQL query against the ARIEL logbook database.

    Use for structural queries that keyword/semantic search can't express:
    counting, grouping, date arithmetic, JSONB filtering, etc.

    Table: enhanced_entries
    Columns: entry_id (text PK), timestamp (timestamptz), author (text),
      source_system (text), raw_text (text), metadata (jsonb),
      summary (text), keywords (text[]), attachments (jsonb),
      attachment_text (text), attachment_captions (jsonb),
      enhancement_status (jsonb), created_at (timestamptz), updated_at (timestamptz)

    attachment_text is the text read from an entry's attachments, and
    attachment_captions the captions of its pictures; both are NULL until the
    attachments are processed. Binary tables are not queryable: attachment_files
    holds the stored file bytes and is refused.

    metadata JSONB keys: logbook, tag, shift, activity_type, logbook_name,
      entry_type, references, event_time, facility_section

    Only SELECT and WITH (CTE) queries are allowed. No writes, no DDL.
    Tables allowed: enhanced_entries, text_embeddings_*.

    Args:
        sql: Read-only SQL statement (SELECT or WITH only).
        max_rows: Maximum rows to return (1-200, default 100).

    Note: timestamp columns (timestamp, created_at, updated_at) are returned as
    stored — UTC with a +00:00 offset. Unlike the other ARIEL tools, this raw
    query path does not convert them to the facility timezone (the columns a query
    selects are arbitrary), so convert client-side if you need facility-local.

    Returns:
        JSON with query results as a list of row objects.
    """
    check_sql_tool_offered()

    if not sql or not sql.strip():
        return make_error(
            "validation_error",
            "Empty SQL query.",
            ["Provide a SELECT query against the enhanced_entries table."],
        )

    try:
        # Fail fast before acquiring a DB connection
        validate_sql_query(sql)

        registry = get_ariel_context()
        service = await registry.service()

        # The SELECT-only login is the boundary. `BEGIN READ ONLY` and the
        # statement allowlist still apply underneath; what the role adds is the
        # half neither can cover, since a transaction mode does not stop a
        # server-side read and no pattern over query text enumerates every
        # dangerous function. So the query runs on that login or not at all,
        # never on the ingestion pool. execute_sql_query re-validates internally.
        pool = service.readonly_pool
        if pool is None:
            reason = service.readonly_pool_error
            make_error(
                "service_unavailable",
                f"The SQL tool is unavailable: the database refused its read-only login ({reason})."
                if reason
                else "The SQL tool is unavailable: no read-only connection is open.",
                [
                    "An administrator can create the role as described in the how-to "
                    "'Search Modes', section 'Read-only role for the SQL tool'."
                ],
            )
        rows = await execute_sql_query(pool, sql, max_rows=max_rows)

        # Egress: rows that select an entry_id column gain a canonical entry_url
        # (config-driven). Rows without entry_id (aggregates, projections) and
        # ARIEL-native entries are left untouched by build_entry_url. Injected
        # before formatting so the text view carries the URL too.
        for row in rows:
            if isinstance(row, dict) and row.get("entry_id"):
                entry_url = build_entry_url(row.get("entry_id"), row.get("source_system"))
                if entry_url is not None:
                    row["entry_url"] = entry_url

        formatted = format_sql_result(rows)

        response = {
            "sql": sql,
            "row_count": len(rows),
            "rows": rows,
            "formatted": formatted,
        }

        return json.dumps(response, default=str)

    except ValueError as exc:
        # Validation errors from validate_sql_query
        return make_error(
            "validation_error",
            str(exc),
            [
                "Only SELECT/WITH queries on enhanced_entries and "
                "text_embeddings_* tables are allowed.",
                "Binary tables are not queryable; attachment text is in the "
                "enhanced_entries columns attachment_text and attachment_captions.",
            ],
        )
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("sql_query failed")
        return make_error(
            "internal_error",
            f"ARIEL SQL query failed: {exc}",
            [
                "Check ARIEL database connectivity.",
                "Verify your SQL syntax is correct.",
            ],
        )
