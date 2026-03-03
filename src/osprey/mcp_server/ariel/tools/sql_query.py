"""MCP tool: sql_query — raw SQL query against the ARIEL database.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Facility-customizable: table schema, metadata JSONB keys
"""

import json
import logging

from osprey.mcp_server.ariel.server_context import get_ariel_context
from osprey.mcp_server.ariel.server import make_error, mcp
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
    query: str,
    max_rows: int = 100,
) -> str:
    """Execute a read-only SQL query against the ARIEL logbook database.

    Use for structural queries that keyword/semantic search can't express:
    counting, grouping, date arithmetic, JSONB filtering, etc.

    Table: enhanced_entries
    Columns: entry_id (text PK), timestamp (timestamptz), author (text),
      source_system (text), raw_text (text), metadata (jsonb),
      summary (text), keywords (text[]), attachments (jsonb),
      enhancement_status (jsonb), created_at (timestamptz), updated_at (timestamptz)

    metadata JSONB keys: logbook, tag, shift, activity_type, logbook_name,
      entry_type, references, event_time, facility_section

    Only SELECT and WITH (CTE) queries are allowed. No writes, no DDL.
    Tables allowed: enhanced_entries, text_embeddings_*.

    Args:
        query: Read-only SQL query (SELECT or WITH only).
        max_rows: Maximum rows to return (1-200, default 100).

    Returns:
        JSON with query results as a list of row objects.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "Empty SQL query.",
                ["Provide a SELECT query against the enhanced_entries table."],
            )
        )

    try:
        # Fail fast before acquiring a DB connection
        validate_sql_query(query)

        registry = get_ariel_context()
        service = await registry.service()

        # execute_sql_query re-validates internally
        rows = await execute_sql_query(service.pool, query, max_rows=max_rows)

        formatted = format_sql_result(rows)

        response = {
            "query": query,
            "row_count": len(rows),
            "rows": rows,
            "formatted": formatted,
        }

        return json.dumps(response, default=str)

    except ValueError as exc:
        # Validation errors from validate_sql_query
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                [
                    "Only SELECT/WITH queries on enhanced_entries and "
                    "text_embeddings_* tables are allowed.",
                ],
            )
        )
    except Exception as exc:
        logger.exception("sql_query failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"ARIEL SQL query failed: {exc}",
                [
                    "Check ARIEL database connectivity.",
                    "Verify your SQL syntax is correct.",
                ],
            )
        )
