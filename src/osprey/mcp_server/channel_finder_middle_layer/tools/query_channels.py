"""MCP tool: query_channels — run SQL queries against the DuckDB channel database.

Provides agents with ad-hoc SQL access to the channel finder data,
including full-text search via the FTS index.
"""

import json
import logging

import duckdb

from osprey.mcp_server.channel_finder_middle_layer.server import make_error, mcp
from osprey.mcp_server.channel_finder_middle_layer.server_context import get_cf_ml_context

logger = logging.getLogger("osprey.mcp_server.channel_finder_middle_layer.tools.query_channels")

_MAX_ROWS = 500


@mcp.tool()
def query_channels(sql: str) -> str:
    """Run a read-only SQL query against the channel finder DuckDB database.

    The database contains these tables:

    **channels** — One row per EPICS PV channel.
    Columns: channel_name (PK), system, family, field, subfield,
    description, units, data_type, mode, member_of, source, updated_at.

    **systems** — Accelerator systems (e.g., SR, BR, BTS).
    Columns: name (PK), description.

    **families** — Device families within systems (e.g., BPM, HCM).
    Columns: system, name, description.  PK: (system, name).

    **device_map** — Physical device layout.
    Columns: system, family, device_index, sector, device, common_name.

    **Full-text search** is available on channels. Example:
        SELECT channel_name, description,
               fts_main_channels.match_bm25(channel_name, 'corrector magnet') AS score
        FROM channels
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT 20

    Only SELECT queries are allowed.  Results are capped at 500 rows.

    Args:
        sql: A SQL SELECT statement to execute.

    Returns:
        JSON with columns, rows, and row_count.  Or an error envelope.
    """
    try:
        ctx = get_cf_ml_context()
        duckdb_path = getattr(ctx, "duckdb_path", None)
        if not duckdb_path:
            return json.dumps(
                make_error(
                    "not_configured",
                    "DuckDB database is not configured for the channel finder.",
                    ["Ensure channel_finder.pipelines.middle_layer.database.duckdb_path is set in config.yml."],
                )
            )

        # Safety: only allow SELECT
        stripped = sql.strip()
        if not stripped.upper().startswith("SELECT"):
            return json.dumps(
                make_error(
                    "invalid_query",
                    "Only SELECT queries are allowed.",
                    ["Rewrite your query as a SELECT statement."],
                )
            )

        con = duckdb.connect(duckdb_path, read_only=True)
        try:
            con.execute("LOAD fts;")
            result = con.execute(stripped)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchmany(_MAX_ROWS)
            row_count = len(rows)

            # Convert to list of dicts for JSON serialisation
            data = [dict(zip(columns, row)) for row in rows]

            return json.dumps(
                {"columns": columns, "rows": data, "row_count": row_count, "truncated": row_count >= _MAX_ROWS},
                default=str,
            )
        finally:
            con.close()

    except duckdb.Error as exc:
        logger.warning("query_channels SQL error: %s", exc)
        return json.dumps(
            make_error(
                "sql_error",
                f"SQL error: {exc}",
                ["Check your SQL syntax.", "Use 'SELECT * FROM channels LIMIT 10' to explore the schema."],
            )
        )
    except Exception as exc:
        logger.exception("query_channels failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Query failed: {exc}",
                ["Check that the DuckDB database exists and is accessible."],
            )
        )
