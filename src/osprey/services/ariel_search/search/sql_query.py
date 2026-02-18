"""ARIEL SQL query module.

Provides read-only SQL access to the ARIEL database with allowlist
validation and safety constraints. This module bypasses the search
service — it's raw DB access, not a search module.

Unlike keyword/semantic modules, this does NOT follow the
SearchToolDescriptor pattern (that's for the LangChain agent executor).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from osprey.utils.logger import get_logger

logger = get_logger("ariel")

# Allowed tables (allowlist — reject everything else)
ALLOWED_TABLES = {"enhanced_entries", "text_embeddings"}

# Forbidden keywords (DML/DDL/DCL)
FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "COPY",
    "GRANT",
    "REVOKE",
    "VACUUM",
    "SET",
    "EXECUTE",
}

# Maximum rows per query
MAX_ROWS = 200


class SqlQueryInput(BaseModel):
    """Input schema for SQL query tool."""

    query: str = Field(description="Read-only SQL query (SELECT or WITH only)")
    max_rows: int = Field(
        default=100,
        ge=1,
        le=MAX_ROWS,
        description=f"Maximum rows to return (1-{MAX_ROWS})",
    )


def validate_sql_query(query: str) -> None:
    """Validate that a SQL query is safe to execute.

    Uses an allowlist approach:
    - Must start with SELECT or WITH (for CTEs)
    - No multi-statement injection (semicolons)
    - No DML/DDL/DCL keywords
    - Only allowed tables

    Args:
        query: The SQL query to validate.

    Raises:
        ValueError: If the query fails validation.
    """
    if not query or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    # Normalize: strip whitespace, remove trailing semicolons
    normalized = query.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()

    upper = normalized.upper()

    # Must start with SELECT or WITH
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError(
            "Only SELECT and WITH (CTE) queries are allowed. "
            f"Query starts with: {normalized.split()[0]!r}"
        )

    # Reject multi-statement (semicolons in the body)
    if ";" in normalized:
        raise ValueError(
            "Multi-statement queries are not allowed. Remove semicolons from the query body."
        )

    # Check for forbidden keywords
    # Use word boundary matching to avoid false positives (e.g. "UPDATED_AT")
    for keyword in FORBIDDEN_KEYWORDS:
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, upper):
            raise ValueError(
                f"Forbidden keyword '{keyword}' found in query. "
                "Only read-only SELECT queries are allowed."
            )

    # Table allowlist check
    # Extract CTE names (WITH x AS ...) so we can skip them in FROM/JOIN checks
    cte_pattern = r"\bWITH\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\b"
    cte_names = {name.lower() for name in re.findall(cte_pattern, normalized, re.IGNORECASE)}

    # Extract table references from FROM and JOIN clauses
    table_pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    table_refs = re.findall(table_pattern, normalized, re.IGNORECASE)

    for table_ref in table_refs:
        table_lower = table_ref.lower()
        # Skip CTE-defined names
        if table_lower in cte_names:
            continue
        # Check exact match or prefix match (for text_embeddings_* tables)
        if not any(
            table_lower == allowed or table_lower.startswith(f"{allowed}_")
            for allowed in ALLOWED_TABLES
        ):
            raise ValueError(
                f"Table '{table_ref}' is not in the allowlist. "
                f"Allowed tables: enhanced_entries, text_embeddings_*"
            )


async def sql_query(
    pool: Any,
    query: str,
    max_rows: int = 100,
) -> list[dict[str, Any]]:
    """Execute a read-only SQL query against the ARIEL database.

    Args:
        pool: psycopg async connection pool.
        query: SQL query (validated before execution).
        max_rows: Maximum rows to return (capped at MAX_ROWS).

    Returns:
        List of row dicts.

    Raises:
        ValueError: If the query fails validation.
    """
    validate_sql_query(query)

    # Cap max_rows
    max_rows = min(max_rows, MAX_ROWS)

    logger.info(f"sql_query: executing query (max_rows={max_rows})")

    from psycopg.rows import dict_row

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Read-only transaction with timeout
            await cur.execute("BEGIN READ ONLY")
            await cur.execute("SET LOCAL statement_timeout = '10s'")

            try:
                await cur.execute(query)
                rows = await cur.fetchmany(max_rows)
                return [dict(row) for row in rows]
            finally:
                await cur.execute("ROLLBACK")


def format_sql_result(rows: list[dict[str, Any]]) -> str:
    """Format SQL query results for agent consumption.

    Args:
        rows: List of row dicts from sql_query().

    Returns:
        Formatted string representation.
    """
    if not rows:
        return "No results found."

    # Build a simple tabular representation
    lines = [f"Results: {len(rows)} row(s)"]
    lines.append("")

    for i, row in enumerate(rows, 1):
        lines.append(f"--- Row {i} ---")
        for key, value in row.items():
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 200:
                str_val = str_val[:200] + "..."
            lines.append(f"  {key}: {str_val}")

    return "\n".join(lines)
