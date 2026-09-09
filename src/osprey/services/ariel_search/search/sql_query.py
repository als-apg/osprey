"""ARIEL SQL query module.

Provides read-only SQL access to the ARIEL database with allowlist
validation and safety constraints. This module bypasses the search
service — it's raw DB access, not a search module.

Unlike keyword/semantic modules, this does NOT follow the
SearchToolDescriptor pattern (that's for the LangChain agent executor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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

# Lexical shapes the FROM-list scan cannot read, refused outright rather than
# resolved: a quoted identifier hides a table name from an unquoted-identifier
# scan, a comment can carry a parenthesis or a comma that desynchronises it,
# dollar quoting opens a string the scan does not terminate, and a backslash in
# an ``E'...'`` string escapes the quote (``E'\''``), which desynchronises the
# scan from the server's lexer. No agent query needs any of them.
REFUSED_LEXEMES = (
    ('"', "Quoted identifiers"),
    ("--", "Line comments"),
    ("/*", "Block comments"),
    ("$", "Dollar quoting and dollar parameters"),
    ("\\", "Backslash escapes"),
)

# Identifiers, single-quoted strings (skipped whole, so a comma or parenthesis
# inside one is not read as syntax), parentheses and commas.
_TOKEN_RE = re.compile(r"'(?:[^']|'')*'|[a-zA-Z_][a-zA-Z0-9_]*|[(),]")

# Keywords that end a FROM clause at their own paren depth. Anything else --
# an alias, ``AS``, ``ON`` and its condition -- leaves the clause open. A
# modifier between the keyword and the name (``FROM ONLY t``, ``JOIN LATERAL
# (``, ``WITH RECURSIVE t AS``, ``WITH t(a) AS``, ``AS MATERIALIZED``) is read
# as the name itself, so those forms are refused as an unknown table rather
# than resolved; the rule is not grown to read them. The same goes for the
# ``FROM`` inside ``EXTRACT(... FROM col)``, ``TRIM(... FROM col)`` and
# ``POSITION(... IN col)``-family calls: the column is read as a table name.
_FROM_ENDING_KEYWORDS = frozenset(
    {
        "WHERE",
        "GROUP",
        "HAVING",
        "ORDER",
        "LIMIT",
        "OFFSET",
        "WINDOW",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "FETCH",
        "FOR",
        "SELECT",
    }
)


@dataclass
class _Scope:
    """Scan state for one paren depth, scoped the way Postgres scopes it.

    ``ctes`` holds the CTE names a reference at this depth (or deeper) may
    resolve to; ``pending`` is a declared name whose body has not closed yet --
    without ``RECURSIVE`` the body cannot see its own name, so it binds only
    when the body's ``)`` returns to this scope.
    """

    in_from: bool = False
    in_with: bool = False
    ctes: set[str] = field(default_factory=set)
    pending: str | None = None


def _scan_relations(normalized: str) -> list[str]:
    """Return the ``FROM``/``JOIN``/``TABLE`` targets that are not a CTE in scope.

    One pass over one token stream is the only reader of the query text. A
    single-quoted string is consumed whole, so nothing inside a literal is
    read as syntax (a backslash, the one escape the scan cannot follow, is
    refused up front). A CTE name is resolved where the reference is read,
    against the declarations visible at that depth: a ``WITH`` inside a
    subquery does not cover a ``JOIN`` at the top level.

    The allowlist resolves one name per ``FROM``/``JOIN``/``TABLE`` (Postgres's
    ``TABLE name`` is ``SELECT * FROM name`` without the keyword), so any shape
    that reaches a second relation from the same clause reaches a table this
    scan never sees. Rather than parse those shapes, the scan refuses them: a
    comma-separated FROM list, and a parenthesised join in place of a subquery.

    Args:
        normalized: The query, stripped and without its trailing semicolon.

    Returns:
        The referenced table names in the order they appear.

    Raises:
        ValueError: If the query carries a refused lexeme or FROM shape.
    """
    for lexeme, label in REFUSED_LEXEMES:
        if lexeme in normalized:
            raise ValueError(f"{label} are not allowed in a query; remove {lexeme!r}.")

    tokens = _TOKEN_RE.findall(normalized)
    refs: list[str] = []
    # One frame per paren depth; a closing paren discards its frame, so the
    # clause the subquery interrupted is still open after it, and a CTE the
    # subquery declared is gone with it.
    scopes = [_Scope()]

    def declares_cte(index: int) -> bool:
        """Is the token at ``index`` followed by ``AS (``, binding a CTE name?

        The trailing "(" is what keeps a select-list alias (", author AS name")
        from laundering a table name into the CTE set.
        """
        return [t.upper() for t in tokens[index + 1 : index + 3]] == ["AS", "("]

    for index, token in enumerate(tokens):
        previous = tokens[index - 1].upper() if index else ""
        upper = token.upper()
        scope = scopes[-1]

        if token == "(":
            if previous in ("FROM", "JOIN"):
                following = tokens[index + 1].upper() if index + 1 < len(tokens) else ""
                if following not in ("SELECT", "WITH"):
                    raise ValueError(
                        "A parenthesised FROM/JOIN target must be a subquery starting "
                        "with SELECT or WITH."
                    )
            scopes.append(_Scope())
        elif token == ")":
            if len(scopes) > 1:
                scopes.pop()
                if scopes[-1].pending:  # the CTE body just closed; its name binds now
                    scopes[-1].ctes.add(scopes[-1].pending)
                    scopes[-1].pending = None
        elif token == ",":
            if scope.in_from:
                raise ValueError("comma-separated FROM lists are not allowed; use JOIN")
        elif token.startswith("'"):
            continue  # a string literal is a value, never syntax
        elif upper in ("FROM", "JOIN", "TABLE"):
            scope.in_from = True
        elif upper == "WITH":
            scope.in_with = True
        elif upper in _FROM_ENDING_KEYWORDS:
            # The statement's own SELECT ends the WITH list along with the clause.
            scope.in_from = scope.in_with = False
        elif previous in ("FROM", "JOIN", "TABLE"):
            if not any(token.lower() in s.ctes for s in scopes):
                refs.append(token)
        elif scope.in_with and previous in ("WITH", ",") and declares_cte(index):
            # ``WITH name AS (`` and every later ``, name AS (`` in that list.
            # Requiring an open WITH list is what stops a ``, name AS (`` in a
            # WINDOW list from binding the name of a table the query joins.
            scope.pending = token.lower()

    return refs


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

    Five rules, all of which must hold:

    - starts with SELECT or WITH (for CTEs);
    - one statement — no semicolons in the body;
    - no DML/DDL/DCL keyword anywhere;
    - every FROM/JOIN target is a shape the allowlist can resolve — no comma
      list, no quoted identifier, no comment, no dollar quoting;
    - reads at least one allowlisted table, and no table outside the
      allowlist.

    That last rule is why the check is an allowlist rather than a denylist, and
    why "names none" is a refusal rather than a pass. A query with no FROM or
    JOIN at all — ``SELECT pg_read_file('/etc/passwd')`` — has nothing to
    check against the allowlist, and the read-only transaction the caller opens
    stops writes, not server-side file reads. So a query that resolves to no
    allowlisted table is refused on that ground alone. A query that *does* name
    one — ``SELECT pg_read_file('/etc/passwd') FROM enhanced_entries`` — still
    passes here: bounding what a function call may read is the read-only
    database role's job, not the allowlist's.

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

    # Table allowlist check. One tokenised pass yields the FROM/JOIN targets
    # with every CTE reference already resolved in its own scope (a CTE is a
    # query of its own, checked by whatever IT reads), refusing the shapes that
    # reach a relation this resolution would never see.
    table_refs = _scan_relations(normalized)

    for table_ref in table_refs:
        table_lower = table_ref.lower()
        # Check exact match or prefix match (for text_embeddings_* tables)
        if not any(
            table_lower == allowed or table_lower.startswith(f"{allowed}_")
            for allowed in ALLOWED_TABLES
        ):
            raise ValueError(
                f"Table '{table_ref}' is not in the allowlist. "
                f"Allowed tables: enhanced_entries, text_embeddings_*"
            )

    # Nothing to check IS the failure: the loop above passes vacuously for a
    # query that reads no table, which is exactly the shape a server-side file
    # read takes.
    if not table_refs:
        raise ValueError(
            "Query reads no allowlisted table. Allowed tables: enhanced_entries, text_embeddings_*"
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
