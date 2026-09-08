"""Tests for the ARIEL read-only SQL query module.

Covers three surfaces of ``search/sql_query.py``:

- ``validate_sql_query`` — the allowlist/denylist gate that stands between an
  agent-authored string and the database.
- ``sql_query`` — the read-only transaction envelope, driven against the
  shared ``_FakePool``.
- ``format_sql_result`` — the agent-facing rendering of returned rows.

The module-level ``ALLOWED_TABLES`` / ``FORBIDDEN_KEYWORDS`` sets are the
security surface: tests read them and pin their contents, but never mutate
them. Variant-set cases go through ``monkeypatch.setattr`` with a *copy*.
"""

from __future__ import annotations

import importlib
import logging

import pytest
from psycopg.rows import dict_row
from pydantic import ValidationError

from osprey.services.ariel_search.search.sql_query import (
    ALLOWED_TABLES,
    FORBIDDEN_KEYWORDS,
    MAX_ROWS,
    SqlQueryInput,
    format_sql_result,
    sql_query,
    validate_sql_query,
)

# search/__init__.py re-exports the sql_query *function*, which shadows the
# submodule of the same name on the package -- so the module handle that
# monkeypatch needs has to come from sys.modules, not from an attribute lookup.
sql_query_module = importlib.import_module("osprey.services.ariel_search.search.sql_query")

# The statements sql_query wraps around every user query, in order.
BEGIN_SQL = "BEGIN READ ONLY"
TIMEOUT_SQL = "SET LOCAL statement_timeout = '10s'"
ROLLBACK_SQL = "ROLLBACK"


class TestSecuritySurface:
    """The allowlist and denylist contents are themselves a contract."""

    def test_allowed_tables_pinned(self):
        """Widening the table allowlist must be a deliberate edit, not a drift."""
        assert ALLOWED_TABLES == {"enhanced_entries", "text_embeddings"}

    def test_forbidden_keywords_pinned(self):
        """Every DML/DDL/DCL verb the validator rejects, enumerated."""
        assert FORBIDDEN_KEYWORDS == {
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

    def test_max_rows_ceiling(self):
        assert MAX_ROWS == 200


class TestValidateSqlQueryAccepts:
    """Queries that clear the gate."""

    def test_plain_select(self):
        validate_sql_query("SELECT * FROM enhanced_entries LIMIT 10")

    def test_lowercase_select(self):
        """The start check runs against the uppercased query."""
        validate_sql_query("select entry_id from enhanced_entries")

    def test_leading_whitespace_and_newlines(self):
        validate_sql_query("\n   SELECT entry_id\n   FROM enhanced_entries\n")

    def test_trailing_semicolon_is_stripped(self):
        """One trailing semicolon is normalized away, not read as a second statement."""
        validate_sql_query("SELECT * FROM enhanced_entries;")

    def test_prefixed_embedding_table(self):
        """``text_embeddings_*`` tables match the allowlist by prefix."""
        validate_sql_query("SELECT * FROM text_embeddings_nomic_embed_text LIMIT 5")

    def test_cte_name_is_not_a_table(self):
        """A name bound by ``WITH x AS`` is skipped by the FROM/JOIN allowlist check."""
        validate_sql_query(
            "WITH stats AS (SELECT author FROM enhanced_entries) SELECT * FROM stats"
        )

    def test_join_between_allowed_tables(self):
        validate_sql_query(
            "SELECT e.entry_id FROM enhanced_entries e "
            "JOIN text_embeddings_nomic t ON t.entry_id = e.entry_id"
        )

    def test_forbidden_keywords_match_on_word_boundaries(self):
        """``UPDATED_AT`` / ``CREATED_AT`` contain UPDATE and CREATE but are columns."""
        validate_sql_query("SELECT updated_at, created_at FROM enhanced_entries")


class TestValidateSqlQueryRejects:
    """Queries the gate turns away, and the message each one produces."""

    @pytest.mark.parametrize("query", ["", "   ", "\n\t "])
    def test_empty_query(self, query):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_sql_query(query)

    def test_non_select_statement_reports_first_token(self):
        with pytest.raises(ValueError, match="Only SELECT and WITH") as exc_info:
            validate_sql_query("EXPLAIN SELECT * FROM enhanced_entries")
        assert "'EXPLAIN'" in str(exc_info.value)

    def test_multi_statement(self):
        with pytest.raises(ValueError, match="Multi-statement queries are not allowed"):
            validate_sql_query("SELECT 1 FROM enhanced_entries; SELECT 2 FROM enhanced_entries")

    def test_semicolon_survives_trailing_strip(self):
        """Stripping the trailing ``;`` still leaves the injected one in the body."""
        with pytest.raises(ValueError, match="Multi-statement queries are not allowed"):
            validate_sql_query("SELECT 1 FROM enhanced_entries; SELECT 2 FROM enhanced_entries;")

    @pytest.mark.parametrize("keyword", sorted(FORBIDDEN_KEYWORDS))
    def test_every_forbidden_keyword(self, keyword):
        """Each denied verb is caught even mid-query, after a valid SELECT prefix."""
        with pytest.raises(ValueError, match=f"Forbidden keyword '{keyword}'"):
            validate_sql_query(f"SELECT entry_id FROM enhanced_entries {keyword} something")

    def test_forbidden_keyword_lowercase(self):
        """The keyword scan runs on the uppercased text, so case does not evade it."""
        with pytest.raises(ValueError, match="Forbidden keyword 'DROP'"):
            validate_sql_query("SELECT * FROM enhanced_entries where 1=1 drop table foo")

    def test_table_outside_allowlist(self):
        with pytest.raises(ValueError, match="Table 'users' is not in the allowlist"):
            validate_sql_query("SELECT * FROM users")

    def test_joined_table_outside_allowlist(self):
        """JOIN targets are checked, not only the FROM table."""
        with pytest.raises(ValueError, match="Table 'pg_authid' is not in the allowlist"):
            validate_sql_query("SELECT * FROM enhanced_entries JOIN pg_authid ON pg_authid.oid = 1")

    def test_lookalike_table_prefix_is_not_enough(self):
        """Prefix matching requires the ``_`` separator, so ``enhanced_entriesx`` fails."""
        with pytest.raises(ValueError, match="Table 'enhanced_entriesx'"):
            validate_sql_query("SELECT * FROM enhanced_entriesx")

    def test_cte_body_table_still_checked(self):
        """Binding a CTE name does not launder the table it selects from."""
        with pytest.raises(ValueError, match="Table 'pg_stat_activity'"):
            validate_sql_query("WITH x AS (SELECT * FROM pg_stat_activity) SELECT * FROM x")

    def test_server_side_file_read_names_no_table(self):
        """A query with no FROM at all had nothing to check, so it used to pass."""
        with pytest.raises(ValueError, match="reads no allowlisted table"):
            validate_sql_query("SELECT pg_read_file('/etc/passwd')")

    def test_tableless_select_is_refused(self):
        """Same shape without the file read: a SELECT that reads nothing is refused."""
        with pytest.raises(ValueError, match="reads no allowlisted table"):
            validate_sql_query("SELECT version()")

    def test_cte_only_query_is_refused(self):
        """Every reference resolving to a CTE leaves no real table behind."""
        with pytest.raises(ValueError, match="reads no allowlisted table"):
            validate_sql_query("WITH x AS (SELECT 1 AS n) SELECT * FROM x")

    def test_the_refusal_names_the_allowed_tables(self):
        """The message an agent reads says what it may query instead."""
        with pytest.raises(ValueError) as exc_info:
            validate_sql_query("SELECT pg_read_file('/etc/passwd')")
        assert "enhanced_entries" in str(exc_info.value)


class TestFromListRule:
    """The FROM/JOIN shape rule.

    The allowlist resolves one identifier per ``FROM``/``JOIN``. Every shape
    below reaches a second table that resolution never sees, so the validator
    refuses the shape rather than trying to resolve it.
    """

    def test_comma_list_after_from(self):
        with pytest.raises(
            ValueError, match="comma-separated FROM lists are not allowed; use JOIN"
        ):
            validate_sql_query("SELECT * FROM enhanced_entries, pg_shadow")

    def test_comma_list_with_aliases(self):
        """An alias between the table and the comma does not change the shape."""
        with pytest.raises(
            ValueError, match="comma-separated FROM lists are not allowed; use JOIN"
        ):
            validate_sql_query("SELECT * FROM enhanced_entries e, pg_shadow s")

    def test_comma_after_a_join_condition(self):
        """``ON`` does not end the FROM clause, so the comma is still a FROM list."""
        with pytest.raises(
            ValueError, match="comma-separated FROM lists are not allowed; use JOIN"
        ):
            validate_sql_query(
                "SELECT * FROM enhanced_entries e "
                "JOIN text_embeddings_nomic t ON t.entry_id = e.entry_id, pg_shadow"
            )

    def test_comma_after_a_parenthesised_subquery(self):
        """Closing the subquery returns to a FROM clause that is still open."""
        with pytest.raises(
            ValueError, match="comma-separated FROM lists are not allowed; use JOIN"
        ):
            validate_sql_query(
                "SELECT * FROM (SELECT entry_id FROM enhanced_entries) AS sub, pg_shadow"
            )

    def test_comma_before_a_set_returning_function(self):
        """The second FROM item need not be a table at all."""
        with pytest.raises(
            ValueError, match="comma-separated FROM lists are not allowed; use JOIN"
        ):
            validate_sql_query("SELECT * FROM enhanced_entries e, pg_ls_dir('/') d")

    def test_quoted_identifier_is_refused(self):
        """A quoted name is invisible to an unquoted-identifier scan."""
        with pytest.raises(ValueError, match="Quoted identifiers are not allowed"):
            validate_sql_query('SELECT * FROM enhanced_entries, "pg_shadow"')

    def test_block_comment_is_refused(self):
        """A comment can carry a parenthesis that would desynchronise the scan."""
        with pytest.raises(ValueError, match="Block comments are not allowed"):
            validate_sql_query("SELECT * FROM enhanced_entries /* ( */ , pg_shadow")

    def test_line_comment_is_refused(self):
        with pytest.raises(ValueError, match="Line comments are not allowed"):
            validate_sql_query("SELECT * FROM enhanced_entries -- (\n, pg_shadow")

    def test_backslash_is_refused(self):
        """An ``E'...'`` string treats ``\\'`` as an escaped quote; the scan does not.

        After ``E'\\''`` the scan and Postgres disagree on where the string
        ends, so the FROM/JOIN targets Postgres reads are inside a literal to
        the scan. The query below read ``pg_shadow`` on the pre-fix code.
        """
        with pytest.raises(ValueError, match="Backslash escapes are not allowed"):
            validate_sql_query(
                "SELECT E'\\'' FROM pg_shadow WHERE usename <> ' FROM enhanced_entries '"
            )

    def test_parenthesised_join_is_refused(self):
        """``FROM (`` may only open a subquery, never a join expression."""
        with pytest.raises(ValueError, match="must be a subquery starting with SELECT or WITH"):
            validate_sql_query("SELECT * FROM (pg_shadow s JOIN enhanced_entries e ON true)")

    # Postgres's ``TABLE name`` is ``SELECT * FROM name`` with neither keyword,
    # legal as a CTE body, a set-operation branch and a subquery. The scan reads
    # it as a third binder, so the name after it is a table the allowlist checks.

    def test_table_command_as_a_cte_body_is_checked(self):
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query("WITH x AS (TABLE pg_shadow) SELECT * FROM x")

    def test_table_command_as_a_cte_body_is_checked_behind_a_decoy_read(self):
        """The decoy ``FROM enhanced_entries`` satisfied the allowlist; the body did not."""
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "WITH x AS (TABLE pg_shadow) SELECT usename FROM enhanced_entries JOIN x ON true"
            )

    def test_table_command_as_a_union_branch_is_checked(self):
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query("SELECT * FROM enhanced_entries UNION ALL TABLE pg_shadow")

    def test_table_command_as_a_from_subquery_is_refused(self):
        """``FROM (TABLE ...)`` stays on the paren rule: only SELECT/WITH open a subquery."""
        with pytest.raises(ValueError, match="must be a subquery starting with SELECT or WITH"):
            validate_sql_query("SELECT * FROM (TABLE pg_shadow) s")

    def test_table_command_with_a_comma_list_is_refused(self):
        with pytest.raises(ValueError, match="comma-separated FROM lists are not allowed"):
            validate_sql_query("WITH x AS (TABLE enhanced_entries, pg_shadow) SELECT * FROM x")

    def test_table_command_on_an_allowlisted_table_is_allowed(self):
        validate_sql_query("WITH x AS (TABLE enhanced_entries) SELECT * FROM x")

    def test_multi_cte_query_is_allowed(self):
        """Every ``, name AS (`` in a WITH list binds a CTE name, not just the first."""
        validate_sql_query(
            "WITH a AS (SELECT entry_id FROM enhanced_entries), "
            "b AS (SELECT entry_id FROM enhanced_entries) "
            "SELECT * FROM a JOIN b ON a.entry_id = b.entry_id"
        )

    def test_commas_outside_a_from_clause_are_allowed(self):
        """Select lists and ORDER BY are comma-separated by construction."""
        validate_sql_query(
            "SELECT entry_id, author FROM enhanced_entries ORDER BY entry_id, author"
        )

    def test_comma_inside_an_in_list_is_allowed(self):
        validate_sql_query("SELECT * FROM enhanced_entries WHERE entry_id IN (1, 2)")

    def test_function_read_on_an_allowlisted_table_still_passes(self):
        """EXPECTED PASS, pinned deliberately.

        Naming an allowlisted table satisfies the allowlist, so a server-side
        function call in the select list goes through. Bounding what the
        session may read is the read-only database role's job — task A2
        ``sql-readonly-role`` — not the allowlist's, and this case is pinned
        here so that gap is recorded rather than implied closed.
        """
        validate_sql_query("SELECT pg_read_file('/etc/passwd') FROM enhanced_entries LIMIT 1")


class TestCteNameBinding:
    """Only a real ``WITH`` list binds a name the allowlist then skips.

    A bound CTE name is removed from the allowlist check, so anything that
    binds a name a query did not declare hands a joined table a free pass. The
    FROM/JOIN targets and the CTE names come out of one token pass, which
    consumes a single-quoted string whole, and a name only binds inside an open
    WITH list.
    """

    def test_literal_mimicking_a_later_cte_binds_no_name(self):
        """A `, name AS (` inside a literal must not launder a joined table."""
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "SELECT ', pg_shadow AS (' FROM enhanced_entries JOIN pg_shadow ON true"
            )

    def test_literal_mimicking_a_with_clause_binds_no_name(self):
        """Same bypass through the first CTE's `WITH name AS` spelling."""
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "SELECT 'WITH pg_shadow AS ' FROM enhanced_entries JOIN pg_shadow ON true"
            )

    def test_window_alias_does_not_bind_a_cte_name(self):
        """``, name AS (`` binds a CTE only inside a WITH list, not in a WINDOW list."""
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "SELECT * FROM enhanced_entries JOIN pg_shadow ON true "
                "WINDOW w AS (), pg_shadow AS ()"
            )

    # Postgres scopes a CTE to the query that declares it: a name bound inside a
    # subquery is a real relation at the outer level, so the allowlist must
    # resolve each FROM/JOIN reference against the declarations visible THERE.

    def test_cte_declared_in_a_from_subquery_does_not_cover_an_outer_join(self):
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "SELECT * FROM (WITH pg_shadow AS (SELECT 1 AS n) "
                "SELECT * FROM enhanced_entries) s JOIN pg_shadow ON true"
            )

    def test_cte_declared_in_a_scalar_subquery_does_not_cover_an_outer_join(self):
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "SELECT (WITH pg_shadow AS (SELECT 1 AS n) SELECT n FROM pg_shadow) "
                "FROM enhanced_entries JOIN pg_shadow ON true"
            )

    def test_cte_declared_in_an_in_subquery_does_not_cover_a_union_branch(self):
        with pytest.raises(ValueError, match="Table 'pg_authid' is not in the allowlist"):
            validate_sql_query(
                "SELECT * FROM enhanced_entries WHERE x IN (WITH pg_authid AS (SELECT 1) "
                "SELECT 1) UNION SELECT * FROM pg_authid"
            )

    def test_non_recursive_cte_body_cannot_see_its_own_name(self):
        """Without RECURSIVE a CTE's name is bound only after its body closes.

        Inside the body the name is the real table, so a CTE named after a
        catalog table reads that table from its own body.
        """
        with pytest.raises(ValueError, match="Table 'pg_shadow' is not in the allowlist"):
            validate_sql_query(
                "WITH pg_shadow AS (SELECT usename FROM pg_shadow) "
                "SELECT * FROM enhanced_entries JOIN pg_shadow ON true"
            )

    def test_cte_declared_in_a_subquery_covers_a_reference_in_that_subquery(self):
        validate_sql_query(
            "SELECT * FROM (WITH x AS (SELECT * FROM enhanced_entries) SELECT * FROM x) s"
        )

    def test_outer_cte_covers_a_reference_inside_a_subquery(self):
        validate_sql_query(
            "WITH x AS (SELECT entry_id FROM enhanced_entries) "
            "SELECT * FROM enhanced_entries WHERE entry_id IN (SELECT entry_id FROM x)"
        )


class TestValidateSqlQueryVariantSets:
    """The two constants are read at call time — swap in copies, never mutate."""

    def test_widened_table_allowlist(self, monkeypatch):
        widened = set(ALLOWED_TABLES) | {"ariel_runs"}
        monkeypatch.setattr(sql_query_module, "ALLOWED_TABLES", widened)

        validate_sql_query("SELECT * FROM ariel_runs")

        assert ALLOWED_TABLES == {"enhanced_entries", "text_embeddings"}

    def test_narrowed_table_allowlist(self, monkeypatch):
        narrowed = {"text_embeddings"}
        monkeypatch.setattr(sql_query_module, "ALLOWED_TABLES", narrowed)

        with pytest.raises(ValueError, match="Table 'enhanced_entries'"):
            validate_sql_query("SELECT * FROM enhanced_entries")

        assert ALLOWED_TABLES == {"enhanced_entries", "text_embeddings"}

    def test_extended_forbidden_keywords(self, monkeypatch):
        extended = set(FORBIDDEN_KEYWORDS) | {"ANALYZE"}
        monkeypatch.setattr(sql_query_module, "FORBIDDEN_KEYWORDS", extended)

        with pytest.raises(ValueError, match="Forbidden keyword 'ANALYZE'"):
            validate_sql_query("SELECT * FROM enhanced_entries ANALYZE")

        assert "ANALYZE" not in FORBIDDEN_KEYWORDS


class TestSqlQueryInput:
    """The pydantic input schema exposed to the agent."""

    def test_defaults(self):
        model = SqlQueryInput(query="SELECT 1 FROM enhanced_entries")
        assert model.max_rows == 100

    def test_max_rows_upper_bound(self):
        assert SqlQueryInput(query="SELECT 1", max_rows=MAX_ROWS).max_rows == MAX_ROWS
        with pytest.raises(ValidationError):
            SqlQueryInput(query="SELECT 1", max_rows=MAX_ROWS + 1)

    def test_max_rows_lower_bound(self):
        assert SqlQueryInput(query="SELECT 1", max_rows=1).max_rows == 1
        with pytest.raises(ValidationError):
            SqlQueryInput(query="SELECT 1", max_rows=0)

    def test_query_is_required(self):
        with pytest.raises(ValidationError):
            SqlQueryInput()

    def test_schema_is_not_the_validator(self):
        """The model accepts any string; safety comes from validate_sql_query."""
        assert SqlQueryInput(query="DROP TABLE enhanced_entries").query


class TestSqlQueryExecution:
    """``sql_query`` against the shared fake pool."""

    QUERY = "SELECT entry_id FROM enhanced_entries LIMIT 5"

    async def test_returns_scripted_rows(self, fake_pool_factory):
        pool = fake_pool_factory(rows_for={"FROM enhanced_entries": [{"entry_id": "e-1"}]})

        rows = await sql_query(pool, self.QUERY)

        assert rows == [{"entry_id": "e-1"}]

    async def test_read_only_transaction_envelope(self, fake_pool_factory):
        """Every query is bracketed by BEGIN READ ONLY, a timeout, and ROLLBACK."""
        pool = fake_pool_factory(rows_for={"FROM enhanced_entries": [{"entry_id": "e-1"}]})

        await sql_query(pool, self.QUERY)

        assert pool.sql == [BEGIN_SQL, TIMEOUT_SQL, self.QUERY, ROLLBACK_SQL]

    async def test_cursor_requests_dict_row(self, fake_pool_factory):
        pool = fake_pool_factory()

        await sql_query(pool, self.QUERY)

        assert [cur.row_factory for cur in pool.conn.cursors] == [dict_row]

    async def test_empty_result_set(self, fake_pool):
        assert await sql_query(fake_pool, self.QUERY) == []

    async def test_rows_are_copied_into_plain_dicts(self, fake_pool_factory):
        """``dict(row)`` detaches the result from whatever the driver handed back."""
        scripted = {"entry_id": "e-1"}
        pool = fake_pool_factory(rows_for={"FROM enhanced_entries": [scripted]})

        rows = await sql_query(pool, self.QUERY)

        assert rows[0] == scripted
        assert rows[0] is not scripted

    async def test_max_rows_limits_the_fetch(self, fake_pool_factory):
        pool = fake_pool_factory(
            rows_for={"FROM enhanced_entries": [{"entry_id": f"e-{i}"} for i in range(5)]}
        )

        rows = await sql_query(pool, self.QUERY, max_rows=2)

        assert [row["entry_id"] for row in rows] == ["e-0", "e-1"]

    async def test_max_rows_capped_at_module_ceiling(self, fake_pool_factory, monkeypatch):
        """A caller asking for more than MAX_ROWS is silently clamped down to it."""
        monkeypatch.setattr(sql_query_module, "MAX_ROWS", 2)
        pool = fake_pool_factory(
            rows_for={"FROM enhanced_entries": [{"entry_id": f"e-{i}"} for i in range(5)]}
        )

        rows = await sql_query(pool, self.QUERY, max_rows=100)

        assert len(rows) == 2

    async def test_validation_runs_before_any_sql(self, fake_pool):
        """A rejected query never reaches the pool — not even the BEGIN."""
        with pytest.raises(ValueError, match="Forbidden keyword 'DROP'"):
            await sql_query(fake_pool, "SELECT * FROM enhanced_entries DROP TABLE foo")

        assert fake_pool.calls == []

    async def test_rollback_runs_when_the_query_raises(self, fake_pool_factory):
        """The ``finally`` arm keeps the transaction from being left open."""
        boom = RuntimeError("relation does not exist")
        pool = fake_pool_factory(rows_for={"FROM enhanced_entries": boom})

        with pytest.raises(RuntimeError, match="relation does not exist"):
            await sql_query(pool, self.QUERY)

        assert pool.sql == [BEGIN_SQL, TIMEOUT_SQL, self.QUERY, ROLLBACK_SQL]

    async def test_logs_the_effective_max_rows(self, fake_pool, caplog):
        with caplog.at_level(logging.INFO, logger="ariel"):
            await sql_query(fake_pool, self.QUERY, max_rows=50)

        assert "sql_query: executing query (max_rows=50)" in caplog.text

    async def test_logged_max_rows_reflects_the_cap(self, fake_pool, caplog):
        with caplog.at_level(logging.INFO, logger="ariel"):
            await sql_query(fake_pool, self.QUERY, max_rows=1000)

        assert f"max_rows={MAX_ROWS}" in caplog.text


class TestFormatSqlResult:
    """Agent-facing rendering of the row list."""

    def test_no_rows(self):
        assert format_sql_result([]) == "No results found."

    def test_header_counts_rows(self):
        out = format_sql_result([{"a": 1}, {"a": 2}])
        assert out.splitlines()[0] == "Results: 2 row(s)"

    def test_rows_are_numbered_from_one(self):
        out = format_sql_result([{"a": 1}, {"a": 2}])
        assert "--- Row 1 ---" in out
        assert "--- Row 2 ---" in out

    def test_columns_rendered_as_key_value_lines(self):
        out = format_sql_result([{"entry_id": "e-1", "author": "jsmith"}])
        assert "  entry_id: e-1" in out
        assert "  author: jsmith" in out

    def test_non_string_values_are_stringified(self):
        out = format_sql_result([{"count": 42, "missing": None}])
        assert "  count: 42" in out
        assert "  missing: None" in out

    def test_long_values_truncated_at_200_chars(self):
        out = format_sql_result([{"raw_text": "x" * 500}])
        rendered = next(line for line in out.splitlines() if line.startswith("  raw_text: "))
        value = rendered.removeprefix("  raw_text: ")
        assert value == "x" * 200 + "..."

    def test_value_at_the_truncation_boundary_is_kept_whole(self):
        out = format_sql_result([{"raw_text": "x" * 200}])
        assert "  raw_text: " + "x" * 200 in out
        assert "..." not in out
