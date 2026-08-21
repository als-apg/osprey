"""Unit tests for :class:`ARIELRepository`, driven by the fake connection pool.

These tests assert on what the repository *emits* -- the SQL text, the bound
parameters, and the exception it raises when a query fails -- never on a
database outcome. Ordering, ranking, jsonb merge semantics and index behaviour
are the container-backed integration suite's job; pinning them here would only
pin the fake.

The error-wrap suite is the reason this file exists in bulk: nearly every
repository method ends in the same ``except Exception -> DatabaseQueryError``
shape, and the ``query=`` breadcrumb it attaches is the only thing that tells an
operator reading a log which statement died. The parametrized case below pins
one breadcrumb per method, and the two methods that deviate from the shape get
their own cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import ARIELRepository
from osprey.services.ariel_search.exceptions import (
    ConfigurationError,
    DatabaseQueryError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ARIELConfig:
    """Config with every search and enhancement module the repository gates on."""
    data: dict[str, Any] = {
        "database": {"uri": "postgresql://unit-test/ariel"},
        "search_modules": {
            "keyword": {"enabled": True},
            "semantic": {"enabled": True, "model": "nomic-embed-text"},
        },
        "enhancement_modules": {
            "text_embedding": {
                "enabled": True,
                "models": [{"name": "nomic-embed-text", "dimension": 768}],
            },
            "semantic_processor": {"enabled": True},
        },
    }
    data.update(overrides)
    return ARIELConfig.from_dict(data)


def _entry_row(**overrides: Any) -> dict[str, Any]:
    """One ``dict_row`` row shaped like ``enhanced_entries``."""
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    row: dict[str, Any] = {
        "entry_id": "e-1",
        "source_system": "als_logbook",
        "timestamp": now,
        "author": "operator",
        "raw_text": "beam lost at 03:04",
        "attachments": [],
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


def _sql_body(sql: str) -> str:
    """SQL with ``--`` comments dropped and whitespace collapsed.

    The upsert statement carries a long explanatory comment; assertions target
    the executable text so a reworded comment never passes or fails a test.
    """
    lines = [line.split("--", 1)[0] for line in sql.splitlines()]
    return " ".join(" ".join(lines).split())


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------

#: ``(method label, coroutine factory, expected ``query=`` breadcrumb)``.
ERROR_WRAP_CASES = [
    pytest.param(
        lambda r: r.get_entry("e-1"),
        "SELECT entry_id=e-1",
        id="get_entry",
    ),
    pytest.param(
        lambda r: r.get_entries_by_ids(["e-1", "e-2"]),
        "SELECT entry_ids=ANY([2 ids])",
        id="get_entries_by_ids",
    ),
    pytest.param(
        lambda r: r.upsert_entry(
            {
                "entry_id": "e-1",
                "source_system": "als_logbook",
                "timestamp": datetime(2026, 1, 2, tzinfo=UTC),
                "raw_text": "text",
            }
        ),
        "UPSERT entry_id=e-1",
        id="upsert_entry",
    ),
    pytest.param(
        lambda r: r.search_by_time_range(),
        "SELECT time_range=(None, None)",
        id="search_by_time_range",
    ),
    pytest.param(
        lambda r: r.count_entries(),
        "SELECT COUNT(*)",
        id="count_entries",
    ),
    pytest.param(
        lambda r: r.get_distinct_authors(),
        "SELECT DISTINCT author",
        id="get_distinct_authors",
    ),
    pytest.param(
        lambda r: r.get_distinct_source_systems(),
        "SELECT DISTINCT source_system",
        id="get_distinct_source_systems",
    ),
    pytest.param(
        lambda r: r.store_attachment("e-1", "a-1", "shot.png", "image/png", b"data", 4),
        "INSERT attachment_files attachment_id=a-1",
        id="store_attachment",
    ),
    pytest.param(
        lambda r: r.get_attachment("a-1"),
        "SELECT attachment_id=a-1",
        id="get_attachment",
    ),
    pytest.param(
        lambda r: r.get_attachments_for_entry("e-1"),
        "SELECT attachments entry_id=e-1",
        id="get_attachments_for_entry",
    ),
    pytest.param(
        lambda r: r.get_incomplete_entries(module_name="text_embedding"),
        "SELECT incomplete module=text_embedding",
        id="get_incomplete_entries",
    ),
    pytest.param(
        lambda r: r.get_enhancement_stats(),
        "SELECT enhancement_stats",
        id="get_enhancement_stats",
    ),
    pytest.param(
        lambda r: r.mark_enhancement_complete("e-1", "text_embedding"),
        "UPDATE entry_id=e-1 module=text_embedding",
        id="mark_enhancement_complete",
    ),
    pytest.param(
        lambda r: r.mark_enhancement_failed("e-1", "text_embedding", "nope"),
        "UPDATE entry_id=e-1 module=text_embedding",
        id="mark_enhancement_failed",
    ),
    pytest.param(
        lambda r: r.get_embedding_tables(),
        "SELECT embedding tables",
        id="get_embedding_tables",
    ),
    pytest.param(
        lambda r: r.validate_search_model_table("nomic-embed-text"),
        "SELECT table exists text_embeddings_nomic_embed_text",
        id="validate_search_model_table",
    ),
    pytest.param(
        lambda r: r.store_text_embedding("e-1", [0.1, 0.2], "nomic-embed-text"),
        "INSERT text_embeddings_nomic_embed_text entry_id=e-1",
        id="store_text_embedding",
    ),
    pytest.param(
        lambda r: r.keyword_search([], [], "quench"),
        "KEYWORD SEARCH: quench",
        id="keyword_search",
    ),
    pytest.param(
        lambda r: r.fuzzy_search("quench"),
        "FUZZY SEARCH: quench",
        id="fuzzy_search",
    ),
    pytest.param(
        lambda r: r.semantic_search([0.1, 0.2], "nomic-embed-text"),
        "SEMANTIC SEARCH model=nomic-embed-text",
        id="semantic_search",
    ),
    pytest.param(
        lambda r: r.start_ingestion_run("als_logbook"),
        "INSERT ingestion_runs",
        id="start_ingestion_run",
    ),
    pytest.param(
        lambda r: r.complete_ingestion_run(7, 1, 2, 3),
        "UPDATE ingestion_runs id=7",
        id="complete_ingestion_run",
    ),
    pytest.param(
        lambda r: r.fail_ingestion_run(7, "boom"),
        "UPDATE ingestion_runs id=7",
        id="fail_ingestion_run",
    ),
    pytest.param(
        lambda r: r.get_last_successful_run("als_logbook"),
        "SELECT MAX(completed_at) source_system=als_logbook",
        id="get_last_successful_run",
    ),
]


class TestErrorWrapping:
    """Every query method turns a driver failure into DatabaseQueryError."""

    @pytest.mark.parametrize(("call", "expected_query"), ERROR_WRAP_CASES)
    async def test_driver_failure_becomes_database_query_error(
        self,
        fake_pool_factory,
        call,
        expected_query: str,
    ) -> None:
        """A failed query is wrapped with the breadcrumb naming what was run.

        ``DatabaseQueryError.technical_details["query"]`` is the only record of
        *which* statement failed once the psycopg exception has been swallowed,
        so each method's breadcrumb is pinned individually. The original
        exception must stay chained -- dropping ``from e`` would leave an
        operator with a message and no traceback into the driver.
        """
        driver_error = RuntimeError("connection reset by peer")
        repo = ARIELRepository(fake_pool_factory(error=driver_error), _make_config())

        with pytest.raises(DatabaseQueryError) as exc_info:
            await call(repo)

        assert exc_info.value.technical_details["query"] == expected_query
        assert "connection reset by peer" in exc_info.value.message
        assert exc_info.value.__cause__ is driver_error

    async def test_validate_search_model_table_reraises_configuration_error(
        self,
        fake_pool_factory,
    ) -> None:
        """A missing embedding table stays a ConfigurationError, not a query error.

        Deviant from the shared shape: the ``except ConfigurationError: raise``
        arm sits ahead of the generic wrap. Without it the actionable
        "run 'osprey ariel migrate'" message would be reboxed as a database
        failure and read as a transient outage.
        """
        repo = ARIELRepository(fake_pool_factory(results=[[(False,)]]), _make_config())

        with pytest.raises(ConfigurationError) as exc_info:
            await repo.validate_search_model_table("nomic-embed-text")

        assert exc_info.value.technical_details["config_key"] == "search_modules.semantic.model"
        assert "text_embeddings_nomic_embed_text" in exc_info.value.message
        assert "osprey ariel migrate" in exc_info.value.message

    async def test_validate_search_model_table_treats_missing_row_as_absent(
        self,
        fake_pool_factory,
    ) -> None:
        """No row back from the EXISTS probe is read as "table absent"."""
        repo = ARIELRepository(fake_pool_factory(results=[[]]), _make_config())

        with pytest.raises(ConfigurationError):
            await repo.validate_search_model_table("nomic-embed-text")

    async def test_validate_search_model_table_passes_when_table_exists(
        self,
        fake_pool_factory,
    ) -> None:
        """An existing table validates silently and probes by table name."""
        pool = fake_pool_factory(results=[[(True,)]])
        repo = ARIELRepository(pool, _make_config())

        await repo.validate_search_model_table("nomic-embed-text")

        assert pool.calls[0][1] == ["text_embeddings_nomic_embed_text"]

    async def test_start_ingestion_run_reraises_its_own_query_error(
        self,
        fake_pool_factory,
    ) -> None:
        """The "no ID returned" DatabaseQueryError passes through unwrapped.

        Deviant from the shared shape: the method raises DatabaseQueryError from
        inside its own ``try``, so it needs ``except DatabaseQueryError: raise``
        ahead of the generic arm. Without it the specific message would be
        wrapped into a second, vaguer one naming an exception instead of the
        missing RETURNING row.
        """
        repo = ARIELRepository(fake_pool_factory(results=[[]]), _make_config())

        with pytest.raises(DatabaseQueryError) as exc_info:
            await repo.start_ingestion_run("als_logbook")

        assert exc_info.value.message == "Failed to start ingestion run: no ID returned"
        assert exc_info.value.__cause__ is None

    async def test_health_check_reports_failure_instead_of_raising(
        self,
        fake_pool_factory,
    ) -> None:
        """health_check is the one method that returns its failure.

        It backs a status endpoint, so an unreachable database has to come back
        as ``(False, message)``; raising would turn a red health tile into a 500.
        """
        repo = ARIELRepository(fake_pool_factory(error=RuntimeError("no route")), _make_config())

        healthy, message = await repo.health_check()

        assert healthy is False
        assert message == "Database unreachable: no route"

    async def test_health_check_probes_with_select_1(self, fake_pool) -> None:
        """A reachable database answers healthy after a trivial probe."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.health_check() == (True, "Database connected")
        assert fake_pool.sql == ["SELECT 1"]


# ---------------------------------------------------------------------------
# Re-ingest hazards (upsert_entry)
# ---------------------------------------------------------------------------


class TestUpsertEntryReingestHazards:
    """The ON CONFLICT clause is re-ingestion's only data-loss guard."""

    @staticmethod
    def _update_clause(pool) -> str:
        sql = _sql_body(pool.sql[0])
        assert "ON CONFLICT (entry_id) DO UPDATE SET" in sql
        return sql.split("DO UPDATE SET", 1)[1]

    async def test_upstream_owned_columns_are_overwritten_unconditionally(
        self,
        fake_pool,
        seed_entry_factory,
    ) -> None:
        """Pins the unconditional overwrite set in ON CONFLICT DO UPDATE.

        Hazard: a re-ingestion poll re-fetches entries that already exist, and
        those five columns are upstream's to own. If one is dropped from the
        DO UPDATE set, an entry edited in the source logbook silently stays
        frozen at whatever ARIEL first saw, with no error anywhere.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.upsert_entry(seed_entry_factory())

        update_clause = self._update_clause(fake_pool)
        for column in ("source_system", "timestamp", "author", "raw_text", "metadata"):
            assert f"{column} = EXCLUDED.{column}" in update_clause

    async def test_empty_incoming_attachments_preserve_stored_ones(
        self,
        fake_pool,
        seed_entry_factory,
    ) -> None:
        """Pins the ``'[]'::jsonb`` CASE that protects ARIEL-native attachments.

        Hazard: the upstream write API cannot accept file uploads, so an entry
        published by ARIEL comes back from the next poll with
        ``attachments = '[]'::jsonb``. Collapsing the CASE into a plain
        ``attachments = EXCLUDED.attachments`` erases web-uploaded attachments
        and orphans their stored blobs -- unrecoverable, and invisible until
        someone opens the entry.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.upsert_entry(seed_entry_factory(attachments=[]))

        update_clause = self._update_clause(fake_pool)
        assert (
            "attachments = CASE WHEN EXCLUDED.attachments = '[]'::jsonb "
            "THEN enhanced_entries.attachments ELSE EXCLUDED.attachments END" in update_clause
        )
        assert "attachments = EXCLUDED.attachments" not in update_clause

    async def test_enhancement_status_is_absent_from_the_update_set(
        self,
        fake_pool,
        seed_entry_factory,
    ) -> None:
        """Pins enhancement_status's absence from ON CONFLICT DO UPDATE.

        Hazard: the column is written on INSERT but must never be re-written on
        conflict. Adding it to the DO UPDATE set would reset every module on an
        already-enhanced entry back to the incoming (usually empty) status on
        each poll, silently re-queueing the whole corpus for re-enhancement.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.upsert_entry(seed_entry_factory(enhancement_status={"text_embedding": {}}))

        sql = _sql_body(fake_pool.sql[0])
        insert_clause, update_clause = sql.split("DO UPDATE SET", 1)
        assert "enhancement_status" in insert_clause
        assert "enhancement_status" not in update_clause

    async def test_json_columns_are_serialized_in_column_order(
        self,
        fake_pool,
        seed_entry_factory,
    ) -> None:
        """The three jsonb columns are bound as JSON text, positionally."""
        entry = seed_entry_factory(
            attachments=[{"url": "http://logbook.invalid/a.png"}],
            metadata={"logbook": "operations"},
            enhancement_status={"text_embedding": {"status": "complete"}},
        )
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.upsert_entry(entry)

        params = fake_pool.calls[0][1]
        assert params[:5] == [
            entry["entry_id"],
            entry["source_system"],
            entry["timestamp"],
            entry["author"],
            entry["raw_text"],
        ]
        assert params[5] == json.dumps(entry["attachments"])
        assert params[6] == json.dumps(entry["metadata"])
        assert params[7] == json.dumps(entry["enhancement_status"])

    async def test_missing_optional_fields_fall_back_to_empty(
        self,
        fake_pool,
    ) -> None:
        """An entry without author/attachments/metadata still binds every slot."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.upsert_entry(
            {
                "entry_id": "e-1",
                "source_system": "als_logbook",
                "timestamp": datetime(2026, 1, 2, tzinfo=UTC),
                "raw_text": "text",
            }
        )

        params = fake_pool.calls[0][1]
        assert params[3] == ""
        assert params[5:] == ["[]", "{}", "{}"]


# ---------------------------------------------------------------------------
# Entry reads
# ---------------------------------------------------------------------------


class TestEntryReads:
    """get_entry / get_entries_by_ids / search_by_time_range / count_entries."""

    async def test_get_entry_returns_converted_row(self, fake_pool_factory) -> None:
        """A found row is converted to an EnhancedLogbookEntry."""
        pool = fake_pool_factory(results=[[_entry_row(entry_id="e-42")]])
        repo = ARIELRepository(pool, _make_config())

        entry = await repo.get_entry("e-42")

        assert entry is not None
        assert entry["entry_id"] == "e-42"
        assert pool.calls[0] == (
            "SELECT * FROM enhanced_entries WHERE entry_id = %s",
            ["e-42"],
        )

    async def test_get_entry_returns_none_when_absent(self, fake_pool) -> None:
        """No row means None, not an exception."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.get_entry("missing") is None

    async def test_get_entries_by_ids_short_circuits_on_empty_input(self, fake_pool) -> None:
        """An empty id list returns [] without opening a connection.

        Hazard: ``= ANY(%s)`` with an empty array is a full-table scan waiting
        to happen on the caller's next refactor; the guard keeps the degenerate
        case off the database entirely.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.get_entries_by_ids([]) == []
        assert fake_pool.calls == []

    async def test_get_entries_by_ids_returns_all_found_rows(self, fake_pool_factory) -> None:
        """Ids are bound as one array parameter."""
        rows = [_entry_row(entry_id="e-1"), _entry_row(entry_id="e-2")]
        pool = fake_pool_factory(results=[rows])
        repo = ARIELRepository(pool, _make_config())

        entries = await repo.get_entries_by_ids(["e-1", "e-2", "gone"])

        assert [e["entry_id"] for e in entries] == ["e-1", "e-2"]
        assert pool.calls[0][1] == [["e-1", "e-2", "gone"]]

    async def test_search_by_time_range_without_filters_uses_true(self, fake_pool) -> None:
        """No filters means ``WHERE TRUE``, with only limit and offset bound."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.search_by_time_range(limit=25, offset=50) == []

        sql, params = fake_pool.calls[0]
        assert "WHERE TRUE" in _sql_body(sql)
        assert params == [25, 50]

    async def test_search_by_time_range_appends_each_filter(self, fake_pool_factory) -> None:
        """Filters are ANDed in declaration order, ahead of limit/offset."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        pool = fake_pool_factory(results=[[_entry_row()]])
        repo = ARIELRepository(pool, _make_config())

        entries = await repo.search_by_time_range(
            start=start,
            end=end,
            limit=10,
            offset=0,
            author="operator",
            source_system="als_logbook",
        )

        assert len(entries) == 1
        sql, params = pool.calls[0]
        assert (
            "WHERE timestamp >= %s AND timestamp <= %s AND author = %s AND source_system = %s"
            in _sql_body(sql)
        )
        assert params == [start, end, "operator", "als_logbook", 10, 0]

    async def test_count_entries_without_filters(self, fake_pool_factory) -> None:
        """An unfiltered count binds no parameters."""
        pool = fake_pool_factory(results=[[(17,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.count_entries() == 17
        assert pool.calls[0] == ("SELECT COUNT(*) FROM enhanced_entries WHERE TRUE", [])

    async def test_count_entries_mirrors_search_filters(self, fake_pool_factory) -> None:
        """The count applies the same filters as ``search_by_time_range``.

        Hazard: the two must stay in step -- a paginated listing derives
        ``total_pages`` from this count, so a filter honoured by one and not the
        other yields pages that render empty.
        """
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        pool = fake_pool_factory(results=[[(3,)]])
        repo = ARIELRepository(pool, _make_config())

        total = await repo.count_entries(
            start=start,
            end=end,
            author="operator",
            source_system="als_logbook",
        )

        assert total == 3
        sql, params = pool.calls[0]
        assert (
            "WHERE timestamp >= %s AND timestamp <= %s AND author = %s AND source_system = %s"
            in _sql_body(sql)
        )
        assert params == [start, end, "operator", "als_logbook"]

    async def test_count_entries_returns_zero_without_a_row(self, fake_pool) -> None:
        """A missing count row reads as zero rather than raising."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.count_entries() == 0

    async def test_get_distinct_authors_flattens_rows(self, fake_pool_factory) -> None:
        """Distinct authors come back as a flat list of strings."""
        pool = fake_pool_factory(results=[[("alice",), ("bob",)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_distinct_authors() == ["alice", "bob"]
        assert "SELECT DISTINCT author FROM enhanced_entries" in _sql_body(pool.sql[0])

    async def test_get_distinct_source_systems_flattens_rows(self, fake_pool_factory) -> None:
        """Distinct source systems come back as a flat list of strings."""
        pool = fake_pool_factory(results=[[("als_logbook",)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_distinct_source_systems() == ["als_logbook"]
        assert "SELECT DISTINCT source_system FROM enhanced_entries" in _sql_body(pool.sql[0])


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


class TestAttachments:
    """Attachment blobs live in their own table, keyed by attachment_id."""

    async def test_store_attachment_binds_columns_in_order(self, fake_pool) -> None:
        """The INSERT binds (attachment_id, entry_id, filename, mime, data, size)."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.store_attachment(
            entry_id="e-1",
            attachment_id="a-1",
            filename="shot.png",
            mime_type="image/png",
            data=b"\x89PNG",
            size_bytes=4,
        )

        sql, params = fake_pool.calls[0]
        assert "INSERT INTO attachment_files" in _sql_body(sql)
        assert params == ["a-1", "e-1", "shot.png", "image/png", b"\x89PNG", 4]

    async def test_get_attachment_returns_row_as_dict(self, fake_pool_factory) -> None:
        """A found attachment comes back as a plain dict including its bytes."""
        row = {"attachment_id": "a-1", "filename": "shot.png", "data": b"\x89PNG"}
        pool = fake_pool_factory(results=[[row]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_attachment("a-1") == row
        assert pool.calls[0][1] == ["a-1"]

    async def test_get_attachment_returns_none_when_absent(self, fake_pool) -> None:
        """A missing attachment is None, not an empty dict."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.get_attachment("gone") is None

    async def test_get_attachments_for_entry_omits_blob_column(self, fake_pool_factory) -> None:
        """The per-entry listing selects metadata only, never ``data``.

        Hazard: this feeds an entry view that may list many attachments, so
        adding ``data`` to the projection would pull every blob into memory to
        render a filename list.
        """
        pool = fake_pool_factory(results=[[{"attachment_id": "a-1", "filename": "shot.png"}]])
        repo = ARIELRepository(pool, _make_config())

        rows = await repo.get_attachments_for_entry("e-1")

        assert rows == [{"attachment_id": "a-1", "filename": "shot.png"}]
        select_clause = _sql_body(pool.sql[0]).split("FROM", 1)[0]
        assert "size_bytes" in select_clause
        assert "data" not in select_clause


# ---------------------------------------------------------------------------
# Enhancement status
# ---------------------------------------------------------------------------


class TestEnhancementStatus:
    """Incomplete-entry queries, stats, and the two status transitions."""

    async def test_get_incomplete_entries_filters_by_module_and_status(
        self,
        fake_pool_factory,
    ) -> None:
        """Both filters present narrows to one module's exact status."""
        pool = fake_pool_factory(results=[[_entry_row()]])
        repo = ARIELRepository(pool, _make_config())

        entries = await repo.get_incomplete_entries(
            module_name="text_embedding", status="failed", limit=5
        )

        assert len(entries) == 1
        sql, params = pool.calls[0]
        assert "WHERE enhancement_status->%s->>'status' = %s" in _sql_body(sql)
        assert params == ["text_embedding", "failed", 5]

    async def test_get_incomplete_entries_by_module_includes_never_attempted(
        self,
        fake_pool,
    ) -> None:
        """Module without status also matches entries the module never touched.

        Hazard: the ``NOT (enhancement_status ? %s)`` arm is what lets a newly
        enabled module pick up the existing corpus; matching only 'failed' and
        'pending' would leave every pre-existing entry permanently unenhanced.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(module_name="text_embedding")

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "NOT (enhancement_status ? %s)" in body
        assert "enhancement_status->%s->>'status' IN ('failed', 'pending')" in body
        assert params == ["text_embedding", "text_embedding", 100]

    async def test_get_incomplete_entries_without_module_scans_all(self, fake_pool) -> None:
        """No module filter selects every entry, oldest first."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(limit=7)

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "WHERE" not in body
        assert "ORDER BY created_at ASC LIMIT %s" in body
        assert params == [7]

    async def test_get_enhancement_stats_maps_row_positions(self, fake_pool_factory) -> None:
        """The seven aggregate columns are unpacked positionally.

        Hazard: the SELECT list order and this mapping are one contract with no
        column names between them -- reordering the FILTER clauses without
        reordering the indices would silently swap 'failed' and 'pending'.
        """
        pool = fake_pool_factory(results=[[(100, 90, 5, 5, 80, 10, 10)]])
        repo = ARIELRepository(pool, _make_config())

        stats = await repo.get_enhancement_stats()

        assert stats == {
            "total_entries": 100,
            "text_embedding": {"complete": 90, "failed": 5, "pending": 5},
            "semantic_processor": {"complete": 80, "failed": 10, "pending": 10},
        }

    async def test_get_enhancement_stats_on_empty_database(self, fake_pool) -> None:
        """No aggregate row degrades to a zero total rather than raising."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.get_enhancement_stats() == {"total_entries": 0}

    async def test_mark_enhancement_complete_targets_module_key(self, fake_pool) -> None:
        """Completion is written under the module's own jsonb key."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_complete("e-1", "text_embedding")

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "UPDATE enhanced_entries SET enhancement_status = jsonb_set(" in body
        assert "'status', 'complete'" in body
        assert params == [["text_embedding"], "e-1"]

    async def test_mark_enhancement_failed_truncates_the_error(self, fake_pool) -> None:
        """A long error message is truncated to 500 characters before binding.

        Hazard: enhancement errors can carry a whole provider response body;
        without the cap every failure would bloat the entry's jsonb status and,
        via the same row, every subsequent read of that entry.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_failed("e-1", "text_embedding", "x" * 600)

        params = fake_pool.calls[0][1]
        assert params[1] == "x" * 500
        assert params == [["text_embedding"], "x" * 500, "e-1"]

    async def test_mark_enhancement_failed_keeps_short_errors_intact(self, fake_pool) -> None:
        """An error under the cap is bound verbatim."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_failed("e-1", "text_embedding", "model timed out")

        assert fake_pool.calls[0][1][1] == "model timed out"


# ---------------------------------------------------------------------------
# Embedding tables
# ---------------------------------------------------------------------------


class TestEmbeddingTables:
    """Discovery of ``text_embeddings_*`` tables and their per-table probes."""

    async def test_each_table_is_probed_for_count_and_dimension(
        self,
        fake_pool_factory,
    ) -> None:
        """Per table: a COUNT and an ``atttypmod`` lookup, flagged against config.

        The result script is positional and mirrors the emitted order --
        discovery, then (count, dimension) for each table in turn.
        """
        pool = fake_pool_factory(
            results=[
                [("text_embeddings_nomic_embed_text",), ("text_embeddings_mxbai_embed_large",)],
                [(7,)],
                [(768,)],
                [(0,)],
                [(1024,)],
            ]
        )
        repo = ARIELRepository(pool, _make_config())

        tables = await repo.get_embedding_tables()

        assert [(t.table_name, t.entry_count, t.dimension, t.is_active) for t in tables] == [
            ("text_embeddings_nomic_embed_text", 7, 768, True),
            ("text_embeddings_mxbai_embed_large", 0, 1024, False),
        ]
        assert "information_schema.tables" in _sql_body(pool.sql[0])
        assert pool.sql[1] == "SELECT COUNT(*) FROM text_embeddings_nomic_embed_text"

    async def test_unmeasurable_table_reports_zero_count_and_no_dimension(
        self,
        fake_pool_factory,
    ) -> None:
        """A table whose probes return nothing usable degrades, not raises.

        ``atttypmod`` is -1 for a column declared without a type modifier, and
        the COUNT can come back empty; both mean "unknown", which the caller
        renders rather than crashing the diagnostics page on.
        """
        pool = fake_pool_factory(
            results=[
                [("text_embeddings_unknown",)],
                [],
                [(-1,)],
            ]
        )
        repo = ARIELRepository(pool, _make_config())

        (table,) = await repo.get_embedding_tables()

        assert table.entry_count == 0
        assert table.dimension is None
        assert table.is_active is False

    async def test_no_table_is_active_when_semantic_search_is_off(
        self,
        fake_pool_factory,
    ) -> None:
        """Without a configured search model nothing can be the active table."""
        config = _make_config(search_modules={"semantic": {"enabled": False}})
        pool = fake_pool_factory(
            results=[
                [("text_embeddings_nomic_embed_text",)],
                [(7,)],
                [(768,)],
            ]
        )
        repo = ARIELRepository(pool, config)

        (table,) = await repo.get_embedding_tables()

        assert table.is_active is False

    async def test_store_text_embedding_formats_vector_literal(self, fake_pool) -> None:
        """The vector is bound as a pgvector literal and upserted by entry_id."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.store_text_embedding("e-1", [0.1, 0.2, 0.3], "nomic-embed-text")

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "INSERT INTO text_embeddings_nomic_embed_text (entry_id, embedding)" in body
        assert "ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding" in body
        assert params == ["e-1", "[0.1,0.2,0.3]"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchQueries:
    """Keyword, fuzzy and semantic search all emit one parametrized statement."""

    async def test_keyword_search_with_highlights(self, fake_pool_factory) -> None:
        """Highlighted search binds the search text twice, ahead of the filters."""
        rows = [
            _entry_row(entry_id="e-1", rank=0.8, headline="beam <b>lost</b>"),
            _entry_row(entry_id="e-2", rank=None, headline=None),
        ]
        pool = fake_pool_factory(results=[rows])
        repo = ARIELRepository(pool, _make_config())

        results = await repo.keyword_search(
            where_clauses=["author = %s"],
            params=["operator"],
            search_text="beam lost",
            max_results=5,
        )

        assert [(entry["entry_id"], score, hl) for entry, score, hl in results] == [
            ("e-1", 0.8, ["beam <b>lost</b>"]),
            ("e-2", 0.0, []),
        ]
        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "ts_headline('english', raw_text || ' ' || COALESCE(summary, '')" in body
        assert "WHERE author = %s" in body
        assert params == ["beam lost", "beam lost", "operator", 5]

    async def test_keyword_search_without_highlights(self, fake_pool) -> None:
        """Skipping highlights drops the ts_headline call and one bound copy."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert (
            await repo.keyword_search(
                where_clauses=[],
                params=[],
                search_text="quench",
                include_highlights=False,
            )
            == []
        )

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "ts_headline" not in body
        assert "NULL AS headline" in body
        assert "WHERE TRUE" in body
        assert params == ["quench", 10]

    async def test_keyword_search_without_semantic_processor_uses_core_fts(self, fake_pool) -> None:
        """Default-off semantic processing leaves keyword search on the core raw_text index."""
        config = _make_config(enhancement_modules={"semantic_processor": {"enabled": False}})
        repo = ARIELRepository(fake_pool, config)

        assert (
            await repo.keyword_search(
                where_clauses=[],
                params=[],
                search_text="quench",
                include_highlights=False,
            )
            == []
        )

        body = _sql_body(fake_pool.calls[0][0])
        assert "to_tsvector('english', raw_text)" in body
        assert "COALESCE(summary, '')" not in body

    async def test_fuzzy_search_without_date_filters(self, fake_pool_factory) -> None:
        """Similarity threshold is the only filter when no dates are given."""
        pool = fake_pool_factory(results=[[_entry_row(sim=0.42)]])
        repo = ARIELRepository(pool, _make_config())

        ((entry, score, highlights),) = await repo.fuzzy_search("quench", threshold=0.25)

        assert (entry["entry_id"], score, highlights) == ("e-1", 0.42, [])
        sql, params = pool.calls[0]
        assert "WHERE similarity(raw_text, %s) >= %s" in _sql_body(sql)
        assert params == ["quench", "quench", 0.25, 10]

    async def test_fuzzy_search_appends_date_filters(self, fake_pool) -> None:
        """Start and end dates are ANDed after the similarity predicate."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.fuzzy_search("quench", start_date=start, end_date=end, max_results=3)

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "timestamp >= %s AND timestamp <= %s" in body
        assert params == ["quench", "quench", 0.3, start, end, 3]

    async def test_fuzzy_search_reports_zero_for_null_similarity(
        self,
        fake_pool_factory,
    ) -> None:
        """A NULL similarity scores 0.0 rather than crashing the conversion."""
        pool = fake_pool_factory(results=[[_entry_row(sim=None)]])
        repo = ARIELRepository(pool, _make_config())

        ((_entry, score, _highlights),) = await repo.fuzzy_search("quench")

        assert score == 0.0

    async def test_semantic_search_without_filters(self, fake_pool_factory) -> None:
        """The embedding is bound twice: once for the projection, once for WHERE."""
        pool = fake_pool_factory(results=[[_entry_row(similarity=0.91)]])
        repo = ARIELRepository(pool, _make_config())

        ((entry, similarity),) = await repo.semantic_search(
            [0.1, 0.2], "nomic-embed-text", max_results=4, similarity_threshold=0.6
        )

        assert (entry["entry_id"], similarity) == ("e-1", 0.91)
        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "JOIN text_embeddings_nomic_embed_text emb ON e.entry_id = emb.entry_id" in body
        assert "WHERE 1 - (emb.embedding <=> %s::vector) >= %s" in body
        assert params == ["[0.1,0.2]", "[0.1,0.2]", 0.6, 4]

    async def test_semantic_search_appends_every_filter(self, fake_pool) -> None:
        """Date, author and source filters are ANDed in declaration order.

        Author matches with ILIKE and wildcards while source_system is exact --
        the asymmetry is deliberate and worth pinning, since a stray wildcard on
        source_system would silently widen every filtered search.
        """
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.semantic_search(
            [0.5],
            "nomic-embed-text",
            start_date=start,
            end_date=end,
            author="oper",
            source_system="als_logbook",
        )

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "e.timestamp >= %s AND e.timestamp <= %s" in body
        assert "e.author ILIKE %s AND e.source_system = %s" in body
        assert params == ["[0.5]", "[0.5]", 0.5, start, end, "%oper%", "als_logbook", 10]

    async def test_semantic_search_reports_zero_for_null_similarity(
        self,
        fake_pool_factory,
    ) -> None:
        """A NULL similarity scores 0.0 rather than crashing the conversion."""
        pool = fake_pool_factory(results=[[_entry_row(similarity=None)]])
        repo = ARIELRepository(pool, _make_config())

        ((_entry, similarity),) = await repo.semantic_search([0.1], "nomic-embed-text")

        assert similarity == 0.0


# ---------------------------------------------------------------------------
# Ingestion runs
# ---------------------------------------------------------------------------


class TestIngestionRuns:
    """The ingestion_runs bookkeeping row, from start to terminal state."""

    async def test_start_ingestion_run_returns_the_new_id(self, fake_pool_factory) -> None:
        """The RETURNING id is what callers pass to complete/fail."""
        pool = fake_pool_factory(results=[[(42,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.start_ingestion_run("als_logbook") == 42

        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "INSERT INTO ingestion_runs (started_at, source_system, status)" in body
        assert "'running'" in body
        assert "RETURNING id" in body
        assert params == ["als_logbook"]

    async def test_complete_ingestion_run_binds_counts_then_id(self, fake_pool) -> None:
        """Success closes the row with the three entry counts."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.complete_ingestion_run(
            run_id=7, entries_added=3, entries_updated=2, entries_failed=1
        )

        sql, params = fake_pool.calls[0]
        assert "status = 'success'" in _sql_body(sql)
        assert params == [3, 2, 1, 7]

    async def test_fail_ingestion_run_truncates_the_error(self, fake_pool) -> None:
        """The failure message is truncated to 500 characters before binding.

        Hazard: the error recorded here is whatever the adapter raised, which
        for an HTTP adapter can be an entire response body; without the cap one
        bad poll writes an unbounded string into the run history.
        """
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.fail_ingestion_run(7, "y" * 600)

        sql, params = fake_pool.calls[0]
        assert "status = 'failed'" in _sql_body(sql)
        assert params == ["y" * 500, 7]

    async def test_get_last_successful_run_returns_completion_time(
        self,
        fake_pool_factory,
    ) -> None:
        """The watermark is the MAX(completed_at) over successful runs only."""
        completed = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
        pool = fake_pool_factory(results=[[(completed,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_successful_run("als_logbook") == completed

        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "SELECT MAX(completed_at) FROM ingestion_runs" in body
        assert "status = 'success'" in body
        assert params == ["als_logbook"]

    async def test_get_last_successful_run_without_any_run(self, fake_pool) -> None:
        """No rows at all means no watermark."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.get_last_successful_run("als_logbook") is None

    async def test_get_last_successful_run_with_null_aggregate(
        self,
        fake_pool_factory,
    ) -> None:
        """MAX over zero successful runs is a NULL, which is also no watermark.

        Hazard: the aggregate always returns one row, so a truthiness check on
        the row alone would hand callers a None timestamp and make the next
        incremental poll compare against nothing.
        """
        pool = fake_pool_factory(results=[[(None,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_successful_run("als_logbook") is None
