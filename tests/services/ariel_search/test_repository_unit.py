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
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from osprey.imaging.formats import CAPTION_NOT_DONE_SQL, image_table_not_done_sql, viewable_sql
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database import repository as repository_module
from osprey.services.ariel_search.database.repository import (
    ATTACHMENT_ROW_COLUMNS,
    ATTACHMENT_SCHEMA_GAP_WARNING,
    MAX_ENHANCEMENT_ATTEMPTS,
    SCHEMA_FACTS_NEGATIVE_TTL_SECONDS,
    ARIELRepository,
    SchemaFacts,
)
from osprey.services.ariel_search.exceptions import (
    ConfigurationError,
    DatabaseQueryError,
    PatternError,
    SearchTimeoutError,
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
        "SELECT MAX(started_at) source_system=als_logbook",
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

        Hazard: the adapter write contract carries no attachments, so an entry
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

    def test_no_select_star_targets_attachment_files_under_src(self) -> None:
        """Every attachment_files reader names its columns, so no query drags both blobs."""
        src = Path(__file__).resolve().parents[3] / "src"
        pattern = re.compile(r"SELECT\s+\*\s+FROM\s+attachment_files", re.IGNORECASE)
        assert src.is_dir(), src
        offenders = [
            str(path.relative_to(src))
            for path in sorted(src.rglob("*"))
            if path.suffix in {".py", ".sql"} and pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_repository_has_no_whole_row_attachment_reader(self) -> None:
        """Originals and renditions are read through their own column-listed readers."""
        assert not hasattr(ARIELRepository, "get_attachment")
        assert hasattr(ARIELRepository, "get_attachment_original")
        assert hasattr(ARIELRepository, "get_rendition")


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
        assert "enhancement_status->%s->>'status' = 'pending'" in body
        assert "COALESCE((enhancement_status->%s->>'attempts')::int, 0) < %s" in body
        assert params == ["text_embedding"] * 4 + [MAX_ENHANCEMENT_ATTEMPTS, 100]

    async def test_get_incomplete_entries_without_module_scans_all(self, fake_pool) -> None:
        """No module filter selects every entry, oldest first."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(limit=7)

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "WHERE" not in body
        assert "ORDER BY created_at ASC LIMIT %s" in body
        assert params == [7]

    async def test_get_enhancement_stats_groups_by_module_key(self, fake_pool_factory) -> None:
        """Whatever modules the store carries are reported, none of them named in SQL.

        The aggregate returns one row per (module, status) pair, so a module a
        facility registered itself is counted the moment its first entry lands.
        """
        pool = fake_pool_factory(
            rows_for={
                "jsonb_each(enhancement_status)": [
                    (100, "text_embedding", "complete", 90),
                    (100, "text_embedding", "failed", 5),
                    (100, "text_embedding", "pending", 5),
                    (100, "facility_tagger", "complete", 80),
                    (100, "facility_tagger", "failed", 10),
                ],
            }
        )
        repo = ARIELRepository(pool, _make_config())

        stats = await repo.get_enhancement_stats()

        assert stats == {
            "total_entries": 100,
            "text_embedding": {"complete": 90, "failed": 5, "pending": 5},
            # 10 entries never reached this module at all: no key, so no row.
            "facility_tagger": {"complete": 80, "failed": 10, "pending": 10},
        }

    async def test_get_enhancement_stats_reads_one_snapshot(self, fake_pool_factory) -> None:
        """Total and per-module counts come from a single statement.

        ``pending`` is derived by subtracting the module's seen rows from the
        total, so the two counts must describe the same snapshot. Split across
        two statements on an autocommit connection, an ingest landing in between
        would make the subtraction negative.
        """
        pool = fake_pool_factory(
            rows_for={"jsonb_each(enhancement_status)": [(4, "text_embedding", "complete", 4)]}
        )
        repo = ARIELRepository(pool, _make_config())

        await repo.get_enhancement_stats()

        assert len(pool.calls) == 1
        body = _sql_body(pool.calls[0][0])
        assert "COUNT(*) AS entries FROM enhanced_entries" in body
        assert "jsonb_each(enhancement_status)" in body

    async def test_get_enhancement_stats_names_no_module_in_sql(self, fake_pool_factory) -> None:
        """Fixed SQL text: a module name never reaches the statement."""
        pool = fake_pool_factory(
            rows_for={"jsonb_each(enhancement_status)": [(1, None, None, None)]}
        )
        repo = ARIELRepository(pool, _make_config())

        await repo.get_enhancement_stats()

        aggregate = [sql for sql, _ in pool.calls if "jsonb_each" in sql]
        assert len(aggregate) == 1
        assert "text_embedding" not in aggregate[0]
        assert "semantic_processor" not in aggregate[0]

    async def test_get_enhancement_stats_reports_a_total_with_no_status_keys(
        self, fake_pool_factory
    ) -> None:
        """A store whose entries carry no status key still reports its total."""
        pool = fake_pool_factory(
            rows_for={"jsonb_each(enhancement_status)": [(7, None, None, None)]}
        )
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_enhancement_stats() == {"total_entries": 7}

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
        assert params == [
            ["text_embedding"],
            "x" * 500,
            "text_embedding",
            "e-1",
            "text_embedding",
        ]

    async def test_mark_enhancement_failed_keeps_short_errors_intact(self, fake_pool) -> None:
        """An error under the cap is bound verbatim."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_failed("e-1", "text_embedding", "model timed out")

        assert fake_pool.calls[0][1][1] == "model timed out"

    async def test_mark_enhancement_failed_counts_the_attempt(self, fake_pool) -> None:
        """The failure adds one to the module's attempt count and reads it back."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_failed("e-1", "text_embedding", "nope")

        body = _sql_body(fake_pool.calls[0][0])
        assert "'attempts', COALESCE((enhancement_status->%s->>'attempts')::int, 0) + 1" in body
        assert "RETURNING" in body

    async def test_mark_enhancement_failed_returns_the_stored_count(
        self, fake_pool_factory
    ) -> None:
        """The count the update stored is returned."""
        pool = fake_pool_factory(rows_for={"RETURNING": [(2,)]})
        repo = ARIELRepository(pool, _make_config())

        assert await repo.mark_enhancement_failed("e-1", "text_embedding", "nope") == 2

    async def test_mark_enhancement_failed_on_an_unknown_entry_returns_zero(
        self, fake_pool
    ) -> None:
        """No row updated means no attempt stored."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.mark_enhancement_failed("missing", "text_embedding", "nope") == 0


class TestMarkerStatus:
    """The marker-aware branches: named placeholders, B1 text untouched."""

    async def test_incomplete_with_marker_uses_the_marker_predicate(self, fake_pool) -> None:
        """A stale marker or a not-given-up pending/failed status makes an entry incomplete."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(module_name="text_embedding", marker="m-a", limit=9)

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "NOT (e.enhancement_status ? %(module)s)" in body
        assert "e.enhancement_status->%(module)s->>'marker' IS DISTINCT FROM %(marker)s" in body
        assert "IN ('pending', 'failed')" in body
        assert "(e.enhancement_status->%(module)s->>'gave_up')::boolean" in body
        assert "%s" not in body
        assert "EXISTS" not in body
        assert params == {"module": "text_embedding", "marker": "m-a", "limit": 9}

    async def test_incomplete_with_marker_orders_pending_newest_first_then_failed(
        self, fake_pool
    ) -> None:
        """One catch-up order: effective status, effective attempts, newest, entry id."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(module_name="text_embedding", marker="m-a")

        body = _sql_body(fake_pool.calls[0][0])
        order = body.split("ORDER BY", 1)[1]
        assert order.index("= 'failed'") < order.index("'attempts')::int, 0)")
        assert order.index("'attempts')::int, 0)") < order.index("e.timestamp DESC")
        assert order.index("e.timestamp DESC") < order.index("e.entry_id")
        assert "THEN 'pending'" in order
        assert "COALESCE(e.enhancement_status->%(module)s->>'status', 'pending')" in order

    async def test_caption_todo_requires_a_viewable_picture_not_done(self, fake_pool) -> None:
        """image_caption walks only entries holding a viewable picture with no caption."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(module_name="image_caption", marker="vis-a")

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "EXISTS ( SELECT 1 FROM attachment_files f WHERE f.entry_id = e.entry_id" in body
        assert viewable_sql("f") in body
        assert CAPTION_NOT_DONE_SQL in body
        assert params["model"] == "vis-a"

    async def test_embedding_todo_reads_the_marker_table(self, fake_pool) -> None:
        """image_embedding's not-done fragment names the image table its marker holds."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(module_name="image_embedding", marker="img_emb_x")

        body = _sql_body(fake_pool.calls[0][0])
        assert image_table_not_done_sql("img_emb_x") in body

    async def test_embedding_todo_refuses_a_non_identifier_marker(self, fake_pool) -> None:
        """A table marker is spliced, so it must be a plain identifier."""
        repo = ARIELRepository(fake_pool, _make_config())

        with pytest.raises(ValueError):
            await repo.get_incomplete_entries(module_name="image_embedding", marker="x; DROP")
        assert fake_pool.calls == []

    async def test_status_filter_keeps_b1_statement_even_with_marker(self, fake_pool) -> None:
        """``status=`` (retry-failed listing) is B1's exact statement."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_incomplete_entries(module_name="text_embedding", status="failed")

        assert fake_pool.calls[0][1] == ["text_embedding", "failed", 100]

    async def test_mark_complete_with_marker_stores_it(self, fake_pool) -> None:
        """The complete object is ``{status, completed_at, marker}``."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_complete("e-1", "image_caption", marker="vis-a")

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "%(path)s::text[]" in body
        assert "'marker', %(marker)s::text" in body
        assert "WHERE entry_id = %(entry_id)s" in body
        assert params == {"path": ["image_caption"], "entry_id": "e-1", "marker": "vis-a"}

    async def test_mark_failed_with_marker_resets_on_a_stale_marker(self, fake_pool) -> None:
        """A different stored marker restarts attempts at 1; gave_up at the cap."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.mark_enhancement_failed("e-1", "image_caption", "y" * 600, marker="vis-a")

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert (
            "CASE WHEN enhancement_status->%(module)s->>'marker' IS DISTINCT FROM %(marker)s"
            " THEN 1 ELSE COALESCE((enhancement_status->%(module)s->>'attempts')::int, 0) + 1 END"
        ) in body
        assert ">= %(max_attempts)s" in body
        assert "'marker', %(marker)s::text" in body
        assert "%s" not in body
        assert params == {
            "path": ["image_caption"],
            "module": "image_caption",
            "error": "y" * 500,
            "marker": "vis-a",
            "max_attempts": MAX_ENHANCEMENT_ATTEMPTS,
            "entry_id": "e-1",
        }

    async def test_stats_with_markers_binds_modules_and_markers(self, fake_pool) -> None:
        """Marker modules and markers are named parameters, never spliced."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_enhancement_stats(markers={"image_caption": "vis-a"})

        sql, params = fake_pool.calls[0]
        assert "image_caption" not in sql
        assert "vis-a" not in sql
        body = _sql_body(sql)
        assert "IS DISTINCT FROM %(marker_0)s THEN 'pending'" in body
        assert params == {"module_0": "image_caption", "marker_0": "vis-a"}

    async def test_stats_with_markers_adds_gave_up_to_marker_modules_only(
        self, fake_pool_factory
    ) -> None:
        """A module absent from ``markers`` keeps B1's three keys exactly."""
        pool = fake_pool_factory(
            rows_for={
                "jsonb_each(enhancement_status)": [
                    (10, "image_caption", "complete", 4, 0),
                    (10, "image_caption", "failed", 3, 2),
                    (10, "image_caption", "pending", 1, 0),
                    (10, "text_embedding", "complete", 9, 0),
                ],
            }
        )
        repo = ARIELRepository(pool, _make_config())

        stats = await repo.get_enhancement_stats(markers={"image_caption": "vis-a"})

        assert stats == {
            "total_entries": 10,
            "image_caption": {"complete": 4, "failed": 3, "pending": 3, "gave_up": 2},
            "text_embedding": {"complete": 9, "failed": 0, "pending": 1},
        }

    async def test_stats_with_empty_markers_is_b1(self, fake_pool) -> None:
        """An empty mapping takes B1's statement, which binds nothing."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.get_enhancement_stats(markers={})

        sql, params = fake_pool.calls[0]
        assert "marker" not in sql
        assert not params


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

    async def test_get_last_successful_run_returns_start_time(
        self,
        fake_pool_factory,
    ) -> None:
        """The watermark is the MAX(started_at) over successful runs only.

        An entry written upstream while a run fetched is newer than the run's
        start, so the start, not the completion, bounds the next poll.
        """
        started = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
        pool = fake_pool_factory(results=[[(started,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_successful_run("als_logbook") == started

        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "SELECT MAX(started_at) FROM ingestion_runs" in body
        assert "completed_at" not in body
        assert "status = 'success'" in body
        assert params == ["als_logbook"]

    async def test_get_last_ingestion_keeps_completion_time(self, fake_pool_factory) -> None:
        """The status time stays the last completion, unlike the poll watermark."""
        completed = datetime(2026, 3, 4, 5, 7, tzinfo=UTC)
        pool = fake_pool_factory(results=[[(completed,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_ingestion() == completed

        sql, _params = pool.calls[0]
        assert "MAX(completed_at)" in _sql_body(sql)

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


# ---------------------------------------------------------------------------
# Keyword search: expanded tsquery and pattern timeout envelope
# ---------------------------------------------------------------------------

#: Transaction-boundary markers the fake below writes into its log.
_TX_OPEN = "BEGIN"
_TX_OK = "COMMIT"
_TX_UNDO = "ROLLBACK"


class _TxPool:
    """Fake pool that also records transaction boundaries.

    The shared ``_FakePool`` in ``conftest`` has no ``transaction()``, and the
    pattern timeout envelope is exactly a transaction boundary plus a
    ``set_config`` -- so the ordered ``log`` here interleaves the block markers
    with the SQL text, which is the only way to pin that ``SET LOCAL`` really
    ran *inside* the block.

    Args:
        rows: Rows every ``execute`` returns.
        error: Raised by the ``enhanced_entries`` statement only, so the
            bookkeeping statements around it still run.
    """

    def __init__(self, rows: list[Any] | None = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.log: list[str] = []
        self.rows = list(rows or [])
        self.error = error
        self.conn = _TxConnection(self)

    def connection(self) -> _TxConnection:
        return self.conn

    def record(self, sql: str, params: Any) -> list[Any]:
        """Log one execute and return (or raise) its scripted result."""
        self.calls.append((sql, params))
        self.log.append(sql)
        if self.error is not None and "enhanced_entries" in sql:
            raise self.error
        return list(self.rows)


class _TxTransaction:
    """``conn.transaction()`` stand-in that marks its own boundaries."""

    def __init__(self, pool: _TxPool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _TxTransaction:
        self.pool.log.append(_TX_OPEN)
        return self

    async def __aexit__(self, exc_type: Any, *rest: object) -> bool:
        self.pool.log.append(_TX_OK if exc_type is None else _TX_UNDO)
        return False


class _TxCursor:
    """Cursor stand-in writing through to the pool's log."""

    def __init__(self, pool: _TxPool, row_factory: Any = None) -> None:
        self.pool = pool
        self.row_factory = row_factory
        self.rows: list[Any] = []

    async def __aenter__(self) -> _TxCursor:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> _TxCursor:
        self.rows = self.pool.record(sql, params)
        return self

    async def fetchall(self) -> list[Any]:
        return list(self.rows)


class _TxConnection:
    """Connection stand-in supporting ``transaction()``, ``cursor()`` and ``execute()``."""

    def __init__(self, pool: _TxPool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _TxConnection:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def transaction(self) -> _TxTransaction:
        return _TxTransaction(self.pool)

    def cursor(self, row_factory: Any = None) -> _TxCursor:
        return _TxCursor(self.pool, row_factory=row_factory)

    async def execute(self, sql: str, params: Any = None) -> _TxCursor:
        cur = self.cursor()
        await cur.execute(sql, params)
        return cur


#: A five-placeholder expanded tsquery, the shape ``build_expanded_tsquery`` emits.
EXPANDED_TSQUERY = (
    "(plainto_tsquery('english', %s) || plainto_tsquery('english', %s)) && "
    "plainto_tsquery('english', %s) && "
    "(phraseto_tsquery('english', %s) || plainto_tsquery('english', %s))"
)
EXPANDED_PARAMS = ["ts", "troubleshoot", "aborted", "beam dump", "beam abort"]

SET_TIMEOUT_SQL = "SELECT set_config('statement_timeout', %s, true)"


def _entry_statement(pool: _TxPool) -> tuple[str, Any]:
    """The one statement that reads ``enhanced_entries``."""
    (call,) = [call for call in pool.calls if "enhanced_entries" in call[0]]
    return call


class TestKeywordSearchTsquerySplice:
    """A caller-supplied tsquery replaces the plain one in rank *and* headline."""

    @pytest.mark.parametrize("include_highlights", [True, False])
    async def test_default_call_keeps_the_plain_tsquery_path(
        self,
        fake_pool,
        include_highlights: bool,
    ) -> None:
        """Omitting `tsquery_sql` emits today's statement, placeholders and params."""
        repo = ARIELRepository(fake_pool, _make_config())

        await repo.keyword_search(
            where_clauses=["author = %s"],
            params=["operator"],
            search_text="beam lost",
            max_results=5,
            include_highlights=include_highlights,
        )

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        assert "plainto_tsquery('english', %s)" in body
        expected = (
            ["beam lost", "beam lost", "operator", 5]
            if include_highlights
            else ["beam lost", "operator", 5]
        )
        assert params == expected
        assert sql.count("%s") == len(params)

    async def test_expanded_tsquery_is_spliced_twice_with_highlights(self) -> None:
        """Rank and headline each take the fragment, so its params bind twice, first."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        await repo.keyword_search(
            where_clauses=["author = %s"],
            params=["operator"],
            search_text="ts aborted",
            max_results=5,
            tsquery_sql=EXPANDED_TSQUERY,
            tsquery_params=EXPANDED_PARAMS,
        )

        sql, params = _entry_statement(pool)
        body = _sql_body(sql)
        assert body.count(_sql_body(EXPANDED_TSQUERY)) == 2
        assert "plainto_tsquery('english', %s) ) AS rank" not in body
        assert params == [*EXPANDED_PARAMS, *EXPANDED_PARAMS, "operator", 5]
        assert sql.count("%s") == len(params)

    async def test_expanded_tsquery_is_spliced_once_without_highlights(self) -> None:
        """No ts_headline means one occurrence of the fragment and one copy of its params."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        await repo.keyword_search(
            where_clauses=["author = %s"],
            params=["operator"],
            search_text="ts aborted",
            max_results=5,
            include_highlights=False,
            tsquery_sql=EXPANDED_TSQUERY,
            tsquery_params=EXPANDED_PARAMS,
        )

        sql, params = _entry_statement(pool)
        body = _sql_body(sql)
        assert body.count(_sql_body(EXPANDED_TSQUERY)) == 1
        assert "ts_headline" not in body
        assert params == [*EXPANDED_PARAMS, "operator", 5]
        assert sql.count("%s") == len(params)

    async def test_rows_are_still_decoded_through_the_expanded_path(self) -> None:
        """Splicing changes the statement, never how rank and headline come back."""
        pool = _TxPool(rows=[_entry_row(entry_id="e-1", rank=0.5, headline="ts <b>aborted</b>")])
        repo = ARIELRepository(pool, _make_config())

        ((entry, score, highlights),) = await repo.keyword_search(
            where_clauses=[],
            params=[],
            search_text="ts aborted",
            tsquery_sql=EXPANDED_TSQUERY,
            tsquery_params=EXPANDED_PARAMS,
        )

        assert (entry["entry_id"], score, highlights) == ("e-1", 0.5, ["ts <b>aborted</b>"])


class TestKeywordSearchOrdering:
    """A pattern-only statement ranks everything 0, so the tie has to be broken."""

    async def test_pattern_only_query_breaks_the_rank_tie_on_timestamp(self) -> None:
        """No search text and no tsquery: order by rank then timestamp."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        await repo.keyword_search(
            where_clauses=["raw_text ~* %s"],
            params=["SR01C___BPM[0-9]+"],
            search_text="",
        )

        body = _sql_body(_entry_statement(pool)[0])
        assert "ORDER BY rank DESC, timestamp DESC" in body

    @pytest.mark.parametrize(
        ("search_text", "tsquery_sql", "tsquery_params"),
        [
            pytest.param("quench", None, None, id="plain-text"),
            pytest.param("", EXPANDED_TSQUERY, EXPANDED_PARAMS, id="expanded-tsquery"),
        ],
    )
    async def test_ranked_query_orders_on_rank_alone(
        self,
        search_text: str,
        tsquery_sql: str | None,
        tsquery_params: list[Any] | None,
    ) -> None:
        """Anything that actually ranks keeps today's single-key ordering."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        await repo.keyword_search(
            where_clauses=[],
            params=[],
            search_text=search_text,
            tsquery_sql=tsquery_sql,
            tsquery_params=tsquery_params,
        )

        body = _sql_body(_entry_statement(pool)[0])
        assert "ORDER BY rank DESC LIMIT" in body


class TestKeywordSearchTimeoutEnvelope:
    """`pattern_timeout_seconds` is the only thing that opens a transaction."""

    async def test_no_timeout_opens_no_transaction(self) -> None:
        """The default path issues one statement and no set_config."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        await repo.keyword_search(where_clauses=[], params=[], search_text="quench")

        assert _TX_OPEN not in pool.log
        assert not [sql for sql in pool.log if "set_config" in sql]
        assert len(pool.calls) == 1

    @pytest.mark.parametrize(
        ("seconds", "rendered"),
        [
            pytest.param(10.0, "10000ms", id="default"),
            pytest.param(0.001, "1ms", id="floor"),
            pytest.param(2.5, "2500ms", id="fractional"),
        ],
    )
    async def test_timeout_runs_set_config_inside_the_transaction(
        self,
        seconds: float,
        rendered: str,
    ) -> None:
        """SET LOCAL is inert outside a block, so it must follow the block marker."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        await repo.keyword_search(
            where_clauses=["raw_text ~* %s"],
            params=["SR01C___BPM[0-9]+"],
            search_text="",
            pattern_timeout_seconds=seconds,
        )

        assert pool.log[0] == _TX_OPEN
        assert pool.log[-1] == _TX_OK
        assert pool.log[1] == SET_TIMEOUT_SQL
        assert pool.calls[0] == (SET_TIMEOUT_SQL, (rendered,))
        assert "enhanced_entries" in pool.log[2]

    async def test_timeout_rounding_down_to_zero_is_refused(self) -> None:
        """0ms disables the timeout in PostgreSQL, so it is never silently sent."""
        pool = _TxPool()
        repo = ARIELRepository(pool, _make_config())

        with pytest.raises(ValueError, match="at least 0.001"):
            await repo.keyword_search(
                where_clauses=[],
                params=[],
                search_text="",
                pattern_timeout_seconds=0.0004,
            )

        assert pool.calls == []


class TestKeywordSearchErrorClassification:
    """Timeouts and bad patterns are named errors, not a generic query failure."""

    async def test_query_canceled_becomes_search_timeout_error(self) -> None:
        """A cancelled statement is the timeout the caller asked for."""
        canceled = psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
        pool = _TxPool(error=canceled)
        repo = ARIELRepository(pool, _make_config())

        with pytest.raises(SearchTimeoutError) as excinfo:
            await repo.keyword_search(
                where_clauses=["raw_text ~* %s"],
                params=["SR01C___BPM[0-9]+"],
                search_text="",
                pattern_timeout_seconds=10.0,
            )

        assert excinfo.value.timeout_seconds == 10.0
        assert excinfo.value.operation == "keyword_search"
        assert excinfo.value.__cause__ is canceled
        assert pool.log[-1] == _TX_UNDO

    async def test_invalid_regular_expression_becomes_pattern_error(self) -> None:
        """PostgreSQL's own message names the expression it refused."""
        invalid = psycopg.errors.InvalidRegularExpression(
            "invalid regular expression: brackets [] not balanced"
        )
        pool = _TxPool(error=invalid)
        repo = ARIELRepository(pool, _make_config())

        with pytest.raises(PatternError) as excinfo:
            await repo.keyword_search(
                where_clauses=["raw_text ~* %s"],
                params=["SR0[1-4"],
                search_text="",
                pattern_timeout_seconds=10.0,
            )

        assert "brackets [] not balanced" in str(excinfo.value)
        assert excinfo.value.pattern is None
        assert excinfo.value.__cause__ is invalid

    async def test_cancel_inside_the_regex_engine_is_still_a_timeout(self) -> None:
        """A statement_timeout that lands mid-regex surfaces as SQLSTATE 2201B, not 57014.

        PostgreSQL's regex engine reports the cancellation as
        ``invalid regular expression: operation cancelled``; the operator asked
        for a timeout and must not be told their pattern is malformed.
        """
        cancelled = psycopg.errors.InvalidRegularExpression(
            "invalid regular expression: operation cancelled"
        )
        pool = _TxPool(error=cancelled)
        repo = ARIELRepository(pool, _make_config())

        with pytest.raises(SearchTimeoutError) as excinfo:
            await repo.keyword_search(
                where_clauses=["raw_text ~* %s"],
                params=["SR01C___BPM[0-9]+ trip [a-z]{4,}"],
                search_text="",
                pattern_timeout_seconds=0.001,
            )

        assert excinfo.value.timeout_seconds == 0.001
        assert excinfo.value.operation == "keyword_search"
        assert excinfo.value.__cause__ is cancelled

    async def test_any_other_failure_still_becomes_database_query_error(self) -> None:
        """The blanket handler and its breadcrumb are unchanged."""
        pool = _TxPool(error=RuntimeError("connection reset"))
        repo = ARIELRepository(pool, _make_config())

        with pytest.raises(DatabaseQueryError) as excinfo:
            await repo.keyword_search(where_clauses=[], params=[], search_text="quench")

        assert excinfo.value.technical_details["query"] == "KEYWORD SEARCH: quench"


# ---------------------------------------------------------------------------
# schema_facts -- per-pool probe of optional schema objects
# ---------------------------------------------------------------------------

_PROBE = "information_schema.columns"


class _Clock:
    """Settable monotonic clock for the negative-probe TTL; nothing sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _probing_repo(pool: Any, config: ARIELConfig | None = None) -> tuple[ARIELRepository, _Clock]:
    repo = ARIELRepository(pool, config or _make_config())
    clock = _Clock()
    repo._clock = clock
    return repo, clock


class TestSchemaFacts:
    async def test_one_statement_probes_both_facts(self, fake_pool):
        fake_pool.recorder.rows_for[_PROBE] = [(False, True)]
        repo, _ = _probing_repo(fake_pool)

        facts = await repo.schema_facts()

        assert facts == SchemaFacts(has_v2_fts=False, has_copy_state=True)
        assert len(fake_pool.calls) == 1
        sql = fake_pool.sql[0]
        assert "to_regclass('idx_entries_raw_text_fts_v2')" in sql
        assert "attachment_files" in sql and "copy_status" in sql

    async def test_semantic_processor_adds_the_second_v2_index(self, fake_pool):
        repo, _ = _probing_repo(fake_pool)

        await repo.schema_facts()

        assert "to_regclass('idx_entries_text_search_v2')" in fake_pool.sql[0]

    async def test_without_semantic_processor_only_the_raw_text_index_counts(self, fake_pool):
        config = _make_config(
            enhancement_modules={"semantic_processor": {"enabled": False}},
        )
        repo, _ = _probing_repo(fake_pool, config)

        await repo.schema_facts()

        assert "idx_entries_text_search_v2" not in fake_pool.sql[0]
        assert "idx_entries_raw_text_fts_v2" in fake_pool.sql[0]

    async def test_negative_is_cached_within_the_ttl(self, fake_pool):
        fake_pool.recorder.rows_for[_PROBE] = [(False, False)]
        repo, clock = _probing_repo(fake_pool)

        await repo.schema_facts()
        fake_pool.recorder.rows_for[_PROBE] = [(False, True)]
        clock.now += SCHEMA_FACTS_NEGATIVE_TTL_SECONDS - 1
        facts = await repo.schema_facts()

        assert facts.has_copy_state is False
        assert len(fake_pool.matching(_PROBE)) == 1

    async def test_negative_is_reprobed_after_the_ttl(self, fake_pool):
        fake_pool.recorder.rows_for[_PROBE] = [(False, False)]
        repo, clock = _probing_repo(fake_pool)

        await repo.schema_facts()
        fake_pool.recorder.rows_for[_PROBE] = [(False, True)]
        clock.now += SCHEMA_FACTS_NEGATIVE_TTL_SECONDS
        facts = await repo.schema_facts()

        assert facts == SchemaFacts(has_v2_fts=False, has_copy_state=True)
        assert len(fake_pool.matching(_PROBE)) == 2

    async def test_positive_is_never_reprobed(self, fake_pool):
        fake_pool.recorder.rows_for[_PROBE] = [(True, True)]
        repo, clock = _probing_repo(fake_pool)

        await repo.schema_facts()
        clock.now += 100 * SCHEMA_FACTS_NEGATIVE_TTL_SECONDS
        facts = await repo.schema_facts()

        assert facts == SchemaFacts(has_v2_fts=True, has_copy_state=True)
        assert len(fake_pool.calls) == 1

    async def test_a_true_fact_stays_true_while_the_other_is_reprobed(self, fake_pool):
        fake_pool.recorder.rows_for[_PROBE] = [(False, True)]
        repo, clock = _probing_repo(fake_pool)

        await repo.schema_facts()
        # A failed later probe must not take back a fact already seen.
        fake_pool.recorder.rows_for[_PROBE] = RuntimeError("connection reset")
        clock.now += SCHEMA_FACTS_NEGATIVE_TTL_SECONDS
        facts = await repo.schema_facts()

        assert facts == SchemaFacts(has_v2_fts=False, has_copy_state=True)
        assert len(fake_pool.matching(_PROBE)) == 2

    async def test_invalidate_drops_the_cached_negative(self, fake_pool):
        fake_pool.recorder.rows_for[_PROBE] = [(False, False)]
        repo, _ = _probing_repo(fake_pool)

        await repo.schema_facts()
        fake_pool.recorder.rows_for[_PROBE] = [(False, True)]
        repo.invalidate_schema_facts()
        facts = await repo.schema_facts()

        assert facts.has_copy_state is True
        assert len(fake_pool.matching(_PROBE)) == 2

    async def test_a_failing_probe_reads_as_absent(self, fake_pool_factory, caplog):
        pool = fake_pool_factory(error=psycopg.OperationalError("db down"))
        repo, _ = _probing_repo(pool)

        with caplog.at_level("WARNING", logger="ariel"):
            facts = await repo.schema_facts()

        assert facts == SchemaFacts(has_v2_fts=False, has_copy_state=False)
        assert "Schema probe failed" in caplog.text

    async def test_no_row_reads_as_absent(self, fake_pool):
        repo, _ = _probing_repo(fake_pool)

        assert await repo.schema_facts() == SchemaFacts(False, False)


# ---------------------------------------------------------------------------
# Attachment readers -- column-explicit, schema-gated
# ---------------------------------------------------------------------------

_MIGRATED = SchemaFacts(has_v2_fts=False, has_copy_state=True)
_UNMIGRATED = SchemaFacts(has_v2_fts=False, has_copy_state=False)


def _gated_repo(pool: Any, facts: SchemaFacts) -> ARIELRepository:
    repo = ARIELRepository(pool, _make_config())

    async def _facts() -> SchemaFacts:
        return facts

    repo.schema_facts = _facts  # type: ignore[method-assign]
    return repo


def _select_clause(sql: str) -> str:
    return _sql_body(sql).split(" FROM ", 1)[0]


def _selected_columns(sql: str) -> tuple[str, ...]:
    clause = _select_clause(sql).removeprefix("SELECT ")
    return tuple(column.strip() for column in clause.split(","))


@pytest.fixture
def schema_gap_flag(monkeypatch):
    """Reset the once-per-process schema-gap flag for the test."""
    monkeypatch.setattr(repository_module, "_attachment_schema_gap_warned", False)


def _gap_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == ATTACHMENT_SCHEMA_GAP_WARNING]


class TestAttachmentRowColumns:
    def test_columns_are_the_thirteen_blob_free_names(self):
        assert ATTACHMENT_ROW_COLUMNS == (
            "attachment_id",
            "entry_id",
            "filename",
            "mime_type",
            "size_bytes",
            "source_url",
            "copy_status",
            "skip_reason",
            "copy_attempts",
            "rendition_mime",
            "rendition_w",
            "rendition_h",
            "rendition_sha256",
        )
        assert "data" not in ATTACHMENT_ROW_COLUMNS
        assert "rendition_bytes" not in ATTACHMENT_ROW_COLUMNS


class TestGetAttachmentRows:
    async def test_groups_rows_by_entry_with_named_any(self, fake_pool_factory):
        rows = [
            {"attachment_id": "a-1", "entry_id": "e-1"},
            {"attachment_id": "a-2", "entry_id": "e-2"},
            {"attachment_id": "a-3", "entry_id": "e-1"},
        ]
        pool = fake_pool_factory(results=[rows])
        repo = _gated_repo(pool, _MIGRATED)

        mapping = await repo.get_attachment_rows(["e-1", "e-2", "e-3"])

        assert mapping == {
            "e-1": [rows[0], rows[2]],
            "e-2": [rows[1]],
        }
        sql, params = pool.calls[0]
        assert len(pool.calls) == 1
        assert "entry_id = ANY(%(entry_ids)s)" in _sql_body(sql)
        assert params == {"entry_ids": ["e-1", "e-2", "e-3"]}
        assert _selected_columns(sql) == ATTACHMENT_ROW_COLUMNS

    async def test_selects_no_blob(self, fake_pool):
        repo = _gated_repo(fake_pool, _MIGRATED)

        assert await repo.get_attachment_rows(["e-1"]) == {}

        columns = _selected_columns(fake_pool.sql[0])
        assert "data" not in columns
        assert "rendition_bytes" not in columns

    async def test_empty_ids_return_empty_without_a_connection(self, fake_pool):
        repo = _gated_repo(fake_pool, _MIGRATED)

        assert await repo.get_attachment_rows([]) == {}
        assert fake_pool.calls == []

    async def test_empty_ids_skip_the_schema_gate(self, fake_pool):
        repo = ARIELRepository(fake_pool, _make_config())

        async def _raising() -> SchemaFacts:
            raise AssertionError("schema_facts must not be consulted")

        repo.schema_facts = _raising  # type: ignore[method-assign]

        assert await repo.get_attachment_rows([]) == {}
        assert fake_pool.calls == []

    @pytest.mark.usefixtures("schema_gap_flag")
    async def test_unmigrated_store_returns_none_and_warns_once(self, fake_pool, caplog):
        caplog.set_level(logging.WARNING, logger="ariel")
        repo = _gated_repo(fake_pool, _UNMIGRATED)

        assert await repo.get_attachment_rows(["e-1"]) is None
        assert await repo.get_attachment_rows(["e-2"]) is None

        assert fake_pool.calls == []
        assert len(_gap_warnings(caplog)) == 1

    async def test_driver_failure_is_wrapped(self, fake_pool_factory):
        driver_error = RuntimeError("connection reset by peer")
        repo = _gated_repo(fake_pool_factory(error=driver_error), _MIGRATED)

        with pytest.raises(DatabaseQueryError) as exc_info:
            await repo.get_attachment_rows(["e-1", "e-2"])

        assert exc_info.value.technical_details["query"] == (
            "SELECT attachment_files rows entry_ids=ANY([2 ids])"
        )
        assert exc_info.value.__cause__ is driver_error


class TestGetRendition:
    async def test_selects_row_columns_plus_rendition_bytes(self, fake_pool_factory):
        row = {"attachment_id": "a-1", "rendition_bytes": b"\xff\xd8"}
        pool = fake_pool_factory(results=[[row]])
        repo = _gated_repo(pool, _MIGRATED)

        assert await repo.get_rendition("a-1") == row

        sql, params = pool.calls[0]
        assert _selected_columns(sql) == (*ATTACHMENT_ROW_COLUMNS, "rendition_bytes")
        assert "data" not in _selected_columns(sql)
        assert "attachment_id = %(attachment_id)s" in _sql_body(sql)
        assert params == {"attachment_id": "a-1"}

    async def test_absent_rendition_is_none(self, fake_pool):
        repo = _gated_repo(fake_pool, _MIGRATED)

        assert await repo.get_rendition("a-1") is None
        assert "rendition_bytes IS NOT NULL" in _sql_body(fake_pool.sql[0])

    @pytest.mark.usefixtures("schema_gap_flag")
    async def test_unmigrated_store_returns_none_without_a_query(self, fake_pool, caplog):
        caplog.set_level(logging.WARNING, logger="ariel")
        repo = _gated_repo(fake_pool, _UNMIGRATED)

        assert await repo.get_rendition("a-1") is None
        assert fake_pool.calls == []
        assert len(_gap_warnings(caplog)) == 1

    async def test_driver_failure_is_wrapped(self, fake_pool_factory):
        driver_error = RuntimeError("connection reset by peer")
        repo = _gated_repo(fake_pool_factory(error=driver_error), _MIGRATED)

        with pytest.raises(DatabaseQueryError) as exc_info:
            await repo.get_rendition("a-1")

        assert exc_info.value.technical_details["query"] == (
            "SELECT attachment_files rendition attachment_id=a-1"
        )
        assert exc_info.value.__cause__ is driver_error


class TestGetAttachmentOriginal:
    async def test_reads_only_copied_rows_with_data(self, fake_pool_factory):
        row = {"filename": "shot.png", "mime_type": "image/png", "data": b"\x89PNG"}
        pool = fake_pool_factory(results=[[row]])
        repo = _gated_repo(pool, _MIGRATED)

        assert await repo.get_attachment_original("a-1") == row

        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "copy_status = 'copied'" in body
        assert "data IS NOT NULL" in body
        assert "rendition_bytes" not in _selected_columns(sql)
        assert "data" in _selected_columns(sql)
        assert params == {"id": "a-1"}

    async def test_absent_original_is_none(self, fake_pool):
        repo = _gated_repo(fake_pool, _MIGRATED)

        assert await repo.get_attachment_original("gone") is None

    @pytest.mark.usefixtures("schema_gap_flag")
    async def test_unmigrated_store_falls_back_to_the_b1_statement(self, fake_pool_factory, caplog):
        caplog.set_level(logging.WARNING, logger="ariel")
        row = {"filename": "shot.png", "mime_type": "image/png", "data": b"\x89PNG"}
        pool = fake_pool_factory(results=[[row]])
        repo = _gated_repo(pool, _UNMIGRATED)

        assert await repo.get_attachment_original("a-1") == row

        sql, params = pool.calls[0]
        assert _sql_body(sql) == (
            "SELECT filename, mime_type, data FROM attachment_files WHERE attachment_id = %(id)s"
        )
        assert params == {"id": "a-1"}
        assert len(_gap_warnings(caplog)) == 1

    async def test_driver_failure_is_wrapped(self, fake_pool_factory):
        driver_error = RuntimeError("connection reset by peer")
        repo = _gated_repo(fake_pool_factory(error=driver_error), _MIGRATED)

        with pytest.raises(DatabaseQueryError) as exc_info:
            await repo.get_attachment_original("a-1")

        assert exc_info.value.technical_details["query"] == (
            "SELECT attachment_files original attachment_id=a-1"
        )
        assert exc_info.value.__cause__ is driver_error


class TestAttachmentSchemaGapWarning:
    @pytest.mark.usefixtures("schema_gap_flag")
    def test_logs_once_per_process(self, caplog):
        caplog.set_level(logging.WARNING, logger="ariel")

        repository_module.warn_attachment_schema_gap_once()
        repository_module.warn_attachment_schema_gap_once()

        records = _gap_warnings(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING

    @pytest.mark.usefixtures("schema_gap_flag")
    async def test_every_reader_shares_the_one_warning(self, fake_pool, caplog):
        caplog.set_level(logging.WARNING, logger="ariel")
        repo = _gated_repo(fake_pool, _UNMIGRATED)

        await repo.get_attachment_rows(["e-1"])
        await repo.get_rendition("a-1")
        await repo.get_attachment_original("a-1")
        repository_module.warn_attachment_schema_gap_once()

        assert len(_gap_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# V2 search expressions -- selected by the has_v2_fts schema fact
# ---------------------------------------------------------------------------


def _raw_config() -> ARIELConfig:
    return _make_config(enhancement_modules={"semantic_processor": {"enabled": False}})


class TestV2SearchStatements:
    """``v2=True`` widens rank, headline and fuzzy similarity to ``attachment_text``."""

    async def test_keyword_search_defaults_to_the_v1_expressions(self, fake_pool) -> None:
        repo = ARIELRepository(fake_pool, _raw_config())

        await repo.keyword_search(where_clauses=[], params=[], search_text="quench")

        body = _sql_body(fake_pool.calls[0][0])
        assert "to_tsvector('english', raw_text)" in body
        assert "ts_headline('english', raw_text, plainto_tsquery('english', %s)" in body
        assert "attachment_text" not in body

    @pytest.mark.parametrize("semantic", [False, True])
    async def test_keyword_search_v2_ranks_and_highlights_the_v2_document(
        self, fake_pool, semantic: bool
    ) -> None:
        from osprey.services.ariel_search.database import search_fts

        config = _make_config() if semantic else _raw_config()
        repo = ARIELRepository(fake_pool, config)

        await repo.keyword_search(where_clauses=[], params=[], search_text="quench", v2=True)

        body = _sql_body(fake_pool.calls[0][0])
        expression, document = (
            (search_fts.SEMANTIC_FTS_EXPRESSION_V2, search_fts.SEMANTIC_TEXT_SEARCH_DOCUMENT_V2)
            if semantic
            else (search_fts.RAW_TEXT_FTS_EXPRESSION_V2, search_fts.RAW_TEXT_SEARCH_DOCUMENT_V2)
        )
        assert f"ts_rank( {expression}, plainto_tsquery('english', %s) )" in body
        assert f"ts_headline('english', {document}, plainto_tsquery('english', %s)" in body

    async def test_fuzzy_search_v2_takes_the_better_of_both_texts(self, fake_pool) -> None:
        repo = ARIELRepository(fake_pool, _raw_config())

        await repo.fuzzy_search("quench", threshold=0.25, v2=True)

        sql, params = fake_pool.calls[0]
        body = _sql_body(sql)
        greatest = (
            "GREATEST(similarity(raw_text, %s), similarity(COALESCE(attachment_text,''), %s))"
        )
        assert f"SELECT e.*, {greatest} AS sim" in body
        assert f"WHERE {greatest} >= %s" in body
        assert "ORDER BY sim DESC" in body
        assert params == ["quench", "quench", "quench", "quench", 0.25, 10]

    async def test_fuzzy_search_v1_is_b1_exact(self, fake_pool) -> None:
        repo = ARIELRepository(fake_pool, _raw_config())

        await repo.fuzzy_search("quench", v2=False)

        body = _sql_body(fake_pool.calls[0][0])
        assert "SELECT e.*, similarity(raw_text, %s) AS sim" in body
        assert "WHERE similarity(raw_text, %s) >= %s" in body
        assert "attachment_text" not in body

    async def test_probe_requires_the_attachment_text_trigram_index(self, fake_pool) -> None:
        repo, _ = _probing_repo(fake_pool)

        await repo.schema_facts()

        assert "to_regclass('idx_entries_attachment_text_trgm') IS NOT NULL" in fake_pool.sql[0]


_RANK_EXPRESSION = re.compile(r"ts_rank\( (.*?), \(?(?:plainto|websearch)_tsquery")


class TestOneQueryOneExpression:
    """The keyword module reads the fact once; match and rank share one constant."""

    @pytest.mark.parametrize("v2", [False, True])
    async def test_rank_and_where_use_the_same_constant(self, fake_pool, v2: bool) -> None:
        from osprey.services.ariel_search.database.search_fts import keyword_fts_expression
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = _raw_config()
        fake_pool.recorder.rows_for[_PROBE] = [(v2, True)]
        repo, _ = _probing_repo(fake_pool, config)

        await keyword_search("quench", repo, config, fuzzy_fallback=False)

        (search_sql, _params) = fake_pool.recorder.matching("ts_rank")[0]
        body = _sql_body(search_sql)
        expected = keyword_fts_expression(config, v2=v2)
        rank = _RANK_EXPRESSION.search(body)
        assert rank is not None
        assert rank.group(1) == expected
        assert f"WHERE {expected} @@ (plainto_tsquery('english', %s))" in body
        assert len(fake_pool.recorder.matching(_PROBE)) == 1

    async def test_a_running_panel_flips_to_v2_within_the_ttl(self, fake_pool) -> None:
        """A store migrated after the panel started is searched with V2 after 60 s."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = _raw_config()
        fake_pool.recorder.rows_for[_PROBE] = [(False, True)]
        repo, clock = _probing_repo(fake_pool, config)

        async def searched_with_v2() -> bool:
            fake_pool.recorder.calls.clear()
            await keyword_search("quench", repo, config, fuzzy_fallback=False)
            (search_sql, _params) = fake_pool.recorder.matching("ts_rank")[0]
            return "attachment_text" in search_sql

        assert not await searched_with_v2()

        fake_pool.recorder.rows_for[_PROBE] = [(True, True)]  # the migration ran
        clock.now += SCHEMA_FACTS_NEGATIVE_TTL_SECONDS - 1
        assert not await searched_with_v2()

        clock.now += 2
        assert await searched_with_v2()
