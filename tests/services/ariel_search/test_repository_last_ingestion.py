"""Unit tests for the facility-wide ingestion watermark.

``ARIELRepository.get_last_ingestion`` is the unfiltered sibling of
``get_last_successful_run``: no ``source_system`` predicate, so it answers
"when did this deployment last ingest anything successfully". Two surfaces read
it -- the ARIEL dashboard panel and ``osprey ariel status`` -- and the point of
these tests is that both read the *same* definition: the service no longer
carries its own copy of the statement.

Assertions target emitted SQL and bound parameters against the fake pool from
``conftest``; nothing here talks to Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import ARIELRepository
from osprey.services.ariel_search.exceptions import DatabaseQueryError
from osprey.services.ariel_search.service import ARIELSearchService

_UNFILTERED_SQL = "SELECT MAX(completed_at) FROM ingestion_runs WHERE status = 'success'"


def _make_config(**overrides: Any) -> ARIELConfig:
    """Minimal config; the watermark query gates on no module."""
    data: dict[str, Any] = {
        "database": {"uri": "postgresql://user:pass@localhost:5432/ariel"},
        "search_modules": {"keyword": {"enabled": True}},
        "enhancement_modules": {"text_embedding": {"enabled": True}},
    }
    data.update(overrides)
    return ARIELConfig.from_dict(data)


def _sql_body(sql: str) -> str:
    """SQL with whitespace collapsed, so a multi-line literal matches one line."""
    return " ".join(str(sql).split())


class TestGetLastIngestion:
    """The repository method itself."""

    async def test_returns_the_max_completed_at_across_all_sources(
        self,
        fake_pool_factory,
    ) -> None:
        """The watermark is MAX(completed_at) over successful runs, unfiltered.

        Hazard: adding a ``source_system`` predicate here would silently turn a
        facility-wide "last ingestion" into a per-adapter one, and a deployment
        with several adapters would report the wrong timestamp.
        """
        completed = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
        pool = fake_pool_factory(results=[[(completed,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_ingestion() == completed

        sql, params = pool.calls[0]
        body = _sql_body(sql)
        assert "SELECT MAX(completed_at) FROM ingestion_runs" in body
        assert "status = 'success'" in body
        assert "source_system" not in body
        assert not params

    async def test_returns_none_without_any_run(self, fake_pool) -> None:
        """No rows at all means no watermark."""
        repo = ARIELRepository(fake_pool, _make_config())

        assert await repo.get_last_ingestion() is None

    async def test_returns_none_for_a_null_aggregate(self, fake_pool_factory) -> None:
        """MAX over zero successful runs is one row holding NULL.

        Hazard: a truthiness check on the row alone would hand callers a None
        timestamp wrapped in a row and report it as a real ingestion time.
        """
        pool = fake_pool_factory(results=[[(None,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_ingestion() is None

    async def test_driver_failure_becomes_database_query_error(
        self,
        fake_pool_factory,
    ) -> None:
        """A failed query is wrapped with a breadcrumb naming what was run."""
        driver_error = RuntimeError("connection reset by peer")
        repo = ARIELRepository(fake_pool_factory(error=driver_error), _make_config())

        with pytest.raises(DatabaseQueryError) as exc_info:
            await repo.get_last_ingestion()

        assert exc_info.value.technical_details["query"] == "SELECT MAX(completed_at) all sources"
        assert "connection reset by peer" in exc_info.value.message
        assert exc_info.value.__cause__ is driver_error

    async def test_get_last_successful_run_still_filters_by_source(
        self,
        fake_pool_factory,
    ) -> None:
        """The per-source watermark is untouched; the two are separate readers."""
        completed = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
        pool = fake_pool_factory(results=[[(completed,)]])
        repo = ARIELRepository(pool, _make_config())

        assert await repo.get_last_successful_run("als_logbook") == completed

        sql, params = pool.calls[0]
        assert "source_system = %s" in _sql_body(sql)
        assert params == ["als_logbook"]


class TestServiceStatusUsesTheRepository:
    """``ARIELService.get_status`` reads the watermark through the repository."""

    @staticmethod
    def _service(pool: Any, repository: Any) -> ARIELSearchService:
        return ARIELSearchService(config=_make_config(), pool=pool, repository=repository)

    async def test_status_reports_what_the_repository_returns(self, fake_pool_factory) -> None:
        """The panel's timestamp comes from ``get_last_ingestion``, not inline SQL."""
        completed = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
        pool = fake_pool_factory(rows_for={"SELECT COUNT(*) FROM enhanced_entries": [(7,)]})
        repository = MagicMock()
        repository.get_embedding_tables = AsyncMock(return_value=[])
        repository.get_last_ingestion = AsyncMock(return_value=completed)

        result = await self._service(pool, repository).get_status()

        assert result.last_ingestion == completed
        assert result.errors == []
        repository.get_last_ingestion.assert_awaited_once_with()
        # The statement lives in the repository only -- the service runs it nowhere.
        assert not [sql for sql in pool.sql if "MAX(completed_at)" in _sql_body(sql)]

    async def test_status_reports_no_watermark_when_there_is_none(
        self,
        fake_pool_factory,
    ) -> None:
        """A None watermark stays None rather than becoming an error."""
        pool = fake_pool_factory(rows_for={"SELECT COUNT(*) FROM enhanced_entries": [(0,)]})
        repository = MagicMock()
        repository.get_embedding_tables = AsyncMock(return_value=[])
        repository.get_last_ingestion = AsyncMock(return_value=None)

        result = await self._service(pool, repository).get_status()

        assert result.last_ingestion is None
        assert result.errors == []

    async def test_a_failing_watermark_query_becomes_a_status_error(
        self,
        fake_pool_factory,
    ) -> None:
        """The repository's DatabaseQueryError is reported, not raised at the caller.

        Hazard: ``osprey ariel status`` exists to describe a sick deployment, so
        a database failure has to arrive as an error string on the result.
        """
        pool = fake_pool_factory(rows_for={"SELECT COUNT(*) FROM enhanced_entries": [(7,)]})
        repository = MagicMock()
        repository.get_embedding_tables = AsyncMock(return_value=[])
        repository.get_last_ingestion = AsyncMock(side_effect=RuntimeError("watermark exploded"))

        result = await self._service(pool, repository).get_status()

        assert result.last_ingestion is None
        assert result.healthy is False
        assert any("watermark exploded" in error for error in result.errors)


def test_the_unfiltered_statement_appears_once_in_the_package() -> None:
    """One definition, two readers: the service must not re-declare the SQL."""
    from pathlib import Path

    import osprey.services.ariel_search as package

    hits = [
        path.relative_to(Path(package.__file__).parent).as_posix()
        for path in Path(package.__file__).parent.rglob("*.py")
        if _UNFILTERED_SQL in _sql_body(path.read_text(encoding="utf-8"))
    ]
    assert hits == ["database/repository.py"]
