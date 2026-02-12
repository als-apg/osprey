"""Tests for ARIEL ingestion scheduler."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    IngestionConfig,
    WatchConfig,
)
from osprey.services.ariel_search.ingestion.scheduler import (
    IngestionPollResult,
    IngestionScheduler,
)


def _make_config(
    *,
    poll_interval: int = 60,
    require_initial: bool = True,
    max_failures: int = 10,
    backoff_multiplier: float = 2.0,
    max_interval: int = 3600,
    source_url: str = "https://api.example.com/logbook",
    adapter: str = "generic_json",
) -> ARIELConfig:
    """Build a minimal ARIELConfig for scheduler tests."""
    return ARIELConfig(
        database=DatabaseConfig(uri="postgresql://localhost/test"),
        ingestion=IngestionConfig(
            adapter=adapter,
            source_url=source_url,
            poll_interval_seconds=poll_interval,
            watch=WatchConfig(
                require_initial_ingest=require_initial,
                max_consecutive_failures=max_failures,
                backoff_multiplier=backoff_multiplier,
                max_interval_seconds=max_interval,
            ),
        ),
    )


def _make_entry(entry_id: str = "e1") -> dict:
    """Build a mock entry dict."""
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        "author": "tester",
        "raw_text": "test entry",
        "attachments": [],
        "metadata": {},
        "enhancement_status": {},
    }


def _mock_adapter(entries: list[dict] | None = None):
    """Create a mock adapter that yields entries."""
    adapter = MagicMock()
    adapter.source_system_name = "test_system"

    async def _fetch(since=None, until=None, limit=None):
        for entry in entries or []:
            yield entry

    adapter.fetch_entries = _fetch
    return adapter


class TestIngestionPollResult:
    """Tests for IngestionPollResult dataclass."""

    def test_creation(self) -> None:
        """IngestionPollResult stores all fields."""
        since = datetime(2024, 1, 1, tzinfo=UTC)
        result = IngestionPollResult(
            entries_added=5,
            entries_updated=2,
            entries_failed=1,
            duration_seconds=3.5,
            since=since,
        )
        assert result.entries_added == 5
        assert result.entries_updated == 2
        assert result.entries_failed == 1
        assert result.duration_seconds == 3.5
        assert result.since == since

    def test_creation_no_since(self) -> None:
        """IngestionPollResult works with since=None."""
        result = IngestionPollResult(
            entries_added=0,
            entries_updated=0,
            entries_failed=0,
            duration_seconds=0.1,
            since=None,
        )
        assert result.since is None


class TestIngestionScheduler:
    """Tests for IngestionScheduler."""

    @pytest.fixture
    def config(self) -> ARIELConfig:
        """Default test config."""
        return _make_config()

    @pytest.fixture
    def repository(self) -> MagicMock:
        """Mock repository with all required async methods."""
        repo = MagicMock()
        repo.pool = MagicMock()
        repo.pool.connection = MagicMock(return_value=AsyncMock())
        repo.start_ingestion_run = AsyncMock(return_value=1)
        repo.complete_ingestion_run = AsyncMock()
        repo.fail_ingestion_run = AsyncMock()
        repo.get_last_successful_run = AsyncMock(return_value=None)
        repo.upsert_entry = AsyncMock()
        repo.mark_enhancement_complete = AsyncMock()
        repo.mark_enhancement_failed = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_poll_once_success(self, config, repository) -> None:
        """poll_once stores entries and records successful run."""
        entries = [_make_entry("e1"), _make_entry("e2")]
        adapter = _mock_adapter(entries)
        last_time = datetime(2024, 1, 1, tzinfo=UTC)
        repository.get_last_successful_run = AsyncMock(return_value=last_time)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 2
        assert result.entries_failed == 0
        assert result.since == last_time
        assert result.duration_seconds >= 0
        repository.start_ingestion_run.assert_called_once_with("test_system")
        repository.complete_ingestion_run.assert_called_once_with(
            1, entries_added=2, entries_updated=0, entries_failed=0
        )
        assert repository.upsert_entry.call_count == 2

    @pytest.mark.asyncio
    async def test_poll_once_no_entries(self, config, repository) -> None:
        """poll_once records empty run when no entries found."""
        adapter = _mock_adapter([])
        last_time = datetime(2024, 1, 1, tzinfo=UTC)
        repository.get_last_successful_run = AsyncMock(return_value=last_time)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 0
        assert result.entries_failed == 0
        repository.complete_ingestion_run.assert_called_once_with(
            1, entries_added=0, entries_updated=0, entries_failed=0
        )

    @pytest.mark.asyncio
    async def test_poll_once_entry_error(self, config, repository) -> None:
        """poll_once counts enhancement failures but still succeeds."""
        entries = [_make_entry("e1")]
        adapter = _mock_adapter(entries)
        last_time = datetime(2024, 1, 1, tzinfo=UTC)
        repository.get_last_successful_run = AsyncMock(return_value=last_time)

        # Create a mock enhancer that raises
        failing_enhancer = MagicMock()
        failing_enhancer.name = "text_embedding"
        failing_enhancer.enhance = AsyncMock(side_effect=RuntimeError("model unavailable"))

        # Mock pool.connection as async context manager
        mock_conn = AsyncMock()
        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[failing_enhancer],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 1
        assert result.entries_failed == 1
        repository.mark_enhancement_failed.assert_called_once()
        # Run still completes (not failed)
        repository.complete_ingestion_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_once_adapter_error(self, config, repository) -> None:
        """poll_once calls fail_ingestion_run when adapter raises."""
        last_time = datetime(2024, 1, 1, tzinfo=UTC)
        repository.get_last_successful_run = AsyncMock(return_value=last_time)

        # Adapter that raises during iteration
        adapter = MagicMock()
        adapter.source_system_name = "test_system"

        async def _fetch_error(**kwargs):
            raise ConnectionError("API unreachable")
            yield  # make it a generator  # noqa: E501

        adapter.fetch_entries = _fetch_error

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            with pytest.raises(ConnectionError, match="API unreachable"):
                await scheduler.poll_once()

        repository.fail_ingestion_run.assert_called_once()
        args = repository.fail_ingestion_run.call_args
        assert args[0][0] == 1  # run_id
        assert "API unreachable" in args[0][1]

    @pytest.mark.asyncio
    async def test_poll_once_dry_run(self, config, repository) -> None:
        """poll_once with dry_run=True does not store entries."""
        entries = [_make_entry("e1"), _make_entry("e2"), _make_entry("e3")]
        adapter = _mock_adapter(entries)
        last_time = datetime(2024, 1, 1, tzinfo=UTC)
        repository.get_last_successful_run = AsyncMock(return_value=last_time)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once(dry_run=True)

        assert result.entries_added == 3
        assert result.entries_failed == 0
        # No repository writes in dry-run mode
        repository.start_ingestion_run.assert_not_called()
        repository.upsert_entry.assert_not_called()
        repository.complete_ingestion_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_since_detection(self, config, repository) -> None:
        """poll_once uses last successful run time as since-parameter."""
        last_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        repository.get_last_successful_run = AsyncMock(return_value=last_time)

        # Track calls to fetch_entries to verify since parameter
        fetch_calls = []
        adapter = MagicMock()
        adapter.source_system_name = "test_system"

        async def _fetch(since=None, until=None, limit=None):
            fetch_calls.append(since)
            return
            yield  # make it a generator  # noqa: E501

        adapter.fetch_entries = _fetch

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.since == last_time
        assert fetch_calls == [last_time]

    @pytest.mark.asyncio
    async def test_auto_since_no_history_requires_initial(self, repository) -> None:
        """poll_once returns early when no history and require_initial_ingest=True."""
        config = _make_config(require_initial=True)
        repository.get_last_successful_run = AsyncMock(return_value=None)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=_mock_adapter([]),
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 0
        # Should not have started a run
        repository.start_ingestion_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_since_no_history_not_required(self, repository) -> None:
        """poll_once proceeds with since=None when require_initial_ingest=False."""
        config = _make_config(require_initial=False)
        repository.get_last_successful_run = AsyncMock(return_value=None)

        entries = [_make_entry("e1")]
        adapter = _mock_adapter(entries)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 1
        assert result.since is None
        repository.start_ingestion_run.assert_called_once()
        repository.complete_ingestion_run.assert_called_once()

    def test_backoff_on_consecutive_failures(self) -> None:
        """Interval increases after consecutive failures."""
        config = _make_config(poll_interval=60, backoff_multiplier=2.0, max_interval=3600)
        repo = MagicMock()
        scheduler = IngestionScheduler(config=config, repository=repo)

        # No failures: base interval
        assert scheduler._get_current_interval() == 60.0

        # 1 failure: 60 * 2^1 = 120
        scheduler._consecutive_failures = 1
        assert scheduler._get_current_interval() == 120.0

        # 2 failures: 60 * 2^2 = 240
        scheduler._consecutive_failures = 2
        assert scheduler._get_current_interval() == 240.0

        # 3 failures: 60 * 2^3 = 480
        scheduler._consecutive_failures = 3
        assert scheduler._get_current_interval() == 480.0

    def test_backoff_capped_at_max(self) -> None:
        """Backoff interval is capped at max_interval_seconds."""
        config = _make_config(poll_interval=60, backoff_multiplier=2.0, max_interval=300)
        repo = MagicMock()
        scheduler = IngestionScheduler(config=config, repository=repo)

        # 10 failures: 60 * 2^10 = 61440, but capped at 300
        scheduler._consecutive_failures = 10
        assert scheduler._get_current_interval() == 300.0

    def test_backoff_resets_on_success(self) -> None:
        """Interval resets to base after a successful poll."""
        config = _make_config(poll_interval=60, backoff_multiplier=2.0)
        repo = MagicMock()
        scheduler = IngestionScheduler(config=config, repository=repo)

        scheduler._consecutive_failures = 5
        assert scheduler._get_current_interval() > 60.0

        # Simulate success
        scheduler._consecutive_failures = 0
        assert scheduler._get_current_interval() == 60.0

    @pytest.mark.asyncio
    async def test_poll_once_enhancement_success(self, config, repository) -> None:
        """poll_once calls mark_enhancement_complete when enhancer succeeds."""
        entries = [_make_entry("e1")]
        adapter = _mock_adapter(entries)
        repository.get_last_successful_run = AsyncMock(
            return_value=datetime(2024, 1, 1, tzinfo=UTC)
        )

        # Create a succeeding enhancer
        succeeding_enhancer = MagicMock()
        succeeding_enhancer.name = "text_embedding"
        succeeding_enhancer.enhance = AsyncMock(return_value=None)

        # Mock pool.connection as async context manager
        mock_conn = AsyncMock()
        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[succeeding_enhancer],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 1
        assert result.entries_failed == 0
        repository.mark_enhancement_complete.assert_called_once_with("e1", "text_embedding")
        repository.mark_enhancement_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_once_mixed_enhancers(self, config, repository) -> None:
        """poll_once handles mixed success/failure across multiple enhancers."""
        entries = [_make_entry("e1")]
        adapter = _mock_adapter(entries)
        repository.get_last_successful_run = AsyncMock(
            return_value=datetime(2024, 1, 1, tzinfo=UTC)
        )

        # First enhancer succeeds, second fails
        good_enhancer = MagicMock()
        good_enhancer.name = "text_embedding"
        good_enhancer.enhance = AsyncMock(return_value=None)

        bad_enhancer = MagicMock()
        bad_enhancer.name = "semantic_processor"
        bad_enhancer.enhance = AsyncMock(side_effect=RuntimeError("model unavailable"))

        # Mock pool.connection as async context manager
        mock_conn = AsyncMock()
        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[good_enhancer, bad_enhancer],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 1
        assert result.entries_failed == 1
        repository.mark_enhancement_complete.assert_called_once_with("e1", "text_embedding")
        repository.mark_enhancement_failed.assert_called_once()
        fail_args = repository.mark_enhancement_failed.call_args
        assert fail_args[0] == ("e1", "semantic_processor", "model unavailable")

    @pytest.mark.asyncio
    async def test_start_exits_after_max_failures(self, repository) -> None:
        """start() exits after max_consecutive_failures without hanging."""
        config = _make_config(max_failures=2, poll_interval=0)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=_mock_adapter([]),
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)

            # Make poll_once always raise
            scheduler.poll_once = AsyncMock(side_effect=ConnectionError("API down"))

            # start() should exit after 2 failures, not hang
            await asyncio.wait_for(scheduler.start(), timeout=5.0)

        assert scheduler._consecutive_failures == 2
        assert scheduler.poll_once.call_count == 2

    @pytest.mark.asyncio
    async def test_start_resets_failures_on_success(self, repository) -> None:
        """start() resets _consecutive_failures to 0 after a successful poll."""
        config = _make_config(max_failures=5, poll_interval=0)

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=_mock_adapter([]),
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)

            call_count = 0
            success_result = IngestionPollResult(
                entries_added=1,
                entries_updated=0,
                entries_failed=0,
                duration_seconds=0.1,
                since=None,
            )

            async def _poll_sequence(dry_run=False):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("temporary blip")
                elif call_count == 2:
                    return success_result
                else:
                    await scheduler.stop()
                    return success_result

            scheduler.poll_once = AsyncMock(side_effect=_poll_sequence)

            await asyncio.wait_for(scheduler.start(), timeout=5.0)

        # After call 1 (fail): _consecutive_failures = 1
        # After call 2 (success): _consecutive_failures = 0 (reset at line 78)
        assert scheduler._consecutive_failures == 0
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_stop_event(self, config, repository) -> None:
        """start() exits when stop_event is set."""
        repository.get_last_successful_run = AsyncMock(
            return_value=datetime(2024, 1, 1, tzinfo=UTC)
        )

        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=_mock_adapter([]),
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)

            # Set stop event immediately so the loop exits after one poll
            async def _stop_after_poll():
                await asyncio.sleep(0.01)
                await scheduler.stop()

            # Run start() and stop in parallel
            await asyncio.gather(scheduler.start(), _stop_after_poll())

        # Verify it ran at least one poll and stopped
        repository.start_ingestion_run.assert_called()


class TestIngestionRunTracking:
    """Tests for repository ingestion run tracking methods.

    These test the method signatures and SQL patterns using mocks.
    """

    @pytest.fixture
    def repository(self):
        """Create a repository with mocked pool."""
        from osprey.services.ariel_search.database.repository import ARIELRepository

        mock_pool = MagicMock()
        mock_config = MagicMock()
        return ARIELRepository(pool=mock_pool, config=mock_config)

    @pytest.mark.asyncio
    async def test_start_ingestion_run(self, repository) -> None:
        """start_ingestion_run returns an integer run ID."""
        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(42,))

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        run_id = await repository.start_ingestion_run("als_elog")
        assert run_id == 42
        mock_conn.execute.assert_called_once()
        call_sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO ingestion_runs" in call_sql
        assert "RETURNING id" in call_sql

    @pytest.mark.asyncio
    async def test_complete_ingestion_run(self, repository) -> None:
        """complete_ingestion_run updates status to success."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        await repository.complete_ingestion_run(
            run_id=1, entries_added=10, entries_updated=2, entries_failed=1
        )
        mock_conn.execute.assert_called_once()
        call_sql = mock_conn.execute.call_args[0][0]
        assert "UPDATE ingestion_runs" in call_sql
        assert "status = 'success'" in call_sql
        call_params = mock_conn.execute.call_args[0][1]
        assert call_params == [10, 2, 1, 1]

    @pytest.mark.asyncio
    async def test_fail_ingestion_run(self, repository) -> None:
        """fail_ingestion_run updates status to failed with error."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        await repository.fail_ingestion_run(run_id=1, error_message="timeout")
        mock_conn.execute.assert_called_once()
        call_sql = mock_conn.execute.call_args[0][0]
        assert "UPDATE ingestion_runs" in call_sql
        assert "status = 'failed'" in call_sql
        call_params = mock_conn.execute.call_args[0][1]
        assert "timeout" in call_params
        assert 1 in call_params

    @pytest.mark.asyncio
    async def test_get_last_successful_run(self, repository) -> None:
        """get_last_successful_run returns datetime from DB."""
        last_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(last_time,))

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        result = await repository.get_last_successful_run("als_elog")
        assert result == last_time

    @pytest.mark.asyncio
    async def test_get_last_successful_run_none(self, repository) -> None:
        """get_last_successful_run returns None when no runs found."""
        mock_result = MagicMock()
        mock_result.fetchone = AsyncMock(return_value=(None,))

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        repository.pool.connection = MagicMock(return_value=conn_cm)

        result = await repository.get_last_successful_run("als_elog")
        assert result is None
