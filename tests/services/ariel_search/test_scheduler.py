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
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.ingestion.scheduler import (
    IngestionPollResult,
    IngestionScheduler,
    StopReason,
)

#: A store without the attachment copy state: ``ingest_one`` takes the plain upsert.
_PLAIN_STORE = SchemaFacts(has_v2_fts=False, has_copy_state=False)


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


def _mock_adapter(entries: list[dict] | None = None, unreadable: int = 0):
    """Create a mock adapter that yields entries and reports ``unreadable`` skips."""
    adapter = MagicMock()
    adapter.source_system_name = "test_system"
    adapter.unreadable_entries = unreadable

    async def _fetch(since=None, until=None, limit=None):  # noqa: ARG001 - the ingestion adapter fetch_entries signature
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
        repo.schema_facts = AsyncMock(return_value=_PLAIN_STORE)
        repo.get_copy_retry_candidates = AsyncMock(return_value=[])
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
            yield  # make it a generator

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
    async def test_poll_once_counts_unreadable_entries_as_failed(self, config, repository) -> None:
        """Entries the adapter could not read count as failed in the run and the result."""
        adapter = _mock_adapter([_make_entry("e1")], unreadable=2)
        repository.get_last_successful_run = AsyncMock(
            return_value=datetime(2024, 1, 1, tzinfo=UTC)
        )

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
        assert result.entries_failed == 2
        repository.complete_ingestion_run.assert_awaited_once_with(
            1, entries_added=1, entries_updated=0, entries_failed=2
        )

    @pytest.mark.asyncio
    async def test_poll_once_dry_run_reports_unreadable_entries(self, config, repository) -> None:
        """A dry run reports the entries the adapter could not read as failed."""
        adapter = _mock_adapter([_make_entry("e1")], unreadable=2)
        repository.get_last_successful_run = AsyncMock(
            return_value=datetime(2024, 1, 1, tzinfo=UTC)
        )

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

        assert result.entries_added == 1
        assert result.entries_failed == 2
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

        async def _fetch(since=None, until=None, limit=None):  # noqa: ARG001 - the ingestion adapter fetch_entries signature
            fetch_calls.append(since)
            return
            yield  # make it a generator

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
    async def test_run_forever_exits_after_max_failures(self, repository) -> None:
        """run_forever() exits after max_consecutive_failures without hanging."""
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

            # run_forever() should exit after 2 failures, not hang
            await asyncio.wait_for(scheduler.run_forever(), timeout=5.0)

        assert scheduler._consecutive_failures == 2
        assert scheduler.poll_once.call_count == 2

    @pytest.mark.asyncio
    async def test_run_forever_resets_failures_on_success(self, repository) -> None:
        """run_forever() resets _consecutive_failures to 0 after a successful poll."""
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

            async def _poll_sequence(dry_run=False):  # noqa: ARG001 - the poll_once signature
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

            await asyncio.wait_for(scheduler.run_forever(), timeout=5.0)

        # After call 1 (fail): _consecutive_failures = 1
        # After call 2 (success): _consecutive_failures = 0 (reset at line 78)
        assert scheduler._consecutive_failures == 0
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_stop_event(self, config, repository) -> None:
        """run_forever() exits when stop_event is set."""
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

            # Run run_forever() and stop in parallel
            await asyncio.gather(scheduler.run_forever(), _stop_after_poll())

        # Verify it ran at least one poll and stopped
        repository.start_ingestion_run.assert_called()

    @pytest.mark.asyncio
    async def test_run_forever_returns_failure_cap(self, repository) -> None:
        """run_forever() reports FAILURE_CAP when the failure cap ends the loop."""
        config = _make_config(max_failures=2, poll_interval=0)
        scheduler = IngestionScheduler(config=config, repository=repository)
        scheduler.poll_once = AsyncMock(side_effect=ConnectionError("API down"))

        reason = await asyncio.wait_for(scheduler.run_forever(), timeout=5.0)

        assert reason is StopReason.FAILURE_CAP
        assert scheduler.poll_once.call_count == 2

    @pytest.mark.asyncio
    async def test_run_forever_returns_signal_on_stop(self, config, repository) -> None:
        """run_forever() reports SIGNAL when stop() ends the loop."""
        scheduler = IngestionScheduler(config=config, repository=repository)
        poll_result = IngestionPollResult(
            entries_added=0,
            entries_updated=0,
            entries_failed=0,
            duration_seconds=0.1,
            since=None,
        )

        async def _poll(dry_run=False):  # noqa: ARG001 - the poll_once signature
            await scheduler.stop()
            return poll_result

        scheduler.poll_once = AsyncMock(side_effect=_poll)

        reason = await asyncio.wait_for(scheduler.run_forever(), timeout=5.0)

        assert reason is StopReason.SIGNAL
        assert scheduler.poll_once.call_count == 1

    @pytest.mark.asyncio
    async def test_run_forever_returns_signal_when_stopped_before_start(
        self, config, repository
    ) -> None:
        """run_forever() reports SIGNAL when stop() was called before it ran."""
        scheduler = IngestionScheduler(config=config, repository=repository)
        scheduler.poll_once = AsyncMock()
        await scheduler.stop()

        reason = await asyncio.wait_for(scheduler.run_forever(), timeout=5.0)

        assert reason is StopReason.SIGNAL
        scheduler.poll_once.assert_not_called()


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


class TestSidecarMetadataWiring:
    """The scheduler hands the sidecar step the adapter, which carries its transport."""

    @pytest.mark.asyncio
    async def test_the_step_receives_the_adapter(self) -> None:
        """Without it, the step has no sidecar names, origins or file base and fetches nothing."""
        config = _make_config()
        adapter = _mock_adapter([_make_entry("e1")])

        repository = MagicMock()
        repository.pool = MagicMock()
        repository.pool.connection = MagicMock(return_value=AsyncMock())
        repository.start_ingestion_run = AsyncMock(return_value=1)
        repository.complete_ingestion_run = AsyncMock()
        repository.fail_ingestion_run = AsyncMock()
        repository.get_last_successful_run = AsyncMock(
            return_value=datetime(2026, 1, 1, tzinfo=UTC)
        )
        repository.upsert_entry = AsyncMock()
        repository.schema_facts = AsyncMock(return_value=_PLAIN_STORE)

        extract = AsyncMock()
        with (
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
            patch(
                "osprey.services.ariel_search.ingestion.ingest.extract_metadata_from_attachments",
                extract,
            ),
        ):
            scheduler = IngestionScheduler(config=config, repository=repository)
            await scheduler.poll_once()

        extract.assert_awaited_once()
        kwargs = extract.await_args.kwargs
        assert kwargs["adapter"] is adapter
        assert "ingestion" not in kwargs


def _poll_patches(adapter, enhancers: list | None = None):
    """Patch the poll's adapter and enhancer factories at their owning modules."""
    return (
        patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter),
        patch(
            "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
            return_value=enhancers or [],
        ),
    )


class _RunLedger:
    """An in-memory ``ingestion_runs``: the watermark is the latest successful start."""

    def __init__(self, seed: datetime) -> None:
        self.runs: dict[int, dict] = {0: {"started_at": seed, "status": "success"}}

    async def start(self, _source: str) -> int:
        run_id = len(self.runs)
        self.runs[run_id] = {"started_at": datetime.now(UTC), "status": "running"}
        return run_id

    async def complete(self, run_id: int, **_counts) -> None:
        self.runs[run_id]["status"] = "success"

    async def fail(self, run_id: int, _message: str) -> None:
        self.runs[run_id]["status"] = "failed"

    async def watermark(self, _source: str) -> datetime | None:
        starts = [r["started_at"] for r in self.runs.values() if r["status"] == "success"]
        return max(starts, default=None)

    def install(self, repo: MagicMock) -> None:
        repo.start_ingestion_run = AsyncMock(side_effect=self.start)
        repo.complete_ingestion_run = AsyncMock(side_effect=self.complete)
        repo.fail_ingestion_run = AsyncMock(side_effect=self.fail)
        repo.get_last_successful_run = AsyncMock(side_effect=self.watermark)


class TestPollThroughIngestOne:
    """Every polled entry goes through ``ingest_one``; only an all-unstored poll fails."""

    @pytest.fixture
    def repository(self) -> MagicMock:
        repo = MagicMock()
        repo.pool = MagicMock()
        repo.pool.connection = MagicMock(return_value=AsyncMock())
        repo.upsert_entry = AsyncMock()
        repo.mark_enhancement_complete = AsyncMock()
        repo.mark_enhancement_failed = AsyncMock()
        repo.schema_facts = AsyncMock(return_value=_PLAIN_STORE)
        repo.get_copy_retry_candidates = AsyncMock(return_value=[])
        _RunLedger(datetime(2024, 1, 1, tzinfo=UTC)).install(repo)
        return repo

    @pytest.mark.asyncio
    async def test_every_entry_failing_leaves_the_watermark_and_returns(self, repository):
        """Nothing stored: the run is failed, the poll returns, the watermark holds."""
        repository.upsert_entry = AsyncMock(side_effect=RuntimeError("db is gone"))
        before = await repository.get_last_successful_run("test_system")
        adapter = _mock_adapter([_make_entry("e1"), _make_entry("e2")])

        get_adapter, get_enhancers = _poll_patches(adapter)
        with get_adapter, get_enhancers:
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 0
        assert result.entries_failed == 2
        repository.fail_ingestion_run.assert_awaited_once()
        repository.complete_ingestion_run.assert_not_awaited()
        assert await repository.get_last_successful_run("test_system") == before

    @pytest.mark.asyncio
    async def test_all_entries_failing_does_not_feed_the_failure_cap(self, repository):
        """run_forever counts only raised polls; an all-unstored poll returns normally."""
        repository.upsert_entry = AsyncMock(side_effect=RuntimeError("db is gone"))
        adapter = _mock_adapter([_make_entry("e1")])
        config = _make_config(max_failures=1, poll_interval=0)

        get_adapter, get_enhancers = _poll_patches(adapter)
        with get_adapter, get_enhancers:
            scheduler = IngestionScheduler(config=config, repository=repository)
            real_poll = scheduler.poll_once
            polls = 0

            async def _poll_twice(dry_run=False):  # noqa: ARG001 - the poll_once signature
                nonlocal polls
                polls += 1
                if polls == 2:
                    await scheduler.stop()
                return await real_poll()

            scheduler.poll_once = _poll_twice
            stop = await asyncio.wait_for(scheduler.run_forever(), timeout=5.0)

        assert stop is StopReason.SIGNAL
        assert scheduler._consecutive_failures == 0
        assert repository.fail_ingestion_run.await_count == 2

    @pytest.mark.asyncio
    async def test_every_enhancer_failing_still_advances_the_watermark(self, repository):
        """The text is stored, so the run succeeds and its start becomes the watermark."""
        before = await repository.get_last_successful_run("test_system")
        failing = MagicMock()
        failing.name = "text_embedding"
        failing.enhance = AsyncMock(side_effect=RuntimeError("model unavailable"))
        adapter = _mock_adapter([_make_entry("e1"), _make_entry("e2")])

        get_adapter, get_enhancers = _poll_patches(adapter, [failing])
        with get_adapter, get_enhancers:
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 2
        assert result.entries_failed == 2
        repository.complete_ingestion_run.assert_awaited_once()
        repository.fail_ingestion_run.assert_not_awaited()
        assert await repository.get_last_successful_run("test_system") > before

    @pytest.mark.asyncio
    async def test_attachment_recording_failure_counts_but_the_run_succeeds(self, repository):
        """A savepoint failure keeps the text: one failed count, no pinned watermark."""
        from osprey.services.ariel_search.ingestion.ingest import EntryIngestOutcome

        adapter = _mock_adapter([_make_entry("e1")])
        outcome = EntryIngestOutcome(enhanced=0, enhancer_failed=0, attachments_recorded=False)

        get_adapter, get_enhancers = _poll_patches(adapter)
        with (
            get_adapter,
            get_enhancers,
            patch(
                "osprey.services.ariel_search.ingestion.ingest.ingest_one",
                AsyncMock(return_value=outcome),
            ),
        ):
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            result = await scheduler.poll_once()

        assert (result.entries_added, result.entries_failed) == (1, 1)
        repository.complete_ingestion_run.assert_awaited_once()
        repository.fail_ingestion_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entry_written_during_a_slow_fetch_is_picked_up_next_poll(self, repository):
        """The watermark is the run's start, so a mid-fetch write is newer than ``since``."""
        old = _make_entry("old")
        old["timestamp"] = datetime(2024, 6, 1, tzinfo=UTC)
        source: list[dict] = [old]
        stored: list[str] = []
        repository.upsert_entry = AsyncMock(side_effect=lambda e: stored.append(e["entry_id"]))

        adapter = MagicMock()
        adapter.source_system_name = "test_system"
        adapter.unreadable_entries = 0

        async def _slow_fetch(since=None, until=None, limit=None):  # noqa: ARG001 - the ingestion adapter fetch_entries signature
            snapshot = [e for e in source if since is None or e["timestamp"] > since]
            await asyncio.sleep(0.01)
            late = _make_entry("written-mid-poll")
            late["timestamp"] = datetime.now(UTC)
            if not any(e["entry_id"] == late["entry_id"] for e in source):
                source.append(late)
            await asyncio.sleep(0.01)
            for entry in snapshot:
                yield entry

        adapter.fetch_entries = _slow_fetch

        get_adapter, get_enhancers = _poll_patches(adapter)
        with get_adapter, get_enhancers:
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            first = await scheduler.poll_once()
            second = await scheduler.poll_once()

        assert first.entries_added == 1
        assert second.entries_added == 1
        assert stored == ["old", "written-mid-poll"]

    @pytest.mark.asyncio
    async def test_no_fetch_for_an_entry_without_attachments(self, repository, attachment_fetch):
        """On a store with the copy state, an entry with no pictures fetches nothing."""
        repository.schema_facts = AsyncMock(
            return_value=SchemaFacts(has_v2_fts=True, has_copy_state=True)
        )
        repository.get_copy_rows = AsyncMock(return_value=[])
        adapter = _mock_adapter([_make_entry("e1")])

        get_adapter, get_enhancers = _poll_patches(adapter)
        with (
            get_adapter,
            get_enhancers,
            patch(
                "osprey.services.ariel_search.ingestion.ingest._store",
                AsyncMock(return_value=True),
            ),
        ):
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            result = await scheduler.poll_once()

        assert (result.entries_added, result.entries_failed) == (1, 0)
        repository.get_copy_rows.assert_awaited_once_with("e1")
        assert attachment_fetch is None or attachment_fetch.calls == []

    @pytest.mark.asyncio
    async def test_one_copy_run_and_breaker_span_exactly_one_poll(self, repository):
        """Every entry of a poll shares one CopyRun; five refused connects trip its breaker.

        The tripped breaker then refuses that host for the rest of the poll, and
        the next poll starts with a fresh breaker.
        """
        from osprey.services.ariel_search.ingestion.ingest import EntryIngestOutcome

        host = ("https", "elog.example", 443)
        runs: list = []

        async def _refusing_ingest(entry, adapter, repo, enhancers, config, copy_run):  # noqa: ARG001 - the ingest_one signature
            runs.append(copy_run)
            assert copy_run.breaker.allow(host)
            copy_run.breaker.record(host, transient=True)
            return EntryIngestOutcome()

        adapter = _mock_adapter([_make_entry(f"e{i}") for i in range(5)])

        get_adapter, get_enhancers = _poll_patches(adapter)
        with (
            get_adapter,
            get_enhancers,
            patch("osprey.services.ariel_search.ingestion.ingest.ingest_one", _refusing_ingest),
        ):
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            await scheduler.poll_once()
            first_poll = runs[:]
            await scheduler.poll_once()

        assert len(first_poll) == 5
        assert all(run is first_poll[0] for run in first_poll)
        assert first_poll[0].breaker.is_open(host)
        assert not first_poll[0].breaker.allow(host)
        assert runs[5] is not first_poll[0]
        assert all(run is runs[5] for run in runs[5:])


# --- copy retry step ------------------------------------------------------------

#: A store carrying the attachment copy state: the poll's retry step runs.
_COPY_STORE = SchemaFacts(has_v2_fts=True, has_copy_state=True)


def _lock_factory(held: bool = True, calls: list | None = None):
    """A stand-in for ``try_advisory_lock`` that yields ``held`` and records its calls."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory(conninfo, key, **kwargs):
        if calls is not None:
            calls.append((conninfo, key, kwargs))
        yield held

    return _factory


def _keyset(pairs: list[tuple[datetime, str]]) -> AsyncMock:
    """``get_copy_retry_candidates`` over ``pairs``: newest first below the cursor."""
    ordered = sorted(pairs, reverse=True)

    async def _get(after, limit):
        return [p for p in ordered if after is None or p < after][:limit]

    return AsyncMock(side_effect=_get)


def _pairs(prefix: str, count: int, *, day: int = 1) -> list[tuple[datetime, str]]:
    """``count`` candidates, one per minute of day ``day`` of 2024."""
    return [(datetime(2024, 1, day, 0, i, tzinfo=UTC), f"{prefix}{i:02d}") for i in range(count)]


class TestCopyRetryStep:
    """After its new entries, a poll retries the copy of other stored entries."""

    @pytest.fixture
    def repository(self) -> MagicMock:
        repo = MagicMock()
        repo.pool = MagicMock()
        repo.pool.conninfo = "postgresql://localhost/test"
        repo.schema_facts = AsyncMock(return_value=_COPY_STORE)
        repo.get_copy_retry_candidates = AsyncMock(return_value=[])
        _RunLedger(datetime(2024, 1, 1, tzinfo=UTC)).install(repo)
        return repo

    @staticmethod
    def _patches(adapter, copy_entry, ingest=None):
        from osprey.services.ariel_search.ingestion.ingest import EntryIngestOutcome

        async def _ingest(*_args, **_kwargs):
            return EntryIngestOutcome()

        get_adapter, get_enhancers = _poll_patches(adapter)
        return (
            get_adapter,
            get_enhancers,
            patch("osprey.services.ariel_search.ingestion.ingest.ingest_one", ingest or _ingest),
            patch("osprey.services.ariel_search.attachments.copy.copy_entry", copy_entry),
        )

    async def _poll(self, scheduler, adapter, copy_entry, polls: int = 1, ingest=None):
        a, b, c, d = self._patches(adapter, copy_entry, ingest)
        results = []
        with a, b, c, d:
            for _ in range(polls):
                results.append(await scheduler.poll_once())
        return results

    @pytest.mark.asyncio
    async def test_retry_step_copies_other_entries_with_the_poll_copy_run(self, repository):
        """New entries are passed over; the others are copied with the poll's CopyRun."""
        ts = datetime(2024, 1, 2, tzinfo=UTC)
        repository.get_copy_retry_candidates = AsyncMock(
            return_value=[(ts, "new-1"), (ts, "old-2"), (ts, "old-1")]
        )
        ingest_runs: list = []

        async def _ingest(entry, adapter, repo, enhancers, config, copy_run):  # noqa: ARG001 - the ingest_one signature
            from osprey.services.ariel_search.ingestion.ingest import EntryIngestOutcome

            ingest_runs.append(copy_run)
            return EntryIngestOutcome()

        copy_entry = AsyncMock()
        lock_calls: list = []
        scheduler = IngestionScheduler(
            _make_config(), repository, lock_factory=_lock_factory(True, lock_calls)
        )
        (result,) = await self._poll(
            scheduler, _mock_adapter([_make_entry("new-1")]), copy_entry, ingest=_ingest
        )

        assert result.entries_added == 1
        assert [c.args[1] for c in copy_entry.await_args_list] == ["old-2", "old-1"]
        assert all(c.args[0] is repository for c in copy_entry.await_args_list)
        assert all(c.args[3] is ingest_runs[0] for c in copy_entry.await_args_list)
        assert lock_calls == [("postgresql://localhost/test", "ariel_copy", {})]
        # The new entry is not counted against the step's budget.
        repository.get_copy_retry_candidates.assert_awaited_once_with(None, 21)

    @pytest.mark.asyncio
    async def test_retry_step_visits_at_most_twenty_entries(self, repository):
        repository.get_copy_retry_candidates = _keyset(_pairs("c", 30))
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_lock_factory())
        await self._poll(scheduler, _mock_adapter([]), copy_entry)

        visited = [c.args[1] for c in copy_entry.await_args_list]
        assert visited == [f"c{i:02d}" for i in range(29, 9, -1)]

    @pytest.mark.asyncio
    async def test_retry_step_rotating_cursor_continues_below_and_wraps(self, repository):
        """45 candidates: polls visit 20, 20, the last 5, then the newest 20 again."""
        repository.get_copy_retry_candidates = _keyset(_pairs("c", 45))
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_lock_factory())

        per_poll = []
        for _ in range(4):
            copy_entry.reset_mock()
            await self._poll(scheduler, _mock_adapter([]), copy_entry)
            per_poll.append([c.args[1] for c in copy_entry.await_args_list])

        newest_first = [f"c{i:02d}" for i in range(44, -1, -1)]
        assert per_poll[0] == newest_first[:20]
        assert per_poll[1] == newest_first[20:40]
        assert per_poll[2] == newest_first[40:]
        assert per_poll[3] == newest_first[:20]

    @pytest.mark.asyncio
    async def test_retry_step_cursor_at_the_end_wraps_in_the_same_poll(self, repository):
        """Exactly 20 candidates: the second poll finds nothing below and restarts at the top."""
        repository.get_copy_retry_candidates = _keyset(_pairs("c", 20))
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_lock_factory())
        await self._poll(scheduler, _mock_adapter([]), copy_entry)
        copy_entry.reset_mock()
        await self._poll(scheduler, _mock_adapter([]), copy_entry)
        assert len(copy_entry.await_args_list) == 20

    @pytest.mark.asyncio
    async def test_retry_step_stuck_newest_entries_do_not_starve_an_older_one(self, repository):
        """20 newest entries that never copy plus one older: the older is visited by poll 2."""
        stuck = _pairs("stuck-", 20, day=3)
        older = (datetime(2024, 1, 2, tzinfo=UTC), "older")
        repository.get_copy_retry_candidates = _keyset([*stuck, older])
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_lock_factory())

        await self._poll(scheduler, _mock_adapter([]), copy_entry)
        assert "older" not in [c.args[1] for c in copy_entry.await_args_list]
        await self._poll(scheduler, _mock_adapter([]), copy_entry)
        assert "older" in [c.args[1] for c in copy_entry.await_args_list]

    @pytest.mark.asyncio
    async def test_retry_step_lock_held_elsewhere_skips_the_step(self, repository):
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(
            _make_config(), repository, lock_factory=_lock_factory(held=False)
        )
        with patch("osprey.services.ariel_search.ingestion.scheduler.logger") as log:
            (result,) = await self._poll(scheduler, _mock_adapter([_make_entry()]), copy_entry)

        assert result.entries_added == 1
        copy_entry.assert_not_awaited()
        repository.get_copy_retry_candidates.assert_not_awaited()
        log.info.assert_any_call("copy: running in another process")

    @pytest.mark.asyncio
    async def test_retry_step_skipped_without_copy_state_schema(self, repository):
        """A store without the copy state: the poll completes, no lock, no walk."""
        repository.schema_facts = AsyncMock(return_value=_PLAIN_STORE)
        lock_calls: list = []
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(
            _make_config(), repository, lock_factory=_lock_factory(True, lock_calls)
        )
        (result,) = await self._poll(scheduler, _mock_adapter([_make_entry()]), copy_entry)

        assert result.entries_added == 1
        assert lock_calls == []
        repository.get_copy_retry_candidates.assert_not_awaited()
        copy_entry.assert_not_awaited()
        repository.complete_ingestion_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_step_copy_failure_does_not_stop_the_step_or_fail_the_poll(
        self, repository
    ):
        ts = datetime(2024, 1, 2, tzinfo=UTC)
        repository.get_copy_retry_candidates = AsyncMock(return_value=[(ts, "b"), (ts, "a")])
        copy_entry = AsyncMock(side_effect=[RuntimeError("boom"), None])
        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_lock_factory())
        (result,) = await self._poll(scheduler, _mock_adapter([]), copy_entry)

        assert [c.args[1] for c in copy_entry.await_args_list] == ["b", "a"]
        assert result.entries_failed == 0
        repository.complete_ingestion_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_step_database_failure_is_absorbed(self, repository):
        """A failing candidate query or lock never reaches run_forever's failure cap."""
        repository.get_copy_retry_candidates = AsyncMock(side_effect=RuntimeError("db down"))
        copy_entry = AsyncMock()
        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_lock_factory())
        (result,) = await self._poll(scheduler, _mock_adapter([]), copy_entry)
        assert result.entries_added == 0
        copy_entry.assert_not_awaited()

        def _broken_lock(*_args, **_kwargs):
            raise OSError("connection refused")

        scheduler = IngestionScheduler(_make_config(), repository, lock_factory=_broken_lock)
        await self._poll(scheduler, _mock_adapter([]), copy_entry)
        copy_entry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retry_step_not_run_on_dry_run_or_before_initial_ingest(self, repository):
        copy_entry = AsyncMock()
        lock_calls: list = []
        repository.get_last_successful_run = AsyncMock(return_value=None)
        scheduler = IngestionScheduler(
            _make_config(require_initial=True),
            repository,
            lock_factory=_lock_factory(True, lock_calls),
        )
        await self._poll(scheduler, _mock_adapter([_make_entry()]), copy_entry)
        a, b, c, d = self._patches(_mock_adapter([_make_entry()]), copy_entry)
        with a, b, c, d:
            await scheduler.poll_once(dry_run=True)
        assert lock_calls == []
        copy_entry.assert_not_awaited()


# ---------------------------------------------------------------------------
# Image modules stay out of the poll
# ---------------------------------------------------------------------------


class _ImageModuleCalls:
    """Every touch of the fake picture module, across the instances a poll builds."""

    events: list[str] = []


def _image_module_classes():
    """A recording inline ``text_embedding`` and a misconfigured ``image_caption``."""
    from osprey.services.ariel_search.enhancement.base import (
        BaseEnhancementModule,
        ImageEntryOutcome,
    )

    class _TextEmbedding(BaseEnhancementModule):
        @property
        def name(self) -> str:
            return "text_embedding"

        async def enhance(self, entry, conn) -> None:  # noqa: ARG002 - the enhancement module signature
            _ImageModuleCalls.events.append(f"text_embedding:{entry['entry_id']}")

    class _ImageCaption(BaseEnhancementModule):
        runs_inline = False

        def __init__(self) -> None:
            _ImageModuleCalls.events.append("image_caption:init")

        @property
        def name(self) -> str:
            return "image_caption"

        def configure(self, config) -> None:  # noqa: ARG002 - the enhancement module signature
            _ImageModuleCalls.events.append("image_caption:configure")
            raise ValueError("image_caption is misconfigured")

        async def enhance(self, entry, conn) -> None:  # noqa: ARG002 - the enhancement module signature
            _ImageModuleCalls.events.append("image_caption:enhance")

        async def run_entry(self, entry, repository, *, gate):  # noqa: ARG002 - the enhancement module signature
            _ImageModuleCalls.events.append("image_caption:run_entry")
            return ImageEntryOutcome.done()

    return _TextEmbedding, _ImageCaption


def _register_image_modules(monkeypatch) -> None:
    """Swap the mocked registry's ``text_embedding`` and add ``image_caption``."""
    from osprey.registry import get_registry
    from osprey.registry.base import ArielEnhancementModuleRegistration

    text_cls, image_cls = _image_module_classes()
    registry = get_registry()
    table = dict(registry.get_ariel_enhancement_module.side_effect.__self__)
    for cls, name, order in ((text_cls, "text_embedding", 20), (image_cls, "image_caption", 40)):
        table[name] = (
            cls,
            ArielEnhancementModuleRegistration(
                name=name,
                module_path=__name__,
                class_name=cls.__name__,
                description=name,
                execution_order=order,
            ),
        )
    ordered = sorted(table, key=lambda n: table[n][1].execution_order)
    monkeypatch.setattr(registry.list_ariel_enhancement_modules, "return_value", ordered)
    monkeypatch.setattr(registry.get_ariel_enhancement_module, "side_effect", table.get)


class TestPollSkipsImageModules:
    """A poll builds inline modules only, through the real factory."""

    @pytest.fixture
    def repository(self) -> MagicMock:
        repo = MagicMock()
        repo.pool = MagicMock()
        repo.pool.connection = MagicMock(return_value=AsyncMock())
        repo.start_ingestion_run = AsyncMock(return_value=1)
        repo.complete_ingestion_run = AsyncMock()
        repo.fail_ingestion_run = AsyncMock()
        # A previous run exists, so the poll ingests instead of asking for an initial ingest.
        repo.get_last_successful_run = AsyncMock(return_value=datetime(2024, 1, 1, tzinfo=UTC))
        repo.upsert_entry = AsyncMock()
        repo.mark_enhancement_complete = AsyncMock()
        repo.mark_enhancement_failed = AsyncMock()
        repo.schema_facts = AsyncMock(return_value=_PLAIN_STORE)
        repo.get_copy_retry_candidates = AsyncMock(return_value=[])
        return repo

    @pytest.fixture(autouse=True)
    def _clear_events(self):
        _ImageModuleCalls.events = []
        yield
        _ImageModuleCalls.events = []

    @pytest.mark.asyncio
    async def test_image_module_misconfigured_poll_stores_and_runs_text_embedding(
        self, monkeypatch, repository
    ) -> None:
        _register_image_modules(monkeypatch)
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "ingestion": {"adapter": "generic_json", "source_url": "https://x/logbook"},
                "enhancement_modules": {
                    "text_embedding": {"enabled": True},
                    "image_caption": {"enabled": True, "provider": "nowhere"},
                },
            }
        )
        adapter = _mock_adapter([_make_entry("e1"), _make_entry("e2")])

        with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
            scheduler = IngestionScheduler(config=config, repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 2
        assert result.entries_failed == 0
        assert repository.upsert_entry.await_count == 2
        assert _ImageModuleCalls.events == ["text_embedding:e1", "text_embedding:e2"]
        marked = [c.args[1] for c in repository.mark_enhancement_complete.await_args_list]
        assert marked == ["text_embedding", "text_embedding"]
        repository.mark_enhancement_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_module_in_the_enhancer_list_gets_zero_calls(self, repository) -> None:
        text_cls, image_cls = _image_module_classes()
        image = image_cls.__new__(image_cls)  # no __init__: count only poll-time calls
        adapter = _mock_adapter([_make_entry("e1")])

        with (
            _poll_patches(adapter, [text_cls(), image])[0],
            _poll_patches(adapter, [text_cls(), image])[1],
        ):
            scheduler = IngestionScheduler(config=_make_config(), repository=repository)
            result = await scheduler.poll_once()

        assert result.entries_added == 1
        assert _ImageModuleCalls.events == ["text_embedding:e1"]
        names = [c.args[1] for c in repository.mark_enhancement_complete.await_args_list]
        assert "image_caption" not in names
        repository.mark_enhancement_failed.assert_not_awaited()
