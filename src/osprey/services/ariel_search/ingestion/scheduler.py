"""ARIEL ingestion scheduler for live polling.

This module provides the IngestionScheduler class that periodically fetches
new logbook entries from a live API, stores them, and runs the enhancement
pipeline.

See 04_OSPREY_INTEGRATION.md for specification context.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = get_logger("ariel.scheduler")


@dataclass
class IngestionPollResult:
    """Result of a single poll cycle.

    Attributes:
        entries_added: Number of new entries stored
        entries_updated: Number of existing entries updated
        entries_failed: Number of entries that failed enhancement
        duration_seconds: Wall-clock time for the poll cycle
        since: The since-timestamp used for this poll (None = full ingest)
    """

    entries_added: int
    entries_updated: int
    entries_failed: int
    duration_seconds: float
    since: datetime | None


class IngestionScheduler:
    """Scheduler that periodically polls a source for new logbook entries.

    Uses the same adapter and enhancement pipeline as `osprey ariel ingest`,
    but runs continuously with configurable poll intervals and backoff.

    Attributes:
        config: ARIEL configuration
        repository: Database repository for entry storage and run tracking
    """

    def __init__(
        self,
        config: ARIELConfig,
        repository: ARIELRepository,
    ) -> None:
        self.config = config
        self.repository = repository
        self._stop_event = asyncio.Event()
        self._consecutive_failures = 0

    async def start(self) -> None:
        """Run the poll loop until stopped.

        Each iteration calls poll_once(), then sleeps for the configured
        interval (with backoff on failures). Exits when stop() is called.
        """
        logger.info("Ingestion scheduler started")

        while not self._stop_event.is_set():
            try:
                result = await self.poll_once()
                self._consecutive_failures = 0
                total = result.entries_added + result.entries_updated
                logger.info(
                    f"Poll complete: {total} entries "
                    f"({result.entries_added} added, {result.entries_updated} updated, "
                    f"{result.entries_failed} failed) in {result.duration_seconds:.1f}s"
                )
            except Exception:
                self._consecutive_failures += 1
                logger.exception(
                    "Poll failed (consecutive failures: %d)",
                    self._consecutive_failures,
                )

                watch_config = self.config.ingestion.watch if self.config.ingestion else None
                max_failures = watch_config.max_consecutive_failures if watch_config else 10
                if self._consecutive_failures >= max_failures:
                    logger.error(
                        f"Stopping scheduler after {self._consecutive_failures} consecutive failures"
                    )
                    break

            interval = self._get_current_interval()
            logger.debug("Sleeping %.0fs until next poll", interval)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                break  # stop_event was set
            except TimeoutError:
                continue  # Timeout means it's time to poll again

        logger.info("Ingestion scheduler stopped")

    async def poll_once(self, dry_run: bool = False) -> IngestionPollResult:
        """Execute a single poll cycle.

        1. Determine since-timestamp from last successful run
        2. Fetch entries via adapter
        3. Store entries and run enhancements
        4. Record the ingestion run

        Args:
            dry_run: If True, parse entries without storing

        Returns:
            IngestionPollResult with counts and timing
        """
        from osprey.services.ariel_search.enhancement import create_enhancers_from_config
        from osprey.services.ariel_search.ingestion import get_adapter

        start_time = time.monotonic()

        adapter = get_adapter(self.config)
        enhancers = create_enhancers_from_config(self.config)
        source_system = adapter.source_system_name

        # Determine since-timestamp from last successful run
        last_run_time = await self.repository.get_last_successful_run(source_system)

        watch_config = self.config.ingestion.watch if self.config.ingestion else None
        require_initial = watch_config.require_initial_ingest if watch_config else True

        if last_run_time is None and require_initial:
            logger.info(
                "No previous ingestion found for '%s' and require_initial_ingest=True. "
                "Run 'osprey ariel ingest' first.",
                source_system,
            )
            return IngestionPollResult(
                entries_added=0,
                entries_updated=0,
                entries_failed=0,
                duration_seconds=time.monotonic() - start_time,
                since=None,
            )

        since = last_run_time

        if dry_run:
            count = 0
            async for _entry in adapter.fetch_entries(since=since):
                count += 1
            return IngestionPollResult(
                entries_added=count,
                entries_updated=0,
                entries_failed=0,
                duration_seconds=time.monotonic() - start_time,
                since=since,
            )

        # Start tracking the ingestion run
        run_id = await self.repository.start_ingestion_run(source_system)

        entries_added = 0
        entries_failed = 0

        try:
            async for entry in adapter.fetch_entries(since=since):
                try:
                    await self.repository.upsert_entry(entry)
                    entries_added += 1

                    # Run enhancement pipeline
                    for enhancer in enhancers:
                        try:
                            async with self.repository.pool.connection() as conn:
                                await enhancer.enhance(entry, conn)
                            await self.repository.mark_enhancement_complete(
                                entry["entry_id"],
                                enhancer.name,
                            )
                        except Exception as e:
                            await self.repository.mark_enhancement_failed(
                                entry["entry_id"],
                                enhancer.name,
                                str(e),
                            )
                            entries_failed += 1
                except Exception:
                    entries_failed += 1
                    logger.exception("Failed to process entry")

            await self.repository.complete_ingestion_run(
                run_id,
                entries_added=entries_added,
                entries_updated=0,
                entries_failed=entries_failed,
            )
        except Exception as e:
            await self.repository.fail_ingestion_run(run_id, str(e))
            raise

        return IngestionPollResult(
            entries_added=entries_added,
            entries_updated=0,
            entries_failed=entries_failed,
            duration_seconds=time.monotonic() - start_time,
            since=since,
        )

    async def stop(self) -> None:
        """Signal the scheduler to stop after the current poll cycle."""
        self._stop_event.set()

    def _get_current_interval(self) -> float:
        """Calculate the current poll interval with backoff.

        Returns base interval on success, increasing exponentially
        on consecutive failures up to max_interval_seconds.

        Returns:
            Poll interval in seconds
        """
        base_interval = float(
            self.config.ingestion.poll_interval_seconds if self.config.ingestion else 3600
        )

        if self._consecutive_failures == 0:
            return base_interval

        watch_config = self.config.ingestion.watch if self.config.ingestion else None
        multiplier = watch_config.backoff_multiplier if watch_config else 2.0
        max_interval = float(watch_config.max_interval_seconds if watch_config else 3600)

        backoff_interval = base_interval * (multiplier**self._consecutive_failures)
        return min(backoff_interval, max_interval)
