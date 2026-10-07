"""ARIEL ingestion scheduler for live polling.

This module provides the IngestionScheduler class that periodically fetches
new logbook entries from a live API, stores them, and runs the enhancement
pipeline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.attachments.copy import CopyRun
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.migrations import LockFactory
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = get_logger("ariel.scheduler")

#: Most entries besides its own new ones that one poll's copy retry step visits.
COPY_RETRY_ENTRIES = 20

#: Advisory-lock name every copier of already-stored entries holds (poll retry, backfill).
COPY_LOCK_KEY = "ariel_copy"


class StopReason(StrEnum):
    """Why the scheduler's poll loop returned.

    Attributes:
        SIGNAL: The loop was asked to stop via ``stop()``.
        FAILURE_CAP: The loop gave up after ``max_consecutive_failures``
            failed poll cycles in a row.
    """

    SIGNAL = "signal"
    FAILURE_CAP = "failure_cap"


@dataclass
class IngestionPollResult:
    """Result of a single poll cycle.

    Attributes:
        entries_added: Number of new entries stored
        entries_updated: Number of existing entries updated
        entries_failed: Number of entries that could not be read, stored or enhanced
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
        *,
        lock_factory: LockFactory | None = None,
    ) -> None:
        """Build a scheduler.

        Args:
            config: ARIEL configuration.
            repository: Database repository for entry storage and run tracking.
            lock_factory: Opens the copy advisory lock as
                ``lock_factory(conninfo, key)``; defaults to ``try_advisory_lock``.
        """
        self.config = config
        self.repository = repository
        self._stop_event = asyncio.Event()
        self._consecutive_failures = 0
        self._lock_factory = lock_factory
        #: ``(timestamp, entry_id)`` of the last entry the retry step visited;
        #: the next step continues below it, None starts from the newest.
        self._copy_cursor: tuple[datetime, str] | None = None

    async def run_forever(self) -> StopReason:
        """Run the poll loop until stopped -- this blocks; it is not a handle.

        Each iteration calls poll_once(), then sleeps for what is left of the
        configured interval (with backoff on failures), measured from the start
        of that poll, so consecutive polls start one interval apart. Returns when stop() is called or
        after ``max_consecutive_failures`` failed cycles in a row.

        Returns:
            ``StopReason.SIGNAL`` when stop() ended the loop,
            ``StopReason.FAILURE_CAP`` when the consecutive-failure cap did.
        """
        logger.info("Ingestion scheduler started")

        stop_reason = StopReason.SIGNAL

        while not self._stop_event.is_set():
            poll_start = time.monotonic()
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
                    stop_reason = StopReason.FAILURE_CAP
                    break

            # The interval runs from poll start to poll start, so the catch-up
            # attached to a poll never delays the next one; an overrunning poll
            # re-polls at once. A failed poll's backoff is part of the interval.
            interval = max(0.0, self._get_current_interval() - (time.monotonic() - poll_start))
            if self._stop_event.is_set():
                break
            if interval <= 0:
                await asyncio.sleep(0)  # let stop() and other tasks run between polls
                continue
            logger.debug("Sleeping %.0fs until next poll", interval)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                break  # stop_event was set
            except TimeoutError:
                continue  # Timeout means it's time to poll again

        logger.info("Ingestion scheduler stopped")
        return stop_reason

    async def poll_once(
        self, dry_run: bool = False, limit: int | None = None
    ) -> IngestionPollResult:
        """Execute a single poll cycle.

        1. Determine since-timestamp from last successful run
        2. Fetch entries via adapter
        3. Store each entry through ``ingest_one`` (text, pictures, enhancers)
        4. Record the ingestion run
        5. Retry the copy of up to ``COPY_RETRY_ENTRIES`` other stored entries

        Every entry of the poll shares one ``CopyRun``, so the per-host
        breaker spans exactly this poll. The run is recorded failed, and the
        watermark stays put, only when entries were fetched and none of them
        was stored; enhancer and attachment-recording failures leave the text
        stored and count in ``entries_failed`` of a successful run. Such a
        failed poll still returns normally, so it does not count toward the
        consecutive-failure cap; an adapter error is recorded failed and raised.
        The retry step shares the poll's ``CopyRun`` and never raises; see
        :meth:`_copy_retry_step`.

        Args:
            dry_run: If True, parse entries without storing
            limit: Most entries to fetch, or None for all

        Returns:
            IngestionPollResult with counts and timing
        """
        from osprey.services.ariel_search.attachments.copy import (
            COPY_CONCURRENCY,
            CopyRun,
            HostBreaker,
        )
        from osprey.services.ariel_search.attachments.fetch import origins_for
        from osprey.services.ariel_search.enhancement import create_enhancers_from_config
        from osprey.services.ariel_search.ingestion import get_adapter
        from osprey.services.ariel_search.ingestion.ingest import ingest_one

        start_time = time.monotonic()

        adapter = get_adapter(self.config)
        # The default stage is inline: a catch-up module never runs in a poll.
        enhancers = create_enhancers_from_config(self.config)
        source_system = adapter.source_system_name

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
            async for _entry in adapter.fetch_entries(since=since, limit=limit):
                count += 1
            return IngestionPollResult(
                entries_added=count,
                entries_updated=0,
                entries_failed=adapter.unreadable_entries,
                duration_seconds=time.monotonic() - start_time,
                since=since,
            )

        run_id = await self.repository.start_ingestion_run(source_system)

        entries_fetched = 0
        entries_added = 0
        entries_failed = 0
        new_ids: set[str] = set()

        try:
            copy_run = CopyRun(
                adapter,
                origins_for(adapter, self.config),
                asyncio.Semaphore(COPY_CONCURRENCY),
                HostBreaker(),
            )
            async for entry in adapter.fetch_entries(since=since, limit=limit):
                entries_fetched += 1
                new_ids.add(str(entry["entry_id"]))
                try:
                    outcome = await ingest_one(
                        entry, adapter, self.repository, enhancers, self.config, copy_run
                    )
                except Exception:
                    entries_failed += 1
                    logger.exception("Failed to process entry")
                    continue
                entries_added += 1
                entries_failed += outcome.enhancer_failed
                if not outcome.attachments_recorded:
                    entries_failed += 1
            entries_failed += adapter.unreadable_entries
        except Exception as e:
            await self.repository.fail_ingestion_run(run_id, str(e))
            raise

        if entries_fetched > 0 and entries_added == 0:
            message = f"none of the {entries_fetched} fetched entries could be stored"
            logger.error("Poll of '%s' failed: %s", source_system, message)
            await self.repository.fail_ingestion_run(run_id, message)
        else:
            await self.repository.complete_ingestion_run(
                run_id,
                entries_added=entries_added,
                entries_updated=0,
                entries_failed=entries_failed,
            )

        await self._copy_retry_step(copy_run, new_ids)

        return IngestionPollResult(
            entries_added=entries_added,
            entries_updated=0,
            entries_failed=entries_failed,
            duration_seconds=time.monotonic() - start_time,
            since=since,
        )

    async def _copy_retry_step(self, copy_run: CopyRun, new_ids: set[str]) -> None:
        """Run ``copy_entry`` over stored entries that still hold copy work.

        Visits at most ``COPY_RETRY_ENTRIES`` entries outside *new_ids* that hold a
        pending row or a copied row without rendition, newest first from a
        rotating cursor: each step continues below the last entry the previous
        step visited and wraps to the newest once the walk is exhausted, so
        entries stuck on a down host cannot starve older ones. The step holds
        the ``ariel_copy`` advisory lock; held elsewhere, it is skipped. On a
        store without the copy state it is skipped. Every failure is logged and
        absorbed, so the step never fails a poll.

        Args:
            copy_run: The poll's shared fetch state.
            new_ids: Entry ids this poll stored, already copied by ``ingest_one``.
        """
        from osprey.services.ariel_search.attachments.copy import copy_entry

        try:
            if not (await self.repository.schema_facts()).has_copy_state:
                return
            lock = self._copy_lock_factory()
            async with lock(self.repository.pool.conninfo, COPY_LOCK_KEY) as held:
                if not held:
                    logger.info("copy: running in another process")
                    return
                for entry_id in await self._next_copy_candidates(new_ids):
                    try:
                        await copy_entry(self.repository, entry_id, self.config, copy_run)
                    except Exception as exc:
                        logger.warning(
                            "%s: picture copy retry failed (%s); a later poll retries it",
                            entry_id,
                            exc,
                        )
        except Exception as exc:
            logger.warning("copy retry step skipped: %s", exc)

    def _copy_lock_factory(self) -> LockFactory:
        """Return the injected lock factory, else ``try_advisory_lock``."""
        if self._lock_factory is not None:
            return self._lock_factory
        from osprey.services.ariel_search.database.connection import try_advisory_lock

        return try_advisory_lock

    async def _next_copy_candidates(self, new_ids: set[str]) -> list[str]:
        """Return the entry ids the retry step visits and advance the cursor.

        Args:
            new_ids: Entry ids to pass over without counting them.

        Returns:
            At most ``COPY_RETRY_ENTRIES`` entry ids, newest first.
        """
        limit = COPY_RETRY_ENTRIES + len(new_ids)
        rows = await self.repository.get_copy_retry_candidates(self._copy_cursor, limit)
        if not rows and self._copy_cursor is not None:
            self._copy_cursor = None
            rows = await self.repository.get_copy_retry_candidates(None, limit)

        picked: list[str] = []
        consumed = 0
        for _timestamp, entry_id in rows:
            consumed += 1
            if entry_id in new_ids:
                continue
            picked.append(entry_id)
            if len(picked) == COPY_RETRY_ENTRIES:
                break

        exhausted = consumed == len(rows) and len(rows) < limit
        self._copy_cursor = None if exhausted or not rows else rows[consumed - 1]
        return picked

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
