"""Ingest-side pipeline tests for :mod:`osprey.services.ariel_search.cli_operations`.

Companion to ``test_cli_operations.py``, which covers the pure-logic and
error-translation contracts. This module drives the *storing* paths — the ones
that open the ARIEL service, walk an adapter stream, run enhancers and record an
ingestion run:

* ``run_sync`` — migrate, incremental poll, enhance cleanup, and the deep-copied
  ``require_initial_ingest=False`` override that lets sync do a full ingest on a
  fresh database without editing the caller's config dict.
* ``run_ingest`` — the storing branch: upsert, per-enhancer success/failure
  accounting, run bookkeeping, and the every-100-entries progress lines.
* ``run_watch`` — the source/adapter/interval config overrides, the single-poll
  result projection, and daemon mode with its SIGINT/SIGTERM handlers.
* ``run_enhance`` — entry selection (force / single module / dedupe across all
  enhancers) and the per-entry enhancement loop.
* ``seed_logbook_entries`` — the progress branches.
* ``IngestionScheduler.poll_once`` — the per-entry exception handler, reached
  through ``run_watch(once=True)`` with a repository that rejects one entry.

Everything crossing a process boundary is faked. Following the package rule that
a patch must name the module that *owns* the symbol (``cli_operations`` imports
each of these lazily, inside the function), the seams are patched at their
source: ``create_ariel_service`` on the ``ariel_search`` package,
``create_connection_pool`` on ``database.connection``, ``run_migrations`` on
``database.migrations``, ``IngestionScheduler`` on ``ingestion.scheduler``,
``get_adapter`` on ``ingestion`` and ``create_enhancers_from_config`` on
``enhancement``. The service stub carries a ``_FakePool`` from the shared
``mock_repository`` fixture, so ``async with service.pool.connection()`` works
and every statement lands in ``repo.pool.calls``.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from osprey.services.ariel_search import cli_operations as ops
from tests.services.ariel_search._cli_ops_doubles import (
    _Adapter,
    _Enhancer,
    _forbid_service,
    _patch_adapter,
    _patch_enhancers,
    _patch_migrations,
    _patch_pool,
    _patch_service,
    _StubService,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# A minimal config dict accepted by ARIELConfig.from_dict.
_DB: dict[str, Any] = {"database": {"uri": "postgresql://localhost/test"}}
_SOURCE = "file:///entries.json"


def _config(**ingestion: Any) -> dict[str, Any]:
    """Fresh config dict; ``ingestion`` kwargs become the ``ingestion`` block."""
    cfg: dict[str, Any] = {"database": dict(_DB["database"])}
    if ingestion:
        cfg["ingestion"] = dict(ingestion)
    return cfg


@pytest.fixture(autouse=True)
def _no_ariel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ingestion config assertions independent of the developer's shell.

    ``IngestionConfig``/``WriteConfig`` fall back to ``ARIEL_SOCKS_PROXY`` /
    ``ARIEL_WRITE_USER`` / ``ARIEL_WRITE_PASSWORD`` when the dict omits them, so
    a host that exports any of them would otherwise leak into the config objects
    these tests build.
    """
    for name in ("ARIEL_SOCKS_PROXY", "ARIEL_WRITE_USER", "ARIEL_WRITE_PASSWORD"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Entry and result factories -- the shared doubles live in ``_cli_ops_doubles``
# ---------------------------------------------------------------------------


def _entries(count: int, prefix: str = "E") -> list[dict[str, Any]]:
    return [{"entry_id": f"{prefix}{i}", "raw_text": f"entry {i}"} for i in range(count)]


def _async_return(value: Any) -> AsyncMock:
    """An awaitable repository stub returning *value* on every call."""
    return AsyncMock(return_value=value)


def _poll_result(
    added: int = 0,
    failed: int = 0,
    duration: float = 0.5,
    since: Any = None,
    updated: int = 0,
):
    from osprey.services.ariel_search.ingestion.scheduler import IngestionPollResult

    return IngestionPollResult(
        entries_added=added,
        entries_updated=updated,
        entries_failed=failed,
        duration_seconds=duration,
        since=since,
    )


# ---------------------------------------------------------------------------
# Scheduler double -- the only seam unique to the pipeline operations
# ---------------------------------------------------------------------------


class _StubScheduler:
    """Recording stand-in for ``IngestionScheduler`` (built by the patch below)."""

    poll_result: Any = None
    made: list[_StubScheduler] = []

    def __init__(self, config: Any, repository: Any) -> None:
        self.config = config
        self.repository = repository
        self.poll_calls: list[dict[str, Any]] = []
        self.run_forever_calls = 0
        self.stop_calls = 0

    async def poll_once(self, dry_run: bool = False, limit: int | None = None):
        self.poll_calls.append({"dry_run": dry_run, "limit": limit})
        return self.poll_result

    async def run_forever(self) -> None:
        self.run_forever_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


def _patch_scheduler(monkeypatch: pytest.MonkeyPatch, poll_result: Any = None) -> list[Any]:
    """Route ``IngestionScheduler`` to a recording stub; returns its instances."""
    import osprey.services.ariel_search.ingestion.scheduler as sched_mod

    made: list[Any] = []

    class _Recorded(_StubScheduler):
        def __init__(self, config, repository):
            super().__init__(config, repository)
            self.poll_result = poll_result if poll_result is not None else _poll_result()
            made.append(self)

    monkeypatch.setattr(sched_mod, "IngestionScheduler", _Recorded)
    return made


# ---------------------------------------------------------------------------
# run_sync -- migrate, incremental poll, enhance cleanup
# ---------------------------------------------------------------------------


class TestRunSync:
    async def test_full_cycle_reports_every_stage(self, monkeypatch, mock_repository, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=["001_initial", "002_embeddings"])
        schedulers = _patch_scheduler(monkeypatch, _poll_result(added=3, failed=1, since=None))
        service = _StubService(mock_repository)
        _patch_service(monkeypatch, service)
        _patch_enhancers(monkeypatch, [_Enhancer("text_embedding")])
        mock_repository.get_incomplete_entries = _async_return([{"entry_id": "OLD1"}])

        config_dict = _config(adapter="generic_json", source_url=_SOURCE)
        messages: list[str] = []

        out = await ops.run_sync(config_dict, limit=5, progress=messages.append)

        assert out.migrations_applied == 2
        assert out.entries_ingested == 3
        assert out.entries_failed == 1
        assert out.was_initial_ingest is True
        # Step 3 catches up the one entry left incomplete by an earlier run.
        assert out.entries_enhanced == 1

        assert fake_pool.closed is True
        assert schedulers[0].poll_calls == [{"dry_run": False, "limit": 5}]
        assert messages[:2] == ["Running migrations...", "  2 migrations applied"]
        assert f"Polling for new entries (source: {_SOURCE})..." in messages
        assert "  3 entries ingested" in messages
        assert "  (initial full ingest)" in messages

    async def test_sync_overrides_require_initial_ingest_without_touching_caller_config(
        self, monkeypatch, mock_repository, fake_pool
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        schedulers = _patch_scheduler(monkeypatch, _poll_result(added=0))
        _patch_service(monkeypatch, _StubService(mock_repository))
        _patch_enhancers(monkeypatch, [])

        config_dict = _config(adapter="generic_json", source_url=_SOURCE)
        config_dict["ingestion"]["watch"] = {
            "require_initial_ingest": True,
            "max_consecutive_failures": 3,
        }

        await ops.run_sync(config_dict)

        watch = schedulers[0].config.ingestion.watch
        # Only the one key is overridden; the rest of the watch block survives.
        assert watch.require_initial_ingest is False
        assert watch.max_consecutive_failures == 3
        # The caller's dict is deep-copied, so its own watch block is untouched.
        assert config_dict["ingestion"]["watch"] == {
            "require_initial_ingest": True,
            "max_consecutive_failures": 3,
        }

    async def test_missing_ingestion_block_gets_the_override_too(
        self, monkeypatch, mock_repository, fake_pool
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        schedulers = _patch_scheduler(monkeypatch, _poll_result(added=0))
        _patch_service(monkeypatch, _StubService(mock_repository))
        _patch_enhancers(monkeypatch, [])

        config_dict = dict(_DB)

        await ops.run_sync(config_dict)

        assert schedulers[0].config.ingestion.watch.require_initial_ingest is False
        # The synthesized block never leaks back to the caller.
        assert "ingestion" not in config_dict

    async def test_up_to_date_and_incremental_poll_report_no_initial_ingest(
        self, monkeypatch, mock_repository, fake_pool
    ):
        since = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_scheduler(monkeypatch, _poll_result(added=2, since=since))
        _patch_service(monkeypatch, _StubService(mock_repository))
        _patch_enhancers(monkeypatch, [])

        messages: list[str] = []
        out = await ops.run_sync(_config(source_url=_SOURCE), progress=messages.append)

        assert out.migrations_applied == 0
        assert out.was_initial_ingest is False
        assert out.entries_enhanced == 0
        assert "  Already up to date" in messages
        assert "  (initial full ingest)" not in messages
        # Step 3 short-circuits when nothing is enabled.
        assert "No enhancement modules enabled or selected" in messages

    async def test_unset_source_url_reports_unknown(self, monkeypatch, mock_repository, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=[])
        _patch_scheduler(monkeypatch, _poll_result())
        _patch_service(monkeypatch, _StubService(mock_repository))
        _patch_enhancers(monkeypatch, [])

        messages: list[str] = []
        await ops.run_sync(_config(adapter="generic_json"), progress=messages.append)

        assert "Polling for new entries (source: unknown)..." in messages

    async def test_runs_silently_without_a_progress_callback(
        self, monkeypatch, mock_repository, fake_pool
    ):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, applied=["001_initial"])
        _patch_scheduler(monkeypatch, _poll_result(added=1, since=None))
        _patch_service(monkeypatch, _StubService(mock_repository))
        _patch_enhancers(monkeypatch, [])

        out = await ops.run_sync(_config(source_url=_SOURCE))

        assert out.migrations_applied == 1
        assert out.entries_ingested == 1

    async def test_migration_failure_closes_the_pool_and_propagates(self, monkeypatch, fake_pool):
        _patch_pool(monkeypatch, fake_pool)
        _patch_migrations(monkeypatch, error=RuntimeError("migration blew up"))
        _forbid_service(monkeypatch)

        with pytest.raises(RuntimeError, match="migration blew up"):
            await ops.run_sync(_config(source_url=_SOURCE))

        assert fake_pool.closed is True


# ---------------------------------------------------------------------------
# run_ingest -- the storing branch
# ---------------------------------------------------------------------------


class TestRunIngestStoring:
    async def test_stores_entries_and_records_the_run(self, monkeypatch, mock_repository):
        adapter = _Adapter(_entries(2), source_system_name="ALS eLog")
        enhancer = _Enhancer("text_embedding")
        _patch_adapter(monkeypatch, adapter)
        _patch_enhancers(monkeypatch, [enhancer])
        service = _StubService(mock_repository)
        _patch_service(monkeypatch, service)

        config_dict = dict(_DB)
        out = await ops.run_ingest(
            config_dict,
            source=_SOURCE,
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=False,
        )

        assert (out.count, out.enhanced_count, out.failed_count) == (2, 2, 0)
        assert out.dry_run is False
        assert out.enhancer_names == ["text_embedding"]
        assert mock_repository.upsert_entry.await_count == 2
        mock_repository.start_ingestion_run.assert_awaited_once_with("ALS eLog")
        mock_repository.complete_ingestion_run.assert_awaited_once_with(
            1, entries_added=2, entries_updated=0, entries_failed=0
        )
        mock_repository.fail_ingestion_run.assert_not_awaited()
        # Every enhancer runs against the connection opened from the service pool.
        assert enhancer.seen == ["E0", "E1"]
        assert set(enhancer.conns) == {service.pool.conn}
        assert service.exits == service.enters == 1

    async def test_source_and_adapter_are_written_into_the_caller_config(
        self, monkeypatch, mock_repository
    ):
        _patch_adapter(monkeypatch, _Adapter())
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository))

        config_dict = dict(_DB)
        await ops.run_ingest(
            config_dict,
            source=_SOURCE,
            adapter="als_logbook",
            since=None,
            limit=None,
            dry_run=False,
        )

        # Unlike run_sync, run_ingest edits the dict it is handed.
        assert config_dict["ingestion"] == {"source_url": _SOURCE, "adapter": "als_logbook"}

    async def test_since_and_limit_reach_the_adapter(self, monkeypatch, mock_repository):
        since = datetime(2026, 5, 5, tzinfo=UTC)
        adapter = _Adapter(_entries(1))
        _patch_adapter(monkeypatch, adapter)
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository))

        await ops.run_ingest(
            dict(_DB),
            source=_SOURCE,
            adapter="generic_json",
            since=since,
            limit=7,
            dry_run=False,
        )

        assert adapter.fetch_calls == [{"since": since, "limit": 7}]

    async def test_enhancer_failure_is_recorded_per_entry_and_does_not_abort(
        self, monkeypatch, mock_repository
    ):
        _patch_adapter(monkeypatch, _Adapter(_entries(3)))
        _patch_enhancers(monkeypatch, [_Enhancer("text_embedding", fails_on={"E1"})])
        _patch_service(monkeypatch, _StubService(mock_repository))

        out = await ops.run_ingest(
            dict(_DB),
            source=_SOURCE,
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=False,
        )

        # All three entries still land; only the one enhancement is counted failed.
        assert (out.count, out.enhanced_count, out.failed_count) == (3, 2, 1)
        mock_repository.mark_enhancement_failed.assert_awaited_once_with(
            "E1", "text_embedding", "enhance failed on E1"
        )
        mock_repository.complete_ingestion_run.assert_awaited_once_with(
            1, entries_added=3, entries_updated=0, entries_failed=1
        )

    async def test_counts_are_per_enhancer_not_per_entry(self, monkeypatch, mock_repository):
        good = _Enhancer("text_embedding")
        bad = _Enhancer("semantic_processor", fails_on={"E0", "E1"})
        _patch_adapter(monkeypatch, _Adapter(_entries(2)))
        _patch_enhancers(monkeypatch, [good, bad])
        _patch_service(monkeypatch, _StubService(mock_repository))

        out = await ops.run_ingest(
            dict(_DB),
            source=_SOURCE,
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=False,
        )

        # 2 entries x 2 enhancers = 4 attempts: 2 complete, 2 failed.
        assert (out.count, out.enhanced_count, out.failed_count) == (2, 2, 2)
        assert out.enhancer_names == ["text_embedding", "semantic_processor"]

    async def test_upsert_failure_marks_the_run_failed_and_reraises(
        self, monkeypatch, mock_repository
    ):
        _patch_adapter(monkeypatch, _Adapter(_entries(2)))
        _patch_enhancers(monkeypatch, [])
        service = _StubService(mock_repository)
        _patch_service(monkeypatch, service)
        mock_repository.upsert_entry.side_effect = RuntimeError("db is gone")

        with pytest.raises(RuntimeError, match="db is gone"):
            await ops.run_ingest(
                dict(_DB),
                source=_SOURCE,
                adapter="generic_json",
                since=None,
                limit=None,
                dry_run=False,
            )

        mock_repository.fail_ingestion_run.assert_awaited_once_with(1, "db is gone")
        mock_repository.complete_ingestion_run.assert_not_awaited()
        assert service.exits == 1

    async def test_adapter_stream_failure_marks_the_run_failed(self, monkeypatch, mock_repository):
        _patch_adapter(monkeypatch, _Adapter(_entries(3), raise_at=2))
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository))

        with pytest.raises(RuntimeError, match="adapter stream failed"):
            await ops.run_ingest(
                dict(_DB),
                source=_SOURCE,
                adapter="generic_json",
                since=None,
                limit=None,
                dry_run=False,
            )

        # The two entries that arrived before the stream died are already stored;
        # the run itself is recorded as failed.
        assert mock_repository.upsert_entry.await_count == 2
        mock_repository.fail_ingestion_run.assert_awaited_once_with(1, "adapter stream failed")

    async def test_progress_every_hundred_entries_names_enhancement(
        self, monkeypatch, mock_repository
    ):
        _patch_adapter(monkeypatch, _Adapter(_entries(100)))
        _patch_enhancers(monkeypatch, [_Enhancer("text_embedding")])
        _patch_service(monkeypatch, _StubService(mock_repository))

        messages: list[str] = []
        out = await ops.run_ingest(
            dict(_DB),
            source=_SOURCE,
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=False,
            progress=messages.append,
        )

        assert out.count == 100
        assert "  Ingested and enhanced 100 entries..." in messages
        assert not any("Ingested 100 entries" in m for m in messages)

    async def test_progress_every_hundred_entries_without_enhancers(
        self, monkeypatch, mock_repository
    ):
        _patch_adapter(monkeypatch, _Adapter(_entries(100)))
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository))

        messages: list[str] = []
        await ops.run_ingest(
            dict(_DB),
            source=_SOURCE,
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=False,
            progress=messages.append,
        )

        assert "  Ingested 100 entries..." in messages

    async def test_dry_run_reports_progress_every_hundred_entries(self, monkeypatch):
        _patch_adapter(monkeypatch, _Adapter(_entries(100)))
        _patch_enhancers(monkeypatch, [])
        _forbid_service(monkeypatch)

        messages: list[str] = []
        out = await ops.run_ingest(
            dict(_DB),
            source=_SOURCE,
            adapter="generic_json",
            since=None,
            limit=None,
            dry_run=True,
            progress=messages.append,
        )

        assert out.count == 100
        assert "  Parsed 100 entries..." in messages


# ---------------------------------------------------------------------------
# run_watch -- config overrides and the single-poll projection
# ---------------------------------------------------------------------------


class TestRunWatchOnce:
    async def test_source_and_adapter_overrides_create_the_ingestion_block(
        self, monkeypatch, mock_repository
    ):
        since = datetime(2026, 4, 1, 9, 30, tzinfo=UTC)
        schedulers = _patch_scheduler(
            monkeypatch, _poll_result(added=4, failed=2, duration=1.25, since=since)
        )
        _patch_service(monkeypatch, _StubService(mock_repository))

        config_dict = dict(_DB)
        messages: list[str] = []
        out = await ops.run_watch(
            config_dict,
            source=_SOURCE,
            adapter="als_logbook",
            once=True,
            interval=None,
            dry_run=False,
            progress=messages.append,
        )

        assert config_dict["ingestion"] == {"source_url": _SOURCE, "adapter": "als_logbook"}
        assert schedulers[0].config.ingestion.adapter == "als_logbook"
        assert out is not None
        assert (out.entries_added, out.entries_failed) == (4, 2)
        assert out.duration_seconds == 1.25
        assert out.since == since
        assert out.dry_run is False
        assert f"Running single poll cycle (source: {_SOURCE})" in messages

    async def test_interval_override_reaches_the_scheduler_config(
        self, monkeypatch, mock_repository
    ):
        schedulers = _patch_scheduler(monkeypatch, _poll_result())
        _patch_service(monkeypatch, _StubService(mock_repository))

        config_dict = _config(adapter="generic_json", source_url=_SOURCE)
        await ops.run_watch(
            config_dict,
            source=None,
            adapter=None,
            once=True,
            interval=30,
            dry_run=False,
        )

        assert config_dict["ingestion"]["poll_interval_seconds"] == 30
        assert schedulers[0].config.ingestion.poll_interval_seconds == 30

    async def test_interval_without_a_source_is_recorded_then_rejected(self, monkeypatch):
        _forbid_service(monkeypatch)

        config_dict = dict(_DB)
        with pytest.raises(ValueError, match="No ingestion source configured"):
            await ops.run_watch(
                config_dict,
                source=None,
                adapter=None,
                once=True,
                interval=45,
                dry_run=False,
            )

        # The override is applied before the source check, so the synthesized
        # block is left behind even though the call is rejected.
        assert config_dict["ingestion"] == {"poll_interval_seconds": 45}

    async def test_the_updated_count_survives_the_projection(self, monkeypatch, mock_repository):
        """``entries_updated`` reaches the CLI result, not just the poll result.

        ``poll_once`` hardcodes 0 today, so this is only observable through a
        scheduler double -- but the projection is where a real upsert/update
        split would be silently dropped, and ``watch --once`` would then
        under-report every entry it refreshed rather than inserted.
        """
        _patch_scheduler(monkeypatch, _poll_result(added=2, updated=7, failed=1))
        _patch_service(monkeypatch, _StubService(mock_repository))

        out = await ops.run_watch(
            _config(source_url=_SOURCE),
            source=None,
            adapter=None,
            once=True,
            interval=None,
            dry_run=False,
        )

        assert out is not None
        assert (out.entries_added, out.entries_updated, out.entries_failed) == (2, 7, 1)

    async def test_dry_run_flag_reaches_poll_once_and_the_result(
        self, monkeypatch, mock_repository
    ):
        schedulers = _patch_scheduler(monkeypatch, _poll_result(added=6))
        _patch_service(monkeypatch, _StubService(mock_repository))

        out = await ops.run_watch(
            _config(source_url=_SOURCE),
            source=None,
            adapter=None,
            once=True,
            interval=None,
            dry_run=True,
        )

        assert schedulers[0].poll_calls == [{"dry_run": True, "limit": None}]
        assert out is not None
        assert out.dry_run is True
        assert out.entries_added == 6


class TestRunWatchDaemon:
    async def test_daemon_mode_installs_signal_handlers_and_runs_the_loop(
        self, monkeypatch, mock_repository
    ):
        schedulers = _patch_scheduler(monkeypatch)
        _patch_service(monkeypatch, _StubService(mock_repository))
        loop = asyncio.get_running_loop()

        messages: list[str] = []
        try:
            out = await ops.run_watch(
                _config(source_url=_SOURCE, poll_interval_seconds=120),
                source=None,
                adapter=None,
                once=False,
                interval=None,
                dry_run=False,
                progress=messages.append,
            )

            assert out is None
            assert schedulers[0].run_forever_calls == 1
            assert schedulers[0].poll_calls == []
            assert messages == [
                f"Watching: {_SOURCE}",
                "Poll interval: 120s",
                "Press Ctrl+C to stop\n",
            ]
            # remove_signal_handler reports whether one was installed -- and
            # takes it back off the loop.
            assert loop.remove_signal_handler(signal.SIGINT) is True
            assert loop.remove_signal_handler(signal.SIGTERM) is True
        finally:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)

    async def test_installed_handler_stops_the_scheduler(self, monkeypatch, mock_repository):
        schedulers = _patch_scheduler(monkeypatch)
        _patch_service(monkeypatch, _StubService(mock_repository))

        loop = asyncio.get_running_loop()
        handlers: dict[int, Callable[[], None]] = {}
        real_add = loop.add_signal_handler

        def _record(sig, callback, *args):
            handlers[sig] = callback
            real_add(sig, callback, *args)

        monkeypatch.setattr(loop, "add_signal_handler", _record)

        try:
            await ops.run_watch(
                _config(source_url=_SOURCE),
                source=None,
                adapter=None,
                once=False,
                interval=None,
                dry_run=False,
            )

            assert set(handlers) == {signal.SIGINT, signal.SIGTERM}

            handlers[signal.SIGTERM]()
            # The handler schedules stop() as a task; give the loop one turn.
            await asyncio.sleep(0)
            assert schedulers[0].stop_calls == 1
        finally:
            for sig in handlers:
                loop.remove_signal_handler(sig)


# ---------------------------------------------------------------------------
# run_enhance -- entry selection and the enhancement loop
# ---------------------------------------------------------------------------


class TestRunEnhance:
    async def test_no_enhancers_reports_through_progress(self, monkeypatch):
        _patch_enhancers(monkeypatch, [])
        _forbid_service(monkeypatch)

        messages: list[str] = []
        out = await ops.run_enhance(
            dict(_DB), module=None, force=False, limit=10, progress=messages.append
        )

        assert out.entries_processed == 0
        assert messages == ["No enhancement modules enabled or selected"]

    async def test_force_reprocesses_the_time_range_instead_of_incomplete_entries(
        self, monkeypatch, mock_repository
    ):
        enhancer = _Enhancer("text_embedding")
        _patch_enhancers(monkeypatch, [enhancer])
        _patch_service(monkeypatch, _StubService(mock_repository))
        mock_repository.search_by_time_range = _async_return(_entries(2))
        mock_repository.get_incomplete_entries = _async_return([])

        out = await ops.run_enhance(dict(_DB), module=None, force=True, limit=25)

        assert out.entries_processed == 2
        assert out.module_names == ["text_embedding"]
        mock_repository.search_by_time_range.assert_awaited_once_with(limit=25)
        mock_repository.get_incomplete_entries.assert_not_awaited()
        assert enhancer.seen == ["E0", "E1"]
        assert mock_repository.mark_enhancement_complete.await_count == 2

    async def test_named_module_narrows_both_the_query_and_the_enhancers(
        self, monkeypatch, mock_repository
    ):
        wanted = _Enhancer("text_embedding")
        other = _Enhancer("semantic_processor")
        _patch_enhancers(monkeypatch, [wanted, other])
        _patch_service(monkeypatch, _StubService(mock_repository))
        mock_repository.get_incomplete_entries = _async_return(_entries(1))

        out = await ops.run_enhance(dict(_DB), module="text_embedding", force=False, limit=5)

        assert out.module_names == ["text_embedding"]
        assert out.entries_processed == 1
        mock_repository.get_incomplete_entries.assert_awaited_once_with(
            module_name="text_embedding", limit=5
        )
        assert other.seen == []

    async def test_all_modules_dedupes_entries_incomplete_for_more_than_one(
        self, monkeypatch, mock_repository
    ):
        from unittest.mock import AsyncMock

        first = _Enhancer("text_embedding")
        second = _Enhancer("semantic_processor")
        _patch_enhancers(monkeypatch, [first, second])
        _patch_service(monkeypatch, _StubService(mock_repository))
        # E2 is incomplete for both modules and must be collected once.
        mock_repository.get_incomplete_entries = AsyncMock(
            side_effect=[
                [{"entry_id": "E1"}, {"entry_id": "E2"}],
                [{"entry_id": "E2"}, {"entry_id": "E3"}],
            ]
        )

        out = await ops.run_enhance(dict(_DB), module=None, force=False, limit=50)

        assert out.entries_processed == 3
        assert first.seen == ["E1", "E2", "E3"]
        # Every collected entry is offered to every enhancer, not just the one
        # whose query produced it.
        assert second.seen == ["E1", "E2", "E3"]
        assert mock_repository.get_incomplete_entries.await_count == 2

    async def test_enhancement_failure_is_recorded_and_the_loop_continues(
        self, monkeypatch, mock_repository
    ):
        _patch_enhancers(monkeypatch, [_Enhancer("text_embedding", fails_on={"E0"})])
        _patch_service(monkeypatch, _StubService(mock_repository))
        mock_repository.get_incomplete_entries = _async_return(_entries(2))

        out = await ops.run_enhance(dict(_DB), module=None, force=False, limit=10)

        assert out.entries_processed == 2
        mock_repository.mark_enhancement_failed.assert_awaited_once_with(
            "E0", "text_embedding", "enhance failed on E0"
        )
        mock_repository.mark_enhancement_complete.assert_awaited_once_with("E1", "text_embedding")

    async def test_progress_reports_the_batch_and_every_tenth_entry(
        self, monkeypatch, mock_repository
    ):
        _patch_enhancers(monkeypatch, [_Enhancer("text_embedding")])
        _patch_service(monkeypatch, _StubService(mock_repository))
        mock_repository.get_incomplete_entries = _async_return(_entries(10))

        messages: list[str] = []
        await ops.run_enhance(
            dict(_DB), module=None, force=False, limit=10, progress=messages.append
        )

        assert "Enhancement modules: ['text_embedding']" in messages
        assert "Processing 10 entries..." in messages
        assert "  Processed 10 entries..." in messages


# ---------------------------------------------------------------------------
# seed_logbook_entries -- progress branches
# ---------------------------------------------------------------------------


class TestSeedLogbookEntriesProgress:
    async def test_progress_reports_every_hundred_and_a_final_total(
        self, monkeypatch, mock_repository
    ):
        _patch_service(monkeypatch, _StubService(mock_repository))

        messages: list[str] = []
        count = await ops.seed_logbook_entries(dict(_DB), _entries(100), progress=messages.append)

        assert count == 100
        assert messages == [
            "  Seeded 100 entries...",
            "✓ Seeded 100 logbook entries.",
        ]

    async def test_final_total_is_reported_even_below_the_batch_size(
        self, monkeypatch, mock_repository
    ):
        _patch_service(monkeypatch, _StubService(mock_repository))

        messages: list[str] = []
        await ops.seed_logbook_entries(dict(_DB), _entries(2), progress=messages.append)

        assert messages == ["✓ Seeded 2 logbook entries."]


# ---------------------------------------------------------------------------
# IngestionScheduler.poll_once -- per-entry failure, driven through run_watch
# ---------------------------------------------------------------------------


class TestSchedulerPollOnceEntryFailures:
    async def test_one_bad_entry_is_counted_and_logged_without_stopping_the_poll(
        self, monkeypatch, mock_repository, caplog
    ):
        # The real scheduler here -- only the adapter, the enhancers and the
        # service are faked.
        _patch_adapter(monkeypatch, _Adapter(_entries(3)))
        _patch_enhancers(monkeypatch, [])
        _patch_service(monkeypatch, _StubService(mock_repository))
        mock_repository.get_last_successful_run = _async_return(datetime(2026, 3, 3, tzinfo=UTC))

        def _reject_e1(entry):
            if entry["entry_id"] == "E1":
                raise RuntimeError("entry rejected")

        mock_repository.upsert_entry.side_effect = _reject_e1

        with caplog.at_level(logging.ERROR, logger="ariel"):
            out = await ops.run_watch(
                _config(adapter="generic_json", source_url=_SOURCE),
                source=None,
                adapter=None,
                once=True,
                interval=None,
                dry_run=False,
            )

        assert out is not None
        assert (out.entries_added, out.entries_failed) == (2, 1)
        assert out.since == datetime(2026, 3, 3, tzinfo=UTC)
        # The poll still completes its run rather than failing it.
        mock_repository.complete_ingestion_run.assert_awaited_once_with(
            1, entries_added=2, entries_updated=0, entries_failed=1
        )
        mock_repository.fail_ingestion_run.assert_not_awaited()
        assert "Failed to process entry" in caplog.text
        assert "entry rejected" in caplog.text
