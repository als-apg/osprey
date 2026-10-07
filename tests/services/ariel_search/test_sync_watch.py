"""Daemon tests for ``run_sync_watch`` and ``osprey ariel sync --watch``.

The composite is the entry point the bundled ARIEL sync container runs: one
asyncio task that syncs once and then keeps polling. What is pinned here is the
behaviour a long-running container depends on and the plain ``sync`` path does
not have:

* a sync that fails is logged and does not stop the daemon, and the watch that
  follows still does a full first ingest, so a container that started before its
  source was reachable ingests everything on the first poll that works;
* one SIGINT/SIGTERM handler, installed before the sync starts, cancels
  whichever half is in flight -- ``run_watch`` installs none of its own;
* the per-poll wrapper runs the qmd resync, the poll and the enhance cleanup, and
  a failing enhancer neither changes the poll result nor counts as an ingestion
  failure, which is what the scheduler's consecutive-failure cap counts;
* the exit code: the failure cap is non-zero, a signal and a cancellation are
  zero.

Everything crossing a process boundary is faked and nothing sleeps: the
scheduler double's ``run_forever`` drives the poll wrapper directly and returns
the stop reason the test wants.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import ariel_group
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.ingestion.scheduler import StopReason
from tests.services.ariel_search._cli_ops_doubles import _patch_service, _StubService

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_SOURCE = "file:///entries.json"


def _config(**ingestion: Any) -> dict[str, Any]:
    """Config dict with a reachable-looking ingestion source and its adapter."""
    return {
        "database": {"uri": "postgresql://localhost/test"},
        "ingestion": {"adapter": "generic_json", "source_url": _SOURCE, **ingestion},
    }


class _Repo:
    """Repository stand-in: a pool plus the schema-facts surface the watch reaches."""

    def __init__(self) -> None:
        self.pool = object()
        self.invalidations = 0

    async def schema_facts(self):
        from osprey.services.ariel_search.database.repository import SchemaFacts

        return SchemaFacts(has_v2_fts=False, has_copy_state=False)

    def invalidate_schema_facts(self) -> None:
        self.invalidations += 1


def _poll_result(added: int = 2):
    from osprey.services.ariel_search.ingestion.scheduler import IngestionPollResult

    return IngestionPollResult(
        entries_added=added,
        entries_updated=0,
        entries_failed=0,
        duration_seconds=0.1,
        since=None,
    )


def _patch_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    polls: int = 1,
    stop_reason: Any = StopReason.SIGNAL,
    order: list[str] | None = None,
    on_poll: Callable[[Any], Awaitable[None]] | None = None,
) -> list[Any]:
    """Route ``IngestionScheduler`` to a double that runs *polls* poll cycles.

    ``run_forever`` calls ``self.poll_once()``, which is the wrapper
    ``run_watch`` installed, and records what it returned. ``on_poll`` is
    awaited with the scheduler's repository inside each inner poll.
    """
    import osprey.services.ariel_search.ingestion.scheduler as sched_mod

    made: list[Any] = []

    class _Recorded:
        def __init__(self, config: Any, repository: Any) -> None:
            self.config = config
            self.repository = repository
            self.inner_polls = 0
            self.results: list[Any] = []
            self.stop_calls = 0
            self._stop_event = asyncio.Event()
            made.append(self)

        async def poll_once(self, dry_run: bool = False, limit: int | None = None):  # noqa: ARG002 - the poll_once signature
            self.inner_polls += 1
            if order is not None:
                order.append("poll")
            if on_poll is not None:
                await on_poll(self.repository)
            return _poll_result()

        async def run_forever(self) -> Any:
            for _ in range(polls):
                self.results.append(await self.poll_once())
            return stop_reason

        async def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setattr(sched_mod, "IngestionScheduler", _Recorded)
    return made


def _patch_poll_neighbours(
    monkeypatch: pytest.MonkeyPatch,
    order: list[str] | None = None,
    enhance_error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Fake the resync pre-step and the catch-up around the poll."""
    enhance_calls: list[dict[str, Any]] = []

    async def _resync(config_dict: dict, progress: Any = None):  # noqa: ARG001 - the resync_qmd_mirror_best_effort signature
        if order is not None:
            order.append("resync")
        return None

    async def _enhance(config_dict: dict, *, budget_s, stop_event, progress=None):  # noqa: ARG001 - the run_catchup signature
        if order is not None:
            order.append("enhance")
        enhance_calls.append({"budget_s": budget_s, "stop_event": stop_event})
        if enhance_error is not None:
            raise enhance_error
        return ops.EnhanceResult(entries_processed=0, module_names=[])

    monkeypatch.setattr(ops, "resync_qmd_mirror_best_effort", _resync)
    monkeypatch.setattr(ops, "run_catchup", _enhance)
    return enhance_calls


def _patch_sync(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception | None = None,
    busy_skipped: list[str] | None = None,
) -> list[str]:
    """Replace the sync half; returns the call log so order can be asserted."""
    order: list[str] = []

    async def _sync(config_dict, limit=None, progress=None):  # noqa: ARG001 - the run_sync signature
        order.append("sync")
        if error is not None:
            raise error
        return ops.SyncResult(0, 0, 0, 0, False, busy_skipped=list(busy_skipped or []))

    monkeypatch.setattr(ops, "run_sync", _sync)
    return order


def _patch_migrate(
    monkeypatch: pytest.MonkeyPatch,
    results: list[Any] | None = None,
    order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fake ``run_migrations_detailed``; each call pops the next result.

    A result is a ``MigrationResult`` or an exception to raise. Once the list is
    exhausted every call reports nothing busy. Returns the call log.
    """
    import osprey.services.ariel_search.database.migrations as mig_mod

    calls: list[dict[str, Any]] = []
    queue = list(results or [])

    async def _migrate(pool, config, **kwargs):  # noqa: ARG001 - the run_migrations_detailed signature
        calls.append({"pool": pool, **kwargs})
        if order is not None:
            order.append("migrate")
        nxt = queue.pop(0) if queue else mig_mod.MigrationResult()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(mig_mod, "run_migrations_detailed", _migrate)
    return calls


@pytest.fixture(autouse=True)
def _no_real_migrations(monkeypatch):
    """Default: the per-poll retry (when reached) finds nothing busy."""
    _patch_migrate(monkeypatch)


def _patch_watch(
    monkeypatch: pytest.MonkeyPatch,
    order: list[str] | None = None,
    reason: Any = None,
    seen: dict[str, Any] | None = None,
) -> None:
    """Replace the watch half, in place of the scheduler-level doubles above.

    ``reason`` is appended to the caller's ``stop_reason_out`` list, standing in
    for a loop that ended on it; ``seen`` collects the arguments the composite
    drove ``run_watch`` with.
    """

    async def _watch(config_dict, source, adapter, once, interval, dry_run, progress=None, **kw):  # noqa: ARG001 - the run_watch signature
        if order is not None:
            order.append("watch")
        if seen is not None:
            seen.update(kw)
            seen["once"] = once
            seen["dry_run"] = dry_run
        if reason is not None:
            kw["stop_reason_out"].append(reason)
        return None

    monkeypatch.setattr(ops, "run_watch", _watch)


def _record_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> dict[int, Callable[[], None]]:
    """Capture the handlers installed on the running loop, still installing them."""
    loop = asyncio.get_running_loop()
    handlers: dict[int, Callable[[], None]] = {}
    real_add = loop.add_signal_handler

    def _record(sig, callback, *args):
        handlers[sig] = callback
        real_add(sig, callback, *args)

    monkeypatch.setattr(loop, "add_signal_handler", _record)
    return handlers


# ---------------------------------------------------------------------------
# run_sync_watch -- the composite
# ---------------------------------------------------------------------------


class TestRunSyncWatch:
    async def test_a_failing_sync_still_reaches_a_full_first_poll(self, monkeypatch, caplog):
        """The daemon's whole point: a dead source at start-up is not fatal."""
        _patch_sync(monkeypatch, error=RuntimeError("source unreachable"))
        _patch_poll_neighbours(monkeypatch)
        schedulers = _patch_scheduler(monkeypatch)
        _patch_service(monkeypatch, _StubService(_Repo()))

        with caplog.at_level(logging.WARNING, logger="ariel"):
            reason = await ops.run_sync_watch(_config())

        assert reason is StopReason.SIGNAL
        # The poll ran, and it was asked for a full ingest rather than the
        # scheduler's default "skip until some earlier run succeeded".
        assert schedulers[0].inner_polls == 1
        assert schedulers[0].config.ingestion.watch.require_initial_ingest is False
        # The warning names the failure rather than swallowing it.
        assert "RuntimeError" in caplog.text
        assert "source unreachable" in caplog.text

    async def test_a_failing_sync_is_reported_through_progress(self, monkeypatch):
        _patch_sync(monkeypatch, error=RuntimeError("source unreachable"))
        _patch_poll_neighbours(monkeypatch)
        _patch_scheduler(monkeypatch)
        _patch_service(monkeypatch, _StubService(_Repo()))

        messages: list[str] = []
        await ops.run_sync_watch(_config(), progress=messages.append)

        assert any("source unreachable" in line for line in messages)

    async def test_the_watch_half_is_driven_as_an_embedded_caller(self, monkeypatch):
        """``run_watch`` gets the composite's three keyword controls, not defaults."""
        seen: dict[str, Any] = {}
        _patch_sync(monkeypatch)
        _patch_watch(monkeypatch, reason=StopReason.FAILURE_CAP, seen=seen)

        reason = await ops.run_sync_watch(_config())

        assert reason is StopReason.FAILURE_CAP
        assert seen["require_initial_ingest"] is False
        assert seen["install_signal_handlers"] is False
        assert seen["once"] is False
        assert seen["dry_run"] is False

    async def test_a_loop_that_reports_no_reason_returns_none(self, monkeypatch):
        _patch_sync(monkeypatch)
        _patch_watch(monkeypatch)

        assert await ops.run_sync_watch(_config()) is None

    async def test_signal_handlers_are_installed_before_the_sync(self, monkeypatch):
        """A container killed during a long first sync must still stop."""
        order = _patch_sync(monkeypatch)
        _patch_watch(monkeypatch, order=order)
        handlers = _record_signal_handlers(monkeypatch)

        await ops.run_sync_watch(_config())

        assert set(handlers) == {signal.SIGINT, signal.SIGTERM}
        assert order == ["sync", "watch"]

    async def test_the_handlers_are_taken_back_off_the_loop(self, monkeypatch):
        _patch_sync(monkeypatch)
        _patch_watch(monkeypatch)

        await ops.run_sync_watch(_config())

        loop = asyncio.get_running_loop()
        assert loop.remove_signal_handler(signal.SIGINT) is False
        assert loop.remove_signal_handler(signal.SIGTERM) is False

    async def test_a_signal_during_the_sync_cancels_the_run(self, monkeypatch):
        """Cancellation reaches the in-flight sync; the watch never starts."""
        started = asyncio.Event()
        reached: list[str] = []

        async def _sync(config_dict, limit=None, progress=None):  # noqa: ARG001 - the run_sync signature
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(ops, "run_sync", _sync)
        _patch_watch(monkeypatch, order=reached)
        handlers = _record_signal_handlers(monkeypatch)

        task = asyncio.ensure_future(ops.run_sync_watch(_config()))
        await started.wait()
        handlers[signal.SIGINT]()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert reached == []


# ---------------------------------------------------------------------------
# The per-poll wrapper -- resync, poll, enhance cleanup
# ---------------------------------------------------------------------------


class TestPerPollWrapper:
    async def test_each_poll_resyncs_then_polls_then_runs_the_enhance_cleanup(self, monkeypatch):
        order: list[str] = []
        enhance_calls = _patch_poll_neighbours(monkeypatch, order=order)
        schedulers = _patch_scheduler(monkeypatch, polls=2, order=order)
        _patch_service(monkeypatch, _StubService(_Repo()))
        _patch_sync(monkeypatch)

        await ops.run_sync_watch(_config())

        assert schedulers[0].inner_polls == 2
        assert order == ["resync", "poll", "enhance", "resync", "poll", "enhance"]
        # Each poll runs the catch-up with the scheduler's own stop event and
        # what is left of the poll interval as its budget.
        assert len(enhance_calls) == 2
        for call in enhance_calls:
            assert call["stop_event"] is schedulers[0]._stop_event
            assert 0 < call["budget_s"] <= schedulers[0].config.ingestion.poll_interval_seconds

    async def test_an_enhancer_failure_leaves_the_poll_result_untouched(self, monkeypatch, caplog):
        """Only ingestion failures may count toward the consecutive-failure cap."""
        _patch_poll_neighbours(monkeypatch, enhance_error=RuntimeError("embedding endpoint down"))
        schedulers = _patch_scheduler(monkeypatch)
        _patch_service(monkeypatch, _StubService(_Repo()))
        _patch_sync(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="ariel"):
            reason = await ops.run_sync_watch(_config())

        assert reason is StopReason.SIGNAL
        # The exception never left the wrapper: the scheduler saw a normal poll.
        assert schedulers[0].inner_polls == 1
        assert [r.entries_added for r in schedulers[0].results] == [2]
        assert "embedding endpoint down" in caplog.text


# ---------------------------------------------------------------------------
# Busy-migration retry -- sync hand-off and per-poll
# ---------------------------------------------------------------------------


def _busy(*names: str):
    from osprey.services.ariel_search.database.migrations import MigrationResult

    return MigrationResult(busy_skipped=list(names))


def _lock_held():
    from osprey.services.ariel_search.database.migrations import MigrationResult

    return MigrationResult(lock_held=True)


def _applied(*names: str):
    from osprey.services.ariel_search.database.migrations import MigrationResult

    return MigrationResult(applied=list(names))


class TestBusyMigrationRetry:
    async def test_sync_hands_its_busy_set_to_the_watch(self, monkeypatch):
        seen: dict[str, Any] = {}
        _patch_sync(monkeypatch, busy_skipped=["004_x"])
        _patch_watch(monkeypatch, seen=seen)

        await ops.run_sync_watch(_config())

        assert seen["initial_busy_skipped"] == ["004_x"]

    async def test_a_failed_sync_hands_over_unknown(self, monkeypatch):
        seen: dict[str, Any] = {}
        _patch_sync(monkeypatch, error=RuntimeError("boom"))
        _patch_watch(monkeypatch, seen=seen)

        await ops.run_sync_watch(_config())

        assert seen["initial_busy_skipped"] is None

    async def test_busy_skip_is_retried_before_each_poll_until_applied(self, monkeypatch):
        order: list[str] = []
        calls = _patch_migrate(monkeypatch, [_busy("004_x"), _busy()], order=order)
        _patch_poll_neighbours(monkeypatch, order=order)
        _patch_scheduler(monkeypatch, polls=3, order=order)
        service = _StubService(_Repo())
        _patch_service(monkeypatch, service)
        _patch_sync(monkeypatch, busy_skipped=["004_x"])

        await ops.run_sync_watch(_config())

        assert order == [
            "migrate", "resync", "poll", "enhance",
            "migrate", "resync", "poll", "enhance",
            "resync", "poll", "enhance",
        ]  # fmt: skip
        assert len(calls) == 2
        assert all(c["lock"] == "try" and c["pool"] is service.pool for c in calls)

    async def test_a_permanent_skip_triggers_no_per_poll_migrate(self, monkeypatch):
        """pgvector-missing skips are not busy skips, so the set is empty."""
        calls = _patch_migrate(monkeypatch)
        _patch_poll_neighbours(monkeypatch)
        _patch_scheduler(monkeypatch, polls=3)
        _patch_service(monkeypatch, _StubService(_Repo()))
        _patch_sync(monkeypatch, busy_skipped=[])

        await ops.run_sync_watch(_config())

        assert calls == []

    async def test_standalone_watch_makes_no_migrate_call(self, monkeypatch):
        calls = _patch_migrate(monkeypatch)
        _patch_poll_neighbours(monkeypatch)
        _patch_scheduler(monkeypatch, polls=2)
        _patch_service(monkeypatch, _StubService(_Repo()))

        await ops.run_watch(
            _config(), None, None, False, None, False, install_signal_handlers=False
        )

        assert calls == []

    async def test_a_raising_sync_still_migrates_once_before_the_first_poll(self, monkeypatch):
        order: list[str] = []
        calls = _patch_migrate(monkeypatch, [_busy()], order=order)
        _patch_poll_neighbours(monkeypatch, order=order)
        _patch_scheduler(monkeypatch, polls=3, order=order)
        _patch_service(monkeypatch, _StubService(_Repo()))
        _patch_sync(monkeypatch, error=RuntimeError("source unreachable"))

        await ops.run_sync_watch(_config())

        assert len(calls) == 1
        assert order[:2] == ["migrate", "resync"]

    async def test_lock_held_elsewhere_skips_the_retry_without_waiting(self, monkeypatch, caplog):
        calls = _patch_migrate(monkeypatch, [_lock_held(), _busy()])
        _patch_poll_neighbours(monkeypatch)
        schedulers = _patch_scheduler(monkeypatch, polls=3)
        _patch_service(monkeypatch, _StubService(_Repo()))
        _patch_sync(monkeypatch, busy_skipped=["004_x"])

        with caplog.at_level(logging.INFO, logger="ariel"):
            await asyncio.wait_for(ops.run_sync_watch(_config()), timeout=5)

        assert "migrate running elsewhere" in caplog.text
        assert schedulers[0].inner_polls == 3
        # The busy set survived the held lock, so the next poll retried.
        assert len(calls) == 2
        assert all(c["lock"] == "try" for c in calls)

    async def test_an_applied_migration_invalidates_the_cached_schema_facts(self, monkeypatch):
        order: list[str] = []
        _patch_migrate(monkeypatch, [_busy("004_x"), _applied("004_x")], order=order)
        _patch_poll_neighbours(monkeypatch)
        _patch_scheduler(monkeypatch, polls=2, order=order)
        repo = _Repo()
        _patch_service(monkeypatch, _StubService(repo))
        _patch_sync(monkeypatch, busy_skipped=["004_x"])

        await ops.run_sync_watch(_config())

        assert order == ["migrate", "poll", "migrate", "poll"]
        # Only the retry that applied something invalidates.
        assert repo.invalidations == 1

    async def test_busy_then_applied_records_copy_rows_in_the_same_poll(
        self, monkeypatch, fake_pool
    ):
        """A negative probe cached before the migrate does not outlive it.

        The repository is real and its pool fake: the first probe sees no
        ``copy_status`` column, the migration that adds it applies on the next
        retry, and the poll right after that retry already records rows -- no
        waiting out the negative TTL on a frozen clock.
        """
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.database.repository import ARIELRepository

        probe = "information_schema.columns"
        fake_pool.recorder.rows_for[probe] = [(False, False)]
        repo = ARIELRepository(fake_pool, ARIELConfig.from_dict(_config()))
        repo._clock = lambda: 1000.0
        assert (await repo.schema_facts()).has_copy_state is False

        recorded: list[int] = []

        async def _record_rows(repository: Any) -> None:
            if (await repository.schema_facts()).has_copy_state:
                recorded.append(len(recorded) + 1)

        import osprey.services.ariel_search.database.migrations as mig_mod

        results = [_busy("005_copy_state"), _applied("005_copy_state")]

        async def _migrate(pool, config, **kwargs):  # noqa: ARG001 - the run_migrations_detailed signature
            result = results.pop(0)
            if result.applied:  # the migration adds the column.
                fake_pool.recorder.rows_for[probe] = [(False, True)]
            return result

        monkeypatch.setattr(mig_mod, "run_migrations_detailed", _migrate)
        _patch_poll_neighbours(monkeypatch)
        _patch_scheduler(monkeypatch, polls=2, on_poll=_record_rows)
        _patch_service(monkeypatch, _StubService(repo))
        _patch_sync(monkeypatch, busy_skipped=["005_copy_state"])

        await ops.run_sync_watch(_config())

        # Poll 1 ran on the busy schema; poll 2 followed the applying retry.
        assert recorded == [1]

    async def test_a_failing_retry_does_not_fail_the_poll(self, monkeypatch):
        _patch_migrate(monkeypatch, [RuntimeError("db down")])
        _patch_poll_neighbours(monkeypatch)
        schedulers = _patch_scheduler(monkeypatch)
        _patch_service(monkeypatch, _StubService(_Repo()))
        _patch_sync(monkeypatch, busy_skipped=["004_x"])

        await ops.run_sync_watch(_config())

        assert schedulers[0].inner_polls == 1


# ---------------------------------------------------------------------------
# osprey ariel sync --watch -- the exit-code contract
# ---------------------------------------------------------------------------


class TestSyncWatchCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture(autouse=True)
    def _config_present(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: _config() if key == "ariel" else default,
        )

    def _patch_composite(self, monkeypatch, result: Any = None, error: BaseException | None = None):
        calls: list[dict[str, Any]] = []

        async def _composite(config_dict, progress=None):  # noqa: ARG001 - the run_sync_watch signature
            calls.append({"config_dict": config_dict})
            if error is not None:
                raise error
            return result

        monkeypatch.setattr(ops, "run_sync_watch", _composite)
        return calls

    def test_help_names_the_flag(self, runner):
        result = runner.invoke(ariel_group, ["sync", "--help"])

        assert result.exit_code == 0
        assert "--watch" in result.output

    def test_the_failure_cap_exits_non_zero(self, runner, monkeypatch):
        self._patch_composite(monkeypatch, result=StopReason.FAILURE_CAP)

        result = runner.invoke(ariel_group, ["sync", "--watch"])

        assert result.exit_code == 1

    def test_a_signal_exits_zero(self, runner, monkeypatch):
        self._patch_composite(monkeypatch, result=StopReason.SIGNAL)

        result = runner.invoke(ariel_group, ["sync", "--watch"])

        assert result.exit_code == 0

    def test_a_cancelled_run_exits_zero(self, runner, monkeypatch):
        self._patch_composite(monkeypatch, error=asyncio.CancelledError())

        result = runner.invoke(ariel_group, ["sync", "--watch"])

        assert result.exit_code == 0

    def test_a_loop_with_no_reason_exits_zero(self, runner, monkeypatch):
        self._patch_composite(monkeypatch, result=None)

        result = runner.invoke(ariel_group, ["sync", "--watch"])

        assert result.exit_code == 0

    def test_without_the_flag_the_sync_stays_one_shot(self, runner, monkeypatch):
        composite_calls = self._patch_composite(monkeypatch, result=None)
        sync_calls: list[Any] = []

        async def _sync(config_dict, limit=None, progress=None):  # noqa: ARG001 - the run_sync signature
            sync_calls.append(limit)
            return ops.SyncResult(
                migrations_applied=0,
                entries_ingested=0,
                entries_enhanced=0,
                entries_failed=0,
                was_initial_ingest=False,
            )

        monkeypatch.setattr(ops, "run_sync", _sync)

        result = runner.invoke(ariel_group, ["sync"])

        assert result.exit_code == 0
        assert sync_calls == [None]
        assert composite_calls == []
