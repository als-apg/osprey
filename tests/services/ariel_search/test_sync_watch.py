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
    from collections.abc import Callable

_SOURCE = "file:///entries.json"


def _config(**ingestion: Any) -> dict[str, Any]:
    """Config dict with a reachable-looking ingestion source."""
    return {
        "database": {"uri": "postgresql://localhost/test"},
        "ingestion": {"source_url": _SOURCE, **ingestion},
    }


class _Repo:
    """Repository stand-in; only its pool is ever reached."""

    def __init__(self) -> None:
        self.pool = object()


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
) -> list[Any]:
    """Route ``IngestionScheduler`` to a double that runs *polls* poll cycles.

    ``run_forever`` calls ``self.poll_once()``, which is the wrapper
    ``run_watch`` installed, and records what it returned.
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
            made.append(self)

        async def poll_once(self, dry_run: bool = False, limit: int | None = None):
            self.inner_polls += 1
            if order is not None:
                order.append("poll")
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
    """Fake the resync pre-step and the enhance cleanup around the poll."""
    enhance_calls: list[dict[str, Any]] = []

    async def _resync(config_dict: dict, progress: Any = None):
        if order is not None:
            order.append("resync")
        return None

    async def _enhance(config_dict: dict, module=None, force=False, limit=0, progress=None):
        if order is not None:
            order.append("enhance")
        enhance_calls.append({"module": module, "force": force, "limit": limit})
        if enhance_error is not None:
            raise enhance_error
        return ops.EnhanceResult(entries_processed=0, module_names=[])

    monkeypatch.setattr(ops, "resync_qmd_mirror_best_effort", _resync)
    monkeypatch.setattr(ops, "run_enhance", _enhance)
    return enhance_calls


def _patch_sync(monkeypatch: pytest.MonkeyPatch, error: Exception | None = None) -> list[str]:
    """Replace the sync half; returns the call log so order can be asserted."""
    order: list[str] = []

    async def _sync(config_dict, limit=None, progress=None):
        order.append("sync")
        if error is not None:
            raise error
        return None

    monkeypatch.setattr(ops, "run_sync", _sync)
    return order


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

    async def _watch(config_dict, source, adapter, once, interval, dry_run, progress=None, **kw):
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

        async def _sync(config_dict, limit=None, progress=None):
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
        assert enhance_calls == [
            {"module": None, "force": False, "limit": 1000},
            {"module": None, "force": False, "limit": 1000},
        ]

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

        async def _composite(config_dict, progress=None):
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

        async def _sync(config_dict, limit=None, progress=None):
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
