"""Cache + single-flight refresh scheduler backing the health ``/checks`` route.

The web health surface must answer ``/checks`` in constant time in every state —
cold start, mid-refresh, and steady state — while never letting a browser poll
trigger a suite run. This module is the only place a suite executes.

Refresh cycle (single-flight; request-kicked, never a browser-blocking run):

1. **Sync phase** through :func:`osprey.health.offload.run_sync` (never
   ``asyncio.to_thread`` — a wedged plugin import must not block process exit):
   the injected :class:`~osprey.health.loader.HealthConfigLoader`
   resolves the config path, mtime-gates ``.env``/``config.yml``, and assembles
   the merged records.
2. **Poll cycle** via :func:`osprey.health.refresh.run_poll_refresh` — the
   sequence shared with the MCP poll surface: reconcile the
   :class:`~osprey.health.lifecycle.HealthRuntimeLifecycle` runtime against the
   ``control_system`` snapshot, run the ``full=False`` / ``categories=None``
   suite so ``on_demand`` categories never execute and the report is never
   filtered, then append the unfiltered plugin-diagnostic and restart-notice
   rows.

The report is cached with an age clock. ``/checks`` serves the cache and, when
its age crosses ``interval_s − suite_timeout_s``, kicks a background refresh
(refresh-ahead) so a well-behaved poller keeps seeing fresh data; ``stale`` is
reported only once age exceeds ``interval_s``. A cold caller gets a ``warming``
envelope and the single background first run; concurrent cold callers share it.

**Circuit breaker (liveness-keyed).** After every cycle, if any abandoned worker
thread is still alive (:func:`~osprey.health.offload.abandoned_alive_count`),
refreshes are suppressed — the cache keeps serving — until ``config.yml``/``.env``
changes or a ``~backoff_factor × interval_s`` backoff elapses, with one
escalating warning per trip. A slow-but-completing check self-prunes and never
trips, so healthy-but-slow facilities keep full cadence; a wedged plugin import
or hung canary (whose thread stays alive) trips it, bounding live-thread growth
to O(#edits) in the weeks-lived terminal process.

**Re-probe.** :meth:`HealthCheckEngine.reprobe` re-runs part of the report on
an operator's request — the categories behind some rows or one component, or
everything (a rescan). It shares the single flight: while a refresh is in
flight it waits for that one instead of starting a second, it is refused while
the breaker suppresses refreshes, and it merges the re-run categories' rows
into the cache. A named slice runs at full tier — the click is the consent an
``on_demand`` row waits for; a rescan stays the poll.

**The envelope** carries, beyond the report, each row's ``probed_at`` and the
panel layout (:func:`osprey.health.layout.build_layers`), plus what the MCP
introspection cache holds — read with
:func:`~osprey.mcp_server.introspect.peek_mcp_servers`, so serving ``/checks``
never spawns an MCP server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from osprey.health import offload
from osprey.health.config import HealthSettings, parse_health_config
from osprey.health.dedupe import SHARED_PROBES, fold_shared
from osprey.health.layout import build_layers
from osprey.health.models import CheckReport
from osprey.health.refresh import run_poll_refresh
from osprey.health.signatures import disk_signature as _resolve_disk_signature

if TYPE_CHECKING:
    from osprey.health.lifecycle import HealthRuntimeLifecycle
    from osprey.health.loader import LoadedHealthConfig

logger = logging.getLogger("osprey.interfaces.health.engine")

#: Default backoff multiple applied to ``interval_s`` when the breaker trips.
DEFAULT_BREAKER_BACKOFF_FACTOR = 10.0


class RefreshSuppressedError(RuntimeError):
    """A re-probe was asked for while the circuit breaker suppresses refreshes."""


class _SyncLoader(Protocol):
    """Structural type for the injected synchronous config loader."""

    def load(self) -> LoadedHealthConfig: ...


class HealthCheckEngine:
    """Owns the cached report and the single-flight refresh that produces it.

    Args:
        loader: The synchronous config-load phase. Called on a daemon
            thread via :func:`offload.run_sync` each refresh.
        lifecycle: The connector-runtime owner. Its
            :meth:`~HealthRuntimeLifecycle.reconcile` is called each cycle and
            its in-flight-task provider is wired to this engine so teardown can
            cancel a running refresh.
        config_path: Explicit ``config.yml`` path, or ``None`` to resolve like the
            CLI. Used only to compute the breaker's disk-change signature.
        settings: Initial cadence settings; defaults to the framework degraded
            defaults (``suite_timeout_s=30``, ``interval_s=60``) until the first
            successful load replaces them.
        clock: Monotonic time source for cache-age accounting (injectable).
        disk_signature: Override for the breaker's disk-change probe (injectable);
            defaults to stat-ing ``config.yml`` and its sibling ``.env``.
        backoff_factor: Multiple of ``interval_s`` a trip suppresses refreshes for.
        wall_clock: Epoch time source for the ``probed_at`` / ``scanned_at``
            stamps the envelope carries (injectable).
    """

    def __init__(
        self,
        *,
        loader: _SyncLoader,
        lifecycle: HealthRuntimeLifecycle,
        config_path: str | Path | None = None,
        settings: HealthSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
        disk_signature: Callable[[], Any] | None = None,
        backoff_factor: float = DEFAULT_BREAKER_BACKOFF_FACTOR,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._loader = loader
        self._lifecycle = lifecycle
        self._config_path = config_path
        self._settings = settings if settings is not None else parse_health_config(None)
        self._clock = clock
        self._disk_signature_fn = disk_signature or self._default_disk_signature
        self._backoff_factor = backoff_factor

        self._report: CheckReport | None = None
        self._cached_at: float = 0.0
        self._config_ok: bool = False
        self._refresh_task: asyncio.Task[None] | None = None
        self._wall = wall_clock
        self._scanned_at: float | None = None
        #: Epoch time each row (by name) was last probed.
        self._probed_at: dict[str, float] = {}
        #: The expanded config of the last completed refresh, for the layout.
        self._expanded: dict[str, Any] | None = None

        # Circuit-breaker state.
        self._suppressed_until: float | None = None
        self._suppressed_sig: Any = None
        self._breaker_trips = 0

        # Teardown must cancel this engine's in-flight refresh before the
        # connector is disconnected; expose the task to the lifecycle.
        self._lifecycle.set_inflight_task_provider(self.current_refresh_task)

    # -- public surface ---------------------------------------------------------

    @property
    def config_ok(self) -> bool:
        """Whether the last completed refresh loaded a usable config."""
        return self._config_ok

    @property
    def expanded(self) -> dict[str, Any] | None:
        """The expanded config the last completed refresh ran against."""
        return self._expanded

    def current_refresh_task(self) -> asyncio.Task[None] | None:
        """Return the in-flight refresh task, or ``None`` when idle/done."""
        if self._refresh_task is not None and not self._refresh_task.done():
            return self._refresh_task
        return None

    def get_checks(self) -> dict[str, Any]:
        """Return the ``/checks`` envelope, kicking a background refresh as needed.

        Constant-time in all states: it never awaits a suite. On the cold path it
        returns a ``warming`` envelope and kicks the single first run (concurrent
        cold callers share it). Otherwise it serves the cached report and, past
        the refresh-ahead threshold, kicks a background refresh; ``stale`` is set
        only once the cache age exceeds ``interval_s``.
        """
        if self._report is None:
            self._maybe_kick()
            return self._envelope(None, warming=True, stale=True)

        age = self._clock() - self._cached_at
        interval_s = self._settings.interval_s
        if age > interval_s - self._settings.suite_timeout_s:
            self._maybe_kick()
        return self._envelope(self._report, warming=False, stale=age > interval_s)

    # -- envelope ---------------------------------------------------------------

    def _envelope(
        self, report: CheckReport | None, *, warming: bool, stale: bool
    ) -> dict[str, Any]:
        # The cache holds unfolded rows so a partial re-probe merges cleanly;
        # what is served is folded.
        if report is not None:
            report = CheckReport(
                results=fold_shared(report.results),
                elapsed_ms=report.elapsed_ms,
                deadline_hit=report.deadline_hit,
            )
        base = (report or CheckReport()).to_dict()
        base["stale"] = stale
        base["warming"] = warming
        base["interval_s"] = self._settings.interval_s
        base["title"] = self._settings.title
        base["scanned_at"] = self._scanned_at if report is not None else None
        for row in base["results"]:
            row["probed_at"] = self._probed_at.get(row["name"], self._scanned_at)
        servers = self._mcp_configured()
        base["layers"] = build_layers(base["results"], self._expanded, mcp_configured=servers)
        base["mcp"] = self._mcp_summary(servers)
        return base

    # -- MCP introspection cache (read-only here) ------------------------------

    def mcp_json_path(self) -> Path | None:
        """The rendered ``.mcp.json`` beside the deployment's ``config.yml``."""
        try:
            if self._config_path is not None:
                config_path = Path(self._config_path)
            else:
                from osprey_connectors.workspace import resolve_config_path

                config_path = resolve_config_path()
        except Exception:
            return None
        return config_path.parent / ".mcp.json"

    def _mcp_configured(self) -> list[str]:
        path = self.mcp_json_path()
        try:
            servers = json.loads(path.read_text(encoding="utf-8")).get("mcpServers") if path else {}
        except (OSError, ValueError, AttributeError):
            return []
        return list(servers) if isinstance(servers, dict) else []

    def _mcp_summary(self, configured: list[str]) -> dict[str, Any]:
        from osprey.mcp_server.introspect import peek_mcp_servers

        path = self.mcp_json_path()
        servers, probed_at = peek_mcp_servers(path) if path is not None else (None, None)
        stdio = [s for s in servers or [] if s.get("transport", "stdio") == "stdio"]
        return {
            "configured": configured,
            "stdio": len(stdio) if servers is not None else None,
            "answered": sum(1 for s in stdio if s.get("tools") is not None)
            if servers is not None
            else None,
            "tools": sum(len(s.get("tools") or []) for s in stdio) if servers is not None else None,
            "probed_at": probed_at,
        }

    # -- re-probe ---------------------------------------------------------------

    def categories_for(
        self, *, checks: Collection[str] = (), component: str | None = None
    ) -> set[str]:
        """The categories whose rows answer for *checks* or *component* in the cache."""
        if self._report is None:
            return set()
        rows = [r.to_dict() for r in self._report.results]
        build_layers(rows, self._expanded)
        wanted = set(checks)
        for group in SHARED_PROBES:
            # A folded row answers for every probe folded into it.
            if wanted & set(group):
                wanted.update(group)
        return {
            r["category"]
            for r in rows
            if r["name"] in wanted or (component is not None and r.get("component") == component)
        }

    async def reprobe(
        self, *, checks: Collection[str] = (), component: str | None = None
    ) -> dict[str, Any]:
        """Re-run the categories behind *checks* / *component* (everything when neither).

        Shares the single flight: an in-flight refresh is awaited rather than
        joined by a second run, and the result is merged into the cache.

        Raises:
            RefreshSuppressedError: the circuit breaker is suppressing refreshes.
        """
        if self._is_suppressed():
            raise RefreshSuppressedError("refreshes are suppressed after abandoned worker threads")
        inflight = self.current_refresh_task()
        if inflight is not None:
            await asyncio.shield(inflight)
            return self.get_checks()
        categories: set[str] | None = None
        if checks or component is not None:
            categories = self.categories_for(checks=checks, component=component)
            if not categories:
                return self.get_checks()
        task = asyncio.ensure_future(self._refresh(categories=categories))
        task.add_done_callback(self._on_refresh_done)
        self._refresh_task = task
        await asyncio.shield(task)
        return self.get_checks()

    # -- refresh scheduling -----------------------------------------------------

    def _maybe_kick(self) -> None:
        """Schedule a background refresh unless one is running or suppressed."""
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        if self._is_suppressed():
            return
        task = asyncio.ensure_future(self._refresh())
        task.add_done_callback(self._on_refresh_done)
        self._refresh_task = task

    def _on_refresh_done(self, task: asyncio.Task[None]) -> None:
        """Retrieve a finished refresh's exception so asyncio never logs it raw."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("health refresh task failed unexpectedly: %s", exc, exc_info=exc)

    def _is_suppressed(self) -> bool:
        """Whether the breaker is currently suppressing refreshes."""
        if self._suppressed_until is None:
            return False
        if self._clock() >= self._suppressed_until:
            return False  # backoff elapsed → allow one attempt
        # A config/.env change since the failure allows one attempt.
        return not (
            self._disk_signature_fn is not None
            and self._disk_signature_fn() != self._suppressed_sig
        )

    async def _refresh(self, categories: Collection[str] | None = None) -> None:
        """Run one refresh cycle: sync load → reconcile → suite → cache.

        With *categories* only those run and their rows replace the cached ones
        (a re-probe); without, the report is replaced whole.

        On any failure of the sync or suite phase the cache is left serving. A
        cancellation (teardown) propagates untouched — only ``Exception`` is
        handled here, never ``CancelledError``.
        """
        events_before = offload.abandoned_count()
        try:
            loaded = await offload.run_sync(
                self._loader.load, timeout_s=self._settings.suite_timeout_s
            )
        except TimeoutError:
            logger.warning(
                "health refresh: sync config phase exceeded %.1fs; serving cached report",
                self._settings.suite_timeout_s,
            )
            self._evaluate_breaker(events_before)
            return
        except Exception:
            logger.warning(
                "health refresh: sync config phase failed; serving cached report", exc_info=True
            )
            self._evaluate_breaker(events_before)
            return

        self._settings = loaded.settings
        if categories is not None:
            # Only what the suite can run: a notice or plugin-diagnostic row
            # names a category no record executes, and is kept as cached.
            categories = {r.name for r in loaded.records} & set(categories)
            if not categories:
                self._evaluate_breaker(events_before)
                return
        try:
            # Shared poll cycle: reconcile → suite → append plugin/notice rows.
            report = await run_poll_refresh(
                loaded, self._lifecycle, categories=categories, fold=False
            )
        except Exception:
            logger.warning("health refresh: suite run failed; serving cached report", exc_info=True)
            self._evaluate_breaker(events_before)
            return

        now = self._wall()
        if categories is None or self._report is None:
            self._report_full_at(now)
            self._cached_at = self._clock()
        for row in report.results:
            self._probed_at[row.name] = now
        if categories is not None and self._report is not None:
            report = self._merged(report, set(categories))
        self._report = report
        self._expanded = loaded.expanded
        self._config_ok = loaded.config_ok
        self._evaluate_breaker(events_before)

    def _report_full_at(self, now: float) -> None:
        self._scanned_at = now
        self._probed_at.clear()

    def _merged(self, part: CheckReport, categories: set[str]) -> CheckReport:
        """The cached report with *categories*' rows replaced by *part*'s, in place."""
        assert self._report is not None
        fresh = list(part.results)
        merged: list[Any] = []
        inserted = False
        for row in self._report.results:
            if row.category in categories:
                if not inserted:
                    merged.extend(fresh)
                    inserted = True
                continue
            merged.append(row)
        if not inserted:
            merged.extend(fresh)
        return CheckReport(
            results=merged,
            elapsed_ms=self._report.elapsed_ms,
            deadline_hit=self._report.deadline_hit,
        )

    # -- circuit breaker --------------------------------------------------------

    def _evaluate_breaker(self, events_before: int) -> None:
        """Trip or clear the breaker from the live abandoned-thread count."""
        events_after = offload.abandoned_count()
        if events_after > events_before:
            logger.warning(
                "health refresh: %d worker thread(s) abandoned this cycle (%d total)",
                events_after - events_before,
                events_after,
            )
        if offload.abandoned_alive_count() > 0:
            self._trip_breaker()
        else:
            self._clear_breaker()

    def _trip_breaker(self) -> None:
        alive = offload.abandoned_alive_count()
        self._breaker_trips += 1
        backoff_s = self._settings.interval_s * self._backoff_factor
        self._suppressed_until = self._clock() + backoff_s
        self._suppressed_sig = self._disk_signature_fn() if self._disk_signature_fn else None
        logger.warning(
            "health refresh: %d abandoned worker thread(s) still alive; suppressing refreshes "
            "for ~%.0fs (trip #%d) until config.yml/.env changes",
            alive,
            backoff_s,
            self._breaker_trips,
        )

    def _clear_breaker(self) -> None:
        if self._suppressed_until is not None:
            logger.info("health refresh: worker threads drained; resuming normal cadence")
        self._suppressed_until = None
        self._suppressed_sig = None
        self._breaker_trips = 0

    # -- default disk-change probe ----------------------------------------------

    def _default_disk_signature(self) -> tuple[Any, ...]:
        """Stat ``config.yml`` and the deployment's env chain for the breaker fast-path."""
        return _resolve_disk_signature(self._config_path)
