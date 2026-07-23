"""Cache + single-flight refresh context backing the stdio health MCP server.

The stdio health server answers an agent's ``poll`` tool by serving a cached
health report and, only when that cache is invalid, running exactly one poll
suite behind a single-flight lock. Unlike the web engine
(:class:`osprey.interfaces.health.engine.HealthCheckEngine`), which is
loop-affine and refresh-*ahead* (a browser poll never blocks on a suite run),
this context is refresh-*inline*: the tool caller awaits the refresh. Concurrent
callers coalesce onto the one in-flight run and then serve its fresh snapshot,
so two poll suites never run at once.

Composition (mirrors the web engine's building blocks):

* One :class:`~osprey.health.loader.HealthConfigLoader` owns the
  synchronous config-load phase (config-path resolution, ``.env``/``config.yml``
  mtime gating, merged-record assembly, and the degrade-to-default-settings
  contract). It is driven off the event loop through
  :func:`osprey.health.offload.run_sync` so a wedged plugin import cannot block
  process exit.
* One :class:`~osprey.health.lifecycle.HealthRuntimeLifecycle` owns
  the single connector runtime: :meth:`~HealthRuntimeLifecycle.reconcile` is
  called each refresh (re-snapshotting the runtime before first connect, and
  surfacing a restart-notice row after) and :meth:`~HealthRuntimeLifecycle.shutdown`
  performs the bounded connector teardown.
* One :func:`osprey.health.refresh.run_poll_refresh` per refresh — the poll
  cycle shared with the web engine (reconcile → ``full=False`` /
  ``categories=None`` suite → append plugin/notice rows). The cache always holds
  the full poll suite so any caller's category selection can be satisfied
  downstream by filtering the returned report; category selection never re-runs
  the suite.

Validity, wedge breaker, and single-flight are the three invariants:

1. **Validity.** A snapshot is valid iff its age (``clock() - cached_at``) does
   not exceed ``interval_s`` *and* the disk signature (``config.yml`` + ``.env``
   ``(mtime_ns, size)``) is unchanged since the snapshot was taken. A changed
   signature forces a refresh regardless of age.
2. **Wedge breaker.** While an abandoned worker thread is still alive
   (:func:`osprey.health.offload.abandoned_alive_count` ``> 0``) and the disk
   signature has not changed since suppression began, refreshes are suppressed:
   an existing snapshot is served flagged ``refresh_suppressed=True``; a cold
   context (no snapshot) raises :class:`HealthRefreshSuppressedError` for the
   tool layer to convert. A signature change permits exactly one refresh attempt
   while wedged (the disk-change escape), after which — if still wedged — the new
   signature is re-anchored and suppression resumes.
3. **Single-flight.** All refreshing happens under one :class:`asyncio.Lock`.
   Validity is re-checked after the lock is acquired, so a caller that queued
   behind an in-flight refresh serves that refresh's fresh snapshot instead of
   running a redundant second suite.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.health import offload
from osprey.health.config import parse_health_config
from osprey.health.lifecycle import HealthRuntimeLifecycle
from osprey.health.loader import HealthConfigLoader, LoadedHealthConfig
from osprey.health.models import CheckReport
from osprey.health.refresh import run_poll_refresh
from osprey.health.signatures import disk_signature as _resolve_disk_signature

logger = logging.getLogger("osprey.mcp_server.health.server_context")


class HealthRefreshSuppressedError(RuntimeError):
    """A refresh was required but suppressed by the wedge breaker with no cache.

    Raised by :meth:`HealthServerContext.get_poll_report` only on the cold path:
    a persistently-wedged suite (an abandoned worker thread still running) forbids
    a fresh run and there is no prior snapshot to serve. The tool layer converts
    this into an error envelope via its ``make_error`` helper — the context never
    calls ``os._exit``.

    Attributes:
        wedged_count: The number of abandoned worker threads still alive when the
            refresh was suppressed.
    """

    def __init__(self, wedged_count: int) -> None:
        self.wedged_count = wedged_count
        super().__init__(
            f"health refresh suppressed: {wedged_count} abandoned worker thread(s) "
            f"still alive and no cached report to serve"
        )


@dataclass(frozen=True)
class _Snapshot:
    """One cached poll cycle. Exactly one of these is held at a time.

    ``signature`` is the disk signature captured at the moment the snapshot was
    produced; validity compares the *current* signature to it.
    """

    loaded: LoadedHealthConfig
    report: CheckReport
    cached_at: float
    signature: Any


@dataclass(frozen=True)
class PollReportResult:
    """Everything the poll tool needs to build its response envelope.

    Attributes:
        loaded: The config inputs the cached ``report`` was produced from. The
            tool layer reads ``loaded.records`` to validate a requested category
            set and ``loaded.settings`` / ``loaded.config_ok`` for the envelope;
            ``loaded`` is always present (never ``None``).
        report: The full poll-suite :class:`~osprey.health.models.CheckReport`
            (``full=False``, all categories). Plugin-load diagnostics
            (``loaded.extra_rows``) and any lifecycle restart-notice rows are
            already appended to ``report.results``. Category selection is the
            tool layer's job: filter ``report.results`` by ``result.category``.
        cached_at: The monotonic clock reading when the suite was run.
        cached: ``True`` when this call served an existing snapshot without
            running the suite; ``False`` when this call ran a fresh suite.
        age_s: ``clock() - cached_at`` at serve time (``0.0`` for a fresh run).
        refresh_suppressed: ``True`` when a stale snapshot was served because the
            wedge breaker suppressed the refresh this call would otherwise have
            run.
    """

    loaded: LoadedHealthConfig
    report: CheckReport
    cached_at: float
    cached: bool
    age_s: float
    refresh_suppressed: bool


class HealthServerContext:
    """Owns the cached poll report and the single-flight refresh that produces it.

    A single instance is created per stdio health server process via
    :func:`initialize_server_context` and reached from tools via
    :func:`get_server_context`.

    Args:
        config_path: Explicit ``config.yml`` path, or ``None`` to resolve like the
            CLI (``OSPREY_CONFIG`` env, else ``./config.yml``). Forwarded to the
            loader and used to compute the default disk signature.
        clock: Monotonic time source for cache-age accounting. Injectable for
            tests; defaults to :func:`time.monotonic`.
        disk_signature: Override for the ``config.yml`` + ``.env`` change probe.
            Injectable for tests; defaults to stat-ing both files.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        disk_signature: Callable[[], Any] | None = None,
    ) -> None:
        cfg_path = Path(config_path) if config_path is not None else None
        self._loader = HealthConfigLoader(cfg_path)
        self._lifecycle = HealthRuntimeLifecycle(restart_hint="restart the health MCP server")
        self._config_path = cfg_path
        self._clock = clock
        self._disk_signature_fn = disk_signature or self._default_disk_signature

        self._lock = asyncio.Lock()
        self._snapshot: _Snapshot | None = None
        # Cadence settings; replaced by each successful load. Degraded config
        # yields the framework defaults (interval_s=60, suite_timeout_s=30).
        self._settings = parse_health_config(None)
        # The disk signature under which we are currently suppressing refreshes,
        # or ``None`` when not wedged. Anchors the disk-change escape.
        self._wedge_sig: Any = None
        self._shutdown_done = False

    # -- public surface ---------------------------------------------------------

    @property
    def lifecycle(self) -> HealthRuntimeLifecycle:
        """The connector-runtime lifecycle (exposed for server loop binding)."""
        return self._lifecycle

    @property
    def loader(self) -> HealthConfigLoader:
        """The signature-gated synchronous config loader.

        Exposed read-only so the approval-gated full-tier tool can run the same
        ``.env``/``config.yml`` load the poll path uses without touching the
        cached poll snapshot. The full tool drives this off the event loop via
        :func:`osprey.health.offload.run_sync` and keeps the result private to
        its own call — it never reads or writes :attr:`_snapshot`.
        """
        return self._loader

    async def get_poll_report(self, categories: Iterable[str] | None = None) -> PollReportResult:
        """Return the poll-tier report, refreshing it inline only when required.

        The cache always holds the full poll suite; ``categories`` is accepted for
        call-site symmetry with the tool contract but does **not** change what is
        run or cached — the tool layer filters ``result.report.results`` against
        ``result.loaded.records``. This keeps the single-flight cache consistent
        for concurrent callers regardless of their category selection.

        Behaviour:

        * A valid snapshot (fresh age and unchanged disk signature) is served
          immediately with ``cached=True, refresh_suppressed=False``.
        * Otherwise, if the wedge breaker is suppressing refreshes: an existing
          snapshot is served with ``refresh_suppressed=True``; with no snapshot a
          :class:`HealthRefreshSuppressedError` is raised.
        * Otherwise a refresh runs under the single-flight lock. A caller that
          queued behind an in-flight refresh re-checks validity on acquiring the
          lock and serves the just-produced snapshot rather than re-running.

        Args:
            categories: Advisory category selection; see above. Ignored for the
                cache/run decision.

        Returns:
            A :class:`PollReportResult` carrying the snapshot and serve flags.

        Raises:
            HealthRefreshSuppressedError: Cold path only — a refresh was required
                but suppressed by the wedge breaker with no snapshot to serve.
        """
        sig = self._disk_signature_fn()

        snapshot = self._snapshot
        if snapshot is not None and self._is_valid(snapshot, sig):
            return self._serve(snapshot, refreshed=False, refresh_suppressed=False)

        if self._is_suppressed(sig):
            if snapshot is not None:
                return self._serve(snapshot, refreshed=False, refresh_suppressed=True)
            raise HealthRefreshSuppressedError(offload.abandoned_alive_count())

        async with self._lock:
            # A refresh may have completed while we awaited the lock; re-check
            # validity against a fresh signature so we never run a redundant suite.
            sig = self._disk_signature_fn()
            snapshot = self._snapshot
            if snapshot is not None and self._is_valid(snapshot, sig):
                return self._serve(snapshot, refreshed=False, refresh_suppressed=False)
            snapshot = await self._refresh(sig)

        return self._serve(snapshot, refreshed=True, refresh_suppressed=False)

    async def shutdown(self) -> None:
        """Tear down the connector runtime exactly once (safe to call twice).

        Delegates to :meth:`HealthRuntimeLifecycle.shutdown`, which cancels any
        in-flight refresh, disconnects the connector on its owning loop, and
        unregisters the process-exit hook. Used by the server lifespan and by
        :func:`reset_server_context`'s async test teardown.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        await self._lifecycle.shutdown()

    # -- serve / validity / suppression -----------------------------------------

    def _serve(
        self, snapshot: _Snapshot, *, refreshed: bool, refresh_suppressed: bool
    ) -> PollReportResult:
        """Build the result envelope for a snapshot at the current clock."""
        age_s = 0.0 if refreshed else max(0.0, self._clock() - snapshot.cached_at)
        return PollReportResult(
            loaded=snapshot.loaded,
            report=snapshot.report,
            cached_at=snapshot.cached_at,
            cached=not refreshed,
            age_s=age_s,
            refresh_suppressed=refresh_suppressed,
        )

    def _is_valid(self, snapshot: _Snapshot, sig: Any) -> bool:
        """Whether *snapshot* may be served without refreshing."""
        if sig != snapshot.signature:
            return False
        age = self._clock() - snapshot.cached_at
        return age <= self._settings.interval_s

    def _is_suppressed(self, sig: Any) -> bool:
        """Whether the wedge breaker forbids a refresh right now.

        Suppression holds while an abandoned worker thread is still alive and the
        disk signature has not changed since suppression began. The first wedged
        observation anchors the signature; a later differing signature releases
        suppression for exactly one attempt (the caller then refreshes, and
        :meth:`_evaluate_wedge` re-anchors if still wedged).
        """
        if offload.abandoned_alive_count() <= 0:
            self._wedge_sig = None
            return False
        if self._wedge_sig is None:
            # Wedge observed but never refreshed-through: anchor and suppress.
            self._wedge_sig = sig
            return True
        if sig != self._wedge_sig:
            return False  # disk-change escape: allow exactly one refresh attempt
        return True

    def _evaluate_wedge(self, sig: Any) -> None:
        """Re-anchor or clear the wedge signature after a refresh.

        Called with the signature the just-completed refresh ran under. If an
        abandoned thread is still alive the signature is re-anchored so a repeat
        call at the same signature is suppressed again; otherwise the breaker
        clears.
        """
        self._wedge_sig = sig if offload.abandoned_alive_count() > 0 else None

    # -- refresh ----------------------------------------------------------------

    async def _refresh(self, sig: Any) -> _Snapshot:
        """Run one full refresh cycle: sync load → poll cycle → cache.

        Must be called holding :attr:`_lock`. Shares the poll cycle
        (:func:`osprey.health.refresh.run_poll_refresh`) with the web engine but
        awaits it inline and always stores the resulting snapshot.
        """
        loaded = await offload.run_sync(self._loader.load, timeout_s=self._settings.suite_timeout_s)
        self._settings = loaded.settings

        report = await run_poll_refresh(loaded, self._lifecycle)

        snapshot = _Snapshot(
            loaded=loaded,
            report=report,
            cached_at=self._clock(),
            signature=sig,
        )
        self._snapshot = snapshot
        self._evaluate_wedge(sig)
        return snapshot

    # -- default disk-change probe ----------------------------------------------

    def _default_disk_signature(self) -> tuple[Any, Any]:
        """Stat ``config.yml`` and its sibling ``.env`` for the validity probe."""
        return _resolve_disk_signature(self._config_path)


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors control_system.get_server_context())
# ---------------------------------------------------------------------------

_context: HealthServerContext | None = None


def get_server_context() -> HealthServerContext:
    """Return the health MCP server context singleton.

    Raises:
        RuntimeError: If :func:`initialize_server_context` has not been called.
    """
    if _context is None:
        raise RuntimeError(
            "Health server context not initialized. Call initialize_server_context() first."
        )
    return _context


def initialize_server_context(
    *,
    config_path: str | Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    disk_signature: Callable[[], Any] | None = None,
) -> HealthServerContext:
    """Create and store the health MCP server context singleton.

    Construction is cheap and side-effect-free (no config load, no connector);
    the first :meth:`HealthServerContext.get_poll_report` performs the work.
    """
    global _context
    _context = HealthServerContext(
        config_path=config_path, clock=clock, disk_signature=disk_signature
    )
    return _context


def reset_server_context() -> None:
    """Drop the singleton (for testing).

    Synchronous, to match the control_system idiom and the autouse test reset.
    Async teardown of a live connector is the caller's responsibility: await
    ``get_server_context().shutdown()`` before calling this from an async test
    that constructed a real connector.
    """
    global _context
    _context = None
