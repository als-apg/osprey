"""The poll-tier refresh cycle shared by the long-lived health surfaces.

Both persistent poll surfaces — the web sidecar's refresh engine
(:class:`osprey.interfaces.health.engine.HealthCheckEngine`) and the stdio MCP
server's context
(:class:`osprey.mcp_server.health.server_context.HealthServerContext`) — turn
one :class:`~osprey.health.loader.LoadedHealthConfig` into a poll-tier
:class:`~osprey.health.models.CheckReport` by the identical sequence: reconcile
the connector runtime against the load's ``control_system`` snapshot, run the
``full=False`` suite against it, and append the unfiltered plugin-diagnostic and
restart-notice rows. This module is the single owner of that sequence; the two
surfaces differ only in how they *schedule* it (refresh-ahead vs. inline
single-flight) and in their error/accounting policy, which stays in each caller.

The row-append order is load-bearing and deliberately single-sourced here:
plugin ``extra_rows`` first, then the lifecycle's ``notice_rows``, both after the
suite results so any downstream category filter treats them uniformly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.health.runner import run_health_suite

if TYPE_CHECKING:
    from osprey.health.lifecycle import HealthRuntimeLifecycle
    from osprey.health.loader import LoadedHealthConfig
    from osprey.health.models import CheckReport


async def run_poll_refresh(
    loaded: LoadedHealthConfig,
    lifecycle: HealthRuntimeLifecycle,
) -> CheckReport:
    """Produce the poll-tier report for one loaded config.

    Reconciles *lifecycle* against the load's ``control_system`` snapshot, runs
    the poll suite (``full=False``, all categories) against the resulting
    runtime, then appends the unfiltered plugin-diagnostic rows
    (``loaded.extra_rows``) followed by any restart-notice rows the reconcile
    produced — in that order.

    The caller owns scheduling and error/accounting policy: this helper neither
    catches suite/load failures nor mutates caller state, so a failure propagates
    for the caller to translate (serve-cache-and-evaluate-breaker for the web
    engine; propagate-under-lock for the MCP context).

    Raises:
        RuntimeError: if the reconcile produced no runtime — a defensive guard,
            since :meth:`~HealthRuntimeLifecycle.reconcile` always constructs one.
    """
    notice_rows = lifecycle.reconcile(loaded.expanded)
    runtime = lifecycle.runtime
    if runtime is None:  # defensive: reconcile always constructs a runtime
        raise RuntimeError("health lifecycle produced no runtime after reconcile")

    report = await run_health_suite(
        loaded.records,
        runtime=runtime,
        config=loaded.expanded,
        full=False,
        categories=None,
        suite_timeout_s=loaded.settings.suite_timeout_s,
        on_demand_timeout_s=loaded.settings.on_demand_timeout_s,
    )
    report.results.extend(loaded.extra_rows)
    report.results.extend(notice_rows)
    return report
