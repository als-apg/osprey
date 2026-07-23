"""MCP tool: health_check — cached poll-tier facility health check."""

from __future__ import annotations

import json
import logging

from osprey.health.models import CheckReport
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.health.server import mcp
from osprey.mcp_server.health.server_context import (
    HealthRefreshSuppressedError,
    get_server_context,
)

logger = logging.getLogger("osprey.mcp_server.health.tools.health_check")


@mcp.tool()
async def health_check(categories: list[str] | None = None) -> str:
    """Run the cached poll tier of the facility health suite.

    The poll tier is the always-on, read-only health picture of the facility:
    config load, connector reachability, provider canaries, archiver freshness,
    plugin diagnostics, and similar fast checks. It never writes to hardware and
    never blocks on operator approval, so call it freely to triage the state of
    the facility.

    Escalation to the full tier:
        Deeper, slower, or side-effect-bearing checks belong to the ``on_demand``
        tier and are *not* run here — they appear in this report as ``skip`` rows.
        To actually run them, call ``health_check_full``, which is approval-gated
        (a human must approve the run). Ignore any shell/CLI command mentioned in
        a skip row's hint or message text: those hints target a human at a
        terminal, not you. Your only escalation path is the ``health_check_full``
        tool.

    Latency and caching:
        Results are served from a process cache. The first call — and the first
        call after ``config.yml`` or ``.env`` is edited — runs the full poll
        suite inline and may take up to ``suite_timeout_s`` (default 30 s); later
        calls within the refresh interval return immediately from cache. Read the
        ``cached`` and ``age_s`` fields to tell a fresh run from a cached serve.

    Args:
        categories: Optional subset of health categories to report on. Each name
            must be a known category (an unknown name is rejected with the list
            of valid names). When omitted, every category is reported, including
            plugin-diagnostic rows. When a subset is given, plugin-diagnostic
            rows are omitted; note that ``elapsed_ms`` and ``deadline_hit`` in the
            response then describe the underlying *unfiltered* suite run, not just
            the selected rows.

    Returns:
        A JSON object with the locked report shape — ``summary``, ``ok``,
        ``warnings``, ``errors``, ``skips``, ``total``, ``elapsed_ms``,
        ``deadline_hit``, ``results`` — plus ``cached`` (served from cache vs.
        freshly run), ``age_s`` (age of the served snapshot in seconds), and
        ``refresh_suppressed`` (a stale snapshot was served because a wedged
        worker thread blocked the refresh).
    """
    ctx = get_server_context()

    try:
        result = await ctx.get_poll_report(categories)
    except HealthRefreshSuppressedError as exc:
        make_error(
            "health_suppressed",
            f"{exc.wedged_count} wedged worker threads; no cached report",
            suggestions=[
                "Retry shortly; the wedge clears once the abandoned worker exits "
                "or config.yml/.env changes on disk.",
            ],
            details={"wedged_count": exc.wedged_count},
        )

    valid_names = {r.name for r in result.loaded.records}

    if categories is not None:
        unknown = [name for name in categories if name not in valid_names]
        if unknown:
            plural = "ies" if len(unknown) > 1 else "y"
            make_error(
                "unknown_category",
                f"Unknown health categor{plural}: {', '.join(unknown)}. "
                f"Valid categories: {', '.join(sorted(valid_names))}",
                suggestions=["Call health_check with no categories to see every category."],
                details={"unknown": unknown, "valid": sorted(valid_names)},
            )

    if categories is None:
        # Unfiltered serve: the snapshot's report already carries plugin
        # extra rows and any restart-notice rows — serve it verbatim.
        report = result.report
    else:
        report = _filter_report(result, requested=set(categories))

    envelope = report.to_dict()
    envelope["cached"] = result.cached
    envelope["age_s"] = result.age_s
    envelope["refresh_suppressed"] = result.refresh_suppressed
    return json.dumps(envelope, default=str)


def _filter_report(result, *, requested: set[str]) -> CheckReport:
    """Scope the cached report to *requested* categories (CLI-parity rules).

    Selects rows whose ``category`` is requested, excludes plugin-diagnostic rows
    (they render only on unfiltered serves), and force-includes ``configuration``
    rows when the config load failed — a global fault must never be scoped out.
    The rebuilt report carries the underlying unfiltered run's ``elapsed_ms`` and
    ``deadline_hit``; counts recompute from the selected rows.
    """
    effective = set(requested)
    if not result.loaded.config_ok:
        effective.add("configuration")

    extra_ids = {id(row) for row in result.loaded.extra_rows}
    selected = [
        row
        for row in result.report.results
        if row.category in effective and id(row) not in extra_ids
    ]
    return CheckReport(
        results=selected,
        elapsed_ms=result.report.elapsed_ms,
        deadline_hit=result.report.deadline_hit,
    )
