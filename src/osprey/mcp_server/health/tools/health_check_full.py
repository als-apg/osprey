"""MCP tool: health_check_full — approval-gated full-tier facility health check.

Unlike ``health_check`` (which serves a cached poll snapshot behind a
single-flight refresh), this tool always runs a fresh full-tier suite per
invocation and never touches the poll snapshot. It mirrors the CLI's
per-invocation model: a private config load, a private
:class:`~osprey.health.runtime.HealthRuntime` opened as an async context manager
(shut down even on error), and one ``full=True`` suite run. Operator approval —
not the poll path's wedge breaker — is the rate limiter, so this tool is never
suppressed and its result is never cached.
"""

import json
import logging

from osprey.health import offload
from osprey.health.config import DEFAULT_SUITE_TIMEOUT_S
from osprey.health.runner import run_health_suite
from osprey.health.runtime import HealthRuntime
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.health.server import mcp
from osprey.mcp_server.health.server_context import get_server_context

logger = logging.getLogger("osprey.mcp_server.health.tools.health_check_full")


@mcp.tool()
async def health_check_full(categories: list[str] | None = None) -> str:
    """Run the approval-gated full tier of the facility health suite.

    This is the escalation path for the on_demand checks that ``health_check``
    reports as ``skip`` rows (live model chat completions, pinned-CLI download
    verification, and other costly probes). Because those probes are expensive,
    running this tool requires operator approval.

    Expect the call to take roughly the sum of the selected on_demand
    categories' time budgets — bounded by the runner's on_demand deadline
    (tunable via ``health.on_demand_timeout_s`` in ``config.yml``) plus the poll
    tier's budget. The suite always runs fresh: this tool never serves a cached
    report and is never suppressed by the poll path's wedge breaker.

    Args:
        categories: Optional subset of health categories to run and report on.
            When omitted, every available category runs (including plugin
            diagnostic rows). Unknown category names are rejected.

    Returns:
        JSON summary of the full-tier health results. The envelope carries the
        locked report wire shape plus ``cached: false``, ``age_s: 0``, and
        ``refresh_suppressed: false`` — this tier is always fresh.
    """
    ctx = get_server_context()

    # Private, per-call config load — the same signature-gated load the poll
    # path uses, but the result is NEVER stored on the context's poll snapshot.
    loaded = await offload.run_sync(ctx.loader.load, timeout_s=DEFAULT_SUITE_TIMEOUT_S)

    requested = list(dict.fromkeys(categories)) if categories else None
    if requested is not None:
        valid_names = {record.name for record in loaded.records}
        unknown = [name for name in requested if name not in valid_names]
        if unknown:
            plural = "ies" if len(unknown) > 1 else "y"
            make_error(
                "unknown_category",
                f"Unknown health categor{plural}: {', '.join(unknown)}. "
                f"Valid categories: {', '.join(sorted(valid_names))}",
                suggestions=["Call health_check with no categories to see every category."],
                details={"unknown": unknown, "valid": sorted(valid_names)},
            )

    # Private per-call runtime, opened as an async context manager so the
    # connector is torn down exactly once even if the suite raises.
    async with HealthRuntime(loaded.control_system) as runtime:
        report = await run_health_suite(
            loaded.records,
            runtime=runtime,
            config=loaded.expanded,
            full=True,
            categories=requested,
            suite_timeout_s=loaded.settings.suite_timeout_s,
            on_demand_timeout_s=loaded.settings.on_demand_timeout_s,
        )

    # Plugin-load diagnostic rows live on the loader result, not in the suite
    # output; the runner never appends them. Include them only for an unfiltered
    # run so they land exactly once and a filtered response excludes them.
    if requested is None:
        report.results.extend(loaded.extra_rows)

    envelope = report.to_dict()
    # This tier is always fresh and never suppressed — operator approval, not the
    # wedge breaker, is the rate limiter.
    envelope["cached"] = False
    envelope["age_s"] = 0
    envelope["refresh_suppressed"] = False
    return json.dumps(envelope, default=str)
