"""Archiver-freshness health probe — is the archiver actually accumulating data?

Verifies the deployment's archiver is both *reachable* and *live*: it queries a
short trailing window for a derived or stated canary channel through the suite's
shared archive (obtained from the
:class:`~osprey.health.runtime.HealthRuntime` on the probe context via
``ctx.runtime.get_archive(spec)``) and grades the age of the newest archived
sample against a threshold. This generalizes the single most valuable check on a
facility dashboard: the archiver *UI being up* is not the same as the archiver
*archiving* — a wedged ingester leaves the web front-end reachable while data
silently stops flowing, and only the age of the newest sample distinguishes the
two.

The config is resolved with the same precedence as
:mod:`~osprey.health.probes.provider_canary`: an explicit per-run ``ctx.config``
(the web surface) is authoritative when present, else the global config
singleton (CLI/standalone). The archive probed is the first target's of the
deployment's archive row (:func:`~osprey_connectors.service_targets.service_targets`):
a health run has no session, so it probes the one archive the deployment has,
whichever target holds it. A declared ``archiver_freshness`` check with **no**
``archiver:`` block configured is a real misconfiguration and grades ``error``
— not a silent pass.

Grading:

* no ``archiver:`` block configured → ``error`` (misconfiguration);
* a block that serves no target, an archive whose settings are refused, or an
  archive that gives no answer → ``error`` whose ``details`` carries the reason;
* a sample is found and its age is ``<= max_age_s`` → ``ok``, message
  ``"Fresh (newest sample 42 s old)"`` and ``value`` the sample reading;
* a sample is found but older than ``max_age_s`` → ``warning`` reporting the
  observed age (a reachable-but-stale archiver);
* no samples in the query window → ``warning`` (the newest sample, if any, is
  older than the whole trailing window — indistinguishable from stale, and
  graded the same).

The staleness verdict is deliberately fixed at ``warning``: a facility tunes the
*timeout* severity through the generic ``timeout_status`` check key (handled by
the runner), and this probe invents no staleness-specific severity param.

``timeout_s`` bounds the ``history`` call directly; the runner's outer
``wait_for`` allows a small margin so the archive's own timeout fires first.
The trailing window is derived from ``max_age_s`` to be comfortably wider than
the threshold, so a moderately stale sample's true age is still observable rather
than collapsing to an empty-window warning. The age comparison is done entirely
in UTC, which ``history``'s ``datetime64[ns, UTC]`` timestamp column guarantees.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status
from osprey.services.pv_directory._connectors import service_table
from osprey_connectors.service_answers import Unavailable

if TYPE_CHECKING:
    import pandas as pd

    from osprey.health.probes import ProbeContext


async def run(spec: Mapping[str, Any], ctx: ProbeContext) -> CheckResult:
    """Query the archiver for a canary channel and grade the newest sample's age.

    Args:
        spec: Parsed check parameters. Recognized keys:

            * ``channel`` (str, required): the canary channel to query. A missing
              or empty value is a config-style ``error`` (the check cannot run).
            * ``max_age_s`` (float): freshness threshold in seconds (default
              ``600``); a newest sample older than this grades ``warning``.
            * ``name`` (str): result name (default ``"archiver_freshness"``).
            * ``category`` (str): result category (default
              ``"archiver_freshness"``).
            * ``timeout_s`` (float): archiver query timeout in seconds (default
              ``10.0``), passed to ``history``.
        ctx: Shared per-run context. ``ctx.runtime.get_archive(spec)`` yields
            the suite's single lazily-constructed archive; the config is read
            from ``ctx.config`` when present, else the global config singleton.

    Returns:
        A :class:`~osprey.health.models.CheckResult` with ``latency_ms`` set on
        every outcome where a query was attempted and, when a sample was found,
        ``value`` carrying the reading.
    """
    name = str(spec.get("name", "archiver_freshness"))
    category = str(spec.get("category", "archiver_freshness"))
    channel = str(spec.get("channel", ""))
    max_age_s = float(spec.get("max_age_s", 600.0))
    timeout_s = float(spec.get("timeout_s", 10.0))

    if not channel:
        return CheckResult(
            name,
            category,
            Status.ERROR,
            "archiver_freshness check requires a 'channel' parameter",
        )

    config = _config(ctx.config)
    archiver_block = config.get("archiver")
    if not isinstance(archiver_block, Mapping) or not archiver_block:
        return CheckResult(
            name,
            category,
            Status.ERROR,
            "no archiver configured (empty or missing 'archiver:' config block)",
        )

    # Query a trailing window wider than the threshold so a moderately stale
    # sample's true age is still captured rather than collapsing to an empty
    # window. Floor at 60 s so a tiny max_age_s still spans a usable window.
    window_s = max(2.0 * max_age_s, 60.0)
    now = datetime.now(UTC)
    start = now - timedelta(seconds=window_s)

    # The first target's archive: a health run has no session to pick one.
    archive_spec = next((s for s in service_table(config).archive.values() if s is not None), None)
    if archive_spec is None:
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"archiver unreachable querying {channel}",
            details="this deployment's archiver block serves no control target",
        )

    t0 = perf_counter()
    try:
        archive = await ctx.runtime.get_archive(archive_spec)
    except Exception as exc:  # a block the archive refuses is an error result
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"archiver unreachable querying {channel}",
            latency_ms=(perf_counter() - t0) * 1000.0,
            details=str(exc),
        )
    frame = await archive.history([channel], start_at=start, end_at=now, timeout_s=timeout_s)
    if isinstance(frame, Unavailable):
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"archiver unreachable querying {channel}",
            latency_ms=(perf_counter() - t0) * 1000.0,
            details=frame.reason,
        )

    latency_ms = (perf_counter() - t0) * 1000.0
    newest = _newest_sample(frame, channel)
    if newest is None:
        return CheckResult(
            name,
            category,
            Status.WARNING,
            f"No samples for {channel} in the last {window_s:g} s",
            latency_ms=latency_ms,
        )

    newest_ts, value = newest
    age_s = max(0.0, (now - newest_ts).total_seconds())
    value_str = str(value)

    if age_s <= max_age_s:
        return CheckResult(
            name,
            category,
            Status.OK,
            f"Fresh (newest sample {age_s:.0f} s old)",
            value=value_str,
            latency_ms=latency_ms,
        )
    return CheckResult(
        name,
        category,
        Status.WARNING,
        f"Stale (newest sample {age_s:.0f} s old, threshold {max_age_s:g} s)",
        value=value_str,
        latency_ms=latency_ms,
    )


def _config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the full config the archive is resolved from, or ``{}`` when unavailable.

    Resolution precedence mirrors :mod:`~osprey.health.probes.provider_canary`:
    an explicit per-run *config* (from ``ctx.config``, e.g. the web surface) is
    authoritative when present — the global singleton is never consulted in that
    case — otherwise the config falls back to the one this process reads
    (:func:`~osprey_connectors.config.current_config`). The whole config, not the
    ``archiver:`` block: which target the archive serves turns on whether the
    deployment built a simulator. Any failure to load config is swallowed to an
    empty config, which the caller grades as a misconfiguration.
    """
    if config is not None:
        return config
    try:
        from osprey_connectors.config import current_config

        return current_config().raw_tree()
    except Exception:  # config unavailability degrades to no config
        return {}


def _newest_sample(frame: pd.DataFrame, channel: str) -> tuple[datetime, object] | None:
    """Return the ``(timestamp, value)`` of the newest non-null sample, or ``None``.

    Filters the long-format archiver frame to rows for ``channel``, drops null
    values, and returns the row with the maximum timestamp; ``None`` means the
    channel has no non-null samples in the window. The timestamp is a UTC-aware
    :class:`pandas.Timestamp` (itself a ``datetime``), deliberately not passed
    through ``to_pydatetime()``, which discards nanoseconds and warns. A frame
    missing the long-format columns is a connector bug and is left to raise.
    """
    sub = frame.loc[frame["address"] == channel].dropna(subset=["value"])
    if sub.empty:
        return None
    row = sub.iloc[sub["timestamp"].argmax()]
    return row["timestamp"], row["value"]
