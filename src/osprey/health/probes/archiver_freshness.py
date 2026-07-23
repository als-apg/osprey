"""Archiver-freshness health probe — is the archiver actually accumulating data?

Verifies the deployment's archiver is both *reachable* and *live*: it queries a
short trailing window for a facility-declared canary channel through the suite's
shared archiver connector (obtained from the
:class:`~osprey.health.runtime.HealthRuntime` on the probe context via
``ctx.runtime.get_archiver(...)``) and grades the age of the newest archived
sample against a threshold. This generalizes the single most valuable check on a
facility dashboard: the archiver *UI being up* is not the same as the archiver
*archiving* — a wedged ingester leaves the web front-end reachable while data
silently stops flowing, and only the age of the newest sample distinguishes the
two.

The archiver config block (``type`` plus the per-type sub-block) is resolved
with the same precedence as :mod:`~osprey.health.probes.provider_canary`: an
explicit per-run ``ctx.config`` (the web surface) is authoritative when present,
else the global config singleton (CLI/standalone). A declared
``archiver_freshness`` check with **no** ``archiver:`` block configured is a real
misconfiguration and grades ``error`` — not a silent pass.

Grading:

* no ``archiver:`` block configured → ``error`` (misconfiguration);
* the connector is unreachable, or the query raises → ``error`` whose ``details``
  carries ``str(exc)``;
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

``timeout_s`` bounds the ``get_data`` call directly; the runner's outer
``wait_for`` allows a small margin so the connector's own timeout fires first.
The trailing window is derived from ``max_age_s`` to be comfortably wider than
the threshold, so a moderately stale sample's true age is still observable rather
than collapsing to an empty-window warning. Archiver timestamps that arrive
timezone-naive are read as UTC (the EPICS Archiver Appliance returns UTC-aware
timestamps; the age comparison is done entirely in UTC).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

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
              ``10.0``), passed to ``get_data``.
        ctx: Shared per-run context. ``ctx.runtime.get_archiver(block)`` yields
            the suite's single lazily-constructed archiver connector; the
            ``archiver:`` config *block* is read from ``ctx.config`` when present,
            else the global config singleton.

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

    archiver_block = _archiver_block(ctx.config)
    if not archiver_block:
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

    t0 = perf_counter()
    try:
        connector = await ctx.runtime.get_archiver(dict(archiver_block))
        frame = await connector.get_data(
            [channel],
            start,
            now,
            timeout=int(timeout_s),
        )
    except Exception as exc:  # noqa: BLE001 - any archiver failure becomes an error result
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"archiver unreachable querying {channel}",
            latency_ms=latency_ms,
            details=str(exc),
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


def _archiver_block(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ``archiver`` config block, or ``{}`` when unavailable.

    Resolution precedence mirrors :mod:`~osprey.health.probes.provider_canary`:
    an explicit per-run *config* (from ``ctx.config``, e.g. the web surface) is
    authoritative when present — the global singleton is never consulted in that
    case — otherwise the block falls back to the global singleton via
    :func:`~osprey.utils.config.get_config_value`. Any failure to load config is
    swallowed to an empty block, which the caller grades as a misconfiguration.
    """
    if config is not None:
        block = config.get("archiver")
        return block if isinstance(block, Mapping) else {}
    try:
        from osprey.utils.config import get_config_value

        block = get_config_value("archiver", {})
    except Exception:  # noqa: BLE001 - config unavailability degrades to no block
        return {}
    return block if isinstance(block, Mapping) else {}


def _newest_sample(frame: pd.DataFrame, channel: str) -> tuple[datetime, Any] | None:
    """Return the ``(timestamp, value)`` of the newest non-null sample, or ``None``.

    Reads the ``channel`` column of the archiver DataFrame (a datetime-indexed
    frame with one column per queried PV), drops nulls, and picks the row with
    the latest index. Returns ``None`` when the column is absent or holds no
    non-null samples in the window. The returned timestamp is normalized to a
    timezone-aware UTC :class:`~datetime.datetime`: a naive index timestamp is
    read as UTC, so the caller's age arithmetic never hits a naive/aware
    ``TypeError``.
    """
    if channel not in frame.columns:
        return None
    series = frame[channel].dropna()
    if series.empty:
        return None
    series = series.sort_index()
    raw_ts = series.index[-1]
    value = series.iloc[-1]

    stamp = raw_ts.to_pydatetime() if hasattr(raw_ts, "to_pydatetime") else raw_ts
    if not isinstance(stamp, datetime):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    else:
        stamp = stamp.astimezone(UTC)
    return stamp, value
