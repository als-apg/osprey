"""Channel-read health probe — one connector-mediated read, graded against bands.

Reads a single control-system channel through the suite's shared connector,
obtained from the :class:`~osprey.health.runtime.HealthRuntime` on the probe
context (``ctx.runtime.get_connector()``). The read is a plain
``connector.read_channel(address, timeout=timeout_s)`` — this probe never writes
and never subscribes; every I/O flows through the one connector the runtime owns.

Grading:

* any read failure (connection, timeout, invalid address) → ``error`` whose
  ``details`` carries ``str(exc)``;
* with ``expect`` set, the value must equal it (``error`` otherwise);
* otherwise, with ``ok_range`` / ``warn_range`` (inclusive numeric bands): a
  value outside ``warn_range`` is an ``error``, inside ``warn_range`` but outside
  ``ok_range`` a ``warning``, and inside ``ok_range`` an ``ok``. A non-numeric
  value with a band configured is an ``error``;
* with no ``expect`` and no bands, a successful read is ``ok`` (a liveness check).

``timeout_s`` is passed to ``read_channel`` explicitly: the runner's outer
``wait_for`` allows a small margin so the connector's own timeout fires first,
surfacing its better error message. ``latency_ms`` is measured on every outcome —
success, out-of-range, and read failure alike — from a single ``perf_counter``
taken just before the read. Units from ``ChannelValue.metadata.units`` are
appended to the reported value (e.g. ``"401.2 mA"``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.probes import ProbeContext


async def run(spec: Mapping[str, Any], ctx: ProbeContext) -> CheckResult:
    """Read one channel through the runtime's connector and grade the value.

    Args:
        spec: Parsed check parameters. Recognized keys:

            * ``address`` (str, required): the channel to read.
            * ``name`` (str): result name (default ``"channel_read"``).
            * ``category`` (str): result category (default ``"channel_read"``).
            * ``expect`` (Any): required exact value; mismatch is an error.
            * ``ok_range`` (``[lo, hi]``): inclusive numeric OK band.
            * ``warn_range`` (``[lo, hi]``): inclusive numeric band; outside it is
              an error.
            * ``timeout_s`` (float): read timeout in seconds (default ``5.0``),
              passed to ``read_channel``.
        ctx: Shared per-run context; ``ctx.runtime.get_connector()`` yields the
            suite's single lazily-constructed connector.

    Returns:
        A :class:`~osprey.health.models.CheckResult` with ``latency_ms`` set on
        every outcome and, when a value was read, ``value`` carrying the reading
        (with units).
    """
    name = str(spec.get("name", "channel_read"))
    category = str(spec.get("category", "channel_read"))
    address = str(spec.get("address", ""))
    timeout_s = float(spec.get("timeout_s", 5.0))
    expect = spec.get("expect")
    ok_range = spec.get("ok_range")
    warn_range = spec.get("warn_range")

    t0 = perf_counter()
    try:
        connector = await ctx.runtime.get_connector()
        channel = await connector.read_channel(address, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - any read failure becomes an error result
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"read {address} failed",
            latency_ms=latency_ms,
            details=str(exc),
        )

    latency_ms = (perf_counter() - t0) * 1000.0
    value = channel.value
    units = getattr(channel.metadata, "units", "") or ""
    value_str = f"{value} {units}".strip()

    if expect is not None:
        if value == expect:
            return CheckResult(
                name,
                category,
                Status.OK,
                f"{address} = {value_str}",
                value=value_str,
                latency_ms=latency_ms,
            )
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"{address} = {value_str}, expected {expect}",
            value=value_str,
            latency_ms=latency_ms,
        )

    if ok_range is not None or warn_range is not None:
        numeric = _as_number(value)
        if numeric is None:
            return CheckResult(
                name,
                category,
                Status.ERROR,
                f"{address} = {value_str} is not numeric",
                value=value_str,
                latency_ms=latency_ms,
            )
        if warn_range is not None and not _in_range(numeric, warn_range):
            lo, hi = _bounds(warn_range)
            return CheckResult(
                name,
                category,
                Status.ERROR,
                f"{address} = {value_str} outside warn range [{lo}, {hi}]",
                value=value_str,
                latency_ms=latency_ms,
            )
        if ok_range is not None and not _in_range(numeric, ok_range):
            lo, hi = _bounds(ok_range)
            return CheckResult(
                name,
                category,
                Status.WARNING,
                f"{address} = {value_str} outside ok range [{lo}, {hi}]",
                value=value_str,
                latency_ms=latency_ms,
            )

    return CheckResult(
        name,
        category,
        Status.OK,
        f"{address} = {value_str}",
        value=value_str,
        latency_ms=latency_ms,
    )


def _as_number(value: Any) -> float | None:
    """Coerce ``value`` to float for band comparison, or ``None`` if not numeric.

    ``bool`` is rejected: a control-system enum read back as ``True``/``False``
    should not be silently graded as ``1``/``0`` against a numeric band.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounds(rng: Sequence[Any]) -> tuple[float, float]:
    """Return the inclusive ``(lo, hi)`` bounds of a two-element range."""
    return float(rng[0]), float(rng[1])


def _in_range(value: float, rng: Sequence[Any]) -> bool:
    """Return whether ``value`` lies within the inclusive range ``rng``."""
    lo, hi = _bounds(rng)
    return lo <= value <= hi
