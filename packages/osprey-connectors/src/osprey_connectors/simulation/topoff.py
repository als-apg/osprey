"""The top-off sawtooth: a stored current that decays with a lifetime and refills on a period.

A ``topoff`` seed ``{ceiling, lifetime_h, period_s}`` describes a float
channel that starts each period at ``ceiling`` and decays exponentially with
the lifetime ``tau = lifetime_h * 3600`` seconds until the next refill. The
unit shape is a pure function of epoch seconds::

    s(t) = exp(-((t mod period_s) / tau))

so every channel with the same seed refills at the same instants, and a live
read and an archived sample at the same instant agree. Its mean over one
period is::

    m = (tau / period_s) * (1 - exp(-period_s / tau))

The channel HOLDS the level ``ceiling * m`` -- the value a limits band and
an archive's start state see -- and the sawtooth is MOTION around it: the
tooth ``level * (s(t) / m - 1)``, added on the level it rides, so at the held
level the reading is ``ceiling * s(t)``, and a level moved to zero carries no
tooth at all.

A ``topoff_lifetime`` seed ``{of}`` names a channel whose seed is ``topoff``
and holds that seed's ``lifetime_h`` as a constant: the lifetime readback that
produces the decay.

The module imports only the standard library, so the envelope rule and the
facility build share it with the texture model. :data:`SHAPE_VERSION` is a
digest of the shape constants; the archive seed fingerprint consumes it, so a
change to the shape re-seeds the stored history.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "DECAY_FORM",
    "LIFETIME_KEY",
    "REFILL_PHASE",
    "SECONDS_PER_HOUR",
    "SHAPE_CONSTANT_NAMES",
    "SHAPE_VERSION",
    "TOPOFF_KEY",
    "held_value",
    "period_mean",
    "shape",
    "shape_version",
    "tau_period",
    "tooth_depth",
    "topoff_record",
]

#: The seed key of a decaying, refilling channel.
TOPOFF_KEY = "topoff"
#: The seed key of a channel holding a ``topoff`` seed's lifetime.
LIFETIME_KEY = "topoff_lifetime"
#: Seconds per lifetime hour.
SECONDS_PER_HOUR = 3600.0
#: Form of the decay between refills; part of the shape digest.
DECAY_FORM = "exponential"
#: Where in a period the refill lands; part of the shape digest.
REFILL_PHASE = "period_start"

#: The constants :data:`SHAPE_VERSION` digests, in a fixed order.
SHAPE_CONSTANT_NAMES: tuple[str, ...] = ("SECONDS_PER_HOUR", "DECAY_FORM", "REFILL_PHASE")


def shape_version() -> str:
    """Digest of the current shape constants (see :data:`SHAPE_CONSTANT_NAMES`)."""
    module = globals()
    text = ";".join(f"{name}={module[name]!r}" for name in SHAPE_CONSTANT_NAMES)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


#: Digest of the shape constants; the archive seed fingerprint consumes it.
SHAPE_VERSION = shape_version()


def topoff_record(seed: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The ``topoff`` record of a seed, or None when it has none."""
    if not seed:
        return None
    record = seed.get(TOPOFF_KEY)
    return record if isinstance(record, Mapping) else None


def tau_period(record: Mapping[str, Any]) -> tuple[float, float]:
    """``(tau, period_s)`` of a ``topoff`` record, both in seconds."""
    return float(record["lifetime_h"]) * SECONDS_PER_HOUR, float(record["period_s"])


def shape(record: Mapping[str, Any], t_s: float) -> float:
    """The unit sawtooth ``s(t)`` of a ``topoff`` record at epoch second ``t_s``.

    Args:
        record: The ``topoff`` record, ``{ceiling, lifetime_h, period_s}``.
        t_s: Absolute epoch seconds.

    Returns:
        ``exp(-((t_s mod period_s) / tau))``: 1.0 at each refill, falling
        towards ``exp(-period_s / tau)`` just before the next.
    """
    tau, period = tau_period(record)
    return math.exp(-(float(t_s) % period) / tau)


def period_mean(record: Mapping[str, Any]) -> float:
    """The mean ``m`` of the unit sawtooth over one period."""
    tau, period = tau_period(record)
    return (tau / period) * -math.expm1(-period / tau)


def tooth_depth(record: Mapping[str, Any]) -> float:
    """How far the sawtooth carries a reading from its held level ``ceiling * m``.

    Returns:
        ``|ceiling| * max(1 - m, m - exp(-period_s / tau))``: the larger of
        the overshoot at a refill and the undershoot just before one.
    """
    tau, period = tau_period(record)
    mean = period_mean(record)
    return abs(float(record["ceiling"])) * max(1.0 - mean, mean - math.exp(-period / tau))


def held_value(
    seed: Mapping[str, Any] | None,
    lookup: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> float | None:
    """The value a ``topoff`` or ``topoff_lifetime`` seed holds.

    Args:
        seed: A channel's seed, or None.
        lookup: Returns the seed of an address, or None; resolves the channel
            a ``topoff_lifetime`` names. Without one a ``topoff_lifetime``
            holds nothing.

    Returns:
        ``ceiling * m`` for a ``topoff`` seed; the referenced seed's
        ``lifetime_h`` for a ``topoff_lifetime`` seed whose channel carries
        ``topoff``; None for any other seed.
    """
    record = topoff_record(seed)
    if record is not None:
        return float(record["ceiling"]) * period_mean(record)
    lifetime = (seed or {}).get(LIFETIME_KEY)
    if not isinstance(lifetime, Mapping) or lookup is None:
        return None
    source = topoff_record(lookup(str(lifetime.get("of"))))
    if source is None:
        return None
    return float(source["lifetime_h"])
