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

The module imports only the standard library and the package's
standard-library-only :mod:`~osprey_connectors.simulation.values`, so the
envelope rule and the facility build share it with the texture model; the
build reads a facility seed as a
:class:`~osprey_connectors.simulation.records.SeedRecord` first. :data:`SHAPE_VERSION` is a
digest of the shape constants; the archive seed fingerprint consumes it, so a
change to the shape re-seeds the stored history.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.records import SeedRecord, TopoffRecord

__all__ = [
    "SHAPE_VERSION",
    "held_value",
    "topoff_record",
]

#: Seconds per lifetime hour.
_SECONDS_PER_HOUR = 3600.0
#: Form of the decay between refills; part of the shape digest.
_DECAY_FORM = "exponential"
#: Where in a period the refill lands; part of the shape digest.
_REFILL_PHASE = "period_start"

#: The constants :data:`SHAPE_VERSION` digests, in a fixed order. Each is
#: digested under its name without the leading underscore, the spelling it had
#: while it was public, so making them private left the digest unchanged.
_SHAPE_CONSTANT_NAMES: tuple[str, ...] = ("_SECONDS_PER_HOUR", "_DECAY_FORM", "_REFILL_PHASE")


def _shape_version() -> str:
    """Digest of the current shape constants (see :data:`_SHAPE_CONSTANT_NAMES`)."""
    module: Mapping[str, object] = globals()
    text = ";".join(f"{name.lstrip('_')}={module[name]!r}" for name in _SHAPE_CONSTANT_NAMES)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


#: Digest of the shape constants; the archive seed fingerprint consumes it.
SHAPE_VERSION = _shape_version()


def topoff_record(seed: SeedRecord | None) -> TopoffRecord | None:
    """The ``topoff`` record of a seed, or None when it has none."""
    return None if seed is None else seed.topoff


def _tau_period(record: TopoffRecord) -> tuple[float, float]:
    """``(tau, period_s)`` of a ``topoff`` record, both in seconds."""
    return record.lifetime_h * _SECONDS_PER_HOUR, float(record.period_s)


def _period_mean(record: TopoffRecord) -> float:
    """The mean ``m`` of the unit sawtooth over one period."""
    return _unit_mean(*_tau_period(record))


def _unit_mean(tau: float, period: float) -> float:
    """The mean of ``exp(-(t mod period) / tau)`` over one period."""
    return (tau / period) * -math.expm1(-period / tau)


def _tooth_depth(record: TopoffRecord) -> float:
    """How far the sawtooth carries a reading from its held level ``ceiling * m``.

    Returns:
        ``|ceiling| * max(1 - m, m - exp(-period_s / tau))``: the larger of
        the overshoot at a refill and the undershoot just before one.
    """
    tau, period = _tau_period(record)
    mean = _period_mean(record)
    return abs(float(record.ceiling)) * max(1.0 - mean, mean - math.exp(-period / tau))


def held_value(
    seed: SeedRecord | None,
    lookup: Callable[[str], SeedRecord | None] | None = None,
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
        tau = float(record.lifetime_h) * _SECONDS_PER_HOUR
        return float(record.ceiling) * _unit_mean(tau, float(record.period_s))
    lifetime = None if seed is None else seed.topoff_lifetime
    if lifetime is None or lookup is None:
        return None
    source = topoff_record(lookup(lifetime.of))
    if source is None:
        return None
    return float(source.lifetime_h)
