"""How far a simulated channel's declared motion can carry its value.

A channel's seed may declare two kinds of motion: ``noise``, a Gaussian draw,
and ``drift``, a slow wander whose ``amplitude`` bounds it (``|wander| <=
amplitude`` by construction). A ``noise`` record states exactly one term:
``{absolute: <sigma>}``, a sigma in the channel's unit, or ``{relative:
<fraction>}``, a sigma as a fraction of the reading. An active scenario may
couple the channel to shared drivers, each adding ``gain * (1 + gain_wander)
* driver`` with ``|driver| <= drive.amplitude``, and may replace its noise.

A served value is therefore the held value plus at most its envelope::

    |drift.amplitude|
      + sum |gain| x (1 + |gain_wander.amplitude|) x |drive.amplitude|
      + MOTION_SIGMAS x noise sigma

where a relative noise sigma is taken at the seed's own ``nominal``. A
channel that declares no motion has an envelope of 0.0.

The module imports only the standard library, so the build, the simulator
view's reader and the texture model share one rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "MOTION_SIGMAS",
    "active_envelopes",
    "declares_motion",
    "motion_envelope",
    "noise_sigma",
]

#: Gaussian headroom on a noise sigma: a per-draw miss rate near 1e-9.
MOTION_SIGMAS = 6.0

_ABSOLUTE = "absolute"
_RELATIVE = "relative"


def _number(value: Any) -> float | None:
    """``value`` as a float when it is a real number, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def noise_sigma(noise: Mapping[str, Any] | None, nominal: Any = None) -> float:
    """The sigma a noise record draws with, in the channel's unit.

    Args:
        noise: A ``noise`` record, ``{absolute: <sigma>}`` or ``{relative:
            <fraction>}``, or None.
        nominal: The seed's ``nominal``, which a relative term is taken of.

    Returns:
        ``|absolute|``, or ``|relative| x |nominal|``; 0.0 for no record.

    Raises:
        ValueError: The record is relative and ``nominal`` is not a number.
    """
    if not noise:
        return 0.0
    if _ABSOLUTE in noise:
        return abs(float(noise[_ABSOLUTE] or 0.0))
    relative = abs(float(noise.get(_RELATIVE) or 0.0))
    if not relative:
        return 0.0
    base = _number(nominal)
    if base is None:
        raise ValueError(f"a relative noise is taken of a numeric nominal, not {nominal!r}")
    return relative * abs(base)


def _term(noise: Mapping[str, Any] | None) -> float:
    """The noise record's one term, signed as stated; 0.0 for no record."""
    if not noise:
        return 0.0
    return float(noise.get(_ABSOLUTE) or noise.get(_RELATIVE) or 0.0)


def declares_motion(seed: Mapping[str, Any] | None) -> bool:
    """Whether a seed states motion: a non-zero noise term, or ``drift``."""
    if not seed:
        return False
    return bool(_term(seed.get("noise")) or seed.get("drift"))


def motion_envelope(
    seed: Mapping[str, Any] | None,
    *,
    noise: Mapping[str, Any] | None = None,
    couplings: Iterable[Mapping[str, Any]] = (),
) -> float:
    """The band a channel's motion keeps its value within around the held value.

    Args:
        seed: The channel's seed, or None.
        noise: A scenario's noise record standing in for the seed's, or None
            for the seed's own.
        couplings: Resolved couplings, each ``{driver, gain, gain_wander?,
            drive}`` with ``drive`` the driver's ``{kind, amplitude,
            period_s}``.

    Returns:
        ``|drift.amplitude| + sum |gain| x (1 + |gain_wander.amplitude|) x
        |drive.amplitude| + MOTION_SIGMAS x noise sigma``; 0.0 for an absent
        seed with no couplings.
    """
    seed = seed or {}
    drift = seed.get("drift") or {}
    envelope = abs(float(drift.get("amplitude") or 0.0))
    for coupling in couplings:
        wander = coupling.get("gain_wander") or {}
        drive = coupling.get("drive") or {}
        envelope += (
            abs(float(coupling.get("gain") or 0.0))
            * (1.0 + abs(float(wander.get("amplitude") or 0.0)))
            * abs(float(drive.get("amplitude") or 0.0))
        )
    record = noise if noise is not None else seed.get("noise")
    return envelope + MOTION_SIGMAS * noise_sigma(record, seed.get("nominal"))


def active_envelopes(
    seeds: Mapping[str, Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
    active: Sequence[str],
) -> dict[str, float]:
    """Every seeded, coupled or noise-replaced channel's envelope under ``active``.

    The active scenarios' ``noise`` and ``couple`` blocks merge in order, a
    later scenario's noise record replacing an earlier one's, and each
    coupling takes its driver's ``drive`` from the merged ``drivers``; a
    coupling to an undeclared driver adds nothing, as it serves nothing.

    Args:
        seeds: Seeds by address.
        scenarios: Scenario records by name.
        active: The active scenario names, in order; a name ``scenarios``
            lacks adds nothing.

    Returns:
        The envelope by address, sorted.
    """
    drivers: dict[str, Mapping[str, Any]] = {}
    couple: dict[str, list[Mapping[str, Any]]] = {}
    noise: dict[str, Mapping[str, Any]] = {}
    for name in active:
        scenario = scenarios.get(name) or {}
        drivers.update(scenario.get("drivers") or {})
        for address, terms in (scenario.get("couple") or {}).items():
            couple.setdefault(str(address), []).extend(terms)
        for address, record in (scenario.get("noise") or {}).items():
            noise[str(address)] = record
    resolved: dict[str, list[Mapping[str, Any]]] = {}
    for address, terms in couple.items():
        for term in terms:
            drive = drivers.get(str(term["driver"]))
            if drive is not None:
                resolved.setdefault(address, []).append({**term, "drive": drive})
    addresses = {str(address) for address in seeds} | set(couple) | set(noise)
    return {
        address: motion_envelope(
            seeds.get(address), noise=noise.get(address), couplings=resolved.get(address, ())
        )
        for address in sorted(addresses)
    }
