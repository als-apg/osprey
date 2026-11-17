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
      + _MOTION_SIGMAS x noise sigma
      + tooth depth

where a relative noise sigma is taken at the seed's own ``nominal``, or at
the level a ``topoff`` seed holds, and the tooth depth is how far a
``topoff`` seed's sawtooth carries the reading from that level. A
channel that declares no motion, or that an active scenario stills, has an
envelope of 0.0.

The module imports the standard library, the standard-library-only
:mod:`~osprey_connectors.simulation.topoff` and the view's records of
:mod:`~osprey_connectors.simulation.records`, so the build, the simulator
view's reader and the texture model share one rule. A caller holding a seed or
a scenario as a mapping (the facility build, reading its YAML) validates it
into its record first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from osprey_connectors.simulation import topoff
from osprey_connectors.simulation.records import (
    AbsoluteNoise,
    CouplingTerm,
    DriveRecord,
    Noise,
    RelativeNoise,
    ResolvedCoupling,
    ScenarioRecord,
    SeedRecord,
)

__all__ = [
    "active_envelopes",
    "declares_motion",
    "motion_envelope",
]

#: Gaussian headroom on a noise sigma: a per-draw miss rate near 1e-9.
_MOTION_SIGMAS = 6.0


def _number(value: object) -> float | None:
    """``value`` as a float when it is a real number, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _noise_sigma(noise: Noise | None, nominal: object = None) -> float:
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
    match noise:
        case None:
            return 0.0
        case AbsoluteNoise(absolute=absolute):
            return abs(float(absolute))
        case RelativeNoise(relative=relative) if relative:
            base = _number(nominal)
            if base is None:
                raise ValueError(f"a relative noise is taken of a numeric nominal, not {nominal!r}")
            return abs(float(relative)) * abs(base)
        case RelativeNoise():
            return 0.0


def _term(noise: Noise | None) -> float:
    """The noise record's one term, signed as stated; 0.0 for no record."""
    match noise:
        case None:
            return 0.0
        case AbsoluteNoise(absolute=term) | RelativeNoise(relative=term):
            return float(term)


def declares_motion(seed: SeedRecord | None) -> bool:
    """Whether a seed states motion: a non-zero noise term, ``drift``, or ``topoff``."""
    if seed is None:
        return False
    return bool(_term(seed.noise) or seed.drift or topoff.topoff_record(seed) is not None)


def motion_envelope(
    seed: SeedRecord | None,
    *,
    noise: Noise | None = None,
    couplings: Iterable[ResolvedCoupling] = (),
    still: bool = False,
) -> float:
    """The band a channel's motion keeps its value within around the held value.

    Args:
        seed: The channel's seed, or None.
        noise: A scenario's noise record standing in for the seed's, or None
            for the seed's own.
        couplings: Resolved couplings, each ``{driver, gain, gain_wander?,
            drive}`` with ``drive`` the driver's ``{kind, amplitude,
            period_s}``.
        still: The active scenarios still the channel.

    Returns:
        ``|drift.amplitude| + sum |gain| x (1 + |gain_wander.amplitude|) x
        |drive.amplitude| + _MOTION_SIGMAS x noise sigma + tooth depth``;
        0.0 for an absent seed with no couplings, and for a stilled channel.
    """
    if still:
        return 0.0
    drift = None if seed is None else seed.drift
    envelope = abs(float(drift.amplitude or 0.0)) if drift is not None else 0.0
    for coupling in couplings:
        wander = coupling.gain_wander
        envelope += (
            abs(float(coupling.gain or 0.0))
            * (1.0 + (abs(float(wander.amplitude or 0.0)) if wander is not None else 0.0))
            * abs(float(coupling.drive.amplitude or 0.0))
        )
    shape = topoff.topoff_record(seed)
    if shape is not None:
        envelope += topoff.tooth_depth(shape)
    held = topoff.held_value(seed)
    record = noise if noise is not None else (None if seed is None else seed.noise)
    nominal = held if held is not None else (None if seed is None else seed.nominal)
    return envelope + _MOTION_SIGMAS * _noise_sigma(record, nominal)


def active_envelopes(
    seeds: Mapping[str, SeedRecord],
    scenarios: Mapping[str, ScenarioRecord],
    active: Sequence[str],
) -> dict[str, float]:
    """Every seeded, coupled or noise-replaced channel's envelope under ``active``.

    The active scenarios' ``noise`` and ``couple`` blocks merge in order, a
    later scenario's noise record replacing an earlier one's, and each
    coupling takes its driver's ``drive`` from the merged ``drivers``; a
    coupling to an undeclared driver adds nothing, as it serves nothing. A
    reading any active scenario's ``still`` list names has an envelope of 0.0.

    Args:
        seeds: Seeds by address.
        scenarios: Scenario records by name; only ``still``, ``drivers``,
            ``couple`` and ``noise`` are read.
        active: The active scenario names, in order; a name ``scenarios``
            lacks adds nothing.

    Returns:
        The envelope by address, sorted.
    """
    drivers: dict[str, DriveRecord] = {}
    couple: dict[str, list[CouplingTerm]] = {}
    noise: dict[str, Noise] = {}
    still: set[str] = set()
    for name in active:
        scenario = scenarios.get(name)
        if scenario is None:
            continue
        still.update(scenario.still)
        drivers.update(scenario.drivers)
        for address, terms in scenario.couple.items():
            couple.setdefault(address, []).extend(terms)
        noise.update(scenario.noise)
    resolved: dict[str, list[ResolvedCoupling]] = {}
    for address, merged in couple.items():
        for term in merged:
            drive = drivers.get(term.driver)
            if drive is not None:
                resolved.setdefault(address, []).append(
                    ResolvedCoupling(
                        driver=term.driver,
                        gain=term.gain,
                        gain_wander=term.gain_wander,
                        drive=drive,
                    )
                )
    addresses = set(seeds) | set(couple) | set(noise)
    return {
        address: motion_envelope(
            seeds.get(address),
            noise=noise.get(address),
            couplings=resolved.get(address, ()),
            still=address in still,
        )
        for address in sorted(addresses)
    }
