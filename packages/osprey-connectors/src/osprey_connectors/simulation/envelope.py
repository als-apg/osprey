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

The module imports only the standard library and the standard-library-only
:mod:`~osprey_connectors.simulation.topoff` and
:mod:`~osprey_connectors.simulation.values`, so the build, the simulator
view's reader and the texture model share one rule.
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from osprey_connectors.simulation import topoff
from osprey_connectors.simulation.values import _real

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.records import (
        NoiseRecord,
        ResolvedCoupling,
        SeedRecord,
    )

__all__ = [
    "active_envelopes",
    "declares_motion",
    "motion_envelope",
]

#: Gaussian headroom on a noise sigma: a per-draw miss rate near 1e-9.
_MOTION_SIGMAS = 6.0

_ABSOLUTE: Final = "absolute"
_RELATIVE: Final = "relative"


def _number(value: object) -> float | None:
    """``value`` as a float when it is a real number, else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


# A facility file's scenario blocks reach :func:`active_envelopes` as the
# schema leaves them, untyped. These read a field the way the plain ``in``,
# ``[]`` and ``.get`` of a mapping do, and raise what those raise on any other
# value, so every block reads as it did before the reader was typed.


def _get(record: object, key: str) -> object:
    """``record.get(key)``."""
    if isinstance(record, Mapping):
        return record.get(key)
    raise AttributeError(f"{type(record).__name__!r} object has no attribute 'get'")


def _item(record: object, key: str) -> object:
    """``record[key]``."""
    if isinstance(record, Mapping):
        return record[key]
    raise TypeError(f"{type(record).__name__!r} object is not subscriptable")


def _has(record: object, key: str) -> bool:
    """``key in record``."""
    if isinstance(record, Container):
        return key in record
    raise TypeError(f"argument of type {type(record).__name__!r} is not iterable")


def _iterate(value: object) -> Iterable[object]:
    """``value`` to iterate over."""
    if isinstance(value, Iterable):
        return value
    raise TypeError(f"{type(value).__name__!r} object is not iterable")


def _items(block: object) -> Iterable[tuple[object, object]]:
    """``block.items()``."""
    if isinstance(block, Mapping):
        return block.items()
    raise AttributeError(f"{type(block).__name__!r} object has no attribute 'items'")


def _update(merged: dict[object, object], block: object) -> None:
    """``merged.update(block)``: a mapping, or an iterable of key-value pairs."""
    if isinstance(block, Mapping):
        merged.update(block)
        return
    for index, pair in enumerate(_iterate(block)):
        if not isinstance(pair, Iterable):
            raise TypeError(
                f"cannot convert dictionary update sequence element #{index} to a sequence"
            )
        items = tuple(pair)
        if len(items) != 2:
            raise ValueError(
                f"dictionary update sequence element #{index} has length {len(items)}; "
                "2 is required"
            )
        merged[items[0]] = items[1]


def _noise_sigma(noise: object, nominal: object = None) -> float:
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
    if _has(noise, _ABSOLUTE):
        return abs(_real(_item(noise, _ABSOLUTE) or 0.0))
    relative = abs(_real(_get(noise, _RELATIVE) or 0.0))
    if not relative:
        return 0.0
    base = _number(nominal)
    if base is None:
        raise ValueError(f"a relative noise is taken of a numeric nominal, not {nominal!r}")
    return relative * abs(base)


def _term(noise: NoiseRecord | None) -> float:
    """The noise record's one term, signed as stated; 0.0 for no record."""
    if not noise:
        return 0.0
    return float(noise.get(_ABSOLUTE) or noise.get(_RELATIVE) or 0.0)


def declares_motion(seed: SeedRecord | None) -> bool:
    """Whether a seed states motion: a non-zero noise term, ``drift``, or ``topoff``."""
    if not seed:
        return False
    return bool(
        _term(seed.get("noise")) or seed.get("drift") or topoff.topoff_record(seed) is not None
    )


def motion_envelope(
    seed: SeedRecord | None,
    *,
    noise: NoiseRecord | None = None,
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
    return _envelope(
        seed,
        noise,
        (
            (coupling.get("gain"), coupling.get("gain_wander"), coupling.get("drive"))
            for coupling in couplings
        ),
        still,
    )


def _envelope(
    seed: SeedRecord | None,
    noise: object,
    couplings: Iterable[tuple[object, object, object]],
    still: bool,
) -> float:
    """:func:`motion_envelope`, each coupling as ``(gain, gain_wander, drive)``."""
    if still:
        return 0.0
    seed = seed or {}
    drift = seed.get("drift")
    envelope = abs(float(drift.get("amplitude") or 0.0)) if drift else 0.0
    for gain, wander, drive in couplings:
        envelope += (
            abs(_real(gain or 0.0))
            * (1.0 + (abs(_real(_get(wander, "amplitude") or 0.0)) if wander else 0.0))
            * (abs(_real(_get(drive, "amplitude") or 0.0)) if drive else 0.0)
        )
    shape = topoff.topoff_record(seed)
    if shape is not None:
        envelope += topoff.tooth_depth(shape)
    held = topoff.held_value(seed)
    record = noise if noise is not None else seed.get("noise")
    nominal = held if held is not None else seed.get("nominal")
    return envelope + _MOTION_SIGMAS * _noise_sigma(record, nominal)


def active_envelopes(
    seeds: Mapping[str, SeedRecord],
    scenarios: Mapping[str, Mapping[str, object]],
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
        scenarios: Scenario records by name, the simulator view's or the
            facility file's; only ``still``, ``drivers``, ``couple`` and
            ``noise`` are read.
        active: The active scenario names, in order; a name ``scenarios``
            lacks adds nothing.

    Returns:
        The envelope by address, sorted.
    """
    # The view copies a scenario's ``drivers``, ``couple`` and ``noise`` blocks
    # from the facility file verbatim, so either source states them as its
    # schema does; its ``still`` and ``faults`` the view resolves first. The
    # facility file's blocks are unchecked here, so each is read through the
    # helpers above, as a plain mapping read would.
    drivers: dict[object, object] = {}
    couple: dict[str, list[object]] = {}
    noise: dict[str, object] = {}
    still: set[str] = set()
    for name in active:
        scenario = scenarios.get(name)
        if scenario is None:
            continue
        still.update(str(address) for address in _iterate(scenario.get("still") or ()))
        _update(drivers, scenario.get("drivers") or {})
        for address, terms in _items(scenario.get("couple") or {}):
            couple.setdefault(str(address), []).extend(_iterate(terms))
        for address, record in _items(scenario.get("noise") or {}):
            noise[str(address)] = record
    resolved: dict[str, list[tuple[object, object, object]]] = {}
    for address, terms in couple.items():
        for term in terms:
            drive = drivers.get(str(_item(term, "driver")))
            if drive is not None:
                coupling = (_get(term, "gain"), _get(term, "gain_wander"), drive)
                resolved.setdefault(address, []).append(coupling)
    addresses = {str(address) for address in seeds} | set(couple) | set(noise)
    return {
        address: _envelope(
            seeds.get(address),
            noise.get(address),
            resolved.get(address, ()),
            address in still,
        )
        for address in sorted(addresses)
    }
