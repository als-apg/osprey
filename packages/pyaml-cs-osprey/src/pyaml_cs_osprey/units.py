"""Scale channel values between their native unit and SI.

A channel reference may carry a unit suffix (``ADDR[mm]``); the control system serves the
value in that unit, while pyAML works in SI. This module maps each suffix spelling the
exported trees and the FODO lattice serve to its SI factor and SI unit word. Lookup is
case-folded and ignores surrounding whitespace; a reference without a suffix (a tune
reference, for instance) has factor 1 and an empty unit word.

The module is pure: it imports nothing from ``osprey`` or ``pyaml``, so an unknown suffix
raises the module-local :class:`UnitError`, which callers may re-raise as their own
exception type.
"""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "UnitError",
    "factor",
    "max_step_si",
    "si_unit",
    "to_native",
    "to_si",
]


class UnitError(ValueError):
    """A unit suffix has no known SI factor.

    Attributes:
        suffix: The suffix as given.
    """

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        super().__init__(f"unknown unit suffix {suffix!r}")


class _HasUnit(Protocol):
    @property
    def unit(self) -> str: ...


Reference = str | _HasUnit
"""A unit suffix, or any object with a ``unit`` attribute (a channel reference)."""

# case-folded suffix -> (SI factor, SI unit word)
_TABLE: dict[str, tuple[float, str]] = {
    "": (1.0, ""),
    "ampere": (1.0, "A"),
    "a": (1.0, "A"),
    "1/m": (1.0, "1/m"),
    "1/m**2": (1.0, "1/m**2"),
    "rad": (1.0, "rad"),
    "m": (1.0, "m"),
    "hz": (1.0, "Hz"),
    "mm": (1e-3, "m"),
    "um": (1e-6, "m"),
    "urad": (1e-6, "rad"),
    "mrad": (1e-3, "rad"),
    "khz": (1e3, "Hz"),
    "mhz": (1e6, "Hz"),
}


def _suffix(reference: Reference) -> str:
    return reference if isinstance(reference, str) else reference.unit


def _entry(suffix: str) -> tuple[float, str]:
    try:
        return _TABLE[suffix.strip().casefold()]
    except KeyError:
        raise UnitError(suffix) from None


def factor(suffix: str) -> float:
    """Return the factor that turns a value in ``suffix`` into SI.

    Raises:
        UnitError: The suffix is not a known spelling.
    """
    return _entry(suffix)[0]


def si_unit(suffix: str) -> str:
    """Return the SI unit word for ``suffix`` (``''`` for no suffix).

    Raises:
        UnitError: The suffix is not a known spelling.
    """
    return _entry(suffix)[1]


def to_si(value: float, reference: Reference) -> float:
    """Convert a native ``value`` of ``reference`` to SI.

    Raises:
        UnitError: The reference's suffix is not a known spelling.
    """
    return value * factor(_suffix(reference))


def to_native(value: float, reference: Reference) -> float:
    """Convert an SI ``value`` to the native unit of ``reference``.

    Raises:
        UnitError: The reference's suffix is not a known spelling.
    """
    return value / factor(_suffix(reference))


def max_step_si(max_step: float | None, reference: Reference) -> float | None:
    """Convert a native ``max_step`` of ``reference`` to SI; ``None`` stays ``None``.

    Raises:
        UnitError: The reference's suffix is not a known spelling.
    """
    if max_step is None:
        return None
    return to_si(max_step, reference)
