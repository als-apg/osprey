"""Central type definitions for the optimization system."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, TypeAlias


class Direction(str, Enum):  # noqa: UP042 - str() and serialized form kept from tuning_scripts
    """Objective direction. ``str`` mixin keeps wire/string comparisons working."""

    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


VariableBounds: TypeAlias = dict[str, list[float]]
VariableInputs: TypeAlias = dict[str, float]
PVValues: TypeAlias = dict[str, float]
BestPoint: TypeAlias = dict[str, float]
ConfigDict: TypeAlias = dict[str, Any]


def validate_min_le_max(min_value: float, max_value: float, *, label: str = "min") -> None:
    """Raise ``ValueError`` if ``min_value`` exceeds ``max_value``.

    Shared by the bound validators in ``config`` and ``protocols`` so the
    ``min > max`` check lives in one place.
    """
    if min_value > max_value:
        raise ValueError(f"{label} ({min_value}) cannot exceed max ({max_value})")


def require_finite(value: float, *, label: str = "value") -> None:
    """Raise ``ValueError`` if ``value`` is NaN or infinite."""
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value}")


def require_finite_positive(value: float, *, label: str = "value") -> None:
    """Raise ``ValueError`` if ``value`` is not a finite, strictly positive number."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{label} must be a finite positive number, got {value}")


# Derived from ``Direction`` so the two can't drift.
DIRECTIONS: frozenset[Direction] = frozenset(Direction)


def parse_direction(value: str) -> Direction:
    """Validate an objective-direction string and return it as a ``Direction``.

    The accepted spelling is exactly ``MAXIMIZE`` / ``MINIMIZE`` (the canonical
    convention); anything else raises ``ValueError``.
    """
    if value not in DIRECTIONS:
        raise ValueError(
            f"Invalid objective direction {value!r}; expected 'MAXIMIZE' or 'MINIMIZE'"
        )
    return Direction(value)


__all__ = [
    "DIRECTIONS",
    "BestPoint",
    "ConfigDict",
    "Direction",
    "PVValues",
    "VariableBounds",
    "VariableInputs",
    "parse_direction",
    "require_finite",
    "require_finite_positive",
    "validate_min_le_max",
]
