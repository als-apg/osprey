"""Read a raw configuration value as a duration in seconds."""

from __future__ import annotations

import math
from typing import SupportsFloat


def positive_seconds(value: object) -> float | None:
    """Return *value* as a positive, finite number of seconds, or ``None``.

    A ``bool`` is refused although ``float(True)`` is ``1.0``: a one-second
    duration is never what ``true`` meant. A string that parses as a number is
    accepted, since ``${VAR}`` interpolation hands readers strings. An
    ``OverflowError`` counts as "not a number", because YAML gives an
    out-of-range integer literal as an ``int`` that ``float()`` cannot convert.
    Zero, negatives, NaN and infinity are not durations.

    The caller decides what ``None`` means: which default applies, what the
    warning says and which key it names.

    Args:
        value: The raw configuration value.

    Returns:
        The duration in seconds, or ``None`` when *value* is not a positive,
        finite number of seconds.
    """
    if isinstance(value, bool) or not isinstance(value, str | SupportsFloat):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None
