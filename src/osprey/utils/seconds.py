"""Read a raw configuration value as a duration in seconds."""

from __future__ import annotations

import math
from typing import SupportsFloat


def positive_seconds(value: object) -> float | None:
    """Return *value* as a positive, finite number of seconds, or ``None``.

    Zero is not a duration here; :func:`non_negative_seconds` is the reader for
    a bound where zero means "do not wait".

    The caller decides what ``None`` means: which default applies, what the
    warning says and which key it names.

    Args:
        value: The raw configuration value.

    Returns:
        The duration in seconds, or ``None`` when *value* is not a positive,
        finite number of seconds.
    """
    seconds = _finite_seconds(value)
    return seconds if seconds is not None and seconds > 0 else None


def non_negative_seconds(value: object) -> float | None:
    """Return *value* as a finite number of seconds of zero or more, or ``None``.

    The reader for a bound where zero is meaningful. A negative number is still
    not a duration and is refused rather than clamped to zero, so a sign typo is
    reported and not read as "do not wait".

    The caller decides what ``None`` means: which default applies, what the
    warning says and which key it names.

    Args:
        value: The raw configuration value.

    Returns:
        The bound in seconds, or ``None`` when *value* is not a finite number of
        seconds of zero or more.
    """
    seconds = _finite_seconds(value)
    return seconds if seconds is not None and seconds >= 0 else None


def _finite_seconds(value: object) -> float | None:
    """Return *value* as a finite float, or ``None``.

    A ``bool`` is refused although ``float(True)`` is ``1.0``: a one-second
    duration is never what ``true`` meant. A string that parses as a number is
    accepted, since ``${VAR}`` interpolation hands readers strings. An
    ``OverflowError`` counts as "not a number", because YAML gives an
    out-of-range integer literal as an ``int`` that ``float()`` cannot convert.
    NaN and infinity are not durations.
    """
    if isinstance(value, bool) or not isinstance(value, str | SupportsFloat):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) else None
