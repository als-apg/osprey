"""The call bound of a control-system connector.

The seconds a control-system connector gives one call when the call names no
bound of its own, read from the connector's own block
(``control_system.connector.<type>``). One key, one default and one check, so
every connector that bounds its calls names and refuses that bound alike.
"""

import math
from collections.abc import Mapping
from typing import Any, SupportsFloat

#: Seconds a control-system call is given when its connector block names none.
DEFAULT_TIMEOUT_S = 5.0

#: The connector block's key for that bound.
TIMEOUT_KEY = "timeout_s"


def call_timeout_s(config: Mapping[str, Any], connector_type: str | None) -> float:
    """The call bound one connector block declares.

    Args:
        config: A connector's own config section — the mapping
            ``control_system.connector.<type>`` resolves to.
        connector_type: The type that block belongs to, so a refusal names the
            key an operator has to go and fix. ``None`` leaves the type a
            placeholder in that message.

    Returns:
        The declared bound in seconds, or :data:`DEFAULT_TIMEOUT_S` when the
        block declares none.

    Raises:
        ValueError: If ``timeout_s`` is a bool, neither a number nor a numeric
            string, not finite, or not positive.
    """
    block = f"control_system.connector.{connector_type or '<type>'}"
    value = config.get(TIMEOUT_KEY, DEFAULT_TIMEOUT_S)
    seconds = _positive_seconds(value)
    if seconds is None:
        raise ValueError(
            f"{block}.{TIMEOUT_KEY} must be a positive number of seconds, got {value!r}"
        )
    return seconds


def _positive_seconds(value: object) -> float | None:
    """*value* as a positive, finite number of seconds, or ``None``.

    The same rule as ``osprey.utils.seconds.positive_seconds``, kept here because
    this package never imports ``osprey``. A ``bool`` is refused although
    ``float(True)`` is ``1.0``. A string that parses as a number is accepted,
    since ``${VAR}`` interpolation hands readers strings. An ``OverflowError``
    counts as "not a number": YAML gives an out-of-range integer literal as an
    ``int`` that ``float()`` cannot convert. Zero, negatives, NaN and infinity
    are not durations.
    """
    if isinstance(value, bool) or not isinstance(value, str | SupportsFloat):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None
