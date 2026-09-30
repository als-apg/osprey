"""The call bound of a control-system connector.

The seconds a control-system connector gives one call when the call names no
bound of its own, read from the connector's own block
(``control_system.connector.<type>``). One key, one default and one check, so
every connector that bounds its calls names and refuses that bound alike.
"""

import math
from collections.abc import Iterator, Mapping
from typing import Any, SupportsFloat

from osprey_connectors.types import DOOCS, EPICS, LIVE_STANDIN, TANGO, VIRTUAL_ACCELERATOR

#: Seconds a control-system call is given when its connector block names none.
DEFAULT_TIMEOUT_S = 5.0

#: The connector block's key for that bound.
TIMEOUT_KEY = "timeout_s"

#: The spelling the connector block no longer reads.
_RENAMED_KEY = "timeout"

#: Every old-spelled key, as the dotted path a whole config names it by.
_RENAMED_PATHS = frozenset(
    f"control_system.connector.{connector_type}.{_RENAMED_KEY}"
    for connector_type in (EPICS, VIRTUAL_ACCELERATOR, LIVE_STANDIN, TANGO, DOOCS)
)


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
        ValueError: If the block still carries ``timeout``, or if ``timeout_s``
            is a bool, neither a number nor a numeric string, not finite, or
            not positive.
    """
    block = f"control_system.connector.{connector_type or '<type>'}"
    if _RENAMED_KEY in config:
        raise ValueError(f"{block}.{_RENAMED_KEY} is renamed to {TIMEOUT_KEY}")
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


def refuse_renamed_timeout_keys(config: Mapping[str, Any]) -> None:
    """Refuse a whole config that still spells a connector's call bound ``timeout``.

    Checked when a config is read, so the old key fails before any connector is
    built; :func:`call_timeout_s` refuses it again at connect. Nested and dotted
    spellings are read alike, so a ``config.yml`` and a profile's ``config:``
    block are both covered.

    Args:
        config: A whole config mapping, nested, dotted or both.

    Raises:
        ValueError: Naming every old-spelled key and :data:`TIMEOUT_KEY`.
    """
    renamed = sorted(path for path in _dotted_paths(config) if path in _RENAMED_PATHS)
    if renamed:
        raise ValueError("; ".join(f"{path} is renamed to {TIMEOUT_KEY}" for path in renamed))


def _dotted_paths(node: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    """Every key of *node*, nested ones included, as a dotted path."""
    for key, value in node.items():
        if not isinstance(key, str):
            continue
        path = f"{prefix}.{key}" if prefix else key
        yield path
        if isinstance(value, Mapping):
            yield from _dotted_paths(value, path)
