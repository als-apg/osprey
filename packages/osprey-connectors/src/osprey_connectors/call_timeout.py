"""The call bounds of a control-system connector.

The seconds a control-system connector gives one call when the call names no
bound of its own, and the ceiling on the fresh read a ``max_step`` check makes
before a write, both read from the connector's own block
(``control_system.connector.<type>``). One key, one default and one check each,
so every connector that bounds its calls names and refuses those bounds alike.
"""

import logging
import math
from collections.abc import Iterable, Iterator, Mapping
from typing import SupportsFloat

from osprey_connectors.errors import ConfigurationError
from osprey_connectors.types import DOOCS, EPICS, LIVE_STANDIN, TANGO, VIRTUAL_ACCELERATOR

__all__ = [
    "DEFAULT_STEP_READ_TIMEOUT_S",
    "DEFAULT_TIMEOUT_S",
    "STEP_READ_TIMEOUT_KEY",
    "TIMEOUT_KEY",
    "call_timeout_s",
    "refuse_renamed_timeout_keys",
    "step_read_timeout_s",
]

#: The logger the step-read fallback warns on: the limits validator's, whose
#: check the budget bounds. Named, not taken from ``osprey_connectors.logger``,
#: which sits above this module.
logger = logging.getLogger("limits_validator")

#: Seconds a control-system call is given when its connector block names none.
DEFAULT_TIMEOUT_S = 5.0

#: The connector block's key for that bound.
TIMEOUT_KEY = "timeout_s"

#: The other spelling of that key, refused wherever a config carries it.
_RENAMED_KEY = "timeout"

#: Ceiling on the fresh read a ``max_step`` check makes, in seconds, when the
#: deployment names none. The check sits in front of a write an operator is
#: waiting on, so a channel that does not answer must fail closed quickly
#: rather than hold the write open: running out of budget answers ``None``,
#: which refuses the write.
#:
#: How long that read takes is a property of the facility's control network,
#: exactly like the connector's own ``timeout_s`` — a gateway two hops away
#: answers slower than a soft IOC on the same host — so a deployment overrides
#: it with ``control_system.connector.<type>.step_read_timeout_s``.
DEFAULT_STEP_READ_TIMEOUT_S = 2.0

#: The per-connector key that overrides it, spelled once so the connector and
#: the python-executor sandbox cannot read two different names.
STEP_READ_TIMEOUT_KEY = "step_read_timeout_s"

#: Every old-spelled key, as the dotted path a whole config names it by.
_RENAMED_PATHS = frozenset(
    f"control_system.connector.{connector_type}.{_RENAMED_KEY}"
    for connector_type in (EPICS, VIRTUAL_ACCELERATOR, LIVE_STANDIN, TANGO, DOOCS)
)


def call_timeout_s(config: Mapping[str, object], connector_type: str | None) -> float:
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
        ConfigurationError: If the block still carries ``timeout``, or if
            ``timeout_s`` is a bool, neither a number nor a numeric string, not
            finite, or not positive. ``key`` is the dotted key at fault.
    """
    block = f"control_system.connector.{connector_type or '<type>'}"
    if _RENAMED_KEY in config:
        raise ConfigurationError(
            f"{block}.{_RENAMED_KEY} is renamed to {TIMEOUT_KEY}", key=f"{block}.{_RENAMED_KEY}"
        )
    value = config.get(TIMEOUT_KEY, DEFAULT_TIMEOUT_S)
    seconds = _positive_seconds(value)
    if seconds is None:
        raise ConfigurationError(
            f"{block}.{TIMEOUT_KEY} must be a positive number of seconds, got {value!r}",
            key=f"{block}.{TIMEOUT_KEY}",
        )
    return seconds


def step_read_timeout_s(
    config: Mapping[str, object] | None, connector_type: str | None = None
) -> float:
    """The ``max_step`` fresh-read budget one connector block declares.

    Args:
        config: A connector's own config section — the mapping
            ``control_system.connector.<type>`` resolves to.
        connector_type: The type that block belongs to, so a warning names the
            key an operator has to go and fix. ``None`` leaves the type a
            placeholder in that message.

    Returns:
        The declared budget, or :data:`DEFAULT_STEP_READ_TIMEOUT_S` when
        the block declares none or declares something that is not a positive
        number. A value that cannot be used is warned about, naming the key and
        what was found, and then falls back rather than raising: this resolves
        on the write path, and a typo in a tuning key must not be the thing
        that takes writes down.
    """
    if not isinstance(config, Mapping):
        return DEFAULT_STEP_READ_TIMEOUT_S
    declared = config.get(STEP_READ_TIMEOUT_KEY)
    if declared is None:
        return DEFAULT_STEP_READ_TIMEOUT_S
    if isinstance(declared, bool) or not isinstance(declared, int | float) or declared <= 0:
        logger.warning(
            "control_system.connector.%s.%s must be a positive number of seconds, "
            "got %r — falling back to %s",
            connector_type or "<type>",
            STEP_READ_TIMEOUT_KEY,
            declared,
            DEFAULT_STEP_READ_TIMEOUT_S,
        )
        return DEFAULT_STEP_READ_TIMEOUT_S
    return float(declared)


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


def refuse_renamed_timeout_keys(config: Mapping[str, object]) -> None:
    """Refuse a whole config that still spells a connector's call bound ``timeout``.

    Checked when a config is read, so the old key fails before any connector is
    built; :func:`call_timeout_s` refuses it again at connect. Nested and dotted
    spellings are read alike, so a ``config.yml`` and a profile's ``config:``
    block are both covered.

    Args:
        config: A whole config mapping, nested, dotted or both.

    Raises:
        ConfigurationError: Naming every old-spelled key and :data:`TIMEOUT_KEY`;
            ``key`` is the first of them.
    """
    renamed = sorted(path for path in _dotted_paths(config) if path in _RENAMED_PATHS)
    if renamed:
        raise ConfigurationError(
            "; ".join(f"{path} is renamed to {TIMEOUT_KEY}" for path in renamed), key=renamed[0]
        )


def _dotted_paths(node: Mapping[str, object], prefix: str = "") -> Iterator[str]:
    """Every string key of *node*, nested ones included, as a dotted path."""
    # A YAML mapping may hold a key of any type, whatever the annotation says.
    items: Iterable[tuple[object, object]] = node.items()
    for key, value in items:
        if not isinstance(key, str):
            continue
        path = f"{prefix}.{key}" if prefix else key
        yield path
        if isinstance(value, Mapping):
            yield from _dotted_paths(value, path)
