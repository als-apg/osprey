"""The time budget of one control-system write.

One write runs three bounded phases on its connector: a step read (at most S
seconds), the put (at most T) and a confirming read (at most T), so the
connector call is bounded by ``D = S + 2T``. Everything that waits on a write
derives its own bound from those numbers: the session's wait adds the margin
:data:`WRITE_MARGIN_S`, the process that holds the write gets ``D + 2`` after
SIGTERM, and the relay in front of it gives up after ``min(D + 3, 27)``.

:func:`write_budget` reads T and S once, as the maxima over every connector
block of a config, so one budget covers whichever connector a write is bound
to. :func:`bound_budget` narrows it to the block a session is actually bound
to.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from osprey_connectors.call_timeout import DEFAULT_TIMEOUT_S, call_timeout_s
from osprey_connectors.control_system import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    STEP_READ_TIMEOUT_KEY,
    step_read_timeout_seconds,
)
from osprey_connectors.logger import get_logger

logger = get_logger("runtime")

__all__ = [
    "MAX_WRITE_BOUND_S",
    "MIN_CALL_BOUND_S",
    "GRACE_MARGIN_S",
    "RELAY_MARGIN_S",
    "MAX_RELAY_TIMEOUT_S",
    "WRITE_MARGIN_S",
    "WriteBudget",
    "write_budget",
    "bound_budget",
]

#: The ceiling on a write's connector bound D, in seconds.
MAX_WRITE_BOUND_S = 24.0

#: The call bound T the step-read cap always leaves room for, in seconds.
MIN_CALL_BOUND_S = 1.0

#: What the holding process's stop grace adds to D.
GRACE_MARGIN_S = 2.0

#: What the relay's timeout adds to D.
RELAY_MARGIN_S = 3.0

#: The ceiling on the relay's timeout, in seconds.
MAX_RELAY_TIMEOUT_S = 27.0

#: The margin M a waiter adds to D before it gives a write up; ``D + M`` stays
#: within both the stop grace and the relay timeout.
WRITE_MARGIN_S = 1.0


@dataclass(frozen=True)
class WriteBudget:
    """The time bounds of one write, fixed when a process starts.

    Attributes:
        T: The call bound: the longest ``timeout_s`` of any connector block,
            clamped so that ``D`` stays within :data:`MAX_WRITE_BOUND_S`.
        S: The step-read bound: the longest step-read ceiling of any connector
            block, capped so that ``T`` keeps at least :data:`MIN_CALL_BOUND_S`.
        D: The bound on a write's connector call: ``S + 2T``.
        grace_s: How long the process holding a write gets after SIGTERM:
            ``D + 2``.
        relay_timeout_s: How long the relay in front of that process waits:
            ``min(D + 3, 27)``.
    """

    T: float
    S: float
    D: float
    grace_s: float
    relay_timeout_s: float


def write_budget(config: Mapping[str, Any]) -> WriteBudget:
    """The write budget *config* gives.

    T and S are the maxima over every ``control_system.connector.<type>``
    block, seeded with the defaults (``timeout_s`` 5 s, step read 2 s), so a
    config with no blocks gives ``(5, 2, 12, 14, 15)``. S is capped at
    ``MAX_WRITE_BOUND_S - 2 * MIN_CALL_BOUND_S``; T is then clamped so that
    ``D = S + 2T`` stays within :data:`MAX_WRITE_BOUND_S`.

    A block that is not a mapping, or whose call bound cannot be read, is
    skipped with a warning: the connector it configures refuses to connect, so
    it never holds a write.

    Args:
        config: A whole config mapping.

    Returns:
        The budget.
    """
    timeouts = [DEFAULT_TIMEOUT_S]
    step_reads = [DEFAULT_STEP_READ_TIMEOUT_SECONDS]
    for connector_type, block in _connector_blocks(config).items():
        if not isinstance(block, Mapping):
            logger.warning(
                "control_system.connector.%s is not a mapping; the write budget skips it",
                connector_type,
            )
            continue
        try:
            timeout = call_timeout_s(block, str(connector_type))
        except ValueError as exc:
            logger.warning("%s; the write budget skips this block", exc)
            continue
        timeouts.append(timeout)
        step_reads.append(step_read_timeout_seconds(block, str(connector_type)))
    step_read = min(max(step_reads), MAX_WRITE_BOUND_S - 2 * MIN_CALL_BOUND_S)
    timeout = min(max(timeouts), (MAX_WRITE_BOUND_S - step_read) / 2)
    bound = step_read + 2 * timeout
    return WriteBudget(
        T=timeout,
        S=step_read,
        D=bound,
        grace_s=bound + GRACE_MARGIN_S,
        relay_timeout_s=min(bound + RELAY_MARGIN_S, MAX_RELAY_TIMEOUT_S),
    )


def bound_budget(
    block: Mapping[str, Any] | None,
    budget: WriteBudget,
    connector_type: str | None = None,
) -> tuple[float, float, float]:
    """The bounds one bound connector block runs a write with.

    Args:
        block: The bound connector's own block
            (``control_system.connector.<type>``); ``None`` declares nothing.
        budget: The process's :class:`WriteBudget`.
        connector_type: The block's type, named in any refusal or warning.

    Returns:
        ``(T_b, S_b, D_b)``: ``T_b = min(block timeout_s, T)``,
        ``S_b = min(block step-read ceiling, S)`` and ``D_b = S_b + 2 * T_b``.
        A block below the budget keeps its own values.

    Raises:
        ValueError: From :func:`call_timeout_s` when the block's call bound
            cannot be read; its connector would refuse to connect alike. Also
            when the block declares a non-finite step read (NaN or infinity),
            or when *budget* carries a non-finite ``T`` or ``S``: a write's
            step, put and confirm phases are never run without a finite bound.
    """
    declared = block if block is not None else {}
    _refuse_non_finite_step_read(declared, connector_type)
    _refuse_non_finite("the write budget's T", budget.T)
    _refuse_non_finite("the write budget's S", budget.S)
    timeout = min(call_timeout_s(declared, connector_type), budget.T)
    step_read = min(step_read_timeout_seconds(declared, connector_type), budget.S)
    return timeout, step_read, step_read + 2 * timeout


def _refuse_non_finite_step_read(block: Mapping[str, Any], connector_type: str | None) -> None:
    """Refuse a block whose step-read ceiling is a non-finite number.

    :func:`step_read_timeout_seconds` falls back on a value it cannot use, but
    NaN passes its ``<= 0`` check and would poison ``min`` with NaN, and an
    infinite ceiling is no bound at all, so both are refused here by name.
    """
    declared = block.get(STEP_READ_TIMEOUT_KEY)
    if isinstance(declared, float) and not math.isfinite(declared):
        raise ValueError(
            f"control_system.connector.{connector_type or '<type>'}.{STEP_READ_TIMEOUT_KEY} "
            f"must be a finite number of seconds, got {declared!r}"
        )


def _refuse_non_finite(name: str, value: float) -> None:
    """Refuse a budget bound *name* whose *value* is NaN or infinite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of seconds, got {value!r}")


def _connector_blocks(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """``control_system.connector`` of *config*, or an empty mapping."""
    control_system = config.get("control_system") if isinstance(config, Mapping) else None
    if not isinstance(control_system, Mapping):
        return {}
    blocks = control_system.get("connector")
    return blocks if isinstance(blocks, Mapping) else {}
