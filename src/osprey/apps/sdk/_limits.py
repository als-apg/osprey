"""Whether a value may be written to one address, judged before any write.

:func:`prewrite_verdict` relays the judgement of the target's connector
limits validator — the same one the connector applies at write time, built
for the target's limits posture — so an application can show a verdict per
setpoint before it asks for approval. No bounds logic lives here: the record,
the mode and the refusal reason are the validator's.

The validator is imported inside the function, so importing the SDK loads
none of it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["WriteVerdictView", "prewrite_verdict"]


@dataclass(frozen=True)
class WriteVerdictView:
    """One address's pre-write verdict under a target's limits posture.

    Attributes:
        allowed: Whether the connector's validator would take the write.
        reason: The validator's refusal, ``"<violation type>: <reason>"``
            (``MAX_STEP_EXCEEDED: Step size ...``); ``None`` when allowed.
        declared: Whether the limits database holds a record for the address.
        enforced: Whether limits checking is on for the target.
        mode: The limits mode the target runs under (``optional`` or
            ``exclusive``), ``None`` when unstated or limits are off.
        limits: The declared record as ``{min, max, max_step}``, ``None``
            when the address has no record.
    """

    allowed: bool
    reason: str | None
    declared: bool
    enforced: bool
    mode: Literal["optional", "exclusive"] | None
    limits: dict[str, float | None] | None


@functools.cache
def _validator(target: str) -> Any:
    """The target's limits validator, built once per target (``None`` = limits off)."""
    from osprey_connectors.control_system import LimitsValidator

    return LimitsValidator.from_config(target=target)


def prewrite_verdict(
    address: str, value: float, *, target: str, current: float | None
) -> WriteVerdictView:
    """Judge a write of *value* to *address* as the *target*'s connector would.

    Args:
        address: The channel address to be written.
        value: The value to be written.
        target: The control target whose connector posture judges the write.
        current: The address's present value, which ``max_step`` is measured
            against. ``None`` on a record with ``max_step`` refuses, as the
            connector does: a step that cannot be measured cannot be approved.

    Returns:
        The verdict. Limits off for the target gives ``enforced=False,
        allowed=True``; a fail-safe validator (incomplete limits block,
        unavailable database) refuses with its reason.
    """
    from osprey_connectors.errors import ChannelLimitsViolationError

    validator = _validator(target)
    if validator is None:
        return WriteVerdictView(
            allowed=True, reason=None, declared=False, enforced=False, mode=None, limits=None
        )

    record = validator.limits.get(address)
    limits = (
        None
        if record is None
        else {"min": record.min_value, "max": record.max_value, "max_step": record.max_step}
    )
    mode = validator.policy.get("mode")
    reader = None if current is None else (lambda _address: current)
    try:
        validator.validate(address, value, read_current=reader)
    except ChannelLimitsViolationError as refusal:
        return WriteVerdictView(
            allowed=False,
            reason=f"{refusal.violation_type}: {refusal.violation_reason}",
            declared=record is not None,
            enforced=True,
            mode=mode,
            limits=limits,
        )
    return WriteVerdictView(
        allowed=True,
        reason=None,
        declared=record is not None,
        enforced=True,
        mode=mode,
        limits=limits,
    )
