"""The person's side of a program's write authorization: what an approval checks and shows.

:mod:`osprey_connectors.write_authorization` owns the files — the holder's
request, the grant, the ledger — and the writer that issues a grant under the
owner's record lock. What it cannot check, because it sits below the
control-system connectors and reads no config, is checked here, once, for the
two surfaces a person approves from: the web terminal's card
(``POST /api/terminal/write-authorizations``) and ``osprey write-authorization
approve``.

**What an approval refuses.** A range must sit inside the deployment's limits
database: the channel listed there, writable, with a finite ``min_value`` and
``max_value`` (a range needs a number on both sides, so a channel the
database bounds on one side only, or not at all, is not authorizable — the
same reading as an enum, a string or a waveform), and the requested
``[min, max]`` inside them. A database that cannot be read refuses every
approval. The surfaces add the login's tier, the read-only run, and (on the
card) a current value that is not a number.

**What the person is shown** (:func:`card`): every channel with its range, the
limits bound it sits inside, and its current value when the surface read one;
the steps, the expiry, the lease, the holder, the label and the completion the
run will perform; and that a write the limits refuse still costs its step,
because the step is spent before the connector's own checks run.
"""

from __future__ import annotations

import math
import numbers
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from osprey_connectors.write_authorization import Grant, WriteRequest

__all__ = [
    "LIMITS_COST_SENTENCE",
    "ApprovalRefused",
    "ChannelCheck",
    "card",
    "check_channels",
    "grant_summary",
    "load_limits",
    "refusal",
]

#: Said on every approval surface: the step is spent before the limits run.
LIMITS_COST_SENTENCE: Final = (
    "A write the limits database refuses still costs its step: the step is "
    "spent before the connector's own checks run."
)

#: Why a channel cannot be approved, as each surface names it.
NOT_IN_LIMITS: Final = "not_in_limits"
NOT_WRITABLE: Final = "not_writable"
NOT_NUMERIC: Final = "not_numeric"
OUTSIDE_LIMITS: Final = "outside_limits"
LIMITS_UNAVAILABLE: Final = "limits_unavailable"

_PROBLEM_WORDS: Final = {
    NOT_IN_LIMITS: "is not in the limits database",
    NOT_WRITABLE: "is not writable in the limits database",
    NOT_NUMERIC: "has no numeric range in the limits database, or a value that is not a number",
    OUTSIDE_LIMITS: "asks for a range outside its limits",
}


class ApprovalRefused(Exception):
    """An approval surface refused, for *reason*, with a sentence a person can act on.

    Attributes:
        reason: One of this module's reason words, or a refusal reason of
            :data:`~osprey_connectors.refusals.REASONS`.
        detail: The sentence.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class ChannelCheck:
    """One requested channel against the limits database.

    Attributes:
        address: The channel, spelled as the holder will write it.
        min: The requested lower bound.
        max: The requested upper bound.
        limit_min: The database's ``min_value``, or ``None``.
        limit_max: The database's ``max_value``, or ``None``.
        problem: Why it cannot be approved, or ``None``.
    """

    address: str
    min: float
    max: float
    limit_min: float | None
    limit_max: float | None
    problem: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "min": self.min,
            "max": self.max,
            "limit_min": self.limit_min,
            "limit_max": self.limit_max,
            "problem": self.problem,
        }


def load_limits(config: Mapping[str, Any], config_path: Path | str | None) -> dict[str, Any]:
    """The deployment's limits database, ``{address: ChannelLimitsConfig}``.

    Read through the one loader every surface uses
    (:meth:`~osprey_connectors.control_system.LimitsValidator.load_configured_database`).

    Raises:
        ApprovalRefused: ``limits_unavailable`` — not configured, or unreadable.
    """
    from osprey_connectors.control_system import LimitsValidator, mapping_config_lookup

    loaded, reason = LimitsValidator.load_configured_database(
        config_lookup=mapping_config_lookup(config),
        config_path=None if config_path is None else str(config_path),
    )
    if loaded is None:
        raise ApprovalRefused(
            LIMITS_UNAVAILABLE,
            f"No approval without the limits database: {reason}.",
        )
    return dict(loaded[0])


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def check_channels(
    request: WriteRequest,
    limits: Mapping[str, Any],
    current: Mapping[str, Any] | None = None,
) -> list[ChannelCheck]:
    """Every requested channel judged against *limits*, in the request's order.

    *current* is ``{address: value}`` as a surface read it: a value that was
    read and is not a finite number makes its channel :data:`NOT_NUMERIC`. A
    value not read (absent, or ``None``) says nothing.
    """
    checks: list[ChannelCheck] = []
    for address, bound in request.channels.items():
        entry = limits.get(address)
        limit_min = None if entry is None else _finite(entry.min_value)
        limit_max = None if entry is None else _finite(entry.max_value)
        problem: str | None = None
        if entry is None:
            problem = NOT_IN_LIMITS
        elif not entry.writable:
            problem = NOT_WRITABLE
        elif limit_min is None or limit_max is None:
            problem = NOT_NUMERIC
        elif bound.min < limit_min or bound.max > limit_max:
            problem = OUTSIDE_LIMITS
        if problem is None and current is not None:
            value = current.get(address)
            if value is not None and _finite(value) is None:
                problem = NOT_NUMERIC
        checks.append(
            ChannelCheck(
                address=address,
                min=bound.min,
                max=bound.max,
                limit_min=limit_min,
                limit_max=limit_max,
                problem=problem,
            )
        )
    return checks


def refusal(checks: list[ChannelCheck]) -> ApprovalRefused | None:
    """The refusal the channels with a problem make, or ``None`` when every one passes.

    The reason is the first one's; the sentence names every one, so a person
    fixes the request once.
    """
    failing = [check for check in checks if check.problem is not None]
    if not failing:
        return None
    return ApprovalRefused(
        str(failing[0].problem),
        " ".join(f"{check.address} {_PROBLEM_WORDS[str(check.problem)]}." for check in failing),
    )


def _clock(at_ns: int) -> str:
    return datetime.fromtimestamp(at_ns / 1e9, tz=UTC).isoformat(timespec="seconds")


def card(
    request: WriteRequest,
    checks: list[ChannelCheck],
    *,
    current: Mapping[str, Any] | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """What a person approving *request* is shown, as JSON.

    Channels with ranges, limits bounds, current values (``None`` where none
    was read) and problems; steps, expiry, lease, holder, owner, label,
    completion; whether it can be approved and, if not, why; and
    :data:`LIMITS_COST_SENTENCE`.
    """
    now = time.time_ns() if now_ns is None else now_ns
    refused = refusal(checks)
    channels = []
    for check in checks:
        row = check.to_payload()
        row["current"] = None if current is None else current.get(check.address)
        channels.append(row)
    return {
        "id": request.id,
        "owner": request.owner,
        "holder": request.holder,
        "target": request.target,
        "label": request.label,
        "completion": request.completion,
        "steps": request.steps,
        "lease_s": request.lease_s,
        "requested_at": _clock(request.requested_at_ns),
        "expires_at": _clock(request.expires_at_ns),
        "expires_in_s": max(0.0, round((request.expires_at_ns - now) / 1e9, 1)),
        "channels": channels,
        "approvable": refused is None and now < request.expires_at_ns,
        "refusal": None
        if refused is None
        else {"reason": refused.reason, "phrase": refused.detail},
        "limits_note": LIMITS_COST_SENTENCE,
    }


def grant_summary(
    grant: Grant, *, used: int | None = None, now_ns: int | None = None
) -> dict[str, Any]:
    """One grant as a surface lists it: who holds it, on what, how much is left, until when.

    *used* is the steps spent, where the surface can read the holder's ledger.
    """
    now = time.time_ns() if now_ns is None else now_ns
    return {
        "id": grant.id,
        "owner": grant.owner,
        "holder": grant.holder,
        "target": grant.target,
        "label": grant.label,
        "steps": grant.steps,
        "used": used,
        "lease_s": grant.lease_s,
        "expires_at": _clock(grant.expires_at_ns),
        "expires_in_s": max(0.0, round((grant.expires_at_ns - now) / 1e9, 1)),
        "approver": {"identity": grant.approver.identity, "surface": grant.approver.surface},
        "channels": {
            address: {"min": bound.min, "max": bound.max}
            for address, bound in grant.channels.items()
        },
        "revoked": None
        if grant.revoked is None
        else {"by": grant.revoked.by, "at": _clock(grant.revoked.at_ns)},
    }
