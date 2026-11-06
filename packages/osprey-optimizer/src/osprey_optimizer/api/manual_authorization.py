"""A one-step write authorization for each manual write the API makes.

The API's routes that move hardware — the results restore, ``PUT
/environments/{name}/variables`` and ``POST /environments/{name}/evaluate`` —
are the optimizer service writing as a program, so each goes out under a write
authorization of its own (:mod:`osprey_connectors.write_authorization`), as a
run does: exactly the channels it writes, each at exactly the value it writes
(``min == max``), one step. A restore and a run therefore meet one mechanism
and one check, and a manual write never rides a person's switch alone.

**One call.** The route issues that grant itself, which the owner's switch for
the machine allows only while it stands at ``allowed``; it writes once and
revokes the grant. Below ``allowed`` nothing is written: the call answers
``403`` with the refusal's reason and asks the operator to set the switch to
allowed. There is no request to approve and no call to repeat.

**The owner** is the ``X-Osprey-Owner`` header, the one a run's start reads:
the connector child that writes is stamped with it, so the write is judged by
that login's tier and record, and a call naming nobody is ``owner_unknown``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from fastapi import HTTPException, status

from osprey_optimizer.environments.write_authority import WriteAuthority, use_write_authority
from osprey_optimizer.models.config import ControlTarget, ExecutionMode
from osprey_optimizer.models.protocols import EnvironmentProtocol
from osprey_optimizer.services.write_authorization import retire_self_issued

logger = logging.getLogger(__name__)

__all__ = [
    "LEASE_S",
    "SET_ALLOWED",
    "SWITCH_NOT_ALLOWED",
    "authorize_manual_write",
    "setpoint_channels",
]

_T = TypeVar("_T")

#: The lease and the life of a manual write's grant: long enough to bring its
#: connector child up and write once.
LEASE_S = 120.0

#: The reason a manual write is refused with while the switch is on but not at
#: ``allowed`` (at ``asked``). The connectors' self-issue answers ``auth_revoked``
#: there, which reads as "an authorization this program held has ended"; a
#: manual write held none, so its caller is told what stands in its way. Local
#: to the optimizer: :mod:`osprey_connectors.refusals` names no such reason.
SWITCH_NOT_ALLOWED = "switch_not_allowed"

#: The refusals of a self-issue that mean the switch stands below ``allowed``.
_BELOW_ALLOWED = frozenset({"auth_revoked", "switch_off", "stopped_since", SWITCH_NOT_ALLOWED})

#: What a refusal below ``allowed`` asks of the operator.
SET_ALLOWED = (
    "A manual write goes out only while the write switch for this machine is at "
    "allowed: set it to allowed from the chip, then repeat this call."
)


def setpoint_channels(env: EnvironmentProtocol, variables: Mapping[str, float]) -> dict[str, float]:
    """``{setpoint channel: value}`` for *variables*, as the environment writes them.

    422 for a variable the environment does not have, or a value that is not a
    finite number inside the variable's range: nothing is asked for a write
    that cannot be made.
    """
    known = env.variables
    channels: dict[str, float] = {}
    for name, value in variables.items():
        meta = known.get(name)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown variable(s): [{name!r}]",
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Variable {name} value {value} is not finite",
            )
        if not meta.min_value <= value <= meta.max_value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Variable {name} value {value} is outside bounds "
                    f"[{meta.min_value}, {meta.max_value}]"
                ),
            )
        channel = meta.setpoint_pv
        if channel in channels and channels[channel] != float(value):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Two variables write {channel} with different values",
            )
        channels[channel] = float(value)
    return channels


def _refused(
    reason: str, detail: str | None, code: int = status.HTTP_403_FORBIDDEN
) -> HTTPException:
    from osprey_connectors.refusals import refusal_phrase

    if reason in _BELOW_ALLOWED:
        phrase = SET_ALLOWED
    else:
        try:
            phrase = refusal_phrase(reason)
        except ValueError:
            phrase = reason
    return HTTPException(
        status_code=code,
        detail={"reason": reason, "message": " ".join(p for p in (phrase, detail) if p)},
    )


def _control_system_section() -> Any:
    from osprey_connectors.config import get_config_value

    return get_config_value("control_system", None)


def authorize_manual_write(
    *,
    env: EnvironmentProtocol,
    variables: Mapping[str, float],
    target: ControlTarget,
    execution_mode: ExecutionMode | None,
    owner: str | None,
    label: str,
    write: Callable[[], _T],
) -> _T:
    """Run *write*, which sets *variables* on *env*, under a one-step authorization for exactly them.

    The grant is issued here, used for this one write and revoked. A readonly
    call needs none: its writes are refused ``readonly_run`` whatever it holds.

    Raises:
        HTTPException: ``422`` for a write that cannot be made
            (:func:`setpoint_channels`); ``403`` for a refusal, its reason in
            the detail (below ``allowed``, with :data:`SET_ALLOWED`); ``409``
            for a lease that lapsed before the write.
    """
    from osprey_connectors import write_authorization as wa

    if execution_mode is not None:
        with use_write_authority(WriteAuthority(owner=owner, required=False)):
            return write()
    channels = setpoint_channels(env, variables)
    try:
        grant = wa.self_issue(
            _control_system_section(),
            target,
            {channel: (value, value) for channel, value in channels.items()},
            steps=1,
            expires_at_ns=time.time_ns() + int(LEASE_S * 1e9),
            lease_s=LEASE_S,
            label=label,
            owner=owner,
        )
    except wa.AuthorizationRefused as refused:
        # A self-issue's auth_revoked means the switch was not at allowed (or
        # left it during the issue): no authorization of this call has ended.
        reason = SWITCH_NOT_ALLOWED if refused.reason == "auth_revoked" else refused.reason
        raise _refused(reason, refused.detail) from refused
    try:
        wa.start_lease(grant.id, owner=owner)
    except wa.AuthorizationRefused as refused:
        retire_self_issued(grant)
        code = (
            status.HTTP_409_CONFLICT
            if refused.reason == "auth_lease_lapsed"
            else status.HTTP_403_FORBIDDEN
        )
        raise _refused(refused.reason, refused.detail, code) from refused
    authority = WriteAuthority(owner=owner, authorization=grant.id)
    try:
        with use_write_authority(authority):
            return write()
    finally:
        authority.end("released")
        retire_self_issued(grant)
        _retire(target, owner, grant.id)


def _retire(target: str, owner: str | None, authorization: str) -> None:
    """Stop the connector child that named *authorization*. Never raises."""
    from osprey_optimizer.environments.interface import existing_bridge

    bridge = existing_bridge()
    if bridge is None:
        return
    try:
        bridge.retire(target, None, owner=owner, authorization=authorization)
    except Exception:  # its lease and its one step are spent; it can write nothing more
        logger.warning("Could not stop the connector child of %s", authorization, exc_info=True)
