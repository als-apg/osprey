"""A run's write authorization: what it asks for, how it waits, and how it holds it.

An optimization run writes to the machine only under one bounded write
authorization (:mod:`osprey_connectors.write_authorization`): the setpoint
channels it will write, a ``[min, max]`` per channel, a step budget, an
expiry and a lease the run renews. The machine's switch can stay off.

**What is asked** (:func:`write_plan`). The channels are the setpoints the run
writes, spelled exactly as written: each tuned variable's setpoint with the
run's resolved window, narrowed to the environment's range (which is already
the limits database's), and each fixed feature at the one value it is written
with. The steps are the resolved evaluations (``n_initial + n_iterations``),
each one ``write_multiple_channels`` call, plus one for the completion write:
``apply_best`` and an explicitly set ``restore`` are one step each, and
``hold`` writes nothing (``completion: none``). An evaluation whose objective is
re-measured re-reads it without writing again, so it spends one step however
many tries it takes.

**How it is had** (:meth:`RunAuthorization.acquire`). While the owner's
switch for the machine stands at ``allowed`` the run writes its own grant
(:func:`~osprey_connectors.write_authorization.self_issue`). Below
``allowed`` (``auth_revoked``, or ``switch_off`` with the switch off) it writes
a request and waits, its job showing ``awaiting``, until a person approves it
from the chip or the CLI; a cancel while it waits withdraws the request, and
so does a wait that outlasts :data:`WAIT_S`. Any other refusal — no owner, a
tier that allows nothing, an unreadable record — ends the run with that
reason before it writes anything.

**How it is held.** The lease is started once, by the worker, and renewed by
a heartbeat at a third of :data:`LEASE_S`. A renewal that fails ends the
authority: the run's next write is refused before any connector is asked, and
the run ends with the reason (``auth_revoked``, ``auth_expired`` or
``auth_lease_lapsed``). A write the connector child refuses with any ``auth_*``
reason ends it the same way (:class:`~osprey_optimizer.environments.interface.ConnectorInterface`).
Nothing asks again.

**The end.** :meth:`RunAuthorization.release` stops the heartbeat, withdraws a
request still pending, marks a self-issued grant revoked (it is in the run's
own directory; one a person approved is left to its lease and expiry), and
stops the connector child that names the authorization. With the heartbeat
stopped the lease lapses within :data:`LEASE_S`, and from then on a person's
grant no longer blocks another program's approval on its channels
(:func:`~osprey_connectors.write_authorization.auth_overlap`). It writes
nothing to the machine: a cancelled or failed run holds the machine where it
is.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from osprey_optimizer.environments.write_authority import WriteAuthority
from osprey_optimizer.models.config import OnComplete
from osprey_optimizer.models.exceptions import AUTHORIZATION_TIMEOUT, AuthorizationEndedError
from osprey_optimizer.models.protocols import EnvironmentProtocol
from osprey_optimizer.utils.interval_thread import IntervalThread

logger = logging.getLogger(__name__)

__all__ = [
    "ASK_INSTEAD",
    "EXPIRES_S",
    "EXPIRY_MARGIN_S",
    "LEASE_S",
    "POLL_S",
    "WAIT_S",
    "Completion",
    "RunAuthorization",
    "WritePlan",
    "completion_for",
    "retire_self_issued",
    "write_plan",
]

Completion = Literal["none", "apply_best", "restore"]

#: How long a run's authorization lasts at most, from the moment it is asked for.
#: A value past the connectors' ceiling (:data:`~osprey_connectors.write_authorization.MAX_EXPIRES_S`,
#: a day) is held to just under it (:data:`EXPIRY_MARGIN_S`) when the run asks.
EXPIRES_S = float(os.environ.get("OPTIMIZER_AUTHORIZATION_EXPIRES_S", 8 * 3600))
#: How far under the ceiling a held expiry stays: an approver's clock behind the
#: run's would otherwise see the expiry more than a day after its issue.
EXPIRY_MARGIN_S = 60.0
#: How long a request waits for a person's approval before the run ends.
WAIT_S = float(os.environ.get("OPTIMIZER_AUTHORIZATION_WAIT_S", 15 * 60))
#: How long one renewal of the run's lease lasts; the heartbeat renews at a third of it.
LEASE_S = 60.0
#: How often a waiting run looks for its grant.
POLL_S = 2.0

#: The reason a run's authority ends with once the run is over, when nothing
#: ended it before.
_RELEASED = "released"

#: The refusals of a self-issue that mean "ask a person instead": the switch is
#: below ``allowed`` (``auth_revoked``) or off (``switch_off``).
ASK_INSTEAD = frozenset({"auth_revoked", "switch_off"})


def completion_for(on_complete: OnComplete) -> Completion:
    """The authorization's completion word for a run's ``on_complete``."""
    return "none" if on_complete == "hold" else on_complete


@dataclass(frozen=True)
class WritePlan:
    """What a run asks to write: its channels and ranges, its evaluations and its completion."""

    target: str
    channels: Mapping[str, tuple[float, float]]
    evaluations: int
    completion: Completion
    label: str = ""

    @property
    def steps(self) -> int:
        """The write calls the run makes: one per evaluation, plus its completion's."""
        return self.evaluations + (0 if self.completion == "none" else 1)


def _narrow(
    channels: dict[str, tuple[float, float]], channel: str, low: float, high: float, what: str
) -> None:
    """Record *channel*'s range, intersected with any range already recorded for it."""
    if channel in channels:
        low, high = max(low, channels[channel][0]), min(high, channels[channel][1])
    if not low <= high:
        raise ValueError(f"{what}: its setpoint {channel} has no value the run may write")
    channels[channel] = (float(low), float(high))


def write_plan(
    env: EnvironmentProtocol,
    windows: Mapping[str, tuple[float, float]],
    *,
    target: str,
    evaluations: int,
    on_complete: OnComplete,
    fixed: Mapping[str, float] | None = None,
    label: str = "",
    limits: Mapping[str, Any] | None = None,
) -> WritePlan:
    """The authorization a run asks for, from its resolved windows.

    Args:
        env: The run's environment; its variables name the setpoint channels
            and the range each may take (the environment's own, narrowed to
            the limits database).
        windows: ``{variable: (min, max)}``, the run's resolved search window.
        target: The control target.
        evaluations: The resolved evaluation count, ``n_initial + n_iterations``.
        on_complete: The run's ``on_complete``.
        fixed: The fixed features written on every candidate, ``{variable: value}``.
        label: The run, shown to the approver.
        limits: The deployment's channel limits for the target
            (:func:`~osprey_optimizer.environments.connector_env.deployment_limits`),
            or ``None`` when limits checking is off. Every range is narrowed to
            its channel's entry, so an authorization never reaches past the
            limits database, whatever environment the run drives.

    Raises:
        ValueError: A variable is not the environment's, its window does not
            meet the environment's range or its channel's limits, or its
            channel is read-only in the limits database.
    """
    variables = env.variables
    channels: dict[str, tuple[float, float]] = {}
    for name, (low, high) in windows.items():
        meta = variables.get(name)
        if meta is None:
            raise ValueError(f"Variable {name!r} is not one of the environment's")
        _narrow(
            channels,
            meta.setpoint_pv,
            max(low, meta.min_value),
            min(high, meta.max_value),
            f"Variable {name!r}",
        )
    for name, value in (fixed or {}).items():
        meta = variables.get(name)
        if meta is None:
            raise ValueError(f"Fixed feature {name!r} is not one of the environment's variables")
        _narrow(channels, meta.setpoint_pv, value, value, f"Fixed feature {name!r}")
    for channel, (low, high) in list(channels.items()):
        entry = (limits or {}).get(channel)
        if entry is None:
            continue
        if not getattr(entry, "writable", True):
            raise ValueError(f"{channel} is read-only in the channel limits database")
        floor = getattr(entry, "min_value", None)
        ceiling = getattr(entry, "max_value", None)
        del channels[channel]
        _narrow(
            channels,
            channel,
            low if floor is None else max(low, floor),
            high if ceiling is None else min(high, ceiling),
            f"Channel {channel} within its limits",
        )
    return WritePlan(
        target=target,
        channels=channels,
        evaluations=evaluations,
        completion=completion_for(on_complete),
        label=label,
    )


def retire_self_issued(grant: Any) -> None:
    """Mark *grant* revoked if this holder issued it itself. Never raises.

    A self-issued grant lives in the holder's own ledger directory, so the
    holder can end it once it is done with it; otherwise it would stand,
    unusable (its lease has lapsed) but shown as in force, until its expiry.
    A grant a person approved is in the owner's directory and is left to its
    lease and its expiry.
    """
    from osprey_connectors import write_authorization as wa
    from osprey_connectors.identity import acting_identity

    if grant is None or grant.approver.surface != wa.SELF_ISSUE_SURFACE:
        return
    try:
        wa.revoke_grants(by=acting_identity(), directory=wa.ledger_dir(), ids=[grant.id])
    except Exception:  # its lease lapses and it expires on its own
        logger.warning("Could not mark self-issued grant %s ended", grant.id, exc_info=True)


def _control_system_section() -> Any:
    from osprey_connectors.config import get_config_value

    return get_config_value("control_system", None)


class RunAuthorization:
    """One run's write authorization, from asking for it to the end of its lease.

    Args:
        authority: The worker's process authority; it names the authorization
            once held, and is ended when the authorization ends.
        owner: The login the run is for, or ``None`` for the owner ladder of
            this process (a worker stamps its run's owner in its environment).
        publish: Records the authorization's state on the job, for its status.
        check_cancellation: Raises ``CancellationError`` once the run is
            cancelled; asked while the run waits for its approval.
    """

    def __init__(
        self,
        authority: WriteAuthority,
        *,
        owner: str | None = None,
        publish: Callable[[dict[str, Any] | None], None] | None = None,
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        self._authority = authority
        self._owner = owner
        self._publish_fn = publish
        self._check_cancellation = check_cancellation
        self._state: dict[str, Any] = {}
        self._pending: str | None = None
        self._grant_id: str | None = None
        self._issued: Any = None
        self._target: str | None = None
        self._heartbeat: IntervalThread | None = None
        authority.on_end = self._ended

    # -- state --------------------------------------------------------------

    def _publish(self, **fields: Any) -> None:
        self._state.update(fields)
        if self._publish_fn is None:
            return
        try:
            self._publish_fn(dict(self._state))
        except Exception:  # the job's status is a report; it never stops the run
            logger.warning("Could not record the write authorization's state", exc_info=True)

    def _ended(self, reason: str, detail: str | None) -> None:
        """The authority ended, whoever ended it: record why on the job."""
        if reason == _RELEASED:
            if self._grant_id is not None:
                self._publish(state="released")
            elif self._state.get("state") in ("requesting", "awaiting"):
                # The run is over before anything was granted (a cancel while it
                # waited): its request is withdrawn, so the job no longer awaits.
                self._publish(
                    state="ended",
                    reason=_RELEASED,
                    detail="The run ended before its request was approved; the request is withdrawn.",
                )
            return
        logger.warning("The run's write authorization ended: %s %s", reason, detail or "")
        self._publish(state="ended", reason=reason, detail=detail)

    def end(self, reason: str, detail: str | None = None) -> AuthorizationEndedError:
        """End the authority for *reason*, and answer the error the run ends with."""
        self._authority.end(reason, detail)
        return AuthorizationEndedError(self._authority.ended or reason, detail)

    # -- acquiring ----------------------------------------------------------

    def acquire(self, plan: WritePlan) -> None:
        """Hold an authorization for *plan* before the run's first write.

        Raises:
            AuthorizationEndedError: None could be had, or it ended at once.
            CancellationError: The run was cancelled while it waited; the
                request is withdrawn.
        """
        from osprey_connectors import write_authorization as wa

        self._target = plan.target
        self._publish(
            state="requesting",
            target=plan.target,
            channels={channel: list(bounds) for channel, bounds in plan.channels.items()},
            steps=plan.steps,
            completion=plan.completion,
            id=None,
            approver=None,
            reason=None,
            detail=None,
        )
        # Under the ceiling by a margin, so an approver's lagging clock still reads it.
        expires_s = min(EXPIRES_S, wa.MAX_EXPIRES_S - EXPIRY_MARGIN_S)
        expires_at_ns = time.time_ns() + int(expires_s * 1e9)
        try:
            grant = wa.self_issue(
                _control_system_section(),
                plan.target,
                plan.channels,
                steps=plan.steps,
                expires_at_ns=expires_at_ns,
                lease_s=LEASE_S,
                label=plan.label,
                owner=self._owner,
            )
        except wa.AuthorizationRefused as refused:
            if refused.reason not in ASK_INSTEAD:
                raise self.end(refused.reason, refused.detail) from refused
            grant = self._ask(plan, expires_at_ns)
        except Exception as exc:  # a self-issue that could not be made gives nothing
            raise self.end("auth_unreadable", str(exc)) from exc
        self._hold(grant)

    def _ask(self, plan: WritePlan, expires_at_ns: int) -> Any:
        """Write a request and wait for its grant; the grant, or the error the run ends with."""
        from osprey_connectors import write_authorization as wa

        try:
            request = wa.write_request(
                plan.target,
                plan.channels,
                steps=plan.steps,
                expires_at_ns=expires_at_ns,
                lease_s=LEASE_S,
                label=plan.label,
                completion=plan.completion,
                owner=self._owner,
            )
        except wa.AuthorizationRefused as refused:
            raise self.end(refused.reason, refused.detail) from refused
        except Exception as exc:  # a request that could not be written asks nobody
            raise self.end("auth_unreadable", str(exc)) from exc
        self._pending = request.id
        self._publish(state="awaiting", id=request.id, owner=request.owner)
        logger.info(
            "Awaiting approval of write authorization %s for %s (%d step(s), %d channel(s))",
            request.id,
            request.owner,
            plan.steps,
            len(plan.channels),
        )
        deadline = time.monotonic() + WAIT_S
        try:
            while True:
                if self._check_cancellation is not None:
                    self._check_cancellation()
                try:
                    grant = wa.approved_grant(request.id, owner=self._owner)
                except wa.AuthorizationRefused as refused:
                    raise self.end(refused.reason, refused.detail) from refused
                if grant is not None:
                    return grant
                if time.monotonic() >= deadline:
                    raise self.end(
                        AUTHORIZATION_TIMEOUT,
                        f"Nobody approved request {request.id} within {WAIT_S:.0f} s.",
                    )
                time.sleep(POLL_S)
        finally:
            self._withdraw()

    def _withdraw(self) -> None:
        """Remove the request still pending, if any. Never raises."""
        from osprey_connectors import write_authorization as wa

        request, self._pending = self._pending, None
        if request is None:
            return
        try:
            wa.withdraw_request(request)
        except Exception:  # a request left behind expires; it never authorizes anything
            logger.warning("Could not withdraw write request %s", request, exc_info=True)

    def _hold(self, grant: Any) -> None:
        """Start the lease, name the authorization to the writes, and start the heartbeat."""
        from osprey_connectors import write_authorization as wa

        self._issued = grant
        try:
            wa.start_lease(grant.id, owner=self._owner)
        except wa.AuthorizationRefused as refused:
            raise self.end(refused.reason, refused.detail) from refused
        except Exception as exc:  # a lease that could not be opened authorizes nothing
            raise self.end("auth_unreadable", str(exc)) from exc
        self._grant_id = grant.id
        self._authority.authorization = grant.id
        self._publish(
            state="granted",
            id=grant.id,
            owner=grant.owner,
            approver=grant.approver.surface,
            expires_at_ns=grant.expires_at_ns,
        )
        logger.info(
            "Holding write authorization %s (approved from %s), %d step(s)",
            grant.id,
            grant.approver.surface,
            grant.steps,
        )
        self._heartbeat = IntervalThread(
            self._renew,
            interval=LEASE_S / 3,
            name=f"write-authorization-lease-{grant.id}",
            swallow_errors=True,
        )
        self._heartbeat.start()

    def _renew(self) -> None:
        """One heartbeat: renew the lease, or end the authority with the reason it lapsed."""
        from osprey_connectors import write_authorization as wa

        grant_id = self._grant_id
        if grant_id is None or self._authority.ended is not None:
            return
        if wa.renew_lease(grant_id, owner=self._owner):
            return
        reason = "auth_lease_lapsed"
        try:
            wa.approved_grant(grant_id, owner=self._owner)
        except wa.AuthorizationRefused as refused:
            reason = refused.reason
        except Exception:  # the lease is what failed; that is the reason given
            logger.debug("Could not read why %s lapsed", grant_id, exc_info=True)
        self.end(reason, "The lease could not be renewed; the run makes no further write.")

    # -- the end ------------------------------------------------------------

    def release(self) -> None:
        """Stop holding the authorization. Writes nothing to the machine; never raises."""
        if self._heartbeat is not None:
            self._heartbeat.stop(timeout=5)
            self._heartbeat = None
        self._withdraw()
        self._authority.end(_RELEASED)
        grant_id, self._grant_id = self._grant_id, None
        issued, self._issued = self._issued, None
        retire_self_issued(issued)
        if grant_id is None or self._target is None:
            return
        from osprey_optimizer.environments.interface import existing_bridge

        bridge = existing_bridge()
        if bridge is None:
            return
        try:
            bridge.retire(self._target, None, owner=self._authority.owner, authorization=grant_id)
        except Exception:  # the child's lease lapses on its own
            logger.warning("Could not stop the connector child of %s", grant_id, exc_info=True)
