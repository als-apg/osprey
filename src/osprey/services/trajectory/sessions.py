"""Sessions: one holder per service target, on the action layer.

A holder is an owner, the door it came through, a ``client_id`` bound to that
``(owner, kind)``, and the stop event the action layer keeps for it. Sessions
live in memory only.

- **Open** is a worker action (:meth:`ActionLayer.submit`): it is admitted,
  probed and run under the lock like any other. Its precondition re-checks in
  the worker that the target has no holder, so of two queued opens the second
  is 409 ``session_held``, naming the holder's door and owner. Open is the only
  way to acquire a holder.
- **Take-over** acts on the control path (:meth:`ActionLayer.control`) after
  ``admit``, in one in-memory step with nothing awaited: it sets the old
  holder's stop event, voids its ``client_id`` and installs the new holder
  with a clear stop event. It never probes and never takes the run lock, so it
  is answered while an action is in flight and on an unreachable target. With
  no holder it is 409 ``no_session``.
- **Release** acts on the control path without ``admit`` and checks only that
  the caller's owner and ``client_id`` are the holder's, so a narrowed holder
  or the holder of an unreachable target can release.
- **A call** is a worker action whose precondition checks the caller's
  ``client_id`` in the worker: a voided one is 409 ``session_taken_over``, an
  unknown one, or one bound to another owner or door, is 409
  ``not_session_holder``.

The door passes ``kind``; it is never read from a request body. A body that
carries any key is 400: ``owner_in_body`` for ``owner``, ``bad_request`` for
anything else. The state view names the holder's owner and door and never its
``client_id``.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from osprey.services.trajectory.actions import Action, ActionLayer, Precondition
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.refusal import Refused

__all__ = [
    "BAD_REQUEST",
    "NOT_SESSION_HOLDER",
    "NO_SESSION",
    "OWNER_IN_BODY",
    "SESSION_HELD",
    "SESSION_TAKEN_OVER",
    "Holder",
    "SessionGrant",
    "SessionRefused",
    "SessionRelease",
    "Sessions",
    "check_body",
]

SESSION_HELD = "session_held"
NO_SESSION = "no_session"
NOT_SESSION_HOLDER = "not_session_holder"
SESSION_TAKEN_OVER = "session_taken_over"
OWNER_IN_BODY = "owner_in_body"
BAD_REQUEST = "bad_request"

#: The HTTP status each refusal raised here answers with.
_STATUS = {
    SESSION_HELD: 409,
    NO_SESSION: 409,
    NOT_SESSION_HOLDER: 409,
    SESSION_TAKEN_OVER: 409,
    OWNER_IN_BODY: 400,
    BAD_REQUEST: 400,
}

T = TypeVar("T")


class SessionRefused(Refused):
    """A session request refused; its code is one of this module's constants."""

    statuses = _STATUS


@dataclass(frozen=True)
class Holder:
    """The one holder of a service target.

    Attributes:
        owner: Who holds it, as admitted.
        kind: The door it was opened or taken over through.
        client_id: The token later calls carry, bound to ``(owner, kind)``.
        stop: The stop event the action layer keeps for this holder.
    """

    owner: str
    kind: str
    client_id: str
    stop: asyncio.Event


@dataclass(frozen=True)
class SessionGrant:
    """What an open or a take-over answers with."""

    client_id: str
    owner: str
    target: str

    def as_body(self) -> dict[str, str]:
        """The response body: ``client_id``, ``owner`` and ``target``."""
        return {"client_id": self.client_id, "owner": self.owner, "target": self.target}


@dataclass(frozen=True)
class SessionRelease:
    """What a release's audit record carries: ``owner``, ``target`` and ``client_id``."""

    owner: str
    target: str
    client_id: str

    def as_record(self) -> dict[str, str]:
        """The audit record of the release."""
        return {"owner": self.owner, "target": self.target, "client_id": self.client_id}


def check_body(body: Mapping[str, Any] | None) -> None:
    """Refuse a session request body that carries any key.

    The owner comes from the door's identity and the door from its route, so a
    session request has nothing to say in its body.

    Raises:
        SessionRefused: ``owner_in_body`` when the body names an owner,
            ``bad_request`` for any other key or a body that is not a mapping.
    """
    if body is None:
        return
    if not isinstance(body, Mapping):
        raise SessionRefused(BAD_REQUEST, "a session request carries no body")
    if "owner" in body:
        raise SessionRefused(
            OWNER_IN_BODY,
            "the owner is the identity the request arrived with, never a body field",
        )
    if body:
        keys = ", ".join(sorted(repr(key) for key in body))
        raise SessionRefused(BAD_REQUEST, f"a session request carries no body fields; got {keys}")


class Sessions:
    """The sessions of one started service.

    Attributes:
        layer: The action layer the sessions act through.
    """

    def __init__(self, layer: ActionLayer) -> None:
        self.layer = layer
        self._holder: Holder | None = None
        self._voided: set[str] = set()

    @property
    def target(self) -> str:
        """The service target the one holder holds."""
        return self.layer.startup.target

    @property
    def holder(self) -> Holder | None:
        """The current holder, if any."""
        return self._holder

    # -- open ----------------------------------------------------------------

    async def open(
        self, owner: Any, kind: str, body: Mapping[str, Any] | None = None
    ) -> SessionGrant:
        """Open a session for *owner* through door *kind*, as a worker action.

        Raises:
            SessionRefused: The body carries a key, or the target has a holder.
            AdmitRefused: ``admit`` refused the owner.
            ActionRefused: The worker refused the open.
        """
        check_body(body)

        async def install(admission: Admission, stop: asyncio.Event) -> SessionGrant:
            return self._install(admission, stop)

        return await self.layer.submit(owner, kind, install, precondition=self._unheld)

    def _unheld(self, _admission: Admission) -> None:
        holder = self._holder
        if holder is not None:
            raise SessionRefused(
                SESSION_HELD,
                f"target {self.target!r} is held by {holder.owner!r} through the "
                f"{holder.kind!r} door; take it over to act here",
            )

    # -- take-over -----------------------------------------------------------

    def take_over(
        self, owner: Any, kind: str, body: Mapping[str, Any] | None = None
    ) -> SessionGrant:
        """Take the target over for *owner* through door *kind*, on the control path.

        Raises:
            SessionRefused: The body carries a key, or the target has no holder.
            AdmitRefused: ``admit`` refused the owner.
        """
        check_body(body)
        return self.layer.control(owner, kind, self._swap)

    def _swap(self, admission: Admission | None) -> SessionGrant:
        if admission is None:  # the control path admits every take-over
            raise RuntimeError("a take-over runs only after admit")
        old = self._holder
        if old is None:
            raise SessionRefused(
                NO_SESSION,
                f"target {self.target!r} has no holder to take over; open a session instead",
            )
        stop = self.layer.renew_stop_event()
        self._voided.add(old.client_id)
        return self._install(admission, stop)

    def _install(self, admission: Admission, stop: asyncio.Event) -> SessionGrant:
        client_id = secrets.token_urlsafe(24)
        self._holder = Holder(
            owner=admission.owner, kind=admission.kind, client_id=client_id, stop=stop
        )
        return SessionGrant(client_id=client_id, owner=admission.owner, target=admission.target)

    # -- release -------------------------------------------------------------

    def release(
        self,
        owner: Any,
        kind: str,
        client_id: str | None,
        body: Mapping[str, Any] | None = None,
    ) -> SessionRelease:
        """Release the session, on the control path and without ``admit``.

        Raises:
            SessionRefused: The body carries a key, or *owner* and *client_id*
                are not the holder's.
        """
        check_body(body)

        def drop(_admission: Admission | None) -> SessionRelease:
            holder = self._holder
            if holder is None or holder.owner != owner or holder.client_id != client_id:
                raise _non_holder(client_id, self._voided)
            self._holder = None
            return SessionRelease(owner=holder.owner, target=self.target, client_id=client_id)

        return self.layer.control(owner, kind, drop, admit=False)

    # -- calls ---------------------------------------------------------------

    def precondition(self, client_id: str | None) -> Precondition:
        """The worker check a holder's call passes, for the caller's *client_id*."""

        def check(admission: Admission) -> None:
            holder = self._holder
            if (
                holder is None
                or holder.client_id != client_id
                or holder.owner != admission.owner
                or holder.kind != admission.kind
            ):
                raise _non_holder(client_id, self._voided)

        return check

    async def call(self, owner: Any, kind: str, client_id: str | None, action: Action[T]) -> T:
        """Run *action* for the holder as a worker action.

        Raises:
            SessionRefused: ``session_taken_over`` or ``not_session_holder``,
                checked in the worker.
            AdmitRefused: ``admit`` refused the owner.
            ActionRefused: The worker refused the action.
        """
        return await self.layer.submit(
            owner, kind, action, precondition=self.precondition(client_id)
        )

    # -- state ---------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The holder's owner and door for a state view; never its ``client_id``."""
        holder = self._holder
        return {
            "target": self.target,
            "owner": holder.owner if holder is not None else None,
            "kind": holder.kind if holder is not None else None,
        }


def _non_holder(client_id: str | None, voided: set[str]) -> SessionRefused:
    if client_id is not None and client_id in voided:
        return SessionRefused(
            SESSION_TAKEN_OVER,
            "this session was taken over; open or take over a session to act again",
        )
    return SessionRefused(NOT_SESSION_HOLDER, "the caller does not hold this target's session")
