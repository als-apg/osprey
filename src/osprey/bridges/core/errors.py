"""Exceptions the bridge engine and its channel adapters share.

One class today. Every other failure an adapter's outbound member raises is opaque to
the engine on purpose: by the failure contracts in :mod:`osprey.bridges.core.ports`, a
raise from ``post_answer`` or ``post_giveup`` means "the message did not land — keep
the entry queued and try again next cycle", whatever the exception was. That is the
right response to an outage. It is the wrong response to a refusal the platform will
repeat on every attempt, and :class:`UndeliverableError` is how an adapter says so.
"""

from __future__ import annotations

__all__ = ["UndeliverableError"]


class UndeliverableError(Exception):
    """The channel permanently refuses this entry's destination.

    Raised (by an adapter's client, through its outbound members) when the platform
    says the destination itself is closed to the app or gone — Google Chat's
    ``403 "This Chat app is not a member of this space."`` after the app was removed
    from a space, or a ``404`` for a space that no longer exists — as opposed to a
    transport or service failure that a later attempt may clear.

    On any other raise the engine leaves the entry queued and retries the post next
    cycle (see the failure contracts in :mod:`osprey.bridges.core.ports`). On this one it
    settles the entry terminal at once, stamping ``give_up_reason`` with
    ``undeliverable: <the platform's reason>``: retrying can never deliver, and an entry
    kept queued by a refusal it cannot outlive would otherwise fail the same post once
    per drain pass until its lifetime cap.

    Adapters raise it only for refusals that are specific to the destination. A missing
    OAuth scope is a ``403`` too, but a bridge misconfiguration that one fix clears for
    every destination at once — that stays an ordinary raise, and stays queued.
    """
