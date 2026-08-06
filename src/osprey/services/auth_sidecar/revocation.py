"""In-memory revocation list for logged-out session ids.

A session cookie is signed, not stateful: once issued it stays cryptographically
valid until its recorded expiry. Logout therefore needs somewhere to record "this
session id is dead", and the verify endpoint consults that record on every
``auth_request`` subrequest.

**Process-local by design.** The sidecar runs single-process (uvicorn default), so
one dict in that process is the whole store — there is no shared cache to keep
coherent. Two consequences the deployment docs state plainly:

* Anything that recreates the container clears the list. A force-recreate on
  ``osprey deploy passwd``, a decommission re-render, or a podman image-drift
  reconcile all drop it. A cookie captured before a logout would work again after
  such a restart, bounded by ``auth.session_lifetime``.
* Password rotation does **not** rely on this store. Invalidation there rests on
  the credential-generation tag carried in each unlocked-user entry, which is
  recomputed from the current stored hash on every verify and so survives
  restarts. Revocation covers logout only.

Memory is bounded by logouts-per-session-lifetime, not by container uptime: an
entry is dropped once the clock passes the expiry that was recorded with it,
since a session that has expired is rejected on its own expiry check and no
longer needs a revocation record.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["RevocationStore"]


class RevocationStore:
    """Tracks revoked session ids until the moment they would have expired.

    Expiry timestamps are absolute epoch seconds, matching the per-user expiry
    carried in the session cookie, so callers can pass the cookie's value
    straight through.

    Not thread-safe on its own; the sidecar's async request handlers touch it
    from a single event loop, and every method completes without awaiting.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        """Create an empty store.

        Args:
            clock: Returns the current time as absolute epoch seconds. Injectable
                so tests can advance time without sleeping.
        """
        self._clock = clock
        self._revoked: dict[str, float] = {}

    def revoke(self, session_id: str, expires_at: float) -> None:
        """Record a session id as revoked until ``expires_at``.

        Re-revoking a known id keeps the later of the two expiries, so a
        refreshed cookie cannot shorten an existing record.

        Args:
            session_id: The session id from the cookie being logged out.
            expires_at: Absolute epoch seconds at which that session expires on
                its own. Passing an already-past value is harmless: the session
                is expired, :meth:`is_revoked` reports ``False`` for it, and the
                entry is dropped on the next sweep.
        """
        self.purge_expired()
        existing = self._revoked.get(session_id)
        if existing is None or expires_at > existing:
            self._revoked[session_id] = expires_at

    def is_revoked(self, session_id: str) -> bool:
        """Whether ``session_id`` was revoked and has not yet reached its expiry.

        Reports ``False`` once the recorded expiry has passed, whether or not the
        entry has been swept yet — the answer never depends on sweep timing. A
        caller reaching that point has an expired session anyway and rejects it on
        the expiry check.
        """
        expires_at = self._revoked.get(session_id)
        if expires_at is None:
            return False
        if self._clock() >= expires_at:
            del self._revoked[session_id]
            return False
        return True

    def purge_expired(self) -> int:
        """Drop every entry whose recorded expiry has passed.

        Called on each :meth:`revoke`, which is what keeps the store bounded by
        logouts-per-lifetime. Exposed so a caller can sweep on its own schedule.

        Returns:
            How many entries were dropped.
        """
        now = self._clock()
        stale = [sid for sid, expires_at in self._revoked.items() if now >= expires_at]
        for session_id in stale:
            del self._revoked[session_id]
        return len(stale)

    def clear(self) -> None:
        """Forget every revocation — equivalent to what a container restart does."""
        self._revoked.clear()

    def __len__(self) -> int:
        """How many entries are held, including any not yet swept."""
        return len(self._revoked)
