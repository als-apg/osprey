"""Per-user brute-force throttle for the login route.

Each roster user carries one "next allowed attempt" timestamp. An attempt that
arrives before it is refused outright — the caller returns 429 and never touches
the stored credential. That ordering is the whole point: rejecting *before*
evaluation makes parallel guessing throughput-bounded rather than merely
latency-delayed, because a hundred concurrent requests cost the sidecar a dict
lookup each instead of a hundred scrypt derivations.

The window grows on each failed attempt (1s, 2s, 4s, … capped at 30s) and is
dropped entirely on success, so an operator who mistypes once pays a second and
an operator who types correctly pays nothing.

**No lockout, ever.** A control-room operator must never be shut out of the
terminals, so there is no failure count that latches. The window only ever
delays the next attempt, and it decays: a user quiet for ``forget_after`` seconds
is forgotten and starts again at ``initial_delay``. That also bounds memory,
since the login route keys on a requested username.

Framework-free on purpose — the login route maps :meth:`AttemptThrottle.retry_after`
onto the 429 and its ``Retry-After`` header. Time is injectable so tests need not
sleep.

Typical wiring::

    delay = throttle.retry_after(user)
    if delay > 0:
        return plain_429(retry_after=math.ceil(delay))   # no credential check
    if verify_password(user, submitted):
        throttle.record_success(user)
        ...
    else:
        throttle.record_failure(user)
        ...

A refused attempt must not call :meth:`record_failure`; only attempts that were
actually evaluated grow the window. Otherwise a flood of parallel requests would
ratchet the window against a legitimate operator, which is the lockout this
module exists to avoid.

That rule is about the *login* window. A caller that wants this escalation shape
for something other than deciding a login — the sidecar bounds how often its
audit ledger repeats a refusal that way, growing the window on refusals that
were never evaluated — must build its OWN instance rather than share the login
one (:func:`~osprey.services.auth_sidecar.app.get_audit_throttle`). Sharing it
would mean one object under two opposite rules, and an unauthenticated caller
could then delay a named operator's real login just by asking for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["AttemptThrottle"]

_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MULTIPLIER = 2.0
_DEFAULT_MAX_DELAY = 30.0
_DEFAULT_FORGET_AFTER = 300.0


@dataclass(slots=True)
class _UserState:
    """One user's current window: how long it is, and when it lifts."""

    delay: float
    next_allowed_at: float


class AttemptThrottle:
    """Tracks the next allowed login attempt per user.

    Times are durations from an arbitrary origin (``time.monotonic`` by default),
    never wall-clock instants, so a clock step cannot open or extend a window.

    Not thread-safe on its own; the sidecar's async request handlers touch it from
    a single event loop, and every method completes without awaiting.
    """

    def __init__(
        self,
        *,
        initial_delay: float = _DEFAULT_INITIAL_DELAY,
        multiplier: float = _DEFAULT_MULTIPLIER,
        max_delay: float = _DEFAULT_MAX_DELAY,
        forget_after: float = _DEFAULT_FORGET_AFTER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty throttle.

        Args:
            initial_delay: Window after the first failed attempt, in seconds.
            multiplier: Factor the window grows by on each further failure.
            max_delay: Ceiling on the window, in seconds. The cap is what bounds
                guessing throughput; it is deliberately low enough that a locked-out
                operator is never more than this many seconds from retrying.
            forget_after: Seconds of quiet, measured past the moment the window
                lifted, after which the user's escalation is discarded.
            clock: Returns a monotonically increasing seconds value. Injectable so
                tests can advance time without sleeping.

        Raises:
            ValueError: If the parameters could not produce a growing, capped
                window (non-positive delays, a multiplier below 1, or a cap below
                the initial delay).
        """
        if initial_delay <= 0:
            raise ValueError(f"initial_delay must be positive, got {initial_delay}")
        if multiplier < 1:
            raise ValueError(f"multiplier must be at least 1, got {multiplier}")
        if max_delay < initial_delay:
            raise ValueError(
                f"max_delay ({max_delay}) must be at least initial_delay ({initial_delay})"
            )
        if forget_after < 0:
            raise ValueError(f"forget_after must not be negative, got {forget_after}")

        self._initial_delay = initial_delay
        self._multiplier = multiplier
        self._max_delay = max_delay
        self._forget_after = forget_after
        self._clock = clock
        self._states: dict[str, _UserState] = {}

    @property
    def max_delay(self) -> float:
        """The window ceiling, in seconds — what bounds guessing throughput."""
        return self._max_delay

    def retry_after(self, user: str) -> float:
        """Seconds the caller must refuse ``user`` for; ``0.0`` when an attempt is allowed.

        A pure query: it never grows the window, so parallel requests arriving
        inside one window cannot ratchet it. Call it before evaluating any
        credential, and return 429 with this value (rounded up) as ``Retry-After``
        when it is positive.
        """
        state = self._states.get(user)
        if state is None:
            return 0.0
        remaining = state.next_allowed_at - self._clock()
        return remaining if remaining > 0 else 0.0

    def record_failure(self, user: str) -> float:
        """Record an evaluated attempt that failed and open the next window.

        The window is ``initial_delay`` on the first failure and grows by
        ``multiplier`` on each subsequent one, capped at ``max_delay``.

        Args:
            user: The roster user whose attempt failed.

        Returns:
            The length of the window just opened, in seconds.
        """
        self.purge_stale()
        now = self._clock()
        state = self._states.get(user)
        if state is None:
            delay = self._initial_delay
        else:
            delay = min(state.delay * self._multiplier, self._max_delay)
        self._states[user] = _UserState(delay=delay, next_allowed_at=now + delay)
        return delay

    def record_success(self, user: str) -> None:
        """Clear ``user``'s window after a successful login.

        A no-op for a user with no recorded failures.
        """
        self._states.pop(user, None)

    def purge_stale(self) -> int:
        """Forget users whose window lifted more than ``forget_after`` seconds ago.

        Called on each :meth:`record_failure`, which is what keeps the throttle
        bounded by recently-active users rather than by every username ever asked
        for. Exposed so a caller can sweep on its own schedule.

        Returns:
            How many users were forgotten.
        """
        cutoff = self._clock() - self._forget_after
        stale = [user for user, state in self._states.items() if state.next_allowed_at <= cutoff]
        for user in stale:
            del self._states[user]
        return len(stale)

    def clear(self) -> None:
        """Forget every user's window — equivalent to what a container restart does."""
        self._states.clear()

    def __len__(self) -> int:
        """How many users are being tracked, including any not yet swept."""
        return len(self._states)
