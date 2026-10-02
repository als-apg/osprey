"""Per-user brute-force throttle for the login route.

Each roster user carries one "next allowed attempt" timestamp. An attempt that
arrives before it is refused outright — the caller returns 429 and never touches
the stored credential. That ordering is the whole point: rejecting *before*
evaluation makes parallel guessing throughput-bounded rather than merely
latency-delayed, because a hundred concurrent requests cost the sidecar a dict
lookup each instead of a hundred scrypt derivations.

The window grows on each failed attempt and is dropped entirely on success. By
default it opens at 1 s and doubles up to a 30 s ceiling, forgotten after 300 s
of quiet, so an operator who mistypes once pays a second and an operator who
types correctly pays nothing. A deployment sets all four through
``modules.web_terminals.auth.throttle``.

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

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DEFAULT_FORGET_AFTER",
    "DEFAULT_INITIAL_DELAY",
    "DEFAULT_MAX_DELAY",
    "DEFAULT_MULTIPLIER",
    "THROTTLE_DEFAULTS",
    "AttemptThrottle",
    "throttle_problems",
]

DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MULTIPLIER = 2.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_FORGET_AFTER = 300.0

#: Each default, keyed by its ``AttemptThrottle`` keyword.
THROTTLE_DEFAULTS: dict[str, float] = {
    "initial_delay": DEFAULT_INITIAL_DELAY,
    "multiplier": DEFAULT_MULTIPLIER,
    "max_delay": DEFAULT_MAX_DELAY,
    "forget_after": DEFAULT_FORGET_AFTER,
}


def _usable_number(value: object) -> bool:
    """A finite real number, and not a bool (which Python counts as an int)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def throttle_problems(
    *, initial_delay: object, multiplier: object, max_delay: object, forget_after: object
) -> dict[str, str]:
    """Name every parameter the throttle cannot be built with.

    The one predicate for the throttle's parameters: the constructor, the
    sidecar's requirements check, the render refusal and the lint rule all call
    it, so no two surfaces can disagree on what a usable throttle is. A
    non-finite value is refused because NaN would disable the throttle and an
    infinite ceiling would eventually lock a user out.

    Args:
        initial_delay: Window after the first failed attempt, in seconds.
        multiplier: Factor the window grows by on each further failure.
        max_delay: Ceiling on the window, in seconds.
        forget_after: Seconds of quiet after which escalation is discarded.

    Returns:
        ``{parameter name: reason}``, empty when all four are usable. Each reason
        is plain words carrying the offending bound, so a caller prefixes the
        name its own surface uses.
    """
    values = {
        "initial_delay": initial_delay,
        "multiplier": multiplier,
        "max_delay": max_delay,
        "forget_after": forget_after,
    }
    numbers: dict[str, float] = {
        name: float(value)  # type: ignore[arg-type]  # _usable_number proved it real
        for name, value in values.items()
        if _usable_number(value)
    }
    problems = {
        name: "is not a finite number"
        if name == "multiplier"
        else "is not a finite number of seconds"
        for name in values
        if name not in numbers
    }
    if numbers.get("initial_delay", 1.0) <= 0:
        problems["initial_delay"] = "must be greater than zero"
    if numbers.get("multiplier", 1.0) < 1:
        problems["multiplier"] = "must be at least 1"
    if "max_delay" in numbers and "initial_delay" in numbers:
        if numbers["max_delay"] < numbers["initial_delay"]:
            problems["max_delay"] = f"must be at least the initial delay ({initial_delay} s)"
    if numbers.get("forget_after", 0.0) < 0:
        problems["forget_after"] = "must not be negative"
    return problems


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
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        multiplier: float = DEFAULT_MULTIPLIER,
        max_delay: float = DEFAULT_MAX_DELAY,
        forget_after: float = DEFAULT_FORGET_AFTER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty throttle.

        Args:
            initial_delay: Window after the first failed attempt, in seconds.
            multiplier: Factor the window grows by on each further failure.
            max_delay: Ceiling on the window, in seconds. The cap is what bounds
                guessing throughput; it is deliberately low enough that a throttled
                operator is never more than this many seconds from retrying.
            forget_after: Seconds of quiet, measured past the moment the window
                lifted, after which the user's escalation is discarded.
            clock: Returns a monotonically increasing seconds value. Injectable so
                tests can advance time without sleeping.

        Raises:
            ValueError: If the parameters could not produce a growing, capped
                window (non-positive delays, a multiplier below 1, a cap below
                the initial delay, a negative ``forget_after``), or if any is
                not a finite number or is a bool. The message names every
                parameter at fault.
        """
        problems = throttle_problems(
            initial_delay=initial_delay,
            multiplier=multiplier,
            max_delay=max_delay,
            forget_after=forget_after,
        )
        if problems:
            raise ValueError("; ".join(f"{name} {reason}" for name, reason in problems.items()))

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
