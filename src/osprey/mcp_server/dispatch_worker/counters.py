"""Process-lifetime failure counters for the dispatch worker.

Monotonic per-failure-class error counts, incremented at *stamp time* by wiring
:func:`increment` into ``failure_class._stamp`` through its
``register_counter_hook`` seam (see :func:`install`).

Why a dedicated counter rather than deriving counts from ``dispatch_api._runs``:
that map is capacity-bounded (old entries are evicted once it exceeds
``_MAX_RUNS``) and its persisted-run reload is lexical and truncated to the same
cap, so any total derived from it is **non-monotonic** — it can shrink as runs
age out and understates lifetime failures. These counters only ever increase
within a process and reset to zero on restart, which the worker's boot nonce
makes observable to pollers (a nonce change means the counters restarted).

The dispatch worker runs a single asyncio event loop; ``_stamp`` (writer) and
the ``/health`` handler (reader) are cooperatively scheduled on it and never
touch these counters across an ``await``, so plain dict access is race-free and
needs no lock — matching ``run_stats``.
"""

from __future__ import annotations

from osprey.mcp_server.dispatch_worker.failure_class import FAILURE_CLASSES, register_counter_hook

# One monotonic counter per known failure class. Seeded with every valid class
# so a reader never has to special-case an absent key.
_counts: dict[str, int] = dict.fromkeys(FAILURE_CLASSES, 0)


def increment(failure_class: str) -> None:
    """Count one stamped error of ``failure_class`` (the counter-hook callback).

    Tolerates an unrecognised class by counting it under its own key, so an
    unexpected value can never silently drop a failure from the lifetime total.
    """
    _counts[failure_class] = _counts.get(failure_class, 0) + 1


def get_counts() -> dict[str, int]:
    """Return a snapshot copy of the process-lifetime per-class counters."""
    return dict(_counts)


def install() -> None:
    """Wire :func:`increment` into ``failure_class._stamp`` via the seam.

    Called once at worker startup. After this, every ``_stamp`` — at any
    dispatch error site — bumps the matching lifetime counter here.
    """
    register_counter_hook(increment)


def reset() -> None:
    """Restore the pristine seeded state (known classes only, all zero).

    For test isolation only — never called in production. Drops any stray
    key added via :func:`increment`'s unknown-class tolerance so a reset
    always yields the exact seeded shape.
    """
    _counts.clear()
    _counts.update(dict.fromkeys(FAILURE_CLASSES, 0))
