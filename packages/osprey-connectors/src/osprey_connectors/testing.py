"""Hooks for tests of code that reads the control state.

Nothing in the package or the framework calls these: they exist so a test can
put a process back the way it started between cases, which production never
needs to do.
"""

from __future__ import annotations

from osprey_connectors import control_context

__all__ = ["invalidate_cache"]


def invalidate_cache() -> None:
    """Forget every control-context record and server report this process has parsed.

    Every read re-stats its file and re-parses only when the signature moved
    (rule 4 of :mod:`~osprey_connectors.control_context`), so a test that writes
    a file inside one filesystem clock tick of the last, or that moves the
    agent-data root under a reader, drops the cache through this before it reads.
    """
    control_context._invalidate_cache()
