"""Per-run execution stats for the dispatch worker.

A module-level dict keyed by dispatch ``run_id`` holding counters that the run's
coroutine observes as it streams SDK messages (e.g. ``num_tool_calls``). The
value is that these counters remain readable from code paths that cannot see the
coroutine's locals: the timeout/cancel branches in ``dispatch_api`` build their
result dict without the runner's return value, so they otherwise have no honest
tool-call count to report. Both the runner (writer) and those stamp sites
(reader) go through this shared map.

The dispatch worker runs a single asyncio event loop; the run coroutine and the
stale-run sweep are cooperatively scheduled on it, never on separate threads, so
plain dict access is race-free and needs no lock.
"""

from __future__ import annotations

# run_id -> {"num_tool_calls": int}. Entries are created lazily on the first
# counted event and removed in dispatch_api._run_dispatch_task's finally.
_run_stats: dict[str, dict[str, int]] = {}


def increment_tool_calls(run_id: str) -> None:
    """Count one tool call for ``run_id``, creating its entry if absent."""
    stats = _run_stats.get(run_id)
    if stats is None:
        stats = {"num_tool_calls": 0}
        _run_stats[run_id] = stats
    stats["num_tool_calls"] += 1


def get_run_stats(run_id: str) -> dict[str, int]:
    """Return the live stats for ``run_id``, or a zeroed default if none exist.

    Read by the dispatch-API stamp sites; the returned dict is the live entry
    when present, so a caller sees the latest count.
    """
    return _run_stats.get(run_id, {"num_tool_calls": 0})


def pop_run_stats(run_id: str) -> dict[str, int]:
    """Remove and return ``run_id``'s stats (called once the run is done).

    Returns a zeroed default when no entry was ever created (e.g. a run that
    made no tool calls), so the caller need not special-case absence.
    """
    return _run_stats.pop(run_id, {"num_tool_calls": 0})
