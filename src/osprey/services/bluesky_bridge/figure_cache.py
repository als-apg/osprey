"""Bounded cache of SETTLED run figures, invalidated on write.

``GET /runs/{id}/figure`` is polled: the panel asks again every tick while a run
is in flight, and keeps asking for a moment after it finishes. Rendering a
finished run's figure means reading its whole table back — from Tiled, over the
network — and computing the same picture from the same rows every time. This
module holds those finished pictures so a second GET on a settled run issues
zero source reads.

Only settled figures belong here. A run still producing rows has a different
figure a second later, so the route caches nothing for it; settledness is
decided by the SOURCE (live buffer no longer partial, or a stop document in
Tiled), never by the run record, whose terminal status can precede the last
rows and survives a re-run of an interrupted item under the same OSPREY run id.

**Recycled run ids are handled by invalidate-on-write.** Validating a cached
figure at read time would cost the Tiled search the cache exists to avoid, so
instead the *write* side invalidates: when a start document arrives for a run id
that already has entries (the operator re-ran an interrupted item), the document
plane calls :func:`evict`, which drops that id's entries for every source and
bumps a per-run generation counter. Stores are compare-and-set on that counter —
:func:`snapshot_generation` is read when the rows are snapshotted and handed
back to :func:`put`, which checks it inside the same lock acquisition as the
write. A figure computation already in flight across an eviction therefore
discards its now-stale result instead of re-poisoning a settled key; the loser
simply recomputes on the next poll.

**Lock discipline.** This is a two-thread structure: the route's threadpool
threads write and read, the document plane's dispatcher thread evicts. Every
get/put/evict is guarded by one module-level lock, mirroring ``live_rows._lock``.
The two locks are never nested, in either order: the document plane's eviction
runs after its ``live_rows`` write has released that lock, and the route never
holds this lock across ``live_rows.get()`` or any Tiled call. Nothing here calls
out to anything, so there is no lock ordering to get wrong — keep it that way.

Values are opaque: whatever the route stores is what a later get returns, by
reference and unvalidated. This module never inspects a figure, which keeps it
free of pydantic and of bluesky alike (like ``figure.py``), so it can be
unit-tested with plain sentinels.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

# Settled figures retained at once, oldest-*used* evicted first: a settled run
# being polled stays hot, and one nobody has asked about since falls out. Sized
# like `live_rows._MAX_RUNS` — the runs whose rows are still retained are
# roughly the runs whose figures are worth keeping.
_MAX_ENTRIES = 50

# Generation counters retained at once. Larger than the entry cap because a
# counter (one small int) is cheap and must outlive its run's cached entries:
# dropping one would reset that run id's generation to zero, which is exactly
# the value an in-flight computation may be holding. Dropping a counter
# therefore drops that run's entries too, so a cached entry always has a
# counter to be compared against (see `_forget_oldest_generation`). One
# residual window is accepted rather than designed away: a put still in
# flight while 500+ distinct runs churn through the cache could find its
# run's counter dropped and re-created at the very value it snapshotted, and
# land a stale figure — practically unreachable at a 1 Hz poll against a
# 50-entry cache.
_MAX_GENERATIONS = 500

FigureKey = tuple[str, str | None, str]
"""``(run_id, plan_name, source)``. The plan name is part of the key because a
run id resolved to a different plan is a different picture, and ``None`` (plan
unknown) is a key in its own right rather than a wildcard. ``source`` is
``"live"`` or ``"tiled"``; both variants of a run id are dropped together."""

_lock = Lock()
_entries: OrderedDict[FigureKey, Any] = OrderedDict()
_generations: OrderedDict[str, int] = OrderedDict()


def make_key(run_id: str, plan_name: str | None, source: str) -> FigureKey:
    """The cache key for *run_id*'s figure of *plan_name* read from *source*.

    A constructor rather than a bare tuple literal at each call site: the run id
    must be the first element for :func:`evict` to find it, and three strings in
    a row are easy to transpose.
    """
    return (run_id, plan_name, source)


def snapshot_generation(run_id: str) -> int:
    """*run_id*'s current generation, to be handed back to :func:`put`.

    Read this when the rows are snapshotted, not when the figure is finished:
    the value's whole job is to say which occupant of the run id those rows came
    from, so a later :func:`evict` can reject the store.
    """
    with _lock:
        return _generations.get(run_id, 0)


def get(key: FigureKey) -> Any | None:
    """The cached figure for *key*, or ``None`` if there is none.

    Returned by reference — this module stores what it was given and never
    copies it, so a caller must treat the result as immutable. A hit refreshes
    the entry's position, so a figure that keeps being asked for is not the one
    evicted to make room.
    """
    with _lock:
        if key not in _entries:
            return None
        _entries.move_to_end(key)
        return _entries[key]


def get_for_source(run_id: str, source: str) -> Any | None:
    """The cached figure for *run_id* read from *source*, whatever its plan name.

    The rotated branch of the figure route needs this: a run whose live buffer
    is gone has its plan name recorded only in the Tiled start document, so a
    lookup that required the full key would cost the very Tiled read the cache
    exists to avoid. A run-scoped read is unambiguous because at most one plan
    name can occupy a ``(run_id, source)`` pair at a time — every start
    document for the id evicts ALL of its names and bumps the generation
    (:func:`evict`), and a bridge restart clears the whole in-process cache —
    so this never has to choose between candidates. Stores still use the exact
    key (:func:`put`, generation-checked); this is a read-side accessor only,
    and like :func:`get` it refreshes the entry's LRU position, returns by
    reference, and never creates anything.
    """
    with _lock:
        for key in _entries:
            if key[0] == run_id and key[2] == source:
                _entries.move_to_end(key)
                return _entries[key]
        return None


def put(key: FigureKey, figure: Any, generation: int) -> bool:
    """Store *figure* under *key*, unless *key*'s run was evicted meanwhile.

    Args:
        key: What was rendered — see :data:`FigureKey`.
        figure: The settled figure, stored as-is and never inspected.
        generation: What :func:`snapshot_generation` returned for the run id at
            the time its rows were read.

    Returns:
        True if the figure was stored; False if the run id's generation moved on
        while the figure was being computed, in which case *figure* describes
        rows that are no longer what that run id means and is dropped. The
        caller has already served it to this requester — the next poll
        recomputes, which is the point.
    """
    run_id = key[0]
    with _lock:
        if _generations.get(run_id, 0) != generation:
            return False
        # Keep a counter alive for every run that has an entry, so an entry can
        # never outlive the generation it would be compared against.
        _generations.setdefault(run_id, generation)
        _generations.move_to_end(run_id)
        _entries[key] = figure
        _entries.move_to_end(key)
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last=False)
        while len(_generations) > _MAX_GENERATIONS:
            _forget_oldest_generation()
        return True


def evict(run_id: str) -> None:
    """Drop every cached figure for *run_id* and invalidate stores in flight.

    Called when a start document lands for a run id that may already be cached:
    the rows behind that id are about to be replaced, so both the entries built
    from the old rows and any computation still running over them are void. The
    generation bump is what voids the latter — see :func:`put`.

    Must not be called while holding ``live_rows._lock`` (the document plane
    calls it after its recorder write returns).
    """
    with _lock:
        for key in [key for key in _entries if key[0] == run_id]:
            del _entries[key]
        _generations[run_id] = _generations.get(run_id, 0) + 1
        _generations.move_to_end(run_id)
        while len(_generations) > _MAX_GENERATIONS:
            _forget_oldest_generation()


def _forget_oldest_generation() -> None:
    """Drop the least recently touched counter, and its run's entries (caller holds the lock)."""
    run_id, _ = _generations.popitem(last=False)
    for key in [key for key in _entries if key[0] == run_id]:
        del _entries[key]


def _clear() -> None:
    """Drop every entry and counter (test isolation only)."""
    with _lock:
        _entries.clear()
        _generations.clear()
