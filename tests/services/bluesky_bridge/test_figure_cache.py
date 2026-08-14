"""Unit tests for the settled-figure cache (`figure_cache.py`).

Figures are opaque to the cache, so these tests store plain sentinels rather
than real `Figure` objects — anything the cache did to a value would show up as
a failure here. The threaded sections drive the two-thread shape the module is
built for: route threads snapshotting a generation, computing, and storing,
against document-plane threads evicting the same run id.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from osprey.services.bluesky_bridge import document_plane, figure_cache, live_rows
from osprey.services.bluesky_bridge.document_plane import RunDocumentRouter
from osprey.services.bluesky_bridge.queue_backend import PLAN_META_KEY, RUN_ID_META_KEY


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Every test gets an empty module-level cache."""
    figure_cache._clear()
    live_rows._clear()
    yield
    figure_cache._clear()
    live_rows._clear()


def _key(run_id: str = "run-1", plan: str | None = "orm", source: str = "tiled"):
    return figure_cache.make_key(run_id, plan, source)


# =========================================================================
# Keys and basic storage
# =========================================================================


def test_make_key_orders_run_id_first() -> None:
    assert figure_cache.make_key("run-1", "orm", "tiled") == ("run-1", "orm", "tiled")


def test_unknown_key_misses() -> None:
    assert figure_cache.get(_key()) is None


def test_a_stored_figure_comes_back_by_identity() -> None:
    figure = object()
    assert figure_cache.put(_key(), figure, generation=0) is True

    assert figure_cache.get(_key()) is figure


def test_a_stored_none_is_indistinguishable_from_a_miss() -> None:
    # Documented consequence of `get` returning `None` for a miss: the route
    # never stores `None`, and this pins that the cache does not pretend
    # otherwise.
    figure_cache.put(_key(), None, generation=0)

    assert figure_cache.get(_key()) is None


def test_the_three_key_fields_are_all_significant() -> None:
    live, tiled = object(), object()
    figure_cache.put(_key(source="live"), live, generation=0)
    figure_cache.put(_key(source="tiled"), tiled, generation=0)

    assert figure_cache.get(_key(source="live")) is live
    assert figure_cache.get(_key(source="tiled")) is tiled
    assert figure_cache.get(_key(plan="grid_scan")) is None
    assert figure_cache.get(_key(run_id="run-2")) is None


def test_an_unknown_plan_name_is_its_own_key_not_a_wildcard() -> None:
    unknown = object()
    figure_cache.put(_key(plan=None), unknown, generation=0)

    assert figure_cache.get(_key(plan=None)) is unknown
    assert figure_cache.get(_key(plan="orm")) is None


def test_a_second_put_replaces_the_entry() -> None:
    first, second = object(), object()
    figure_cache.put(_key(), first, generation=0)
    figure_cache.put(_key(), second, generation=0)

    assert figure_cache.get(_key()) is second


# =========================================================================
# Run-scoped reads (get_for_source — the rotated branch's lookup, which has
# no plan name until it pays for the Tiled read the cache exists to avoid)
# =========================================================================


def test_get_for_source_finds_an_exact_key_put_without_the_plan_name() -> None:
    figure = object()
    figure_cache.put(_key(plan="orm", source="tiled"), figure, generation=0)

    assert figure_cache.get_for_source("run-1", "tiled") is figure


def test_get_for_source_discriminates_source_and_run_id() -> None:
    live, tiled = object(), object()
    figure_cache.put(_key(source="live"), live, generation=0)
    figure_cache.put(_key(source="tiled"), tiled, generation=0)

    assert figure_cache.get_for_source("run-1", "live") is live
    assert figure_cache.get_for_source("run-1", "tiled") is tiled
    assert figure_cache.get_for_source("run-2", "tiled") is None


def test_get_for_source_misses_after_evict() -> None:
    figure_cache.put(_key(source="tiled"), object(), generation=0)
    figure_cache.evict("run-1")

    assert figure_cache.get_for_source("run-1", "tiled") is None


def test_get_for_source_refreshes_the_lru_position_like_get() -> None:
    kept = object()
    figure_cache.put(_key(run_id="hot", source="tiled"), kept, generation=0)
    for i in range(figure_cache._MAX_ENTRIES - 1):
        figure_cache.put(_key(run_id=f"filler-{i}"), object(), generation=0)

    # A run-scoped read must keep its entry hot, exactly as `get` does...
    assert figure_cache.get_for_source("hot", "tiled") is kept
    figure_cache.put(_key(run_id="one-more"), object(), generation=0)

    # ...so the overflow evicts the oldest filler, not the just-read entry.
    assert figure_cache.get_for_source("hot", "tiled") is kept
    assert figure_cache.get(_key(run_id="filler-0")) is None


# =========================================================================
# Bounding
# =========================================================================


def test_the_cache_is_bounded_and_drops_the_least_recently_used() -> None:
    for index in range(figure_cache._MAX_ENTRIES):
        figure_cache.put(_key(run_id=f"run-{index}"), index, generation=0)
    # Touch the oldest so it is no longer the eviction candidate.
    assert figure_cache.get(_key(run_id="run-0")) == 0

    figure_cache.put(_key(run_id="overflow"), "new", generation=0)

    assert len(figure_cache._entries) == figure_cache._MAX_ENTRIES
    assert figure_cache.get(_key(run_id="run-0")) == 0
    assert figure_cache.get(_key(run_id="run-1")) is None
    assert figure_cache.get(_key(run_id="overflow")) == "new"


def test_generations_are_bounded_and_take_their_runs_entries_with_them() -> None:
    # A counter that outlives its entries is harmless; an entry that outlives
    # its counter would be compared against a generation reset to zero — which
    # is exactly the value a stalled computation may be holding.
    figure_cache.put(_key(run_id="oldest"), "figure", generation=0)
    for index in range(figure_cache._MAX_GENERATIONS):
        figure_cache.evict(f"run-{index}")

    assert "oldest" not in figure_cache._generations
    assert figure_cache.get(_key(run_id="oldest")) is None
    assert len(figure_cache._generations) == figure_cache._MAX_GENERATIONS


def test_every_cached_entry_keeps_a_generation_counter() -> None:
    figure_cache.put(_key(), "figure", generation=0)

    assert figure_cache._generations["run-1"] == 0


# =========================================================================
# Eviction and the generation counter
# =========================================================================


def test_evict_drops_every_source_variant_of_the_run() -> None:
    figure_cache.put(_key(source="live"), "live", generation=0)
    figure_cache.put(_key(source="tiled"), "tiled", generation=0)
    figure_cache.put(_key(plan="grid_scan", source="tiled"), "other-plan", generation=0)

    figure_cache.evict("run-1")

    assert figure_cache.get(_key(source="live")) is None
    assert figure_cache.get(_key(source="tiled")) is None
    assert figure_cache.get(_key(plan="grid_scan", source="tiled")) is None


def test_evict_leaves_other_runs_alone() -> None:
    figure_cache.put(_key(run_id="run-1"), "one", generation=0)
    figure_cache.put(_key(run_id="run-2"), "two", generation=0)

    figure_cache.evict("run-1")

    assert figure_cache.get(_key(run_id="run-2")) == "two"
    assert figure_cache.snapshot_generation("run-2") == 0


def test_evict_bumps_the_generation_even_with_nothing_cached() -> None:
    # The bump is for computations in flight, which leave no trace in the map.
    assert figure_cache.snapshot_generation("run-1") == 0

    figure_cache.evict("run-1")

    assert figure_cache.snapshot_generation("run-1") == 1


def test_generations_accumulate_across_evictions() -> None:
    for expected in range(1, 4):
        figure_cache.evict("run-1")
        assert figure_cache.snapshot_generation("run-1") == expected


# =========================================================================
# Compare-and-set stores
# =========================================================================


def test_a_store_at_the_current_generation_lands() -> None:
    figure_cache.evict("run-1")
    generation = figure_cache.snapshot_generation("run-1")

    assert figure_cache.put(_key(), "figure", generation) is True
    assert figure_cache.get(_key()) == "figure"


def test_a_store_from_before_an_eviction_is_dropped() -> None:
    generation = figure_cache.snapshot_generation("run-1")
    figure_cache.evict("run-1")

    assert figure_cache.put(_key(), "stale", generation) is False
    assert figure_cache.get(_key()) is None


def test_a_stale_store_does_not_overwrite_a_fresh_entry() -> None:
    stale_generation = figure_cache.snapshot_generation("run-1")
    figure_cache.evict("run-1")
    figure_cache.put(_key(), "fresh", figure_cache.snapshot_generation("run-1"))

    assert figure_cache.put(_key(), "stale", stale_generation) is False
    assert figure_cache.get(_key()) == "fresh"


def test_a_store_from_the_future_is_dropped_too() -> None:
    # Not reachable through the route, but the check is an equality on purpose:
    # a generation the cache never issued is not evidence of freshness.
    assert figure_cache.put(_key(), "figure", generation=7) is False
    assert figure_cache.get(_key()) is None


# =========================================================================
# Threaded: evictions against reads and stores
# =========================================================================


def test_a_stalled_computation_interleaved_with_an_eviction_does_not_land() -> None:
    """The stalled-render race, driven for real rather than by call ordering."""
    snapshotted = threading.Event()
    evicted = threading.Event()
    stored: list[bool] = []

    def compute_and_store() -> None:
        generation = figure_cache.snapshot_generation("run-1")
        snapshotted.set()
        assert evicted.wait(timeout=5.0)
        stored.append(figure_cache.put(_key(), "stale", generation))

    worker = threading.Thread(target=compute_and_store)
    worker.start()
    assert snapshotted.wait(timeout=5.0)
    figure_cache.evict("run-1")
    evicted.set()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert stored == [False]
    assert figure_cache.get(_key()) is None


def test_concurrent_evictions_and_reads_neither_deadlock_nor_lose_an_eviction() -> None:
    """Route threads storing against document-plane threads evicting.

    The invariant asserted is the one the cache exists to keep: whatever is
    cached at the end was stored at the generation still current for that run.
    An entry stamped with an older generation would be a lost eviction — a
    figure built from rows that run id no longer means.
    """
    run_ids = [f"run-{index}" for index in range(4)]
    stop = threading.Event()
    failures: list[str] = []
    barrier = threading.Barrier(6)

    def reader(run_id: str) -> None:
        barrier.wait(timeout=5.0)
        try:
            while not stop.is_set():
                generation = figure_cache.snapshot_generation(run_id)
                key = figure_cache.make_key(run_id, "orm", "tiled")
                figure_cache.get(key)
                # Stand in for the source read + render the route does here.
                time.sleep(0.0005)
                figure_cache.put(key, generation, generation)
        except Exception as exc:  # pragma: no cover - a failure path
            failures.append(f"reader {run_id}: {exc!r}")

    def evictor() -> None:
        barrier.wait(timeout=5.0)
        try:
            while not stop.is_set():
                for run_id in run_ids:
                    figure_cache.evict(run_id)
                time.sleep(0.0005)
        except Exception as exc:  # pragma: no cover - a failure path
            failures.append(f"evictor: {exc!r}")

    threads = [threading.Thread(target=reader, args=(run_id,)) for run_id in run_ids]
    threads += [threading.Thread(target=evictor) for _ in range(2)]
    for thread in threads:
        thread.start()
    time.sleep(0.5)
    stop.set()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not [thread for thread in threads if thread.is_alive()], "threads did not finish"
    assert failures == []
    assert figure_cache._lock.acquire(timeout=1.0), "cache lock was left held"
    figure_cache._lock.release()
    for run_id in run_ids:
        assert figure_cache.snapshot_generation(run_id) > 0, "the evictors never ran"
        cached = figure_cache.get(figure_cache.make_key(run_id, "orm", "tiled"))
        if cached is not None:
            assert cached == figure_cache.snapshot_generation(run_id)


# =========================================================================
# Document-plane wiring
# =========================================================================


def _start_doc(run_uid: str, run_id: str | None = "run-1") -> dict:
    doc: dict = {"uid": run_uid, "num_points": 3}
    if run_id is not None:
        doc[RUN_ID_META_KEY] = run_id
        doc[PLAN_META_KEY] = {"name": "orm", "kwargs": {}}
    return doc


def test_a_start_document_evicts_the_run_ids_cached_figures() -> None:
    figure_cache.put(_key(source="live"), "live", generation=0)
    figure_cache.put(_key(source="tiled"), "tiled", generation=0)

    RunDocumentRouter()("start", _start_doc("uid-1"))

    assert figure_cache.get(_key(source="live")) is None
    assert figure_cache.get(_key(source="tiled")) is None
    assert figure_cache.snapshot_generation("run-1") == 1


def test_a_start_document_without_a_run_id_evicts_under_the_run_uid() -> None:
    # Same fallback identity the recorder keys its buffer by.
    figure_cache.put(_key(run_id="uid-1"), "figure", generation=0)

    RunDocumentRouter()("start", _start_doc("uid-1", run_id=None))

    assert figure_cache.get(_key(run_id="uid-1")) is None
    assert figure_cache.snapshot_generation("uid-1") == 1


def test_a_start_document_leaves_other_run_ids_cached() -> None:
    figure_cache.put(_key(run_id="run-2"), "untouched", generation=0)

    RunDocumentRouter()("start", _start_doc("uid-1"))

    assert figure_cache.get(_key(run_id="run-2")) == "untouched"


def test_the_rows_are_already_the_new_runs_when_the_generation_is_bumped() -> None:
    """Ordering: eviction runs after the recorder write, not before it.

    A figure computation that snapshots the post-bump generation must be reading
    the NEW run's rows — otherwise it could cache the previous occupant's rows
    under a generation that looks current.
    """
    router = RunDocumentRouter()
    router("start", _start_doc("uid-1"))
    router("descriptor", {"uid": "desc-1", "run_start": "uid-1"})
    router("event", {"descriptor": "desc-1", "data": {"x": 1.0}})
    router("stop", {"run_start": "uid-1"})
    assert live_rows.get("run-1")["total_seen"] == 1

    observed: list[int] = []

    class _WatchingRecorder:
        """Records the generation as of the moment the buffer is reset."""

        def __init__(self, key=None, plan=None):
            self._inner = live_rows.LiveRowRecorder(key=key, plan=plan)

        def __call__(self, name: str, doc: dict) -> None:
            self._inner(name, doc)
            if name == "start":
                observed.append(figure_cache.snapshot_generation("run-1"))
                observed.append(live_rows.get("run-1")["total_seen"])

    before = figure_cache.snapshot_generation("run-1")
    original = document_plane.LiveRowRecorder
    document_plane.LiveRowRecorder = _WatchingRecorder  # type: ignore[misc]
    try:
        router("start", _start_doc("uid-2"))
    finally:
        document_plane.LiveRowRecorder = original  # type: ignore[misc]

    # Buffer already reset while the generation was still the old one, so the
    # bump strictly follows the row reset.
    assert observed == [before, 0]
    assert figure_cache.snapshot_generation("run-1") == before + 1


def test_the_eviction_hook_never_holds_both_locks() -> None:
    """`live_rows`' lock is free while the cache's is held, and vice versa.

    The two locks are never nested in either order, so there is no ordering for
    a future caller to violate. Asserted from inside each critical section.
    """
    held: list[bool] = []

    original_evict = figure_cache.evict

    def watching_evict(run_id: str) -> None:
        with figure_cache._lock:
            held.append(live_rows._lock.acquire(blocking=False))
            if held[-1]:
                live_rows._lock.release()
        original_evict(run_id)

    figure_cache.evict = watching_evict  # type: ignore[assignment]
    document_plane.figure_cache.evict = watching_evict  # type: ignore[assignment]
    try:
        RunDocumentRouter()("start", _start_doc("uid-1"))
    finally:
        figure_cache.evict = original_evict  # type: ignore[assignment]
        document_plane.figure_cache.evict = original_evict  # type: ignore[assignment]

    assert held == [True], "live_rows' lock was still held when the cache's was taken"


# =========================================================================
# Module hygiene
# =========================================================================


def test_the_cache_module_imports_neither_bluesky_nor_pydantic() -> None:
    # It is imported by the bridge process and, transitively, by anything that
    # imports the document plane; it needs neither stack to hold opaque values.
    source = Path(figure_cache.__file__).read_text()
    assert "import bluesky" not in source
    assert "import pydantic" not in source
    assert "from pydantic" not in source
