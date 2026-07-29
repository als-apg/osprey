"""DedupStore conformance suite.

Two concerns, kept in their original sections:

  * **prune / timestamp stamping** — ``claim()`` stamps ``claimed_at`` and,
    before claiming, reaps terminal entries whose settle time is more than
    :data:`~osprey.bridges.core.dedup.PRUNE_MAX_AGE_SECONDS` old; terminal
    transitions stamp ``settled_at``.
  * **CAS transition / retry-queue selection** — the concurrency primitive the
    whole retry queue is built on.

Both drive a REAL store on ``tmp_path`` (no mocks) and the actual lock; the
prune section injects a fake clock through the ``now`` seam.
"""

import json
import threading

from osprey.bridges.core.dedup import PRUNE_MAX_AGE_SECONDS, DedupStore


class FakeClock:
    """Injectable monotone clock (epoch seconds) advanced explicitly by tests."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _clocked_store(tmp_path, clock):
    return DedupStore(str(tmp_path / "dedup.json"), now=clock)


def _store(tmp_path):
    return DedupStore(str(tmp_path / "dedup.json"))


def _write_raw(tmp_path, data):
    """Seed the store file directly, simulating a pre-feature persisted store."""
    with open(str(tmp_path / "dedup.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


# ===========================================================================
# Section 1 — prune + timestamp stamping
# ===========================================================================


# --- timestamp stamping ----------------------------------------------------
def test_claim_stamps_claimed_at(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("m1")
    assert d.get("m1")["claimed_at"] == clock.t


def test_terminal_transition_stamps_settled_at(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("m1")
    d.record("m1", run_id="r1", status="in_flight")
    # Non-terminal transition does NOT stamp settled_at.
    assert "settled_at" not in d.get("m1")

    clock.advance(5.0)
    assert d.transition("m1", "in_flight", "completed") is True
    assert d.get("m1")["settled_at"] == clock.t


def test_record_terminal_stamps_settled_at(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("m1")
    d.record("m1", run_id="r1", status="completed")
    assert d.get("m1")["settled_at"] == clock.t


# --- prune: age boundary ---------------------------------------------------
def test_prune_keeps_terminal_entry_at_exact_age_boundary(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("old")
    d.transition("old", "pending", "completed")  # settled_at = t0

    # Exactly PRUNE_MAX_AGE_SECONDS old: age is NOT > the threshold, so it stays.
    clock.advance(PRUNE_MAX_AGE_SECONDS)
    d.claim("trigger")  # prune runs at the start of claim()
    assert d.seen("old") is True


def test_prune_drops_terminal_entry_past_age_boundary(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("old")
    d.transition("old", "pending", "completed")  # settled_at = t0

    clock.advance(PRUNE_MAX_AGE_SECONDS + 1)
    d.claim("trigger")
    assert d.seen("old") is False
    assert d.seen("trigger") is True


def test_prune_uses_claimed_at_when_settled_at_absent(tmp_path):
    """A terminal entry with claimed_at but no settled_at ages from claimed_at."""
    clock = FakeClock()
    _write_raw(tmp_path, {"old": {"run_id": "r", "status": "error", "claimed_at": clock.t}})
    d = _clocked_store(tmp_path, clock)

    clock.advance(PRUNE_MAX_AGE_SECONDS + 1)
    d.claim("trigger")
    assert d.seen("old") is False


# --- prune: only terminal entries ------------------------------------------
def test_prune_leaves_non_terminal_entries(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("pending1")
    d.claim("inflight1")
    d.record("inflight1", run_id="r1", status="in_flight")
    d.claim("queued1")
    d.transition("queued1", "pending", "queued")

    # Far past any prune horizon — non-terminal work is never reaped.
    clock.advance(PRUNE_MAX_AGE_SECONDS * 10)
    d.claim("trigger")
    assert d.seen("pending1") is True
    assert d.seen("inflight1") is True
    assert d.seen("queued1") is True


def test_prune_reaps_each_terminal_status(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    for i, status in enumerate(("completed", "error", "post_failed", "superseded")):
        mid = f"m{i}"
        d.claim(mid)
        d.transition(mid, "pending", status)

    clock.advance(PRUNE_MAX_AGE_SECONDS + 1)
    d.claim("trigger")
    assert [d.seen(f"m{i}") for i in range(4)] == [False, False, False, False]


# --- prune: grandfathering ts-less terminal entries ------------------------
def test_prune_grandfathers_tsless_terminal_entry_on_first_observation(tmp_path):
    """A pre-feature terminal entry (no settled_at, no claimed_at) survives the
    first prune and gets settled_at = now stamped and persisted."""
    clock = FakeClock()
    _write_raw(tmp_path, {"legacy": {"run_id": "r", "status": "completed"}})
    d = _clocked_store(tmp_path, clock)

    d.claim("trigger")  # first prune observation
    assert d.seen("legacy") is True
    assert d.get("legacy")["settled_at"] == clock.t

    # The stamp is persisted: a fresh store reading the file sees it.
    d2 = _clocked_store(tmp_path, clock)
    assert d2.get("legacy")["settled_at"] == clock.t


def test_prune_grandfathered_entry_ages_out_one_window_later(tmp_path):
    clock = FakeClock()
    _write_raw(tmp_path, {"legacy": {"run_id": "r", "status": "completed"}})
    d = _clocked_store(tmp_path, clock)

    d.claim("trigger1")  # grandfathers: settled_at = t0
    grandfather_t = clock.t

    # Not yet a full window past the grandfather stamp: still present.
    clock.advance(PRUNE_MAX_AGE_SECONDS)
    d.claim("trigger2")
    assert d.seen("legacy") is True
    assert d.get("legacy")["settled_at"] == grandfather_t  # not re-stamped

    # One tick past the window from the grandfather stamp: reaped.
    clock.advance(1)
    d.claim("trigger3")
    assert d.seen("legacy") is False


# --- prune: redelivery re-claims cleanly -----------------------------------
def test_prune_redelivery_of_pruned_message_reclaims_cleanly(tmp_path):
    clock = FakeClock()
    d = _clocked_store(tmp_path, clock)
    d.claim("m1", space="s1")
    d.transition("m1", "pending", "completed")

    clock.advance(PRUNE_MAX_AGE_SECONDS + 1)
    # A redelivery of the long-since-pruned m1: its stale entry is reaped by the
    # prune at the start of claim(), so this call re-claims it as brand new.
    assert d.claim("m1", space="s2") is True
    entry = d.get("m1")
    assert entry["status"] == "pending"
    assert entry["run_id"] is None
    assert entry["space"] == "s2"  # fresh claim, not the stale s1
    assert entry["claimed_at"] == clock.t


# ===========================================================================
# Section 2 — CAS transition + retry-queue selection
# ===========================================================================


def test_transition_matches_from_status(tmp_path):
    d = _store(tmp_path)
    d.claim("m1")  # status == "pending"

    assert d.transition("m1", "pending", "queued") is True
    assert d.get("m1")["status"] == "queued"


def test_transition_returns_false_on_status_mismatch(tmp_path):
    d = _store(tmp_path)
    d.claim("m1")  # status == "pending"

    assert d.transition("m1", "queued", "in_flight") is False
    # No write: the entry keeps its real status.
    assert d.get("m1")["status"] == "pending"


def test_transition_returns_false_when_absent(tmp_path):
    d = _store(tmp_path)
    assert d.transition("nope", "queued", "in_flight") is False
    assert d.get("nope") is None


def test_transition_merges_meta_on_success(tmp_path):
    d = _store(tmp_path)
    d.claim("m1", space="s1")

    assert d.transition("m1", "pending", "queued", attempts=1) is True
    entry = d.get("m1")
    assert entry["status"] == "queued"
    assert entry["attempts"] == 1
    assert entry["space"] == "s1"  # pre-existing meta preserved


def test_transition_does_not_merge_meta_on_failure(tmp_path):
    d = _store(tmp_path)
    d.claim("m1")

    assert d.transition("m1", "queued", "in_flight", attempts=9) is False
    assert "attempts" not in d.get("m1")


def test_transition_persists_across_reload(tmp_path):
    path = str(tmp_path / "dedup.json")
    d = DedupStore(path)
    d.claim("m1")
    assert d.transition("m1", "pending", "queued") is True

    # A fresh store reading the same file must see the flipped status.
    d2 = DedupStore(path)
    assert d2.get("m1")["status"] == "queued"


def test_cas_race_leaves_exactly_one_winner(tmp_path):
    """Concurrent flip (drain: queued -> in_flight) vs coalesce-drop
    (queued -> superseded) over the same entry: exactly one transition wins."""
    d = _store(tmp_path)
    d.claim("m1")
    assert d.transition("m1", "pending", "queued") is True

    start = threading.Barrier(2)
    results: dict[str, bool] = {}

    def flip():
        start.wait()
        results["drain"] = d.transition("m1", "queued", "in_flight")

    def drop():
        start.wait()
        results["coalesce"] = d.transition("m1", "queued", "superseded")

    threads = [threading.Thread(target=flip), threading.Thread(target=drop)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one CAS observed the "queued" status and won.
    assert sum(results.values()) == 1
    final = d.get("m1")["status"]
    if results["drain"]:
        assert final == "in_flight"
    else:
        assert final == "superseded"


def test_many_threads_flip_one_winner(tmp_path):
    """N threads all racing the same queued -> in_flight flip: exactly one True."""
    d = _store(tmp_path)
    d.claim("m1")
    d.transition("m1", "pending", "queued")

    n = 32
    start = threading.Barrier(n)
    wins: list[bool] = []
    wins_lock = threading.Lock()

    def worker():
        start.wait()
        won = d.transition("m1", "queued", "in_flight")
        with wins_lock:
            wins.append(won)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(wins) == 1
    assert d.get("m1")["status"] == "in_flight"


def test_no_ghost_resurrection_via_record_after_terminal(tmp_path):
    """A terminal entry is final: a late/duplicate record() must not resurrect
    it into a non-terminal state (which reconcile would re-dispatch)."""
    d = _store(tmp_path)
    d.claim("m1", space="s1")
    d.record("m1", run_id="r1", status="in_flight")
    assert d.transition("m1", "in_flight", "completed") is True

    # A stale worker callback fires record() again — default status is
    # non-terminal ("in_flight"). It must be ignored.
    d.record("m1", run_id="r1", status="in_flight")

    entry = d.get("m1")
    assert entry["status"] == "completed"  # still terminal
    assert entry["run_id"] == "r1"
    assert len(d.reconcile()) == 0  # no ghost re-surfaces for re-dispatch
    # And no phantom second entry was created.
    with d._lock:
        assert list(d._data.keys()) == ["m1"]


def test_record_still_upserts_non_terminal_entry(tmp_path):
    """The terminal guard must not break the normal in-flight upsert path."""
    d = _store(tmp_path)
    d.claim("m1")  # pending
    d.record("m1", run_id="r1", status="in_flight")
    assert d.get("m1")["status"] == "in_flight"
    assert d.get("m1")["run_id"] == "r1"


def test_queued_returns_only_queued_entries(tmp_path):
    d = _store(tmp_path)
    d.claim("q1")
    d.claim("q2")
    d.claim("p1")
    d.claim("done")
    d.transition("q1", "pending", "queued")
    d.transition("q2", "pending", "queued")
    d.record("done", run_id="r", status="completed")

    q = d.queued()
    assert set(q) == {"q1", "q2"}
    assert all(e["status"] == "queued" for e in q.values())


def test_reconcile_excludes_queued(tmp_path):
    d = _store(tmp_path)
    d.claim("inflight")
    d.claim("queued1")
    d.claim("done")
    d.record("inflight", run_id="r1", status="in_flight")
    d.transition("queued1", "pending", "queued")
    d.record("done", run_id="r2", status="completed")

    rec = d.reconcile()
    # in_flight (non-terminal, non-queued) reconciles; queued is owned by the
    # drain loop; completed is terminal.
    assert set(rec) == {"inflight"}


def test_queued_not_in_terminal_statuses():
    """Guard: 'queued' must NOT be a terminal status — other consumers key off
    the frozenset, and a queued entry is live work, not done."""
    from osprey.bridges.core.config import TERMINAL_STATUSES

    assert "queued" not in TERMINAL_STATUSES
