"""drain_once + supervision helpers, driven with fake callbacks, a fake
DispatchClient.status, and a REAL DedupStore on a temp file.

Covers: gate-closed no-op, min-age eligibility, coalescing (before the age
filter; supersede notes; terminal superseded), the redispatch safety gate
(KNOWN num_tool_calls == 0 or give up — never re-execute unproven work), fresh
redispatch (CAS; CAS-loser takes no action), re-attach (completed deliver /
non-terminal skip / terminal-failure retry decision / consecutive-404 give up),
the give-up ceilings, deliver-failure -> post_failed, per-entry isolation, and
the thread supervision helpers."""

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.drain import (
    DrainCallbacks,
    DrainDeps,
    drain_once,
    ensure_alive,
    run_drain_thread,
)

NOW = 1_000_000.0
PAST_MIN = 2000.0  # > retry_min_age (1200)


def _cfg(**over):
    base = {
        "retry_min_age": 1200.0,
        "retry_give_up": 172800.0,
        "retry_lifetime_cap": 604800.0,
        "drain_interval": 0.01,
    }
    base.update(over)
    return SimpleNamespace(**base)


@dataclass
class FakeCalls:
    """Records callback invocations; deliver/give_up can be told to raise."""

    dispatched: list = field(default_factory=list)
    delivered: list = field(default_factory=list)
    gave_up: list = field(default_factory=list)
    deliver_raises: bool = False
    give_up_raises: bool = False

    def as_callbacks(self):
        def dispatch(mid, entry):
            self.dispatched.append(mid)

        def deliver(mid, entry, body):
            self.delivered.append((mid, body.get("status")))
            if self.deliver_raises:
                raise RuntimeError("send failed")

        def give_up(mid, entry):
            self.gave_up.append(mid)
            if self.give_up_raises:
                raise RuntimeError("notice failed")

        return DrainCallbacks(dispatch=dispatch, deliver=deliver, give_up=give_up)


class FakeDispatcher:
    """Stands in for DispatchClient.status: returns queued bodies by run_id."""

    def __init__(self, bodies=None):
        # run_id -> body dict (None means 404)
        self.bodies = bodies or {}

    def status(self, run_id):
        return self.bodies.get(run_id)


def _deps(store, calls, dispatcher=None, gate=True, now=NOW):
    return DrainDeps(
        cfg=_cfg(),
        dedup=store,
        dispatcher=dispatcher or FakeDispatcher(),
        callbacks=calls.as_callbacks(),
        gate=(lambda: gate),
        now=(lambda: now),
    )


def _store(tmp_path):
    return DedupStore(str(tmp_path / "dedup.json"))


def _enqueue(store, mid, age=PAST_MIN, **meta):
    """Create a queued entry aged `age` seconds before NOW."""
    store.claim(mid, **meta)
    store.transition(mid, "pending", "queued", queued_at=NOW - age, **meta)


# --- gate ------------------------------------------------------------------


def test_gate_closed_is_noop(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1")
    calls = FakeCalls()
    drain_once(_deps(store, calls, gate=False))
    assert calls.dispatched == [] and store.get("m1")["status"] == "queued"


# --- eligibility -----------------------------------------------------------


def test_too_young_entry_skipped(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "young", age=100.0)  # < retry_min_age
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.dispatched == []
    assert store.get("young")["status"] == "queued"


# --- fresh redispatch (CAS) ------------------------------------------------


def test_fresh_entry_redispatched_and_flipped(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi", num_tool_calls=0)  # known side-effect free
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.dispatched == ["m1"]
    entry = store.get("m1")
    assert entry["status"] == "in_flight"
    assert entry["retried"] is True


def test_unknown_tool_calls_never_redispatched(tmp_path):
    """No num_tool_calls in the entry meta -> the failed attempt cannot be
    PROVEN side-effect free -> honest give-up, never a re-execution."""
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi")  # num_tool_calls unknown
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.dispatched == []
    assert calls.gave_up == ["m1"]
    assert store.get("m1")["status"] == "error"


def test_nonzero_tool_calls_never_redispatched(tmp_path):
    """A failed attempt that made tool calls may have acted on the machine —
    re-running it is forbidden regardless of failure class."""
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi", num_tool_calls=3)
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.dispatched == []
    assert calls.gave_up == ["m1"]
    assert store.get("m1")["status"] == "error"


def test_cas_loser_takes_no_action(tmp_path):
    """An entry flipped away between the queue snapshot and the drain's CAS is
    skipped entirely — the loser must not double-dispatch."""
    store = _store(tmp_path)
    _enqueue(store, "m1", num_tool_calls=0)
    _enqueue(store, "m2", num_tool_calls=0)
    dispatched = []

    def dispatch(mid, entry):
        dispatched.append(mid)
        if mid == "m1":  # simulate a racing actor claiming m2 mid-pass
            assert store.transition("m2", "queued", "superseded")

    cbs = DrainCallbacks(dispatch=dispatch, deliver=lambda m, e, b: None, give_up=lambda m, e: None)
    deps = DrainDeps(
        cfg=_cfg(),
        dedup=store,
        dispatcher=FakeDispatcher(),
        callbacks=cbs,
        gate=(lambda: True),
        now=(lambda: NOW),
    )
    drain_once(deps)
    assert dispatched == ["m1"]  # m2's CAS lost -> no dispatch for it
    assert store.get("m2")["status"] == "superseded"


# --- coalescing ------------------------------------------------------------


def test_coalesce_keeps_newest_supersedes_rest(tmp_path):
    store = _store(tmp_path)
    # Two fresh entries, same coalesce key; m_new is newer (smaller age).
    _enqueue(store, "m_old", age=3000.0, coalesce_key=["space", "thread"], num_tool_calls=0)
    _enqueue(store, "m_new", age=2000.0, coalesce_key=["space", "thread"], num_tool_calls=0)
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.dispatched == ["m_new"]
    assert store.get("m_old")["status"] == "superseded"
    assert store.get("m_new")["status"] == "in_flight"


def test_coalesce_fires_supersede_callback(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m_old", age=3000.0, coalesce_key=["s", "t"], num_tool_calls=0)
    _enqueue(store, "m_new", age=2000.0, coalesce_key=["s", "t"], num_tool_calls=0)
    superseded = []
    calls = FakeCalls()
    cbs = calls.as_callbacks()
    cbs.supersede = lambda mid, entry: superseded.append(mid)
    deps = DrainDeps(
        cfg=_cfg(),
        dedup=store,
        dispatcher=FakeDispatcher(),
        callbacks=cbs,
        gate=(lambda: True),
        now=(lambda: NOW),
    )
    drain_once(deps)
    assert superseded == ["m_old"]


def test_coalesce_runs_before_age_filter(tmp_path):
    """A newest-but-too-young sibling still supersedes an older ELIGIBLE one —
    otherwise the old question is answered this pass and the new one later."""
    store = _store(tmp_path)
    _enqueue(store, "m_old", age=3000.0, coalesce_key=["k"], num_tool_calls=0)
    _enqueue(store, "m_new", age=100.0, coalesce_key=["k"], num_tool_calls=0)  # < min age
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert store.get("m_old")["status"] == "superseded"
    assert calls.dispatched == []  # the newest waits out retry_min_age
    assert store.get("m_new")["status"] == "queued"


def test_coalesce_newest_by_first_queued_at(tmp_path):
    # A re-parked older request carries a FRESHER queued_at than a newer request
    # parked once; "newest" must follow the request's first park time, not the
    # latest re-park, or a retry loop shields an old question from coalescing.
    store = _store(tmp_path)
    _enqueue(
        store,
        "m_old",
        age=100.0,
        coalesce_key=["k"],
        num_tool_calls=0,
        first_queued_at=NOW - 5000.0,
    )
    _enqueue(
        store,
        "m_new",
        age=3000.0,
        coalesce_key=["k"],
        num_tool_calls=0,
        first_queued_at=NOW - 3000.0,
    )
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert store.get("m_old")["status"] == "superseded"
    assert store.get("m_new")["status"] == "in_flight"


def test_give_up_ceiling_survives_repark(tmp_path):
    # A cycling entry (fresh queued_at from its latest re-park) must still trip
    # the give-up ceiling through its first_queued_at anchor.
    store = _store(tmp_path)
    _enqueue(
        store,
        "m1",
        age=PAST_MIN,
        first_queued_at=NOW - 172801.0,
        retried=True,
        num_tool_calls=0,
    )
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.gave_up == ["m1"]
    assert store.get("m1")["status"] == "error"


def test_superseded_is_terminal_and_never_reconciled(tmp_path):
    """'superseded' must be a terminal drop: reconcile() re-running a
    deliberately dropped question on restart is forbidden."""
    from osprey.bridges.core.config import TERMINAL_STATUSES

    assert "superseded" in TERMINAL_STATUSES
    store = _store(tmp_path)
    _enqueue(store, "m_old", age=3000.0, coalesce_key=["k"], num_tool_calls=0)
    _enqueue(store, "m_new", age=2000.0, coalesce_key=["k"], num_tool_calls=0)
    drain_once(_deps(store, FakeCalls()))
    assert store.get("m_old")["status"] == "superseded"
    assert "m_old" not in store.reconcile()
    assert "m_old" not in store.queued()


def test_empty_coalesce_key_never_coalesces(tmp_path):
    store = _store(tmp_path)
    # No coalesce_key -> each stands alone, both redispatched.
    _enqueue(store, "a", age=3000.0, num_tool_calls=0)
    _enqueue(store, "b", age=2000.0, num_tool_calls=0)
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert sorted(calls.dispatched) == ["a", "b"]


# --- re-attach -------------------------------------------------------------


def test_reattach_terminal_delivers_and_marks(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1")
    store.record("m1", run_id="R1", status="queued")  # has a run_id
    dispatcher = FakeDispatcher({"R1": {"status": "completed", "text_output": "done"}})
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=dispatcher))
    assert calls.delivered == [("m1", "completed")]
    assert calls.dispatched == []  # re-attach, not redispatch
    assert store.get("m1")["status"] == "completed"


def test_reattach_nonterminal_skips(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1")
    store.record("m1", run_id="R1", status="queued")
    dispatcher = FakeDispatcher({"R1": {"status": "running"}})
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=dispatcher))
    assert calls.delivered == [] and calls.gave_up == []
    assert store.get("m1")["status"] == "queued"


def test_reattach_single_404_defers(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1")
    store.record("m1", run_id="R1", status="queued")
    dispatcher = FakeDispatcher({})  # 404 for R1
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=dispatcher))
    assert calls.gave_up == []
    entry = store.get("m1")
    assert entry["status"] == "queued"
    assert entry["notfound_count"] == 1


def test_reattach_two_consecutive_404s_give_up(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1")
    store.record("m1", run_id="R1", status="queued")
    dispatcher = FakeDispatcher({})  # always 404
    calls = FakeCalls()
    deps = _deps(store, calls, dispatcher=dispatcher)
    drain_once(deps)  # miss 1
    assert store.get("m1")["notfound_count"] == 1
    drain_once(deps)  # miss 2 -> give up
    assert calls.gave_up == ["m1"]
    assert store.get("m1")["status"] == "error"


def test_reattach_failure_retryable_and_safe_redispatches(tmp_path):
    """A terminal FAILURE body is the run's definitive outcome: retryable class
    + known-zero tool calls -> one fresh dispatch, not a delivery."""
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi")
    store.record("m1", run_id="R1", status="queued")
    body = {"status": "error", "failure_class": "provider", "num_tool_calls": 0}
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=FakeDispatcher({"R1": body})))
    assert calls.dispatched == ["m1"]
    assert calls.delivered == []  # an error body is never "delivered"
    entry = store.get("m1")
    assert entry["status"] == "in_flight"
    assert entry["retried"] is True
    assert entry["run_id"] is None  # cleared; the channel's persister stamps the new one


def test_reattach_failure_with_tool_calls_gives_up(tmp_path):
    """Retryable class but the failed run made tool calls -> it may have acted
    on the machine, so it is abandoned, never re-executed."""
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi")
    store.record("m1", run_id="R1", status="queued")
    body = {"status": "error", "failure_class": "infrastructure", "num_tool_calls": 2}
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=FakeDispatcher({"R1": body})))
    assert calls.dispatched == [] and calls.delivered == []
    assert calls.gave_up == ["m1"]
    assert store.get("m1")["status"] == "error"


def test_reattach_failure_not_retryable_gives_up(tmp_path):
    """An agent/user-level failure would fail identically on redispatch."""
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi")
    store.record("m1", run_id="R1", status="queued")
    body = {"status": "error", "failure_class": "agent", "num_tool_calls": 0}
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=FakeDispatcher({"R1": body})))
    assert calls.dispatched == []
    assert calls.gave_up == ["m1"]
    assert store.get("m1")["status"] == "error"


def test_reattach_404_streak_resets_on_hit(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1")
    store.record("m1", run_id="R1", status="queued")
    dispatcher = FakeDispatcher({})
    calls = FakeCalls()
    deps = _deps(store, calls, dispatcher=dispatcher)
    drain_once(deps)  # miss 1
    dispatcher.bodies["R1"] = {"status": "running"}  # now the run appears
    drain_once(deps)  # non-terminal -> streak cleared
    assert store.get("m1")["notfound_count"] == 0


# --- give-up ceilings ------------------------------------------------------


def test_give_up_lifetime_cap(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "old", age=604800.0)  # >= lifetime cap
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.gave_up == ["old"]
    assert calls.dispatched == []
    assert store.get("old")["status"] == "error"


def test_give_up_age_extends_on_first_pass_then_fires(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1", age=200000.0, text="hi", num_tool_calls=0)  # > give_up age, attempts 0
    calls = FakeCalls()
    deps = _deps(store, calls)
    # First pass: gate seen open now but not yet persisted from a PRIOR pass, and
    # never retried -> extend (redispatch), and stamp gate_seen_open.
    drain_once(deps)
    assert calls.gave_up == []
    assert store.get("m1")["status"] == "in_flight"  # got its fair attempt


def test_give_up_age_fires_when_gate_seen_open_prior(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1", age=200000.0)
    store.record("m1", run_id=None, status="queued", gate_seen_open=True)
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert calls.gave_up == ["m1"]


# --- deliver failure -> stays queued, retried -------------------------------


def test_deliver_failure_leaves_queued_and_retries_next_cycle(tmp_path):
    """A raising deliver must NOT terminally drop the answer: the entry stays
    queued (run_id intact) and the next cycle re-attaches and re-delivers."""
    store = _store(tmp_path)
    _enqueue(store, "m1")
    store.record("m1", run_id="R1", status="queued")
    dispatcher = FakeDispatcher({"R1": {"status": "completed", "text_output": "hi"}})
    calls = FakeCalls(deliver_raises=True)
    deps = _deps(store, calls, dispatcher=dispatcher)
    drain_once(deps)  # send fails
    entry = store.get("m1")
    assert entry["status"] == "queued"
    assert entry["run_id"] == "R1"
    calls.deliver_raises = False
    drain_once(deps)  # transient failure cleared -> re-delivered
    assert calls.delivered == [("m1", "completed"), ("m1", "completed")]
    assert store.get("m1")["status"] == "completed"


# --- give-up mechanics -------------------------------------------------------


def test_dispatch_increments_attempts(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1", num_tool_calls=0, attempts=0)
    _enqueue(store, "m2", num_tool_calls=0, attempts=2)
    calls = FakeCalls()
    drain_once(_deps(store, calls))
    assert store.get("m1")["attempts"] == 1
    assert store.get("m2")["attempts"] == 3


def test_give_up_reason_stamped_in_store_meta(tmp_path):
    """The drain's machine-side reason lands in the terminal transition meta
    (audit trail) — the callback surface stays (message_id, entry)."""
    store = _store(tmp_path)
    _enqueue(store, "m_age", age=604800.0)  # lifetime cap
    _enqueue(store, "m_run", text="hi")
    store.record("m_run", run_id="R1", status="queued")
    body = {"status": "error", "failure_class": "agent", "num_tool_calls": 0}
    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=FakeDispatcher({"R1": body})))
    assert sorted(calls.gave_up) == ["m_age", "m_run"]
    assert "lifetime cap" in store.get("m_age")["give_up_reason"]
    assert "cannot be safely re-run" in store.get("m_run")["give_up_reason"]


def test_give_up_notice_failure_leaves_queued(tmp_path):
    """Callback first, CAS second: a give-up notice that fails to send leaves
    the entry queued so the next cycle retries the notice — never a silent
    terminal drop."""
    store = _store(tmp_path)
    _enqueue(store, "m1", age=604800.0)
    calls = FakeCalls(give_up_raises=True)
    deps = _deps(store, calls)
    drain_once(deps)
    assert calls.gave_up == ["m1"]  # attempted
    assert store.get("m1")["status"] == "queued"  # not dropped
    calls.give_up_raises = False
    drain_once(deps)  # notice now sends -> terminal
    assert store.get("m1")["status"] == "error"


# --- per-entry isolation ---------------------------------------------------


def test_one_entry_failure_does_not_abort_pass(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "bad")
    store.record("bad", run_id="RB", status="queued")
    _enqueue(store, "good", text="hi", num_tool_calls=0)

    class Boom(FakeDispatcher):
        def status(self, run_id):
            raise RuntimeError("worker exploded")

    calls = FakeCalls()
    drain_once(_deps(store, calls, dispatcher=Boom()))
    # The good fresh entry is still redispatched despite the bad one raising.
    assert calls.dispatched == ["good"]
    assert store.get("bad")["status"] == "queued"


# --- supervision helpers ---------------------------------------------------


def test_run_drain_thread_runs_and_stops(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, "m1", text="hi", num_tool_calls=0)
    calls = FakeCalls()
    deps = _deps(store, calls)
    stop = threading.Event()
    thread = run_drain_thread(deps, stop=stop)
    # Poll until the fresh entry gets dispatched, then stop.
    for _ in range(200):
        if calls.dispatched:
            break
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2.0)
    assert calls.dispatched == ["m1"]
    assert not thread.is_alive()
    assert thread.daemon and thread.name == "bridge-drain"


def test_ensure_alive_returns_live_thread():
    alive = threading.Thread(target=lambda: time.sleep(1.0))
    alive.start()
    try:
        assert ensure_alive(alive, factory=lambda: pytest.fail("should not rebuild")) is alive
    finally:
        alive.join()


def test_ensure_alive_rebuilds_dead_thread():
    made = []

    def factory():
        t = threading.Thread(target=lambda: None)
        made.append(t)
        return t

    result = ensure_alive(None, factory)
    assert result is made[0]
