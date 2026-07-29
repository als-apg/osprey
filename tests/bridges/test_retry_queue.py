"""Retry-queue write-side specification: park, supersede-at-enqueue, give-up notice.

Every test drives a REAL :class:`~osprey.bridges.core.dedup.DedupStore` (and, where
history is involved, a real :class:`~osprey.bridges.core.history.HistoryStore`) on
``tmp_path`` — the CAS being tested IS the store's, so mocking it would test nothing —
against the shared ``RecordingChannelOps`` double for the channel side.

The section that matters most is "supersession preconditions". Supersession is the only
place the engine cancels work a user asked for, and it is justified *solely* by a newer
message taking the older one's place. The three named tests there pin each side of that:
a newer message that parks supersedes; a newer message that succeeds or is still in
flight supersedes nothing. Getting it wrong silently drops a user's first question while
answering their second, with no notice for either — invisible in production.
"""

import time

import pytest

from osprey.bridges.core.config import TERMINAL_STATUSES
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.history import HistoryStore
from osprey.bridges.core.retry import is_retryable
from osprey.bridges.core.retry_queue import (
    SUPERSEDED_STATUS,
    give_up,
    park,
    supersede_at_enqueue,
    supersede_note,
)
from tests.bridges.conftest import RecordingChannelOps

CHAT_KEY = ["space/1", "users/alice"]

RETRYABLE = {"failure_class": "provider", "num_tool_calls": 0, "status": "error"}


class FakeClock:
    """Injectable monotone clock (epoch seconds) advanced explicitly by tests."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def dedup(tmp_path):
    return DedupStore(str(tmp_path / "dedup.json"))


def _ops(**kwargs) -> RecordingChannelOps:
    """A recording double whose channel coalesce key defaults to the Chat-style
    ``[history_key, sender_id]`` grouping."""
    kwargs.setdefault("coalesce", CHAT_KEY)
    return RecordingChannelOps(**kwargs)


def _claim(dedup: DedupStore, message_id: str, **meta):
    """Claim a message the way the pipeline does, and return the persisted entry."""
    dedup.claim(message_id, text="why is the beam down?", history_key="space/1", **meta)
    return dedup.get(message_id)


def _seed_queued(dedup: DedupStore, message_id: str, *, key=CHAT_KEY, **meta):
    """Seed an already-parked entry (an older sibling waiting in the queue)."""
    entry = _claim(dedup, message_id, **meta)
    dedup.transition(
        message_id,
        entry["status"],
        "queued",
        coalesce_key=key,
        queued_at=1_000.0,
        first_queued_at=1_000.0,
        attempts=0,
    )
    return dedup.get(message_id)


# ===========================================================================
# Section 1 — park: what lands in the store
# ===========================================================================


def test_park_flips_pending_to_queued(dedup):
    entry = _claim(dedup, "m1")

    assert park(_ops(), dedup, "m1", entry, RETRYABLE) is True

    assert dedup.get("m1")["status"] == "queued"
    assert "m1" in dedup.queued()


def test_park_stamps_retry_bookkeeping(dedup):
    clock = FakeClock()
    entry = _claim(dedup, "m1")

    park(_ops(), dedup, "m1", entry, RETRYABLE, now=clock)

    parked = dedup.get("m1")
    assert parked["failure_class"] == "provider"
    assert parked["num_tool_calls"] == 0
    assert parked["queued_at"] == clock.t
    assert parked["first_queued_at"] == clock.t
    assert parked["attempts"] == 0


def test_park_preserves_unknown_tool_count_as_none(dedup):
    """``None`` means *unknown*, and ``may_redispatch`` is fail-closed on unknown.
    Coercing it to 0 would let the drain re-run a side-effecting agent."""
    entry = _claim(dedup, "m1")

    park(_ops(), dedup, "m1", entry, {"failure_class": "infrastructure"})

    parked = dedup.get("m1")
    assert parked["num_tool_calls"] is None
    assert parked["failure_class"] == "infrastructure"


def test_park_persists_normalized_channel_coalesce_key(dedup):
    """The channel supplies the grouping input; the engine normalizes it through
    ``retry.coalesce_key`` and persists it JSON-plain so it round-trips on disk."""
    ops = _ops(coalesce="thread-root-42")
    entry = _claim(dedup, "m1")

    park(ops, dedup, "m1", entry, RETRYABLE)

    assert dedup.get("m1")["coalesce_key"] == ["thread-root-42"]
    assert ops.of("coalesce_key")[0]["entry"] == entry


def test_park_defaults_to_time_time(dedup):
    """The clock is injectable but must default to wall time — the drain's ageing
    compares ``queued_at`` against ``time.time()``."""
    entry = _claim(dedup, "m1")
    before = time.time()

    park(_ops(), dedup, "m1", entry, RETRYABLE)

    assert before <= dedup.get("m1")["queued_at"] <= time.time()


def test_repark_refreshes_pacing_clock_but_not_the_giveup_anchor(dedup):
    """``queued_at`` paces the next attempt and refreshes; ``first_queued_at`` anchors
    the give-up ceilings and must NOT, or a cycling entry re-ages forever."""
    clock = FakeClock()
    entry = _claim(dedup, "m1")
    park(_ops(), dedup, "m1", entry, RETRYABLE, now=clock)

    clock.advance(3600.0)
    # The drain redispatched and failed again: queued -> in_flight -> re-park.
    dedup.transition("m1", "queued", "in_flight", retried=True, attempts=1)
    park(_ops(), dedup, "m1", dedup.get("m1"), RETRYABLE, now=clock)

    reparked = dedup.get("m1")
    assert reparked["queued_at"] == 1_003_600.0
    assert reparked["first_queued_at"] == 1_000_000.0


def test_repark_never_resets_the_attempt_count(dedup):
    """``attempts`` is the drain's; only the FIRST park seeds it at 0."""
    entry = _claim(dedup, "m1")
    park(_ops(), dedup, "m1", entry, RETRYABLE)
    dedup.transition("m1", "queued", "in_flight", retried=True, attempts=3)

    park(_ops(), dedup, "m1", dedup.get("m1"), RETRYABLE)

    assert dedup.get("m1")["attempts"] == 3


def test_park_of_a_pre_first_queued_at_entry_anchors_on_queued_at(dedup):
    """An entry persisted before ``first_queued_at`` existed still gets a stable
    anchor, taken from its older ``queued_at`` rather than reset to now."""
    _claim(dedup, "m1")
    dedup.transition("m1", "pending", "in_flight", queued_at=500.0)

    park(_ops(), dedup, "m1", dedup.get("m1"), RETRYABLE, now=FakeClock())

    assert dedup.get("m1")["first_queued_at"] == 500.0


def test_park_writes_no_history(dedup, tmp_path):
    """The exchange is unfinished — history is appended by the drain when the queued
    run finally completes, never at park time."""
    history = HistoryStore(str(tmp_path / "history.json"))
    entry = _claim(dedup, "m1")

    park(_ops(), dedup, "m1", entry, RETRYABLE)

    assert history.recent("space/1") == []


# ===========================================================================
# Section 2 — park: the queued notice, and its ordering
# ===========================================================================


def test_first_park_posts_the_queued_notice(dedup):
    ops = _ops()
    entry = _claim(dedup, "m1")

    park(ops, dedup, "m1", entry, RETRYABLE)

    assert ops.count("post_queued") == 1
    assert ops.of("post_queued")[0]["result"] == RETRYABLE


def test_repark_is_silent(dedup):
    """The user was told once; the next message in the thread is the answer or the
    give-up notice, not another "still queued"."""
    ops = _ops()
    entry = _claim(dedup, "m1")
    park(ops, dedup, "m1", entry, RETRYABLE)
    dedup.transition("m1", "queued", "in_flight", retried=True, attempts=1)

    park(ops, dedup, "m1", dedup.get("m1"), RETRYABLE)

    assert ops.count("post_queued") == 1


def test_entry_is_already_queued_when_the_notice_is_posted(dedup):
    """``post_queued`` MAY raise, so the engine parks FIRST: the entry handed to the
    notice is the persisted one, already in ``queued``."""
    ops = _ops()
    entry = _claim(dedup, "m1")

    park(ops, dedup, "m1", entry, RETRYABLE)

    posted = ops.of("post_queued")[0]["entry"]
    assert posted["status"] == "queued"
    assert posted["queued_at"] == dedup.get("m1")["queued_at"]


def test_failed_queued_notice_leaves_the_request_parked(dedup):
    """A lost notice costs the user a message; it must never cost them the request."""
    ops = _ops()
    ops.fail_next("post_queued", RuntimeError("chat 503"))
    entry = _claim(dedup, "m1")

    with pytest.raises(RuntimeError):
        park(ops, dedup, "m1", entry, RETRYABLE)

    assert dedup.get("m1")["status"] == "queued"
    assert "m1" in dedup.queued()


def test_failed_queued_notice_still_superseded_the_older_sibling(dedup):
    """Queue state is committed before the notice, so a raise cannot leave the queue
    half-collapsed."""
    ops = _ops()
    ops.fail_next("post_queued", RuntimeError("chat 503"))
    _seed_queued(dedup, "old")
    entry = _claim(dedup, "new")

    with pytest.raises(RuntimeError):
        park(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("old")["status"] == SUPERSEDED_STATUS


# ===========================================================================
# Section 3 — park: the CAS
# ===========================================================================


def test_park_loses_the_cas_when_the_entry_already_moved(dedup):
    """A racing drain (or a late terminal callback) owns the entry now — parking on
    top of it would resurrect work someone else settled."""
    ops = _ops()
    entry = _claim(dedup, "m1")
    dedup.update_status("m1", "completed")

    assert park(ops, dedup, "m1", entry, RETRYABLE) is False

    assert dedup.get("m1")["status"] == "completed"
    assert ops.count("post_queued") == 0


def test_a_park_that_lost_the_cas_supersedes_nothing(dedup):
    """The precondition again, at CAS granularity: no park, no supersession."""
    ops = _ops()
    _seed_queued(dedup, "old")
    entry = _claim(dedup, "new")
    dedup.update_status("new", "completed")

    assert park(ops, dedup, "new", entry, RETRYABLE) is False

    assert dedup.get("old")["status"] == "queued"
    assert ops.count("post_superseded") == 0


def test_park_of_an_unknown_message_is_a_no_op(dedup):
    ops = _ops()

    assert park(ops, dedup, "ghost", {"status": "pending"}, RETRYABLE) is False

    assert dedup.get("ghost") is None
    assert ops.count("post_queued") == 0


# ===========================================================================
# Section 4 — supersession preconditions (the invariant that is easy to lose)
# ===========================================================================


def _newer_message_finishes(ops, dedup, message_id, entry, result):
    """The caller contract park depends on, spelled out: the pipeline parks a message
    ONLY when its failure is retryable, and otherwise settles it terminally. These
    three tests drive that rule rather than calling ``park`` directly, so they fail if
    supersession ever migrates onto a non-park path."""
    if is_retryable(result):
        return park(ops, dedup, message_id, entry, result)
    dedup.record(message_id, entry.get("run_id"), result.get("status", "completed"))
    return False


def test_older_sibling_superseded_when_newer_message_parks(dedup):
    ops = _ops()
    _seed_queued(dedup, "old")
    entry = _claim(dedup, "new")

    _newer_message_finishes(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("old")["status"] == SUPERSEDED_STATUS
    assert ops.count("post_superseded") == 1
    assert ops.of("post_superseded")[0]["entry"]["status"] == SUPERSEDED_STATUS
    assert dedup.get("new")["status"] == "queued"


def test_older_sibling_untouched_when_newer_message_dispatches_successfully(dedup):
    """The failure mode this guards: the user's older queued question is cancelled by
    a newer one that then succeeds — they never get the first answer and are never
    told why."""
    ops = _ops()
    _seed_queued(dedup, "old")
    entry = _claim(dedup, "new")

    _newer_message_finishes(
        ops, dedup, "new", entry, {"status": "completed", "answer": "beam is up"}
    )

    assert dedup.get("old")["status"] == "queued"
    assert "old" in dedup.queued()
    assert ops.count("post_superseded") == 0


def test_older_sibling_untouched_while_newer_message_is_in_flight(dedup):
    """An in-flight newer message has not taken the older one's place — it may yet
    succeed, fail non-retryably, or crash. Only its park may supersede."""
    ops = _ops()
    _seed_queued(dedup, "old")
    _claim(dedup, "new")

    dedup.record("new", "run-7", "in_flight")

    assert dedup.get("old")["status"] == "queued"
    assert "old" in dedup.queued()
    assert ops.count("post_superseded") == 0


def test_non_retryable_failure_of_the_newer_message_supersedes_nothing(dedup):
    """A failed-but-unretryable newer message ends with its own error reply; the older
    queued question is still owed an answer."""
    ops = _ops()
    _seed_queued(dedup, "old")
    entry = _claim(dedup, "new")

    _newer_message_finishes(ops, dedup, "new", entry, {"status": "error", "failure_class": "agent"})

    assert dedup.get("old")["status"] == "queued"
    assert ops.count("post_superseded") == 0


# ===========================================================================
# Section 5 — supersede-at-enqueue: matching, the CAS, and best-effort notes
# ===========================================================================


def test_supersession_skips_the_parking_entry_itself(dedup):
    """The entry is already ``queued`` by the time siblings are scanned, so the
    self-exclusion is load-bearing: without it a park cancels itself."""
    ops = _ops()
    entry = _claim(dedup, "m1")

    park(ops, dedup, "m1", entry, RETRYABLE)

    assert dedup.get("m1")["status"] == "queued"
    assert ops.count("post_superseded") == 0


def test_sibling_with_a_different_key_is_untouched(dedup):
    ops = _ops()
    _seed_queued(dedup, "other", key=["space/2", "users/bob"])
    entry = _claim(dedup, "new")

    park(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("other")["status"] == "queued"
    assert ops.count("post_superseded") == 0


def test_keyless_entries_stand_alone(dedup):
    """``""``/``()`` is the no-group sentinel, not a group of its own — two keyless
    entries must never collapse into each other."""
    ops = _ops(coalesce="")
    _seed_queued(dedup, "old", key=[])
    entry = _claim(dedup, "new")

    park(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("old")["status"] == "queued"
    assert ops.count("post_superseded") == 0


def test_scalar_and_sequence_keys_normalize_to_the_same_group(dedup):
    """A sibling persisted from a scalar channel key still matches a scalar park."""
    ops = _ops(coalesce="thread-root")
    _seed_queued(dedup, "old", key="thread-root")
    entry = _claim(dedup, "new")

    park(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("old")["status"] == SUPERSEDED_STATUS


def test_all_older_siblings_in_the_group_are_dropped(dedup):
    ops = _ops()
    _seed_queued(dedup, "old1")
    _seed_queued(dedup, "old2")
    entry = _claim(dedup, "new")

    park(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("old1")["status"] == SUPERSEDED_STATUS
    assert dedup.get("old2")["status"] == SUPERSEDED_STATUS
    assert ops.count("post_superseded") == 2


def test_cas_is_respected_for_an_entry_that_left_the_queue(dedup):
    """A drain that already claimed the sibling (``queued -> in_flight``) wins the
    race: we neither drop it nor post a note it would contradict."""
    ops = _ops()
    _seed_queued(dedup, "old")
    dedup.transition("old", "queued", "in_flight", retried=True)
    entry = _claim(dedup, "new")

    park(ops, dedup, "new", entry, RETRYABLE)

    assert dedup.get("old")["status"] == "in_flight"
    assert ops.count("post_superseded") == 0


def test_failed_supersede_note_does_not_disturb_the_pass(dedup):
    """The coalesce CAS is already committed — a failing note must not abort the
    remaining siblings, nor the park's own queued notice."""
    ops = _ops()
    ops.fail_next("post_superseded", RuntimeError("chat 503"))
    _seed_queued(dedup, "old1")
    _seed_queued(dedup, "old2")
    entry = _claim(dedup, "new")

    assert park(ops, dedup, "new", entry, RETRYABLE) is True

    assert dedup.get("old1")["status"] == SUPERSEDED_STATUS
    assert dedup.get("old2")["status"] == SUPERSEDED_STATUS
    assert ops.count("post_superseded") == 2
    assert ops.count("post_queued") == 1


def test_supersede_at_enqueue_returns_the_dropped_ids(dedup):
    ops = _ops()
    _seed_queued(dedup, "old")
    _seed_queued(dedup, "other", key=["space/2", "users/bob"])

    dropped = supersede_at_enqueue(ops, dedup, "new", ("space/1", "users/alice"))

    assert dropped == ["old"]


def test_supersede_at_enqueue_ignores_the_empty_key(dedup):
    ops = _ops()
    _seed_queued(dedup, "old", key=[])

    assert supersede_at_enqueue(ops, dedup, "new", ()) == []

    assert dedup.get("old")["status"] == "queued"
    assert ops.count("post_superseded") == 0


def test_superseded_status_is_terminal(dedup):
    """If it were not terminal, ``reconcile()`` would re-surface a dropped question on
    the next restart and re-run it."""
    assert SUPERSEDED_STATUS in TERMINAL_STATUSES

    _seed_queued(dedup, "old")
    park(_ops(), dedup, "new", _claim(dedup, "new"), RETRYABLE)

    assert "old" not in dedup.queued()
    assert "old" not in dedup.reconcile()


# ===========================================================================
# Section 6 — give-up and the drain-side supersede note
# ===========================================================================


def test_give_up_posts_the_persisted_entry(dedup):
    ops = _ops()
    entry = _claim(dedup, "m1")
    park(ops, dedup, "m1", entry, RETRYABLE)

    give_up(ops, dedup, "m1", entry)

    assert ops.count("post_giveup") == 1
    assert ops.of("post_giveup")[0]["entry"]["status"] == "queued"


def test_give_up_leaves_the_terminal_mark_to_the_drain(dedup):
    """The drain marks terminal only AFTER this returns — so a notice that fails to
    send leaves the entry queued for the next cycle instead of silently dropped."""
    ops = _ops()
    entry = _claim(dedup, "m1")
    park(ops, dedup, "m1", entry, RETRYABLE)

    give_up(ops, dedup, "m1", entry)

    assert dedup.get("m1")["status"] == "queued"


def test_give_up_raises_when_the_notice_fails(dedup):
    """``post_giveup`` MUST raise and the engine MUST let it through: a lost give-up
    notice is silence, worse than a rare duplicate."""
    ops = _ops()
    ops.fail_next("post_giveup", RuntimeError("chat 503"))
    entry = _claim(dedup, "m1")

    with pytest.raises(RuntimeError):
        give_up(ops, dedup, "m1", entry)

    assert dedup.get("m1")["status"] == "pending"


def test_give_up_records_the_failed_turn_in_history(dedup, tmp_path):
    history = HistoryStore(str(tmp_path / "history.json"))
    ops = _ops()
    entry = _claim(dedup, "m1")

    give_up(ops, dedup, "m1", entry, history=history)

    turns = history.recent("space/1")
    assert len(turns) == 1
    assert turns[0]["question"] == "why is the beam down?"


def test_give_up_skips_history_when_the_notice_failed(dedup, tmp_path):
    """No notice landed, so the drain will retry the whole give-up next cycle — a
    history turn written now would be duplicated."""
    history = HistoryStore(str(tmp_path / "history.json"))
    ops = _ops()
    ops.fail_next("post_giveup", RuntimeError("chat 503"))
    entry = _claim(dedup, "m1")

    with pytest.raises(RuntimeError):
        give_up(ops, dedup, "m1", entry, history=history)

    assert history.recent("space/1") == []


def test_give_up_survives_a_failing_history_write(dedup):
    """The notice already landed; a history failure must not turn a delivered notice
    into a retried one."""

    class BrokenHistory:
        def append_failed(self, key, question):
            raise RuntimeError("disk full")

    ops = _ops()
    entry = _claim(dedup, "m1")

    give_up(ops, dedup, "m1", entry, history=BrokenHistory())

    assert ops.count("post_giveup") == 1


def test_give_up_without_a_history_key_posts_only(dedup, tmp_path):
    history = HistoryStore(str(tmp_path / "history.json"))
    ops = _ops()
    dedup.claim("m1", text="hi", history_key="")

    give_up(ops, dedup, "m1", dedup.get("m1"), history=history)

    assert ops.count("post_giveup") == 1
    assert history.recent("") == []


def test_supersede_note_posts_the_persisted_entry(dedup):
    ops = _ops()
    entry = _seed_queued(dedup, "old")

    supersede_note(ops, dedup, "old", entry)

    assert ops.count("post_superseded") == 1
    assert ops.of("post_superseded")[0]["entry"]["text"] == "why is the beam down?"


def test_supersede_note_never_raises(dedup):
    ops = _ops()
    ops.fail_next("post_superseded", RuntimeError("chat 503"))
    entry = _seed_queued(dedup, "old")

    supersede_note(ops, dedup, "old", entry)

    assert ops.count("post_superseded") == 1
