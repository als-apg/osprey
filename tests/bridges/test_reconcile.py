"""Startup crash-recovery suite.

The scenarios are written as *crashes*: a real :class:`DedupStore` on ``tmp_path`` is
left in exactly the state a process death would leave it in, then
:func:`reconcile_inflight` runs against it and the store's own contents are the
assertion. The dispatcher is a recording fake — the point of most of these tests is
which dispatcher calls did NOT happen.

The load-bearing one is
:func:`test_reattach_polls_existing_run_and_never_dispatches`: an entry that already
carries a run_id must be POLLED, never re-fired, because that run may already have
reached hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.reconcile import ReconcileReport, deliver_terminal, reconcile_inflight

from .conftest import RecordingChannelOps


class FakeDispatcher:
    """A ``DispatchClient`` stand-in recording every call it receives.

    ``poll_result`` / ``run_result`` are the terminal bodies handed back; ``fire`` is
    present (and counted) purely so a test can prove it was never reached.
    """

    def __init__(
        self,
        *,
        poll_result: dict[str, Any] | None = None,
        run_result: dict[str, Any] | None = None,
        run_id: str = "run-new",
    ) -> None:
        self.poll_result = poll_result or {"status": "completed", "text_output": "polled"}
        self.run_result = run_result or {"status": "completed", "text_output": "ran"}
        self.run_id = run_id
        self.polled: list[str] = []
        self.ran: list[tuple[str, dict[str, Any] | None]] = []
        self.fired: list[str] = []
        self.raise_on_poll: BaseException | None = None
        self.raise_on_run: BaseException | None = None

    # -- DispatchClient surface ----------------------------------------------

    def poll_worker(self, run_id: str, deadline: float | None = None) -> dict[str, Any]:
        self.polled.append(run_id)
        if self.raise_on_poll is not None:
            raise self.raise_on_poll
        return dict(self.poll_result, run_id=run_id)

    def run(self, question, extra=None, *, on_run_id=None) -> dict[str, Any]:
        self.ran.append((question, extra))
        if self.raise_on_run is not None:
            raise self.raise_on_run
        if on_run_id is not None:
            on_run_id(self.run_id)
        return dict(self.run_result, run_id=self.run_id)

    def fire(self, question: str, extra: dict[str, Any] | None = None) -> str:
        self.fired.append(question)
        return "dispatch-1"

    @property
    def dispatch_calls(self) -> int:
        """Every call that would START a new run — the number that must stay 0 on the
        re-attach path."""
        return len(self.ran) + len(self.fired)


@dataclass
class Deps:
    """A ``ReconcileDeps``-shaped bundle (structural, not a subclass)."""

    dedup: DedupStore
    dispatcher: FakeDispatcher
    ops: RecordingChannelOps = field(default_factory=RecordingChannelOps)


@pytest.fixture
def store(tmp_path) -> DedupStore:
    return DedupStore(str(tmp_path / "dedup.json"))


def _crashed_mid_poll(store: DedupStore, message_id: str = "m1", run_id: str = "run-42") -> None:
    """Leave the store as a crash between the dispatcher handshake and the answer:
    claimed with its question, a run_id recorded, status still non-terminal."""
    store.claim(message_id, text="how is the beam?", history_key="room-1")
    store.record(message_id, run_id=run_id, status="in_flight")


def _crashed_before_run_id(store: DedupStore, message_id: str = "m1") -> None:
    """Leave the store as a crash after the claim but before any run existed."""
    store.claim(message_id, text="how is the beam?", history_key="room-1")


# --- re-attach: the case that must never re-dispatch --------------------------


def test_reattach_polls_existing_run_and_never_dispatches(store):
    """An entry carrying a run_id is polled to terminal and delivered with ZERO
    additional dispatch calls.

    This is the phase's crash-mid-run criterion. A run_id means a run was started and
    may already have moved hardware; re-dispatching would duplicate those side effects.
    Asserting only "the answer arrived" would pass even for a re-dispatch, so the
    assertion that matters is the dispatch count.
    """
    _crashed_mid_poll(store, run_id="run-42")
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert deps.dispatcher.dispatch_calls == 0, "re-attach must not start a second run"
    assert deps.dispatcher.ran == []
    assert deps.dispatcher.fired == []
    assert deps.dispatcher.polled == ["run-42"]
    assert report == ReconcileReport(reattached=1)
    assert deps.ops.count("post_answer") == 1
    assert deps.ops.of("post_answer")[0]["result"]["text_output"] == "polled"
    assert store.get("m1")["status"] == "completed"


def test_reattach_delivers_files_after_the_answer(store):
    _crashed_mid_poll(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    reconcile_inflight(deps)

    deps.ops.assert_order("post_answer", "deliver_files")


def test_reattach_marks_the_result_status_terminal(store):
    _crashed_mid_poll(store)
    deps = Deps(
        dedup=store,
        dispatcher=FakeDispatcher(poll_result={"status": "error", "error": "worker blew up"}),
    )

    reconcile_inflight(deps)

    assert store.get("m1")["status"] == "error"


def test_reattach_coerces_an_unknown_status_to_error(store):
    """A worker body with a status outside the terminal set must still settle: leaving
    it non-terminal would make the entry reconcilable forever."""
    _crashed_mid_poll(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher(poll_result={"status": "weird"}))

    reconcile_inflight(deps)

    assert store.get("m1")["status"] == "error"


# --- re-dispatch: no run was ever started -------------------------------------


def test_redispatch_when_claimed_before_a_run_id_existed(store):
    _crashed_before_run_id(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert [q for q, _ in deps.dispatcher.ran] == ["how is the beam?"]
    assert deps.dispatcher.polled == []
    assert report == ReconcileReport(redispatched=1)
    assert store.get("m1")["status"] == "completed"


def test_redispatch_persists_the_new_run_id_before_the_poll(store):
    """The re-dispatch installs the same crash-safety persister the live path uses, so
    a crash during reconcile's own dispatch is re-attachable next startup rather than
    dispatched a second time."""
    seen: list[str | None] = []
    _crashed_before_run_id(store)
    dispatcher = FakeDispatcher(run_id="run-99")

    def run(question, extra=None, *, on_run_id=None):
        dispatcher.ran.append((question, extra))
        on_run_id("run-99")
        seen.append(store.get("m1")["run_id"])  # visible BEFORE the terminal result
        return {"status": "completed", "text_output": "ran", "run_id": "run-99"}

    dispatcher.run = run  # type: ignore[method-assign]
    deps = Deps(dedup=store, dispatcher=dispatcher)

    reconcile_inflight(deps)

    assert seen == ["run-99"]
    assert store.get("m1")["run_id"] == "run-99"


def test_redispatch_without_a_builder_ships_the_question_alone(store):
    _crashed_before_run_id(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    reconcile_inflight(deps)

    assert deps.dispatcher.ran == [("how is the beam?", None)]


def test_build_extra_seam_supplies_the_redispatch_context(store):
    """The runtime injects the pipeline's context builder so a replayed question
    carries the same conversation/attachment context a live dispatch would."""
    _crashed_before_run_id(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())
    entries: list[str] = []

    def build_extra(entry):
        entries.append(entry["history_key"])
        return {"conversation_so_far": [{"q": "earlier"}]}

    reconcile_inflight(deps, build_extra=build_extra)

    assert entries == ["room-1"]
    assert deps.dispatcher.ran == [
        ("how is the beam?", {"conversation_so_far": [{"q": "earlier"}]})
    ]


def test_empty_build_extra_result_is_omitted_from_the_dispatch(store):
    """An empty context must leave the payload byte-identical to the no-context path,
    not ship ``extra={}``."""
    _crashed_before_run_id(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    reconcile_inflight(deps, build_extra=lambda entry: {})

    assert deps.dispatcher.ran == [("how is the beam?", None)]


# --- retried entries belong to the drain --------------------------------------


def test_retried_entry_without_a_run_id_goes_back_to_the_queue(store):
    """A drain redispatch that crashed before a run existed is handed back to the
    queue, never dispatched from startup — the health gate and retry min-age must keep
    mediating it."""
    store.claim("m1", text="how is the beam?")
    store.transition("m1", "pending", "queued")
    store.transition("m1", "queued", "in_flight", retried=True, attempts=1)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert store.get("m1")["status"] == "queued"
    assert deps.dispatcher.dispatch_calls == 0
    assert deps.dispatcher.polled == []
    assert deps.ops.calls == []
    assert report == ReconcileReport(requeued=1)


def test_retried_entry_with_a_run_id_is_still_re_attached(store):
    """``retried`` only diverts entries with nothing to attach to. A retried entry that
    DID reach a run is a crash mid-poll like any other."""
    store.claim("m1", text="how is the beam?")
    store.transition("m1", "pending", "queued")
    store.transition("m1", "queued", "in_flight", retried=True)
    store.record("m1", run_id="run-7", status="in_flight")
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert deps.dispatcher.polled == ["run-7"]
    assert deps.dispatcher.dispatch_calls == 0
    assert report == ReconcileReport(reattached=1)


def test_queued_entries_are_left_to_the_drain(store):
    store.claim("m1", text="how is the beam?")
    store.transition("m1", "pending", "queued")
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert report == ReconcileReport()
    assert store.get("m1")["status"] == "queued"
    assert deps.dispatcher.dispatch_calls == 0


def test_terminal_entries_are_untouched(store):
    store.claim("m1", text="how is the beam?")
    store.update_status("m1", "completed")
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    assert reconcile_inflight(deps) == ReconcileReport()
    assert deps.ops.calls == []


# --- nothing to do / failure handling -----------------------------------------


def test_entry_with_neither_run_id_nor_text_is_left_alone(store):
    """Guessing would either invent a dispatch or drop a live dedup claim."""
    store.claim("m1", history_key="room-1")
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(skipped=1)
    assert store.get("m1")["status"] == "pending"
    assert deps.dispatcher.dispatch_calls == 0
    assert deps.ops.calls == []


def test_a_raising_poll_leaves_the_entry_for_the_next_startup(store):
    _crashed_mid_poll(store)
    dispatcher = FakeDispatcher()
    dispatcher.raise_on_poll = RuntimeError("worker unreachable")
    deps = Deps(dedup=store, dispatcher=dispatcher)

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(failed=1)
    assert store.get("m1")["status"] == "in_flight"  # still reconcilable
    assert deps.ops.count("post_answer") == 0


def test_a_raising_dispatch_leaves_the_entry_for_the_next_startup(store):
    _crashed_before_run_id(store)
    dispatcher = FakeDispatcher()
    dispatcher.raise_on_run = RuntimeError("dispatcher down")
    deps = Deps(dedup=store, dispatcher=dispatcher)

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(failed=1)
    assert store.get("m1")["status"] == "pending"


def test_one_failing_entry_does_not_abort_the_pass(store):
    """Reconcile runs before the bridge subscribes; one bad entry must not stop the
    others (or crashloop the process)."""
    _crashed_mid_poll(store, message_id="bad", run_id="run-bad")
    _crashed_mid_poll(store, message_id="good", run_id="run-good")
    dispatcher = FakeDispatcher()
    original = dispatcher.poll_worker

    def poll(run_id, deadline=None):
        if run_id == "run-bad":
            dispatcher.polled.append(run_id)
            raise RuntimeError("boom")
        return original(run_id, deadline)

    dispatcher.poll_worker = poll  # type: ignore[method-assign]
    deps = Deps(dedup=store, dispatcher=dispatcher)

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(reattached=1, failed=1)
    assert store.get("bad")["status"] == "in_flight"
    assert store.get("good")["status"] == "completed"


def test_a_failed_post_marks_the_entry_post_failed(store):
    """``post_answer`` raises by contract when the answer did not land. At startup that
    is terminal: a permanently unpostable destination must not be retried on every
    restart, and the raise must not escape into the process's boot path."""
    _crashed_mid_poll(store)
    ops = RecordingChannelOps()
    ops.fail_next("post_answer", RuntimeError("thread deleted"))
    deps = Deps(dedup=store, dispatcher=FakeDispatcher(), ops=ops)

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(post_failed=1)
    assert store.get("m1")["status"] == "post_failed"
    assert ops.count("deliver_files") == 0


def test_a_raising_file_delivery_does_not_un_deliver_the_answer(store):
    """``deliver_files`` must never raise, but if an adapter does, the text has already
    landed — the entry still settles terminal."""
    _crashed_mid_poll(store)
    ops = RecordingChannelOps()
    ops.fail_next("deliver_files", RuntimeError("upload failed"))
    deps = Deps(dedup=store, dispatcher=FakeDispatcher(), ops=ops)

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(reattached=1)
    assert store.get("m1")["status"] == "completed"


# --- settlement seam ----------------------------------------------------------


def test_settle_seam_replaces_the_default_settlement(store):
    """The runtime injects the pipeline's settlement so a retryable failure found at
    startup is parked for the drain instead of being marked terminal."""
    _crashed_mid_poll(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher(poll_result={"status": "error"}))
    settled: list[tuple[str, str]] = []

    def settle(message_id, entry, result):
        settled.append((message_id, entry["text"]))
        store.update_status(message_id, "queued")

    report = reconcile_inflight(deps, settle=settle)

    assert settled == [("m1", "how is the beam?")]
    assert store.get("m1")["status"] == "queued"
    assert deps.ops.calls == []  # the seam owns all channel I/O
    assert report == ReconcileReport(reattached=1)


def test_settle_sees_the_entry_as_stored_after_the_redispatch(store):
    """The entry handed to settlement is re-read, so it carries the run_id the
    re-dispatch just persisted rather than the stale pre-crash copy."""
    _crashed_before_run_id(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher(run_id="run-99"))
    seen: list[Any] = []

    reconcile_inflight(deps, settle=lambda mid, entry, result: seen.append(entry["run_id"]))

    assert seen == ["run-99"]


def test_deliver_terminal_is_reusable_on_its_own(store):
    """The default settlement is a public function so the runtime can compose it."""
    _crashed_mid_poll(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())
    entry = store.get("m1")

    assert deliver_terminal(deps, "m1", entry, {"status": "completed"}) is True
    assert store.get("m1")["status"] == "completed"


# --- pass-level behavior ------------------------------------------------------


def test_a_mixed_store_settles_every_entry_in_one_pass(store):
    _crashed_mid_poll(store, message_id="attach", run_id="run-1")
    _crashed_before_run_id(store, message_id="redispatch")
    store.claim("retried", text="q")
    store.transition("retried", "pending", "queued")
    store.transition("retried", "queued", "in_flight", retried=True)
    store.claim("bare")
    store.claim("done", text="q")
    store.update_status("done", "completed")
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    report = reconcile_inflight(deps)

    assert report == ReconcileReport(reattached=1, redispatched=1, requeued=1, skipped=1)
    assert report.total == 4
    assert deps.dispatcher.polled == ["run-1"]
    assert len(deps.dispatcher.ran) == 1


def test_an_empty_store_is_a_no_op(store):
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    assert reconcile_inflight(deps) == ReconcileReport()
    assert deps.dispatcher.dispatch_calls == 0


def test_a_second_pass_finds_nothing_left(store):
    """Idempotence across restarts: a settled entry is terminal, so a bridge that
    restarts twice never re-answers the same message."""
    _crashed_mid_poll(store)
    deps = Deps(dedup=store, dispatcher=FakeDispatcher())

    reconcile_inflight(deps)
    second = reconcile_inflight(deps)

    assert second == ReconcileReport()
    assert deps.dispatcher.polled == ["run-42"]
    assert deps.ops.count("post_answer") == 1
