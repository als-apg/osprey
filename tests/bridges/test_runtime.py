"""Tests for the runtime shell: collaborator wiring, startup recovery, drain supervision.

Stores are the real ``DedupStore``/``HistoryStore`` on temp files; the dispatcher is a
stub that records its payload, marks itself into the ``RecordingChannelOps`` timeline and
returns a canned terminal result. ``run_drain_thread`` is monkeypatched wherever a test
cares about *whether* a drain started rather than what it does — the drain loop itself is
covered by ``test_drain.py``, and no test here sleeps or waits on real time.

The load-bearing invariants, each with a named test:

* collaborators come from cfg + ops        — ``test_build_deps_constructs_collaborators_from_cfg_and_ops``
* recovery completes BEFORE the drain runs — ``test_reconcile_runs_before_the_drain_starts``
* exactly ONE drain thread                 — ``test_start_starts_exactly_one_drain_thread``
* a dead drain is replaced                 — ``test_supervise_restarts_a_dead_drain``
* the gate is the health gate              — ``test_gate_defaults_to_health_gate_gate_open``
* callbacks hit the right ChannelOps member— ``test_deliver_callback_posts_answer_then_files_then_history`` and friends
* a re-dispatch replays the live payload   — ``test_rebuild_extra_replays_history_reply_and_inputs``
"""

from __future__ import annotations

import base64
import dataclasses
import threading
from typing import Any

import pytest

from osprey.bridges.core import runtime
from osprey.bridges.core.config import CoreConfig
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.dispatch_client import DispatchClient
from osprey.bridges.core.history import HistoryStore
from osprey.bridges.core.pipeline import PipelineDeps
from osprey.bridges.core.ports import InputDownload
from osprey.bridges.core.reconcile import ReconcileDeps
from tests.bridges.conftest import RecordingChannelOps

QUESTION = "what is the orbit doing?"
HISTORY_KEY = "room/AAA"

COMPLETED = {
    "status": "completed",
    "text_output": "The orbit is stable.",
    "run_id": "run-1",
    "num_tool_calls": 3,
}

# A bridge-local infrastructure failure: retryable, so the pipeline settlement parks it.
RETRYABLE = {
    "status": "error",
    "text_output": "",
    "run_id": "run-1",
    "error": "dispatcher unreachable",
    "failure_class": "infrastructure",
    "num_tool_calls": None,
}


class FakeDispatcher:
    """Stands in for ``DispatchClient``: records dispatches, marks the timeline."""

    def __init__(self, ops: RecordingChannelOps, result: dict[str, Any] | None = None) -> None:
        self.ops = ops
        self.result = result if result is not None else COMPLETED
        self.runs: list[tuple[str, dict[str, Any] | None]] = []
        self.polled: list[str] = []
        self.bodies: dict[str, dict[str, Any] | None] = {}

    def run(self, question, extra=None, *, on_run_id=None):
        self.runs.append((question, extra))
        self.ops.mark("dispatch", question=question, extra=extra)
        if on_run_id is not None:
            on_run_id("run-new")
        return dict(self.result)

    def poll_worker(self, run_id, deadline=None):
        self.polled.append(run_id)
        self.ops.mark("poll_worker", run_id=run_id)
        return dict(self.result)

    def status(self, run_id):
        return self.bodies.get(run_id)


def _cfg(tmp_path, **over: Any) -> CoreConfig:
    fields: dict[str, Any] = {
        "dedup_path": str(tmp_path / "dedup.json"),
        "history_path": str(tmp_path / "history.json"),
        "drain_interval": 0.01,
    }
    fields.update(over)
    return CoreConfig(**fields)


def _deps(tmp_path, ops: RecordingChannelOps, **over: Any) -> PipelineDeps:
    """A deps bundle on real stores with a stub dispatcher and a capable pair."""
    cfg = over.pop("cfg", None) or _cfg(tmp_path)
    deps = PipelineDeps(
        cfg=cfg,
        ops=ops,
        dedup=DedupStore(cfg.dedup_path),
        dispatcher=over.pop("dispatcher", None) or FakeDispatcher(ops),
        history=HistoryStore(cfg.history_path),
        probe_capability=lambda cfg, capability: True,
    )
    return dataclasses.replace(deps, **over) if over else deps


@pytest.fixture
def spawn():
    """Make real ``Thread`` objects in a known liveness state, with no drain loop and no
    sleeping: each blocks on its own event, which the fixture releases at teardown."""
    releases: list[threading.Event] = []

    def make(alive: bool = True) -> threading.Thread:
        release = threading.Event()
        releases.append(release)
        thread = threading.Thread(target=release.wait, daemon=True)
        thread.start()
        if not alive:
            release.set()
            thread.join(timeout=5.0)
        return thread

    yield make
    for release in releases:
        release.set()


@pytest.fixture
def no_drain(monkeypatch, spawn):
    """Replace ``run_drain_thread`` with a recorder; returns the list of DrainDeps it was
    started with (so ``len(started)`` is the thread count)."""
    started: list[Any] = []

    def fake_run_drain_thread(deps, stop=None):
        started.append(deps)
        return spawn()

    monkeypatch.setattr(runtime, "run_drain_thread", fake_run_drain_thread)
    return started


def _claim(store: DedupStore, message_id: str, **meta: Any) -> dict[str, Any]:
    store.claim(message_id, **meta)
    return store.get(message_id) or {}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# --- collaborator construction -----------------------------------------------


def test_build_deps_constructs_collaborators_from_cfg_and_ops(tmp_path):
    cfg = _cfg(tmp_path)
    ops = RecordingChannelOps()

    deps = runtime.build_deps(cfg, ops)

    assert deps.cfg is cfg
    assert deps.ops is ops
    assert isinstance(deps.dedup, DedupStore) and deps.dedup.path == cfg.dedup_path
    assert isinstance(deps.history, HistoryStore) and deps.history.path == cfg.history_path
    assert isinstance(deps.dispatcher, DispatchClient)


def test_build_deps_bundle_also_satisfies_the_reconcile_surface(tmp_path):
    """One bundle serves pipeline, reconcile and the drain wiring — reconcile reads
    exactly ``dedup`` / ``dispatcher`` / ``ops`` off it, so there is no second place a
    collaborator could be constructed differently."""
    ops = RecordingChannelOps()
    deps = runtime.build_deps(_cfg(tmp_path), ops)
    surface: ReconcileDeps = deps

    assert surface.ops is ops
    assert surface.dedup is deps.dedup
    assert surface.dispatcher is deps.dispatcher


def test_build_deps_uses_injected_stores_verbatim(tmp_path):
    ops = RecordingChannelOps()
    dedup = DedupStore(str(tmp_path / "other-dedup.json"))
    history = HistoryStore(str(tmp_path / "other-history.json"))
    dispatcher = FakeDispatcher(ops)

    deps = runtime.build_deps(
        _cfg(tmp_path), ops, dedup=dedup, history=history, dispatcher=dispatcher
    )

    assert deps.dedup is dedup and deps.history is history and deps.dispatcher is dispatcher


# --- startup ordering ---------------------------------------------------------


def test_reconcile_runs_before_the_drain_starts(tmp_path, monkeypatch, spawn):
    """Crash recovery must complete before anything new can be accepted: a message the
    user is already waiting on is answered before the drain competes for the worker."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    deps.dedup.record("m1", run_id="run-1", status="in_flight")

    def marking_run_drain_thread(drain_deps, stop=None):
        ops.mark("drain_started")
        return spawn()

    monkeypatch.setattr(runtime, "run_drain_thread", marking_run_drain_thread)
    runtime.start(deps.cfg, ops, deps=deps)

    ops.assert_order("poll_worker", "post_answer", "drain_started")
    ops.assert_never_before("drain_started", "poll_worker")
    assert deps.dedup.get("m1")["status"] == "completed"


def test_startup_recovery_settles_through_the_pipeline(tmp_path, no_drain):
    """The injected settlement is the pipeline's: a retryable failure found at startup is
    PARKED for the drain (queued + queued notice), not reported as a delivered error."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops, dispatcher=None)
    deps = dataclasses.replace(deps, dispatcher=FakeDispatcher(ops, RETRYABLE))
    _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    deps.dedup.record("m1", run_id="run-1", status="in_flight")

    runtime.start(deps.cfg, ops, deps=deps)

    assert deps.dedup.get("m1")["status"] == "queued"
    assert ops.count("post_queued") == 1
    assert ops.count("post_answer") == 0


def test_startup_recovery_replays_the_live_payload(tmp_path, no_drain):
    """A re-dispatch (crashed before a run id existed) carries the same context a live
    dispatch would — proof the pipeline's context builder is the injected one."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    deps.history.append(HISTORY_KEY, "earlier?", "earlier answer")
    _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)

    runtime.start(deps.cfg, ops, deps=deps)

    (question, extra) = deps.dispatcher.runs[0]
    assert question == QUESTION
    assert [turn["answer"] for turn in extra["conversation_so_far"]] == ["earlier answer"]


# --- drain supervision --------------------------------------------------------


def test_start_starts_exactly_one_drain_thread(tmp_path, no_drain):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)

    bridge = runtime.start(deps.cfg, ops, deps=deps)

    assert len(no_drain) == 1
    assert no_drain[0] is bridge.drain
    assert bridge.thread.is_alive()


def test_supervise_is_a_noop_while_the_drain_lives(tmp_path, no_drain):
    ops = RecordingChannelOps()
    bridge = runtime.start(_cfg(tmp_path), ops, deps=_deps(tmp_path, ops))
    live = bridge.thread

    assert bridge.supervise() is live
    assert len(no_drain) == 1


def test_supervise_restarts_a_dead_drain(tmp_path, no_drain, spawn):
    ops = RecordingChannelOps()
    bridge = runtime.start(_cfg(tmp_path), ops, deps=_deps(tmp_path, ops))
    dead = spawn(alive=False)
    bridge.thread = dead

    replacement = bridge.supervise()

    assert replacement is not dead and replacement.is_alive()
    assert bridge.thread is replacement
    assert len(no_drain) == 2


def test_shutdown_signals_the_drain_to_stop(tmp_path, no_drain):
    ops = RecordingChannelOps()
    bridge = runtime.start(_cfg(tmp_path), ops, deps=_deps(tmp_path, ops))

    assert not bridge.stop.is_set()
    bridge.shutdown()
    assert bridge.stop.is_set()


def test_the_real_drain_thread_is_a_named_daemon(tmp_path):
    """No monkeypatch here: the ONE thread the runtime really starts must be a daemon so
    it never blocks process exit. The gate is injected, so the loop does no I/O."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)

    bridge = runtime.start(deps.cfg, ops, deps=deps, gate=lambda: False)
    try:
        assert bridge.thread.daemon and bridge.thread.name == "bridge-drain"
        drains = [t for t in threading.enumerate() if t.name == "bridge-drain"]
        assert drains == [bridge.thread]
    finally:
        bridge.shutdown()
        bridge.thread.join(timeout=5.0)
    assert not bridge.thread.is_alive()


# --- the health gate ----------------------------------------------------------


def test_gate_defaults_to_health_gate_gate_open(tmp_path, monkeypatch):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    seen: list[Any] = []
    monkeypatch.setattr(runtime, "gate_open", lambda cfg: seen.append(cfg) or False)

    assert runtime.build_drain_deps(deps).gate_is_open() is False
    assert seen == [deps.cfg]


def test_injected_gate_replaces_the_health_gate(tmp_path, monkeypatch):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    monkeypatch.setattr(
        runtime, "gate_open", lambda cfg: pytest.fail("health gate must not be consulted")
    )

    assert runtime.build_drain_deps(deps, gate=lambda: True).gate_is_open() is True


# --- drain callbacks over ChannelOps ------------------------------------------


def test_dispatch_callback_redispatches_and_settles(tmp_path):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)

    runtime.build_drain_callbacks(deps).dispatch("m1", entry)

    assert deps.dispatcher.runs[0][0] == QUESTION
    ops.assert_order("dispatch", "post_answer")
    assert deps.dedup.get("m1")["status"] == "completed"


def test_dispatch_callback_abandons_an_entry_with_no_text(tmp_path):
    """Nothing to replay and nothing to re-attach: terminal, or the drain retries it
    forever."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", history_key=HISTORY_KEY)

    runtime.build_drain_callbacks(deps).dispatch("m1", entry)

    assert deps.dispatcher.runs == []
    assert deps.dedup.get("m1")["status"] == "error"


def test_deliver_callback_posts_answer_then_files_then_history(tmp_path):
    ops = RecordingChannelOps(file_urls={"a1": "https://files/a1.png"})
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    deps.dedup.record("m1", run_id="run-1", status="queued")

    runtime.build_drain_callbacks(deps).deliver(
        "m1", entry, {"status": "completed", "text_output": "The orbit is stable."}
    )

    ops.assert_order("post_answer", "deliver_files")
    # The raw worker body omits the run id; the binding grafts the persisted one on.
    assert ops.of("post_answer")[0]["result"]["run_id"] == "run-1"
    assert [turn["answer"] for turn in deps.history.recent(HISTORY_KEY)] == ["The orbit is stable."]
    # The drain owns the terminal mark; the callback must not write status.
    assert deps.dedup.get("m1")["status"] == "queued"


def test_deliver_callback_propagates_a_failed_post(tmp_path):
    """The raise is the drain's at-least-once signal — swallowing it would drop the
    answer, and nothing after the failed post may run."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    ops.fail_next("post_answer", RuntimeError("send failed"))

    with pytest.raises(RuntimeError, match="send failed"):
        runtime.build_drain_callbacks(deps).deliver("m1", entry, dict(COMPLETED))

    assert ops.count("deliver_files") == 0
    assert deps.history.recent(HISTORY_KEY) == []


def test_deliver_callback_survives_a_raising_file_delivery(tmp_path):
    """``deliver_files`` must never raise by contract, but the answer has already landed:
    a misbehaving adapter must not cost the history turn or force a re-delivery."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    ops.fail_next("deliver_files", RuntimeError("upload exploded"))

    runtime.build_drain_callbacks(deps).deliver("m1", entry, dict(COMPLETED))

    assert [turn["answer"] for turn in deps.history.recent(HISTORY_KEY)] == [
        COMPLETED["text_output"]
    ]


def test_give_up_callback_posts_the_notice_and_records_the_failed_turn(tmp_path):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)

    runtime.build_drain_callbacks(deps).give_up("m1", entry)

    assert ops.count("post_giveup") == 1
    assert deps.history.recent(HISTORY_KEY) != []


def test_give_up_callback_propagates_a_failed_notice(tmp_path):
    """A lost give-up notice is silence: the raise leaves the entry queued for the next
    cycle rather than marking it terminal."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    ops.fail_next("post_giveup", RuntimeError("notice failed"))

    with pytest.raises(RuntimeError, match="notice failed"):
        runtime.build_drain_callbacks(deps).give_up("m1", entry)


def test_supersede_callback_is_best_effort(tmp_path):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    entry = _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    ops.fail_next("post_superseded", RuntimeError("note failed"))

    runtime.build_drain_callbacks(deps).supersede("m1", entry)  # must not raise

    assert ops.count("post_superseded") == 1
    assert deps.history.recent(HISTORY_KEY) == []  # a dropped turn writes no history


def test_adapters_implement_channel_ops_only(tmp_path):
    """The recording double implements ``ChannelOps`` and nothing else; every drain
    callback is bound from it, so no adapter ever implements both surfaces."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)

    callbacks = runtime.build_drain_callbacks(deps)

    assert not hasattr(ops, "dispatch") and not hasattr(ops, "deliver")
    assert callbacks.supersede is not None
    assert all(callable(cb) for cb in (callbacks.dispatch, callbacks.deliver, callbacks.give_up))


# --- re-dispatch payload ------------------------------------------------------


def test_rebuild_extra_replays_history_reply_and_inputs(tmp_path):
    ops = RecordingChannelOps(
        inputs=InputDownload(
            files=[
                {
                    "filename": "orbit.png",
                    "mime": "image/png",
                    "content_b64": _b64(b"bytes"),
                    "ingest": True,
                }
            ]
        )
    )
    deps = _deps(tmp_path, ops)
    deps.history.append(HISTORY_KEY, "earlier?", "earlier answer")
    _claim(deps.dedup, "m1", text=QUESTION, history_key=HISTORY_KEY)
    deps.dedup.record(
        "m1", run_id=None, status="pending", reply_to={"sender": "Pat", "text": "the plot"}
    )

    extra = runtime.rebuild_extra(deps, deps.dedup.get("m1"))

    assert [turn["answer"] for turn in extra["conversation_so_far"]] == ["earlier answer"]
    assert extra["reply_to"]["sender"] == "Pat"
    assert [f["filename"] for f in extra["input_files"]] == ["orbit.png"]
    # A COPY of the persisted reply context: per-dispatch keys must not leak into the store.
    assert extra["reply_to"] is not deps.dedup.get("m1")["reply_to"]


def test_rebuild_extra_is_empty_for_a_text_only_entry(tmp_path):
    """A text-only re-dispatch must stay byte-identical to a live text-only dispatch."""
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    _claim(deps.dedup, "m1", text=QUESTION, history_key="")

    assert runtime.rebuild_extra(deps, deps.dedup.get("m1")) == {}


# --- the adapter entry point --------------------------------------------------


def test_run_forever_serves_the_adapter_loop_then_stops(tmp_path, no_drain):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    seen: dict[str, Any] = {}

    def serve(bridge: runtime.BridgeRuntime) -> None:
        seen["bridge"] = bridge
        seen["running"] = not bridge.stop.is_set()
        # The adapter owns ingestion and feeds the engine one event at a time.
        seen["outcome"] = bridge.handle_event({"raw": "wire event"})
        seen["supervised"] = bridge.supervise()

    runtime.run_forever(deps.cfg, ops, serve, deps=deps)

    assert seen["running"] is True
    assert seen["outcome"] == "ignored"  # the default double parses nothing
    assert seen["supervised"] is seen["bridge"].thread
    assert seen["bridge"].stop.is_set()
    assert len(no_drain) == 1


def test_run_forever_stops_the_drain_when_the_loop_raises(tmp_path, no_drain):
    ops = RecordingChannelOps()
    deps = _deps(tmp_path, ops)
    captured: dict[str, Any] = {}

    def serve(bridge: runtime.BridgeRuntime) -> None:
        captured["bridge"] = bridge
        raise RuntimeError("subscription died")

    with pytest.raises(RuntimeError, match="subscription died"):
        runtime.run_forever(deps.cfg, ops, serve, deps=deps)

    assert captured["bridge"].stop.is_set()


# --- the package export surface -----------------------------------------------


def test_package_exports_the_documented_engine_surface():
    import osprey.bridges.core as core
    from osprey.bridges.core import pipeline, ports

    assert len(core.__all__) == len(set(core.__all__))
    missing = [name for name in core.__all__ if not hasattr(core, name)]
    assert missing == []
    assert core.ChannelOps is ports.ChannelOps
    assert core.RESERVED_ENTRY_KEYS is ports.RESERVED_ENTRY_KEYS
    assert core.handle_event is pipeline.handle_event
    assert core.run_forever is runtime.run_forever
