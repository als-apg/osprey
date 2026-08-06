"""Tests for the channel-agnostic per-message pipeline (``handle_event``).

This suite DEFINES the pipeline's ordering contract — it is not a port. The stores are
the real ``DedupStore``/``HistoryStore`` (tmp-file backed, subclassed only to ``mark``
their calls into the ``RecordingChannelOps`` timeline); the dispatcher is a stub that
records its payload and returns a canned terminal result; the retry-queue park defaults
to the REAL :func:`osprey.bridges.core.retry_queue.park` unless a test injects a seam.

The load-bearing invariants, each with a named test:

* claim precedes dispatch            — ``test_claim_precedes_dispatch``
* ack precedes dispatch              — ``test_ack_precedes_dispatch``
* history only after successful post — ``test_failed_post_answer_keeps_entry_queued_and_skips_history``
* ONE memoized probe per dispatch    — ``test_capability_probe_runs_once_for_all_three_consumers``
* downgrade seen on NEXT dispatch    — ``test_probe_downgrade_observed_on_next_dispatch``
* no-op InputDownload is invisible   — ``test_noop_input_download_leaves_payload_byte_identical``
* duplicate fires nothing            — ``test_duplicate_delivery_returns_duplicate_and_fires_nothing``
* unparseable event never claims     — ``test_unparseable_event_is_ignored_and_never_claims``
"""

from __future__ import annotations

import base64
import copy
import logging
from typing import Any

import pytest

from osprey.bridges.core import pipeline
from osprey.bridges.core.artifacts import FetchedArtifact
from osprey.bridges.core.config import CoreConfig
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.history import HistoryStore
from osprey.bridges.core.pipeline import (
    EXPIRED_NOTE,
    OVER_BUDGET_REASON,
    SERVER_TOO_OLD_REASON,
    TOO_MANY_FILES_REASON,
    PipelineDeps,
    handle_event,
)
from osprey.bridges.core.ports import InboundEvent, InputDownload, ReplyContext
from tests.bridges.conftest import RecordingChannelOps

MSG = "spaces/AAA/messages/1"
MSG2 = "spaces/AAA/messages/2"
QUESTION = "what is the orbit doing?"
HISTORY_KEY = "spaces/AAA"

COMPLETED = {
    "status": "completed",
    "text_output": "The orbit is stable.",
    "run_id": "run-1",
    "num_tool_calls": 3,
}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def make_event(**over: Any) -> InboundEvent:
    fields: dict[str, Any] = {
        "message_id": MSG,
        "text": QUESTION,
        "sender_id": "users/7",
        "sender_display": "Pat",
        "history_key": HISTORY_KEY,
        "claim_meta": {"space": "spaces/AAA", "thread": "spaces/AAA/threads/T"},
    }
    fields.update(over)
    return InboundEvent(**fields)


def make_file(filename: str, data: bytes, mime: str = "image/png") -> dict[str, Any]:
    return {"filename": filename, "mime": mime, "content_b64": b64(data), "ingest": True}


class MarkingDedup(DedupStore):
    """Real dedup store that marks ``claim`` into the shared timeline."""

    def __init__(self, path: str, ops: RecordingChannelOps) -> None:
        super().__init__(path)
        self._ops = ops

    def claim(self, message_id: str, **meta: Any) -> bool:
        claimed = super().claim(message_id, **meta)
        self._ops.mark("claim", message_id=message_id, claimed=claimed)
        return claimed


class MarkingHistory(HistoryStore):
    """Real history store that marks reads/writes into the shared timeline.

    Note ``append_failed`` delegates to ``append`` internally, so a failed turn
    marks BOTH ``history_append_failed`` and ``history_append``.
    """

    def __init__(self, path: str, ops: RecordingChannelOps) -> None:
        super().__init__(path)
        self._ops = ops

    def recent(self, key: str) -> list[dict[str, Any]]:
        turns = super().recent(key)
        self._ops.mark("history_recent", key=key, turns=len(turns))
        return turns

    def append(
        self,
        key: str,
        question: str,
        answer: str,
        run_id: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ops.mark("history_append", key=key, question=question, answer=answer)
        super().append(key, question, answer, run_id=run_id, artifacts=artifacts)

    def append_failed(self, key: str, question: str) -> None:
        self._ops.mark("history_append_failed", key=key, question=question)
        super().append_failed(key, question)


class StubDispatcher:
    """Records each dispatch payload (deep-copied at call time) and marks the shared
    timeline; fires ``on_run_id`` before returning the canned terminal result, the
    way the real client persists the run id mid-handshake."""

    def __init__(self, ops: RecordingChannelOps, result: dict[str, Any] | None = None) -> None:
        self._ops = ops
        self.result = dict(COMPLETED if result is None else result)
        self.calls: list[dict[str, Any]] = []

    def run(self, question: str, extra: dict | None = None, *, on_run_id=None) -> dict[str, Any]:
        self.calls.append({"question": question, "extra": copy.deepcopy(extra)})
        self._ops.mark("dispatch", question=question)
        if on_run_id is not None and self.result.get("run_id"):
            on_run_id(self.result["run_id"])
        return dict(self.result)


class CountingProbe:
    """A ``probe_capability`` double: counts calls, returns scripted verdicts."""

    def __init__(self, *verdicts: bool) -> None:
        self.verdicts = list(verdicts) or [True]
        self.calls = 0

    def __call__(self, cfg: Any, capability: str) -> bool:
        assert capability == "input_files"
        self.calls += 1
        return self.verdicts[min(self.calls - 1, len(self.verdicts) - 1)]


def _prior(data: bytes, content_type: str | None = None):
    """A ``fetch_prior_artifact`` seam serving *data* under *content_type*."""
    return lambda cfg, rid, desc, mb: FetchedArtifact(data=data, content_type=content_type)


def make_deps(
    tmp_path,
    ops: RecordingChannelOps,
    *,
    result: dict[str, Any] | None = None,
    history: bool = True,
    probe=None,
    fetch_prior=None,
    park=None,
) -> PipelineDeps:
    kwargs: dict[str, Any] = {
        "cfg": CoreConfig(),
        "ops": ops,
        "dedup": MarkingDedup(str(tmp_path / "dedup.json"), ops),
        "dispatcher": StubDispatcher(ops, result=result),
        "history": MarkingHistory(str(tmp_path / "history.json"), ops) if history else None,
        "probe_capability": probe if probe is not None else CountingProbe(True),
        "fetch_prior_artifact": (
            fetch_prior if fetch_prior is not None else lambda cfg, rid, desc, mb: None
        ),
    }
    if park is not None:
        kwargs["park"] = park
    return PipelineDeps(**kwargs)


# --- ignore / duplicate -------------------------------------------------------------


def test_unparseable_event_is_ignored_and_never_claims(tmp_path):
    ops = RecordingChannelOps()  # parse_result=None -> ignore
    deps = make_deps(tmp_path, ops)

    assert handle_event({"type": "MEMBERSHIP"}, deps) == "ignored"

    assert ops.names == ["parse_event"]  # no claim, no ack, no dispatch, nothing
    assert deps.dedup.get(MSG) is None


def test_duplicate_delivery_returns_duplicate_and_fires_nothing(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops)
    assert handle_event({}, deps) == "handled"
    entry_before = deps.dedup.get(MSG)
    first_pass = len(ops.names)

    assert handle_event({}, deps) == "duplicate"

    # The redelivery paid for the parse and the losing claim, nothing else.
    assert ops.names[first_pass:] == ["parse_event", "claim"]
    assert len(deps.dispatcher.calls) == 1
    assert ops.count("post_ack") == 1
    assert ops.count("resolve_reply_context") == 1
    assert deps.dedup.get(MSG) == entry_before


# --- claim / ack / dispatch ordering ------------------------------------------------


def test_claim_precedes_dispatch(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops)

    handle_event({}, deps)

    ops.assert_never_before("dispatch", "claim")


def test_ack_precedes_dispatch(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops)

    handle_event({}, deps)

    ops.assert_never_before("post_ack", "claim")
    ops.assert_never_before("dispatch", "post_ack")


def test_ack_precedes_reply_context(tmp_path):
    # The ack exists to eliminate latency: resolve_reply_context may cost the
    # adapter network round-trips, so the user's "on it" signal must never wait
    # on it. Strict check — no resolve may occur before the first ack.
    reply = ReplyContext(reply_to={"sender": "Sam", "text": "the original question"})
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=reply)
    deps = make_deps(tmp_path, ops)

    handle_event({}, deps)

    ops.assert_never_before("resolve_reply_context", "post_ack")
    assert ops.count("post_ack") == 1
    assert ops.count("resolve_reply_context") == 1


def test_claim_persists_full_meta_before_anything_fires(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops)

    handle_event({}, deps)

    # The ack — the very first thing fired — already sees the fully persisted claim:
    # first-class fields AND the adapter's claim_meta, so a crash at any later point
    # can re-dispatch and re-address from the store alone.
    ack_entry = ops.of("post_ack")[0]["entry"]
    assert ack_entry["text"] == QUESTION
    assert ack_entry["sender_id"] == "users/7"
    assert ack_entry["sender_display"] == "Pat"
    assert ack_entry["history_key"] == HISTORY_KEY
    assert ack_entry["space"] == "spaces/AAA"
    assert ack_entry["thread"] == "spaces/AAA/threads/T"
    assert ack_entry["status"] == "pending"


# --- reply context ------------------------------------------------------------------


def test_reply_context_resolved_after_claim_and_persisted(tmp_path):
    reply = ReplyContext(
        reply_to={"sender": "Sam", "text": "the original question"},
        meta={"quoted_attachments": [{"resource_name": "r1", "content_name": "q.png"}]},
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=reply)
    deps = make_deps(tmp_path, ops)

    handle_event({}, deps)

    # After the claim (duplicates never pay for it), before the dispatch.
    ops.assert_order("claim", "post_ack", "resolve_reply_context", "dispatch")
    # Persisted for every re-dispatch path: base reply_to plus the reply meta.
    entry = deps.dedup.get(MSG)
    assert entry["reply_to"] == {"sender": "Sam", "text": "the original question"}
    assert entry["quoted_attachments"] == [{"resource_name": "r1", "content_name": "q.png"}]
    # Shipped with the dispatch.
    extra = deps.dispatcher.calls[0]["extra"]
    assert extra["reply_to"] == {"sender": "Sam", "text": "the original question"}
    # download_inputs ran off the REFRESHED entry, so it can fetch the quoted refs.
    assert ops.of("download_inputs")[0]["entry"]["quoted_attachments"] == [
        {"resource_name": "r1", "content_name": "q.png"}
    ]


# --- history in the payload ---------------------------------------------------------


def test_history_rides_the_payload(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops)
    deps.history.append(HISTORY_KEY, "prior question", "prior answer", run_id="run-0")

    handle_event({}, deps)

    turns = deps.dispatcher.calls[0]["extra"]["conversation_so_far"]
    assert [t["question"] for t in turns] == ["prior question"]
    # And the settled exchange became the next turn.
    assert [t["answer"] for t in deps.history.recent(HISTORY_KEY)] == [
        "prior answer",
        "The orbit is stable.",
    ]


def test_empty_history_key_skips_history_entirely(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event(history_key=""))
    deps = make_deps(tmp_path, ops)

    assert handle_event({}, deps) == "handled"

    assert ops.count("history_recent") == 0
    assert ops.count("history_append") == 0
    assert deps.dispatcher.calls[0]["extra"] is None


# --- terminal settle ----------------------------------------------------------------


def test_settle_order_answer_then_files_then_history(tmp_path):
    result = dict(
        COMPLETED,
        artifacts=[{"artifact_id": "art-1", "filename": "plot.png", "delivered_mime": "image/png"}],
    )
    ops = RecordingChannelOps(
        parse_result=make_event(), file_urls={"art-1": "https://pub/art-1.png"}
    )
    deps = make_deps(tmp_path, ops, result=result)

    assert handle_event({}, deps) == "handled"

    # Strict "only ever after" checks: nothing history- or file-shaped may occur
    # before the first successful post (greedy subsequence matching would miss a
    # stray earlier occurrence).
    ops.assert_never_before("history_append", "post_answer")
    ops.assert_never_before("history_append", "deliver_files")
    ops.assert_never_before("deliver_files", "post_answer")
    entry = deps.dedup.get(MSG)
    assert entry["status"] == "completed"
    assert entry["run_id"] == "run-1"
    # The deliver_files return was stamped onto this turn's OUTPUT descriptor.
    (turn,) = deps.history.recent(HISTORY_KEY)
    assert turn["artifacts"] == [
        {
            "entry_id": "art-1",
            "filename": "plot.png",
            "delivered_mime": "image/png",
            "origin": "output",
            "public_url": "https://pub/art-1.png",
        }
    ]


def test_settle_survives_a_raising_file_delivery(tmp_path):
    """``deliver_files`` never raises by contract, but the answer has already landed: a
    misbehaving adapter must not cost the terminal mark or the history turn — the same
    guarantee the drain and reconcile delivery paths give."""
    ops = RecordingChannelOps(parse_result=make_event())
    ops.fail_next("deliver_files", RuntimeError("upload exploded"))
    deps = make_deps(tmp_path, ops)

    assert handle_event({}, deps) == "handled"

    assert deps.dedup.get(MSG)["status"] == "completed"
    assert [turn["answer"] for turn in deps.history.recent(HISTORY_KEY)] == [
        COMPLETED["text_output"]
    ]


def test_failed_post_answer_keeps_entry_queued_and_skips_history(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    ops.fail_next("post_answer", RuntimeError("channel 500"))
    deps = make_deps(tmp_path, ops)

    assert handle_event({}, deps) == "handled"

    # No terminal mark, no file delivery, no history of any kind.
    assert ops.count("deliver_files") == 0
    assert ops.count("history_append") == 0
    assert ops.count("history_append_failed") == 0
    assert deps.history.recent(HISTORY_KEY) == []
    # The entry stays QUEUED with its run id intact, so the drain's re-attach path
    # re-delivers the already-computed answer next cycle instead of dropping it.
    entry = deps.dedup.get(MSG)
    assert entry["status"] == "queued"
    assert entry["run_id"] == "run-1"
    assert entry["queued_at"] > 0
    assert entry["first_queued_at"] > 0


def test_error_result_posts_wording_and_appends_failed_marker(tmp_path):
    result = {"status": "error", "failure_class": "fatal", "run_id": "run-1"}
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, result=result)

    assert handle_event({}, deps) == "handled"

    # Non-retryable: the channel posts its error wording (post_answer), the entry is
    # terminal, and the QUESTION is kept with the failed-turn marker — the raw error
    # text is never replayed into history.
    assert ops.count("post_answer") == 1
    assert ops.count("post_queued") == 0
    assert deps.dedup.get(MSG)["status"] == "error"
    assert ops.of("history_append_failed") == [{"key": HISTORY_KEY, "question": QUESTION}]


# --- retryable failures park --------------------------------------------------------

RETRYABLE = {
    "status": "error",
    "failure_class": "infrastructure",
    "num_tool_calls": 0,
    "run_id": "run-1",
}


def test_retryable_failure_parks_via_real_retry_queue(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, result=RETRYABLE)  # default park = retry_queue.park

    assert handle_event({}, deps) == "handled"

    # Parked, not settled: queued notice instead of an answer, no history.
    assert ops.count("post_answer") == 0
    assert ops.count("post_queued") == 1
    assert ops.count("history_append") == 0
    assert deps.history.recent(HISTORY_KEY) == []
    entry = deps.dedup.get(MSG)
    assert entry["status"] == "queued"
    assert entry["attempts"] == 0
    assert entry["failure_class"] == "infrastructure"
    assert entry["queued_at"] > 0


def test_injected_park_seam_receives_persisted_entry_and_result(tmp_path):
    parked: list[tuple[str, str, str]] = []

    def park(ops_, dedup_, message_id, entry, result):
        parked.append((message_id, entry["text"], result["failure_class"]))

    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, result=RETRYABLE, park=park)

    assert handle_event({}, deps) == "handled"

    assert parked == [(MSG, QUESTION, "infrastructure")]
    assert ops.count("post_answer") == 0


# --- the single memoized capability probe -------------------------------------------


def test_capability_probe_runs_once_for_all_three_consumers(tmp_path):
    # All three probe consumers in one dispatch: the message's own attachment, a
    # quoted-message attachment (both via download_inputs), and a prior-image
    # re-injection from history.
    inputs = InputDownload(files=(make_file("own.png", b"OWN"), make_file("quoted.png", b"QUOTED")))
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)
    probe = CountingProbe(True)
    deps = make_deps(tmp_path, ops, probe=probe, fetch_prior=_prior(b"PRIOR"))
    deps.history.append(
        HISTORY_KEY,
        "plot it",
        "done",
        run_id="run-0",
        artifacts=[
            {
                "entry_id": "art-0",
                "filename": "prior.png",
                "delivered_mime": "image/png",
                "origin": "output",
            }
        ],
    )

    handle_event({}, deps)

    assert probe.calls == 1
    files = deps.dispatcher.calls[0]["extra"]["input_files"]
    # Own file first, quoted after (list order from download_inputs), re-injected
    # prior last — fresh files ingest=True, replayed prior ingest=False.
    assert [f["filename"] for f in files] == ["own.png", "quoted.png", "prior.png"]
    assert [f["ingest"] for f in files] == [True, True, False]
    assert files[2]["content_b64"] == b64(b"PRIOR")


def test_probe_downgrade_observed_on_next_dispatch(tmp_path):
    inputs = InputDownload(files=(make_file("own.png", b"OWN"),))
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)
    probe = CountingProbe(True, False)
    deps = make_deps(tmp_path, ops, probe=probe, history=False)

    handle_event({}, deps)
    ops.parse_result = make_event(message_id=MSG2)
    handle_event({}, deps)

    # The memo is per-dispatch, never global: the second dispatch re-probed and
    # observed the downgrade — no bytes shipped, an honest skip note instead.
    assert probe.calls == 2
    first, second = (call["extra"] for call in deps.dispatcher.calls)
    assert [f["filename"] for f in first["input_files"]] == ["own.png"]
    assert "input_files" not in second
    assert second["skipped_attachments"] == [
        {"filename": "own.png", "reason": SERVER_TOO_OLD_REASON}
    ]


def test_probe_is_lazy_when_nothing_would_ship(tmp_path):
    # Skip notes alone ship without ever probing: there are no bytes to gate.
    inputs = InputDownload(skipped=({"filename": "big.bin", "reason": "too large"},))
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)
    probe = CountingProbe(True)
    deps = make_deps(tmp_path, ops, probe=probe, history=False)

    handle_event({}, deps)

    assert probe.calls == 0
    assert deps.dispatcher.calls[0]["extra"] == {
        "skipped_attachments": [{"filename": "big.bin", "reason": "too large"}]
    }


# --- the no-op InputDownload invariant ----------------------------------------------


def test_noop_input_download_leaves_payload_byte_identical(tmp_path):
    # A text-only message: the adapter returns the empty InputDownload(). The
    # dispatch payload must be byte-identical to the no-attachment path — the
    # dispatcher is called with NO extra at all, and the probe never fires.
    ops = RecordingChannelOps(parse_result=make_event())  # inputs = InputDownload()
    probe = CountingProbe(True)
    deps = make_deps(tmp_path, ops, probe=probe, history=False)

    assert handle_event({}, deps) == "handled"

    assert ops.count("download_inputs") == 1
    assert deps.dispatcher.calls[0] == {"question": QUESTION, "extra": None}
    assert probe.calls == 0


# --- the engine-owned fresh-file budget ---------------------------------------------


def test_fresh_budget_count_cap_in_list_order(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_FILES", 2)
    inputs = InputDownload(
        files=(make_file("f1", b"aa"), make_file("f2", b"b"), make_file("f3", b"cc"))
    )
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)
    deps = make_deps(tmp_path, ops, history=False)

    handle_event({}, deps)

    extra = deps.dispatcher.calls[0]["extra"]
    assert [f["filename"] for f in extra["input_files"]] == ["f1", "f2"]
    assert extra["skipped_attachments"] == [{"filename": "f3", "reason": TOO_MANY_FILES_REASON}]


def test_fresh_budget_bytes_cap_can_decline_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_TOTAL_BYTES", 1)
    inputs = InputDownload(files=(make_file("f1", b"too-big"),))
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)
    probe = CountingProbe(True)
    deps = make_deps(tmp_path, ops, probe=probe, history=False)

    handle_event({}, deps)

    # Everything declined -> skips-only payload, and no probe (nothing to send).
    assert probe.calls == 0
    assert deps.dispatcher.calls[0]["extra"] == {
        "skipped_attachments": [{"filename": "f1", "reason": OVER_BUDGET_REASON}]
    }


# --- quoted-bucket provenance routing -----------------------------------------------


def make_reply(**meta: Any) -> ReplyContext:
    return ReplyContext(reply_to={"sender": "Sam", "text": "the original question"}, **meta)


def test_quoted_files_deliver_into_reply_to_provenance(tmp_path):
    inputs = InputDownload(
        files=(make_file("own.png", b"OWN"),),
        quoted_files=(make_file("quoted.png", b"QUOTED"),),
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=make_reply(), inputs=inputs)
    deps = make_deps(tmp_path, ops, history=False)

    handle_event({}, deps)

    extra = deps.dispatcher.calls[0]["extra"]
    # Bytes ride input_files own-first; provenance splits: own top-level, quoted
    # inside reply_to (delivered filenames), exactly the worker payload contract.
    assert [f["filename"] for f in extra["input_files"]] == ["own.png", "quoted.png"]
    assert extra["skipped_attachments"] == []
    assert extra["reply_to"]["attachments"] == ["quoted.png"]
    assert "skipped_attachments" not in extra["reply_to"]


def test_quoted_file_trimmed_by_budget_skips_into_reply_to_bucket(tmp_path, monkeypatch):
    # The shared budget runs across files THEN quoted_files: with room for one file,
    # the own attachment wins and the quoted one is trimmed — and its skip note lands
    # in reply_to["skipped_attachments"], NOT the top-level own bucket.
    monkeypatch.setattr(pipeline, "MAX_FILES", 1)
    inputs = InputDownload(
        files=(make_file("own.png", b"OWN"),),
        quoted_files=(make_file("quoted.png", b"QUOTED"),),
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=make_reply(), inputs=inputs)
    deps = make_deps(tmp_path, ops, history=False)

    handle_event({}, deps)

    extra = deps.dispatcher.calls[0]["extra"]
    assert [f["filename"] for f in extra["input_files"]] == ["own.png"]
    assert extra["skipped_attachments"] == []
    assert extra["reply_to"]["skipped_attachments"] == [
        {"filename": "quoted.png", "reason": TOO_MANY_FILES_REASON}
    ]
    assert "attachments" not in extra["reply_to"]
    # The persisted reply_to is untouched — provenance notes ride the payload only.
    assert deps.dedup.get(MSG)["reply_to"] == {"sender": "Sam", "text": "the original question"}


def test_budget_bytes_shared_across_own_then_quoted(tmp_path, monkeypatch):
    # The byte cap is one budget over both buckets, consumed in own-then-quoted
    # order: own fits, the quoted file would bust the remainder and is trimmed into
    # its own bucket with the over-budget reason.
    monkeypatch.setattr(pipeline, "MAX_TOTAL_BYTES", 4)
    inputs = InputDownload(
        files=(make_file("own.png", b"abc"),),
        quoted_files=(make_file("quoted.png", b"de"),),
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=make_reply(), inputs=inputs)
    deps = make_deps(tmp_path, ops, history=False)

    handle_event({}, deps)

    extra = deps.dispatcher.calls[0]["extra"]
    assert [f["filename"] for f in extra["input_files"]] == ["own.png"]
    assert extra["reply_to"]["skipped_attachments"] == [
        {"filename": "quoted.png", "reason": OVER_BUDGET_REASON}
    ]


def test_non_capable_pair_routes_skips_to_matching_buckets(tmp_path):
    # Non-capable pair: no bytes ship anywhere, and each undeliverable file's
    # SERVER_TOO_OLD note joins the adapter's own skips in the MATCHING bucket.
    inputs = InputDownload(
        files=(make_file("own.png", b"OWN"),),
        skipped=({"filename": "own-huge.bin", "reason": "too large"},),
        quoted_files=(make_file("quoted.png", b"QUOTED"),),
        quoted_skipped=({"filename": "quoted-drive.doc", "reason": "drive file"},),
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=make_reply(), inputs=inputs)
    deps = make_deps(tmp_path, ops, probe=CountingProbe(False), history=False)

    handle_event({}, deps)

    extra = deps.dispatcher.calls[0]["extra"]
    assert "input_files" not in extra
    assert extra["skipped_attachments"] == [
        {"filename": "own-huge.bin", "reason": "too large"},
        {"filename": "own.png", "reason": SERVER_TOO_OLD_REASON},
    ]
    assert extra["reply_to"]["skipped_attachments"] == [
        {"filename": "quoted-drive.doc", "reason": "drive file"},
        {"filename": "quoted.png", "reason": SERVER_TOO_OLD_REASON},
    ]
    assert "attachments" not in extra["reply_to"]


def test_quoted_skips_alone_ride_reply_to_without_probing(tmp_path):
    # Skip notes alone (both buckets) ship without probing: nothing to send.
    inputs = InputDownload(
        skipped=({"filename": "own.bin", "reason": "too large"},),
        quoted_skipped=({"filename": "quoted.bin", "reason": "too large"},),
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=make_reply(), inputs=inputs)
    probe = CountingProbe(True)
    deps = make_deps(tmp_path, ops, probe=probe, history=False)

    handle_event({}, deps)

    assert probe.calls == 0
    extra = deps.dispatcher.calls[0]["extra"]
    assert extra["skipped_attachments"] == [{"filename": "own.bin", "reason": "too large"}]
    assert extra["reply_to"]["skipped_attachments"] == [
        {"filename": "quoted.bin", "reason": "too large"}
    ]


def test_quoted_bucket_without_reply_context_is_dropped_with_warning(tmp_path, caplog):
    # Quoted files with no resolved reply are an adapter contract violation: there is
    # no reply_to to fold their provenance into, so the bucket is dropped loudly and
    # the dispatch proceeds.
    inputs = InputDownload(quoted_files=(make_file("quoted.png", b"QUOTED"),))
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)  # no reply_context
    deps = make_deps(tmp_path, ops, history=False)

    with caplog.at_level(logging.WARNING, logger="osprey.bridges.core.pipeline"):
        assert handle_event({}, deps) == "handled"

    assert deps.dispatcher.calls[0]["extra"] is None
    assert any("quoted" in rec.message for rec in caplog.records)


# --- reserved-key collision detection -----------------------------------------------


def test_claim_meta_reserved_key_collision_fails_before_anything_fires(tmp_path):
    # A Talk adapter stamping claim_meta["reply_to"] would have it clobbered by the
    # engine's persisted reply context — the collision must fail fast, before the
    # claim, so nothing is half-processed.
    event = make_event(claim_meta={"space": "spaces/AAA", "reply_to": {"sender": "x"}})
    ops = RecordingChannelOps(parse_result=event)
    deps = make_deps(tmp_path, ops)

    with pytest.raises(ValueError, match="reply_to"):
        handle_event({}, deps)

    assert ops.names == ["parse_event"]  # no claim, no ack, no dispatch
    assert deps.dedup.get(MSG) is None


def test_reply_meta_reserved_key_collision_is_stripped_loudly(tmp_path, caplog):
    # The claim has already committed when reply meta arrives, so a colliding key
    # cannot fail the message — it is stripped (the drain's bookkeeping stays
    # uncorrupted) and logged as an error, while legitimate meta persists.
    reply = ReplyContext(
        reply_to={"sender": "Sam", "text": "the original question"},
        meta={"attempts": 99, "quoted_attachments": [{"resource_name": "r1"}]},
    )
    ops = RecordingChannelOps(parse_result=make_event(), reply_context=reply)
    deps = make_deps(tmp_path, ops)

    with caplog.at_level(logging.ERROR, logger="osprey.bridges.core.pipeline"):
        assert handle_event({}, deps) == "handled"

    entry = deps.dedup.get(MSG)
    assert "attempts" not in entry  # the drain give-up ceiling field survived intact
    assert entry["quoted_attachments"] == [{"resource_name": "r1"}]
    assert any("attempts" in rec.getMessage() for rec in caplog.records)


# --- prior-image re-injection edge cases --------------------------------------------


def _seed_prior_image(deps: PipelineDeps) -> None:
    deps.history.append(
        HISTORY_KEY,
        "plot it",
        "done",
        run_id="run-0",
        artifacts=[
            {
                "entry_id": "art-0",
                "filename": "prior.png",
                "delivered_mime": "image/png",
                "origin": "output",
            }
        ],
    )


def test_unfetchable_prior_flagged_expired_without_store_pollution(tmp_path):
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, fetch_prior=lambda cfg, rid, d, mb: None)
    _seed_prior_image(deps)

    assert handle_event({}, deps) == "handled"

    extra = deps.dispatcher.calls[0]["extra"]
    assert "input_files" not in extra
    # The outgoing payload's descriptor carries the note...
    (turn,) = extra["conversation_so_far"]
    assert turn["artifacts"][0]["note"] == EXPIRED_NOTE
    # ...but the persisted history copy was never mutated (the settled exchange
    # appended a second turn behind it; the seeded turn is the first).
    stored = HistoryStore(str(tmp_path / "history.json")).recent(HISTORY_KEY)[0]
    assert "note" not in stored["artifacts"][0]


def test_reinjected_prior_dedupes_against_own_attachment(tmp_path):
    # The user re-attached the very plot they are asking about: the prior fetch
    # returns identical bytes, which must not ship twice.
    inputs = InputDownload(files=(make_file("own-copy.png", b"SAME-BYTES"),))
    ops = RecordingChannelOps(parse_result=make_event(), inputs=inputs)
    deps = make_deps(tmp_path, ops, fetch_prior=_prior(b"SAME-BYTES"))
    _seed_prior_image(deps)

    handle_event({}, deps)

    files = deps.dispatcher.calls[0]["extra"]["input_files"]
    assert [f["filename"] for f in files] == ["own-copy.png"]


def test_a_prior_whose_bytes_are_not_an_image_is_not_reinjected(tmp_path):
    # osprey #503: the descriptor predicted image/png, but the render failed and
    # the byte route served the original CSV. Shipping those bytes labelled
    # image/png would hand the model a file that is not the image it is told to
    # look at; the descriptor still rides conversation_so_far either way.
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, fetch_prior=_prior(b"x,sin_x\n0,0\n", "text/csv"))
    _seed_prior_image(deps)

    assert handle_event({}, deps) == "handled"

    extra = deps.dispatcher.calls[0]["extra"]
    assert "input_files" not in extra
    (turn,) = extra["conversation_so_far"]
    assert turn["artifacts"][0]["delivered_mime"] == "image/png"


def test_a_reinjected_prior_is_labelled_with_the_served_type(tmp_path):
    # The prediction said PNG, the byte route served JPEG: the model is told what
    # it actually got.
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, fetch_prior=_prior(b"JPEG-BYTES", "image/jpeg"))
    _seed_prior_image(deps)

    handle_event({}, deps)

    (file,) = deps.dispatcher.calls[0]["extra"]["input_files"]
    assert file["mime"] == "image/jpeg"


def test_a_prior_served_without_a_content_type_keeps_the_predicted_mime(tmp_path):
    # An older worker sends no Content-Type; the prediction is all there is.
    ops = RecordingChannelOps(parse_result=make_event())
    deps = make_deps(tmp_path, ops, fetch_prior=_prior(b"PRIOR"))
    _seed_prior_image(deps)

    handle_event({}, deps)

    (file,) = deps.dispatcher.calls[0]["extra"]["input_files"]
    assert file["mime"] == "image/png"
