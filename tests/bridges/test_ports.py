"""The ``ChannelOps`` seam: protocol conformance of the recording double, and the
recording/ordering ergonomics downstream suites (drain, pipeline, retry_queue,
reconcile) assert against."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from osprey.bridges.core.ports import ChannelOps, InboundEvent, InputDownload, ReplyContext
from tests.bridges.conftest import RecordedCall, RecordingChannelOps

# Every protocol member, in the canonical live-path call order (where one exists):
# claim -> ack -> reply context -> dispatch. The ack is the FIRST thing after a won
# claim — resolve_reply_context may cost the adapter network round-trips and must
# never delay the user's "on it" signal.
PROTOCOL_MEMBERS = [
    "parse_event",
    "post_ack",
    "resolve_reply_context",
    "download_inputs",
    "post_answer",
    "deliver_files",
    "post_queued",
    "post_giveup",
    "post_superseded",
    "coalesce_key",
]


def _event(**overrides: Any) -> InboundEvent:
    fields: dict[str, Any] = {
        "message_id": "room1:42",
        "text": "what is the orbit doing?",
        "sender_id": "users/7",
        "sender_display": "Alice",
        "history_key": "spaces/AAA",
    }
    fields.update(overrides)
    return InboundEvent(**fields)


# --- protocol conformance -----------------------------------------------------


def test_double_satisfies_channel_ops_protocol():
    assert isinstance(RecordingChannelOps(), ChannelOps)


def test_every_protocol_member_is_callable_on_the_double():
    ops = RecordingChannelOps()
    for name in PROTOCOL_MEMBERS:
        assert callable(getattr(ops, name)), name


def test_protocol_is_structural_not_inheritance():
    """A from-scratch class with the right members conforms — adapters never need to
    subclass anything from the core package."""

    class MinimalOps:
        def parse_event(self, event: Any) -> InboundEvent | None:
            return None

        def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
            return None

        def post_ack(self, entry: Mapping[str, Any]) -> None:
            pass

        def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
            return InputDownload()

        def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
            pass

        def deliver_files(
            self, entry: Mapping[str, Any], result: Mapping[str, Any]
        ) -> Mapping[str, str]:
            return {}

        def post_queued(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
            pass

        def post_giveup(self, entry: Mapping[str, Any]) -> None:
            pass

        def post_superseded(self, entry: Mapping[str, Any]) -> None:
            pass

        def coalesce_key(self, entry: Mapping[str, Any]) -> str | Sequence[str]:
            return ""

    assert isinstance(MinimalOps(), ChannelOps)


def test_incomplete_class_does_not_satisfy_protocol():
    class NotOps:
        def parse_event(self, event: Any) -> None:
            return None

    assert not isinstance(NotOps(), ChannelOps)


# --- parsed-event / payload types ---------------------------------------------


def test_inbound_event_is_frozen_with_neutral_defaults():
    event = _event()
    assert event.claim_meta == {}
    assert event.raw is None
    with pytest.raises(AttributeError):
        event.text = "mutated"  # type: ignore[misc]


def test_inbound_event_carries_claim_meta_and_raw_opaquely():
    wire = {"chat": {"messagePayload": {}}}
    event = _event(claim_meta={"space": "spaces/AAA", "thread": "threads/T"}, raw=wire)
    assert event.claim_meta["thread"] == "threads/T"
    assert event.raw is wire


def test_input_download_default_is_the_no_op():
    empty = InputDownload()
    assert empty.files == () and empty.skipped == ()
    # The quoted bucket defaults empty too, so a channel with no quote mechanic
    # (email, Talk without ``parent``) never touches the new fields.
    assert empty.quoted_files == () and empty.quoted_skipped == ()


def test_reply_context_defaults_to_empty_meta():
    ctx = ReplyContext(reply_to={"sender": "Osprey", "text": "the orbit is stable"})
    assert ctx.meta == {}


# --- recording ----------------------------------------------------------------


def test_records_protocol_calls_in_order_with_arguments():
    ops = RecordingChannelOps()
    entry = {"space": "spaces/AAA", "text": "q"}
    result = {"status": "completed", "text_output": "a"}

    ops.post_ack(entry)
    ops.post_answer(entry, result)
    ops.deliver_files(entry, result)

    assert ops.names == ["post_ack", "post_answer", "deliver_files"]
    assert ops.of("post_answer") == [{"entry": entry, "result": result}]
    assert ops.calls[0] == RecordedCall("post_ack", {"entry": entry})


def test_mark_interleaves_engine_events_into_the_timeline():
    """Engine-side steps (claim, dispatch, history) marked by store/dispatcher stubs
    land in the same timeline, so cross-boundary ordering is one assertion."""
    ops = RecordingChannelOps()
    entry = {"text": "q"}

    ops.mark("claim", message_id="m1")
    ops.post_ack(entry)
    ops.mark("dispatch", message_id="m1")
    ops.post_answer(entry, {"status": "completed"})
    ops.mark("history_append", key="spaces/AAA")

    ops.assert_order("claim", "post_ack", "dispatch")
    ops.assert_order("post_answer", "history_append")
    assert ops.of("claim") == [{"message_id": "m1"}]


def test_assert_order_fails_on_violation_and_on_missing_op():
    ops = RecordingChannelOps()
    ops.post_answer({}, {})
    ops.post_ack({})

    with pytest.raises(AssertionError, match="post_answer"):
        ops.assert_order("post_ack", "post_answer")
    with pytest.raises(AssertionError, match="post_giveup"):
        ops.assert_order("post_giveup")


def test_assert_order_matches_repeated_ops_as_a_subsequence():
    """A failed first attempt does not shadow the retry: greedy matching walks past
    it, so 'answer posted, then history' holds across at-least-once redelivery."""
    ops = RecordingChannelOps()
    ops.post_answer({}, {})  # attempt 1 (failed at the transport, say)
    ops.post_answer({}, {})  # retry succeeds
    ops.mark("history_append")

    ops.assert_order("post_answer", "post_answer", "history_append")
    with pytest.raises(AssertionError):
        ops.assert_order("post_answer", "post_answer", "post_answer")


def test_assert_never_before_catches_what_greedy_subsequence_misses():
    """The timeline ['history_append', 'post_answer', 'history_append'] — an engine
    bug appending history BEFORE any successful post — passes assert_order (greedy
    subsequence matching walks past the stray first append) but must fail the strict
    never-before check. This is the invariant the settle phase exists to protect."""
    ops = RecordingChannelOps()
    ops.mark("history_append")
    ops.post_answer({}, {})
    ops.mark("history_append")

    ops.assert_order("post_answer", "history_append")  # the false pass, demonstrated
    with pytest.raises(AssertionError, match="before the first"):
        ops.assert_never_before("history_append", "post_answer")


def test_assert_never_before_passes_clean_timelines_and_requires_the_anchor():
    ops = RecordingChannelOps()
    ops.post_answer({}, {})
    ops.mark("history_append")

    ops.assert_never_before("history_append", "post_answer")
    # An op that never occurs at all trivially satisfies "never before".
    ops.assert_never_before("post_giveup", "post_answer")
    # But the anchor MUST occur — an invariant about an event that never happened
    # would silently assert nothing.
    with pytest.raises(AssertionError, match="deliver_files"):
        ops.assert_never_before("history_append", "deliver_files")


def test_count_and_of_track_repeat_calls():
    ops = RecordingChannelOps()
    ops.post_superseded({"space": "s1"})
    ops.post_superseded({"space": "s2"})
    assert ops.count("post_superseded") == 2
    assert [d["entry"]["space"] for d in ops.of("post_superseded")] == ["s1", "s2"]


# --- canned returns & failure injection ---------------------------------------


def test_configured_returns_flow_through():
    event = _event()
    ctx = ReplyContext(reply_to={"sender": "Bob", "text": "earlier"})
    inputs = InputDownload(files=({"filename": "a.png", "content_b64": "aGk="},))
    ops = RecordingChannelOps(
        parse_result=event,
        reply_context=ctx,
        inputs=inputs,
        file_urls={"art-1": "https://files.example/a.png"},
        coalesce=["spaces/AAA", "users/7"],
    )

    assert ops.parse_event({"raw": True}) is event
    assert ops.resolve_reply_context(event) is ctx
    assert ops.download_inputs({"attachments": []}) is inputs
    assert ops.deliver_files({}, {}) == {"art-1": "https://files.example/a.png"}
    assert ops.coalesce_key({}) == ["spaces/AAA", "users/7"]


def test_default_double_ignores_events_and_never_coalesces():
    ops = RecordingChannelOps()
    assert ops.parse_event({"anything": 1}) is None
    assert ops.resolve_reply_context(_event()) is None
    assert ops.download_inputs({}) == InputDownload()
    assert ops.deliver_files({}, {}) == {}
    assert ops.coalesce_key({}) == ""


# --- callable canned returns ---------------------------------------------------
# Every knob also takes a callable over the member's own argument. Without it a
# single double answers identically for every call, which is fine for one-message
# tests and useless for the multi-room ones: two wire events would parse to the same
# ``message_id``, dedup would collapse them into one claim, and "room B dispatches
# while room A is in flight" could not be expressed at all.


def test_callable_parse_result_gives_each_event_its_own_message_id():
    ops = RecordingChannelOps(
        parse_result=lambda raw: _event(
            message_id=f"{raw['room']}:{raw['id']}", history_key=raw["room"]
        ),
    )

    first = ops.parse_event({"room": "roomA", "id": 1})
    second = ops.parse_event({"room": "roomB", "id": 2})

    assert first is not None and second is not None
    # Distinct ids from one instance: the property dedup needs to see two messages.
    assert (first.message_id, second.message_id) == ("roomA:1", "roomB:2")
    assert first.history_key != second.history_key
    # A callable knob changes what is returned, never what is recorded.
    assert [call.details["event"]["room"] for call in ops.calls] == ["roomA", "roomB"]


def test_callable_coalesce_varies_per_entry():
    ops = RecordingChannelOps(coalesce=lambda entry: [entry["history_key"], entry["sender_id"]])

    assert ops.coalesce_key({"history_key": "roomA", "sender_id": "users/1"}) == [
        "roomA",
        "users/1",
    ]
    assert ops.coalesce_key({"history_key": "roomB", "sender_id": "users/2"}) == [
        "roomB",
        "users/2",
    ]


def test_callable_reply_context_inputs_and_file_urls_answer_per_argument():
    ops = RecordingChannelOps(
        reply_context=lambda event: ReplyContext(reply_to={"text": event.text}),
        inputs=lambda entry: InputDownload(files=({"filename": f"{entry['room']}.png"},)),
        file_urls=lambda entry: {entry["room"]: f"https://files.example/{entry['room']}.png"},
    )

    ctx = ops.resolve_reply_context(_event(text="earlier"))
    assert ctx is not None and ctx.reply_to == {"text": "earlier"}
    assert ops.download_inputs({"room": "roomA"}).files[0]["filename"] == "roomA.png"
    # ``file_urls`` is the one asymmetry: its callable takes the ENTRY only, not the
    # run result, so a double cannot derive urls from a run's artifact list.
    assert ops.deliver_files({"room": "roomB"}, {"status": "completed"}) == {
        "roomB": "https://files.example/roomB.png"
    }


def test_callable_knob_is_resolved_at_call_time_not_construction():
    """A knob may be reassigned mid-test, and a callable sees state changed after
    construction — the shape a poller test uses to flip one room's behaviour partway
    through a run."""
    rooms = ["roomA"]
    ops = RecordingChannelOps(coalesce=lambda entry: rooms[-1])

    assert ops.coalesce_key({}) == "roomA"
    rooms.append("roomB")
    assert ops.coalesce_key({}) == "roomB"

    ops.parse_result = lambda raw: _event(message_id=raw["id"])
    parsed = ops.parse_event({"id": "roomC:9"})
    assert parsed is not None and parsed.message_id == "roomC:9"


def test_deliver_files_copies_a_callables_result():
    """Same guarantee the constant form has always given: a caller mutating the
    returned mapping cannot reach back into whatever the knob handed over."""
    shared = {"art-1": "https://files.example/a.png"}
    ops = RecordingChannelOps(file_urls=lambda entry: shared)

    returned = ops.deliver_files({}, {})
    returned["art-2"] = "https://files.example/b.png"

    assert shared == {"art-1": "https://files.example/a.png"}


def test_fail_next_records_the_call_then_raises_then_recovers():
    """Drain semantics in miniature: the failed post is visible in the timeline, and
    the NEXT call succeeds — 'first deliver fails, second cycle re-delivers'."""
    ops = RecordingChannelOps()
    ops.fail_next("post_answer", RuntimeError("chat 502"))

    with pytest.raises(RuntimeError, match="chat 502"):
        ops.post_answer({"space": "s"}, {"status": "completed"})
    ops.post_answer({"space": "s"}, {"status": "completed"})  # recovered

    assert ops.count("post_answer") == 2


def test_fail_next_queues_fifo_per_op():
    ops = RecordingChannelOps()
    ops.fail_next("post_giveup", RuntimeError("first"))
    ops.fail_next("post_giveup", RuntimeError("second"))

    with pytest.raises(RuntimeError, match="first"):
        ops.post_giveup({})
    with pytest.raises(RuntimeError, match="second"):
        ops.post_giveup({})
    ops.post_giveup({})
    assert ops.count("post_giveup") == 3


def test_channel_ops_fixture_provides_a_fresh_default_double(channel_ops):
    assert isinstance(channel_ops, RecordingChannelOps)
    assert channel_ops.calls == []
