"""The assembled SlackOps against the ``ChannelOps`` seam.

Each member has its own suite. What is asserted here only becomes true once they sit
on ONE class the engine holds one instance of: conformance, the parse delegation, no
per-dispatch instance state, entry-key disjointness, and that the keys the outbound
members read are the keys a real parse wrote.
"""

from __future__ import annotations

from typing import Any

from osprey.bridges.core import RESERVED_ENTRY_KEYS, ChannelOps, InboundEvent, InputDownload
from osprey.bridges.slack import events
from osprey.bridges.slack.events import SL_PARENT_FILES
from osprey.bridges.slack.ops import SlackOps, _static_conformance, quote_prefix
from tests.bridges.slack._posting import (
    CHAN,
    CTX,
    DM,
    QUESTION,
    TS,
    RecordingClient,
    dm_payload,
    make_config,
    make_ops,
    mention_payload,
)
from tests.bridges.test_ports import PROTOCOL_MEMBERS

# Every attribute a constructed instance is allowed to hold. Anything else appearing here
# is per-dispatch state on an instance several threads share.
COLLABORATORS = {"_cfg", "_client", "_ctx"}
# The channel roster: built once at construction, owns its lock and its bound, and is
# never rebound — the same shared collaborator the Teams, Chat and Nextcloud ops hold.
ROSTER = {"_roster"}


def parsed(ops: SlackOps, wire: dict[str, Any] | None = None) -> InboundEvent:
    event = ops.parse_event(mention_payload() if wire is None else wire)
    assert event is not None
    return event


def persisted(ops: SlackOps, wire: dict[str, Any] | None = None) -> dict[str, Any]:
    """The flat entry the engine would persist for that payload."""
    event = parsed(ops, wire)
    return {
        "message_id": event.message_id,
        "text": event.text,
        "sender_id": event.sender_id,
        "sender_display": event.sender_display,
        "history_key": event.history_key,
        **event.claim_meta,
    }


def completed() -> dict[str, Any]:
    return {"status": "completed", "text_output": "42 mA", "run_id": "R1", "artifacts": []}


# --- protocol conformance ------------------------------------------------------


def test_the_adapter_satisfies_the_channel_ops_protocol() -> None:
    ops, _ = make_ops()
    assert isinstance(ops, ChannelOps)


def test_every_protocol_member_is_present_and_callable() -> None:
    ops, _ = make_ops()
    for name in PROTOCOL_MEMBERS:
        assert callable(getattr(ops, name, None)), name


def test_static_conformance_helper_is_the_type_check_seam() -> None:
    ops, _ = make_ops()
    assert _static_conformance(ops) is ops


def test_parse_event_delegates_with_the_instances_own_identity() -> None:
    ops = SlackOps(make_config(), RecordingClient(), CTX)  # type: ignore[arg-type]
    wire = mention_payload()
    assert ops.parse_event(wire) == events.parse_event(wire, CTX)
    assert isinstance(ops.parse_event(wire), InboundEvent)


def test_parse_event_never_raises_on_garbage() -> None:
    ops, _ = make_ops()
    for wire in (None, 7, "x", {}, {"event": "nope"}):
        assert ops.parse_event(wire) is None


def test_the_unfilled_seams_are_inert() -> None:
    ops, client = make_ops()
    entry = persisted(ops)
    assert ops.resolve_reply_context(parsed(ops)) is None
    assert ops.download_inputs(entry) == InputDownload()
    assert ops.deliver_files(entry, completed()) == {}
    assert client.calls == []


# --- one shared instance ---------------------------------------------------------


def test_the_instance_holds_its_collaborators_and_the_roster_only() -> None:
    ops, _ = make_ops()
    assert set(vars(ops)) == COLLABORATORS | ROSTER


def test_a_full_dispatch_adds_no_instance_state_and_rebinds_nothing() -> None:
    ops, client = make_ops()
    entry = persisted(ops)
    before = dict(vars(ops))

    ops.post_ack(entry)
    ops.resolve_reply_context(parsed(ops))
    ops.download_inputs(entry)
    ops.post_answer(entry, completed())
    ops.deliver_files(entry, completed())
    ops.coalesce_key(entry)
    ops.post_queued(entry, {"status": "failed"})
    ops.post_resumed(entry, {"status": "completed"})
    ops.post_giveup(entry)
    ops.post_superseded(entry)

    assert set(vars(ops)) == COLLABORATORS | ROSTER
    assert all(vars(ops)[name] is value for name, value in before.items())
    assert len(client.calls) == 6


# --- entry keys --------------------------------------------------------------------


def test_the_adapter_writes_no_key_the_engine_owns() -> None:
    ops, _ = make_ops()
    keys = set(parsed(ops).claim_meta)
    assert keys.isdisjoint(RESERVED_ENTRY_KEYS)
    assert all(key.startswith("sl_") for key in keys)


def test_every_declared_claim_key_is_one_the_parse_member_writes() -> None:
    """``SL_PARENT_FILES`` is reply-context meta, written by the reply lookup, not
    the claim."""
    ops, _ = make_ops()
    declared = {
        value
        for name, value in vars(events).items()
        if name.startswith("SL_") and isinstance(value, str)
    } - {SL_PARENT_FILES}
    assert set(parsed(ops).claim_meta) == declared
    assert set(parsed(ops, dm_payload()).claim_meta) == declared


# --- reader/writer agreement ---------------------------------------------------------


def test_a_parsed_channel_question_is_answered_in_its_thread() -> None:
    ops, client = make_ops()
    ops.post_ack(persisted(ops))
    channel, text, thread_ts = client.calls[0]
    assert (channel, thread_ts) == (CHAN, TS)
    assert not text.startswith(">")


def test_a_parsed_top_level_dm_is_answered_top_level_with_a_quote() -> None:
    ops, client = make_ops()
    entry = persisted(ops, dm_payload())
    ops.post_answer(entry, completed())
    assert client.calls == [(DM, f"> {QUESTION}\n\n42 mA", None)]
    assert quote_prefix(entry) == f"> {QUESTION}\n\n"


def test_a_parsed_threaded_dm_is_answered_in_its_thread() -> None:
    ops, client = make_ops()
    entry = persisted(ops, dm_payload(thread_ts="1699999999.000001"))
    ops.post_ack(entry)
    assert client.calls[0][0] == DM
    assert client.calls[0][2] == "1699999999.000001"
    assert not client.calls[0][1].startswith(">")


def test_coalesce_key_reads_the_parsed_history_key() -> None:
    ops, _ = make_ops()
    entry = persisted(ops)
    assert ops.coalesce_key(entry) == [f"{CHAN}:{TS}", entry["sender_id"]]
