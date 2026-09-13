"""The assembled class against the ``ChannelOps`` seam: does the engine's contract hold?

Each member has its own suite (parsing, posting, delivery). What is asserted here is
everything that only becomes true once the ten sit on ONE class the engine holds one
instance of:

* **conformance** — every member present, callable, and signature-compatible. The runtime
  ``isinstance`` check looks at member NAMES only; the full signature check is
  ``ops._static_conformance``, verified by the repo's mypy gate rather than by an
  assertion here, and re-checked structurally below.
* **the two delegating members are wired at all.** ``parse_event`` and
  ``resolve_reply_context`` live in :mod:`~osprey.bridges.teams.events`; the engine calls
  the class and never the module function, so a missing delegation would leave the adapter
  silently ignoring every message.
* **no per-dispatch instance state.** The engine shares ONE instance across the Service
  Bus receive threads and the retry-drain thread. This adapter holds nothing but its
  injected collaborators — not even the one documented stash its Chat sibling keeps — and
  that emptiness is asserted rather than assumed.
* **entry-key disjointness.** ``claim_meta`` lands in the same flat persisted entry as the
  engine's bookkeeping, so a collision is a silent overwrite in one direction or the
  other; the ``ms_`` prefix is the whole mechanism that prevents it.
* **reader/writer agreement.** The outbound members address the Bot Connector from keys
  the parse member wrote, across two modules; a rename on one side alone would deliver
  nothing and raise nothing, so the entry driving the posts here is the one a real parse
  produced rather than a hand-written dict.

Hermetic: the Connector leg is the posting suite's recording stand-in, so there is no
tenant, no bearer and no network in this file.
"""

from __future__ import annotations

import inspect
from typing import Any

from osprey.bridges.core import (
    RESERVED_ENTRY_KEYS,
    ChannelOps,
    InboundEvent,
    InputDownload,
    ReplyContext,
)
from osprey.bridges.teams import events
from osprey.bridges.teams.events import (
    CHANNEL_CONVERSATION,
    MS_ACTIVITY_ID,
    MS_CONVERSATION_ID,
    MS_CONVERSATION_TYPE,
    MS_SERVICE_URL,
    MS_TENANT_ID,
    PERSONAL_CONVERSATION,
)
from osprey.bridges.teams.ops import TeamsOps, _static_conformance, quote_prefix
from tests.bridges.teams.test_posting import (
    ACTIVITY_ID,
    CHANNEL_CONVERSATION_ID,
    QUESTION,
    SENDER_ID,
    SERVICE_URL,
    RecordingConnector,
    activity,
    chat_activity,
    make_config,
    make_ops,
)
from tests.bridges.test_ports import PROTOCOL_MEMBERS

# Every attribute a constructed instance is allowed to hold. Anything else appearing here
# is per-dispatch state on an instance several threads share — the failure this set exists
# to make loud, since the corruption it causes is intermittent and load-dependent.
COLLABORATORS = {"_cfg", "_client", "_http", "_fetch_artifact"}


# --- builders ---------------------------------------------------------------


def parsed(ops: TeamsOps, wire: dict[str, Any] | None = None) -> tuple[InboundEvent, ReplyContext]:
    """One activity through both inbound members, asserted non-``None`` so the callers
    below read the real thing rather than a skipped test."""
    event = ops.parse_event(activity() if wire is None else wire)
    assert event is not None
    context = ops.resolve_reply_context(event)
    assert context is not None
    return event, context


def persisted(ops: TeamsOps, wire: dict[str, Any] | None = None) -> dict[str, Any]:
    """The flat entry the engine would persist for that activity — the engine's own claim
    fields, the adapter's claim meta and the reply meta merged exactly as the pipeline
    merges them."""
    event, context = parsed(ops, wire)
    return {
        "message_id": event.message_id,
        "text": event.text,
        "sender_id": event.sender_id,
        "sender_display": event.sender_display,
        "history_key": event.history_key,
        **event.claim_meta,
        **context.meta,
    }


def completed() -> dict[str, Any]:
    """A terminal result with no artifacts: the delivery member's bytes path is the
    delivery suite's business, and reaching for it here would want a worker."""
    return {"status": "completed", "text_output": "42 mA", "run_id": "R1", "artifacts": []}


# ==========================================================================
# Protocol conformance
# ==========================================================================


def test_the_adapter_satisfies_the_channel_ops_protocol() -> None:
    # ChannelOps is @runtime_checkable, so this is a real assertion and not a tautology
    # about the class body — but it verifies member NAMES only. Signatures are checked
    # statically by ops._static_conformance under mypy, and structurally below.
    ops, _ = make_ops()
    assert isinstance(ops, ChannelOps)


def test_every_protocol_member_is_present_and_callable() -> None:
    # The canonical member list is imported from the core seam's own suite rather than
    # respelled, so a member added to the protocol fails here instead of being forgotten.
    ops, _ = make_ops()
    for name in PROTOCOL_MEMBERS:
        assert callable(getattr(ops, name, None)), name


def test_every_member_signature_matches_the_protocol() -> None:
    """Parameter names, order, and count, member by member.

    ``isinstance`` cannot see any of this, and a mismatch is not a type error the engine
    would notice until it called the member in production — the retry drain's members in
    particular may not run for hours.
    """
    for name in PROTOCOL_MEMBERS:
        expected = inspect.signature(getattr(ChannelOps, name))
        actual = inspect.signature(getattr(TeamsOps, name))
        assert list(actual.parameters) == list(expected.parameters), name


def test_static_conformance_helper_is_the_type_check_seam() -> None:
    """Present, and an identity — the assertion mypy makes is the point of it.

    Guards against the helper being deleted as apparently-unused code: without it nothing
    would check the full signatures at all, since the runtime protocol check does not.
    """
    ops, _ = make_ops()
    assert _static_conformance(ops) is ops


def test_parsed_types_are_the_engines_own() -> None:
    """Not duck-typed lookalikes: the engine unpacks these dataclasses by field."""
    ops, _ = make_ops()
    event, context = parsed(ops)

    assert isinstance(event, InboundEvent)
    assert isinstance(context, ReplyContext)
    assert isinstance(ops.download_inputs(dict(event.claim_meta)), InputDownload)


# ==========================================================================
# Delegation: the class adds arity, not a second implementation
# ==========================================================================


def test_parse_event_delegates_with_the_instances_own_config() -> None:
    """The seam passes one argument and the module function takes two, so the config is
    the class's whole contribution — and it must be the instance's, since the app id it
    carries is what decides whether a channel message was addressed to this bot at all."""
    cfg = make_config()
    ops = TeamsOps(cfg, RecordingConnector())
    wire = activity()

    assert ops.parse_event(wire) == events.parse_event(wire, cfg)


def test_resolve_reply_context_delegates_to_the_events_module() -> None:
    """Same shape with nothing added: Teams delivers no inbound quote to this bot, and the
    answer lives in ``events`` so that module's suite can pin it without an adapter."""
    ops, _ = make_ops()
    event, context = parsed(ops)

    assert context == events.resolve_reply_context(event)


# ==========================================================================
# The two members with no platform leg at all
# ==========================================================================


def test_download_inputs_is_the_no_op_for_every_entry() -> None:
    """This deployment asks for no attachment-reading consent, so the claim stamped no
    file references and the dispatch payload must stay byte-identical to the text-only
    path — which is what the bare ``InputDownload()`` guarantees."""
    ops, connector = make_ops()

    assert ops.download_inputs(persisted(ops)) == InputDownload()
    # An entry from an older bridge, or one stripped to nothing, answers the same way
    # rather than raising into a dispatch that was about to succeed.
    assert ops.download_inputs({}) == InputDownload()
    # And no leg was taken to find that out: the member must cost the drain nothing.
    assert connector.calls == []


def test_coalesce_key_groups_by_conversation_and_sender() -> None:
    """One person's repeated asking in one conversation collapses; two people's questions
    never do."""
    ops, _ = make_ops()
    entry = persisted(ops)

    assert ops.coalesce_key(entry) == [entry["history_key"], entry["sender_id"]]
    # A different sender in the same conversation keys differently, which is what stops
    # the drain from superseding someone else's question.
    assert ops.coalesce_key({**entry, "sender_id": "29:someone-else"}) != ops.coalesce_key(entry)


def test_coalesce_key_stands_alone_when_either_half_is_missing() -> None:
    """``()`` is the engine's "never coalesce" sentinel, and the fail-closed direction: a
    wrong group silently discards a question, a missed group costs a duplicate run."""
    ops, _ = make_ops()

    assert ops.coalesce_key({"sender_id": SENDER_ID}) == ()
    assert ops.coalesce_key({"history_key": CHANNEL_CONVERSATION_ID}) == ()
    assert ops.coalesce_key({"history_key": "", "sender_id": SENDER_ID}) == ()
    assert ops.coalesce_key({"history_key": CHANNEL_CONVERSATION_ID, "sender_id": 7}) == ()


# ==========================================================================
# One shared instance: no per-dispatch state
# ==========================================================================


def test_the_instance_holds_nothing_but_its_collaborators() -> None:
    """The engine shares ONE instance across the receive threads and the drain thread.

    So the instance dict must hold nothing but the injected collaborators: any
    per-dispatch field (a "current conversation", a cached entry) would be written by one
    thread and read by another, and the corruption would be intermittent and
    load-dependent. Asserted structurally, since a race cannot be asserted directly.

    The held objects are themselves safe to share: ``TeamsBridgeConfig`` is frozen, the
    connector client serializes its own HTTP leg, ``httpx.Client`` is safe to share, and
    the fetcher is a pure function of its arguments.
    """
    ops, _ = make_ops()
    assert set(vars(ops)) == COLLABORATORS


def test_a_full_dispatch_adds_no_instance_state_and_rebinds_nothing() -> None:
    """Every member the live path and the drain call, in order, against one instance.

    The attribute set must come out identical and no attribute may be *replaced* — unlike
    its Chat sibling this adapter carries no cross-call map at all, so nothing here is
    even allowed to change.
    """
    ops, connector = make_ops()
    entry = persisted(ops)
    before = dict(vars(ops))

    ops.post_ack(entry)
    ops.resolve_reply_context(parsed(ops)[0])
    ops.download_inputs(entry)
    ops.post_answer(entry, completed())
    assert ops.deliver_files(entry, completed()) == {}
    ops.coalesce_key(entry)
    ops.post_queued(entry, {"status": "failed"})
    ops.post_giveup(entry)
    ops.post_superseded(entry)

    assert set(vars(ops)) == COLLABORATORS
    assert all(vars(ops)[name] is value for name, value in before.items())
    # The dispatch really did reach the Connector, so the emptiness above is the
    # instance's and not a sign that every member returned early.
    assert len(connector.calls) == 5


# ==========================================================================
# Entry keys: the adapter's namespace against the engine's
# ==========================================================================


def test_the_adapter_writes_no_key_the_engine_owns() -> None:
    """The adapter's namespace lands in the SAME flat persisted entry as the engine's
    bookkeeping, so a collision silently overwrites one side or the other — a stamped
    ``attempts`` would corrupt the drain's give-up ceiling, a persisted ``reply_to`` would
    clobber the engine's."""
    ops, _ = make_ops()
    event, context = parsed(ops)

    claim_keys = set(event.claim_meta)
    meta_keys = set(context.meta)

    assert claim_keys.isdisjoint(RESERVED_ENTRY_KEYS)
    assert meta_keys.isdisjoint(RESERVED_ENTRY_KEYS)
    # The two adapter namespaces must not collide with each other either: they are
    # written by different members and merged into one dict.
    assert claim_keys.isdisjoint(meta_keys)
    # The ``ms_`` prefix is the whole mechanism that keeps all three true.
    assert all(key.startswith("ms_") for key in claim_keys | meta_keys)


def test_the_reply_context_contributes_no_entry_keys_at_all() -> None:
    """Teams resolves no quote, so the persisted entry is the claim's keys and nothing
    else — and a future quote source adding meta here would have to clear the
    disjointness above rather than inherit it by being empty."""
    ops, _ = make_ops()
    event, context = parsed(ops)

    assert context.meta == {}
    assert set(persisted(ops)) == {
        "message_id",
        "text",
        "sender_id",
        "sender_display",
        "history_key",
    } | set(event.claim_meta)


def test_every_declared_entry_key_is_one_the_parse_member_writes() -> None:
    """Both directions, against the constants themselves: a key declared and never written
    is one the outbound members would read as absent forever, and a key written without a
    constant is one the reader cannot import and will eventually respell."""
    ops, _ = make_ops()
    event, _ = parsed(ops)

    declared = {
        value
        for name, value in vars(events).items()
        if name.startswith("MS_") and isinstance(value, str)
    }

    assert set(event.claim_meta) == declared


# ==========================================================================
# Reader/writer agreement across the two modules
# ==========================================================================


def test_the_keys_the_outbound_members_read_are_the_keys_the_parse_member_writes() -> None:
    """End to end, across modules.

    ``ops`` addresses every post from the service url, the conversation id and the
    activity id off the persisted entry, and ``events`` writes all three; a rename on one
    side alone would deliver nothing and raise nothing, so the entry driving the post here
    is the one a real parse produced.
    """
    ops, connector = make_ops()
    entry = persisted(ops)

    assert {MS_SERVICE_URL, MS_CONVERSATION_ID, MS_ACTIVITY_ID, MS_TENANT_ID} <= set(entry)

    ops.post_ack(entry)

    service_url, conversation_id, activity_id, _ = connector.calls[0]
    # The conversation id is the AS-SENT one the activity carried, not the normalized
    # history key: posting to the normalized form would address the thread root instead.
    assert (service_url, conversation_id, activity_id) == (
        SERVICE_URL,
        CHANNEL_CONVERSATION_ID,
        ACTIVITY_ID,
    )


def test_the_quote_is_driven_by_the_conversation_type_the_parse_member_wrote() -> None:
    """``quote_prefix`` reads ``ms_conversation_type`` and the engine's own ``text``; both
    are written elsewhere, so the two shapes are asserted from real parses rather than
    from a hand-built entry that could agree with neither."""
    ops, connector = make_ops()
    channel_entry = persisted(ops)
    chat_entry = persisted(ops, chat_activity())

    assert channel_entry[MS_CONVERSATION_TYPE] == CHANNEL_CONVERSATION
    assert chat_entry[MS_CONVERSATION_TYPE] == PERSONAL_CONVERSATION

    ops.post_ack(channel_entry)
    ops.post_ack(chat_entry)

    # Threaded under the question in a channel, so it quotes nothing; a flat chat has no
    # threads, so the reply names what it answers.
    assert quote_prefix(channel_entry) == ""
    assert not connector.texts[0].startswith(">")
    assert connector.texts[1].startswith(quote_prefix(chat_entry))
    assert connector.texts[1].startswith(f"> {QUESTION}\n\n")
