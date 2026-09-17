"""Tests for what the Teams adapter actually posts into a conversation.

Every test drives :class:`~osprey.bridges.teams.ops.TeamsOps` over a recording
stand-in for the Bot Connector, so the suite needs no tenant, no bearer and no
network: what is asserted is the activity body the adapter handed the connector,
compared against the module's own exported wording.

Comparing by equality against :func:`~osprey.bridges.teams.ops.ack_text` and
:func:`~osprey.bridges.teams.ops.quote_prefix` rather than against a string
spelled here is deliberate. A test that re-spells the ack proves only that
someone typed it twice; a test that composes it from the exported helpers pins
the thing that matters — that the helper the e2e lane and the operator-facing
wording agree on is the one the posting path actually used.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.bridges.core import InboundEvent, InputDownload, ReplyContext
from osprey.bridges.core.text import chunk_text
from osprey.bridges.teams.client import ConnectorError, MessageSizeTooBig, TokenError
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.events import (
    CHANNEL_CONVERSATION,
    MESSAGE_ACTIVITY_TYPE,
    MS_ACTIVITY_ID,
    MS_CONVERSATION_ID,
    MS_CONVERSATION_TYPE,
    MS_SERVICE_URL,
    MS_TENANT_ID,
    PERSONAL_CONVERSATION,
    bot_actor_id,
)
from osprey.bridges.teams.formatting import markdown_to_teams
from osprey.bridges.teams.ops import (
    ACK_TEXT,
    ANSWER_CHUNK_CHARS,
    EMPTY_ANSWER_TEXT,
    ERROR_TEXT,
    GIVEUP_TEXT,
    QUEUED_TEXT,
    QUOTE_LINE_CHARS,
    SUPERSEDED_TEXT,
    TeamsOps,
    ack_text,
    quote_prefix,
)

APP_ID = "11111111-2222-3333-4444-555555555555"
TENANT = "66666666-7777-8888-9999-000000000000"
SERVICE_URL = "https://smba.trafficmanager.net/amer/"
CHANNEL_CONVERSATION_ID = "19:room@thread.tacv2;messageid=1700000000000"
CHAT_CONVERSATION_ID = "19:chat@thread.v2"
ACTIVITY_ID = "1700000000123"
QUESTION = "What is the stored beam current right now?"
VERSION = "2026.9.0"
SENDER_ID = "29:a-stable-user-id"
SENDER_NAME = "Ada Lovelace"
MENTION_SPAN = "<at>Osprey</at>"


def make_config(version_tag: str = VERSION) -> TeamsBridgeConfig:
    """A config carrying only what the posting members read."""
    return TeamsBridgeConfig(
        app_id=APP_ID,
        app_secret="a-client-secret",
        tenant_id=TENANT,
        version_tag=version_tag,
    )


def make_entry(
    conversation_type: str = "channel",
    *,
    text: str = QUESTION,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """A persisted entry as the claim stamped it, for one conversation type.

    Shaped like the real thing — the engine's own claim fields beside the
    adapter's ``ms_`` namespace — because every posting member reads the entry
    and nothing else, and a test that fed it a hand-trimmed dict would not
    exercise that.
    """
    if conversation_id is None:
        conversation_id = (
            CHANNEL_CONVERSATION_ID if conversation_type == "channel" else CHAT_CONVERSATION_ID
        )
    return {
        "text": text,
        "sender_id": SENDER_ID,
        "sender_display": SENDER_NAME,
        "history_key": conversation_id,
        MS_SERVICE_URL: SERVICE_URL,
        MS_CONVERSATION_ID: conversation_id,
        MS_ACTIVITY_ID: ACTIVITY_ID,
        MS_CONVERSATION_TYPE: conversation_type,
        MS_TENANT_ID: TENANT,
    }


def mention() -> dict[str, Any]:
    """A ``mention`` entity naming this bot — the access control for a channel."""
    return {
        "type": "mention",
        "text": MENTION_SPAN,
        "mentioned": {"id": bot_actor_id(APP_ID), "name": "Osprey"},
    }


def activity(**overrides: Any) -> dict[str, Any]:
    """A channel question that mentions the bot, as the relay enqueued it.

    Deliberately the richest activity this adapter accepts, so one parse writes every
    ``ms_`` key there is. The inbound counterpart of :func:`make_entry`: what the
    claim stamps from this activity is the entry that function spells.
    """
    wire: dict[str, Any] = {
        "type": MESSAGE_ACTIVITY_TYPE,
        "id": ACTIVITY_ID,
        "serviceUrl": SERVICE_URL,
        "from": {"id": SENDER_ID, "name": SENDER_NAME},
        "conversation": {
            "id": CHANNEL_CONVERSATION_ID,
            "conversationType": CHANNEL_CONVERSATION,
            "tenantId": TENANT,
        },
        "text": f"{MENTION_SPAN} {QUESTION}",
        "entities": [mention()],
    }
    wire.update(overrides)
    return wire


def chat_activity() -> dict[str, Any]:
    """The same question in a 1:1 chat: no thread, no mention, no mention entity."""
    return activity(
        conversation={
            "id": CHAT_CONVERSATION_ID,
            "conversationType": PERSONAL_CONVERSATION,
            "tenantId": TENANT,
        },
        text=QUESTION,
        entities=[],
    )


class RecordingConnector:
    """A stand-in for :class:`~osprey.bridges.teams.client.ConnectorClient`.

    Records every ``reply`` and can be told to fail the next one, which is how
    the swallow contracts are proven without a transport at all.
    """

    def __init__(self, fail_with: BaseException | None = None) -> None:
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []
        self.fail_with = fail_with

    def reply(
        self,
        service_url: str,
        conversation_id: str,
        activity_id: str,
        activity: dict[str, Any],
    ) -> None:
        self.calls.append((service_url, conversation_id, activity_id, activity))
        if self.fail_with is not None:
            raise self.fail_with

    @property
    def texts(self) -> list[str]:
        """The ``text`` of every activity posted, in order."""
        return [activity.get("text", "") for _, _, _, activity in self.calls]


class SizeRejectingConnector(RecordingConnector):
    """A connector that answers 413 for chosen replies, counted from one.

    Modelled on the only 413 that can actually happen: the Connector measures the
    serialized activity in bytes while the adapter splits in characters, so ONE
    message of a long answer can be refused while its neighbours land. Rejecting
    by position is what lets a test say "the second chunk was too big" without
    constructing 100 KB of multi-byte text.
    """

    def __init__(self, reject: set[int]) -> None:
        super().__init__()
        self.reject = set(reject)
        self.landed: list[str] = []

    def reply(
        self,
        service_url: str,
        conversation_id: str,
        activity_id: str,
        activity: dict[str, Any],
    ) -> None:
        super().reply(service_url, conversation_id, activity_id, activity)
        if len(self.calls) in self.reject:
            raise MessageSizeTooBig("connector rejected the activity as too large (HTTP 413)")
        self.landed.append(activity.get("text", ""))


def long_answer(lines: int = 500, width: int = 99) -> str:
    """An answer long enough to be split, built from lines of known length.

    Lines rather than one unbroken run because that is what a real answer is, and
    because it makes every split land on a newline — so the chunk boundaries are
    the ones the chunker prefers rather than the degenerate hard-cut path, which
    :mod:`tests.bridges.test_text` already pins.

    Every line is numbered, which is load-bearing rather than decorative: with
    identical filler each piece of the same length is the same string, and a test
    asking "was this chunk posted exactly once?" would silently be counting some
    other piece that happened to match.
    """
    return "\n".join(f"{index:04d}{'x' * (width - 4)}" for index in range(lines))


def expected_chunks(text: str) -> list[str]:
    """The chunks the answer path must produce for ``text``.

    Composed from the same two exported pieces the adapter uses rather than
    spelled out here: a test that re-implemented the split would pass while the
    adapter split differently, which is the one failure that matters.
    """
    return chunk_text(markdown_to_teams(text), ANSWER_CHUNK_CHARS)


def make_ops(
    connector: RecordingConnector | None = None,
    *,
    version_tag: str = VERSION,
) -> tuple[TeamsOps, RecordingConnector]:
    """A ``TeamsOps`` wired to a recording connector, and that connector."""
    connector = connector if connector is not None else RecordingConnector()
    return TeamsOps(make_config(version_tag), connector), connector


# --- the ack wording --------------------------------------------------------


def test_ack_text_appends_the_version_parenthetical() -> None:
    """A tagged deployment names its version, two spaces after the sentence."""
    assert ack_text(VERSION) == f"{ACK_TEXT}  (OSPREY {VERSION})"


def test_ack_text_without_a_tag_carries_no_parenthetical() -> None:
    """A source checkout resolves no tag, and must not post an empty ``()``."""
    assert ack_text("") == ACK_TEXT
    assert "(" not in ack_text("")


# --- the quote prefix -------------------------------------------------------


def test_quote_prefix_is_empty_in_a_channel() -> None:
    """A channel reply is threaded under the question, so it needs no quote."""
    assert quote_prefix(make_entry("channel")) == ""


def test_quote_prefix_blockquotes_the_first_line_in_a_chat() -> None:
    """A flat chat reply opens by naming the question it answers."""
    entry = make_entry("personal", text=f"{QUESTION}\nand also the lifetime?")
    assert quote_prefix(entry) == f"> {QUESTION}\n\n"


def test_quote_prefix_truncates_a_long_question_with_an_ellipsis() -> None:
    """The quote is a pointer at the question, never a second copy of it."""
    question = "b" * (QUOTE_LINE_CHARS + 40)
    prefix = quote_prefix(make_entry("groupChat", text=question))
    assert prefix == f"> {'b' * QUOTE_LINE_CHARS}…\n\n"


def test_quote_prefix_is_empty_when_the_entry_carries_no_question() -> None:
    """An empty quote beats a blockquote of nothing above the ack."""
    assert quote_prefix(make_entry("personal", text="   ")) == ""
    entry = make_entry("personal")
    del entry["text"]
    assert quote_prefix(entry) == ""


# --- post_ack ---------------------------------------------------------------


def test_post_ack_in_a_channel_posts_exactly_the_ack_text() -> None:
    """The threaded channel ack is the wording and nothing else."""
    ops, connector = make_ops()
    ops.post_ack(make_entry("channel"))
    assert connector.texts == [ack_text(VERSION)]


def test_post_ack_addresses_the_activity_the_entry_names() -> None:
    """Replies go back to the host, conversation and activity as they arrived."""
    ops, connector = make_ops()
    ops.post_ack(make_entry("channel"))
    service_url, conversation_id, activity_id, activity = connector.calls[0]
    assert (service_url, conversation_id, activity_id) == (
        SERVICE_URL,
        CHANNEL_CONVERSATION_ID,
        ACTIVITY_ID,
    )
    assert activity["type"] == "message"


def test_post_ack_in_a_chat_opens_with_the_blockquote() -> None:
    """A 1:1 reply is flat, so it carries the question it is answering."""
    ops, connector = make_ops()
    entry = make_entry("personal")
    ops.post_ack(entry)
    assert connector.texts == [quote_prefix(entry) + ack_text(VERSION)]
    assert connector.texts[0].startswith("> ")


def test_post_ack_in_a_group_chat_opens_with_the_blockquote() -> None:
    """Group chats are flat for the same reason 1:1 chats are."""
    ops, connector = make_ops()
    entry = make_entry("groupChat")
    ops.post_ack(entry)
    assert connector.texts == [quote_prefix(entry) + ack_text(VERSION)]


def test_post_ack_without_a_version_tag_omits_the_parenthetical() -> None:
    """The untagged ack is posted verbatim, not with an empty ``(OSPREY )``."""
    ops, connector = make_ops(version_tag="")
    ops.post_ack(make_entry("channel"))
    assert connector.texts == [ACK_TEXT]


@pytest.mark.parametrize(
    "failure",
    [ConnectorError("connector refused"), TokenError("no bearer"), RuntimeError("boom")],
    ids=["connector", "token", "unexpected"],
)
def test_post_ack_swallows_every_transport_failure(failure: BaseException) -> None:
    """A lost courtesy message must never cost the user their answer."""
    ops, connector = make_ops(RecordingConnector(fail_with=failure))
    ops.post_ack(make_entry("channel"))
    assert len(connector.calls) == 1


def test_post_ack_swallows_an_entry_that_names_no_destination() -> None:
    """A malformed entry is a bridge bug to log, not a dispatch to abort."""
    ops, connector = make_ops()
    entry = make_entry("channel")
    del entry[MS_SERVICE_URL]
    ops.post_ack(entry)
    assert connector.calls == []


# --- post_answer: the answer itself -----------------------------------------


def test_post_answer_posts_the_answer_rewritten_for_teams() -> None:
    """Teams renders no headings, so the answer arrives already degraded to bold."""
    ops, connector = make_ops()
    ops.post_answer(
        make_entry("channel"),
        {"status": "completed", "text_output": "# Beam current\n\n500 mA"},
    )
    assert connector.texts == ["**Beam current**\n\n500 mA"]


def test_post_answer_leaves_the_callers_result_untouched() -> None:
    """The rewrite is a presentation concern and must not follow the text into history."""
    ops, _ = make_ops()
    result = {"status": "completed", "text_output": "# Beam current"}
    ops.post_answer(make_entry("channel"), result)
    assert result["text_output"] == "# Beam current"


def test_post_answer_in_a_channel_carries_no_quote() -> None:
    """The answer is threaded under the question, which already names itself."""
    ops, connector = make_ops()
    ops.post_answer(make_entry("channel"), {"status": "completed", "text_output": "500 mA"})
    assert connector.texts == ["500 mA"]


@pytest.mark.parametrize("conversation_type", ["personal", "groupChat"], ids=["dm", "group"])
def test_post_answer_in_a_flat_chat_opens_with_the_quote(conversation_type: str) -> None:
    """A flat transcript needs the answer to say which question it answers."""
    ops, connector = make_ops()
    entry = make_entry(conversation_type)
    ops.post_answer(entry, {"status": "completed", "text_output": "500 mA"})
    assert connector.texts == [f"{quote_prefix(entry)}500 mA"]


def test_post_answer_posts_every_chunk_in_order() -> None:
    """A long answer is split at the activity ceiling and arrives whole, in order."""
    ops, connector = make_ops()
    answer = long_answer()
    chunks = expected_chunks(answer)
    assert len(chunks) == 3
    ops.post_answer(make_entry("channel"), {"status": "completed", "text_output": answer})
    assert connector.texts == chunks


def test_post_answer_quotes_only_the_first_chunk_of_a_long_answer() -> None:
    """Repeating the quote on every chunk would bury the answer in its own question."""
    ops, connector = make_ops()
    entry = make_entry("personal")
    answer = long_answer()
    ops.post_answer(entry, {"status": "completed", "text_output": answer})
    assert connector.texts[0].startswith(quote_prefix(entry))
    assert not any(text.startswith("> ") for text in connector.texts[1:])


# --- post_answer: the 413 re-split ------------------------------------------


def test_post_answer_resplits_only_the_chunk_the_connector_refused() -> None:
    """Three chunks with a 413 on the second land as four posts.

    The refused chunk is halved and both halves are posted; the chunks around it
    are posted exactly once each, because re-posting a chunk that already landed
    duplicates it in the conversation and re-sizing the later ones would punish
    the whole answer for one over-large message.
    """
    connector = SizeRejectingConnector({2})
    ops, _ = make_ops(connector)
    answer = long_answer()
    chunks = expected_chunks(answer)
    ops.post_answer(make_entry("channel"), {"status": "completed", "text_output": answer})
    halves = chunk_text(chunks[1], ANSWER_CHUNK_CHARS // 2)
    assert len(halves) == 2
    assert connector.landed == [chunks[0], *halves, chunks[2]]
    assert connector.landed.count(chunks[0]) == 1
    assert connector.landed.count(chunks[2]) == 1


def test_post_answer_raises_when_a_half_chunk_is_still_refused() -> None:
    """Two halvings mean the cap is not what this path assumes — hand it to the engine."""
    connector = SizeRejectingConnector({1, 2})
    ops, _ = make_ops(connector)
    with pytest.raises(MessageSizeTooBig):
        ops.post_answer(
            make_entry("channel"),
            {"status": "completed", "text_output": long_answer()},
        )
    assert connector.landed == []


# --- post_answer: the non-answers -------------------------------------------


def test_post_answer_posts_fixed_wording_for_a_failed_run() -> None:
    """A failure is reported in the adapter's own words, never the run's."""
    ops, connector = make_ops()
    ops.post_answer(
        make_entry("channel"),
        {
            "status": "error",
            "error": "Traceback: KeyError('token=s3cr3t') at frame 3",
            "run_id": "run-7f3c9a",
            "text_output": "half an answer",
        },
    )
    assert connector.texts == [ERROR_TEXT]


@pytest.mark.parametrize("status", ["error", "timeout", "cancelled"], ids=str)
def test_post_answer_never_leaks_the_machine_side_detail(status: str) -> None:
    """Every non-completed status lands on the same wording and leaks nothing."""
    ops, connector = make_ops()
    ops.post_answer(
        make_entry("personal"),
        {"status": status, "error": "s3cr3t-detail", "run_id": "run-7f3c9a"},
    )
    posted = connector.texts[0]
    assert "s3cr3t-detail" not in posted
    assert "run-7f3c9a" not in posted
    assert posted.endswith(ERROR_TEXT)


@pytest.mark.parametrize(
    "result",
    [
        {"status": "completed"},
        {"status": "completed", "text_output": ""},
        {"status": "completed", "text_output": "   \n\n"},
        {"status": "completed", "text_output": None},
    ],
    ids=["absent", "empty", "blank", "not-a-string"],
)
def test_post_answer_posts_a_placeholder_for_a_silent_run(result: dict[str, Any]) -> None:
    """A successful-but-silent run must not be indistinguishable from a dead bridge."""
    ops, connector = make_ops()
    ops.post_answer(make_entry("channel"), result)
    assert connector.texts == [EMPTY_ANSWER_TEXT]


# --- post_answer: the failure contract --------------------------------------


@pytest.mark.parametrize(
    "failure",
    [ConnectorError("connector refused"), TokenError("no bearer")],
    ids=["connector", "token"],
)
def test_post_answer_raises_on_a_transport_failure(failure: BaseException) -> None:
    """The engine re-queues on the raise; swallowing would drop the user's answer."""
    ops, _ = make_ops(RecordingConnector(fail_with=failure))
    with pytest.raises(type(failure)):
        ops.post_answer(make_entry("channel"), {"status": "completed", "text_output": "500 mA"})


def test_post_answer_raises_when_the_entry_names_no_destination() -> None:
    """There is no weaker place to post an answer, so a malformed entry is fatal here."""
    ops, connector = make_ops()
    entry = make_entry("channel")
    del entry[MS_CONVERSATION_ID]
    with pytest.raises(ValueError):
        ops.post_answer(entry, {"status": "completed", "text_output": "500 mA"})
    assert connector.calls == []


# --- post_queued ------------------------------------------------------------


def test_post_queued_posts_the_park_notice() -> None:
    """The user is told to wait rather than to re-send."""
    ops, connector = make_ops()
    ops.post_queued(make_entry("channel"), {"status": "error"})
    assert connector.texts == [QUEUED_TEXT]


def test_post_queued_in_a_chat_opens_with_the_quote() -> None:
    """A flat transcript needs the notice to name the question it parks."""
    ops, connector = make_ops()
    entry = make_entry("personal")
    ops.post_queued(entry, {"status": "error"})
    assert connector.texts == [f"{quote_prefix(entry)}{QUEUED_TEXT}"]


def test_post_queued_never_names_the_failure_that_parked_it() -> None:
    """What the user can do about a park is the same whatever caused it."""
    ops, connector = make_ops()
    ops.post_queued(
        make_entry("channel"),
        {"status": "error", "error": "s3cr3t-detail", "run_id": "run-7f3c9a"},
    )
    assert connector.texts == [QUEUED_TEXT]


def test_post_queued_may_raise_on_a_transport_failure() -> None:
    """The engine parks BEFORE calling this and logs the raise, so raising is safe."""
    ops, _ = make_ops(RecordingConnector(fail_with=ConnectorError("connector refused")))
    with pytest.raises(ConnectorError):
        ops.post_queued(make_entry("channel"), {"status": "error"})


# --- post_giveup ------------------------------------------------------------


def test_post_giveup_posts_the_abandonment_notice() -> None:
    """A user who is never told their question was abandoned waits forever."""
    ops, connector = make_ops()
    ops.post_giveup(make_entry("channel"))
    assert connector.texts == [GIVEUP_TEXT]


def test_post_giveup_in_a_chat_opens_with_the_quote() -> None:
    """The note names the question it abandons, since nothing else in a chat does."""
    ops, connector = make_ops()
    entry = make_entry("groupChat")
    ops.post_giveup(entry)
    assert connector.texts == [f"{quote_prefix(entry)}{GIVEUP_TEXT}"]


def test_post_giveup_never_names_the_give_up_reason() -> None:
    """The machine-side reason stays in store meta, even when the entry carries it."""
    ops, connector = make_ops()
    entry = make_entry("channel")
    entry["give_up_reason"] = "dispatch unavailable for 6 attempts"
    ops.post_giveup(entry)
    assert connector.texts == [GIVEUP_TEXT]


@pytest.mark.parametrize(
    "failure",
    [ConnectorError("connector refused"), TokenError("no bearer")],
    ids=["connector", "token"],
)
def test_post_giveup_raises_on_a_transport_failure(failure: BaseException) -> None:
    """A raise leaves the entry queued so the notice is attempted again next cycle."""
    ops, _ = make_ops(RecordingConnector(fail_with=failure))
    with pytest.raises(type(failure)):
        ops.post_giveup(make_entry("channel"))


# --- post_superseded --------------------------------------------------------


def test_post_superseded_posts_the_superseded_note() -> None:
    """The dropped question is accounted for rather than silently abandoned."""
    ops, connector = make_ops()
    ops.post_superseded(make_entry("channel"))
    assert connector.texts == [SUPERSEDED_TEXT]


def test_post_superseded_in_a_chat_quotes_the_question_it_dropped() -> None:
    """In a flat chat the quote is the only thing saying WHICH question was dropped."""
    ops, connector = make_ops()
    entry = make_entry("personal")
    ops.post_superseded(entry)
    assert connector.texts == [f"{quote_prefix(entry)}{SUPERSEDED_TEXT}"]


@pytest.mark.parametrize(
    "failure",
    [ConnectorError("connector refused"), TokenError("no bearer"), RuntimeError("boom")],
    ids=["connector", "token", "unexpected"],
)
def test_post_superseded_swallows_every_transport_failure(failure: BaseException) -> None:
    """The coalesce CAS has already committed; a lost note must not disturb the pass."""
    ops, connector = make_ops(RecordingConnector(fail_with=failure))
    ops.post_superseded(make_entry("channel"))
    assert len(connector.calls) == 1


def test_post_superseded_swallows_an_entry_that_names_no_destination() -> None:
    """A malformed entry is a bridge bug to log, not a coalesce pass to abort."""
    ops, connector = make_ops()
    entry = make_entry("channel")
    del entry[MS_ACTIVITY_ID]
    ops.post_superseded(entry)
    assert connector.calls == []


# --- the members that touch no wire -----------------------------------------


def test_download_inputs_is_the_empty_download() -> None:
    """No RSC permission means no attachment refs, so there is nothing to fetch."""
    ops, connector = make_ops()
    assert ops.download_inputs(make_entry("channel")) == InputDownload()
    assert connector.calls == []


def test_download_inputs_never_raises_on_a_degenerate_entry() -> None:
    """It reads nothing, so no entry shape can make it fail."""
    ops, _ = make_ops()
    assert ops.download_inputs({}) == InputDownload()


def test_resolve_reply_context_is_the_empty_context() -> None:
    """Teams delivers no inbound quote to this bot, so there is nothing to resolve."""
    ops, connector = make_ops()
    event = InboundEvent(
        message_id=f"{CHAT_CONVERSATION_ID}:{ACTIVITY_ID}",
        text=QUESTION,
        sender_id="29:a-stable-user-id",
        sender_display="Ada Lovelace",
        history_key=CHAT_CONVERSATION_ID,
    )
    assert ops.resolve_reply_context(event) == ReplyContext(reply_to={}, meta={})
    assert connector.calls == []


def test_coalesce_key_groups_by_conversation_and_sender() -> None:
    """One person's repeated asking in one conversation collapses; two people's never do."""
    ops, _ = make_ops()
    entry = make_entry("channel")
    assert ops.coalesce_key(entry) == [CHANNEL_CONVERSATION_ID, "29:a-stable-user-id"]


def test_coalesce_key_of_two_senders_in_one_conversation_differs() -> None:
    """Collapsing two people's questions into one would silently discard somebody's."""
    ops, _ = make_ops()
    mine = make_entry("channel")
    theirs = make_entry("channel")
    theirs["sender_id"] = "29:another-user-id"
    assert ops.coalesce_key(mine) != ops.coalesce_key(theirs)


@pytest.mark.parametrize("field", ["history_key", "sender_id"], ids=str)
@pytest.mark.parametrize("value", [None, "", 17], ids=["absent", "blank", "not-a-string"])
def test_coalesce_key_stands_alone_when_a_component_is_missing(field: str, value: Any) -> None:
    """Standing alone costs a duplicate run; a wrong group costs someone their answer."""
    ops, _ = make_ops()
    entry = make_entry("channel")
    if value is None:
        del entry[field]
    else:
        entry[field] = value
    assert ops.coalesce_key(entry) == ()
