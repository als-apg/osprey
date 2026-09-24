"""parse_event: which Teams activities become questions, and what rides the claim.

Hermetic — the module under test imports nothing but the standard library and the
engine's port types, so there is no transport to fake.

The properties worth stating plainly, because each is a silent failure in
production if it breaks:

* the **channel mention filter** — it is the access control for a room, so a
  mention of a colleague must not summon the agent and neither must the literal
  text ``@Osprey``, while a 1:1 chat must work with no mention at all;
* **other mentions survive as names** — "ask <at>Alice</at>" reaching the agent as
  "ask" is a question with its subject deleted;
* the **thread-root normalization** — Teams gives a root post the bare channel id
  and its replies the ``;messageid=`` form, so a parser that keys on the id as
  sent starts a fresh transcript on every turn of the same thread;
* the **as-sent conversation id on the claim** — outbound posts address it, and
  persisting the normalized key instead would reply to the thread root;
* the **loop guard** — the bot answering its own acks is an unbounded loop in a
  live channel;
* **claim_meta completeness and namespace** — the posting members run off the
  persisted entry long after the activity is gone, and an ``ms_`` key colliding
  with the engine's own is a silent overwrite;
* **never raising** — this runs before the dedup claim, so an exception leaves
  the Service Bus message unsettled and redelivered until its lock expires.

The reply-context cells live here too: :func:`resolve_reply_context` is the other
half of this module's inbound seam, and its whole contract is that it answers the
same empty context for every entry without reading anything.
"""

import json

import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS, ReplyContext
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.events import (
    MS_ACTIVITY_ID,
    MS_CONVERSATION_ID,
    MS_CONVERSATION_TYPE,
    MS_SERVICE_URL,
    MS_TENANT_ID,
    bot_actor_id,
    parse_event,
    people_seen,
    resolve_reply_context,
    roster_conversation_id,
)

APP_ID = "11111111-2222-3333-4444-555555555555"
BOT_ID = f"28:{APP_ID}"
CFG = TeamsBridgeConfig(app_id=APP_ID)

CHANNEL_ID = "19:abcdef@thread.tacv2"
ROOT_ID = "1700000000000"
REPLY_ID = "1700000000999"
CHAT_ID = "a:1a2b3c"
SERVICE_URL = "https://smba.trafficmanager.net/emea/"
TENANT = "99999999-8888-7777-6666-555555555555"
USER_ID = "29:alice-aad-id"


def _mention(actor_id=BOT_ID, name="Osprey", span=None):
    """A ``mention`` entity, shaped as the Bot Connector sends one."""
    entity = {
        "type": "mention",
        "mentioned": {"id": actor_id, "name": name},
    }
    if span is not None:
        entity["text"] = span
    elif name:
        entity["text"] = f"<at>{name}</at>"
    return entity


def _activity(**overrides):
    """A plausible activity: a channel root post that @mentions the bot."""
    activity = {
        "type": "message",
        "id": ROOT_ID,
        "serviceUrl": SERVICE_URL,
        "from": {"id": USER_ID, "name": "Alice Example"},
        "conversation": {
            "id": CHANNEL_ID,
            "conversationType": "channel",
            "tenantId": TENANT,
        },
        "text": "<at>Osprey</at> what is the beam current?",
        "entities": [_mention()],
    }
    activity.update(overrides)
    return activity


def _reply_activity(**overrides):
    """A reply posted into the thread the default root post started."""
    return _activity(
        id=REPLY_ID,
        conversation={
            "id": f"{CHANNEL_ID};messageid={ROOT_ID}",
            "conversationType": "channel",
            "tenantId": TENANT,
        },
        text="<at>Osprey</at> and the lifetime?",
        **overrides,
    )


def _chat_activity(**overrides):
    """A 1:1 chat message, which carries no mention entity at all."""
    return _activity(
        id=REPLY_ID,
        conversation={"id": CHAT_ID, "conversationType": "personal", "tenantId": TENANT},
        text="what is the beam current?",
        entities=[],
        **overrides,
    )


# --- the mention filter ----------------------------------------------------


def test_a_mentioned_channel_message_is_a_question():
    parsed = parse_event(_activity(), CFG)

    assert parsed is not None
    assert parsed.text == "what is the beam current?"
    assert parsed.sender_id == USER_ID
    assert parsed.sender_display == "Alice Example"


def test_an_unmentioned_channel_message_is_ignored():
    activity = _activity(text="what is the beam current?", entities=[])

    assert parse_event(activity, CFG) is None


def test_the_literal_mention_text_without_an_entity_is_not_addressing():
    # The access control is the entity, not the characters: anyone can type the
    # bot's name, and a channel where that summons a run has no filter at all.
    activity = _activity(text="@Osprey what is the beam current?", entities=[])

    assert parse_event(activity, CFG) is None


def test_a_mention_of_somebody_else_does_not_summon_the_bot():
    activity = _activity(
        text="<at>Alice</at> what is the beam current?",
        entities=[_mention(actor_id="29:alice", name="Alice")],
    )

    assert parse_event(activity, CFG) is None


def test_a_group_chat_still_requires_the_mention():
    conversation = {"id": CHAT_ID, "conversationType": "groupChat", "tenantId": TENANT}

    assert parse_event(_activity(conversation=conversation, entities=[]), CFG) is None
    assert parse_event(_activity(conversation=conversation), CFG) is not None


def test_a_personal_chat_needs_no_mention():
    parsed = parse_event(_chat_activity(), CFG)

    assert parsed is not None
    assert parsed.text == "what is the beam current?"


def test_a_missing_conversation_type_still_requires_the_mention():
    # Fail closed: an unrecognizable conversation is treated as a room, not as a
    # 1:1 chat where everything is a question.
    conversation = {"id": CHAT_ID, "tenantId": TENANT}

    assert parse_event(_activity(conversation=conversation, entities=[]), CFG) is None


def test_a_bridge_with_no_app_id_answers_nobody():
    # "28:" matches no mentioned id, so an unconfigured bridge is silent rather
    # than answering every message in every channel it is installed in.
    assert parse_event(_activity(), TeamsBridgeConfig(app_id="")) is None


def test_the_mention_entity_type_is_matched_case_insensitively():
    entity = dict(_mention(), type="Mention")

    assert parse_event(_activity(entities=[entity]), CFG) is not None


def test_a_lone_entity_object_is_read_like_a_single_entry_list():
    assert parse_event(_activity(entities=_mention()), CFG) is not None


def test_a_mention_without_a_text_span_still_addresses_the_bot():
    # Nothing can be sliced out, but the message IS for the bot; dropping it
    # would lose a genuine question.
    parsed = parse_event(
        _activity(text="what is the beam current?", entities=[_mention(span="")]), CFG
    )

    assert parsed is not None
    assert parsed.text == "what is the beam current?"


def test_bot_actor_id_prefixes_a_configured_id_and_nothing_else():
    assert bot_actor_id(APP_ID) == BOT_ID
    assert bot_actor_id("") == ""


# --- text rewriting --------------------------------------------------------


def test_two_mentions_keep_the_colleague_and_lose_only_the_bot():
    activity = _activity(
        text="<at>Osprey</at> ask <at>Alice</at> about the magnets",
        entities=[_mention(), _mention(actor_id="29:alice", name="Alice")],
    )

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.text == "ask Alice about the magnets"


def test_a_mention_span_in_the_middle_is_removed_where_it_stands():
    activity = _activity(text="please <at>Osprey</at> report the current")

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.text == "please  report the current"


def test_interior_whitespace_survives_so_pasted_code_does():
    activity = _activity(text="<at>Osprey</at> run:\n\n    a = 1\n        b = 2\n")

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.text == "run:\n\n    a = 1\n        b = 2"


def test_a_typed_at_tag_without_an_entity_is_left_as_content():
    activity = _activity(text="<at>Osprey</at> what does <at>foo</at> mean?")

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.text == "what does <at>foo</at> mean?"


def test_a_mention_of_somebody_with_no_name_collapses_to_nothing():
    # The span still has to go — leaving raw markup in the question would ship
    # the agent a tag instead of a person.
    activity = _activity(
        text="<at>Osprey</at> ask <at>?</at> about it",
        entities=[_mention(), _mention(actor_id="29:bob", name="", span="<at>?</at>")],
    )

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.text == "ask  about it"


def test_a_message_that_is_nothing_but_the_mention_is_ignored():
    assert parse_event(_activity(text="<at>Osprey</at>"), CFG) is None


def test_a_message_with_no_text_at_all_is_ignored():
    assert parse_event(_activity(text=None), CFG) is None


# --- identity and dedup keys ----------------------------------------------


def test_the_message_id_scopes_the_activity_id_by_its_conversation():
    parsed = parse_event(_activity(), CFG)

    assert parsed is not None
    assert parsed.message_id == f"{CHANNEL_ID}:{ROOT_ID}"


def test_the_message_id_of_a_reply_uses_the_conversation_id_as_sent():
    parsed = parse_event(_reply_activity(), CFG)

    assert parsed is not None
    assert parsed.message_id == f"{CHANNEL_ID};messageid={ROOT_ID}:{REPLY_ID}"


def test_a_bot_authored_message_is_ignored_by_role():
    activity = _activity(**{"from": {"id": "29:someone", "name": "Osprey", "role": "bot"}})

    assert parse_event(activity, CFG) is None


def test_a_bot_authored_message_is_ignored_by_actor_id():
    # A relay that forwards the activity without its role must not defeat the
    # loop guard.
    activity = _activity(**{"from": {"id": BOT_ID, "name": "Osprey"}})

    assert parse_event(activity, CFG) is None


def test_a_non_message_activity_is_ignored():
    for activity_type in ("conversationUpdate", "messageReaction", "typing", "invoke", ""):
        assert parse_event(_activity(type=activity_type), CFG) is None


def test_an_activity_with_no_conversation_id_is_ignored():
    conversation = {"conversationType": "channel", "tenantId": TENANT}

    assert parse_event(_activity(conversation=conversation), CFG) is None


def test_an_activity_with_no_activity_id_is_ignored():
    assert parse_event(_activity(id=""), CFG) is None
    assert parse_event(_activity(id=None), CFG) is None


# --- history key -----------------------------------------------------------


def test_a_root_post_and_a_reply_in_its_thread_share_a_history_key():
    root = parse_event(_activity(), CFG)
    reply = parse_event(_reply_activity(), CFG)

    assert root is not None and reply is not None
    assert root.history_key == f"{CHANNEL_ID};messageid={ROOT_ID}"
    assert reply.history_key == root.history_key


def test_two_threads_in_one_channel_do_not_share_a_history_key():
    other_root = parse_event(_activity(id="1700000001234"), CFG)
    root = parse_event(_activity(), CFG)

    assert other_root is not None and root is not None
    assert other_root.history_key != root.history_key


def test_a_chat_keys_on_its_conversation_id_as_sent():
    parsed = parse_event(_chat_activity(), CFG)

    assert parsed is not None
    assert parsed.history_key == CHAT_ID


def test_a_group_chat_keys_on_its_conversation_id_as_sent():
    conversation = {"id": CHAT_ID, "conversationType": "groupChat", "tenantId": TENANT}

    parsed = parse_event(_activity(conversation=conversation), CFG)

    assert parsed is not None
    assert parsed.history_key == CHAT_ID


# --- claim_meta ------------------------------------------------------------


def test_claim_meta_carries_exactly_the_ms_namespace():
    parsed = parse_event(_activity(), CFG)

    assert parsed is not None
    assert dict(parsed.claim_meta) == {
        MS_SERVICE_URL: SERVICE_URL,
        MS_CONVERSATION_ID: CHANNEL_ID,
        MS_ACTIVITY_ID: ROOT_ID,
        MS_CONVERSATION_TYPE: "channel",
        MS_TENANT_ID: TENANT,
    }
    assert all(key.startswith("ms_") for key in parsed.claim_meta)


def test_claim_meta_keys_are_disjoint_from_the_engines_own():
    parsed = parse_event(_activity(), CFG)

    assert parsed is not None
    assert not set(parsed.claim_meta) & RESERVED_ENTRY_KEYS


def test_claim_meta_is_json_plain_so_it_round_trips_through_the_store():
    parsed = parse_event(_activity(), CFG)

    assert parsed is not None
    assert json.loads(json.dumps(dict(parsed.claim_meta))) == dict(parsed.claim_meta)


def test_the_claimed_conversation_id_is_the_one_the_connector_addresses():
    # The normalized history key names the thread ROOT; a post addressed to that
    # would land on the wrong message. The claim keeps the as-sent id.
    parsed = parse_event(_reply_activity(), CFG)

    assert parsed is not None
    assert parsed.claim_meta[MS_CONVERSATION_ID] == f"{CHANNEL_ID};messageid={ROOT_ID}"
    assert parsed.claim_meta[MS_ACTIVITY_ID] == REPLY_ID


def test_the_tenant_falls_back_to_channel_data():
    activity = _activity(
        conversation={"id": CHANNEL_ID, "conversationType": "channel"},
        channelData={"tenant": {"id": TENANT}, "channel": {"id": CHANNEL_ID}},
    )

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.claim_meta[MS_TENANT_ID] == TENANT


def test_a_missing_tenant_is_empty_rather_than_fatal():
    activity = _activity(conversation={"id": CHANNEL_ID, "conversationType": "channel"})

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.claim_meta[MS_TENANT_ID] == ""


def test_a_missing_service_url_is_empty_rather_than_fatal():
    parsed = parse_event(_activity(serviceUrl=None), CFG)

    assert parsed is not None
    assert parsed.claim_meta[MS_SERVICE_URL] == ""


# --- never raising ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "a string",
        42,
        [],
        [{"type": "message"}],
        {},
        {"type": "message"},
        {"type": ["message"]},
        {"type": "message", "id": ROOT_ID, "conversation": "19:room"},
        {"type": "message", "id": ROOT_ID, "conversation": {"id": CHANNEL_ID}, "entities": "x"},
        {
            "type": "message",
            "id": ROOT_ID,
            "conversation": {"id": CHAT_ID, "conversationType": "personal"},
            "from": "alice",
            "text": 7,
        },
        {
            "type": "message",
            "id": 17,
            "conversation": {"id": CHAT_ID, "conversationType": "personal"},
            "text": "hi",
        },
        {
            "type": "message",
            "id": ROOT_ID,
            "conversation": {"id": CHANNEL_ID, "conversationType": "channel"},
            "entities": [None, 3, {"type": "mention"}, {"type": "mention", "mentioned": "x"}],
            "text": "hi",
        },
    ],
)
def test_junk_is_ignored_rather_than_raised(raw):
    # Before the claim: an exception leaves the queue message unsettled and the
    # same activity redelivered until its lock expires.
    assert parse_event(raw, CFG) is None


def test_a_personal_chat_survives_a_junk_sender():
    activity = _chat_activity(**{"from": "alice"})

    parsed = parse_event(activity, CFG)

    assert parsed is not None
    assert parsed.sender_id == ""
    assert parsed.sender_display == ""


# --- reply context ---------------------------------------------------------


def test_reply_context_on_a_channel_reply_is_empty():
    parsed = parse_event(_reply_activity(), CFG)

    assert parsed is not None
    context = resolve_reply_context(parsed)

    assert isinstance(context, ReplyContext)
    assert dict(context.reply_to) == {}
    assert dict(context.meta) == {}


def test_reply_context_on_a_one_to_one_message_is_empty():
    parsed = parse_event(_chat_activity(), CFG)

    assert parsed is not None
    context = resolve_reply_context(parsed)

    assert isinstance(context, ReplyContext)
    assert dict(context.reply_to) == {}
    assert dict(context.meta) == {}


def test_reply_context_hands_back_a_fresh_mapping_each_time():
    # The engine copies what it persists, but a shared mutable default would let
    # one dispatch's fold leak into the next conversation's context.
    first = resolve_reply_context(parse_event(_activity(), CFG))
    second = resolve_reply_context(parse_event(_chat_activity(), CFG))

    assert first.reply_to is not second.reply_to
    assert first.meta is not second.meta


# --- names the conversation shows, for the room roster ------------------------


def test_people_seen_records_the_sender_and_mentioned_people():
    activity = _activity(
        text="<at>Osprey</at> ask <at>Carol</at>",
        entities=[_mention(), _mention("29:carol-aad-id", "Carol")],
    )
    assert people_seen(activity, APP_ID) == (
        CHANNEL_ID,
        {USER_ID: "Alice Example", "29:carol-aad-id": "Carol"},
    )


def test_people_seen_skips_the_bot_and_other_bots():
    activity = _activity(
        **{"from": {"id": "28:other-bot", "name": "Other", "role": "bot"}},
        entities=[_mention(), _mention("28:another", "Another bot")],
    )
    assert people_seen(activity, APP_ID) == (CHANNEL_ID, {})
    own = _activity(**{"from": {"id": BOT_ID, "name": "Osprey"}})
    assert people_seen(own, APP_ID) == (CHANNEL_ID, {})


def test_people_seen_keys_a_channel_reply_and_its_root_on_one_conversation():
    root, _ = people_seen(_activity(), APP_ID)
    reply, _ = people_seen(_reply_activity(), APP_ID)
    assert root == reply == CHANNEL_ID


@pytest.mark.parametrize("raw", [None, "junk", [], {"type": "message"}, {"conversation": "nope"}])
def test_people_seen_on_junk_is_empty(raw):
    assert people_seen(raw, APP_ID) == ("", {})


def test_roster_conversation_strips_only_the_thread_suffix():
    assert roster_conversation_id(f"{CHANNEL_ID};messageid={ROOT_ID}") == CHANNEL_ID
    assert roster_conversation_id(CHANNEL_ID) == CHANNEL_ID
    assert roster_conversation_id(CHAT_ID) == CHAT_ID
    assert roster_conversation_id("") == ""
