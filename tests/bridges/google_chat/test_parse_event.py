"""parse_event: which Chat events become questions, and what rides the claim.

Hermetic — the module under test imports nothing but the standard library and
the engine's port types, so there is no transport to fake.

The properties worth stating plainly, because each is a silent failure in
production if it breaks:

* **both wire shapes parse** — Pub/Sub delivers the Workspace add-on envelope,
  not the classic event every published example shows, so a parser that handles
  only the documented shape answers nothing at all;
* the **conditional graft** — the payload's space copy has been seen untyped,
  and overwriting the message's typed one with it would defeat DM detection for
  every direct message;
* the **room mention filter** — it is the access control, so a mention of
  somebody else must not summon the agent and neither must the literal text;
* the **loop guard** — the app answering its own answers is an unbounded loop in
  a live room;
* **claim_meta completeness** — the posting and download members run off the
  persisted entry long after the wire event is gone, so a reference the entry
  cannot address is an attachment the agent silently never sees;
* **never raising** — this runs before the dedup claim, so an exception leaves
  the Pub/Sub message unacknowledged and redelivered forever.
"""

import json

import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import (
    GC_ATTACHMENTS,
    GC_DRIVE_ATTACHMENTS,
    GC_IS_DM,
    GC_MESSAGE_NAME,
    GC_SPACE,
    GC_THREAD,
    parse_attachments,
    parse_event,
)

APP_ID = "users/999"
CFG = GoogleChatBridgeConfig(app_id=APP_ID)

SPACE = "spaces/AAAA"
THREAD = "spaces/AAAA/threads/DDDD"
MESSAGE = "spaces/AAAA/messages/BBBB.CCCC"


def _mention(app_name=APP_ID, start=0, length=7):
    """A ``USER_MENTION`` annotation, shaped as Chat sends one."""
    annotation = {
        "type": "USER_MENTION",
        "userMention": {
            "user": {"name": app_name, "displayName": "Osprey", "type": "BOT"},
            "type": "MENTION",
        },
    }
    if start is not None:
        annotation["startIndex"] = start
    if length is not None:
        annotation["length"] = length
    return annotation


def _message(**overrides):
    """A plausible Chat Message: a room question that @mentions the app."""
    message = {
        "name": MESSAGE,
        "sender": {"name": "users/111", "displayName": "Alice", "type": "HUMAN"},
        "text": "@Osprey what is the ring current?",
        "argumentText": " what is the ring current?",
        "thread": {"name": THREAD},
        "space": {"name": SPACE, "type": "ROOM"},
        "annotations": [_mention()],
    }
    message.update(overrides)
    return message


def _classic(**overrides):
    """The classic Chat event: ``{"type": "MESSAGE", "message": {...}}``."""
    return {
        "type": "MESSAGE",
        "eventTime": "2026-06-30T12:00:00.000000Z",
        "message": _message(**overrides),
    }


def _addon(*, payload_key="messagePayload", space_in_message=True, **overrides):
    """The Workspace add-on envelope Pub/Sub actually delivers.

    ``space_in_message=False`` drops the message's own space so the payload's
    copy is the only one — the case the graft exists for.
    """
    message = _message(**overrides)
    space = message["space"]
    if not space_in_message:
        message = {key: value for key, value in message.items() if key != "space"}
    return {
        "commonEventObject": {"hostApp": "CHAT", "platform": "WEB"},
        "chat": {
            "user": {"name": "users/111", "displayName": "Alice", "type": "HUMAN"},
            "eventTime": "2026-06-30T12:00:00.000000Z",
            payload_key: {"message": message, "space": space},
        },
    }


def _uploaded(content_name="ring.png", resource_name="spaces/AAAA/attachments/xyz"):
    """An ``UPLOADED_CONTENT`` attachment entry, camelCase as REST sends it."""
    return {
        "name": f"{MESSAGE}/attachments/xyz",
        "contentName": content_name,
        "contentType": "image/png",
        "attachmentDataRef": {"resourceName": resource_name},
        "source": "UPLOADED_CONTENT",
    }


def _drive(content_name="report.pdf"):
    """A ``DRIVE_FILE`` attachment entry — no app-readable resource name."""
    return {
        "name": f"{MESSAGE}/attachments/drv",
        "contentName": content_name,
        "contentType": "application/pdf",
        "driveDataRef": {"driveFileId": "1a2b3c"},
        "source": "DRIVE_FILE",
    }


# ==========================================================================
# The happy path, and the shape of what it produces
# ==========================================================================


def test_a_room_mention_is_accepted_and_the_mention_stripped():
    event = parse_event(_classic(), CFG)

    assert event is not None
    assert event.message_id == MESSAGE
    assert event.text == "what is the ring current?"
    assert event.sender_id == "users/111"
    assert event.sender_display == "Alice"


def test_the_message_name_is_the_dedup_key_unscoped():
    """Chat message names embed their space, so unlike a Talk message id they
    are already globally unique and need no extra scoping."""
    event = parse_event(_classic(), CFG)

    assert event is not None
    assert event.message_id == MESSAGE


def test_the_normalized_message_is_carried_for_the_reply_context():
    """Reply resolution reads the Message resource inline, so what is carried
    has to be the extracted message — not the envelope it arrived in."""
    event = parse_event(_addon(), CFG)

    assert event is not None
    assert event.raw["name"] == MESSAGE
    assert event.raw["space"]["name"] == SPACE


# ==========================================================================
# history_key — a DM is one conversation, a room conversation is its thread
# ==========================================================================


def test_a_room_message_keys_history_on_its_thread():
    event = parse_event(_classic(), CFG)

    assert event is not None
    assert event.history_key == THREAD


def test_a_dm_keys_history_on_the_space():
    """Each DM message technically starts its own thread, so keying a DM by
    thread would never link two consecutive turns."""
    event = parse_event(_classic(space={"name": SPACE, "spaceType": "DIRECT_MESSAGE"}), CFG)

    assert event is not None
    assert event.history_key == SPACE


def test_a_room_message_with_no_thread_falls_back_to_the_space():
    event = parse_event(_classic(thread={}), CFG)

    assert event is not None
    assert event.history_key == SPACE


# ==========================================================================
# The two wire shapes, and the conditional graft between them
# ==========================================================================


def test_the_addon_envelope_parses_like_the_classic_event():
    event = parse_event(_addon(), CFG)

    assert event is not None
    assert event.message_id == MESSAGE
    assert event.text == "what is the ring current?"
    assert event.history_key == THREAD
    assert event.claim_meta[GC_SPACE] == SPACE


def test_the_payload_space_is_grafted_when_the_message_carries_none():
    event = parse_event(_addon(space_in_message=False), CFG)

    assert event is not None
    assert event.claim_meta[GC_SPACE] == SPACE


def test_the_payload_space_never_overwrites_the_messages_own():
    """The message's space is the typed one; the payload's copy has been seen
    without a ``spaceType``. Grafting over the typed object would defeat DM
    detection for every direct message, so a present key always wins."""
    event = _addon(space={"name": SPACE, "spaceType": "DIRECT_MESSAGE"}, annotations=[])
    event["chat"]["messagePayload"]["space"] = {"name": "spaces/WRONG"}

    parsed = parse_event(event, CFG)

    # Parsed at all only because the typed DM space survived: with no mention,
    # the untyped payload copy would have made this a room message and dropped it.
    assert parsed is not None
    assert parsed.claim_meta[GC_SPACE] == SPACE
    assert parsed.claim_meta[GC_IS_DM] is True


def test_the_envelope_user_is_grafted_as_the_sender_when_the_message_has_none():
    event = _addon()
    del event["chat"]["messagePayload"]["message"]["sender"]

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert parsed.sender_id == "users/111"
    assert parsed.sender_display == "Alice"


def test_the_messages_own_sender_wins_over_the_envelope_user():
    event = _addon(sender={"name": "users/222", "displayName": "Bob", "type": "HUMAN"})

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert parsed.sender_id == "users/222"


def test_the_input_event_is_not_mutated_by_the_graft():
    event = _addon(space_in_message=False)

    assert parse_event(event, CFG) is not None
    assert "space" not in event["chat"]["messagePayload"]["message"]


# ==========================================================================
# Which events are questions at all
# ==========================================================================


@pytest.mark.parametrize("event_type", ["ADDED_TO_SPACE", "REMOVED_FROM_SPACE", "CARD_CLICKED"])
def test_a_non_message_classic_event_is_ignored(event_type):
    event = _classic()
    event["type"] = event_type

    assert parse_event(event, CFG) is None


@pytest.mark.parametrize("payload_key", ["addedToSpacePayload", "removedFromSpacePayload"])
def test_a_non_message_addon_payload_is_ignored(payload_key):
    """The add-on envelope keys its payload by event kind, so any key other
    than ``messagePayload`` is how a non-message event announces itself."""
    assert parse_event(_addon(payload_key=payload_key), CFG) is None


def test_a_classic_event_with_no_message_is_ignored():
    assert parse_event({"type": "MESSAGE"}, CFG) is None


def test_an_event_with_no_type_and_no_chat_key_is_ignored():
    assert parse_event({"message": _message()}, CFG) is None


def test_a_message_with_no_resource_name_is_ignored():
    """Without the name there is no dedup key, so a redelivery could not be
    recognized and the same question would be answered twice."""
    assert parse_event(_classic(name=""), CFG) is None
    message = _message()
    del message["name"]
    assert parse_event({"type": "MESSAGE", "message": message}, CFG) is None


# ==========================================================================
# Addressing — the room mention filter and the DM implicit rule
# ==========================================================================


def test_a_room_message_without_a_mention_is_ignored():
    assert parse_event(_classic(annotations=[]), CFG) is None


def test_a_room_mention_of_another_user_is_ignored():
    assert parse_event(_classic(annotations=[_mention(app_name="users/12345")]), CFG) is None


def test_the_literal_app_name_in_the_text_is_not_a_mention():
    """The filter reads annotations, not text: anything else would let a user
    summon the agent by typing its name."""
    event = _classic(annotations=[], text="@Osprey what is the ring current?")

    assert parse_event(event, CFG) is None


def test_a_mention_of_the_app_beside_another_user_is_accepted():
    event = _classic(annotations=[_mention(app_name="users/12345"), _mention()])

    assert parse_event(event, CFG) is not None


def test_a_mention_without_offsets_still_addresses_the_app():
    """Offsets are needed to slice the mention out of the text, not to decide
    that the message was addressed — a mention Chat sent without them is still
    a question for the app."""
    event = _classic(annotations=[_mention(start=None, length=None)])

    assert parse_event(event, CFG) is not None


@pytest.mark.parametrize(
    "space", [{"name": SPACE, "spaceType": "DIRECT_MESSAGE"}, {"name": SPACE, "type": "DM"}]
)
def test_a_dm_needs_no_mention(space):
    """Both space schemas are live on the wire: ``spaceType`` on the current
    Space resource, ``type`` on its deprecated predecessor."""
    event = _classic(space=space, annotations=[], argumentText=None, text="what is the current?")
    del event["message"]["argumentText"]

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert parsed.text == "what is the current?"
    assert parsed.claim_meta[GC_IS_DM] is True


def test_an_addon_dm_needs_no_mention():
    event = _addon(
        space={"name": SPACE, "spaceType": "DIRECT_MESSAGE"},
        annotations=[],
        text="what is the current?",
    )
    del event["chat"]["messagePayload"]["message"]["argumentText"]

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert parsed.text == "what is the current?"


def test_a_dm_still_strips_a_mention_that_is_present():
    event = _classic(space={"name": SPACE, "spaceType": "DIRECT_MESSAGE"})

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert parsed.text == "what is the ring current?"


def test_the_apps_own_message_is_ignored():
    assert parse_event(_classic(sender={"name": APP_ID, "type": "BOT"}), CFG) is None


def test_the_loop_guard_holds_in_a_dm_where_no_mention_is_needed():
    event = _classic(
        space={"name": SPACE, "spaceType": "DIRECT_MESSAGE"},
        annotations=[],
        sender={"name": APP_ID, "type": "BOT"},
    )

    assert parse_event(event, CFG) is None


# ==========================================================================
# Loose app-id matching
# ==========================================================================


@pytest.mark.parametrize("configured", ["users/999", "999"])
def test_the_app_id_matches_either_spelling(configured):
    """The id is copied out of the Chat console by hand and both spellings are
    natural to paste, so the match is on the last path segment."""
    assert parse_event(_classic(), GoogleChatBridgeConfig(app_id=configured)) is not None


def test_a_prefix_of_the_app_id_does_not_match():
    assert parse_event(_classic(), GoogleChatBridgeConfig(app_id="99")) is None


def test_an_unconfigured_app_id_matches_nothing():
    """An empty id must not match every mention — a bridge started without
    ``GCHAT_APP_ID`` would otherwise answer every message in every room."""
    assert parse_event(_classic(), GoogleChatBridgeConfig()) is None


# ==========================================================================
# Text — argumentText first, offset slicing as the fallback
# ==========================================================================


def test_argument_text_is_preferred_over_slicing():
    event = _classic(text="@Osprey  hello", argumentText=" hello")

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert parsed.text == "hello"


def test_the_offsets_slice_the_mention_out_when_there_is_no_argument_text():
    message = _message(text="@Osprey plot the current", annotations=[_mention(length=7)])
    del message["argumentText"]

    parsed = parse_event({"type": "MESSAGE", "message": message}, CFG)

    assert parsed is not None
    assert parsed.text == "plot the current"


def test_repeated_mentions_are_all_sliced_out():
    """Back-to-front, so removing an earlier span cannot invalidate a later
    span's offsets."""
    message = _message(
        text="@Osprey ring current @Osprey now",
        annotations=[_mention(start=0, length=7), _mention(start=21, length=7)],
    )
    del message["argumentText"]

    parsed = parse_event({"type": "MESSAGE", "message": message}, CFG)

    assert parsed is not None
    assert parsed.text == "ring current  now"


def test_a_mention_of_another_user_is_left_in_the_text():
    """Someone else's mention is content, not addressing."""
    message = _message(
        text="@Osprey ask @Bob about it",
        annotations=[
            _mention(start=0, length=7),
            _mention(app_name="users/12345", start=12, length=4),
        ],
    )
    del message["argumentText"]

    parsed = parse_event({"type": "MESSAGE", "message": message}, CFG)

    assert parsed is not None
    assert parsed.text == "ask @Bob about it"


def test_a_mention_without_offsets_leaves_the_text_alone():
    message = _message(text="@Osprey hello", annotations=[_mention(start=None, length=None)])
    del message["argumentText"]

    parsed = parse_event({"type": "MESSAGE", "message": message}, CFG)

    assert parsed is not None
    assert parsed.text == "@Osprey hello"


def test_interior_whitespace_is_left_alone():
    """Pasted code and markdown have to survive: only the ends are stripped."""
    body = " why does this fail?\n\n    for i in range(3):\n        print(i)\n"

    parsed = parse_event(_classic(argumentText=body), CFG)

    assert parsed is not None
    assert parsed.text == "why does this fail?\n\n    for i in range(3):\n        print(i)"


def test_an_empty_argument_text_is_taken_as_the_whole_answer():
    """A bare address with no question: Chat says so by sending an empty
    ``argumentText``, and that is preferred over re-deriving it from ``text``."""
    parsed = parse_event(_classic(argumentText=""), CFG)

    assert parsed is not None
    assert parsed.text == ""


# ==========================================================================
# Attachments — uploaded refs the app can fetch, Drive names it cannot
# ==========================================================================


def test_a_text_only_message_carries_no_attachment_references():
    event = parse_event(_classic(), CFG)

    assert event is not None
    assert event.claim_meta[GC_ATTACHMENTS] == []
    assert event.claim_meta[GC_DRIVE_ATTACHMENTS] == []


def test_an_uploaded_attachment_becomes_a_fetchable_reference():
    event = parse_event(_classic(attachment=[_uploaded()]), CFG)

    assert event is not None
    # Every field download_inputs needs to re-fetch from the entry alone.
    assert event.claim_meta[GC_ATTACHMENTS] == [
        {
            "content_name": "ring.png",
            "content_type": "image/png",
            "resource_name": "spaces/AAAA/attachments/xyz",
            "source": "UPLOADED_CONTENT",
        }
    ]
    assert event.claim_meta[GC_DRIVE_ATTACHMENTS] == []


def test_a_lone_attachment_object_is_normalized_to_a_list():
    """The repeated field arrives as a list over REST but has been observed as
    a single object in event payloads."""
    event = parse_event(_classic(attachment=_uploaded()), CFG)

    assert event is not None
    assert [ref["resource_name"] for ref in event.claim_meta[GC_ATTACHMENTS]] == [
        "spaces/AAAA/attachments/xyz"
    ]


def test_snake_case_attachment_fields_are_accepted():
    entry = {
        "content_name": "trace.csv",
        "content_type": "text/csv",
        "attachment_data_ref": {"resource_name": "spaces/AAAA/attachments/snk"},
        "source": "UPLOADED_CONTENT",
    }

    event = parse_event(_classic(attachment=[entry]), CFG)

    assert event is not None
    assert event.claim_meta[GC_ATTACHMENTS] == [
        {
            "content_name": "trace.csv",
            "content_type": "text/csv",
            "resource_name": "spaces/AAAA/attachments/snk",
            "source": "UPLOADED_CONTENT",
        }
    ]


def test_a_drive_file_is_reported_by_name_rather_than_referenced():
    """The app's own credentials cannot read a ``driveDataRef``, so the name is
    kept to tell the user the file was skipped."""
    event = parse_event(_classic(attachment=[_drive("weekly.pdf")]), CFG)

    assert event is not None
    assert event.claim_meta[GC_ATTACHMENTS] == []
    assert event.claim_meta[GC_DRIVE_ATTACHMENTS] == ["weekly.pdf"]


def test_mixed_attachments_are_split_into_the_two_buckets():
    event = parse_event(_classic(attachment=[_uploaded(), _drive("design.pdf")]), CFG)

    assert event is not None
    assert [ref["content_name"] for ref in event.claim_meta[GC_ATTACHMENTS]] == ["ring.png"]
    assert event.claim_meta[GC_DRIVE_ATTACHMENTS] == ["design.pdf"]


def test_multiple_attachments_keep_their_order():
    event = parse_event(
        _classic(
            attachment=[
                _uploaded(content_name="a.png", resource_name="spaces/AAAA/attachments/a"),
                _uploaded(content_name="b.png", resource_name="spaces/AAAA/attachments/b"),
            ]
        ),
        CFG,
    )

    assert event is not None
    assert [ref["content_name"] for ref in event.claim_meta[GC_ATTACHMENTS]] == ["a.png", "b.png"]


def test_an_attachment_without_a_resource_name_is_dropped():
    """A reference nothing could fetch is worse than no reference: it looks
    like an attachment the agent should have received."""
    broken = {"contentName": "broken.png", "contentType": "image/png", "source": "UPLOADED_CONTENT"}

    event = parse_event(_classic(attachment=[broken, _uploaded()]), CFG)

    assert event is not None
    assert [ref["content_name"] for ref in event.claim_meta[GC_ATTACHMENTS]] == ["ring.png"]


def test_an_attachment_of_an_unknown_source_is_dropped():
    entry = dict(_uploaded(), source="SOMETHING_NEW")

    event = parse_event(_classic(attachment=[entry]), CFG)

    assert event is not None
    assert event.claim_meta[GC_ATTACHMENTS] == []
    assert event.claim_meta[GC_DRIVE_ATTACHMENTS] == []


def test_an_attachment_with_no_source_is_taken_as_uploaded():
    entry = {k: v for k, v in _uploaded().items() if k != "source"}

    event = parse_event(_classic(attachment=[entry]), CFG)

    assert event is not None
    assert [ref["source"] for ref in event.claim_meta[GC_ATTACHMENTS]] == ["UPLOADED_CONTENT"]


def test_the_addon_wire_shape_carries_attachments():
    event = _addon()
    event["chat"]["messagePayload"]["message"]["attachment"] = [_uploaded(), _drive("d.pdf")]

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert [ref["content_name"] for ref in parsed.claim_meta[GC_ATTACHMENTS]] == ["ring.png"]
    assert parsed.claim_meta[GC_DRIVE_ATTACHMENTS] == ["d.pdf"]


def test_a_dm_attachment_is_parsed_without_a_mention():
    event = _classic(space={"name": SPACE, "spaceType": "DIRECT_MESSAGE"}, annotations=[])
    event["message"]["attachment"] = [_uploaded()]

    parsed = parse_event(event, CFG)

    assert parsed is not None
    assert len(parsed.claim_meta[GC_ATTACHMENTS]) == 1


def test_parse_attachments_reads_a_rest_message_view_directly():
    """It is shared with the quote-reply path, which sees the quoted message
    only through ``spaces.messages.get``."""
    references, drive_names = parse_attachments(
        {"name": MESSAGE, "attachment": [_uploaded(), _drive("q.pdf")]}
    )

    assert [ref["content_name"] for ref in references] == ["ring.png"]
    assert drive_names == ["q.pdf"]


@pytest.mark.parametrize("message", [None, "not a message", [], {}, {"attachment": "nonsense"}])
def test_parse_attachments_yields_empties_for_anything_unusable(message):
    assert parse_attachments(message) == ([], [])


# ==========================================================================
# claim_meta — the posting and download members read only this
# ==========================================================================


def test_claim_meta_carries_the_destination_and_the_message_name():
    event = parse_event(_classic(), CFG)

    assert event is not None
    assert event.claim_meta[GC_SPACE] == SPACE
    assert event.claim_meta[GC_THREAD] == THREAD
    assert event.claim_meta[GC_MESSAGE_NAME] == MESSAGE
    assert event.claim_meta[GC_IS_DM] is False


def test_a_message_that_started_no_thread_carries_a_null_thread():
    event = parse_event(_classic(thread={}), CFG)

    assert event is not None
    assert event.claim_meta[GC_THREAD] is None


def test_claim_meta_keys_are_prefixed_and_disjoint_from_the_engines():
    """The adapter's meta lands in the SAME flat entry dict as the engine's
    bookkeeping, so a collision is a silent overwrite in one direction or the
    other. The ``gc_`` prefix is what keeps the two namespaces apart."""
    event = parse_event(_classic(attachment=[_uploaded()]), CFG)

    assert event is not None
    keys = set(event.claim_meta)
    assert keys and all(key.startswith("gc_") for key in keys)
    assert keys.isdisjoint(RESERVED_ENTRY_KEYS)


def test_claim_meta_is_json_plain():
    """It round-trips through the on-disk dedup store, so anything that is not
    plain JSON would fail at claim time — or worse, come back a different shape
    after a restart."""
    event = parse_event(_classic(attachment=[_uploaded(), _drive()]), CFG)

    assert event is not None
    meta = dict(event.claim_meta)
    assert json.loads(json.dumps(meta)) == meta


# ==========================================================================
# Never raising — this runs before the claim, so an exception redelivers
# ==========================================================================


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "a string",
        42,
        [],
        [{"type": "MESSAGE"}],
        {},
        {"type": "MESSAGE", "message": "not a mapping"},
        {"chat": "not a mapping"},
        {"chat": {}},
        {"chat": {"messagePayload": "not a mapping"}},
        {"chat": {"messagePayload": {}}},
        {"chat": {"messagePayload": {"message": None}}},
    ],
)
def test_junk_payloads_are_ignored_rather_than_raising(raw):
    assert parse_event(raw, CFG) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": 17},
        {"sender": "Alice"},
        {"sender": []},
        {"space": "spaces/AAAA"},
        {"thread": "a thread"},
        {"text": 5, "argumentText": None},
        {"argumentText": 5},
        {"annotations": "not a list"},
        {"annotations": [None, "junk", 7]},
        {"annotations": [{"type": "USER_MENTION", "userMention": "junk"}]},
        {"annotations": _mention()},
        {"attachment": "not a list"},
        {"attachment": [None, 7]},
        {"attachment": [{"attachmentDataRef": "junk", "source": "UPLOADED_CONTENT"}]},
    ],
)
def test_malformed_message_fields_never_raise(overrides):
    """Every one of these is untrusted input off Pub/Sub, not a programming
    error, so the answer is a parse result or ``None`` — never an exception."""
    parse_event(_classic(**overrides), CFG)


def test_a_message_field_that_is_not_a_string_is_read_as_absent():
    event = parse_event(_classic(name=17), CFG)

    assert event is None
