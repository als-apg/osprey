"""resolve_reply_context: which quote resolves, from where, and at what cost.

Hermetic — the reads are an injected callable, so there is no transport to fake.

The properties worth stating plainly, because each is a silent failure in
production if it breaks:

* **the event is preferred** — the snapshot is usually already on the wire, and
  re-reading a message that carried one would spend a Chat call per question for
  nothing;
* **the fallback fires, and fires once** — only the REST view is confirmed to
  carry the snapshot, so a message without one is re-read; a second read of the
  same view would be a per-message cost with no new information;
* **name-only metadata is not a quote** — it must resolve to ``None`` at parse
  time, or the fallback never fires and the agent gets an empty quote instead of
  the words;
* **the quoted message's attachments** — a quote-reply to an image carries no
  text at all, so without that second read the agent knows a message was pointed
  at but cannot see it;
* **provenance stays separate** — quoted files ride their own meta keys, since
  an entry that cannot tell them from the message's own would tell the agent the
  user had just attached a file they only pointed at;
* **never fatal** — reply context is enrichment, so a failing read costs the
  quote and never the answer.
"""

import json
import logging

import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS, InboundEvent
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import (
    GC_ATTACHMENTS,
    GC_DRIVE_ATTACHMENTS,
    GC_IS_DM,
    GC_MESSAGE_NAME,
    GC_QUOTED_ATTACHMENTS,
    GC_QUOTED_DRIVE_ATTACHMENTS,
    GC_SPACE,
    GC_THREAD,
    REPLY_TO_MAX_CHARS,
    QuotedMessage,
    parse_event,
    parse_quoted,
    resolve_reply_context,
)

APP_ID = "users/999"
CFG = GoogleChatBridgeConfig(app_id=APP_ID)

SPACE = "spaces/AAAA"
MESSAGE = "spaces/AAAA/messages/BBBB.CCCC"
QUOTED = "spaces/AAAA/messages/QQQQ.RRRR"

QUOTED_TEXT = "On it 🦅 — running your request, one moment…"


def _quoted_metadata(name=QUOTED, sender="Osprey", text=QUOTED_TEXT):
    """A ``quotedMessageMetadata`` block as the live Chat API delivers one: the
    quoted message's name, the quote type, and an inline snapshot carrying who
    wrote it and what it said."""
    return {
        "name": name,
        "lastUpdateTime": "2026-07-23T14:06:56.078127Z",
        "quoteType": "REPLY",
        "quotedMessageSnapshot": {"sender": sender, "text": text},
    }


def _classic(**message_overrides):
    """The classic Chat event: a room question that @mentions the app."""
    message = {
        "name": MESSAGE,
        "sender": {"name": "users/111", "displayName": "Alice", "type": "HUMAN"},
        "text": "@Osprey what is this?",
        "argumentText": " what is this?",
        "thread": {"name": "spaces/AAAA/threads/DDDD"},
        "space": {"name": SPACE, "type": "ROOM"},
        "annotations": [
            {
                "type": "USER_MENTION",
                "startIndex": 0,
                "length": 7,
                "userMention": {"user": {"name": APP_ID, "type": "BOT"}, "type": "MENTION"},
            }
        ],
    }
    message.update(message_overrides)
    return {"type": "MESSAGE", "message": message}


def _addon(**message_overrides):
    """The Workspace add-on envelope Pub/Sub actually delivers."""
    classic = _classic(**message_overrides)
    return {
        "commonEventObject": {"hostApp": "CHAT"},
        "chat": {
            "user": {"name": "users/111", "displayName": "Alice", "type": "HUMAN"},
            "messagePayload": {"message": classic["message"], "space": {"name": SPACE}},
        },
    }


def _event(**message_overrides):
    """A parsed :class:`InboundEvent`, the way reply resolution receives one."""
    event = parse_event(_classic(**message_overrides), CFG)
    assert event is not None
    return event


def _uploaded(content_name, resource_name="spaces/AAAA/attachments/q1"):
    """An uploaded (fetchable) attachment on a Message resource."""
    return {
        "source": "UPLOADED_CONTENT",
        "contentName": content_name,
        "contentType": "image/png",
        "attachmentDataRef": {"resourceName": resource_name},
    }


class Reader:
    """A recording ``get_message``: answers per resource name, remembers the
    order it was asked, so a test can state both what resolved and what it cost.
    """

    def __init__(self, bodies=None, raises=None):
        self.bodies = bodies or {}
        self.raises = raises or set()
        self.calls: list[str] = []

    def __call__(self, name):
        self.calls.append(name)
        if name in self.raises:
            raise RuntimeError("Chat is having a day")
        return self.bodies.get(name)


# --- parse_quoted ----------------------------------------------------------


def test_an_inline_snapshot_parses_into_a_quoted_message():
    quoted = parse_quoted({"name": MESSAGE, "quotedMessageMetadata": _quoted_metadata()})
    assert quoted == QuotedMessage(message_name=QUOTED, sender="Osprey", text=QUOTED_TEXT)


def test_the_snake_case_spelling_of_the_quote_fields_is_accepted():
    quoted = parse_quoted(
        {
            "quoted_message_metadata": {
                "name": QUOTED,
                "quoted_message_snapshot": {"sender": "Alice", "text": "earlier words"},
            }
        }
    )
    assert quoted == QuotedMessage(message_name=QUOTED, sender="Alice", text="earlier words")


def test_metadata_without_a_snapshot_is_not_a_quote():
    """Name-only metadata is unusable context on its own. Answering None here is
    what lets the caller's re-read fire and resolve the actual content."""
    metadata = {"name": QUOTED, "quoteType": "REPLY"}
    assert parse_quoted({"quotedMessageMetadata": metadata}) is None


def test_a_quoted_message_with_a_sender_but_no_text_is_still_a_quote():
    """A quoted image-only or card-only message carries no text; the sender alone
    is still usable context ('replying to a message from Osprey')."""
    quoted = parse_quoted({"quotedMessageMetadata": _quoted_metadata(text="")})
    assert quoted == QuotedMessage(message_name=QUOTED, sender="Osprey", text="")


def test_a_quote_whose_metadata_named_no_message_still_parses():
    quoted = parse_quoted({"quotedMessageMetadata": {"quotedMessageSnapshot": {"text": "hi"}}})
    assert quoted == QuotedMessage(message_name="", sender="", text="hi")


@pytest.mark.parametrize(
    "message",
    [None, "a string", 17, {}, {"text": "not a reply"}, {"quotedMessageMetadata": "junk"}],
    ids=["none", "string", "number", "empty", "no-metadata", "metadata-not-a-mapping"],
)
def test_anything_that_is_not_a_quote_reply_parses_to_none(message):
    assert parse_quoted(message) is None


# --- resolution order ------------------------------------------------------


def test_a_quote_on_the_event_resolves_without_re_reading_the_message():
    """The snapshot was on the wire, so the only read is the quoted message's
    attachment check — never the current message."""
    reader = Reader()
    context = resolve_reply_context(
        _event(quotedMessageMetadata=_quoted_metadata()),
        reader,
    )

    assert context is not None
    assert context.reply_to == {"sender": "Osprey", "text": QUOTED_TEXT}
    assert reader.calls == [QUOTED]


def test_the_add_on_wire_shape_resolves_the_same_inline_quote():
    """Pub/Sub delivers the add-on envelope, and parse_event normalizes it, so
    resolution reads one message shape regardless of which wire it arrived on."""
    event = parse_event(_addon(quotedMessageMetadata=_quoted_metadata()), CFG)
    assert event is not None

    context = resolve_reply_context(event, Reader())
    assert context is not None
    assert context.reply_to["sender"] == "Osprey"


def test_an_event_without_a_snapshot_resolves_through_one_re_read():
    """Only the REST view is confirmed to carry the snapshot, so a message that
    arrived without one is re-read once and parsed off that view instead."""
    reader = Reader(
        {
            MESSAGE: {
                "name": MESSAGE,
                "quotedMessageMetadata": _quoted_metadata(
                    sender="Alice", text="the earlier question"
                ),
            }
        }
    )
    context = resolve_reply_context(_event(), reader)

    assert context is not None
    assert context.reply_to == {"sender": "Alice", "text": "the earlier question"}
    # The fallback read of the current message, then the quoted message's
    # attachment check — one each, in that order.
    assert reader.calls == [MESSAGE, QUOTED]


def test_name_only_metadata_on_the_event_still_lets_the_re_read_resolve_it():
    """The case the parse-time None exists for: the event named a quoted message
    but carried no snapshot, and the re-read supplies the words."""
    reader = Reader({MESSAGE: {"quotedMessageMetadata": _quoted_metadata(sender="Bob")}})
    event = _event(quotedMessageMetadata={"name": QUOTED, "quoteType": "REPLY"})

    context = resolve_reply_context(event, reader)
    assert context is not None
    assert context.reply_to["sender"] == "Bob"
    assert reader.calls == [MESSAGE, QUOTED]


def test_a_message_that_is_not_a_reply_resolves_to_no_context():
    """Neither view carried a quote: no context, and the re-read is not retried
    — one guarded read is the whole cost of asking."""
    reader = Reader({MESSAGE: {"name": MESSAGE, "text": "plain"}})
    assert resolve_reply_context(_event(), reader) is None
    assert reader.calls == [MESSAGE]


def test_the_resolving_path_is_named_in_the_log(caplog):
    """Production logs have to be able to say whether the event ever carries the
    snapshot, which is what decides if the fallback can be retired."""
    with caplog.at_level(logging.INFO, logger="osprey.bridges.google_chat.events"):
        resolve_reply_context(_event(quotedMessageMetadata=_quoted_metadata()), Reader())
        resolve_reply_context(
            _event(), Reader({MESSAGE: {"quotedMessageMetadata": _quoted_metadata()}})
        )
        resolve_reply_context(_event(), Reader())

    resolved = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(resolved) == 3
    assert "the event" in resolved[0]
    assert "messages.get" in resolved[1]
    assert "no reply context" in resolved[2]


# --- the quoted text -------------------------------------------------------


def test_quoted_text_is_truncated_to_a_hard_character_bound():
    """The quote identifies what is being replied to; it does not replay it. The
    elision marker counts against the cap, so the cap is the whole bound."""
    long_quote = "x" * (REPLY_TO_MAX_CHARS + 500)
    context = resolve_reply_context(
        _event(quotedMessageMetadata=_quoted_metadata(text=long_quote)),
        Reader(),
    )

    assert context is not None
    text = context.reply_to["text"]
    assert len(text) == REPLY_TO_MAX_CHARS
    assert text.endswith("…")


def test_a_quote_at_the_cap_is_shipped_whole():
    exact = "y" * REPLY_TO_MAX_CHARS
    context = resolve_reply_context(
        _event(quotedMessageMetadata=_quoted_metadata(text=exact)),
        Reader(),
    )

    assert context is not None
    assert context.reply_to["text"] == exact


# --- the quoted message's attachments --------------------------------------


def test_the_quoted_messages_attachments_ride_the_reply_meta():
    """The live case a quote-reply exists for: replying to an image-only message.
    The snapshot carries no text, and the file is only on the quoted message's
    own view — without this read the agent cannot see what it was shown."""
    reader = Reader({QUOTED: {"name": QUOTED, "attachment": [_uploaded("table.png")]}})
    context = resolve_reply_context(
        _event(quotedMessageMetadata=_quoted_metadata(sender="Simon", text="")),
        reader,
    )

    assert context is not None
    assert context.meta[GC_QUOTED_ATTACHMENTS] == [
        {
            "content_name": "table.png",
            "content_type": "image/png",
            "resource_name": "spaces/AAAA/attachments/q1",
            "source": "UPLOADED_CONTENT",
        }
    ]
    assert context.meta[GC_QUOTED_DRIVE_ATTACHMENTS] == []


def test_a_drive_file_on_the_quoted_message_is_reported_by_name():
    """The app's own credentials cannot read a Drive attachment, so it is named
    rather than fetched and the answer can say it was skipped."""
    reader = Reader(
        {QUOTED: {"attachment": [{"source": "DRIVE_FILE", "contentName": "quarterly.pdf"}]}}
    )
    context = resolve_reply_context(_event(quotedMessageMetadata=_quoted_metadata()), reader)

    assert context is not None
    assert context.meta[GC_QUOTED_ATTACHMENTS] == []
    assert context.meta[GC_QUOTED_DRIVE_ATTACHMENTS] == ["quarterly.pdf"]


def test_a_quote_with_no_attachments_still_states_both_meta_keys():
    """A persisted entry answers 'were there quoted files?' the same way whether
    or not there were any, so no reader has to tell absent from empty."""
    context = resolve_reply_context(_event(quotedMessageMetadata=_quoted_metadata()), Reader())

    assert context is not None
    assert context.meta == {GC_QUOTED_ATTACHMENTS: [], GC_QUOTED_DRIVE_ATTACHMENTS: []}


def test_metadata_that_named_no_quoted_message_skips_the_attachment_read():
    """There is nothing to address, so the quote resolves on its text alone."""
    reader = Reader()
    metadata = {"quotedMessageSnapshot": {"sender": "Alice", "text": "words"}}
    context = resolve_reply_context(_event(quotedMessageMetadata=metadata), reader)

    assert context is not None
    assert context.reply_to == {"sender": "Alice", "text": "words"}
    assert reader.calls == []


# --- never fatal -----------------------------------------------------------


def test_a_failing_attachment_read_keeps_the_reply_context():
    """The text is the context and the files are a bonus: losing the quote
    because the second read failed would be the worse answer."""
    reader = Reader(raises={QUOTED})
    context = resolve_reply_context(_event(quotedMessageMetadata=_quoted_metadata()), reader)

    assert context is not None
    assert context.reply_to["sender"] == "Osprey"
    assert context.meta == {GC_QUOTED_ATTACHMENTS: [], GC_QUOTED_DRIVE_ATTACHMENTS: []}


def test_a_failing_re_read_of_the_current_message_resolves_to_no_context():
    assert resolve_reply_context(_event(), Reader(raises={MESSAGE})) is None


@pytest.mark.parametrize(
    "body",
    ["a string", 17, ["a", "list"], None],
    ids=["string", "number", "list", "nothing"],
)
def test_a_re_read_answering_something_that_is_not_a_message_resolves_to_no_context(body):
    assert resolve_reply_context(_event(), Reader({MESSAGE: body})) is None


def test_a_quoted_view_that_is_not_a_message_costs_only_its_attachments():
    reader = Reader({QUOTED: "not a message"})
    context = resolve_reply_context(_event(quotedMessageMetadata=_quoted_metadata()), reader)

    assert context is not None
    assert context.meta[GC_QUOTED_ATTACHMENTS] == []


def test_an_event_carrying_no_raw_message_falls_back_to_the_re_read():
    """The engine hands back whatever the adapter put on the event; a re-dispatch
    view with no raw message must still resolve rather than raise."""
    event = InboundEvent(
        message_id=MESSAGE,
        text="what is this?",
        sender_id="users/111",
        sender_display="Alice",
        history_key=SPACE,
    )
    reader = Reader({MESSAGE: {"quotedMessageMetadata": _quoted_metadata()}})

    context = resolve_reply_context(event, reader)
    assert context is not None
    assert context.reply_to["sender"] == "Osprey"


# --- the entry namespace ---------------------------------------------------


def test_the_quoted_meta_keys_collide_with_nothing_else_in_the_entry():
    """Reply meta, claim meta and the engine's own bookkeeping land in ONE flat
    entry: a collision is a silent overwrite in one direction or the other."""
    quoted_keys = {GC_QUOTED_ATTACHMENTS, GC_QUOTED_DRIVE_ATTACHMENTS}
    claim_keys = {
        GC_SPACE,
        GC_THREAD,
        GC_MESSAGE_NAME,
        GC_IS_DM,
        GC_ATTACHMENTS,
        GC_DRIVE_ATTACHMENTS,
    }

    assert quoted_keys.isdisjoint(claim_keys)
    assert quoted_keys.isdisjoint(RESERVED_ENTRY_KEYS)


def test_the_resolved_context_is_json_plain():
    """It is persisted on the entry, so every value has to survive a round trip
    through the store."""
    reader = Reader({QUOTED: {"attachment": [_uploaded("table.png")]}})
    context = resolve_reply_context(_event(quotedMessageMetadata=_quoted_metadata()), reader)

    assert context is not None
    round_tripped = json.loads(json.dumps({"reply_to": context.reply_to, **context.meta}))
    assert round_tripped["reply_to"]["sender"] == "Osprey"
    assert round_tripped[GC_QUOTED_ATTACHMENTS][0]["content_name"] == "table.png"
