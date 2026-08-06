"""The durable in-repo pin of the Workspace add-on envelope contract.

Google publishes the add-on envelope to Pub/Sub; nothing in this repo can ask
Google for its shape, so the acceptance harness that drives live round-trips
fabricates it by hand — one function, one dict literal, the single synthetic
surface in an otherwise all-real harness. Every other object that harness feeds
the bridge is a genuine resource read back from a live API call.

That fabrication is what this file pins. :func:`_addon_event` below restates the
harness's builder verbatim, and each test asserts the real :func:`parse_event`
accepts what it produces. The harness cannot be imported — it lives in the
operator's profile repo, not here — so the restatement *is* the contract: if
Google's published shape moves, or the harness's builder is corrected to follow
it, this file is what fails and says so, instead of a live acceptance case
timing out with no diagnostic after a hard-capped provider budget has been spent.

Deliberately not a second copy of ``test_parse_event.py``. That suite states the
parser's behaviour against every wire shape it must survive; this one states one
narrow thing — that the exact bytes the acceptance harness publishes are bytes
this parser understands — and each test here is a property some live acceptance
case depends on. The typed-payload-space guard is restated along with the
builder, because it is precisely what makes the name-only hazard below a
message-side hazard rather than a payload-side one.
"""

import pytest

from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import (
    GC_IS_DM,
    GC_MESSAGE_NAME,
    GC_SPACE,
    GC_THREAD,
    parse_event,
)

APP_ID = "users/999"
CFG = GoogleChatBridgeConfig(app_id=APP_ID)

SPACE_NAME = "spaces/AAAA"
THREAD_NAME = "spaces/AAAA/threads/DDDD"
MESSAGE_NAME = "spaces/AAAA/messages/BBBB.CCCC"
CREATE_TIME = "2026-08-02T12:00:00.123456Z"

SENDER = {"name": "users/111", "displayName": "Alice", "type": "HUMAN"}

DM_SPACE = {"name": SPACE_NAME, "spaceType": "DIRECT_MESSAGE"}
ROOM_SPACE = {"name": SPACE_NAME, "spaceType": "SPACE", "displayName": "Osprey acceptance"}

MENTION_ANNOTATION = {
    "type": "USER_MENTION",
    "startIndex": 0,
    "length": 7,
    "userMention": {
        "user": {"name": APP_ID, "displayName": "Osprey", "type": "BOT"},
        "type": "MENTION",
    },
}

QUESTION = "what is the ring current?"


def _addon_event(message, space):
    """The add-on envelope, in the exact shape the acceptance harness builds it.

    Restated rather than imported: the harness is forbidden from depending on
    this package (it is the parity oracle for the rewrite that produced it), and
    it lives in a different repository besides. Keep this body identical to the
    harness's ``build_addon_event`` — that correspondence is the whole contract,
    and a divergence here is a green suite pinning an envelope nobody publishes.

    ``chat.user`` and ``chat.eventTime`` are *derived* from the message rather
    than invented, exactly as the harness derives them, so they agree with the
    genuine resource they wrap.
    """
    if not (space.get("spaceType") or space.get("type")):
        raise RuntimeError(
            f"payload space {space!r} carries no spaceType/type — the harness rejects "
            "an untyped payload space rather than publishing one"
        )
    return {
        "commonEventObject": {"hostApp": "CHAT", "platform": "WEB"},
        "chat": {
            "user": dict(message.get("sender") or {}),
            "eventTime": message.get("createTime", ""),
            "messagePayload": {"message": dict(message), "space": dict(space)},
        },
    }


def _captured_message(**overrides):
    """A Message resource shaped like ``spaces.messages.get`` returns one.

    Stands in for what the harness hands its envelope builder at run time: a
    genuine resource read back from Chat, carrying a real ``name``, ``sender``,
    ``thread``, ``createTime`` and ``space``.
    """
    message = {
        "name": MESSAGE_NAME,
        "sender": dict(SENDER),
        "createTime": CREATE_TIME,
        "text": QUESTION,
        "thread": {"name": THREAD_NAME},
        "space": dict(DM_SPACE),
    }
    message.update(overrides)
    return message


def _room_message(**overrides):
    """A captured room message that @mentions the app, as a live case posts one."""
    return _captured_message(
        space=dict(ROOM_SPACE),
        text=f"@Osprey {QUESTION}",
        annotations=[MENTION_ANNOTATION],
        **overrides,
    )


def test_a_direct_message_in_the_harness_envelope_parses_with_every_expected_field():
    """A DM carries no @mention and must parse on space type alone.

    The whole chain in one assertion block: the harness's key path is one
    ``_extract_message`` understands, and every field a live acceptance case
    asserts on survives the round trip off the *captured* resource rather than
    being invented by the envelope.
    """
    parsed = parse_event(_addon_event(_captured_message(), DM_SPACE), CFG)

    assert parsed is not None
    assert parsed.message_id == MESSAGE_NAME
    assert parsed.sender_id == "users/111"
    assert parsed.sender_display == "Alice"
    assert parsed.text == QUESTION
    assert parsed.claim_meta[GC_MESSAGE_NAME] == MESSAGE_NAME
    assert parsed.claim_meta[GC_SPACE] == SPACE_NAME
    assert parsed.claim_meta[GC_THREAD] == THREAD_NAME
    assert parsed.claim_meta[GC_IS_DM] is True
    # Each DM message technically starts its own thread, so keying history by
    # thread would never link two consecutive turns: the space is the scope.
    assert parsed.history_key == SPACE_NAME


def test_the_payload_space_supplies_the_type_when_the_captured_message_has_none():
    """A capture with no ``space`` key at all is completed from the payload.

    The graft is conditional, and this is the branch where the condition holds.
    The harness always publishes a typed payload space alongside the message, so
    this branch is live for every capture whose ``space`` Google omitted.
    """
    message = _captured_message()
    del message["space"]

    parsed = parse_event(_addon_event(message, DM_SPACE), CFG)

    assert parsed is not None
    assert parsed.claim_meta[GC_IS_DM] is True
    assert parsed.claim_meta[GC_SPACE] == SPACE_NAME


def test_a_direct_message_is_dropped_when_its_captured_space_carries_only_a_name():
    """HAZARD PIN: a name-only ``message.space`` defeats DM detection entirely.

    The payload space is grafted in only when the message has no ``space`` key,
    so a capture whose ``space`` is ``{"name": ...}`` — present, but with no
    ``spaceType`` — shadows the typed payload space and leaves ``is_dm`` false.
    The message then falls under the room rule, which demands an @mention a DM
    does not carry, and the event is dropped without a trace.

    Asserted here as the specified failure mode, not as an endorsement. Whether
    a live capture hits it depends on what ``spaces.messages.get`` projects into
    ``message.space`` under the operator's user OAuth, which no hermetic test can
    settle — so the harness repairs the stub at capture time and this pin gives a
    DM acceptance case that mysteriously times out a named cause to check first.
    """
    message = _captured_message(space={"name": SPACE_NAME})

    assert parse_event(_addon_event(message, DM_SPACE), CFG) is None


def test_a_room_message_parses_and_arrives_with_its_mention_stripped():
    """In a room, a USER_MENTION of this app is the whole of the addressing.

    Exercises the mention filter and the preferred text path — Chat's own
    ``argumentText`` — against the fabricated envelope: the configured app id is
    matched to the annotation's ``userMention.user.name``, and the text the agent
    receives is mention-free.
    """
    message = _room_message(argumentText=f" {QUESTION}")

    parsed = parse_event(_addon_event(message, ROOM_SPACE), CFG)

    assert parsed is not None
    assert parsed.message_id == MESSAGE_NAME
    assert parsed.text == QUESTION
    assert parsed.claim_meta[GC_IS_DM] is False
    assert parsed.claim_meta[GC_SPACE] == SPACE_NAME
    # A room conversation is scoped to its thread, so replies stay one exchange.
    assert parsed.claim_meta[GC_THREAD] == THREAD_NAME
    assert parsed.history_key == THREAD_NAME


def test_the_offset_fallback_strips_the_mention_when_the_capture_omits_argument_text():
    """A capture without ``argumentText`` must still yield mention-free text.

    The fallback slices the annotation's ``startIndex``/``length`` span out of
    ``text``. Pinned against the envelope because which of the two paths a live
    capture takes is Google's choice, not ours.
    """
    parsed = parse_event(_addon_event(_room_message(), ROOM_SPACE), CFG)

    assert parsed is not None
    assert parsed.text == QUESTION


@pytest.mark.parametrize("configured", ["users/999", "999"])
def test_the_configured_app_id_addresses_the_app_in_either_spelling(configured):
    """``GCHAT_APP_ID`` may be pasted fully-qualified or bare, on either side.

    The harness restates this matching rule as its own rather than importing it,
    so an app id that works for its answer-matching must also work for
    addressing — otherwise a live run addresses the app it cannot then recognise
    the reply of.
    """
    event = _addon_event(_room_message(argumentText=f" {QUESTION}"), ROOM_SPACE)

    assert parse_event(event, GoogleChatBridgeConfig(app_id=configured)) is not None
