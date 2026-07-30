"""resolve_reply_context: Talk's inline quote, and the refs that outlive the wire event.

Hermetic and I/O-free by construction — Talk ships the quoted message as a
``parent`` object on the event itself, so there is nothing to fetch and no client
in these tests at all.

Two properties carry the weight:

* **it never raises.** Reply context is enrichment; the engine dispatches the
  question whether or not a quote resolved, so a malformed ``parent`` must cost
  the user their quote and nothing more.
* **the quoted file refs are persisted under their own key.** After a restart the
  retry drain has only the persisted entry, and ``download_inputs`` reports
  own-message and quoted attachments to the agent differently — so the quoted
  bucket must be present, correctly shaped, and impossible to confuse with
  ``parse_event``'s own-message bucket.
"""

import json

import httpx
import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS, InboundEvent
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, RoomDirectory, TalkClient
from osprey.bridges.nextcloud_talk.events import (
    ELLIPSIS,
    MAX_QUOTED_TEXT,
    QUOTED_FILES_KEY,
    parse_event,
    resolve_reply_context,
)

BOT = "osprey-bot"
ROOM = "group1"

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account=BOT,
    app_password="app-pw",
    rooms=(ROOM,),
)

BOT_MENTION = {"type": "user", "id": BOT, "name": "OSPREY Bot"}

FILE_PARAM = {
    "type": "file",
    "id": "4711",
    "name": "orbit.png",
    "path": "Talk/orbit.png",
    "size": "20481",
    "mimetype": "image/png",
}


def _parent(**overrides):
    """A plausible Talk `parent`: the message being quoted."""
    parent = {
        "id": 41,
        "actorType": "users",
        "actorId": "bob",
        "actorDisplayName": "Bob",
        "message": "the orbit looks off after the last correction",
        "messageParameters": [],
        "systemMessage": "",
        "messageType": "comment",
    }
    parent.update(overrides)
    return parent


def _event(parent=None, **overrides):
    """A parsed InboundEvent whose raw payload optionally quotes ``parent``.

    Built by running the real :func:`parse_event`, so what the reply resolver
    reads is exactly what the parser produces — the two halves cannot drift apart
    in these tests.
    """
    raw = {
        "id": 42,
        "token": ROOM,
        "actorType": "users",
        "actorId": "alice",
        "actorDisplayName": "Alice",
        "message": "{mention-user1} why?",
        "messageParameters": {"mention-user1": dict(BOT_MENTION)},
        "systemMessage": "",
        "messageType": "comment",
    }
    if parent is not None:
        raw["parent"] = parent
    raw.update(overrides)

    def handler(request):
        return httpx.Response(
            200,
            json={
                "ocs": {
                    "meta": {"status": "ok", "statuscode": 200},
                    "data": {"type": 2, "displayName": ROOM},
                }
            },
        )

    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(handler)))
    rooms = RoomDirectory(client, sleep=lambda _: None)
    assert rooms.resolve_once(ROOM) is not None
    event = parse_event(raw, ROOM, CFG, rooms)
    assert event is not None
    return event


# ==========================================================================
# A resolved quote
# ==========================================================================


def test_a_reply_resolves_to_sender_and_text():
    context = resolve_reply_context(_event(_parent()))

    assert context is not None
    assert context.reply_to == {
        "sender": "Bob",
        "text": "the orbit looks off after the last correction",
    }


def test_reply_to_carries_exactly_the_two_contract_keys():
    """``{"sender", "text"}`` is the payload shape the worker already consumes."""
    context = resolve_reply_context(_event(_parent()))

    assert context is not None
    assert set(context.reply_to) == {"sender", "text"}


def test_the_sender_falls_back_to_the_actor_id():
    context = resolve_reply_context(_event(_parent(actorDisplayName="")))

    assert context is not None
    assert context.reply_to["sender"] == "bob"


def test_quoting_the_bots_own_answer_resolves():
    """The common shape in practice: a user replies to the agent's answer to ask
    a follow-up."""
    context = resolve_reply_context(
        _event(
            _parent(actorId=BOT, actorDisplayName="OSPREY Bot", message="beam current is 500 mA")
        )
    )

    assert context is not None
    assert context.reply_to == {"sender": "OSPREY Bot", "text": "beam current is 500 mA"}


# ==========================================================================
# Not a reply, or a quote worth nothing
# ==========================================================================


def test_a_message_that_is_not_a_reply_returns_none():
    assert resolve_reply_context(_event()) is None


@pytest.mark.parametrize(
    "parent",
    [
        pytest.param([], id="php_empty_array"),
        pytest.param({}, id="empty_dict"),
        pytest.param(None, id="null"),
        pytest.param("nonsense", id="string"),
        pytest.param(42, id="int"),
        pytest.param([{"id": 41}], id="list"),
    ],
)
def test_a_malformed_parent_returns_none(parent):
    assert resolve_reply_context(_event(parent)) is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"systemMessage": "user_added"}, id="system_parent"),
        pytest.param({"systemMessage": "message_deleted"}, id="deleted_by_system_message"),
        pytest.param({"messageType": "comment_deleted"}, id="deleted_comment"),
        pytest.param({"messageType": "system"}, id="type_system"),
    ],
)
def test_a_system_or_deleted_parent_returns_none(overrides):
    """Quoting a join notice or a since-deleted message is not context worth
    shipping — and Talk replaces a deleted message's text with a placeholder that
    would otherwise read as if the user had written it."""
    assert resolve_reply_context(_event(_parent(**overrides))) is None


def test_a_parent_with_neither_text_nor_files_returns_none():
    assert resolve_reply_context(_event(_parent(message="   "))) is None


def test_a_parent_with_only_a_file_still_resolves():
    """The text is empty but the attachment is the whole point of the quote —
    returning None here would drop the file the user pointed at."""
    context = resolve_reply_context(
        _event(_parent(message="", messageParameters={"file0": dict(FILE_PARAM)}))
    )

    assert context is not None
    assert context.reply_to["text"] == ""
    assert context.meta[QUOTED_FILES_KEY][0]["path"] == "Talk/orbit.png"


# ==========================================================================
# Never raises — enrichment must not be able to fail a dispatch
# ==========================================================================


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"message": None}, id="null_message"),
        pytest.param({"message": 17}, id="numeric_message"),
        pytest.param({"messageParameters": "nonsense"}, id="string_params"),
        pytest.param({"messageParameters": {"file0": None}}, id="null_param"),
        pytest.param({"messageParameters": {"file0": "nonsense"}}, id="string_param"),
        pytest.param({"actorId": None}, id="null_actor"),
        pytest.param({"actorDisplayName": 5}, id="numeric_display_name"),
        pytest.param({"systemMessage": None}, id="null_system_message"),
        pytest.param({"messageType": None}, id="null_message_type"),
    ],
)
def test_malformed_parent_fields_never_raise(overrides):
    result = resolve_reply_context(_event(_parent(**overrides)))
    assert result is None or set(result.reply_to) == {"sender", "text"}


def test_a_raw_payload_that_is_not_a_mapping_returns_none():
    """An adapter-internal field, but the contract is never-raises, so a caller
    that hands over a hand-built event must not be able to break the dispatch."""
    event = InboundEvent(
        message_id=f"{ROOM}:42",
        text="why?",
        sender_id="alice",
        sender_display="Alice",
        history_key=f"nextcloud:{ROOM}",
        raw="not a mapping",
    )

    assert resolve_reply_context(event) is None


def test_a_missing_raw_returns_none():
    event = InboundEvent(
        message_id=f"{ROOM}:42",
        text="why?",
        sender_id="alice",
        sender_display="Alice",
        history_key=f"nextcloud:{ROOM}",
    )

    assert resolve_reply_context(event) is None


def test_an_exploding_parent_lookup_is_reported_as_no_quote(caplog):
    """The catch-all is the contract, not a formality: whatever goes wrong in
    here, the question still gets dispatched without its quote."""

    class Hostile(dict):
        def get(self, key, default=None):
            raise RuntimeError("boom")

    event = InboundEvent(
        message_id=f"{ROOM}:42",
        text="why?",
        sender_id="alice",
        sender_display="Alice",
        history_key=f"nextcloud:{ROOM}",
        raw=Hostile(),
    )

    with caplog.at_level("WARNING", logger="osprey.bridges.nextcloud_talk.events"):
        assert resolve_reply_context(event) is None

    assert "could not resolve reply context" in caplog.text


# ==========================================================================
# Truncation
# ==========================================================================


def test_a_long_quote_is_truncated_to_the_limit():
    context = resolve_reply_context(_event(_parent(message="x" * (MAX_QUOTED_TEXT * 3))))

    assert context is not None
    text = context.reply_to["text"]
    # The marker counts against the cap, so the limit is a hard bound.
    assert len(text) == MAX_QUOTED_TEXT
    assert text.endswith(ELLIPSIS)


def test_a_quote_exactly_at_the_limit_is_untouched():
    body = "y" * MAX_QUOTED_TEXT

    context = resolve_reply_context(_event(_parent(message=body)))

    assert context is not None
    assert context.reply_to["text"] == body
    assert ELLIPSIS not in context.reply_to["text"]


def test_a_short_quote_gains_no_marker():
    context = resolve_reply_context(_event(_parent()))

    assert context is not None
    assert ELLIPSIS not in context.reply_to["text"]


# ==========================================================================
# Quoted file refs — download_inputs reads these from the persisted entry
# ==========================================================================


def test_quoted_file_ref_shape_matches_the_own_file_shape():
    """download_inputs fetches both buckets the same way, so the two ref shapes
    must be identical — only the entry key they ride under differs."""
    event = _event(_parent(messageParameters={"file0": dict(FILE_PARAM)}))
    context = resolve_reply_context(event)

    assert context is not None
    quoted = context.meta[QUOTED_FILES_KEY]
    assert quoted == [
        {
            "id": "4711",
            "name": "orbit.png",
            "path": "Talk/orbit.png",
            "size": 20481,
            "mimetype": "image/png",
        }
    ]

    own = parse_event(
        dict(event.raw, messageParameters={"mention-user1": dict(BOT_MENTION), "f": FILE_PARAM}),
        ROOM,
        CFG,
        _rooms_of(event),
    )
    assert own is not None
    assert set(own.claim_meta["nc_files"][0]) == set(quoted[0])


def _rooms_of(event):
    """A resolved directory for the room ``event`` came from."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "ocs": {
                    "meta": {"status": "ok", "statuscode": 200},
                    "data": {"type": 2, "displayName": ROOM},
                }
            },
        )

    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(handler)))
    rooms = RoomDirectory(client, sleep=lambda _: None)
    assert rooms.resolve_once(event.claim_meta["nc_room"]) is not None
    return rooms


def test_the_quoted_bucket_is_always_present_when_a_context_resolves():
    """One rule for download_inputs: key absent means there was no quote at all,
    key present means this is the list — empty included."""
    context = resolve_reply_context(_event(_parent()))

    assert context is not None
    assert context.meta[QUOTED_FILES_KEY] == []


def test_multiple_quoted_files_keep_their_order():
    context = resolve_reply_context(
        _event(
            _parent(
                messageParameters={
                    "file0": dict(FILE_PARAM, id="1", name="a.png", path="Talk/a.png"),
                    "file1": dict(FILE_PARAM, id="2", name="b.png", path="Talk/b.png"),
                }
            )
        )
    )

    assert context is not None
    assert [ref["path"] for ref in context.meta[QUOTED_FILES_KEY]] == ["Talk/a.png", "Talk/b.png"]


def test_a_quoted_file_without_a_dav_path_is_dropped():
    context = resolve_reply_context(
        _event(
            _parent(
                messageParameters={
                    "file0": {"type": "file", "id": "9", "name": "ghost.png"},
                    "file1": dict(FILE_PARAM),
                }
            )
        )
    )

    assert context is not None
    assert [ref["path"] for ref in context.meta[QUOTED_FILES_KEY]] == ["Talk/orbit.png"]


def test_quoted_meta_is_json_plain():
    context = resolve_reply_context(_event(_parent(messageParameters={"file0": dict(FILE_PARAM)})))

    assert context is not None
    assert json.loads(json.dumps(dict(context.meta))) == dict(context.meta)


# ==========================================================================
# Key disjointness — the assertion task 2.5 depends on
# ==========================================================================


def test_quoted_meta_keys_are_prefixed_and_disjoint_from_the_engines():
    context = resolve_reply_context(_event(_parent(messageParameters={"file0": dict(FILE_PARAM)})))

    assert context is not None
    keys = set(context.meta)
    assert keys and all(key.startswith("nc_") for key in keys)
    assert keys.isdisjoint(RESERVED_ENTRY_KEYS)


def test_quoted_meta_does_not_collide_with_the_own_message_claim_meta():
    """Both namespaces land in the SAME flat entry dict, so a shared key would be
    a silent overwrite — and would make own-message and quoted attachments
    indistinguishable to download_inputs."""
    event = _event(_parent(messageParameters={"file0": dict(FILE_PARAM)}))
    context = resolve_reply_context(event)

    assert context is not None
    assert set(context.meta).isdisjoint(set(event.claim_meta))
    # Specifically the two file buckets, which is the pair that would silently
    # merge the two provenances.
    assert QUOTED_FILES_KEY != "nc_files"


def test_the_whole_entry_namespace_is_collision_free():
    """The union of what both members stamp, checked as one set the way the
    pipeline sees it."""
    event = _event(_parent(messageParameters={"file0": dict(FILE_PARAM)}))
    context = resolve_reply_context(event)

    assert context is not None
    claim, meta = set(event.claim_meta), set(context.meta)
    assert len(claim | meta) == len(claim) + len(meta)
    assert (claim | meta).isdisjoint(RESERVED_ENTRY_KEYS)


# ==========================================================================
# Rendering inside a quote
# ==========================================================================


def test_a_mention_inside_the_quote_is_rendered_not_stripped():
    """In the quoted message the mention is content — part of what was actually
    said — so rewriting it would misrepresent the quote."""
    context = resolve_reply_context(
        _event(
            _parent(
                message="{mention-user1} please look at the orbit",
                messageParameters={"mention-user1": dict(BOT_MENTION)},
            )
        )
    )

    assert context is not None
    assert context.reply_to["text"] == "OSPREY Bot please look at the orbit"


def test_a_quoted_file_placeholder_renders_to_its_name():
    context = resolve_reply_context(
        _event(
            _parent(
                message="see {file0}",
                messageParameters={"file0": dict(FILE_PARAM)},
            )
        )
    )

    assert context is not None
    assert context.reply_to["text"] == "see orbit.png"


def test_no_request_is_ever_made_resolving_a_quote():
    """Talk ships the quote inline, so resolution is pure parsing."""
    requests: list[httpx.Request] = []

    def handler(request):
        requests.append(request)
        raise AssertionError("resolve_reply_context must not perform I/O")

    event = InboundEvent(
        message_id=f"{ROOM}:42",
        text="why?",
        sender_id="alice",
        sender_display="Alice",
        history_key=f"nextcloud:{ROOM}",
        claim_meta={"nc_room": ROOM},
        raw={"id": 42, "parent": _parent(messageParameters={"file0": dict(FILE_PARAM)})},
    )
    # A client exists and would record any call; nothing here should reach it.
    TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(handler)))

    context = resolve_reply_context(event)

    assert context is not None
    assert requests == []
