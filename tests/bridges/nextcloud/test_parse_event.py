"""parse_event: which Talk messages become questions, and what rides the claim.

Hermetic — no transport at all. ``parse_event`` is required to be I/O-free, so a
directory whose cache has been populated through the public API is all the
collaborator it needs, and the tests assert that nothing here ever needed a
client.

The properties worth stating plainly, because each is a silent failure in
production if it breaks:

* the **loop guard** — the bridge answering its own answers is an unbounded loop
  in a live room, and casing must not be able to disable it;
* the **group-room mention filter** — it is the access control, so ``@all`` and a
  mention of somebody else must not summon the agent, and the literal text
  ``@osprey-bot`` must not either;
* **claim_meta completeness** — ``download_inputs`` runs off the persisted entry
  long after the wire event is gone, so a file the entry cannot address is a file
  the agent silently never sees;
* **an unresolved room raises rather than returning None** — ``None`` would drop
  the message with no claim taken and the cursor already past it.
"""

import httpx
import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS
from osprey.bridges.nextcloud_talk import (
    NextcloudBridgeConfig,
    RoomDirectory,
    RoomTypeUnresolved,
    TalkClient,
)
from osprey.bridges.nextcloud_talk.client import ROOM_TYPE_ONE_TO_ONE
from osprey.bridges.nextcloud_talk.events import parse_event

BOT = "osprey-bot"

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account=BOT,
    app_password="app-pw",
    rooms=("group1", "direct1"),
)

GROUP = "group1"
DIRECT = "direct1"

BOT_MENTION = {"type": "user", "id": BOT, "name": "OSPREY Bot"}
OTHER_MENTION = {"type": "user", "id": "bob", "name": "Bob"}
ALL_MENTION = {"type": "call", "id": "group1", "name": "Ops room"}

FILE_PARAM = {
    "type": "file",
    "id": "4711",
    "name": "orbit.png",
    "path": "Talk/orbit.png",
    "size": "20481",
    "mimetype": "image/png",
    "preview-available": "yes",
}


def _rooms(**types):
    """A RoomDirectory pre-populated through its public API.

    Types are declared per room token; a room left out stays unresolved, which is
    how the "caller bypassed the hold" case is set up. The transport asserts the
    point of the whole fixture: once population is done, parse_event must not
    cause another request.
    """
    resolved: dict[str, int] = dict(types)
    served: list[str] = []

    def handler(request):
        token = request.url.path.rsplit("/", 1)[-1]
        served.append(token)
        return httpx.Response(
            200,
            json={
                "ocs": {
                    "meta": {"status": "ok", "statuscode": 200},
                    "data": {"type": resolved[token], "displayName": token},
                }
            },
        )

    client = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(handler)))
    directory = RoomDirectory(client, sleep=lambda _: None)
    for token in resolved:
        assert directory.resolve_once(token) is not None
    served.clear()
    return directory, served


def _msg(**overrides):
    """A plausible Talk comment: a group-room question that mentions the bot."""
    message = {
        "id": 42,
        "token": GROUP,
        "actorType": "users",
        "actorId": "alice",
        "actorDisplayName": "Alice",
        "message": "{mention-user1} what is the beam current?",
        "messageParameters": {"mention-user1": dict(BOT_MENTION)},
        "systemMessage": "",
        "messageType": "comment",
    }
    message.update(overrides)
    return message


def _parse(raw, room=GROUP, rooms=None, cfg=CFG):
    directory = rooms
    if directory is None:
        directory, _ = _rooms(**{GROUP: 2, DIRECT: ROOM_TYPE_ONE_TO_ONE})
    return parse_event(raw, room, cfg, directory)


# ==========================================================================
# The happy path, and the shape of what it produces
# ==========================================================================


def test_group_room_mention_is_accepted_and_the_token_stripped():
    event = _parse(_msg())

    assert event is not None
    # The mention is addressing, not content: it is gone, and no leading space
    # survives it.
    assert event.text == "what is the beam current?"
    assert event.message_id == f"{GROUP}:42"
    assert event.sender_id == "alice"
    assert event.sender_display == "Alice"
    assert event.history_key == f"nextcloud:{GROUP}"


def test_one_to_one_needs_no_mention():
    event = _parse(
        _msg(token=DIRECT, message="what is the beam current?", messageParameters=[]),
        room=DIRECT,
    )

    assert event is not None
    assert event.text == "what is the beam current?"
    assert event.message_id == f"{DIRECT}:42"
    assert event.history_key == f"nextcloud:{DIRECT}"


def test_one_to_one_still_strips_a_mention_that_is_present():
    event = _parse(_msg(token=DIRECT), room=DIRECT)

    assert event is not None
    assert event.text == "what is the beam current?"


def test_message_id_is_room_scoped():
    """Talk ids are unique only within a room, so the same id in two rooms must
    not collide into one dedup claim."""
    directory, _ = _rooms(**{GROUP: 2, DIRECT: ROOM_TYPE_ONE_TO_ONE})

    in_group = _parse(_msg(id=7), room=GROUP, rooms=directory)
    in_direct = _parse(_msg(id=7, messageParameters=[]), room=DIRECT, rooms=directory)

    assert in_group is not None and in_direct is not None
    assert in_group.message_id == f"{GROUP}:7"
    assert in_direct.message_id == f"{DIRECT}:7"
    assert in_group.message_id != in_direct.message_id


def test_raw_event_is_carried_for_the_reply_context():
    raw = _msg()

    event = _parse(raw)

    assert event is not None
    # resolve_reply_context reads Talk's inline `parent` off this instead of
    # making a call.
    assert event.raw is raw


def test_sender_display_falls_back_to_the_actor_id():
    event = _parse(_msg(actorDisplayName=""))

    assert event is not None
    assert event.sender_display == "alice"


# ==========================================================================
# The loop guard
# ==========================================================================


def test_the_bots_own_message_is_ignored():
    assert _parse(_msg(actorId=BOT, actorDisplayName="OSPREY Bot")) is None


@pytest.mark.parametrize("spelling", ["OSPREY-BOT", "Osprey-Bot", "osprey-Bot"])
def test_the_loop_guard_survives_account_casing(spelling):
    """Nextcloud logins are case-insensitive while its API echoes canonical
    casing, so a config that differs only in case must not turn the loop guard
    off — that would have the bridge answering its own answers forever."""
    cfg = NextcloudBridgeConfig(
        base_url=CFG.base_url,
        bot_account=spelling,
        app_password="app-pw",
        rooms=CFG.rooms,
    )

    assert _parse(_msg(actorId=BOT), cfg=cfg) is None


def test_a_mention_matches_across_account_casing():
    cfg = NextcloudBridgeConfig(
        base_url=CFG.base_url,
        bot_account="OSPREY-Bot",
        app_password="app-pw",
        rooms=CFG.rooms,
    )

    event = _parse(_msg(), cfg=cfg)

    assert event is not None
    assert event.text == "what is the beam current?"


# ==========================================================================
# The group-room mention filter — this is the access control
# ==========================================================================


def test_group_room_message_without_a_mention_is_ignored():
    assert _parse(_msg(message="what is the beam current?", messageParameters=[])) is None


def test_group_room_at_all_is_ignored():
    """``@all`` is a `call`-type parameter, not a `user` one: a room-wide ping
    must not summon the agent for everybody in the room."""
    assert (
        _parse(
            _msg(
                message="{mention-call1} standup in five",
                messageParameters={"mention-call1": dict(ALL_MENTION)},
            )
        )
        is None
    )


def test_group_room_mention_of_another_user_is_ignored():
    assert (
        _parse(
            _msg(
                message="{mention-user1} can you check the orbit?",
                messageParameters={"mention-user1": dict(OTHER_MENTION)},
            )
        )
        is None
    )


def test_a_mention_of_the_bot_beside_another_user_is_accepted():
    event = _parse(
        _msg(
            message="{mention-user1} {mention-user2} please compare notes",
            messageParameters={
                "mention-user1": dict(OTHER_MENTION),
                "mention-user2": dict(BOT_MENTION),
            },
        )
    )

    assert event is not None
    # Only the bot's own token is removed; the other user renders as Talk would
    # display them, so the question still reads as written.
    assert event.text == "Bob  please compare notes"


@pytest.mark.parametrize(
    "param",
    [
        pytest.param({"type": "call", "id": BOT, "name": "Ops"}, id="call_type"),
        pytest.param({"type": "group", "id": BOT, "name": "Ops"}, id="group_type"),
        pytest.param({"type": "guest", "id": BOT, "name": "Ops"}, id="guest_type"),
        pytest.param({"id": BOT, "name": "Ops"}, id="no_type"),
    ],
)
def test_only_a_user_type_parameter_counts_as_a_mention(param):
    """The filter matches on the parsed parameter, not on the id alone — a
    non-`user` parameter that happens to carry the bot's id is not a mention."""
    assert _parse(_msg(messageParameters={"mention-user1": param})) is None


def test_the_literal_bot_name_in_text_is_not_a_mention():
    """Matching on messageParameters rather than on the text is what makes this
    exact: typing the bot's handle without Talk's mention picker does not
    address it."""
    assert _parse(_msg(message="@osprey-bot what is the beam current?", messageParameters=[])) is (
        None
    )


# ==========================================================================
# System events, non-comments, and empty questions
# ==========================================================================


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"systemMessage": "user_added"}, id="system_message"),
        pytest.param({"systemMessage": "call_started"}, id="call_started"),
        pytest.param({"messageType": "system"}, id="type_system"),
        pytest.param({"messageType": "comment_deleted"}, id="type_comment_deleted"),
        pytest.param({"messageType": "voice-message"}, id="type_voice_message"),
    ],
)
def test_non_comment_messages_are_ignored(overrides):
    assert _parse(_msg(**overrides)) is None


def test_a_missing_message_type_is_treated_as_a_comment():
    """Dropping a real question over an absent field would be a silent failure,
    and systemMessage has already excluded the system events."""
    raw = _msg()
    del raw["messageType"]

    assert _parse(raw) is not None


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("\n\t ", id="newlines"),
    ],
)
def test_empty_text_is_ignored(message):
    assert _parse(_msg(message=message, messageParameters=[])) is None


def test_a_bare_mention_with_no_question_is_ignored():
    """Addressing the bot and saying nothing leaves no question to dispatch."""
    assert _parse(_msg(message="{mention-user1}")) is None


def test_a_message_without_a_usable_id_is_ignored():
    """No id means no dedup key, and an unclaimable message could be answered
    again on every redelivery."""
    assert _parse(_msg(id=None)) is None


# ==========================================================================
# Never raises on malformed wire data
# ==========================================================================


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="none"),
        pytest.param("a string", id="string"),
        pytest.param(42, id="int"),
        pytest.param([], id="empty_list"),
        pytest.param([{"id": 1}], id="list_of_messages"),
        pytest.param({}, id="empty_dict"),
    ],
)
def test_junk_payloads_are_ignored_rather_than_raising(raw):
    assert _parse(raw) is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"messageParameters": []}, id="php_empty_array_params"),
        pytest.param({"messageParameters": None}, id="null_params"),
        pytest.param({"messageParameters": "nonsense"}, id="string_params"),
        pytest.param({"messageParameters": {"mention-user1": None}}, id="null_param_value"),
        pytest.param({"messageParameters": {"mention-user1": "nonsense"}}, id="string_param"),
        pytest.param({"message": None}, id="null_message"),
        pytest.param({"message": 17}, id="numeric_message"),
        pytest.param({"actorId": None}, id="null_actor"),
        pytest.param({"id": "not-a-number"}, id="unparseable_id"),
        pytest.param({"actorDisplayName": None}, id="null_display_name"),
    ],
)
def test_malformed_fields_never_raise(overrides):
    # Some of these parse and some are ignored; the assertion is that none of
    # them raise, since bad wire data must never take a poll thread down.
    result = _parse(_msg(**overrides))
    assert result is None or result.message_id == f"{GROUP}:42"


def test_a_string_id_is_accepted():
    """Talk stringifies some numeric fields; an id that arrives as "42" is the
    same message as one that arrives as 42."""
    event = _parse(_msg(id="42"))

    assert event is not None
    assert event.message_id == f"{GROUP}:42"


# ==========================================================================
# An unresolved room raises — it must never be answered with None
# ==========================================================================


def test_unresolved_room_raises_rather_than_dropping_the_message():
    """Reaching this means the caller bypassed the poller-side hold. Returning
    None instead would discard the message for good: no dedup claim is taken and
    the poller's cursor has already advanced past it."""
    directory, _ = _rooms()  # nothing resolved

    with pytest.raises(RoomTypeUnresolved):
        parse_event(
            _msg(message="what is the beam current?", messageParameters=[]), GROUP, CFG, directory
        )


def test_unresolved_room_raises_even_for_a_mentioning_message():
    directory, _ = _rooms()

    with pytest.raises(RoomTypeUnresolved):
        parse_event(_msg(), GROUP, CFG, directory)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"actorId": BOT}, id="own_message"),
        pytest.param({"systemMessage": "user_added"}, id="system_message"),
        pytest.param({"message": "", "messageParameters": []}, id="empty_text"),
        pytest.param({"id": None}, id="no_id"),
    ],
)
def test_the_universal_drops_do_not_need_the_room_type(overrides):
    """The cheap, room-type-independent drops are decided first, so an
    unresolved room does not raise over messages that could never be dispatched
    anyway."""
    directory, _ = _rooms()

    assert parse_event(_msg(**overrides), GROUP, CFG, directory) is None


# ==========================================================================
# claim_meta — download_inputs and the posting members read only this
# ==========================================================================


def test_claim_meta_addressing_fields():
    event = _parse(_msg())

    assert event is not None
    meta = event.claim_meta
    assert meta["nc_room"] == GROUP
    assert meta["nc_message_id"] == 42
    assert meta["nc_actor_id"] == "alice"
    assert meta["nc_actor_type"] == "users"


def test_claim_meta_file_refs_shape():
    event = _parse(
        _msg(
            message="{mention-user1} what is wrong in {file0}?",
            messageParameters={
                "mention-user1": dict(BOT_MENTION),
                "file0": dict(FILE_PARAM),
            },
        )
    )

    assert event is not None
    # Every field download_inputs needs to re-fetch from the entry alone: path
    # for the DAV GET, name and mimetype for the worker payload, size for logs.
    assert event.claim_meta["nc_files"] == [
        {
            "id": "4711",
            "name": "orbit.png",
            "path": "Talk/orbit.png",
            "size": 20481,
            "mimetype": "image/png",
        }
    ]
    # The file placeholder renders the way Talk displays it, so the question
    # still names the attachment the agent is handed.
    assert event.text == "what is wrong in orbit.png?"


def test_multiple_file_refs_keep_their_order():
    event = _parse(
        _msg(
            messageParameters={
                "mention-user1": dict(BOT_MENTION),
                "file0": dict(FILE_PARAM, id="1", name="a.png", path="Talk/a.png"),
                "file1": dict(FILE_PARAM, id="2", name="b.png", path="Talk/b.png"),
            }
        )
    )

    assert event is not None
    assert [ref["path"] for ref in event.claim_meta["nc_files"]] == ["Talk/a.png", "Talk/b.png"]


def test_a_file_ref_without_a_dav_path_is_dropped():
    """A reference download_inputs could not fetch is worse than no reference:
    it would look like an attachment the agent should have received."""
    event = _parse(
        _msg(
            messageParameters={
                "mention-user1": dict(BOT_MENTION),
                "file0": {"type": "file", "id": "9", "name": "ghost.png"},
                "file1": dict(FILE_PARAM),
            }
        )
    )

    assert event is not None
    assert [ref["path"] for ref in event.claim_meta["nc_files"]] == ["Talk/orbit.png"]


def test_an_unparseable_size_becomes_none():
    event = _parse(
        _msg(
            messageParameters={
                "mention-user1": dict(BOT_MENTION),
                "file0": dict(FILE_PARAM, size="unknown"),
            }
        )
    )

    assert event is not None
    assert event.claim_meta["nc_files"][0]["size"] is None


def test_no_attachments_yields_an_empty_ref_list():
    event = _parse(_msg())

    assert event is not None
    assert event.claim_meta["nc_files"] == []


def test_claim_meta_keys_are_prefixed_and_disjoint_from_the_engines():
    """The adapter's meta lands in the SAME flat entry dict as the engine's
    bookkeeping, so a collision is a silent overwrite in one direction or the
    other. The nc_ prefix is what keeps the two namespaces apart."""
    event = _parse(
        _msg(messageParameters={"mention-user1": dict(BOT_MENTION), "file0": dict(FILE_PARAM)})
    )

    assert event is not None
    keys = set(event.claim_meta)
    assert keys and all(key.startswith("nc_") for key in keys)
    assert keys.isdisjoint(RESERVED_ENTRY_KEYS)


def test_claim_meta_is_json_plain():
    """It round-trips through the on-disk dedup store, so anything that is not
    plain JSON would fail at claim time — or worse, come back a different shape
    after a restart."""
    import json

    event = _parse(
        _msg(messageParameters={"mention-user1": dict(BOT_MENTION), "file0": dict(FILE_PARAM)})
    )

    assert event is not None
    assert json.loads(json.dumps(dict(event.claim_meta))) == dict(event.claim_meta)


# ==========================================================================
# Text rendering
# ==========================================================================


def test_interior_whitespace_is_left_alone():
    """Pasted code and markdown have to survive: only the ends are stripped."""
    body = "{mention-user1} why does this fail?\n\n    for i in range(3):\n        print(i)\n"

    event = _parse(_msg(message=body))

    assert event is not None
    assert event.text == "why does this fail?\n\n    for i in range(3):\n        print(i)"


def test_an_unknown_placeholder_is_left_verbatim():
    """A brace the user typed themselves is not a Talk parameter, and deleting
    text nobody can account for is worse than showing a stray brace."""
    event = _parse(_msg(message="{mention-user1} what does {beam_current} mean?"))

    assert event is not None
    assert event.text == "what does {beam_current} mean?"


def test_a_parameter_with_no_label_is_left_verbatim():
    event = _parse(
        _msg(
            message="{mention-user1} look at {file0}",
            messageParameters={
                "mention-user1": dict(BOT_MENTION),
                "file0": {"type": "file", "id": "9", "path": "Talk/x.png"},
            },
        )
    )

    assert event is not None
    assert event.text == "look at {file0}"


def test_a_repeated_mention_token_is_fully_stripped():
    event = _parse(_msg(message="{mention-user1} hello {mention-user1}"))

    assert event is not None
    assert event.text == "hello"


# ==========================================================================
# No network I/O — parse_event runs before the claim, on every redelivery
# ==========================================================================


def test_parsing_issues_no_requests():
    directory, served = _rooms(**{GROUP: 2, DIRECT: ROOM_TYPE_ONE_TO_ONE})

    for _ in range(50):
        parse_event(_msg(), GROUP, CFG, directory)
        parse_event(_msg(actorId=BOT), GROUP, CFG, directory)
        parse_event(_msg(token=DIRECT, messageParameters=[]), DIRECT, CFG, directory)

    assert served == []
