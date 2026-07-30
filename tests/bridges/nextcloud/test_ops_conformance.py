"""The assembled class against the ``ChannelOps`` seam: does the engine's contract hold?

The individual members have their own suites (posting, downloads, delivery, parsing).
What is asserted here is everything that only becomes true once the ten members sit on
ONE class the engine can hold one instance of:

* **conformance** — every member present, callable, and signature-compatible. The runtime
  ``isinstance`` check only looks for the names; the full signature check is
  ``ops._static_conformance``, which mypy verifies and which is therefore covered by the
  repo's type-check gate rather than by an assertion here.
* **the two inbound members are wired at all.** They exist as module functions in
  :mod:`~osprey.bridges.nextcloud_talk.events` and the class delegates to them, so a
  missing delegation would leave the adapter silently ignoring every message — the engine
  calls ``ops.parse_event``, never the module function.
* **the room ``parse_event`` cannot be given.** The seam passes one argument, so the room
  is read from the message's own ``token``. A message that names none must be ignored, and
  ``RoomTypeUnresolved`` must still escape — returning ``None`` for it would discard a
  message with the poll cursor already past it.
* **entry-key disjointness.** ``claim_meta`` and ``ReplyContext.meta`` land in the same
  flat persisted entry as the engine's bookkeeping. A collision is a silent overwrite in
  one direction or the other, which is why the keys are asserted against
  ``RESERVED_ENTRY_KEYS`` here as well as at their source.
* **no per-dispatch instance state.** The engine shares ONE instance across every room
  thread and the drain thread.
"""

import inspect

import httpx
import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS, ChannelOps, InboundEvent, ReplyContext
from osprey.bridges.nextcloud_talk import (
    NextcloudBridgeConfig,
    RoomDirectory,
    RoomTypeUnresolved,
    TalkClient,
    events,
)
from osprey.bridges.nextcloud_talk.client import ROOM_TYPE_ONE_TO_ONE
from osprey.bridges.nextcloud_talk.ops import (
    NC_FILES,
    NC_MESSAGE_ID,
    NC_ROOM,
    NextcloudTalkOps,
    _static_conformance,
)
from tests.bridges.test_ports import PROTOCOL_MEMBERS

BOT = "osprey-bot"
GROUP = "group1"
DIRECT = "direct1"
UNRESOLVED = "never-fetched"
ROOM_TYPE_GROUP = 2

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account=BOT,
    app_password="app-pw",
    rooms=(GROUP, DIRECT),
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


def _directory():
    """``(directory, served)`` — a RoomDirectory populated through its public API.

    ``GROUP`` and ``DIRECT`` resolve; :data:`UNRESOLVED` deliberately does not, which is
    how "a caller reached past the poller-side hold" is set up. ``served`` is cleared
    after population, so it records only requests ``parse_event`` itself caused — the
    member is required to be I/O-free.
    """
    resolved = {GROUP: ROOM_TYPE_GROUP, DIRECT: ROOM_TYPE_ONE_TO_ONE}
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


def _ops(rooms=None):
    """A fully injected ops instance. No member here reaches the network."""
    unreachable = httpx.MockTransport(lambda request: httpx.Response(502))
    return NextcloudTalkOps(
        CFG,
        TalkClient(CFG, client=httpx.Client(transport=unreachable)),
        httpx.Client(transport=unreachable),
        rooms if rooms is not None else _directory()[0],
    )


def _msg(**overrides):
    """A plausible Talk comment: a group-room question that mentions the bot."""
    message = {
        "id": 42,
        "token": GROUP,
        "actorType": "users",
        "actorId": "alice",
        "actorDisplayName": "Alice",
        "message": "{mention-user1} what is the beam current?",
        "messageParameters": {"mention-user1": dict(BOT_MENTION), "file0": dict(FILE_PARAM)},
        "systemMessage": "",
        "messageType": "comment",
    }
    message.update(overrides)
    return message


def _parent(**overrides):
    """A plausible Talk ``parent``: the message being quoted."""
    parent = {
        "id": 41,
        "actorType": "users",
        "actorId": "bob",
        "actorDisplayName": "Bob",
        "message": "the orbit looks off",
        "messageParameters": {"file0": dict(FILE_PARAM)},
        "systemMessage": "",
        "messageType": "comment",
    }
    parent.update(overrides)
    return parent


# ==========================================================================
# Protocol conformance
# ==========================================================================


def test_the_adapter_satisfies_the_channel_ops_protocol():
    # ChannelOps is @runtime_checkable, so this is a real assertion and not a tautology
    # about the class body — but it verifies member NAMES only. Signatures are checked
    # statically by ops._static_conformance under mypy, and structurally below.
    assert isinstance(_ops(), ChannelOps)


def test_every_protocol_member_is_present_and_callable():
    # The canonical member list is imported from the core seam's own suite rather than
    # respelled, so a member added to the protocol fails here instead of being forgotten.
    ops = _ops()
    for name in PROTOCOL_MEMBERS:
        assert callable(getattr(ops, name, None)), name


def test_every_member_signature_matches_the_protocol():
    """Parameter names, order, and count, member by member.

    ``isinstance`` cannot see any of this, and a mismatch is not a type error the engine
    would notice until it called the member in production — the retry drain's members in
    particular may not run for hours.
    """
    for name in PROTOCOL_MEMBERS:
        expected = inspect.signature(getattr(ChannelOps, name))
        actual = inspect.signature(getattr(NextcloudTalkOps, name))
        assert list(actual.parameters) == list(expected.parameters), name


def test_static_conformance_helper_is_the_type_check_seam():
    """Present, and an identity — the assertion mypy makes is the point of it.

    Guards against the helper being deleted as apparently-unused code: without it nothing
    would check the full signatures at all, since the runtime protocol check does not.
    """
    ops = _ops()
    assert _static_conformance(ops) is ops


# ==========================================================================
# Constructor: explicit, injectable collaborators
# ==========================================================================


def test_injected_collaborators_are_held_as_given():
    rooms, _ = _directory()
    talk = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(lambda r: None)))
    worker = httpx.Client(transport=httpx.MockTransport(lambda r: None))

    ops = NextcloudTalkOps(CFG, talk, worker, rooms)

    assert ops._cfg is CFG
    assert ops._client is talk
    assert ops._http is worker
    assert ops._rooms is rooms


def test_a_room_directory_is_built_from_the_talk_client_when_none_is_injected():
    """The default keeps the nine non-parsing members constructible in one line.

    It is NOT a production wiring: only the poller populates a directory, so a private
    one stays empty and every parse would raise. The constructor docstring says so; this
    just pins the defaulting behaviour itself.
    """
    talk = TalkClient(CFG, client=httpx.Client(transport=httpx.MockTransport(lambda r: None)))
    ops = NextcloudTalkOps(CFG, talk)

    assert isinstance(ops._rooms, RoomDirectory)
    with pytest.raises(RoomTypeUnresolved):
        ops.parse_event(_msg())


def test_no_per_dispatch_instance_state_exists():
    """The engine shares ONE instance across every room thread and the drain thread.

    So the instance dict must hold nothing but the injected collaborators: any
    per-dispatch field (a "current room", a cached entry) would be written by one thread
    and read by another, and the corruption would be intermittent and load-dependent.
    Asserted structurally, since a race cannot be asserted directly.

    The three held objects are themselves safe to share: ``RoomDirectory`` locks its own
    dict, ``NextcloudBridgeConfig`` is frozen, and both HTTP clients are
    :class:`httpx.Client`, which is documented thread-safe and pools connections across
    threads by design (that pooling is a reason to share one, not to build per-thread
    copies).
    """
    ops = _ops()
    assert set(vars(ops)) == {"_cfg", "_client", "_http", "_rooms"}

    # And a full dispatch's worth of calls adds none.
    ops.parse_event(_msg())
    ops.coalesce_key({"history_key": "nextcloud:group1", "sender_id": "alice"})
    ops.post_ack({NC_ROOM: GROUP, NC_MESSAGE_ID: 42})  # swallows the 502
    assert set(vars(ops)) == {"_cfg", "_client", "_http", "_rooms"}


# ==========================================================================
# parse_event: wired, and given a room the seam does not supply
# ==========================================================================


def test_parse_event_delegates_and_sources_the_room_from_the_message_token():
    rooms, served = _directory()
    ops = _ops(rooms)

    event = ops.parse_event(_msg())

    assert event is not None
    # Room-derived fields prove the token was found and used: both are room-scoped, and
    # neither could be produced from the message alone.
    assert event.message_id == f"{GROUP}:42"
    assert event.history_key == f"nextcloud:{GROUP}"
    assert event.claim_meta[NC_ROOM] == GROUP
    # I/O-free: the directory was consulted as a pure cache read.
    assert served == []


def test_parse_event_returns_exactly_what_the_module_function_returns():
    """Delegation, not a second parser. The two must not be able to drift apart."""
    rooms, _ = _directory()
    ops = _ops(rooms)
    raw = _msg()

    assert ops.parse_event(raw) == events.parse_event(raw, GROUP, CFG, rooms)


def test_a_one_to_one_token_routes_to_the_one_to_one_rules():
    """The token decides which room's TYPE is looked up, so it decides the mention rule.

    A message with no mention is accepted here and would be ignored in a group room —
    which is the sharpest available evidence that the token, not a default, chose the
    room.
    """
    ops = _ops()

    event = ops.parse_event(
        _msg(token=DIRECT, message="what is the beam current?", messageParameters=[])
    )

    assert event is not None
    assert event.message_id == f"{DIRECT}:42"


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(42, id="non-string-int"),
        pytest.param(["group1"], id="non-string-list"),
        pytest.param({"token": GROUP}, id="non-string-mapping"),
    ],
)
def test_a_message_with_no_usable_room_token_is_ignored(token):
    """``None``, never an exception: ``parse_event``'s contract is never to raise on bad
    input, and a message with no room could be neither claimed nor answered by any later
    attempt either."""
    ops = _ops()
    raw = _msg()
    if token is None:
        del raw["token"]
    else:
        raw["token"] = token

    assert ops.parse_event(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="none"),
        pytest.param("not a message", id="string"),
        pytest.param([], id="php-empty-array"),
        pytest.param(42, id="int"),
    ],
)
def test_a_non_message_payload_is_ignored(raw):
    """Whatever the network hands the poller reaches this member unmodified."""
    assert _ops().parse_event(raw) is None


def test_room_type_unresolved_propagates():
    """The one documented exception to never-raises, and it must NOT be swallowed.

    ``None`` here would mean "not for us": no dedup claim is taken and the poll cursor has
    already moved past the message, so it would be gone for good. Raising leaves the
    persisted offset unadvanced and the message is re-fetched once the type resolves.
    """
    ops = _ops()

    with pytest.raises(RoomTypeUnresolved):
        ops.parse_event(_msg(token=UNRESOLVED))


# ==========================================================================
# resolve_reply_context: wired
# ==========================================================================


def test_resolve_reply_context_delegates_to_the_module_function():
    ops = _ops()
    event = ops.parse_event(_msg(parent=_parent()))
    assert event is not None

    assert ops.resolve_reply_context(event) == events.resolve_reply_context(event)


def test_resolve_reply_context_resolves_a_quote_through_the_class():
    ops = _ops()
    event = ops.parse_event(_msg(parent=_parent()))
    assert event is not None

    context = ops.resolve_reply_context(event)

    assert context is not None
    assert context.reply_to["sender"] == "Bob"
    assert context.reply_to["text"] == "the orbit looks off"
    assert [ref["path"] for ref in context.meta[events.QUOTED_FILES_KEY]] == ["Talk/orbit.png"]


def test_a_message_that_quotes_nothing_has_no_reply_context():
    ops = _ops()
    event = ops.parse_event(_msg())
    assert event is not None

    assert ops.resolve_reply_context(event) is None


# ==========================================================================
# Entry keys: the adapter's namespace against the engine's
# ==========================================================================


def test_the_adapter_writes_no_key_the_engine_owns():
    """Both adapter namespaces land in the SAME flat persisted entry as the engine's
    bookkeeping, so a collision silently overwrites one side or the other — a stamped
    ``attempts`` would corrupt the drain's give-up ceiling, a persisted ``reply_to`` would
    clobber the engine's."""
    ops = _ops()
    event = ops.parse_event(_msg(parent=_parent()))
    assert event is not None
    context = ops.resolve_reply_context(event)
    assert context is not None

    claim_keys = set(event.claim_meta)
    meta_keys = set(context.meta)

    assert claim_keys.isdisjoint(RESERVED_ENTRY_KEYS)
    assert meta_keys.isdisjoint(RESERVED_ENTRY_KEYS)
    # The two adapter namespaces must not collide with each other either: they are
    # written by different members and merged into one dict.
    assert claim_keys.isdisjoint(meta_keys)
    # The ``nc_`` prefix is the whole mechanism that keeps all three true.
    assert all(key.startswith("nc_") for key in claim_keys | meta_keys)


def test_the_keys_the_outbound_members_read_are_the_keys_the_parse_members_write():
    """Reader/writer agreement across modules.

    ``ops`` reads :data:`NC_ROOM` / :data:`NC_MESSAGE_ID` / :data:`NC_FILES` from the
    entry and ``events`` writes them; a rename on one side alone would deliver nothing and
    raise nothing.
    """
    ops = _ops()
    event = ops.parse_event(_msg(parent=_parent()))
    assert event is not None
    context = ops.resolve_reply_context(event)
    assert context is not None

    assert {NC_ROOM, NC_MESSAGE_ID, NC_FILES} <= set(event.claim_meta)
    assert events.QUOTED_FILES_KEY in context.meta


def test_parsed_types_are_the_engines_own():
    """Not a duck-typed lookalike: the engine unpacks these dataclasses by field."""
    ops = _ops()
    event = ops.parse_event(_msg(parent=_parent()))

    assert isinstance(event, InboundEvent)
    assert isinstance(ops.resolve_reply_context(event), ReplyContext)
