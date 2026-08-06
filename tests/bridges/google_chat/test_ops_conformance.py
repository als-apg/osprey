"""The assembled class against the ``ChannelOps`` seam: does the engine's contract hold?

The individual members have their own suites (parsing, posting, downloads, delivery).
What is asserted here is everything that only becomes true once the ten members sit on
ONE class the engine holds one instance of:

* **conformance** — every member present, callable, and signature-compatible. The runtime
  ``isinstance`` check only looks for the names; the full signature check is
  ``ops._static_conformance``, which mypy verifies and which is therefore covered by the
  repo's type-check gate rather than by an assertion here.
* **the three delegating members are wired at all.** ``parse_event`` and
  ``resolve_reply_context`` live in :mod:`~osprey.bridges.google_chat.events` and
  ``download_inputs``' fetch in :mod:`~osprey.bridges.google_chat.inbound`; the engine
  calls the class, never the module function, so a missing delegation would leave the
  adapter silently ignoring every message or every attachment.
* **no per-dispatch instance state.** The engine shares ONE instance across the subscriber
  thread and the drain thread. The published-URL stash is the single documented exception,
  and it is asserted to be the only one — and to be bounded, since it is the one thing on
  this instance that can grow.
* **entry-key disjointness.** ``claim_meta`` and ``ReplyContext.meta`` land in the same
  flat persisted entry as the engine's bookkeeping. A collision is a silent overwrite in
  one direction or the other, which is why the keys are asserted against
  ``RESERVED_ENTRY_KEYS`` here as well as at their source.
* **reader/writer agreement.** The outbound members address Chat from keys the parse
  members wrote, across two modules; a rename on one side alone would deliver nothing and
  raise nothing, so the entry a real parse produces is what drives the posts here.

Hermetic: the Chat REST leg runs through a ``MagicMock`` discovery service, the media GET
through an ``httpx.Client`` over a ``MockTransport``, and both artifact publishers are
fakes — no network, no credential, and no Google library anywhere in this file.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import httpx

from osprey.bridges.core import (
    RESERVED_ENTRY_KEYS,
    ChannelOps,
    InboundEvent,
    InputDownload,
    ReplyContext,
)
from osprey.bridges.google_chat import client as client_module
from osprey.bridges.google_chat import events
from osprey.bridges.google_chat.client import ChatClient
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import (
    GC_ATTACHMENTS,
    GC_DRIVE_ATTACHMENTS,
    GC_MESSAGE_NAME,
    GC_QUOTED_ATTACHMENTS,
    GC_QUOTED_DRIVE_ATTACHMENTS,
    GC_SPACE,
    GC_THREAD,
)
from osprey.bridges.google_chat.gcs import publish_artifacts, publish_documents
from osprey.bridges.google_chat.inbound import download_attachments
from osprey.bridges.google_chat.ops import STASH_LIMIT, GoogleChatOps, _static_conformance
from tests.bridges.test_ports import PROTOCOL_MEMBERS

APP_ID = "users/1234567890"

CFG = GoogleChatBridgeConfig(
    sa_key="/etc/osprey/sa-key.json",
    subscription="projects/p/subscriptions/s",
    app_id=APP_ID,
    gcs_bucket="test-bucket",
)

SPACE = "spaces/AAAA"
THREAD = "spaces/AAAA/threads/TTTT"
MESSAGE = "spaces/AAAA/messages/MMMM"
QUOTED = "spaces/AAAA/messages/QQQQ"

OWN_RESOURCE = "attachments/OWN"
QUOTED_RESOURCE = "attachments/QUOTED"

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
QUOTED_PNG = b"\x89PNG\r\n\x1a\n" + b"quoted pixels"

RUN_ID = "R1"
PLOT_URL = "https://storage.googleapis.com/test-bucket/plots/1.png"

# Every attribute a constructed instance is allowed to hold, split by what each one is.
# Anything appearing here that is not a collaborator, a lock, or the documented stash is
# per-dispatch state on an instance two threads share — the failure this set exists to
# make loud, since the corruption it causes is intermittent and load-dependent.
COLLABORATORS = {"_cfg", "_client", "_publisher", "_doc_publisher", "_media"}
LOCKS = {"_media_lock", "_stash_lock"}
STASH = {"_stashed_urls"}


# --- builders ---------------------------------------------------------------


def mention() -> dict[str, Any]:
    """A ``USER_MENTION`` annotation naming this app — the access control for a room."""
    return {
        "type": "USER_MENTION",
        "startIndex": 0,
        "length": 7,
        "userMention": {"user": {"name": APP_ID, "displayName": "Osprey", "type": "BOT"}},
    }


def uploaded(resource: str = OWN_RESOURCE, name: str = "ring.png") -> dict[str, Any]:
    """An ``UPLOADED_CONTENT`` attachment entry, camelCase as REST sends one."""
    return {
        "contentName": name,
        "contentType": "image/png",
        "attachmentDataRef": {"resourceName": resource},
        "source": "UPLOADED_CONTENT",
    }


def drive(name: str = "report.pdf") -> dict[str, Any]:
    """A ``DRIVE_FILE`` attachment — no resource name the app's credential could read."""
    return {
        "contentName": name,
        "contentType": "application/pdf",
        "driveDataRef": {"driveFileId": "1a2b3c"},
        "source": "DRIVE_FILE",
    }


def chat_event(**overrides: Any) -> dict[str, Any]:
    """A classic-shape room question: mentions the app, attaches a file, quotes a reply.

    Deliberately the richest event this adapter accepts, so one parse produces every
    ``gc_`` key both parse members can write.
    """
    message: dict[str, Any] = {
        "name": MESSAGE,
        "sender": {"name": "users/9", "displayName": "Alice", "type": "HUMAN"},
        "text": "@Osprey what is the ring current?",
        "argumentText": "what is the ring current?",
        "thread": {"name": THREAD},
        "space": {"name": SPACE, "spaceType": "SPACE"},
        "annotations": [mention()],
        "attachment": [uploaded(), drive()],
        "quotedMessageMetadata": {
            "name": QUOTED,
            "quotedMessageSnapshot": {"sender": "Bob", "text": "the orbit looks off"},
        },
    }
    message.update(overrides)
    return {"type": "MESSAGE", "message": message}


def quoted_body() -> dict[str, Any]:
    """The ``spaces.messages.get`` view of the quoted message — the only view carrying
    its attachments."""
    return {
        "name": QUOTED,
        "text": "the orbit looks off",
        "attachment": [uploaded(QUOTED_RESOURCE, "orbit.png")],
    }


def make_service() -> MagicMock:
    """A Chat discovery stand-in: creates succeed, and ``messages.get`` answers the
    quoted message's own view."""
    service = MagicMock()
    messages = service.spaces.return_value.messages.return_value
    messages.create.return_value.execute.return_value = {"name": MESSAGE}
    messages.get.return_value.execute.return_value = quoted_body()
    return service


def creates(service: MagicMock) -> Any:
    """The ``messages.create`` mock, whose call list is the posted-message record."""
    return service.spaces.return_value.messages.return_value.create


def media_session() -> httpx.Client:
    """A media session serving both attachments; anything else 404s."""
    served = {OWN_RESOURCE: PNG, QUOTED_RESOURCE: QUOTED_PNG}

    def handler(request: httpx.Request) -> httpx.Response:
        resource = str(request.url.path).split("/v1/media/", 1)[-1]
        if resource not in served:
            return httpx.Response(404)
        return httpx.Response(200, content=served[resource])

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_ops(
    *, publisher: Any = None, session: httpx.Client | None = None
) -> tuple[GoogleChatOps, MagicMock]:
    """A fully injected ops instance. No member here reaches the network or GCS."""
    service = make_service()
    ops = GoogleChatOps(
        CFG,
        ChatClient(CFG, service),
        publisher if publisher is not None else MagicMock(return_value=[PLOT_URL]),
        MagicMock(return_value=[]),
        session if session is not None else media_session(),
    )
    return ops, service


def image_descriptor(artifact_id: str = "a1") -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "filename": f"{artifact_id}.png",
        "source_mime": "image/png",
        "delivered_mime": "image/png",
        "convertible": True,
    }


def completed(run_id: str = RUN_ID) -> dict[str, Any]:
    return {
        "status": "completed",
        "text_output": "42 mA",
        "run_id": run_id,
        "error": None,
        "artifacts": [image_descriptor()],
    }


def parsed(ops: GoogleChatOps) -> tuple[InboundEvent, ReplyContext]:
    """One event through both parse members, asserted non-``None`` so the callers below
    read the real thing rather than a skipped test."""
    event = ops.parse_event(chat_event())
    assert event is not None
    context = ops.resolve_reply_context(event)
    assert context is not None
    return event, context


def persisted(ops: GoogleChatOps) -> dict[str, Any]:
    """The flat entry the engine would persist for that event — claim meta and reply meta
    merged exactly as the pipeline merges them, plus the engine's own fields."""
    event, context = parsed(ops)
    return {
        "message_id": event.message_id,
        "history_key": event.history_key,
        "sender_id": event.sender_id,
        **event.claim_meta,
        **context.meta,
    }


# ==========================================================================
# Protocol conformance
# ==========================================================================


def test_the_adapter_satisfies_the_channel_ops_protocol():
    # ChannelOps is @runtime_checkable, so this is a real assertion and not a tautology
    # about the class body — but it verifies member NAMES only. Signatures are checked
    # statically by ops._static_conformance under mypy, and structurally below.
    ops, _ = make_ops()
    assert isinstance(ops, ChannelOps)


def test_every_protocol_member_is_present_and_callable():
    # The canonical member list is imported from the core seam's own suite rather than
    # respelled, so a member added to the protocol fails here instead of being forgotten.
    ops, _ = make_ops()
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
        actual = inspect.signature(getattr(GoogleChatOps, name))
        assert list(actual.parameters) == list(expected.parameters), name


def test_static_conformance_helper_is_the_type_check_seam():
    """Present, and an identity — the assertion mypy makes is the point of it.

    Guards against the helper being deleted as apparently-unused code: without it nothing
    would check the full signatures at all, since the runtime protocol check does not.
    """
    ops, _ = make_ops()
    assert _static_conformance(ops) is ops


# ==========================================================================
# Constructor: explicit, injectable collaborators
# ==========================================================================


def test_injected_collaborators_are_held_as_given():
    service = make_service()
    chat = ChatClient(CFG, service)
    publisher = MagicMock(return_value=[])
    doc_publisher = MagicMock(return_value=[])
    session = media_session()

    ops = GoogleChatOps(CFG, chat, publisher, doc_publisher, session)

    assert ops._cfg is CFG
    assert ops._client is chat
    assert ops._publisher is publisher
    assert ops._doc_publisher is doc_publisher
    assert ops._media is session


def test_the_publisher_defaults_are_the_gcs_modules_own():
    """Omitting them must not quietly wire a different publisher: these two are the whole
    artifact-delivery path, and a stand-in left in place would publish nothing while every
    test that injects its own still passed."""
    ops = GoogleChatOps(CFG, ChatClient(CFG, make_service()))

    assert ops._publisher is publish_artifacts
    assert ops._doc_publisher is publish_documents


def test_the_media_session_is_not_built_until_a_download_needs_it():
    """Building it reads a key file and imports ``google-auth``, and this class is
    constructed on paths that download nothing at all — so the default is ``None`` and the
    session appears on first use, which is also what keeps the whole suite runnable with no
    Google library installed."""
    ops = GoogleChatOps(CFG, ChatClient(CFG, make_service()))

    assert ops._media is None
    # And a text-only entry still does not build one: the member returns before it looks.
    assert ops.download_inputs({GC_SPACE: SPACE}) == InputDownload()
    assert ops._media is None


def test_a_chat_client_is_built_from_the_config_when_none_is_injected(monkeypatch):
    """The default that production actually takes. Nothing else in the suite exercises it,
    because every other test injects a client over a mock service."""
    built = MagicMock()
    factory = MagicMock(return_value=built)
    monkeypatch.setattr(client_module, "build_chat_service", factory)

    ops = GoogleChatOps(CFG)

    assert isinstance(ops._client, ChatClient)
    assert factory.call_args.args == (CFG,)


# ==========================================================================
# One shared instance: no per-dispatch state
# ==========================================================================


def test_the_stash_is_the_only_state_the_instance_holds():
    """The engine shares ONE instance across the subscriber thread and the drain thread.

    So the instance dict must hold nothing but the injected collaborators, their locks,
    and the one documented cross-call map: any per-dispatch field (a "current space", a
    cached entry) would be written by one thread and read by another, and the corruption
    would be intermittent and load-dependent. Asserted structurally, since a race cannot
    be asserted directly.

    The held objects are themselves safe to share: ``GoogleChatBridgeConfig`` is frozen,
    ``ChatClient`` serializes its own HTTP leg, the publishers are pure functions of their
    arguments, and both mutable pieces — the memoized media session and the stash — sit
    behind the two locks.
    """
    ops, _ = make_ops()
    assert set(vars(ops)) == COLLABORATORS | LOCKS | STASH


def test_a_full_dispatch_adds_no_instance_state_and_rebinds_nothing():
    """Every member the live path and the drain call, in order, against one instance.

    The attribute set must come out identical and no attribute may be *replaced* — the
    stash is mutated in place under its lock, which is the one thing that may change here.
    """
    ops, _ = make_ops()
    entry = persisted(ops)
    before = dict(vars(ops))

    ops.post_ack(entry)
    ops.download_inputs(entry)
    ops.post_answer(entry, completed())
    assert ops.deliver_files(entry, completed()) == {"a1": PLOT_URL}
    ops.coalesce_key(entry)
    ops.post_queued(entry, {"status": "failed"})
    ops.post_giveup(entry)
    ops.post_superseded(entry)

    assert set(vars(ops)) == COLLABORATORS | LOCKS | STASH
    assert all(vars(ops)[name] is value for name, value in before.items())
    # Popped, not merely read: the map is gone once the engine has taken it, so an
    # answered run leaves nothing behind on an instance that lives as long as the process.
    assert ops._stashed_urls == {}


def test_the_stash_cannot_grow_past_its_bound():
    """The runs whose ``deliver_files`` never arrives — an answer that raised after
    publishing, an interrupted process — are the only ones that linger, and on a
    process-lifetime instance they would otherwise pile up without limit."""
    ops, _ = make_ops()
    entry = persisted(ops)

    for index in range(STASH_LIMIT + 3):
        ops.post_answer(entry, completed(f"run-{index}"))

    assert len(ops._stashed_urls) == STASH_LIMIT
    # The oldest went, the newest stayed: eviction runs in the direction that matches how
    # the stash is read, since the map just written is the one about to be popped.
    assert "run-0" not in ops._stashed_urls
    assert ops._pop_urls(f"run-{STASH_LIMIT + 2}") == {"a1": PLOT_URL}


# ==========================================================================
# Delegation: the class adds arity, not a second implementation
# ==========================================================================


def test_parse_event_delegates_with_the_instances_own_config():
    """The seam passes one argument and the module function takes two, so the config is
    the class's whole contribution — and it must be the instance's, since the app id it
    carries is what decides whether a room message was addressed to this app at all."""
    ops, _ = make_ops()
    raw = chat_event()

    assert ops.parse_event(raw) == events.parse_event(raw, CFG)


def test_resolve_reply_context_delegates_through_the_injected_clients_reader():
    """Same shape, with the message reader supplied instead: the quoted message's own
    attachments are visible on no other view, so a delegation wired to some other
    transport would resolve a quote with no files and raise nothing."""
    ops, service = make_ops()
    event = ops.parse_event(chat_event())
    assert event is not None

    context = ops.resolve_reply_context(event)

    assert context == events.resolve_reply_context(event, ops._client.get_message)
    # The read went through the injected service, which is what makes the reader the
    # client's rather than a second one built somewhere in the ops layer.
    assert service.spaces.return_value.messages.return_value.get.call_args.kwargs == {
        "name": QUOTED
    }


def test_download_inputs_delegates_the_fetch_to_the_inbound_downloader():
    """The class's contribution is the entry-to-bucket read and the Drive pairing; the
    bytes are the module function's, and the two must not be able to drift apart."""
    session = media_session()
    ops, _ = make_ops(session=session)
    entry = persisted(ops)

    result = ops.download_inputs(entry)

    files, skipped = download_attachments(session, entry[GC_ATTACHMENTS])
    quoted_files, quoted_skipped = download_attachments(session, entry[GC_QUOTED_ATTACHMENTS])
    assert result.files == files
    assert result.quoted_files == quoted_files
    # The Drive names the parse step split off are the class's own addition, so they
    # follow that bucket's download skips rather than replacing them.
    assert list(result.skipped)[: len(skipped)] == skipped
    assert list(result.quoted_skipped)[: len(quoted_skipped)] == quoted_skipped
    assert [note["filename"] for note in result.skipped[len(skipped) :]] == ["report.pdf"]


# ==========================================================================
# Entry keys: the adapter's namespace against the engine's
# ==========================================================================


def test_the_adapter_writes_no_key_the_engine_owns():
    """Both adapter namespaces land in the SAME flat persisted entry as the engine's
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
    # The ``gc_`` prefix is the whole mechanism that keeps all three true.
    assert all(key.startswith("gc_") for key in claim_keys | meta_keys)


def test_the_prefix_is_needed_only_where_a_key_lands_flat():
    """The attachment references are nested under one ``gc_`` key and carry unprefixed
    keys of their own. That is not an inconsistency: nesting keeps them out of the flat
    entry entirely, so they cannot collide with anything, and prefixing them would only
    make the payload harder to read."""
    ops, _ = make_ops()
    event, _ = parsed(ops)

    references = event.claim_meta[GC_ATTACHMENTS]
    assert references and all(isinstance(ref, Mapping) for ref in references)
    assert set(references[0]) == {"content_name", "content_type", "resource_name", "source"}


def test_every_declared_entry_key_is_one_a_parse_member_writes():
    """Both directions, against the constants themselves: a key declared and never written
    is one the outbound members would read as absent forever, and a key written without a
    constant is one the reader cannot import and will eventually respell."""
    ops, _ = make_ops()
    event, context = parsed(ops)

    declared = {
        value
        for name, value in vars(events).items()
        if name.startswith("GC_") and isinstance(value, str)
    }

    assert set(event.claim_meta) | set(context.meta) == declared


def test_the_keys_the_outbound_members_read_are_the_keys_the_parse_members_write():
    """Reader/writer agreement across modules, end to end.

    ``ops`` reads the space, the thread and both attachment buckets off the persisted
    entry and ``events`` writes them; a rename on one side alone would deliver nothing and
    raise nothing, so the entry driving the post here is the one a real parse produced.
    """
    ops, service = make_ops()
    entry = persisted(ops)

    assert {GC_SPACE, GC_THREAD, GC_MESSAGE_NAME, GC_ATTACHMENTS, GC_DRIVE_ATTACHMENTS} <= set(
        entry
    )
    assert {GC_QUOTED_ATTACHMENTS, GC_QUOTED_DRIVE_ATTACHMENTS} <= set(entry)

    ops.post_ack(entry)

    posted = creates(service).call_args
    assert posted.kwargs["parent"] == SPACE
    assert posted.kwargs["body"]["thread"] == {"name": THREAD}
    # The download reached both buckets, which is only possible if both keys were read
    # under the names the parse members wrote them under.
    downloaded = ops.download_inputs(entry)
    assert [file["filename"] for file in downloaded.files] == ["ring.png"]
    assert [file["filename"] for file in downloaded.quoted_files] == ["orbit.png"]


def test_parsed_types_are_the_engines_own():
    """Not duck-typed lookalikes: the engine unpacks these dataclasses by field."""
    ops, _ = make_ops()
    event, context = parsed(ops)

    assert isinstance(event, InboundEvent)
    assert isinstance(context, ReplyContext)
    assert isinstance(ops.download_inputs(dict(event.claim_meta)), InputDownload)
    assert ops.coalesce_key({"history_key": SPACE, "sender_id": "users/9"}) == [SPACE, "users/9"]
