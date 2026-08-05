"""GoogleChatOps' inbound half: the two parse members, the file download, the group key.

The parse members own no wire knowledge — ``osprey.bridges.google_chat.events`` does, and
its own suites cover the wire shapes exhaustively. What is asserted here is the wiring
those suites cannot see: that the config threaded into ``parse_event`` is the instance's
own, and that the message reader threaded into ``resolve_reply_context`` is the injected
Chat client rather than some other transport. Both are wired once and, if wired wrongly,
fail in a way no events-level test can catch.

``download_inputs`` is where the real behavior is, and its frame is the persisted entry:
every reference is read off ``entry`` and nothing else, because by the time the retry drain
or a post-restart reconcile calls this member the wire event is gone. The properties under
the most scrutiny are the ones a live bridge loses quietly:

* the own and quoted buckets never bleed into each other — the engine reports them to the
  agent as different provenances, so a crossed bucket misattributes whose file was lost;
* a Drive attachment, which the app's credential can never fetch, is reported by name
  instead of vanishing;
* a text-only entry returns exactly ``InputDownload()``, so the dispatch payload stays
  byte-identical to the no-attachment path;
* nothing raises — not a failing GET, not a malformed entry, and not a media session that
  cannot be built at all.

Hermetic throughout: the media GET runs through an ``httpx.Client`` over a
``MockTransport`` and the Chat REST leg through a ``MagicMock`` discovery service, so no
network, no credential, and no Google library is involved anywhere in this file.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from osprey.bridges.core import InboundEvent, InputDownload
from osprey.bridges.google_chat import ops as ops_module
from osprey.bridges.google_chat.client import ChatClient
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import (
    GC_ATTACHMENTS,
    GC_DRIVE_ATTACHMENTS,
    GC_QUOTED_ATTACHMENTS,
    GC_QUOTED_DRIVE_ATTACHMENTS,
)
from osprey.bridges.google_chat.inbound import (
    DEFAULT_FILENAME,
    DOWNLOAD_FAILED_REASON,
    DRIVE_SKIP_REASON,
)
from osprey.bridges.google_chat.ops import GoogleChatOps

CFG = GoogleChatBridgeConfig(
    sa_key="/etc/osprey/sa-key.json",
    subscription="projects/p/subscriptions/s",
    app_id="users/1234567890",
    gcs_bucket="test-bucket",
)

SPACE = "spaces/AAAA"
MESSAGE = "spaces/AAAA/messages/MMMM"
QUOTED_MESSAGE = "spaces/AAAA/messages/QQQQ"

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
OTHER_PNG = b"\x89PNG\r\n\x1a\n" + b"other pixels"


# --- builders ---------------------------------------------------------------


def ref(resource: str = "attachments/AAA", name: str = "plot.png") -> dict[str, Any]:
    """One fetchable attachment reference, as ``events.parse_attachments`` builds it."""
    return {
        "content_name": name,
        "content_type": "image/png",
        "resource_name": resource,
        "source": "UPLOADED_CONTENT",
    }


def session_for(by_resource: Mapping[str, bytes]) -> httpx.Client:
    """A media session serving ``resource_name -> bytes``; anything else 404s."""

    def handler(request: httpx.Request) -> httpx.Response:
        resource = str(request.url.path).split("/v1/media/", 1)[-1]
        if resource not in by_resource:
            return httpx.Response(404)
        return httpx.Response(200, content=by_resource[resource])

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_service(get_bodies: list[Any] | None = None) -> MagicMock:
    """A Chat discovery stand-in whose ``messages.get`` answers ``get_bodies`` in order."""
    service = MagicMock()
    messages = service.spaces.return_value.messages.return_value
    if get_bodies is not None:
        messages.get.return_value.execute.side_effect = get_bodies
    return service


def gets(service: MagicMock) -> Any:
    """The ``messages.get`` mock, whose call list is the record of what was read."""
    return service.spaces.return_value.messages.return_value.get


def make_ops(
    cfg: GoogleChatBridgeConfig = CFG,
    *,
    media_session: Any = None,
    service: MagicMock | None = None,
) -> tuple[GoogleChatOps, MagicMock]:
    """An ops instance over a mock Chat service and (optionally) a fake media session."""
    service = service if service is not None else make_service()
    ops = GoogleChatOps(cfg, ChatClient(cfg, service), media_session=media_session)
    return ops, service


def dm_event(text: str = "how is the orbit?", **message: Any) -> dict[str, Any]:
    """A classic-shape Chat event in a direct message, which needs no @mention."""
    return {
        "type": "MESSAGE",
        "message": {
            "name": MESSAGE,
            "text": text,
            "sender": {"name": "users/9", "displayName": "Alice", "type": "HUMAN"},
            "space": {"name": SPACE, "spaceType": "DIRECT_MESSAGE"},
            **message,
        },
    }


def decoded(payload_file: Mapping[str, Any]) -> bytes:
    return base64.b64decode(payload_file["content_b64"])


def names(notes: Any) -> list[str]:
    return [note["filename"] for note in notes]


def reasons(notes: Any) -> list[str]:
    return [note["reason"] for note in notes]


# --- download_inputs: the two provenance buckets ----------------------------


def test_an_entrys_own_attachment_lands_in_the_own_bucket():
    ops, _ = make_ops(media_session=session_for({"attachments/AAA": PNG}))

    result = ops.download_inputs({GC_ATTACHMENTS: [ref()]})

    assert len(result.files) == 1
    assert result.files[0]["filename"] == "plot.png"
    assert decoded(result.files[0]) == PNG
    assert result.files[0]["ingest"] is True
    assert not result.skipped
    assert not result.quoted_files
    assert not result.quoted_skipped


def test_a_quoted_messages_attachment_lands_in_the_quoted_bucket():
    ops, _ = make_ops(media_session=session_for({"attachments/QQQ": PNG}))

    result = ops.download_inputs({GC_QUOTED_ATTACHMENTS: [ref("attachments/QQQ", "quoted.png")]})

    assert not result.files
    assert not result.skipped
    assert len(result.quoted_files) == 1
    assert result.quoted_files[0]["filename"] == "quoted.png"
    assert decoded(result.quoted_files[0]) == PNG


def test_the_two_buckets_do_not_bleed_into_each_other():
    """The engine reports own files at the payload's top level and quoted ones inside
    ``reply_to``, so a crossed bucket tells the agent the wrong person sent the file."""
    ops, _ = make_ops(
        media_session=session_for({"attachments/OWN": PNG, "attachments/QUO": OTHER_PNG})
    )

    result = ops.download_inputs(
        {
            GC_ATTACHMENTS: [ref("attachments/OWN", "own.png")],
            GC_QUOTED_ATTACHMENTS: [ref("attachments/QUO", "quoted.png")],
        }
    )

    assert [f["filename"] for f in result.files] == ["own.png"]
    assert decoded(result.files[0]) == PNG
    assert [f["filename"] for f in result.quoted_files] == ["quoted.png"]
    assert decoded(result.quoted_files[0]) == OTHER_PNG


def test_several_attachments_keep_the_order_chat_listed_them():
    ops, _ = make_ops(media_session=session_for({"attachments/1": PNG, "attachments/2": OTHER_PNG}))

    result = ops.download_inputs(
        {GC_ATTACHMENTS: [ref("attachments/1", "first.png"), ref("attachments/2", "second.png")]}
    )

    assert [f["filename"] for f in result.files] == ["first.png", "second.png"]


# --- download_inputs: Drive attachments -------------------------------------


def test_a_drive_attachment_is_reported_by_name_rather_than_dropped():
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_DRIVE_ATTACHMENTS: ["quarterly.pdf"]})

    assert names(result.skipped) == ["quarterly.pdf"]
    assert reasons(result.skipped) == [DRIVE_SKIP_REASON]
    assert not result.files


def test_a_quoted_drive_attachment_is_reported_in_the_quoted_bucket():
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_QUOTED_DRIVE_ATTACHMENTS: ["shared.pdf"]})

    assert names(result.quoted_skipped) == ["shared.pdf"]
    assert reasons(result.quoted_skipped) == [DRIVE_SKIP_REASON]
    assert not result.skipped


def test_an_entry_of_nothing_but_drive_names_is_not_the_no_op():
    """The user attached files; they just cannot be fetched. Answering as though the
    message carried none would lose the only signal that anything was lost."""
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs(
        {GC_DRIVE_ATTACHMENTS: ["a.pdf"], GC_QUOTED_DRIVE_ATTACHMENTS: ["b.pdf"]}
    )

    assert result != InputDownload()
    assert names(result.skipped) == ["a.pdf"]
    assert names(result.quoted_skipped) == ["b.pdf"]


def test_drive_notes_follow_the_download_skips_of_their_own_bucket():
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs(
        {
            GC_ATTACHMENTS: [ref("attachments/GONE", "deleted.png")],
            GC_DRIVE_ATTACHMENTS: ["from-drive.pdf"],
        }
    )

    assert names(result.skipped) == ["deleted.png", "from-drive.pdf"]
    assert reasons(result.skipped) == [DOWNLOAD_FAILED_REASON, DRIVE_SKIP_REASON]


def test_a_drive_name_carrying_crlf_is_sanitized_like_every_other_note():
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_DRIVE_ATTACHMENTS: ["evil\r\nname.pdf"]})

    assert names(result.skipped) == ["evilname.pdf"]


def test_a_drive_attachment_with_no_usable_name_still_gets_a_note():
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_DRIVE_ATTACHMENTS: ["", "   "]})

    assert names(result.skipped) == [DEFAULT_FILENAME, DEFAULT_FILENAME]


# --- download_inputs: the no-op and malformed entries ------------------------


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {GC_ATTACHMENTS: [], GC_DRIVE_ATTACHMENTS: []},
        {GC_QUOTED_ATTACHMENTS: [], GC_QUOTED_DRIVE_ATTACHMENTS: []},
        {"history_key": SPACE, "sender_id": "users/9"},
    ],
)
def test_an_entry_referencing_no_file_is_exactly_the_no_op(entry):
    """Byte-identical to the text-only path, which is what ``InputDownload()`` means."""
    ops, _ = make_ops(media_session=session_for({}))

    assert ops.download_inputs(entry) == InputDownload()


@pytest.mark.parametrize("junk", ["a string", 17, None, {"not": "a list"}])
def test_a_malformed_attachment_list_reads_as_no_attachments(junk):
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_ATTACHMENTS: junk, GC_DRIVE_ATTACHMENTS: junk})

    assert result == InputDownload()


def test_a_non_mapping_reference_is_dropped_without_costing_its_siblings():
    ops, _ = make_ops(media_session=session_for({"attachments/AAA": PNG}))

    result = ops.download_inputs({GC_ATTACHMENTS: ["junk", ref(), 17]})

    assert [f["filename"] for f in result.files] == ["plot.png"]


def test_a_non_string_drive_name_is_dropped_rather_than_reported_unnamed():
    """A note exists to say WHICH file was lost; one that cannot name it says nothing."""
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_DRIVE_ATTACHMENTS: [17, "real.pdf", None]})

    assert names(result.skipped) == ["real.pdf"]


def test_a_failing_download_becomes_a_skip_note_rather_than_a_raise():
    ops, _ = make_ops(media_session=session_for({}))

    result = ops.download_inputs({GC_ATTACHMENTS: [ref(name="lost.png")]})

    assert not result.files
    assert names(result.skipped) == ["lost.png"]
    assert reasons(result.skipped) == [DOWNLOAD_FAILED_REASON]


def test_the_entry_is_never_mutated():
    ops, _ = make_ops(media_session=session_for({"attachments/AAA": PNG}))
    entry = {GC_ATTACHMENTS: [ref()], GC_DRIVE_ATTACHMENTS: ["d.pdf"]}
    before = repr(entry)

    ops.download_inputs(entry)

    assert repr(entry) == before


def test_downloading_inputs_makes_no_chat_rest_call():
    """The media endpoint is a different host and a different session; a REST call here
    would mean the member had reached for the wire event's message instead of the entry."""
    ops, service = make_ops(media_session=session_for({"attachments/AAA": PNG}))

    ops.download_inputs({GC_ATTACHMENTS: [ref()]})

    assert not service.spaces.called


# --- download_inputs: the media session seam --------------------------------


def test_an_injected_session_is_used_as_given(monkeypatch):
    monkeypatch.setattr(
        ops_module, "build_media_session", MagicMock(side_effect=AssertionError("built one"))
    )
    ops, _ = make_ops(media_session=session_for({"attachments/AAA": PNG}))

    result = ops.download_inputs({GC_ATTACHMENTS: [ref()]})

    assert len(result.files) == 1


def test_no_session_is_built_for_an_entry_with_nothing_to_fetch(monkeypatch):
    """Building one reads a key file and imports ``google-auth``; a text-only
    conversation and a Drive-only one must pay neither."""
    build = MagicMock(side_effect=AssertionError("built a session for nothing"))
    monkeypatch.setattr(ops_module, "build_media_session", build)
    ops, _ = make_ops()

    assert ops.download_inputs({}) == InputDownload()
    assert names(ops.download_inputs({GC_DRIVE_ATTACHMENTS: ["a.pdf"]}).skipped) == ["a.pdf"]
    assert not build.called


def test_the_session_is_built_from_the_configured_key_once_and_reused(monkeypatch):
    build = MagicMock(return_value=session_for({"attachments/AAA": PNG}))
    monkeypatch.setattr(ops_module, "build_media_session", build)
    ops, _ = make_ops()

    ops.download_inputs({GC_ATTACHMENTS: [ref()]})
    ops.download_inputs({GC_ATTACHMENTS: [ref()]})

    build.assert_called_once_with(CFG.sa_key)


def test_a_session_that_cannot_be_built_reports_every_reference_as_failed(monkeypatch):
    """A missing key file or an uninstalled ``google-auth`` must not read to the user as
    a message that carried no attachments."""
    monkeypatch.setattr(
        ops_module, "build_media_session", MagicMock(side_effect=OSError("no such key file"))
    )
    ops, _ = make_ops()

    result = ops.download_inputs(
        {
            GC_ATTACHMENTS: [ref(name="own.png")],
            GC_QUOTED_ATTACHMENTS: [ref(name="quoted.png")],
        }
    )

    assert not result.files
    assert not result.quoted_files
    assert names(result.skipped) == ["own.png"]
    assert names(result.quoted_skipped) == ["quoted.png"]
    assert reasons(result.skipped) == [DOWNLOAD_FAILED_REASON]


# --- parse_event: delegation --------------------------------------------------


def test_parse_event_returns_the_events_modules_inbound_event():
    ops, _ = make_ops()

    event = ops.parse_event(dm_event())

    assert isinstance(event, InboundEvent)
    assert event.message_id == MESSAGE
    assert event.text == "how is the orbit?"
    assert event.history_key == SPACE


def test_parse_event_threads_this_instances_own_app_id_through():
    """The only thing the member adds is the config the engine does not supply — and a
    room message is a question for this app or not exactly according to it."""
    room_message = {
        "type": "MESSAGE",
        "message": {
            "name": MESSAGE,
            "text": "@Osprey how is the orbit?",
            "argumentText": "how is the orbit?",
            "sender": {"name": "users/9", "displayName": "Alice", "type": "HUMAN"},
            "space": {"name": SPACE, "spaceType": "SPACE"},
            "annotations": [
                {"type": "USER_MENTION", "userMention": {"user": {"name": "users/1234567890"}}}
            ],
        },
    }
    ours, _ = make_ops()
    other_app, _ = make_ops(GoogleChatBridgeConfig(app_id="users/999", sa_key=CFG.sa_key))

    assert ours.parse_event(room_message) is not None
    assert other_app.parse_event(room_message) is None


@pytest.mark.parametrize("junk", [None, "a string", 17, {}, {"type": "ADDED_TO_SPACE"}])
def test_parse_event_ignores_what_it_cannot_parse_rather_than_raising(junk):
    """It runs before the claim: a raise would leave the Pub/Sub message unacknowledged
    and the same event redelivered forever."""
    ops, _ = make_ops()

    assert ops.parse_event(junk) is None


def test_parse_event_makes_no_chat_rest_call():
    ops, service = make_ops()

    ops.parse_event(dm_event())

    assert not service.spaces.called


# --- resolve_reply_context: delegation ---------------------------------------


def test_resolve_reply_context_reads_through_the_injected_chat_client():
    """The fallback re-read and the quoted message's attachment read must both go through
    the client the posts use; wired to anything else, reply context resolves in tests
    against a stand-in and silently never resolves in production."""
    current = {
        "name": MESSAGE,
        "quotedMessageMetadata": {
            "name": QUOTED_MESSAGE,
            "quotedMessageSnapshot": {"sender": "Bob", "text": "the orbit looks off"},
        },
    }
    quoted = {
        "name": QUOTED_MESSAGE,
        "attachment": [
            {
                "contentName": "orbit.png",
                "contentType": "image/png",
                "source": "UPLOADED_CONTENT",
                "attachmentDataRef": {"resourceName": "attachments/QUO"},
            }
        ],
    }
    ops, service = make_ops(service=make_service([current, quoted]))
    event = ops.parse_event(dm_event())
    assert event is not None

    reply = ops.resolve_reply_context(event)

    assert reply is not None
    assert reply.reply_to == {"sender": "Bob", "text": "the orbit looks off"}
    assert [c.kwargs["name"] for c in gets(service).call_args_list] == [MESSAGE, QUOTED_MESSAGE]
    assert reply.meta[GC_QUOTED_ATTACHMENTS][0]["resource_name"] == "attachments/QUO"


def test_a_message_that_quotes_nothing_resolves_to_no_context():
    ops, _ = make_ops(service=make_service([{"name": MESSAGE}]))
    event = ops.parse_event(dm_event())
    assert event is not None

    assert ops.resolve_reply_context(event) is None


def test_a_failing_read_costs_the_quote_and_not_the_answer():
    """Reply context is enrichment: it degrades to ``None``, never to a raise the
    dispatch would have to survive."""
    ops, _ = make_ops(service=make_service([RuntimeError("chat api 500")]))
    event = ops.parse_event(dm_event())
    assert event is not None

    assert ops.resolve_reply_context(event) is None


def test_resolved_quoted_attachments_are_what_the_quoted_bucket_downloads():
    """The seam between the two members: whatever ``resolve_reply_context`` persists into
    the entry is exactly what ``download_inputs`` reads back out of it."""
    current = {
        "name": MESSAGE,
        "quotedMessageMetadata": {
            "name": QUOTED_MESSAGE,
            "quotedMessageSnapshot": {"sender": "Bob", "text": "look at this"},
        },
    }
    quoted = {
        "name": QUOTED_MESSAGE,
        "attachment": [
            {
                "contentName": "orbit.png",
                "contentType": "image/png",
                "source": "UPLOADED_CONTENT",
                "attachmentDataRef": {"resourceName": "attachments/QUO"},
            },
            {"contentName": "notes.pdf", "source": "DRIVE_FILE"},
        ],
    }
    ops, _ = make_ops(
        media_session=session_for({"attachments/QUO": PNG}),
        service=make_service([current, quoted]),
    )
    event = ops.parse_event(dm_event())
    assert event is not None
    reply = ops.resolve_reply_context(event)
    assert reply is not None

    result = ops.download_inputs({**event.claim_meta, **reply.meta})

    assert [f["filename"] for f in result.quoted_files] == ["orbit.png"]
    assert decoded(result.quoted_files[0]) == PNG
    assert names(result.quoted_skipped) == ["notes.pdf"]
    assert reasons(result.quoted_skipped) == [DRIVE_SKIP_REASON]
    assert not result.files
    assert not result.skipped


# --- coalesce_key -------------------------------------------------------------


def test_the_group_key_is_the_conversation_and_the_sender():
    ops, _ = make_ops()

    assert ops.coalesce_key({"history_key": SPACE, "sender_id": "users/9"}) == [SPACE, "users/9"]


def test_two_people_asking_in_one_conversation_are_never_grouped_together():
    ops, _ = make_ops()

    alice = ops.coalesce_key({"history_key": SPACE, "sender_id": "users/9"})
    bob = ops.coalesce_key({"history_key": SPACE, "sender_id": "users/10"})

    assert alice != bob


def test_one_person_asking_twice_in_one_conversation_groups_together():
    ops, _ = make_ops()
    entry = {"history_key": SPACE, "sender_id": "users/9"}

    assert ops.coalesce_key(dict(entry)) == ops.coalesce_key(dict(entry))


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"history_key": SPACE},
        {"sender_id": "users/9"},
        {"history_key": "", "sender_id": "users/9"},
        {"history_key": SPACE, "sender_id": ""},
        {"history_key": 17, "sender_id": "users/9"},
        {"history_key": SPACE, "sender_id": None},
    ],
)
def test_an_entry_missing_either_half_stands_alone(entry):
    """Fail-closed: a wrong group silently discards somebody's question, while a missed
    group costs only a duplicate run."""
    ops, _ = make_ops()

    assert ops.coalesce_key(entry) == ()


def test_the_group_key_is_pure():
    ops, service = make_ops()
    entry = {"history_key": SPACE, "sender_id": "users/9"}

    ops.coalesce_key(entry)

    assert entry == {"history_key": SPACE, "sender_id": "users/9"}
    assert not service.spaces.called
