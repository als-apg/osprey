"""The Teams posting suites' world: ids, activities and entries, and a recording Connector."""

from __future__ import annotations

from typing import Any

from osprey.bridges.core.text import chunk_text
from osprey.bridges.teams.client import MessageSizeTooBig
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
from osprey.bridges.teams.ops import ANSWER_CHUNK_CHARS, TeamsOps

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
    the swallow contracts are proven without a transport at all. ``members`` is what
    ``list_members`` answers — a list, returned as ``(members, False)``, or an
    exception to raise; its calls are recorded in ``member_calls``.
    """

    def __init__(
        self,
        fail_with: BaseException | None = None,
        members: list[dict[str, Any]] | BaseException = (),  # type: ignore[assignment]
    ) -> None:
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []
        self.fail_with = fail_with
        self.members = members
        self.member_calls: list[tuple[str, str, int | None]] = []

    def list_members(
        self, service_url: str, conversation_id: str, *, limit: int | None
    ) -> tuple[list[dict[str, Any]], bool]:
        self.member_calls.append((service_url, conversation_id, limit))
        if isinstance(self.members, BaseException):
            raise self.members
        return list(self.members), False

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
