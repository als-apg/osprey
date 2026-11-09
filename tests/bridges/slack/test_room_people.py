"""SlackOps as a room roster: who is in the conversation, and live mentions in answers."""

from __future__ import annotations

from typing import Any

from osprey.bridges.core import MENTION_RULE, CoreConfig, RoomMember, RoomRoster
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.pipeline import PipelineDeps, handle_event
from osprey.bridges.slack.client import SlackApiError
from osprey.bridges.slack.events import SL_CHANNEL
from tests.bridges.conftest import StubDispatcher
from tests.bridges.slack._posting import (
    BOT_USER,
    CHAN,
    MPIM,
    SENDER_ID,
    SENDER_NAME,
    RecordingClient,
    make_entry,
    make_ops,
    mention_payload,
)


def person(ident: str, display: str = "", real: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "id": ident,
        "real_name": real,
        "profile": {"display_name": display, "real_name": real},
        **extra,
    }


class RosterClient(RecordingClient):
    """A recording client that also answers the roster's two Web API calls."""

    def __init__(
        self,
        users: dict[str, Any],
        members: list[str] | BaseException | None = None,
        more: bool = False,
    ) -> None:
        super().__init__()
        self.users = users
        self.member_ids = members if members is not None else list(users)
        self.more = more
        self.member_calls: list[tuple[str, int | None]] = []
        self.info_calls: list[str] = []

    def conversations_members(self, channel: str, *, limit: int | None) -> tuple[list[str], bool]:
        self.member_calls.append((channel, limit))
        if isinstance(self.member_ids, BaseException):
            raise self.member_ids
        return list(self.member_ids), self.more

    def users_info(self, user: str) -> dict[str, Any]:
        self.info_calls.append(user)
        info = self.users[user]
        if isinstance(info, BaseException):
            raise info
        return info


def channel_client(**kwargs: Any) -> RosterClient:
    return RosterClient(
        {
            SENDER_ID: person(SENDER_ID, display=SENDER_NAME),
            "UBOB0002": person("UBOB0002", real="Bob Real"),
            "UCAROL03": person("UCAROL03"),
            BOT_USER: person(BOT_USER, display="osprey", is_bot=True),
        },
        **kwargs,
    )


# --- room_people ---------------------------------------------------------------


def test_the_adapter_satisfies_the_room_roster_protocol():
    ops, _ = make_ops()
    assert isinstance(ops, RoomRoster)


def test_room_people_lists_a_channel_without_the_bot():
    client = channel_client()
    ops, _ = make_ops(client)

    people = ops.room_people(make_entry("channel"))

    assert people is not None
    assert list(people.members) == [
        RoomMember(SENDER_ID, SENDER_NAME),
        RoomMember("UBOB0002", "Bob Real"),
        RoomMember("UCAROL03", None),
    ]
    assert people.more_not_listed is False
    assert client.member_calls and client.member_calls[0][0] == CHAN


def test_room_people_lists_private_channels_and_group_dms():
    for kind, channel in (("group", "G0PRIVATE"), ("mpim", MPIM)):
        client = channel_client()
        ops, _ = make_ops(client)
        people = ops.room_people(make_entry(kind, channel=channel))
        assert people is not None
        assert [m.id for m in people.members] == [SENDER_ID, "UBOB0002", "UCAROL03"]
        assert [call[0] for call in client.member_calls] == [channel]


def test_room_people_is_none_for_a_one_to_one_dm_and_makes_no_call():
    client = channel_client()
    ops, _ = make_ops(client)
    assert ops.room_people(make_entry("im")) is None
    assert client.member_calls == []
    assert client.info_calls == []


def test_room_people_names_the_asker_from_the_entry_after_a_restart():
    client = RosterClient({SENDER_ID: person(SENDER_ID), "UBOB0002": person("UBOB0002")})
    ops, _ = make_ops(client)
    people = ops.room_people(make_entry("channel"))
    assert people is not None
    assert list(people.members) == [
        RoomMember(SENDER_ID, SENDER_NAME),
        RoomMember("UBOB0002", None),
    ]


def test_room_people_never_adds_an_unlisted_asker():
    client = RosterClient({"UBOB0002": person("UBOB0002", display="bob")})
    ops, _ = make_ops(client)
    people = ops.room_people(make_entry("channel"))
    assert people is not None
    assert list(people.members) == [RoomMember("UBOB0002", "bob")]


def test_room_people_never_uses_the_asker_id_as_their_name():
    client = RosterClient({SENDER_ID: person(SENDER_ID)})
    ops, _ = make_ops(client)
    entry = make_entry("channel")
    entry["sender_display"] = SENDER_ID
    people = ops.room_people(entry)
    assert people is not None
    assert list(people.members) == [RoomMember(SENDER_ID, None)]


def test_room_people_keeps_a_listed_name_over_the_entry():
    client = RosterClient({SENDER_ID: person(SENDER_ID, display="ali")})
    ops, _ = make_ops(client)
    people = ops.room_people(make_entry("channel"))
    assert people is not None
    assert list(people.members) == [RoomMember(SENDER_ID, "ali")]


def test_room_people_carries_the_mentions_setting():
    on, _ = make_ops(channel_client(), mentions=True)
    off, _ = make_ops(channel_client(), mentions=False)
    on_people = on.room_people(make_entry("channel"))
    off_people = off.room_people(make_entry("channel"))
    assert on_people is not None and on_people.mentions is True
    assert off_people is not None and off_people.mentions is False


def test_room_people_carries_more_not_listed():
    ops, _ = make_ops(channel_client(more=True))
    people = ops.room_people(make_entry("channel"))
    assert people is not None
    assert people.more_not_listed is True


def test_room_people_is_none_for_an_entry_with_no_channel_and_makes_no_call():
    client = channel_client()
    ops, _ = make_ops(client)
    assert ops.room_people({}) is None
    no_channel = make_entry("channel")
    no_channel[SL_CHANNEL] = ""
    assert ops.room_people(no_channel) is None
    bad_channel = make_entry("channel")
    bad_channel[SL_CHANNEL] = 7
    assert ops.room_people(bad_channel) is None
    assert client.member_calls == []


def test_room_people_is_none_when_the_listing_fails():
    ops, _ = make_ops(
        channel_client(members=SlackApiError("listing failed", method="conversations.members"))
    )
    assert ops.room_people(make_entry("channel")) is None


def test_room_people_is_none_over_a_client_without_roster_calls():
    ops, _ = make_ops(RecordingClient())
    assert ops.room_people(make_entry("channel")) is None


def test_room_people_never_raises_on_a_hostile_entry():
    ops, _ = make_ops(channel_client())

    class Hostile(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    assert ops.room_people(Hostile()) is None


def test_the_room_reaches_the_dispatch_payload(tmp_path):
    client = RosterClient({SENDER_ID: person(SENDER_ID), "UBOB0002": person("UBOB0002", "bob")})
    ops, _ = make_ops(client)
    dispatcher = StubDispatcher()
    deps = PipelineDeps(
        cfg=CoreConfig(),
        ops=ops,
        dedup=DedupStore(str(tmp_path / "dedup.json")),
        dispatcher=dispatcher,  # type: ignore[arg-type]
        probe_capability=lambda cfg, capability: True,
    )

    assert handle_event(mention_payload(), deps) == "handled"

    [extra] = dispatcher.extras
    assert extra["asker"] == {"id": SENDER_ID, "name": SENDER_NAME}
    assert extra["room"] == {
        "members": [
            {"id": SENDER_ID, "name": SENDER_NAME},
            {"id": "UBOB0002", "name": "bob"},
        ],
        "mentions": MENTION_RULE,
    }


# --- mentions in the answer ---------------------------------------------------


def completed(text: str) -> dict[str, Any]:
    return {"status": "completed", "text_output": text}


def test_an_answer_mentioning_a_listed_member_posts_a_live_mention():
    client = channel_client()
    ops, _ = make_ops(client)
    ops.post_answer(make_entry("channel"), completed("Ask <@UBOB0002> about it."))
    assert client.texts == ["Ask <@UBOB0002> about it."]
    assert [call[0] for call in client.member_calls] == [CHAN]


def test_an_unlisted_mention_posts_as_plain_text():
    client = channel_client()
    ops, _ = make_ops(client)
    ops.post_answer(make_entry("channel"), completed("Ask <@UZED9999> about it."))
    assert client.texts == ["Ask @UZED9999 about it."]


def test_an_answer_without_a_placeholder_makes_no_roster_call():
    client = channel_client()
    ops, _ = make_ops(client)
    ops.post_answer(make_entry("channel"), completed("The current is 500 mA."))
    assert client.texts == ["The current is 500 mA."]
    assert client.member_calls == []
    assert client.info_calls == []


def test_mentions_off_posts_the_listed_name():
    client = channel_client()
    ops, _ = make_ops(client, mentions=False)
    ops.post_answer(make_entry("channel"), completed("Ask <@UBOB0002> about it."))
    assert client.texts == ["Ask @Bob Real about it."]


def test_a_roster_failure_posts_plain_ids_and_still_answers():
    client = channel_client(
        members=SlackApiError("listing failed", method="conversations.members", error="ratelimited")
    )
    ops, _ = make_ops(client)
    ops.post_answer(make_entry("channel"), completed("Ask <@UBOB0002> about it."))
    assert client.texts == ["Ask @UBOB0002 about it."]


def test_the_roster_is_cached_between_answers():
    client = channel_client()
    ops, _ = make_ops(client)
    ops.room_people(make_entry("channel"))
    ops.post_answer(make_entry("channel"), completed("Ask <@UBOB0002> about it."))
    assert len(client.member_calls) == 1
    assert client.texts == ["Ask <@UBOB0002> about it."]
