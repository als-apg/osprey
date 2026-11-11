"""ChannelRoster: who is in a Slack channel, listed name first, then seen, then unknown."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import parse_qs

import httpx

from osprey.bridges.core import CoreConfig
from osprey.bridges.slack import roster as roster_module
from osprey.bridges.slack.client import SlackApiError, SlackWebClient
from osprey.bridges.slack.config import SlackBridgeConfig
from osprey.bridges.slack.roster import (
    ROSTER_MAX_CHANNELS,
    ROSTER_MAX_MEMBERS,
    ROSTER_TTL_SECONDS,
    SEEN_MAX_PER_CHANNEL,
    ChannelRoster,
)
from tests.bridges.conftest import FakeClock

BOT_USER = "UBOT0001"
CHANNEL = "C0123ROOM"


def person(ident: str, display: str = "", real: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "id": ident,
        "real_name": real,
        "profile": {"display_name": display, "real_name": real},
        **extra,
    }


class FakeClient:
    """Answers ``conversations_members`` and ``users_info`` from dicts, recording calls."""

    def __init__(
        self,
        users: dict[str, Any],
        members: list[str] | BaseException | None = None,
        more: bool = False,
    ) -> None:
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


def test_members_names_by_listing_then_seen_then_unknown():
    client = FakeClient(
        {
            "U1": person("U1", display="alice", real="Alice Example"),
            "U2": person("U2", real="Bob Real"),
            "U3": person("U3"),
            "U4": person("U4"),
        }
    )
    roster = ChannelRoster(client, BOT_USER)
    roster.note_seen(CHANNEL, {"U1": "Ally", "U3": "Carol"})
    assert roster.members(CHANNEL) == (
        {"U1": "alice", "U2": "Bob Real", "U3": "Carol", "U4": None},
        False,
    )


def test_the_handle_is_never_used_as_a_name():
    client = FakeClient({"U1": {"id": "U1", "name": "ahandle", "profile": {}}})
    assert ChannelRoster(client, BOT_USER).members(CHANNEL) == ({"U1": None}, False)


def test_members_keeps_only_people_and_never_the_bot():
    client = FakeClient(
        {
            "U1": person("U1", display="alice"),
            BOT_USER: person(BOT_USER, display="osprey"),
            "U9": person("U9", display="other bot", is_bot=True),
            "U8": person("U8", display="gone", deleted=True),
            "USLACKBOT": person("USLACKBOT", display="Slackbot"),
        }
    )
    roster = ChannelRoster(client, BOT_USER)
    assert roster.members(CHANNEL) == ({"U1": "alice"}, False)
    assert BOT_USER not in client.info_calls


def test_a_repeated_member_id_is_listed_and_looked_up_once():
    client = FakeClient({"U1": person("U1", display="alice")}, members=["U1", "U1"])
    assert ChannelRoster(client, BOT_USER).members(CHANNEL) == ({"U1": "alice"}, False)
    assert client.info_calls == ["U1"]


def test_members_is_cached_within_the_ttl_and_refetched_after():
    clock = FakeClock()
    client = FakeClient({"U1": person("U1", display="alice")})
    roster = ChannelRoster(client, BOT_USER, now=clock)
    roster.members(CHANNEL)
    clock.advance(ROSTER_TTL_SECONDS - 1)
    roster.members(CHANNEL)
    assert len(client.member_calls) == 1
    assert client.info_calls == ["U1"]
    clock.advance(2)
    roster.members(CHANNEL)
    assert len(client.member_calls) == 2


def test_a_name_seen_after_the_fetch_fills_the_cached_roster_without_a_refetch():
    client = FakeClient({"U1": person("U1")})
    roster = ChannelRoster(client, BOT_USER)
    assert roster.members(CHANNEL) == ({"U1": None}, False)
    roster.note_seen(CHANNEL, {"U1": "Alice"})
    assert roster.members(CHANNEL) == ({"U1": "Alice"}, False)
    assert len(client.member_calls) == 1


def test_a_seen_person_who_is_not_listed_is_not_added():
    roster = ChannelRoster(FakeClient({"U1": person("U1")}), BOT_USER)
    roster.note_seen(CHANNEL, {"U7": "Stranger"})
    assert roster.members(CHANNEL) == ({"U1": None}, False)


def test_the_newest_seen_name_wins_among_seen_names():
    roster = ChannelRoster(FakeClient({"U1": person("U1")}), BOT_USER)
    roster.note_seen(CHANNEL, {"U1": "Old"})
    roster.note_seen(CHANNEL, {"U1": "New"})
    assert roster.members(CHANNEL) == ({"U1": "New"}, False)


def test_a_failing_listing_returns_none_logs_and_is_not_cached(caplog):
    client = FakeClient({"U1": person("U1", display="alice")}, members=RuntimeError("down"))
    roster = ChannelRoster(client, BOT_USER)
    with caplog.at_level(logging.WARNING, logger=roster_module.__name__):
        assert roster.members(CHANNEL) is None
    assert any("members listing failed" in rec.message for rec in caplog.records)
    client.member_ids = ["U1"]
    assert roster.members(CHANNEL) == ({"U1": "alice"}, False)
    assert len(client.member_calls) == 2


def test_a_failing_user_lookup_fails_the_listing_and_caches_nothing():
    client = FakeClient(
        {
            "U1": person("U1", display="alice"),
            "U2": SlackApiError("slack users.info failed: user_not_found", method="users.info"),
        }
    )
    roster = ChannelRoster(client, BOT_USER)
    assert roster.members(CHANNEL) is None
    assert roster._cache == {}
    client.users["U2"] = person("U2", display="bob")
    assert roster.members(CHANNEL) == ({"U1": "alice", "U2": "bob"}, False)


def test_a_malformed_user_object_fails_the_listing():
    client = FakeClient({"U1": "not a dict"})
    assert ChannelRoster(client, BOT_USER).members(CHANNEL) is None


def test_an_empty_channel_answers_none_without_a_call():
    client = FakeClient({})
    assert ChannelRoster(client, BOT_USER).members("") is None
    assert client.member_calls == []


def test_the_cache_and_the_seen_map_are_bounded():
    roster = ChannelRoster(FakeClient({"U1": person("U1")}), BOT_USER)
    for index in range(ROSTER_MAX_CHANNELS + 5):
        roster.members(f"C{index}")
        roster.note_seen(f"C{index}", {"U1": "Alice"})
    assert len(roster._cache) == ROSTER_MAX_CHANNELS
    assert len(roster._seen) == ROSTER_MAX_CHANNELS
    assert "C0" not in roster._cache
    assert "C0" not in roster._seen
    roster.note_seen(CHANNEL, {f"U{i}": f"N{i}" for i in range(SEEN_MAX_PER_CHANNEL + 7)})
    assert len(roster._seen[CHANNEL]) == SEEN_MAX_PER_CHANNEL
    assert "U0" not in roster._seen[CHANNEL]


def test_the_listing_is_bounded_and_the_cap_reports_more_not_listed():
    client = FakeClient({"U1": person("U1")}, more=True)
    roster = ChannelRoster(client, BOT_USER)
    assert roster.members(CHANNEL) == ({"U1": None}, True)
    assert client.member_calls == [(CHANNEL, ROSTER_MAX_MEMBERS)]


def test_the_bounds_match_the_other_bridges():
    assert ROSTER_TTL_SECONDS == 600
    assert ROSTER_MAX_MEMBERS == 200
    assert ROSTER_MAX_CHANNELS == 256
    assert SEEN_MAX_PER_CHANNEL == 500


def test_note_seen_ignores_an_empty_channel_or_no_people():
    roster = ChannelRoster(FakeClient({}), BOT_USER)
    roster.note_seen("", {"U1": "Alice"})
    roster.note_seen(CHANNEL, {})
    assert roster._seen == {}


# --- over the real Web API client -------------------------------------------------


def test_the_roster_speaks_conversations_members_and_users_info_on_the_wire():
    requests: list[tuple[str, dict[str, list[str]]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        form = parse_qs(request.content.decode())
        requests.append((method, form))
        if method == "conversations.members":
            return httpx.Response(
                200,
                json={"ok": True, "members": ["U1", BOT_USER], "response_metadata": {}},
            )
        if method == "users.info":
            user = form["user"][0]
            return httpx.Response(
                200, json={"ok": True, "user": person(user, display=f"name-{user}")}
            )
        return httpx.Response(200, json={"ok": False, "error": "unknown_method"})

    cfg = SlackBridgeConfig(
        bot_token="xoxb-1-2-secret",
        app_token="xapp-1-A-3-secret",  # gitleaks:allow - a fake test token
        core=CoreConfig(),
    )
    client = SlackWebClient(cfg, http=httpx.Client(transport=httpx.MockTransport(handler)))
    roster = ChannelRoster(client, BOT_USER)
    assert roster.members(CHANNEL) == ({"U1": "name-U1"}, False)
    assert [m for m, _ in requests] == ["conversations.members", "users.info"]
    assert requests[0][1]["channel"] == [CHANNEL]
    assert requests[0][1]["limit"] == [str(ROSTER_MAX_MEMBERS)]
    assert requests[1][1]["user"] == ["U1"]
