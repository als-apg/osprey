"""ConversationRoster: who is in a Teams conversation, listing-first, then seen, then unknown."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from osprey.bridges.core import MENTION_RULE, CoreConfig, RoomMember
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.pipeline import PipelineDeps, handle_event
from osprey.bridges.teams import roster as roster_module
from osprey.bridges.teams.client import ConnectorClient, TokenSource
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.events import MS_CONVERSATION_ID, MS_SERVICE_URL
from osprey.bridges.teams.ops import TeamsOps
from osprey.bridges.teams.roster import (
    ROSTER_MAX_CONVERSATIONS,
    ROSTER_MAX_MEMBERS,
    ROSTER_TTL_SECONDS,
    SEEN_MAX_PER_CONVERSATION,
    ConversationRoster,
)
from tests.bridges.teams.test_posting import RecordingConnector

APP_ID = "11111111-2222-3333-4444-555555555555"
BOT = f"28:{APP_ID}"
SERVICE_URL = "https://smba.trafficmanager.net/amer/"
CHANNEL = "19:room@thread.tacv2"
REPLY_CONVERSATION = f"{CHANNEL};messageid=1700000000000"
CFG = TeamsBridgeConfig(app_id=APP_ID)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Lister:
    def __init__(self, members: Any = (), more: bool = False) -> None:
        self.members = members
        self.more = more
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, service_url: str, conversation: str, limit: int) -> tuple[list[Any], bool]:
        self.calls.append((service_url, conversation, limit))
        if isinstance(self.members, BaseException):
            raise self.members
        return list(self.members), self.more


def test_members_names_by_listing_then_seen_then_unknown():
    roster = ConversationRoster(
        Lister([{"id": "29:1", "name": "Alice"}, {"id": "29:2"}, {"id": "29:3", "name": ""}]),
        APP_ID,
    )
    roster.note_seen(CHANNEL, {"29:1": "Ally", "29:2": "Carol"})
    assert roster.members(SERVICE_URL, CHANNEL) == (
        {"29:1": "Alice", "29:2": "Carol", "29:3": None},
        False,
    )


def test_members_keeps_only_people_and_never_the_bot():
    roster = ConversationRoster(
        Lister(
            [
                {"id": "29:1", "name": "Alice"},
                {"id": BOT, "name": "Osprey"},
                {"id": "28:other", "name": "Other bot"},
                {"id": "29:9", "name": "Bot in disguise", "role": "bot"},
                {"id": "8:orgid:x", "name": "Not a 29 id"},
                "junk",
            ]
        ),
        APP_ID,
    )
    assert roster.members(SERVICE_URL, CHANNEL) == ({"29:1": "Alice"}, False)


def test_members_is_cached_within_the_ttl_and_refetched_after():
    clock = Clock()
    lister = Lister([{"id": "29:1", "name": "Alice"}])
    roster = ConversationRoster(lister, APP_ID, now=clock)
    roster.members(SERVICE_URL, CHANNEL)
    clock.t += ROSTER_TTL_SECONDS - 1
    roster.members("https://other.example.org/", CHANNEL)
    assert len(lister.calls) == 1
    clock.t += 2
    roster.members(SERVICE_URL, CHANNEL)
    assert len(lister.calls) == 2


def test_a_name_seen_after_the_fetch_fills_the_cached_roster_without_a_refetch():
    lister = Lister([{"id": "29:1"}])
    roster = ConversationRoster(lister, APP_ID)
    assert roster.members(SERVICE_URL, CHANNEL) == ({"29:1": None}, False)
    roster.note_seen(CHANNEL, {"29:1": "Alice"})
    assert roster.members(SERVICE_URL, CHANNEL) == ({"29:1": "Alice"}, False)
    assert len(lister.calls) == 1


def test_a_failing_list_returns_none_logs_and_is_not_cached(caplog):
    lister = Lister(RuntimeError("members down"))
    roster = ConversationRoster(lister, APP_ID)
    with caplog.at_level(logging.WARNING, logger=roster_module.__name__):
        assert roster.members(SERVICE_URL, CHANNEL) is None
    assert any("members listing failed" in rec.message for rec in caplog.records)
    lister.members = [{"id": "29:1", "name": "Alice"}]
    assert roster.members(SERVICE_URL, CHANNEL) == ({"29:1": "Alice"}, False)
    assert len(lister.calls) == 2


def test_the_cache_and_the_seen_map_are_bounded():
    roster = ConversationRoster(Lister([{"id": "29:1"}]), APP_ID)
    for index in range(ROSTER_MAX_CONVERSATIONS + 5):
        roster.members(SERVICE_URL, f"19:c{index}")
        roster.note_seen(f"19:c{index}", {"29:1": "Alice"})
    assert len(roster._cache) == ROSTER_MAX_CONVERSATIONS
    assert len(roster._seen) == ROSTER_MAX_CONVERSATIONS
    assert "19:c0" not in roster._cache
    roster.note_seen(CHANNEL, {f"29:{i}": f"U{i}" for i in range(SEEN_MAX_PER_CONVERSATION + 7)})
    assert len(roster._seen[CHANNEL]) == SEEN_MAX_PER_CONVERSATION
    assert "29:0" not in roster._seen[CHANNEL]


def test_the_cap_reports_more_not_listed():
    lister = Lister([{"id": "29:1"}], more=True)
    roster = ConversationRoster(lister, APP_ID)
    assert roster.members(SERVICE_URL, CHANNEL)[1] is True
    assert lister.calls == [(SERVICE_URL, CHANNEL, ROSTER_MAX_MEMBERS)]


# --- through the adapter --------------------------------------------------------


def token_route(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})


def paged(*pages: Any):
    served = list(pages)
    requests: list[httpx.Request] = []

    def route(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=served[min(len(requests) - 1, len(served) - 1)])

    route.requests = requests  # type: ignore[attr-defined]
    return route


def connector_over(route: Any) -> ConnectorClient:
    tokens = TokenSource(CFG, httpx.Client(transport=httpx.MockTransport(token_route)))
    return ConnectorClient(CFG, tokens, httpx.Client(transport=httpx.MockTransport(route)))


def activity(sender: str, name: str, *mentions: tuple[str, str]) -> dict[str, Any]:
    entities: list[dict[str, Any]] = [
        {"type": "mention", "text": "<at>Osprey</at>", "mentioned": {"id": BOT, "name": "Osprey"}}
    ]
    text = "<at>Osprey</at> tell"
    for ident, display in mentions:
        entities.append(
            {
                "type": "mention",
                "text": f"<at>{display}</at>",
                "mentioned": {"id": ident, "name": display},
            }
        )
        text += f" <at>{display}</at>"
    return {
        "type": "message",
        "id": "1700000000123",
        "serviceUrl": SERVICE_URL,
        "from": {"id": sender, "name": name},
        "conversation": {"id": REPLY_CONVERSATION, "conversationType": "channel"},
        "text": text,
        "entities": entities,
    }


def entry(**over: Any) -> dict[str, Any]:
    return {MS_SERVICE_URL: SERVICE_URL, MS_CONVERSATION_ID: REPLY_CONVERSATION, **over}


def test_room_people_via_ops_over_a_mock_connector():
    route = paged(
        {
            "members": [{"id": "29:111", "name": "Alice"}, {"id": "29:222"}],
            "continuationToken": "c",
        },
        {"members": [{"id": "29:333"}, {"id": BOT, "name": "Osprey"}]},
    )
    ops = TeamsOps(CFG, connector_over(route))
    ops.parse_event(activity("29:444", "Dave", ("29:222", "Carol")))

    people = ops.room_people(entry())

    assert people is not None
    assert list(people.members) == [
        RoomMember("29:111", "Alice"),
        RoomMember("29:222", "Carol"),
        RoomMember("29:333", None),
    ]
    # The channel itself is listed, never the thread's own id.
    assert all(f"/v3/conversations/{CHANNEL}/pagedmembers" in str(r.url) for r in route.requests)


def test_room_people_names_the_asker_from_the_entry_after_a_restart():
    ops = TeamsOps(CFG, RecordingConnector(members=[{"id": "29:111"}, {"id": "29:222"}]))
    people = ops.room_people(entry(sender_id="29:111", sender_display="Alice"))
    assert people is not None
    assert list(people.members) == [RoomMember("29:111", "Alice"), RoomMember("29:222", None)]
    outsider = TeamsOps(CFG, RecordingConnector(members=[{"id": "29:222"}]))
    people = outsider.room_people(entry(sender_id="29:111", sender_display="Alice"))
    assert people is not None
    assert list(people.members) == [RoomMember("29:222", None)]


def test_room_people_carries_the_mentions_setting():
    members = [{"id": "29:111", "name": "Alice"}]
    on = TeamsOps(CFG, RecordingConnector(members=members))
    off = TeamsOps(
        TeamsBridgeConfig(app_id=APP_ID, mentions=False), RecordingConnector(members=members)
    )
    assert on.room_people(entry()).mentions is True
    assert off.room_people(entry()).mentions is False


def test_room_people_is_none_for_an_entry_with_no_address():
    connector = RecordingConnector(members=[{"id": "29:111", "name": "Alice"}])
    ops = TeamsOps(CFG, connector)
    assert ops.room_people({}) is None
    assert ops.room_people({MS_SERVICE_URL: SERVICE_URL}) is None
    assert ops.room_people({MS_CONVERSATION_ID: CHANNEL, MS_SERVICE_URL: 7}) is None
    assert connector.member_calls == []


class StubDispatcher:
    def __init__(self) -> None:
        self.extras: list[Any] = []

    def run(self, question: str, extra: Any = None, *, on_run_id: Any = None) -> dict[str, Any]:
        del question, on_run_id
        self.extras.append(extra)
        return {"status": "completed", "text_output": "done", "run_id": "run-1"}


def test_the_room_reaches_the_dispatch_payload(tmp_path):
    ops = TeamsOps(CFG, RecordingConnector(members=[{"id": "29:111"}, {"id": "29:222"}]))
    dispatcher = StubDispatcher()
    deps = PipelineDeps(
        cfg=CoreConfig(),
        ops=ops,
        dedup=DedupStore(str(tmp_path / "dedup.json")),
        dispatcher=dispatcher,  # type: ignore[arg-type]
        probe_capability=lambda cfg, capability: True,
    )

    assert handle_event(activity("29:111", "Alice", ("29:222", "Carol")), deps) == "handled"

    [extra] = dispatcher.extras
    assert extra["asker"] == {"id": "29:111", "name": "Alice"}
    assert extra["room"] == {
        "members": [{"id": "29:111", "name": "Alice"}, {"id": "29:222", "name": "Carol"}],
        "mentions": MENTION_RULE,
    }
