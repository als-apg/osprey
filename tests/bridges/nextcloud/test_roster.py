"""ParticipantRoster: who is in a Talk room, listing-first, then seen, then unknown."""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any

import httpx

from osprey.bridges.core import MENTION_RULE, CoreConfig, RoomMember
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.runtime import BridgeRuntime, build_deps, build_drain_deps
from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, RoomDirectory, TalkClient
from osprey.bridges.nextcloud_talk import roster as roster_module
from osprey.bridges.nextcloud_talk.ops import NC_ROOM, NextcloudTalkOps
from osprey.bridges.nextcloud_talk.roster import (
    ROSTER_MAX_MEMBERS,
    ROSTER_MAX_ROOMS,
    ROSTER_TTL_SECONDS,
    SEEN_MAX_PER_ROOM,
    ParticipantRoster,
)

BOT = "osprey-bot"
ROOM = "roomA"
CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org", bot_account=BOT, app_password="pw", rooms=(ROOM,)
)


def user(ident: str, name: str = "", **over: Any) -> dict[str, Any]:
    return {"actorType": "users", "actorId": ident, "displayName": name, **over}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Lister:
    def __init__(self, attendees: Any = ()) -> None:
        self.attendees = attendees
        self.calls: list[str] = []

    def __call__(self, room: str) -> list[Any]:
        self.calls.append(room)
        if isinstance(self.attendees, BaseException):
            raise self.attendees
        return list(self.attendees)


def test_members_are_named_by_the_listing_first_then_seen_then_unknown():
    roster = ParticipantRoster(Lister([user("alice", "Alice"), user("carol"), user("dave")]), BOT)
    roster.note_seen(ROOM, {"carol": "Carol"})
    assert roster.members(ROOM) == ({"alice": "Alice", "carol": "Carol", "dave": None}, False)


def test_a_seen_name_never_overrides_a_listing_name():
    roster = ParticipantRoster(Lister([user("alice", "Alice")]), BOT)
    roster.note_seen(ROOM, {"alice": "Ally"})
    assert roster.members(ROOM) == ({"alice": "Alice"}, False)


def test_a_name_seen_after_the_fetch_fills_an_empty_listing_name_without_a_refetch():
    lister = Lister([user("carol")])
    roster = ParticipantRoster(lister, BOT)
    assert roster.members(ROOM) == ({"carol": None}, False)
    roster.note_seen(ROOM, {"carol": "Carol"})
    assert roster.members(ROOM) == ({"carol": "Carol"}, False)
    assert lister.calls == [ROOM]


def test_members_skip_guests_federated_users_the_bot_and_everyone():
    roster = ParticipantRoster(
        Lister(
            [
                user("alice", "Alice"),
                {"actorType": "guests", "actorId": "abc", "displayName": "Visitor"},
                {"actorType": "federated_users", "actorId": "x@remote", "displayName": "Far"},
                user("OSPREY-Bot", "OSPREY Bot"),
                user("ALL", "Everyone"),
                user("", "No id"),
                user("alice", "Duplicate"),
                "junk",
            ]
        ),
        BOT,
    )
    assert roster.members(ROOM) == ({"alice": "Alice"}, False)


def test_members_are_cached_within_the_ttl_and_refetched_after():
    clock = Clock()
    lister = Lister([user("alice", "Alice")])
    roster = ParticipantRoster(lister, BOT, now=clock)
    roster.members(ROOM)
    clock.t += ROSTER_TTL_SECONDS - 1
    roster.members(ROOM)
    assert len(lister.calls) == 1
    clock.t += 2
    roster.members(ROOM)
    assert len(lister.calls) == 2


def test_a_failing_list_returns_none_logs_and_is_not_cached(caplog):
    lister = Lister(RuntimeError("412 lobby"))
    roster = ParticipantRoster(lister, BOT)
    with caplog.at_level(logging.WARNING, logger=roster_module.__name__):
        assert roster.members(ROOM) is None
    assert any("participants list failed" in rec.message for rec in caplog.records)
    lister.attendees = [user("alice", "Alice")]
    assert roster.members(ROOM) == ({"alice": "Alice"}, False)
    assert len(lister.calls) == 2


def test_the_cap_reports_more_not_listed():
    lister = Lister([user(f"u{i}") for i in range(ROSTER_MAX_MEMBERS + 3)])
    names, more = ParticipantRoster(lister, BOT).members(ROOM)
    assert len(names) == ROSTER_MAX_MEMBERS
    assert more is True
    exact = Lister([user(f"u{i}") for i in range(ROSTER_MAX_MEMBERS)])
    assert ParticipantRoster(exact, BOT).members(ROOM)[1] is False


def test_the_cache_and_the_seen_maps_are_bounded():
    roster = ParticipantRoster(Lister([user("alice")]), BOT)
    for index in range(ROSTER_MAX_ROOMS + 5):
        roster.members(f"room{index}")
        roster.note_seen(f"room{index}", {"alice": "Alice"})
    assert len(roster._cache) == ROSTER_MAX_ROOMS
    assert len(roster._seen) == ROSTER_MAX_ROOMS
    assert "room0" not in roster._cache
    roster.note_seen(ROOM, {f"u{i}": f"U{i}" for i in range(SEEN_MAX_PER_ROOM + 7)})
    assert len(roster._seen[ROOM]) == SEEN_MAX_PER_ROOM
    assert "u0" not in roster._seen[ROOM]


# --- through the adapter --------------------------------------------------------


def _ocs(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"ocs": {"meta": {"statuscode": status}, "data": data}})


def talk(participants: Any, *, room_type: int = 2) -> Any:
    """A Talk that answers participants (a list, or an int status), room info and posts."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/participants"):
            if isinstance(participants, int):
                return httpx.Response(participants, text="lobby")
            return _ocs(participants)
        if "/api/v4/room/" in path:
            return _ocs({"type": room_type, "displayName": ROOM})
        return _ocs({"id": 99})

    return handler


def make_ops(participants: Any, cfg: NextcloudBridgeConfig = CFG) -> NextcloudTalkOps:
    client = TalkClient(cfg, client=httpx.Client(transport=httpx.MockTransport(talk(participants))))
    rooms = RoomDirectory(client)
    rooms.resolve_once(ROOM)
    return NextcloudTalkOps(
        cfg, client, httpx.Client(transport=httpx.MockTransport(talk([]))), rooms
    )


def talk_message(*mentions: tuple[str, str]) -> dict[str, Any]:
    params: dict[str, Any] = {"b": {"type": "user", "id": BOT, "name": "OSPREY Bot"}}
    body = "{b} tell"
    for index, (ident, name) in enumerate(mentions):
        params[f"m{index}"] = {"type": "user", "id": ident, "name": name}
        body += f" {{m{index}}}"
    return {
        "id": 42,
        "token": ROOM,
        "actorType": "users",
        "actorId": "alice",
        "actorDisplayName": "Alice",
        "message": body,
        "messageParameters": params,
        "systemMessage": "",
        "messageType": "comment",
    }


def test_room_people_over_a_mock_transport():
    ops = make_ops(
        [
            user("alice", "Alice"),
            {"actorType": "guests", "actorId": "g1", "displayName": "Guest"},
            user(BOT, "OSPREY Bot"),
            user("carol", ""),
        ]
    )
    ops.parse_event(talk_message(("carol", "Carol")))

    people = ops.room_people({NC_ROOM: ROOM})

    assert people is not None
    assert list(people.members) == [RoomMember("alice", "Alice"), RoomMember("carol", "Carol")]


def test_room_people_names_the_asker_from_the_entry_after_a_restart():
    ops = make_ops([user("alice"), user("carol")])
    people = ops.room_people({NC_ROOM: ROOM, "sender_id": "alice", "sender_display": "Alice"})
    assert people is not None
    assert list(people.members) == [RoomMember("alice", "Alice"), RoomMember("carol", None)]


def test_room_people_never_names_the_asker_from_the_id_fallback():
    ops = make_ops([user("alice")])
    people = ops.room_people({NC_ROOM: ROOM, "sender_id": "alice", "sender_display": "alice"})
    assert people is not None
    assert list(people.members) == [RoomMember("alice", None)]


def test_room_people_carries_the_mentions_setting():
    on = make_ops([user("alice", "Alice")])
    off = make_ops(
        [user("alice", "Alice")],
        NextcloudBridgeConfig(
            base_url=CFG.base_url, bot_account=BOT, app_password="pw", mentions=False
        ),
    )
    assert on.room_people({NC_ROOM: ROOM}).mentions is True
    assert off.room_people({NC_ROOM: ROOM}).mentions is False


def test_room_people_is_none_for_an_entry_without_a_room():
    ops = make_ops([user("alice", "Alice")])
    assert ops.room_people({}) is None
    assert ops.room_people({NC_ROOM: 7}) is None


def test_room_people_is_none_when_the_listing_fails():
    assert make_ops(412).room_people({NC_ROOM: ROOM}) is None


class StubDispatcher:
    def __init__(self) -> None:
        self.extras: list[Any] = []

    def run(self, question: str, extra: Any = None, *, on_run_id: Any = None) -> dict[str, Any]:
        del question, on_run_id
        self.extras.append(extra)
        return {"status": "completed", "text_output": "done", "run_id": "run-1"}


def test_the_room_reaches_the_dispatch_payload(tmp_path):
    ops = make_ops([user("alice"), user("carol")])
    core = CoreConfig(
        dedup_path=str(tmp_path / "dedup.json"), history_path=str(tmp_path / "history.json")
    )
    dispatcher = StubDispatcher()
    deps = dataclasses.replace(
        build_deps(core, ops, dedup=DedupStore(core.dedup_path), dispatcher=dispatcher),  # type: ignore[arg-type]
        probe_capability=lambda cfg, capability: True,
    )
    stop = threading.Event()
    bridge = BridgeRuntime(
        deps=deps,
        drain=build_drain_deps(deps),
        stop=stop,
        thread=threading.Thread(target=stop.wait, daemon=True),
    )

    assert bridge.handle_event(talk_message(("carol", "Carol"))) == "handled"

    [extra] = dispatcher.extras
    assert extra["asker"] == {"id": "alice", "name": "Alice"}
    assert extra["room"] == {
        "members": [{"id": "alice", "name": "Alice"}, {"id": "carol", "name": "Carol"}],
        "mentions": MENTION_RULE,
    }
