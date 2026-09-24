"""SpaceRoster: who is in a Chat space, named listing-first, then seen, then unknown."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from osprey.bridges.core import MENTION_RULE, CoreConfig, RoomMember
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.pipeline import PipelineDeps, handle_event
from osprey.bridges.google_chat import roster as roster_module
from osprey.bridges.google_chat.client import ChatClient
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.events import GC_SPACE
from osprey.bridges.google_chat.ops import GoogleChatOps
from osprey.bridges.google_chat.roster import (
    ROSTER_MAX_MEMBERS,
    ROSTER_MAX_SPACES,
    ROSTER_TTL_SECONDS,
    SEEN_MAX_PER_SPACE,
    SpaceRoster,
)

APP_ID = "users/999"
SPACE = "spaces/AAAA"
CFG = GoogleChatBridgeConfig(app_id=APP_ID)


def human(ident: str, name: str | None = None, **over: Any) -> dict[str, Any]:
    member: dict[str, Any] = {"name": ident, "type": "HUMAN"}
    if name is not None:
        member["displayName"] = name
    return {"member": member, "state": "JOINED", **over}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Lister:
    """A ``list_members`` stub: counts calls, answers a list or raises."""

    def __init__(self, memberships: Any = (), more: bool = False) -> None:
        self.memberships = memberships
        self.more = more
        self.calls: list[tuple[str, int]] = []

    def __call__(self, space: str, limit: int) -> tuple[list[Any], bool]:
        self.calls.append((space, limit))
        if isinstance(self.memberships, BaseException):
            raise self.memberships
        return list(self.memberships), self.more


def test_an_api_display_name_wins_over_a_seen_name():
    roster = SpaceRoster(Lister([human("users/1", "Alice")]), APP_ID)
    roster.note_seen(SPACE, {"users/1": "Ally"})
    assert roster.members(SPACE) == ({"users/1": "Alice"}, False)


def test_a_seen_name_fills_a_missing_api_name():
    roster = SpaceRoster(Lister([human("users/1")]), APP_ID)
    roster.note_seen(SPACE, {"users/1": "Alice"})
    assert roster.members(SPACE) == ({"users/1": "Alice"}, False)


def test_a_member_with_neither_name_is_unknown():
    roster = SpaceRoster(Lister([human("users/1")]), APP_ID)
    roster.note_seen("spaces/OTHER", {"users/1": "Alice"})
    assert roster.members(SPACE) == ({"users/1": None}, False)


def test_a_name_seen_after_the_fetch_fills_the_cached_roster_without_a_refetch():
    lister = Lister([human("users/1")])
    roster = SpaceRoster(lister, APP_ID)
    assert roster.members(SPACE) == ({"users/1": None}, False)
    roster.note_seen(SPACE, {"users/1": "Alice"})
    assert roster.members(SPACE) == ({"users/1": "Alice"}, False)
    assert len(lister.calls) == 1


def test_members_skips_bots_the_app_and_non_joined():
    roster = SpaceRoster(
        Lister(
            [
                human("users/1", "Alice"),
                {"member": {"name": "users/2", "type": "BOT"}, "state": "JOINED"},
                human(APP_ID, "Osprey"),
                human("999", "Osprey bare id"),
                human("users/3", "Invited", state="INVITED"),
                human("groups/4", "A group"),
                {"member": {"name": "users/5", "displayName": "No state"}},
                "junk",
            ]
        ),
        "999",
    )
    assert roster.members(SPACE) == ({"users/1": "Alice", "users/5": "No state"}, False)


def test_members_is_cached_within_the_ttl_and_refetched_after():
    clock = Clock()
    lister = Lister([human("users/1", "Alice")])
    roster = SpaceRoster(lister, APP_ID, now=clock)
    roster.members(SPACE)
    clock.t += ROSTER_TTL_SECONDS - 1
    roster.members(SPACE)
    assert len(lister.calls) == 1
    clock.t += 2
    roster.members(SPACE)
    assert len(lister.calls) == 2


def test_a_failing_list_returns_none_logs_and_is_not_cached(caplog):
    lister = Lister(RuntimeError("members down"))
    roster = SpaceRoster(lister, APP_ID)
    with caplog.at_level(logging.WARNING, logger=roster_module.__name__):
        assert roster.members(SPACE) is None
    assert any("members.list failed" in rec.message for rec in caplog.records)
    lister.memberships = [human("users/1", "Alice")]
    assert roster.members(SPACE) == ({"users/1": "Alice"}, False)
    assert len(lister.calls) == 2


def test_the_cache_and_the_seen_map_are_bounded():
    roster = SpaceRoster(Lister([human("users/1")]), APP_ID)
    for index in range(ROSTER_MAX_SPACES + 5):
        roster.members(f"spaces/{index}")
        roster.note_seen(f"spaces/{index}", {"users/1": "Alice"})
    assert len(roster._cache) == ROSTER_MAX_SPACES
    assert len(roster._seen) == ROSTER_MAX_SPACES
    assert "spaces/0" not in roster._cache
    roster.note_seen(SPACE, {f"users/{i}": f"U{i}" for i in range(SEEN_MAX_PER_SPACE + 7)})
    assert len(roster._seen[SPACE]) == SEEN_MAX_PER_SPACE
    assert "users/0" not in roster._seen[SPACE]


def test_the_cap_reports_more_not_listed():
    lister = Lister([human(f"users/{i}") for i in range(3)], more=True)
    roster = SpaceRoster(lister, APP_ID)
    assert roster.members(SPACE)[1] is True
    assert lister.calls == [(SPACE, ROSTER_MAX_MEMBERS)]


# --- through the adapter --------------------------------------------------------


def members_page(memberships: list[Any], token: str | None = None) -> dict[str, Any]:
    page: dict[str, Any] = {"memberships": memberships}
    if token:
        page["nextPageToken"] = token
    return page


def make_ops(*pages: Any, cfg: GoogleChatBridgeConfig = CFG) -> tuple[GoogleChatOps, MagicMock]:
    service = MagicMock()
    listing = service.spaces.return_value.members.return_value.list
    listing.return_value.execute.side_effect = list(pages)
    create = service.spaces.return_value.messages.return_value.create
    create.return_value.execute.return_value = {"name": f"{SPACE}/messages/X"}
    return GoogleChatOps(cfg, ChatClient(cfg, service)), service


def chat_event(sender: str, name: str, *mentions: tuple[str, str]) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = [
        {
            "type": "USER_MENTION",
            "userMention": {"user": {"name": APP_ID, "displayName": "Osprey", "type": "BOT"}},
        }
    ]
    for ident, display in mentions:
        annotations.append(
            {
                "type": "USER_MENTION",
                "userMention": {"user": {"name": ident, "displayName": display}},
            }
        )
    return {
        "type": "MESSAGE",
        "message": {
            "name": f"{SPACE}/messages/M1",
            "sender": {"name": sender, "displayName": name, "type": "HUMAN"},
            "text": "@Osprey tell Carol",
            "argumentText": "tell Carol",
            "thread": {"name": f"{SPACE}/threads/T"},
            "space": {"name": SPACE, "type": "ROOM"},
            "annotations": annotations,
        },
    }


def test_room_people_via_ops_over_a_mock_service():
    ops, _ = make_ops(
        members_page([human("users/111", "Alice"), human("users/222")], token="p2"),
        members_page([human("users/333")]),
    )
    ops.parse_event(chat_event("users/444", "Dave", ("users/222", "Carol")))

    people = ops.room_people({GC_SPACE: SPACE})

    assert people is not None
    assert list(people.members) == [
        RoomMember("users/111", "Alice"),
        RoomMember("users/222", "Carol"),
        RoomMember("users/333", None),
    ]
    assert people.more_not_listed is False


def test_room_people_names_the_asker_from_the_entry_after_a_restart():
    ops, _ = make_ops(members_page([human("users/111"), human("users/222")]))
    entry = {GC_SPACE: SPACE, "sender_id": "users/111", "sender_display": "Alice"}

    people = ops.room_people(entry)

    assert people is not None
    assert list(people.members) == [
        RoomMember("users/111", "Alice"),
        RoomMember("users/222", None),
    ]


def test_room_people_carries_the_mentions_setting():
    on, _ = make_ops(members_page([human("users/111", "Alice")]))
    off, _ = make_ops(
        members_page([human("users/111", "Alice")]),
        cfg=GoogleChatBridgeConfig(app_id=APP_ID, mentions=False),
    )
    assert on.room_people({GC_SPACE: SPACE}).mentions is True
    assert off.room_people({GC_SPACE: SPACE}).mentions is False
    assert on.room_people({}) is None


class StubDispatcher:
    def __init__(self) -> None:
        self.extras: list[Any] = []

    def run(self, question: str, extra: Any = None, *, on_run_id: Any = None) -> dict[str, Any]:
        del question, on_run_id
        self.extras.append(extra)
        return {"status": "completed", "text_output": "done", "run_id": "run-1"}


def test_the_room_reaches_the_dispatch_payload(tmp_path):
    ops, _ = make_ops(members_page([human("users/111"), human("users/222")]))
    dispatcher = StubDispatcher()
    deps = PipelineDeps(
        cfg=CoreConfig(),
        ops=ops,
        dedup=DedupStore(str(tmp_path / "dedup.json")),
        dispatcher=dispatcher,  # type: ignore[arg-type]
        probe_capability=lambda cfg, capability: True,
    )

    assert handle_event(chat_event("users/111", "Alice", ("users/222", "Carol")), deps) == "handled"

    [extra] = dispatcher.extras
    assert extra["asker"] == {"id": "users/111", "name": "Alice"}
    assert extra["room"] == {
        "members": [
            {"id": "users/111", "name": "Alice"},
            {"id": "users/222", "name": "Carol"},
        ],
        "mentions": MENTION_RULE,
    }
