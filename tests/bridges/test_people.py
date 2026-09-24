"""Tests for :mod:`osprey.bridges.core.people`: who asked, and who is in the room."""

from __future__ import annotations

import re

import pytest

from osprey.bridges.core.people import (
    MENTION_PLACEHOLDER_RE,
    MENTION_RULE,
    MENTIONS_OFF_NOTE,
    asker_of,
    room_payload,
)
from osprey.bridges.core.ports import RoomMember, RoomPeople


def test_asker_of_reads_the_persisted_sender():
    entry = {"sender_id": "users/111", "sender_display": "Alice", "text": "q"}
    assert asker_of(entry) == {"id": "users/111", "name": "Alice"}


def test_asker_of_keeps_a_name_without_an_id_and_an_id_without_a_name():
    assert asker_of({"sender_display": "Alice"}) == {"id": None, "name": "Alice"}
    assert asker_of({"sender_id": "users/111", "sender_display": ""}) == {
        "id": "users/111",
        "name": None,
    }


def test_asker_of_is_none_for_an_entry_with_no_sender():
    assert asker_of({}) is None
    assert asker_of({"sender_id": "", "sender_display": ""}) is None


def test_asker_of_ignores_non_string_sender_fields():
    assert asker_of({"sender_id": 7, "sender_display": None}) is None
    assert asker_of({"sender_id": ["users/1"], "sender_display": "Alice"}) == {
        "id": None,
        "name": "Alice",
    }


# --- who is in the room ---------------------------------------------------------

ALICE = RoomMember("users/111", "Alice")
CAROL = RoomMember("users/222", "Carol")
UNKNOWN = RoomMember("users/333")


def test_room_payload_lists_members_and_the_mention_rule():
    payload = room_payload(RoomPeople(members=(ALICE, CAROL), mentions=True))
    assert payload == {
        "members": [
            {"id": "users/111", "name": "Alice"},
            {"id": "users/222", "name": "Carol"},
        ],
        "mentions": MENTION_RULE,
    }


def test_room_payload_with_mentions_off_carries_the_off_note_instead():
    payload = room_payload(RoomPeople(members=(ALICE,), mentions=False))
    assert payload is not None
    assert payload["mentions"] == MENTIONS_OFF_NOTE
    assert MENTIONS_OFF_NOTE != MENTION_RULE


def test_room_payload_marks_a_capped_list():
    payload = room_payload(RoomPeople(members=(ALICE,), more_not_listed=True))
    assert payload is not None
    assert payload["more_not_listed"] is True
    assert "more_not_listed" not in (room_payload(RoomPeople(members=(ALICE,))) or {})


def test_room_payload_is_none_for_no_members():
    assert room_payload(RoomPeople()) is None
    assert room_payload(RoomPeople(members=(), mentions=True, more_not_listed=True)) is None


def test_unknown_names_stay_null_in_the_payload():
    payload = room_payload(RoomPeople(members=(UNKNOWN,), mentions=True))
    assert payload is not None
    assert payload["members"] == [{"id": "users/333", "name": None}]


@pytest.mark.parametrize(
    ("text", "ident"),
    [
        ("<@users/222>", "users/222"),
        ("<@29:1GcS4E_yB-oS>", "29:1GcS4E_yB-oS"),
        ("<@carol@example.org>", "carol@example.org"),
        ("<users/222>", None),
        ("<@ >", None),
        ("<@a b>", None),
    ],
)
def test_the_placeholder_pattern_reads_the_id(text, ident):
    match = MENTION_PLACEHOLDER_RE.fullmatch(text)
    assert (match.group(1) if match else None) == ident


NARROW_PLACEHOLDER_RE = re.compile(r"<@([^\s<>@]+)>")


@pytest.mark.parametrize(
    "answer",
    [
        "tell <@users/222> about it",
        "<@users/222><@users/111>",
        "(<@users/222>), <@users/111>.",
        "**<@users/222>** and `<@users/111>` in a list:\n- <@users/333>",
        "tell <@29:1GcS4E_yB-oS> about it",
        "<@29:1GcS4E_yB-oS><@29:222>",
        "(<@29:1GcS4E_yB-oS>), <@29:222>.",
        "**<@29:1GcS4E_yB-oS>** x_<@29:222>_y",
    ],
)
def test_widening_the_placeholder_leaves_chat_and_teams_ids_unchanged(answer):
    assert MENTION_PLACEHOLDER_RE.findall(answer) == NARROW_PLACEHOLDER_RE.findall(answer)
