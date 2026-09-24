"""Tests for :mod:`osprey.bridges.core.people`: who asked, and who is in the room."""

from __future__ import annotations

from osprey.bridges.core.people import asker_of


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
